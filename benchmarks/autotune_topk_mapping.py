"""
Auto-tuner for TopK mapping hyperparameters.

Sweeps all (mode, hyperparameter) combinations using the topk_hit_rate
kernel and ranks by Stage 1 resolution rate.

Supports real-data score distributions via --real-histograms: loads the
raw_histograms.npy from calibration and synthesizes score tensors that
match the real bin distribution (by reversing the convert_to_uint8 mapping).

Sweep grid:
  - Mode 3 (power):  p     in [0.1, 0.25, 0.75, 0.9]
  - Mode 6 (asinh):  beta  in [0.1, 0.5, 1, 2, 4]
  - Mode 7 (log1p):  alpha in [0.1, 0.5, 0.75, 1, 2, 4, 8]
  - Baselines: mode 0 (none), mode 4 (log)

Usage:
    python benchmarks/autotune_topk_mapping.py --topk-val 30 --real-histograms calibration/raw_histograms.npy
    python benchmarks/autotune_topk_mapping.py --topk-val 30 --output-json results.json
"""

import argparse
import json
import math
from typing import List

import numpy as np
import torch

from bench_topk import make_topk_inputs, bench_kernel, compute_histogram_stats
from vortex_torch_C import topk_profile_histogram, topk_profile_counters, topk_output_sglang



SWEEP_GRID = {
    # (mode, param_name, param_values)
    3: ("power_exp", [0.1, 0.25, 0.5, 0.75, 0.9, 2.0, 4.0]),
    6: ("beta", [0.1, 0.5, 1.0, 2.0, 4.0]),
    7: ("alpha", [0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]),
    9: ("alpha", [0.1, 0.5, 1.0, 2.0, 4.0]),
    10: ("alpha", [0.1, 0.5, 1.0, 2.0, 4.0]),
    13: ("alpha", [0.5, 1.0, 2.0, 4.0, 8.0]),
    14: ("rho", [2.0, 4.0, 8.0, 16.0]),
}
BASELINES = {
    0: ("none", 0.5),
    4: ("log", 0.5),
    8: ("trunc8", 0.5),
    11: ("subtract", 0.5),
}
# Noscale baselines for parametric transform modes (skip auto-range pre-pass)
NOSCALE_BASELINES = {
    3: ("power_noscale", [0.5]),
    6: ("asinh_noscale", [1.0]),
    7: ("log1p_noscale", [1.0]),
    9: ("erf_noscale", [1.0]),
    10: ("tanh_noscale", [1.0]),
    13: ("exp_stretch_noscale", [1.0, 4.0]),
}
MODE_NAMES = {
    0: "none",
    3: "power",
    4: "log",
    6: "asinh",
    7: "log1p",
    8: "trunc8",
    9: "erf",
    10: "tanh",
    11: "subtract",
    13: "exp_stretch",
    14: "topk_window",
}


def _key_to_fp16(key: int) -> np.float16:
    """Invert the convert_to_uint8 sign-flip for a single 16-bit key."""
    if key >= 0x8000:
        bits = key & 0x7FFF
    else:
        bits = (~key) & 0xFFFF
    return np.array([bits], dtype=np.uint16).view(np.float16)[0]


def build_bin_range_table():
    """Build per-bin (lo, hi) fp16 value tables by iterating all 65536 fp16 bit patterns.

    For each fp16 value, compute its bin via convert_to_uint8 logic, then track
    the min/max fp16 value that lands in each bin.

    Returns:
        (bin_lo, bin_hi): two [256] float32 arrays — the min and max fp16 values per bin.
    """
    # Generate all 65536 fp16 bit patterns
    all_bits = np.arange(65536, dtype=np.uint16)
    all_fp16 = all_bits.view(np.float16)

    # Compute convert_to_uint8 for each: key = sign-flip, bin = key >> 8
    keys = np.where(
        (all_bits & 0x8000).astype(bool),
        (~all_bits).astype(np.uint16),
        all_bits | np.uint16(0x8000),
    )
    bins = (keys >> 8).astype(np.uint8)

    # Convert to float32 for min/max (fp16 has NaNs/Infs, filter them)
    all_f32 = all_fp16.astype(np.float32)
    valid = np.isfinite(all_f32)

    bin_lo = np.full(256, np.inf, dtype=np.float32)
    bin_hi = np.full(256, -np.inf, dtype=np.float32)

    for b in range(256):
        mask = (bins == b) & valid
        if mask.any():
            vals = all_f32[mask]
            bin_lo[b] = vals.min()
            bin_hi[b] = vals.max()

    # For any bin with no valid fp16 values, fall back to midpoint
    empty = bin_lo > bin_hi
    for b in np.where(empty)[0]:
        mid_key = (int(b) << 8) | 0x80
        val = float(_key_to_fp16(mid_key))
        bin_lo[b] = val
        bin_hi[b] = val

    return bin_lo, bin_hi


def generate_remap_lut(mode: int, param: float) -> np.ndarray:
    """Generate a 256-entry uint8 LUT that approximates a transform mode.

    For each of the 256 fp16 radix bins, compute the transform of the
    bin's midpoint value, then linearly map transformed values to [0,255].
    The resulting LUT can be used with mode=1 (LUT CDF) infrastructure,
    replacing expensive per-element transcendental math with a single
    shared memory lookup.

    Args:
        mode: TopKMappingMode (3=Power, 4=Log, 6=Asinh, 7=Log1p, 9=Erf, 10=Tanh)
        param: power_exp/beta/alpha for the transform

    Returns:
        lut: [256] uint8 array mapping original_bin -> remapped_bin
    """
    bin_lo, bin_hi = build_bin_range_table()
    midpoints = (bin_lo + bin_hi) / 2.0  # [256] float32

    # Apply transform
    if mode == 3:  # power
        transformed = np.sign(midpoints) * np.abs(midpoints) ** param
    elif mode == 4:  # log
        transformed = np.sign(midpoints) * np.log(np.abs(midpoints) + 1.0)
    elif mode == 6:  # asinh
        transformed = np.arcsinh(param * midpoints)
    elif mode == 7:  # log1p
        transformed = np.sign(midpoints) * np.log1p(param * np.abs(midpoints))
    elif mode == 9:  # erf
        from scipy.special import erf
        transformed = erf(param * midpoints)
    elif mode == 10:  # tanh
        transformed = np.tanh(param * midpoints)
    else:
        # Identity fallback
        transformed = midpoints.copy()

    # Handle NaN/Inf from edge cases
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    # Linear map to [0, 255]
    tmin, tmax = transformed.min(), transformed.max()
    if tmax > tmin:
        lut = np.clip(((transformed - tmin) / (tmax - tmin) * 255), 0, 255).astype(np.uint8)
    else:
        lut = np.full(256, 128, dtype=np.uint8)

    return lut


def scores_from_histogram(
    histogram: np.ndarray,
    total_pages: int,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate score tensor matching a real bin distribution.

    For each sampled bin, generates a uniform random fp16 value within the
    bin's actual value range (not just the midpoint), so that mapped transforms
    see diverse input values.

    Args:
        histogram: [256] aggregated bin counts from calibration
        total_pages: number of score entries to generate
        device: torch device

    Returns:
        scores: [total_pages, 1, 1] bfloat16 tensor
    """
    bin_lo, bin_hi = build_bin_range_table()

    # Normalize histogram to probability distribution
    counts = histogram.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return torch.zeros(total_pages, 1, 1, dtype=torch.bfloat16, device=device)
    probs = counts / total

    # Sample bin indices according to the real distribution
    bin_indices = np.random.choice(256, size=total_pages, p=probs)

    # Uniform random within each bin's fp16 range
    lo = bin_lo[bin_indices]
    hi = bin_hi[bin_indices]
    rand = np.random.uniform(0, 1, size=total_pages).astype(np.float32)
    scores_f32 = lo + rand * (hi - lo)

    # Convert float32 -> bfloat16 tensor
    scores = torch.from_numpy(scores_f32).to(torch.bfloat16)
    return scores.reshape(total_pages, 1, 1).to(device)


def make_real_inputs(
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    page_size: int,
    topk_val: int,
    reserved_bos: int,
    reserved_eos: int,
    histogram: np.ndarray,
    device: str = "cuda",
) -> dict:
    """Build CSR-formatted inputs with scores matching a real histogram."""
    eff_batch_size = batch_size * num_kv_heads
    num_pages_per_seg = math.ceil(seq_len / page_size)
    total_dense_pages = eff_batch_size * num_pages_per_seg
    sparse_per_seg = min(topk_val + reserved_bos + reserved_eos, num_pages_per_seg)
    total_sparse_pages = eff_batch_size * sparse_per_seg

    dense_kv_indptr = torch.arange(
        0, (eff_batch_size + 1) * num_pages_per_seg, num_pages_per_seg,
        dtype=torch.int32, device=device,
    )
    sparse_kv_indptr = torch.arange(
        0, (eff_batch_size + 1) * sparse_per_seg, sparse_per_seg,
        dtype=torch.int32, device=device,
    )
    dense_kv_indices = torch.arange(total_dense_pages, dtype=torch.int32, device=device)
    sparse_kv_indices = torch.zeros(total_sparse_pages, dtype=torch.int32, device=device)

    x = scores_from_histogram(histogram, total_dense_pages, device=device)

    return {
        "x": x,
        "dense_kv_indptr": dense_kv_indptr,
        "sparse_kv_indptr": sparse_kv_indptr,
        "dense_kv_indices": dense_kv_indices,
        "sparse_kv_indices": sparse_kv_indices,
        "eff_batch_size": eff_batch_size,
        "num_pages_per_seg": num_pages_per_seg,
        "sparse_per_seg": sparse_per_seg,
    }


def run_sweep(args) -> List[dict]:
    """Run all (mode, hyperparam) combos and return ranked results."""
    results = []

    # Load real histogram if provided
    real_histogram = None
    if args.real_histograms:
        raw = np.load(args.real_histograms)  # [num_segments, 256]
        real_histogram = raw.sum(axis=0) if raw.ndim > 1 else raw  # aggregate to [256]

    distributions = args.distributions
    if real_histogram is not None:
        distributions = ["real"]

    for dist in distributions:
        if dist == "real":
            inputs = make_real_inputs(
                batch_size=args.batch_size,
                num_kv_heads=args.num_kv_heads,
                seq_len=args.seq_len,
                page_size=args.page_size,
                topk_val=args.topk_val,
                reserved_bos=args.reserved_bos,
                reserved_eos=args.reserved_eos,
                histogram=real_histogram,
            )
        else:
            inputs = make_topk_inputs(
                batch_size=args.batch_size,
                num_kv_heads=args.num_kv_heads,
                seq_len=args.seq_len,
                page_size=args.page_size,
                topk_val=args.topk_val,
                reserved_bos=args.reserved_bos,
                reserved_eos=args.reserved_eos,
                score_dtype=torch.bfloat16,
                distribution=dist,
            )

        eff_bs = inputs["eff_batch_size"]

        def evaluate(mode: int, power: float, label: str, noscale: bool = False,
                    lut_tensor=None):
            hists = torch.zeros(eff_bs, 256, dtype=torch.int32, device="cuda")
            topk_profile_histogram(
                inputs["x"],
                inputs["dense_kv_indptr"],
                hists,
                eff_bs,
                args.reserved_bos,
                args.reserved_eos,
                mode,
                power,
                lut_tensor,  # lut
                None,  # quantiles
                noscale,
            )
            torch.cuda.synchronize()
            stats = compute_histogram_stats(hists)
            result = {
                "label": label,
                "mode": mode,
                "mode_name": MODE_NAMES.get(mode, f"m{mode}"),
                "param": power,
                "noscale": noscale,
                "distribution": dist,
                "gini": stats["gini"],
                "max_mean_ratio": stats["max_mean_ratio"],
                "num_nonzero_bins": stats["num_nonzero_bins"],
            }

            # Counter-based metrics (Stage 2 cost analysis)
            if args.counters:
                inputs["sparse_kv_indices"].zero_()
                counter_buf = torch.zeros(eff_bs, 6, dtype=torch.int32, device="cuda")
                topk_profile_counters(
                    inputs["x"],
                    inputs["dense_kv_indptr"],
                    inputs["sparse_kv_indptr"],
                    inputs["dense_kv_indices"],
                    inputs["sparse_kv_indices"],
                    counter_buf,
                    eff_bs,
                    args.topk_val,
                    args.reserved_bos,
                    args.reserved_eos,
                    inputs["num_pages_per_seg"],
                    mode,
                    power,
                    lut_tensor,  # lut
                    None,  # quantiles
                    noscale,
                )
                torch.cuda.synchronize()
                c = counter_buf.float()
                result["num_equal_mean"] = c[:, 2].mean().item()
                result["remaining_k_mean"] = c[:, 3].mean().item()
                result["refine_rounds_mean"] = c[:, 4].mean().item()
                result["stage2_input_mean"] = c[:, 5].mean().item()
                result["res_rate_mean"] = (c[:, 3] == 0).float().mean().item()

            return result

        # Baselines
        for mode, (name, default_power) in BASELINES.items():
            r = evaluate(mode, default_power, f"m{mode}_{name}")
            results.append(r)

        # Parametric sweep (scaled)
        for mode, (param_name, values) in SWEEP_GRID.items():
            mname = MODE_NAMES[mode]
            for val in values:
                label = f"m{mode}_{mname}_{param_name}={val}"
                r = evaluate(mode, val, label)
                results.append(r)

        # Noscale sweep for parametric modes
        for mode, (name, values) in NOSCALE_BASELINES.items():
            mname = MODE_NAMES[mode]
            for val in values:
                label = f"m{mode}_{mname}_noscale_{val}"
                r = evaluate(mode, val, label, noscale=True)
                results.append(r)

        # LUT approximation sweep: generate a LUT for each (mode, param) and
        # evaluate via mode=1 (LUT CDF). This replaces per-element transcendentals
        # with a single shared memory lookup.
        if args.lut_sweep:
            lut_modes = {
                3: [0.25, 0.5, 0.75],
                6: [0.5, 1.0, 2.0],
                7: [0.5, 1.0, 2.0],
                9: [0.5, 1.0, 2.0],
                10: [0.5, 1.0, 2.0],
            }
            for src_mode, params in lut_modes.items():
                src_name = MODE_NAMES[src_mode]
                for p in params:
                    try:
                        lut_np = generate_remap_lut(src_mode, p)
                        lut_t = torch.from_numpy(lut_np).cuda()
                        label = f"lut_{src_name}_{p}"
                        # Evaluate as mode=1 (LUT CDF) with the generated LUT
                        r = evaluate(1, 0.5, label, lut_tensor=lut_t)
                        r["lut_source_mode"] = src_mode
                        r["lut_source_param"] = p
                        results.append(r)
                    except ImportError:
                        # scipy not available for erf
                        pass

    return results


def print_table(results: List[dict], show_latency: bool = False):
    """Print ranked results as a formatted table."""
    has_counters = any("res_rate_mean" in r for r in results)
    has_latency = any("full_kernel_ms" in r for r in results)

    # Primary ranking: by res_rate_mean (higher=better) if counters, else by gini (lower=better)
    if has_counters:
        ranked = sorted(results, key=lambda r: -r.get("res_rate_mean", 0.0))
        rank_label = "ranked by res_rate, higher=better"
    else:
        ranked = sorted(results, key=lambda r: r["gini"])
        rank_label = "ranked by Gini, lower=better"

    # Build header
    cols = f"{'Rank':>4s}  {'Label':<35s}  {'Dist':<12s}  {'Gini':>6s}  {'Max/Mean':>8s}  {'NZBins':>6s}"
    if has_counters:
        cols += f"  {'ResRate':>7s}  {'RemK':>5s}  {'Rnds':>4s}  {'S2In':>5s}"
    if has_latency and show_latency:
        cols += f"  {'LatMs':>9s}  {'LatRk':>5s}"

    print(f"\n{'=' * len(cols)}")
    print(f"TopK Mapping Auto-Tune Results ({rank_label})")
    print("=" * len(cols))
    print(cols)
    print("-" * len(cols))

    for i, r in enumerate(ranked):
        noscale_tag = " [NS]" if r.get("noscale", False) else ""
        line = (
            f"{i+1:4d}  {r['label'] + noscale_tag:<35s}  {r['distribution']:<12s}  "
            f"{r['gini']:6.3f}  "
            f"{r['max_mean_ratio']:8.2f}  {r['num_nonzero_bins']:6d}"
        )
        if has_counters:
            rr = r.get("res_rate_mean", 0.0)
            rk = r.get("remaining_k_mean", 0.0)
            rnds = r.get("refine_rounds_mean", 0.0)
            s2in = r.get("stage2_input_mean", 0.0)
            line += f"  {rr:7.3f}  {rk:5.0f}  {rnds:4.1f}  {s2in:5.0f}"
        if has_latency and show_latency:
            lat = r.get("full_kernel_ms", float("nan"))
            lat_rank = r.get("latency_rank", "-")
            line += f"  {lat:9.4f}  {lat_rank:>5s}" if isinstance(lat_rank, str) else f"  {lat:9.4f}  {lat_rank:5d}"
        print(line)

    print("=" * len(cols))
    if ranked:
        best = ranked[0]
        msg = (
            f"\nBest overall: {best['label']} (dist={best['distribution']}) "
            f"— gini={best['gini']:.3f}, max/mean={best['max_mean_ratio']:.2f}"
        )
        if has_counters:
            msg += f", res_rate={best.get('res_rate_mean', 0):.3f}"
        if "full_kernel_ms" in best:
            msg += f", latency={best['full_kernel_ms']:.4f}ms"
        print(msg)

    # If latency data available, also print best by latency
    if has_latency and show_latency:
        lat_ranked = sorted([r for r in results if "full_kernel_ms" in r],
                            key=lambda r: r["full_kernel_ms"])
        if lat_ranked:
            best_lat = lat_ranked[0]
            print(
                f"Best by latency: {best_lat['label']} (dist={best_lat['distribution']}) "
                f"— latency={best_lat['full_kernel_ms']:.4f}ms, gini={best_lat['gini']:.3f}"
            )

    # Per-mode best summary
    mode_best = {}
    for r in results:
        m = r["mode"]
        if has_counters:
            is_better = m not in mode_best or r.get("res_rate_mean", 0) > mode_best[m].get("res_rate_mean", 0)
        else:
            is_better = m not in mode_best or r["gini"] < mode_best[m]["gini"]
        if is_better:
            mode_best[m] = r

    if mode_best:
        print("\nBest per mode:")
        for m in sorted(mode_best.keys()):
            r = mode_best[m]
            mname = MODE_NAMES.get(m, f"m{m}")
            if m in SWEEP_GRID:
                param_name = SWEEP_GRID[m][0]
                param_str = f"{param_name}={r['param']}"
            else:
                param_str = "(baseline)"
            ns_str = " noscale" if r.get("noscale", False) else ""
            lat_str = f"  latency={r['full_kernel_ms']:.4f}ms" if "full_kernel_ms" in r else ""
            counter_str = f"  res_rate={r.get('res_rate_mean', 0):.3f}" if has_counters else ""
            print(
                f"  Mode {m:d} ({mname:>5s}{ns_str}):  {param_str:<20s}  "
                f"gini={r['gini']:.3f}  max/mean={r['max_mean_ratio']:.2f}{counter_str}{lat_str}"
            )


def latency_rerank(results: List[dict], args) -> List[dict]:
    """Re-rank top Gini candidates by actual kernel latency."""
    # Sort by Gini, take top N
    ranked = sorted(results, key=lambda r: r["gini"])
    finalists = ranked[:args.latency_top_n]

    print(f"\n--- Latency re-ranking: timing top {len(finalists)} Gini finalists ---")

    # Build inputs for latency measurement
    real_histogram = None
    if args.real_histograms:
        raw = np.load(args.real_histograms)
        real_histogram = raw.sum(axis=0) if raw.ndim > 1 else raw

    if real_histogram is not None:
        inputs = make_real_inputs(
            batch_size=args.batch_size,
            num_kv_heads=args.num_kv_heads,
            seq_len=args.seq_len,
            page_size=args.page_size,
            topk_val=args.topk_val,
            reserved_bos=args.reserved_bos,
            reserved_eos=args.reserved_eos,
            histogram=real_histogram,
        )
    else:
        inputs = make_topk_inputs(
            batch_size=args.batch_size,
            num_kv_heads=args.num_kv_heads,
            seq_len=args.seq_len,
            page_size=args.page_size,
            topk_val=args.topk_val,
            reserved_bos=args.reserved_bos,
            reserved_eos=args.reserved_eos,
            score_dtype=torch.bfloat16,
            distribution="normal",
        )

    eff_bs = inputs["eff_batch_size"]
    pages_per_seg = inputs["num_pages_per_seg"]

    for r in finalists:
        inputs["sparse_kv_indices"].zero_()
        # For LUT-generated entries, regenerate the LUT tensor
        lut_tensor = None
        if "lut_source_mode" in r:
            lut_np = generate_remap_lut(r["lut_source_mode"], r["lut_source_param"])
            lut_tensor = torch.from_numpy(lut_np).cuda()
        call_args = (
            inputs["x"],
            inputs["dense_kv_indptr"],
            inputs["sparse_kv_indptr"],
            inputs["dense_kv_indices"],
            inputs["sparse_kv_indices"],
            eff_bs,
            args.topk_val,
            args.reserved_bos,
            args.reserved_eos,
            pages_per_seg,
            r["mode"],
            r["param"],
            lut_tensor,  # lut
            None,  # quantiles
            r.get("noscale", False),
        )
        latency = bench_kernel(topk_output_sglang, call_args,
                               warmup=10, repeat=args.latency_repeat)
        r["full_kernel_ms"] = latency["mean_ms"]
        print(f"  {r['label']:<35s}  gini={r['gini']:.3f}  latency={latency['mean_ms']:.4f}ms")

    # Re-rank finalists by latency
    finalists.sort(key=lambda r: r["full_kernel_ms"])
    for i, r in enumerate(finalists):
        r["latency_rank"] = i + 1
        r["gini_rank"] = next(j+1 for j, x in enumerate(ranked) if x is r)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune TopK mapping hyperparameters"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--topk-val", type=int, default=30)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--reserved-bos", type=int, default=1)
    parser.add_argument("--reserved-eos", type=int, default=2)
    parser.add_argument(
        "--distributions", nargs="+",
        default=["normal"],
        help="Score distributions for synthetic data (ignored when --real-histograms is set)",
    )
    parser.add_argument(
        "--real-histograms", type=str, default=None,
        help="Path to raw_histograms.npy from calibration. When set, auto-tunes on "
             "real score distribution instead of synthetic data.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument("--latency-rerank", action="store_true",
                        help="Re-rank top Gini finalists by actual kernel latency")
    parser.add_argument("--latency-top-n", type=int, default=10,
                        help="Number of Gini finalists to re-rank by latency (default: 10)")
    parser.add_argument("--latency-repeat", type=int, default=50,
                        help="Kernel timing repetitions for latency measurement (default: 50)")
    parser.add_argument("--counters", action="store_true",
                        help="Collect counter-based metrics (Stage 2 cost analysis) for each config")
    parser.add_argument("--lut-sweep", action="store_true",
                        help="Generate and evaluate LUT approximations for parametric transform modes")
    args = parser.parse_args()

    source = f"real ({args.real_histograms})" if args.real_histograms else f"synthetic ({args.distributions})"
    print(f"Auto-tuning TopK mapping hyperparameters")
    print(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"topk_val={args.topk_val}, num_kv_heads={args.num_kv_heads}")
    print(f"  score source: {source}")
    n_parametric = sum(len(v) for _, v in SWEEP_GRID.values())
    n_baselines = len(BASELINES)
    n_dists = 1 if args.real_histograms else len(args.distributions)
    print(f"  sweep: {n_parametric} parametric + {n_baselines} baselines "
          f"= {n_parametric + n_baselines} combos x {n_dists} dists")

    results = run_sweep(args)

    if args.latency_rerank:
        results = latency_rerank(results, args)

    print_table(results, show_latency=args.latency_rerank)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
