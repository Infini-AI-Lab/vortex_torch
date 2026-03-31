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

from bench_topk import make_topk_inputs, compute_histogram_stats
from vortex_torch_C import topk_profile_histogram



SWEEP_GRID = {
    # (mode, param_name, param_values)
    3: ("power_exp", [0.1, 0.25, 0.75, 0.9]),
    6: ("beta", [0.1, 0.5, 1.0, 2.0, 4.0]),
    7: ("alpha", [0.1, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]),
}
BASELINES = {
    0: ("none", 0.5),
    4: ("log", 0.5),
}
MODE_NAMES = {
    0: "none",
    3: "power",
    4: "log",
    6: "asinh",
    7: "log1p",
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

        def evaluate(mode: int, power: float, label: str):
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
                None,  # lut
                None,  # quantiles
            )
            torch.cuda.synchronize()
            stats = compute_histogram_stats(hists)
            return {
                "label": label,
                "mode": mode,
                "mode_name": MODE_NAMES.get(mode, f"m{mode}"),
                "param": power,
                "distribution": dist,
                "gini": stats["gini"],
                "max_mean_ratio": stats["max_mean_ratio"],
                "num_nonzero_bins": stats["num_nonzero_bins"],
            }

        # Baselines
        for mode, (name, default_power) in BASELINES.items():
            r = evaluate(mode, default_power, f"m{mode}_{name}")
            results.append(r)

        # Parametric sweep
        for mode, (param_name, values) in SWEEP_GRID.items():
            mname = MODE_NAMES[mode]
            for val in values:
                label = f"m{mode}_{mname}_{param_name}={val}"
                r = evaluate(mode, val, label)
                results.append(r)

    return results


def print_table(results: List[dict]):
    """Print ranked results as a formatted table."""
    # Sort by Gini ascending (lower = more uniform = better)
    ranked = sorted(results, key=lambda r: r["gini"])

    header = (
        f"{'Rank':>4s}  {'Label':<35s}  {'Dist':<12s}  "
        f"{'Gini':>6s}  {'Max/Mean':>8s}  {'NZBins':>6s}"
    )
    print("\n" + "=" * len(header))
    print("TopK Mapping Auto-Tune Results (ranked by Gini, lower=better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for i, r in enumerate(ranked):
        print(
            f"{i+1:4d}  {r['label']:<35s}  {r['distribution']:<12s}  "
            f"{r['gini']:6.3f}  "
            f"{r['max_mean_ratio']:8.2f}  {r['num_nonzero_bins']:6d}"
        )

    print("=" * len(header))
    if ranked:
        best = ranked[0]
        print(
            f"\nBest overall: {best['label']} (dist={best['distribution']}) "
            f"— gini={best['gini']:.3f}, max/mean={best['max_mean_ratio']:.2f}"
        )

    # Per-mode best summary (lowest gini per mode)
    mode_best = {}
    for r in results:
        m = r["mode"]
        if m not in mode_best or r["gini"] < mode_best[m]["gini"]:
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
            print(
                f"  Mode {m:d} ({mname:>5s}):  {param_str:<20s}  "
                f"gini={r['gini']:.3f}  max/mean={r['max_mean_ratio']:.2f}"
            )


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
    args = parser.parse_args()

    source = f"real ({args.real_histograms})" if args.real_histograms else f"synthetic ({args.distributions})"
    print(f"Auto-tuning TopK mapping hyperparameters")
    print(f"  batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"topk_val={args.topk_val}, num_kv_heads={args.num_kv_heads}")
    print(f"  score source: {source}")
    n_parametric = sum(len(v) for _, v in SWEEP_GRID.values())
    n_dists = 1 if args.real_histograms else len(args.distributions)
    print(f"  sweep: {n_parametric} parametric + {len(BASELINES)} baselines "
          f"= {n_parametric + len(BASELINES)} combos x {n_dists} dists")

    results = run_sweep(args)
    print_table(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
