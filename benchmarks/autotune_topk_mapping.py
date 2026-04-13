"""
Auto-tune TopK mapping hyperparameters by profiled kernel latency.

For each (mode, hyperparameter) combo in the sweep grid, this script runs
the fused remap+topk kernel (topk_output_sglang_fused) on synthetic or
real-distribution inputs, measures end-to-end latency with CUDA events,
and picks the hyperparameter with the lowest measured latency per mode.

Distribution statistics (gini, max/mean, counter-based Stage-2 cost) are
still collected for diagnostics, but they do NOT drive the ranking — the
ranking is purely latency-driven.

Usage:
    python benchmarks/autotune_topk_mapping.py \\
        --topk-val 2048 --batch-size 4 --seq-len 65536 --num-kv-heads 8 \\
        --real-histograms calibration/raw_histograms.npy \\
        --output-json autotune_results.json
"""

import argparse
import json
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from bench_topk import make_topk_inputs, bench_kernel, compute_histogram_stats
from vortex_torch_C import (
    topk_output_sglang_fused,
    topk_profile_histogram,
    topk_profile_counters,
)


# Only parametric modes need auto-tuning. Mode 0 (none) and mode 4 (log)
# have no knob; mode 0 is always the baseline.
SWEEP_GRID: Dict[int, List[float]] = {
    3:  [0.1, 0.25, 0.5, 0.75, 0.9],          # power: p
    6:  [0.1, 0.5, 1.0, 2.0, 4.0],            # asinh: beta
    7:  [0.1, 0.5, 1.0, 2.0, 4.0, 8.0],       # log1p: alpha
    9:  [0.1, 0.5, 1.0, 2.0, 4.0],            # erf: alpha
    10: [0.1, 0.5, 1.0, 2.0, 4.0],            # tanh: alpha
    11: [-1.0, -0.5, 0.0, 0.5, 1.0],          # subtract: pivot (free hparam)
    13: [0.5, 1.0, 2.0, 4.0, 8.0],            # exp_stretch: alpha
}

PARAM_NAME = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha", 11: "pivot", 13: "alpha"}
MODE_NAMES = {
    0: "none", 1: "lut_cdf", 2: "quantile",
    3: "power", 4: "log", 6: "asinh", 7: "log1p",
    8: "trunc8", 9: "erf", 10: "tanh", 11: "subtract", 13: "exp_stretch",
}

# Non-parametric modes — no knob to sweep; timed once as a reference point.
# LUT_CDF (1) and QUANTILE (2) are added here at runtime when the caller
# passes --lut-path / --quantiles-path.
BASELINES = [(0, 0.5), (4, 0.5), (8, 0.5)]


# ---------- Real-distribution score generation ----------

def _key_to_fp16(key: int) -> np.float16:
    """Invert convert_to_uint8's sign-flip for a single 16-bit key."""
    bits = (key & 0x7FFF) if key >= 0x8000 else ((~key) & 0xFFFF)
    return np.array([bits], dtype=np.uint16).view(np.float16)[0]


def _build_bin_range_table():
    """Return per-bin (lo, hi) fp16 value tables for all 256 radix bins."""
    all_bits = np.arange(65536, dtype=np.uint16)
    all_fp16 = all_bits.view(np.float16)
    keys = np.where(
        (all_bits & 0x8000).astype(bool),
        (~all_bits).astype(np.uint16),
        all_bits | np.uint16(0x8000),
    )
    bins = (keys >> 8).astype(np.uint8)
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
    empty = bin_lo > bin_hi
    for b in np.where(empty)[0]:
        val = float(_key_to_fp16((int(b) << 8) | 0x80))
        bin_lo[b] = val
        bin_hi[b] = val
    return bin_lo, bin_hi


def _scores_from_histogram(histogram: np.ndarray, total_pages: int, device="cuda") -> torch.Tensor:
    bin_lo, bin_hi = _build_bin_range_table()
    counts = histogram.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return torch.zeros(total_pages, 1, 1, dtype=torch.bfloat16, device=device)
    probs = counts / total
    bin_indices = np.random.choice(256, size=total_pages, p=probs)
    lo = bin_lo[bin_indices]
    hi = bin_hi[bin_indices]
    rand = np.random.uniform(0, 1, size=total_pages).astype(np.float32)
    scores_f32 = lo + rand * (hi - lo)
    return torch.from_numpy(scores_f32).to(torch.bfloat16).reshape(total_pages, 1, 1).to(device)


def _make_real_inputs(args, histogram: np.ndarray) -> dict:
    eff_bs = args.batch_size * args.num_kv_heads
    num_pages_per_seg = math.ceil(args.seq_len / args.page_size)
    total_dense = eff_bs * num_pages_per_seg
    sparse_per_seg = min(args.topk_val + args.reserved_bos + args.reserved_eos, num_pages_per_seg)

    dense_kv_indptr = torch.arange(
        0, (eff_bs + 1) * num_pages_per_seg, num_pages_per_seg,
        dtype=torch.int32, device="cuda",
    )
    sparse_kv_indptr = torch.arange(
        0, (eff_bs + 1) * sparse_per_seg, sparse_per_seg,
        dtype=torch.int32, device="cuda",
    )
    dense_kv_indices = torch.arange(total_dense, dtype=torch.int32, device="cuda")
    sparse_kv_indices = torch.zeros(eff_bs * sparse_per_seg, dtype=torch.int32, device="cuda")
    x = _scores_from_histogram(histogram, total_dense)

    return {
        "x": x,
        "dense_kv_indptr": dense_kv_indptr,
        "sparse_kv_indptr": sparse_kv_indptr,
        "dense_kv_indices": dense_kv_indices,
        "sparse_kv_indices": sparse_kv_indices,
        "eff_batch_size": eff_bs,
        "num_pages_per_seg": num_pages_per_seg,
        "sparse_per_seg": sparse_per_seg,
    }


# ---------- Latency-based evaluation ----------

def _time_fused(inputs, args, mode: int, power: float) -> dict:
    eff_bs = inputs["eff_batch_size"]
    pages_per_seg = inputs["num_pages_per_seg"]
    inputs["sparse_kv_indices"].zero_()
    lut_t = getattr(args, "_mapping_lut", None) if mode == 1 else None
    q_t   = getattr(args, "_mapping_quantiles", None) if mode == 2 else None
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
        mode,
        power,
        lut_t,
        q_t,
    )
    return bench_kernel(topk_output_sglang_fused, call_args,
                        warmup=args.warmup, repeat=args.repeat)


def _collect_diagnostics(inputs, args, mode: int, power: float) -> dict:
    """Optional distribution/counter stats for reporting only (post-timing)."""
    eff_bs = inputs["eff_batch_size"]
    pages_per_seg = inputs["num_pages_per_seg"]
    diag = {}
    lut_t = getattr(args, "_mapping_lut", None) if mode == 1 else None
    q_t   = getattr(args, "_mapping_quantiles", None) if mode == 2 else None

    if args.collect_stats:
        hist = torch.zeros(eff_bs, 256, dtype=torch.int32, device="cuda")
        topk_profile_histogram(
            inputs["x"], inputs["dense_kv_indptr"], hist,
            eff_bs, args.reserved_bos, args.reserved_eos,
            mode, power, lut_t, q_t,
        )
        torch.cuda.synchronize()
        diag.update(compute_histogram_stats(hist))

        counter_buf = torch.zeros(eff_bs, 6, dtype=torch.int32, device="cuda")
        inputs["sparse_kv_indices"].zero_()
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
            pages_per_seg,
            mode,
            power,
            lut_t,
            q_t,
        )
        torch.cuda.synchronize()
        c = counter_buf.float()
        diag["threshold_bin_mean"] = c[:, 0].mean().item()
        diag["num_equal_mean"]     = c[:, 2].mean().item()
        diag["refine_rounds_mean"] = c[:, 4].mean().item()

    return diag


def _run_sweep(args, inputs, dist_label: str) -> List[dict]:
    results = []

    # Baselines: time them but their param is fixed.
    for mode, power in BASELINES:
        lat = _time_fused(inputs, args, mode, power)
        entry = {
            "mode": mode,
            "mode_name": MODE_NAMES.get(mode, f"m{mode}"),
            "param_name": "(baseline)",
            "param": power,
            "distribution": dist_label,
            "latency_ms": lat["mean_ms"],
            "latency_median_ms": lat["median_ms"],
            "latency_min_ms": lat["min_ms"],
        }
        entry.update(_collect_diagnostics(inputs, args, mode, power))
        results.append(entry)
        print(
            f"  mode={mode:>2d} ({MODE_NAMES[mode]:>5s}) baseline                      "
            f"  latency={lat['mean_ms']:.4f} ms"
        )

    # Parametric sweep, one (mode, param) combo at a time.
    for mode, values in SWEEP_GRID.items():
        pname = PARAM_NAME[mode]
        for val in values:
            lat = _time_fused(inputs, args, mode, float(val))
            entry = {
                "mode": mode,
                "mode_name": MODE_NAMES.get(mode, f"m{mode}"),
                "param_name": pname,
                "param": float(val),
                "distribution": dist_label,
                "latency_ms": lat["mean_ms"],
                "latency_median_ms": lat["median_ms"],
                "latency_min_ms": lat["min_ms"],
            }
            entry.update(_collect_diagnostics(inputs, args, mode, float(val)))
            results.append(entry)
            print(
                f"  mode={mode:>2d} ({MODE_NAMES[mode]:>5s}) {pname}={val:<6.3f}                    "
                f"  latency={lat['mean_ms']:.4f} ms"
            )

    return results


def _print_ranked(results: List[dict]) -> None:
    ranked = sorted(results, key=lambda r: r["latency_ms"])
    header = (
        f"{'Rank':>4s}  {'Mode':<12s}  {'Param':<14s}  {'Dist':<10s}  {'Latency (ms)':>14s}"
    )
    print("\n" + "=" * len(header))
    print("TopK auto-tune results (ranked by measured kernel latency, lower is better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(ranked):
        param_str = f"{r['param_name']}={r['param']}" if r["param_name"] != "(baseline)" else "(baseline)"
        print(
            f"{i + 1:4d}  {r['mode_name']:<12s}  {param_str:<14s}  "
            f"{r['distribution']:<10s}  {r['latency_ms']:14.4f}"
        )
    print("=" * len(header))

    # Best per mode.
    best: Dict[int, dict] = {}
    for r in results:
        m = r["mode"]
        if m not in best or r["latency_ms"] < best[m]["latency_ms"]:
            best[m] = r
    print("\nBest per mode (by latency):")
    for m in sorted(best.keys()):
        r = best[m]
        param_str = f"{r['param_name']}={r['param']}" if r["param_name"] != "(baseline)" else "(baseline)"
        print(
            f"  mode {m:>2d} ({r['mode_name']:>5s}):  {param_str:<16s}  "
            f"latency={r['latency_ms']:.4f} ms"
        )


def main():
    parser = argparse.ArgumentParser("TopK mapping hyperparameter auto-tuner (latency-driven)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--topk-val", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--reserved-bos", type=int, default=1)
    parser.add_argument("--reserved-eos", type=int, default=2)
    parser.add_argument("--distributions", type=str, nargs="+",
                        default=["normal"],
                        help="Synthetic distributions when --real-histograms is not set.")
    parser.add_argument("--real-histograms", type=str, default=None,
                        help="Path to raw_histograms.npy from calibration.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--collect-stats", action="store_true",
                        help="Also collect histogram + counter diagnostics (post-timing, no cost).")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--lut-path", type=str, default=None,
                        help="Path to .npy uint8[256] LUT for MAPPING_LUT_CDF (mode 1).")
    parser.add_argument("--quantiles-path", type=str, default=None,
                        help="Path to .npy float32[256] quantile table for MAPPING_QUANTILE (mode 2).")
    args = parser.parse_args()

    args._mapping_lut = None
    args._mapping_quantiles = None
    # Include modes 1/2 as baselines when calibration tables are provided.
    if args.lut_path:
        lut_np = np.load(args.lut_path).astype(np.uint8)
        args._mapping_lut = torch.from_numpy(lut_np).cuda()
        BASELINES.append((1, 0.5))
        print(f"[autotune] loaded LUT from {args.lut_path}")
    if args.quantiles_path:
        q_np = np.load(args.quantiles_path).astype(np.float32)
        args._mapping_quantiles = torch.from_numpy(q_np).cuda()
        BASELINES.append((2, 0.5))
        print(f"[autotune] loaded quantiles from {args.quantiles_path}")

    real_histogram: Optional[np.ndarray] = None
    if args.real_histograms:
        raw = np.load(args.real_histograms)
        real_histogram = raw.sum(axis=0) if raw.ndim > 1 else raw

    all_results: List[dict] = []

    if real_histogram is not None:
        inputs = _make_real_inputs(args, real_histogram)
        print("\n=== Latency sweep on REAL distribution "
              f"(batch={args.batch_size} heads={args.num_kv_heads} seq={args.seq_len} topk={args.topk_val}) ===")
        all_results += _run_sweep(args, inputs, "real")
    else:
        for dist in args.distributions:
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
            print(f"\n=== Latency sweep on synthetic dist={dist} ===")
            all_results += _run_sweep(args, inputs, dist)

    _print_ranked(all_results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
