"""
Auto-tune TopK mapping hyperparameters by profiled kernel latency.

For each (mode, hyperparameter) combo in the sweep grid, this script picks
the hyperparameter whose remapped score distribution produces the lowest
*unfused* topk kernel latency. The measurement is a split-phase:

  1. topk_remap_only(x, mode, power) → float32 buffer  [NOT timed]
  2. topk_output_sglang(remapped)                      [TIMED]

Timing only step 2 isolates the Stage-2 radix cost, which is what bucket
uniformity actually affects. The remap cost is the same constant regardless
of power, so it would only pollute the ranking.

Non-arithmetic baselines (MAPPING_LUT_CDF=1, MAPPING_QUANTILE=2,
MAPPING_TRUNC8=8) route their mapping through compute_stage1_bin, not
apply_transform, so split-phase is a no-op for them. Those are timed via
the fused kernel and marked `timing_mode="fused_fallback"` in the output.

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

from bench_topk import (
    make_topk_inputs,
    bench_kernel,
    compute_histogram_stats,
    scores_from_histogram,
)
from vortex_torch_C import (
    topk_output_sglang,
    topk_output_sglang_fused,
    topk_remap_only,
    topk_profile_histogram,
    topk_profile_counters,
)


# Modes where topk_mapping.cuh::apply_transform is a genuine value-space
# transform (power / asinh / log / log1p / erf / tanh / subtract / exp_stretch,
# plus the top-spreading shift_pow2 / shift_pow3 / linear_steep family) and
# also mode 0 (identity). For these the split-phase `remap_only + unfused
# topk` is correct. Modes 1/2/8 (LUT_CDF / QUANTILE / TRUNC8) apply their
# mapping inside compute_stage1_bin, so split-phase is a no-op.
ARITHMETIC_MODES = {0, 3, 4, 6, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20}


# Only parametric modes need auto-tuning. Mode 0 (none) and mode 4 (log)
# have no knob; mode 0 is always the baseline. Sweep grids widened so the
# autotune actually explores the tails of each transform.
SWEEP_GRID: Dict[int, List[float]] = {
    3:  [0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 9.0],               # power: p
    6:  [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],              # asinh: beta
    7:  [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],              # log1p: alpha
    9:  [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],              # erf: alpha
    10: [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],              # tanh: alpha
    11: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],            # subtract: pivot
    13: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],             # exp_stretch: alpha
    15: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],          # shift_pow2: pivot
    16: [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],      # shift_pow3: pivot (widened)
    17: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],             # linear_steep: k
    18: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],          # half_square: pivot
    19: [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],          # half_cube: pivot
    # dense_mant clamp: sweep a wide range because real attention scores
    # can span [-400, +200] on some models (raw logits), not just [0, 1].
    20: [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0],          # dense_mant: clamp pivot
}

PARAM_NAME = {
    3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha",
    11: "pivot", 13: "alpha",
    15: "pivot", 16: "pivot", 17: "k",
    18: "pivot", 19: "pivot",
    20: "clamp",
}
MODE_NAMES = {
    0: "none", 1: "lut_cdf", 2: "quantile",
    3: "power", 4: "log", 6: "asinh", 7: "log1p",
    8: "trunc8", 9: "erf", 10: "tanh", 11: "subtract", 13: "exp_stretch",
    15: "shift_pow2", 16: "shift_pow3", 17: "linear_steep",
    18: "half_square", 19: "half_cube",
    20: "dense_mant",
}

# Non-parametric modes — no knob to sweep; timed once as a reference point.
# LUT_CDF (1) and QUANTILE (2) are added here at runtime when the caller
# passes --lut-path / --quantiles-path.
BASELINES = [(0, 0.5), (4, 0.5), (8, 0.5)]


# ---------- Real-distribution score generation ----------
# _build_bin_range_table / scores_from_histogram now live in bench_topk.py
# so both autotune and bench_topk draw scores from the same sampler.


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
    x = scores_from_histogram(histogram, total_dense, device="cuda",
                               score_dtype=torch.bfloat16)
    remapped = torch.empty(total_dense, dtype=torch.float32, device="cuda").reshape(x.shape)

    return {
        "x": x,
        "remapped": remapped,
        "dense_kv_indptr": dense_kv_indptr,
        "sparse_kv_indptr": sparse_kv_indptr,
        "dense_kv_indices": dense_kv_indices,
        "sparse_kv_indices": sparse_kv_indices,
        "eff_batch_size": eff_bs,
        "num_pages_per_seg": num_pages_per_seg,
        "sparse_per_seg": sparse_per_seg,
    }


def _ensure_remapped_buffer(inputs: dict) -> torch.Tensor:
    """Lazy-allocate a float32 buffer matching x.shape for the split-phase."""
    buf = inputs.get("remapped")
    if buf is None:
        x = inputs["x"]
        buf = torch.empty(x.numel(), dtype=torch.float32, device=x.device).reshape(x.shape)
        inputs["remapped"] = buf
    return buf


# ---------- Latency-based evaluation ----------

def _time_fused(inputs, args, mode: int, power: float) -> dict:
    """Fused remap+topk kernel latency (used as fallback for modes 1/2/8)."""
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


def _time_unfused_on_remapped(inputs, args, mode: int, power: float) -> dict:
    """Time the unfused topk kernel on pre-remapped scores.

    For mode 0 the original scores are used directly. For every other
    arithmetic mode we run topk_remap_only once (not timed) into a
    pre-allocated float32 buffer, then time topk_output_sglang on that
    buffer with bench_kernel's warmup + repeat loop. This isolates the
    Stage-2 radix cost from the remap pass.
    """
    eff_bs = inputs["eff_batch_size"]
    pages_per_seg = inputs["num_pages_per_seg"]

    if mode == 0:
        src = inputs["x"]
    else:
        remapped = _ensure_remapped_buffer(inputs)
        topk_remap_only(
            inputs["x"],
            inputs["dense_kv_indptr"],
            remapped,
            eff_bs,
            args.reserved_bos,
            args.reserved_eos,
            mode,
            float(power),
        )
        torch.cuda.synchronize()
        src = remapped

    inputs["sparse_kv_indices"].zero_()
    call_args = (
        src,
        inputs["dense_kv_indptr"],
        inputs["sparse_kv_indptr"],
        inputs["dense_kv_indices"],
        inputs["sparse_kv_indices"],
        eff_bs,
        args.topk_val,
        args.reserved_bos,
        args.reserved_eos,
        pages_per_seg,
    )
    return bench_kernel(topk_output_sglang, call_args,
                        warmup=args.warmup, repeat=args.repeat)


def _time_mode(inputs, args, mode: int, power: float) -> tuple:
    """Returns (latency_dict, timing_mode_str)."""
    if mode in ARITHMETIC_MODES:
        return _time_unfused_on_remapped(inputs, args, mode, power), "unfused_on_remapped"
    return _time_fused(inputs, args, mode, power), "fused_fallback"


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
        # selected_from_thr = topk_val - num_above (clamped >= 0). Used as
        # a tie-breaker by bench_topk._load_autotune_hparams when several
        # modes have indistinguishable latency.
        sel_from_thr = (float(args.topk_val) - c[:, 1]).clamp(min=0.0)
        diag["selected_from_thr_mean"] = sel_from_thr.mean().item()

    return diag


def _run_sweep(args, inputs, dist_label: str) -> List[dict]:
    results = []

    # Baselines: time them but their param is fixed.
    for mode, power in BASELINES:
        lat, tmode = _time_mode(inputs, args, mode, power)
        entry = {
            "mode": mode,
            "mode_name": MODE_NAMES.get(mode, f"m{mode}"),
            "param_name": "(baseline)",
            "param": power,
            "distribution": dist_label,
            "timing_mode": tmode,
            "latency_ms": lat["mean_ms"],
            "latency_median_ms": lat["median_ms"],
            "latency_min_ms": lat["min_ms"],
        }
        entry.update(_collect_diagnostics(inputs, args, mode, power))
        results.append(entry)
        print(
            f"  mode={mode:>2d} ({MODE_NAMES[mode]:>10s}) baseline                "
            f"  [{tmode:>20s}]  latency={lat['mean_ms']:.4f} ms"
        )

    # Parametric sweep, one (mode, param) combo at a time.
    for mode, values in SWEEP_GRID.items():
        pname = PARAM_NAME[mode]
        for val in values:
            lat, tmode = _time_mode(inputs, args, mode, float(val))
            entry = {
                "mode": mode,
                "mode_name": MODE_NAMES.get(mode, f"m{mode}"),
                "param_name": pname,
                "param": float(val),
                "distribution": dist_label,
                "timing_mode": tmode,
                "latency_ms": lat["mean_ms"],
                "latency_median_ms": lat["median_ms"],
                "latency_min_ms": lat["min_ms"],
            }
            entry.update(_collect_diagnostics(inputs, args, mode, float(val)))
            results.append(entry)
            print(
                f"  mode={mode:>2d} ({MODE_NAMES[mode]:>10s}) {pname}={val:<6.3f}              "
                f"  [{tmode:>20s}]  latency={lat['mean_ms']:.4f} ms"
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

    # Modes 1 (LUT_CDF) and 2 (Quantile) are no longer evaluated — they
    # don't use topk_mapping::apply_transform (their mapping is done inside
    # compute_stage1_bin) and are kept out of the comparison entirely.
    args._mapping_lut = None
    args._mapping_quantiles = None

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
