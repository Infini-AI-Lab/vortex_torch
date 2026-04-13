"""
TopK kernel benchmarking suite.

Lean rewrite after the remap-benchmark refactor. Exposes three public
helpers used by autotune_topk_mapping.py (make_topk_inputs, bench_kernel,
compute_histogram_stats) and a CLI with two modes:

  - default      : time the baseline (unmapped) kernel and the fused
                   kernel across a grid of (mode, power, batch, seq_len,
                   topk_val, distribution) configs.
  - --remap-bench: time baseline vs fused vs split-phase (remap-only +
                   unmapped-topk-on-remapped) and report threshold stats
                   from topk_profile_counters.
"""

import argparse
import json
import math
import statistics
from typing import Dict, List

import numpy as np
import torch

from vortex_torch_C import (
    topk_output,
    topk_output_sglang,          # unmapped baseline
    topk_output_sglang_fused,    # fused remap + topk
    topk_remap_only,             # standalone remap
    topk_profile_histogram,
    topk_profile_counters,
)


MAPPING_MODE_NAMES = {
    0: "None",
    1: "LUT_CDF",
    2: "Quantile",
    3: "Power",
    4: "Log",
    6: "Asinh",
    7: "Log1p",
    8: "Trunc8",
    9: "Erf",
    10: "Tanh",
    11: "Subtract",
    13: "ExpStretch",
}


def _load_autotune_hparams(path: str) -> Dict[int, float]:
    """Load per-mode best hyperparameters from an autotune_results.json.

    The JSON is produced by autotune_topk_mapping.py and contains a list of
    {mode, param, latency_ms, ...} entries. For each mode we pick the entry
    with the lowest measured latency and return {mode: best_param}.

    Modes with no parametric sweep (0=None, 4=Log) return a dummy 0.5; the
    caller should override to taste.
    """
    with open(path) as f:
        data = json.load(f)
    best: Dict[int, dict] = {}
    for r in data:
        m = r.get("mode")
        lat = r.get("latency_ms")
        if m is None or lat is None:
            continue
        if m not in best or lat < best[m]["latency_ms"]:
            best[m] = r
    return {m: float(r["param"]) for m, r in best.items()}


def make_topk_inputs(
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    page_size: int,
    topk_val: int,
    reserved_bos: int,
    reserved_eos: int,
    score_dtype: torch.dtype,
    distribution: str = "normal",
    device: str = "cuda",
) -> dict:
    """Synthesize CSR-formatted paged attention inputs for kernel timing."""
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

    if distribution == "normal":
        x = torch.randn(total_dense_pages, 1, 1, device=device)
    elif distribution == "lognormal":
        x = torch.randn(total_dense_pages, 1, 1, device=device).exp()
    elif distribution == "uniform":
        x = torch.rand(total_dense_pages, 1, 1, device=device)
    elif distribution == "bucket_uniform":
        # Uniform across all 256 fp16 radix buckets. Random uint16 bit
        # patterns → interpret as fp16. NaN/Inf patterns collapse to ±0.
        raw_bits = torch.randint(0, 65536, (total_dense_pages,), dtype=torch.int32, device=device)
        abs_bits = raw_bits & 0x7FFF
        raw_bits[abs_bits >= 0x7C00] = raw_bits[abs_bits >= 0x7C00] & 0x8000
        x = raw_bits.to(torch.int16).view(torch.float16).float().reshape(total_dense_pages, 1, 1)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    x = x.to(score_dtype)

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


def bench_kernel(kernel_fn, args, warmup: int = 10, repeat: int = 100) -> dict:
    """Time a kernel with CUDA events. Returns latency stats in ms."""
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        start_events[i].record()
        kernel_fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def compute_histogram_stats(histograms: torch.Tensor) -> dict:
    """Bin distribution statistics from histogram tensor [B, 256]."""
    h = histograms.float()
    h_sum = h.sum(dim=0)  # [256]
    nonzero = h_sum[h_sum > 0]
    if len(nonzero) == 0:
        return {
            "max_mean_ratio": 0.0, "std": 0.0, "gini": 0.0,
            "num_nonzero_bins": 0, "entropy": 0.0, "effective_bins": 0.0,
        }
    mean_val = nonzero.mean().item()
    max_val = nonzero.max().item()
    std_val = nonzero.std().item() if len(nonzero) > 1 else 0.0
    sorted_bins = nonzero.sort().values
    n = len(sorted_bins)
    idx = torch.arange(1, n + 1, device=sorted_bins.device, dtype=torch.float32)
    gini = (2.0 * (idx * sorted_bins).sum() / (n * sorted_bins.sum()) - (n + 1) / n).item()
    p = nonzero / nonzero.sum()
    entropy = -(p * p.log2()).sum().item()
    return {
        "max_mean_ratio": max_val / mean_val if mean_val > 0 else 0.0,
        "std": std_val,
        "gini": max(0.0, gini),
        "num_nonzero_bins": int(len(nonzero)),
        "entropy": entropy,
        "effective_bins": 2 ** entropy,
    }


def _collect_threshold_stats(inputs, topk_val, pages_per_seg, args, mode: int, power: float) -> dict:
    """Run topk_profile_counters once and aggregate threshold-bin stats.

    Profile kernel is invoked AFTER all latency measurements have finished,
    so the counter writes never contaminate timing.
    """
    eff_bs = inputs["eff_batch_size"]
    counter_buf = torch.zeros(eff_bs, 6, dtype=torch.int32, device="cuda")
    inputs["sparse_kv_indices"].zero_()
    lut_t = getattr(args, "_mapping_lut", None) if mode == 1 else None
    q_t   = getattr(args, "_mapping_quantiles", None) if mode == 2 else None
    topk_profile_counters(
        inputs["x"],
        inputs["dense_kv_indptr"],
        inputs["sparse_kv_indptr"],
        inputs["dense_kv_indices"],
        inputs["sparse_kv_indices"],
        counter_buf,
        eff_bs,
        topk_val,
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
    # Selected from threshold bin = topk_val - num_above (clamped >= 0).
    sel_from_thr = (float(topk_val) - c[:, 1]).clamp(min=0.0)
    return {
        "threshold_bin_mean": c[:, 0].mean().item(),
        "threshold_bin_max":  c[:, 0].max().item(),
        "num_above_mean":     c[:, 1].mean().item(),
        "threshold_bin_size_mean": c[:, 2].mean().item(),   # NUM_EQUAL
        "threshold_bin_size_max":  c[:, 2].max().item(),
        "selected_from_thr_mean":  sel_from_thr.mean().item(),
        "selected_from_thr_max":   sel_from_thr.max().item(),
        "refine_rounds_mean": c[:, 4].mean().item(),
    }


def _resolve_hparam(args, mode: int) -> float:
    """Pick the hyperparameter for a mode: autotune JSON wins, then --mapping-hparam."""
    if mode == 0:
        return 0.5  # unused for MAPPING_NONE
    hparams: Dict[int, float] = getattr(args, "_autotune_hparams", {}) or {}
    if mode in hparams:
        return hparams[mode]
    return args.mapping_hparam


def _remap_bench_one_config(args, batch_size, num_kv_heads, seq_len, topk_val,
                            distribution, modes: List[int]) -> dict:
    """Time baseline, fused, and split-phase for each mode at one config."""
    inputs = make_topk_inputs(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        page_size=args.page_size,
        topk_val=topk_val,
        reserved_bos=args.reserved_bos,
        reserved_eos=args.reserved_eos,
        score_dtype=torch.bfloat16,
        distribution=distribution,
    )
    eff_bs = inputs["eff_batch_size"]
    pages_per_seg = inputs["num_pages_per_seg"]
    total_dense = inputs["x"].numel()

    # Baseline: unmapped topk.
    baseline_args = (
        inputs["x"],
        inputs["dense_kv_indptr"],
        inputs["sparse_kv_indptr"],
        inputs["dense_kv_indices"],
        inputs["sparse_kv_indices"],
        eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
    )
    inputs["sparse_kv_indices"].zero_()
    baseline = bench_kernel(topk_output_sglang, baseline_args, args.warmup, args.repeat)

    # Pre-allocate the float32 buffer used for the split-phase (remap → baseline).
    remapped = torch.empty(total_dense, dtype=torch.float32, device="cuda").reshape(inputs["x"].shape)

    config = {
        "batch_size": batch_size,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
        "topk_val": topk_val,
        "distribution": distribution,
        "pages_per_seg": pages_per_seg,
        "baseline_ms": baseline["mean_ms"],
        "modes": [],
    }

    for mode in modes:
        power = _resolve_hparam(args, mode)

        lut_t = getattr(args, "_mapping_lut", None) if mode == 1 else None
        q_t   = getattr(args, "_mapping_quantiles", None) if mode == 2 else None
        fused_args = (
            inputs["x"],
            inputs["dense_kv_indptr"],
            inputs["sparse_kv_indptr"],
            inputs["dense_kv_indices"],
            inputs["sparse_kv_indices"],
            eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
            mode, power, lut_t, q_t,
        )
        inputs["sparse_kv_indices"].zero_()
        fused = bench_kernel(topk_output_sglang_fused, fused_args, args.warmup, args.repeat)

        # Split-phase timing: first the standalone remap, then the unmapped
        # topk on the remapped buffer.
        remap_args = (
            inputs["x"],
            inputs["dense_kv_indptr"],
            remapped,
            eff_bs, args.reserved_bos, args.reserved_eos,
            mode, power,
        )
        remap_only = bench_kernel(topk_remap_only, remap_args, args.warmup, args.repeat)

        split_topk_args = (
            remapped,
            inputs["dense_kv_indptr"],
            inputs["sparse_kv_indptr"],
            inputs["dense_kv_indices"],
            inputs["sparse_kv_indices"],
            eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
        )
        # Run remap once so the buffer is populated for warmup of topk-on-remapped.
        topk_remap_only(*remap_args)
        torch.cuda.synchronize()
        inputs["sparse_kv_indices"].zero_()
        split_topk = bench_kernel(topk_output_sglang, split_topk_args, args.warmup, args.repeat)

        # Counter collection is run AFTER all timing measurements for this mode
        # so it cannot affect the timings.
        stats = _collect_threshold_stats(inputs, topk_val, pages_per_seg, args, mode, power)

        row = {
            "mode": mode,
            "mode_name": MAPPING_MODE_NAMES.get(mode, f"m{mode}"),
            "power": power,
            "remap_ms": remap_only["mean_ms"],
            "topk_after_remap_ms": split_topk["mean_ms"],
            "split_total_ms": remap_only["mean_ms"] + split_topk["mean_ms"],
            "fused_ms": fused["mean_ms"],
            **stats,
        }
        config["modes"].append(row)

    return config


def _print_remap_table(results: List[dict]) -> None:
    header = (
        f"{'mode':<12s}  {'remap_us':>9s}  {'topk_us':>9s}  {'split_us':>9s}  "
        f"{'fused_us':>9s}  {'base_us':>9s}  {'thr_bin':>7s}  {'thr_size':>8s}  {'sel_thr':>7s}"
    )
    for cfg in results:
        banner = (
            f"\n[batch={cfg['batch_size']} heads={cfg['num_kv_heads']} "
            f"seq_len={cfg['seq_len']} topk={cfg['topk_val']} "
            f"dist={cfg['distribution']} pages_per_seg={cfg['pages_per_seg']}]"
        )
        print(banner)
        print("  Baseline: mapping_mode=0 (raw fp16 bucketing)")
        print(header)
        print("-" * len(header))
        base_us = cfg["baseline_ms"] * 1000.0
        for row in cfg["modes"]:
            label = f"{row['mode_name']}(p={row['power']})" if row["mode"] != 0 else "None"
            print(
                f"{label:<12s}  "
                f"{row['remap_ms'] * 1000.0:9.2f}  "
                f"{row['topk_after_remap_ms'] * 1000.0:9.2f}  "
                f"{row['split_total_ms'] * 1000.0:9.2f}  "
                f"{row['fused_ms'] * 1000.0:9.2f}  "
                f"{base_us:9.2f}  "
                f"{row['threshold_bin_mean']:7.1f}  "
                f"{row['threshold_bin_size_mean']:8.1f}  "
                f"{row['selected_from_thr_mean']:7.1f}"
            )


def _run_remap_bench(args) -> None:
    modes = [int(m) for m in args.mapping_modes]
    if 0 not in modes:
        modes = [0] + modes

    results = []
    for bs in args.batch_sizes:
        for heads in args.num_kv_heads:
            for seq_len in args.seq_lens:
                for topk_val in args.topk_vals:
                    for dist in args.distributions:
                        cfg = _remap_bench_one_config(
                            args, bs, heads, seq_len, topk_val, dist, modes,
                        )
                        results.append(cfg)

    _print_remap_table(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


def _run_latency_sweep(args) -> None:
    """Simple baseline-vs-fused latency sweep (no split-phase, no counters)."""
    modes = [int(m) for m in args.mapping_modes]
    results = []
    for bs in args.batch_sizes:
        for heads in args.num_kv_heads:
            for seq_len in args.seq_lens:
                for topk_val in args.topk_vals:
                    for dist in args.distributions:
                        inputs = make_topk_inputs(
                            batch_size=bs, num_kv_heads=heads, seq_len=seq_len,
                            page_size=args.page_size, topk_val=topk_val,
                            reserved_bos=args.reserved_bos, reserved_eos=args.reserved_eos,
                            score_dtype=torch.bfloat16, distribution=dist,
                        )
                        eff_bs = inputs["eff_batch_size"]
                        pages_per_seg = inputs["num_pages_per_seg"]
                        row_modes = []
                        for mode in modes:
                            power = _resolve_hparam(args, mode)
                            inputs["sparse_kv_indices"].zero_()
                            if mode == 0:
                                call = topk_output_sglang
                                call_args = (
                                    inputs["x"], inputs["dense_kv_indptr"],
                                    inputs["sparse_kv_indptr"], inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs, topk_val,
                                    args.reserved_bos, args.reserved_eos, pages_per_seg,
                                )
                            else:
                                call = topk_output_sglang_fused
                                lut_t = getattr(args, "_mapping_lut", None) if mode == 1 else None
                                q_t   = getattr(args, "_mapping_quantiles", None) if mode == 2 else None
                                call_args = (
                                    inputs["x"], inputs["dense_kv_indptr"],
                                    inputs["sparse_kv_indptr"], inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs, topk_val,
                                    args.reserved_bos, args.reserved_eos, pages_per_seg,
                                    mode, power, lut_t, q_t,
                                )
                            stats = bench_kernel(call, call_args, args.warmup, args.repeat)
                            row_modes.append({
                                "mode": mode, "mode_name": MAPPING_MODE_NAMES.get(mode, f"m{mode}"),
                                "power": power, "mean_ms": stats["mean_ms"],
                                "median_ms": stats["median_ms"],
                            })
                            print(
                                f"bs={bs} h={heads} seq={seq_len} topk={topk_val} "
                                f"dist={dist} mode={mode:>2d}  lat={stats['mean_ms']:.4f} ms"
                            )
                        results.append({
                            "batch_size": bs, "num_kv_heads": heads, "seq_len": seq_len,
                            "topk_val": topk_val, "distribution": dist, "modes": row_modes,
                        })

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


def main():
    p = argparse.ArgumentParser("TopK kernel benchmarks")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4])
    p.add_argument("--num-kv-heads", type=int, nargs="+", default=[8])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[8192])
    p.add_argument("--topk-vals", type=int, nargs="+", default=[30])
    p.add_argument("--distributions", type=str, nargs="+",
                   default=["normal"],
                   choices=["normal", "lognormal", "uniform", "bucket_uniform"])
    p.add_argument("--mapping-modes", type=int, nargs="+",
                   default=[0, 3, 6, 7],
                   help="Mapping modes to sweep (0=None, 3=Power, 6=Asinh, 7=Log1p, etc.)")
    p.add_argument("--mapping-hparam", "--mapping-power", type=float, default=0.5,
                   dest="mapping_hparam",
                   help="Fallback hyperparameter for every non-zero mapping mode when "
                        "no --autotune-json is provided: p for mode 3 (power), beta for "
                        "mode 6 (asinh), alpha for modes 7/9/10/13 (log1p/erf/tanh/exp_stretch).")
    p.add_argument("--autotune-json", type=str, default=None,
                   help="Path to autotune_results.json produced by autotune_topk_mapping.py. "
                        "When set, the per-mode hyperparameter with the lowest measured "
                        "latency in that file is used instead of --mapping-hparam.")
    p.add_argument("--lut-path", type=str, default=None,
                   help="Path to .npy uint8[256] LUT for MAPPING_LUT_CDF (mode 1).")
    p.add_argument("--quantiles-path", type=str, default=None,
                   help="Path to .npy float32[256] quantile table for MAPPING_QUANTILE (mode 2).")
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--reserved-bos", type=int, default=1)
    p.add_argument("--reserved-eos", type=int, default=2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--remap-bench", action="store_true",
                   help="Run the split-phase remap/topk/fused/baseline benchmark.")
    args = p.parse_args()

    args._autotune_hparams = {}
    if args.autotune_json:
        args._autotune_hparams = _load_autotune_hparams(args.autotune_json)
        print(f"[autotune] using best-latency hyperparameters from {args.autotune_json}:")
        for m, v in sorted(args._autotune_hparams.items()):
            print(f"  mode {m:>2d} -> {v}")

    args._mapping_lut = None
    args._mapping_quantiles = None
    if args.lut_path:
        lut_np = np.load(args.lut_path).astype(np.uint8)
        assert lut_np.shape == (256,), f"LUT must be [256], got {lut_np.shape}"
        args._mapping_lut = torch.from_numpy(lut_np).cuda()
        print(f"[mapping] loaded LUT from {args.lut_path}")
    if args.quantiles_path:
        q_np = np.load(args.quantiles_path).astype(np.float32)
        assert q_np.shape == (256,), f"quantiles must be [256], got {q_np.shape}"
        args._mapping_quantiles = torch.from_numpy(q_np).cuda()
        print(f"[mapping] loaded quantiles from {args.quantiles_path}")

    if args.remap_bench:
        _run_remap_bench(args)
    else:
        _run_latency_sweep(args)


if __name__ == "__main__":
    main()
