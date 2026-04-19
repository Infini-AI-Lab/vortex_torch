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
    topk_output,                 # full CUB BlockRadixSort topk (max 4096 pages/seg)
    topk_output_sglang,          # 2-stage radix approximate topk (unmapped baseline)
    topk_output_sglang_fused,    # fused remap + 2-stage radix topk
    topk_output_sglang_ori,      # original SGLang reference kernel
    topk_output_sglang_parallel, # multi-CTA split+merge variant of the fused kernel
    topk_remap_only,             # standalone value-space remap
    topk_profile_histogram,
    topk_profile_counters,
)

# topk_output's template ladder tops out at 8192 pages per segment
# (see topk.cu::topk_output, branches up to <= 8192). Runs larger than
# that hit TORCH_CHECK(false).
TOPK_OUTPUT_MAX_PAGES = 8192

# The ori kernel has TopK baked in at compile time. If setup.py was built
# with a different value, calls will fail; this is the topk_val that
# matches the current build of topk_sglang_ori.cu.
TOPK_ORI_BAKED_IN = 30


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
    15: "ShiftPow2",
    16: "ShiftPow3",
    17: "LinearSteep",
    18: "HalfSquare",
    19: "HalfCube",
    20: "DenseMant",
}

# Modes whose value-space transform is a real apply_transform() pass. Modes
# 1 (LUT_CDF), 2 (QUANTILE) and 8 (TRUNC8) apply their mapping inside
# compute_stage1_bin, not apply_transform — so `topk_remap_only` cannot
# reproduce them (the fp32 buffer would just contain the raw values). For
# those modes the split-phase numbers are N/A; only the fused kernel is a
# meaningful reference.
ARITHMETIC_MODES = {0, 3, 4, 6, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20}


_AUTOTUNE_TIE_TOLERANCE_MS = 0.0002  # ≈ CUDA event noise floor at this kernel size


def _auto_num_splits(eff_batch_size: int, pages_per_seg: int, topk_val: int) -> int:
    """Pick num_splits to balance Phase-1 and Phase-2 work on the parallel
    kernel.

    Phase-1 per CTA does O(pages/splits) work and runs eff_batch_size*splits
    CTAs in parallel; Phase-2 runs eff_batch_size CTAs each doing
    O(splits*topk) work on the merged candidate list. Assuming both phases
    hit SM saturation, total ≈ (pages/splits + splits*topk)/throughput,
    minimized at splits = sqrt(pages/topk). Cap at the SM-budget for
    eff_batch_size and the max_safe value (pages_per_seg // topk_val, past
    which Phase 1 partitions are smaller than topk_val and gain nothing).

    Returns 1 when splitting cannot help.
    """
    max_safe = max(1, pages_per_seg // max(1, topk_val))
    if max_safe <= 1 or eff_batch_size <= 0:
        return 1
    try:
        sm = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        sm = 132
    balanced = max(1, int(round((pages_per_seg / max(1, topk_val)) ** 0.5)))
    sm_budget = max(1, sm // max(1, eff_batch_size))
    return max(1, min(balanced, sm_budget, max_safe))


def _load_autotune_hparams(path: str) -> Dict[int, float]:
    """Load per-mode best hyperparameters from an autotune_results.json.

    The JSON is produced by autotune_topk_mapping.py and contains a list of
    {mode, param, latency_ms, num_equal_mean, selected_from_thr_mean, ...}
    entries. For each mode we group all sweep entries, find the lowest
    latency, then break ties (within `_AUTOTUNE_TIE_TOLERANCE_MS`) by:

    1. Smallest `num_equal_mean` (= thr_size). Stage-2 cost is O(thr_size),
       so a smaller threshold bin is a better proxy for real fused
       latency than the noisy `latency_ms` measurement.
    2. Smallest `selected_from_thr_mean`. How many pages the topk has to
       pull from the threshold bin during refinement.
    3. Lowest `latency_ms` again (final fallback).

    Modes with no parametric sweep (0=None, 4=Log) return a dummy 0.5;
    the caller should override to taste.
    """
    with open(path) as f:
        data = json.load(f)
    grouped: Dict[int, list] = {}
    for r in data:
        m = r.get("mode")
        lat = r.get("latency_ms")
        if m is None or lat is None:
            continue
        grouped.setdefault(m, []).append(r)

    best: Dict[int, dict] = {}
    for m, entries in grouped.items():
        min_lat = min(e["latency_ms"] for e in entries)
        contenders = [
            e for e in entries
            if e["latency_ms"] - min_lat <= _AUTOTUNE_TIE_TOLERANCE_MS
        ]
        # Tie-breakers: lowest num_equal_mean, then lowest sel_thr,
        # then lowest latency. Missing diagnostic fields → +inf so they
        # lose tie-breaks (we still keep them as fallback candidates).
        def _rank_key(e):
            return (
                e.get("num_equal_mean", float("inf")),
                e.get("selected_from_thr_mean", float("inf")),
                e["latency_ms"],
            )
        best[m] = min(contenders, key=_rank_key)

    return {m: float(r["param"]) for m, r in best.items()}


def _key_to_fp16(key: int) -> np.float16:
    """Invert convert_to_uint8's sign-flip for a single 16-bit key."""
    bits = (key & 0x7FFF) if key >= 0x8000 else ((~key) & 0xFFFF)
    return np.array([bits], dtype=np.uint16).view(np.float16)[0]


def build_bin_range_table():
    """Per-bin (lo, hi) fp16 value tables for the 256 Stage-1 radix bins.

    Shared by the real-distribution samplers in bench_topk.py and
    autotune_topk_mapping.py so both scripts generate identical inputs.
    """
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


def scores_from_histogram(
    histogram: np.ndarray,
    total_pages: int,
    device: str = "cuda",
    score_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Sample `total_pages` scores whose Stage-1 bucket distribution matches
    the given 256-bin histogram (produced by calibration). Each bucket is
    sampled uniformly over the fp16 range that maps into it."""
    bin_lo, bin_hi = build_bin_range_table()
    counts = histogram.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return torch.zeros(total_pages, 1, 1, dtype=score_dtype, device=device)
    probs = counts / total
    bin_indices = np.random.choice(256, size=total_pages, p=probs)
    lo = bin_lo[bin_indices]
    hi = bin_hi[bin_indices]
    rand = np.random.uniform(0, 1, size=total_pages).astype(np.float32)
    scores_f32 = lo + rand * (hi - lo)
    return torch.from_numpy(scores_f32).to(score_dtype).reshape(total_pages, 1, 1).to(device)


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
    real_histogram: np.ndarray = None,
    device: str = "cuda",
) -> dict:
    """Synthesize CSR-formatted paged attention inputs for kernel timing.

    When `real_histogram` is provided, scores are drawn from that 256-bin
    distribution (ignoring `distribution`) so the benchmark sees the same
    Stage-1 bucket distribution as the calibrated model.
    """
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

    if real_histogram is not None:
        x = scores_from_histogram(real_histogram, total_dense_pages, device=device,
                                  score_dtype=score_dtype)
    elif distribution == "normal":
        x = torch.randn(total_dense_pages, 1, 1, device=device).to(score_dtype)
    elif distribution == "lognormal":
        x = torch.randn(total_dense_pages, 1, 1, device=device).exp().to(score_dtype)
    elif distribution == "uniform":
        x = torch.rand(total_dense_pages, 1, 1, device=device).to(score_dtype)
    elif distribution == "bucket_uniform":
        # Uniform across all 256 fp16 radix buckets. Random uint16 bit
        # patterns → interpret as fp16. NaN/Inf patterns collapse to ±0.
        raw_bits = torch.randint(0, 65536, (total_dense_pages,), dtype=torch.int32, device=device)
        abs_bits = raw_bits & 0x7FFF
        raw_bits[abs_bits >= 0x7C00] = raw_bits[abs_bits >= 0x7C00] & 0x8000
        x = raw_bits.to(torch.int16).view(torch.float16).float().reshape(total_dense_pages, 1, 1).to(score_dtype)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

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
    """Run topk_profile_counters + topk_profile_histogram once and aggregate
    threshold-bin / bucket-distribution stats. Profile kernels run AFTER all
    latency measurements, so their writes never contaminate timing.
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

    # Run the 256-bin histogram profile to compute the rank_target_bins
    # metric: how many bins ABOVE the threshold bin (i.e. the bins whose
    # pages are selected without Stage-2 refinement) actually contain
    # selected pages, and the mean pages-per-such-bin.
    hist_buf = torch.zeros(eff_bs, 256, dtype=torch.int32, device="cuda")
    topk_profile_histogram(
        inputs["x"],
        inputs["dense_kv_indptr"],
        hist_buf,
        eff_bs,
        args.reserved_bos,
        args.reserved_eos,
        mode,
        power,
        lut_t,
        q_t,
    )
    torch.cuda.synchronize()

    thr_idx = counter_buf[:, 0].to(torch.int64)  # [eff_bs]
    hist = hist_buf.to(torch.int64)               # [eff_bs, 256]
    bin_ids = torch.arange(256, device="cuda", dtype=torch.int64).unsqueeze(0)  # [1, 256]
    above_mask = bin_ids > thr_idx.unsqueeze(1)   # [eff_bs, 256]
    above_populated = ((hist > 0) & above_mask).sum(dim=1).float()  # bins >thr with any pages
    pages_above = (hist * above_mask.to(torch.int64)).sum(dim=1).float()  # total pages in those bins
    # Mean pages per populated above-threshold bin (per-segment, then
    # averaged). Guard against divide-by-zero.
    pages_per_bin = torch.where(
        above_populated > 0,
        pages_above / above_populated,
        torch.zeros_like(above_populated),
    )

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
        # Rank-target metrics: how the top pages are actually spread.
        "above_bins_mean":       above_populated.mean().item(),
        "pages_per_above_bin_mean": pages_per_bin.mean().item(),
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
                            distribution, modes: List[int],
                            head_label: str = "all") -> dict:
    """Time baseline, fused, and split-phase for each mode at one config.

    `head_label` is metadata: ``"all"`` for the aggregated table (default),
    or a stringified head index ``"0".."N-1"`` for per-head benches. The
    caller is responsible for setting ``args._real_histogram`` to the
    head-sliced sub-histogram before invoking this function in per-head mode.
    """
    real_hist = getattr(args, "_real_histogram", None) if distribution == "real" else None
    inputs = make_topk_inputs(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        page_size=args.page_size,
        topk_val=topk_val,
        reserved_bos=args.reserved_bos,
        reserved_eos=args.reserved_eos,
        score_dtype=torch.bfloat16,
        distribution=distribution if distribution != "real" else "normal",
        real_histogram=real_hist,
    )
    eff_bs = inputs["eff_batch_size"]
    pages_per_seg = inputs["num_pages_per_seg"]
    total_dense = inputs["x"].numel()

    # Baseline = unmapped topk_output_sglang (CUB two-stage radix, the
    # kernel every mapped mode's split-phase ends up calling). This is
    # the `base_us` column and also what the `None` row reports, so
    # None's topk_us == base_us by construction.
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

    # Optional extra row: the full CUB BlockRadixSort topk from topk.cu.
    # This is a "true naive" — exact sort, no bucketing tricks — for A/B
    # against the 2-stage approximate baseline. Only runs when pages_per_seg
    # fits the kernel's template ladder (<= TOPK_OUTPUT_MAX_PAGES = 4096).
    naive_ms = None
    if pages_per_seg <= TOPK_OUTPUT_MAX_PAGES:
        naive_args = (
            inputs["x"],
            inputs["dense_kv_indptr"],
            inputs["dense_kv_indices"],   # NOTE: topk_output arg order differs
            inputs["sparse_kv_indptr"],   #       from topk_output_sglang
            inputs["sparse_kv_indices"],
            eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
        )
        inputs["sparse_kv_indices"].zero_()
        naive_ms = bench_kernel(
            topk_output, naive_args, args.warmup, args.repeat
        )["mean_ms"]

    # Optional extra row: the original SGLang kernel from topk_sglang_ori.cu,
    # compiled with TopK=TOPK_ORI_BAKED_IN. Only runs when topk_val matches
    # that constant; otherwise the row is skipped with a warning. It is NOT
    # used as the baseline — this is a separate A/B point so you can see the
    # ori-vs-naive gap at a glance.
    sglang_ori_ms = None
    if topk_val == TOPK_ORI_BAKED_IN:
        ori_indices = torch.empty(eff_bs, TOPK_ORI_BAKED_IN,
                                  dtype=torch.int32, device="cuda")
        ori_args = (
            inputs["x"],
            inputs["dense_kv_indptr"],
            ori_indices,
            eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
        )
        sglang_ori_ms = bench_kernel(
            topk_output_sglang_ori, ori_args, args.warmup, args.repeat
        )["mean_ms"]

    # Pre-allocate the float32 buffer used for the split-phase (remap → baseline).
    # Split-phase remapped buffer is **float32** to preserve Stage-2
    # refinement precision. The fused kernel computes transforms in
    # fp32 internally (so its Stage-2 sub-bin keys carry transform-
    # dependent bits in positions [15:0]); a narrower remapped buffer
    # (bf16 or fp16) would zero those bits on round-trip and change
    # the Stage-2 tie-break ordering vs the fused path. fp32 is the
    # only lossless choice. The kernel supports bf16 output too (see
    # topk_remap_only's dispatch table) for experimental paths, but we
    # don't use it here because correctness matters more than the
    # small memory-bandwidth win.
    remapped = torch.empty(total_dense, dtype=torch.float32, device="cuda").reshape(inputs["x"].shape)

    config = {
        "batch_size": batch_size,
        "num_kv_heads": num_kv_heads,
        "seq_len": seq_len,
        "topk_val": topk_val,
        "distribution": distribution,
        "pages_per_seg": pages_per_seg,
        "head": head_label,
        "baseline_ms": baseline["mean_ms"],
        "naive_ms": naive_ms,
        "sglang_ori_ms": sglang_ori_ms,
        "modes": [],
    }

    # Naive row — full CUB BlockRadixSort from topk.cu. No mapping, no
    # remap, no fused. Only populated when pages_per_seg fits the kernel.
    if naive_ms is not None:
        config["modes"].append({
            "mode": -2,            # sentinel so ranking/autotune skip it
            "mode_name": "Naive",
            "power": 0.5,
            "remap_ms": None,
            "topk_after_remap_ms": naive_ms,
            "split_total_ms": None,
            "fused_ms": None,
            "parallel_ms": None,
            "parallel_splits": None,
            "threshold_bin_mean": 0.0,
            "threshold_bin_max": 0.0,
            "num_above_mean": 0.0,
            "threshold_bin_size_mean": 0.0,
            "threshold_bin_size_max": 0.0,
            "selected_from_thr_mean": 0.0,
            "selected_from_thr_max": 0.0,
            "refine_rounds_mean": 0.0,
            "above_bins_mean": 0.0,
            "pages_per_above_bin_mean": 0.0,
        })

    # The None row is a pass-through to the naive baseline: no remap, no
    # fused, and topk_us == base_us by construction. Distribution metrics
    # are populated by running the profile kernels with mode=0 so the user
    # can see the unmapped Stage-1 bucket layout as a reference.
    none_stats = _collect_threshold_stats(
        inputs, topk_val, pages_per_seg, args, mode=0, power=0.5
    )
    config["modes"].append({
        "mode": 0,
        "mode_name": "None",
        "power": 0.5,
        "remap_ms": None,
        "topk_after_remap_ms": baseline["mean_ms"],
        "split_total_ms": None,
        "fused_ms": None,
        "parallel_ms": None,
        "parallel_splits": None,
        **none_stats,
    })

    # Extra row for the original SGLang kernel — only populated when the
    # build's baked-in TopK matches topk_val. Also a pass-through (no
    # remap, no fused); topk_us is the ori kernel latency.
    if sglang_ori_ms is not None:
        config["modes"].append({
            "mode": -1,           # sentinel so ranking/autotune skip it
            "mode_name": "sglang_ori",
            "power": 0.5,
            "remap_ms": None,
            "topk_after_remap_ms": sglang_ori_ms,
            "split_total_ms": None,
            "fused_ms": None,
            "parallel_ms": None,
            "parallel_splits": None,
            "threshold_bin_mean": 0.0,
            "threshold_bin_max": 0.0,
            "num_above_mean": 0.0,
            "threshold_bin_size_mean": 0.0,
            "threshold_bin_size_max": 0.0,
            "selected_from_thr_mean": 0.0,
            "selected_from_thr_max": 0.0,
            "refine_rounds_mean": 0.0,
            "above_bins_mean": 0.0,
            "pages_per_above_bin_mean": 0.0,
        })
    else:
        print(f"[bench-remap] sglang_ori row SKIPPED: topk_val={topk_val} != "
              f"TOPK_ORI_BAKED_IN ({TOPK_ORI_BAKED_IN}). Rebuild topk_sglang_ori.cu "
              f"with a matching TopK to enable the row.")

    for mode in modes:
        # Mode 0 is already emitted as the `None` row above (pass-through
        # to the ori baseline with no remap/fused). Skip to avoid a
        # duplicate row and a spurious fused-mode-0 measurement.
        if mode == 0:
            continue

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

        # Multi-CTA split+merge variant of the fused kernel. num_splits <= 1
        # delegates to the single-CTA fused path, so this is only a
        # meaningful extra data point when we can actually split.
        parallel_ms = None
        parallel_splits_used = None
        if getattr(args, "bench_parallel", False):
            splits = getattr(args, "num_splits", -1)
            if splits is None or splits < 1:
                splits = _auto_num_splits(eff_bs, pages_per_seg, topk_val)
            parallel_args = (
                inputs["x"],
                inputs["dense_kv_indptr"],
                inputs["sparse_kv_indptr"],
                inputs["dense_kv_indices"],
                inputs["sparse_kv_indices"],
                eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
                splits,
                mode, power, lut_t, q_t,
            )
            inputs["sparse_kv_indices"].zero_()
            parallel = bench_kernel(
                topk_output_sglang_parallel, parallel_args, args.warmup, args.repeat
            )
            parallel_ms = parallel["mean_ms"]
            parallel_splits_used = splits

        # Split-phase timing is only meaningful for arithmetic modes.
        # MAPPING_LUT_CDF / QUANTILE / TRUNC8 apply their mapping inside
        # compute_stage1_bin, which topk_remap_only cannot reproduce, so we
        # report N/A for the split-phase fields and rely on the fused kernel
        # as the only valid reference latency.
        if mode in ARITHMETIC_MODES:
            remap_args = (
                inputs["x"],
                inputs["dense_kv_indptr"],
                remapped,
                eff_bs, args.reserved_bos, args.reserved_eos,
                mode, power,
            )
            remap_only = bench_kernel(topk_remap_only, remap_args, args.warmup, args.repeat)

            # Populate the remapped buffer once so the unfused-topk warmup
            # iterations don't read stale data.
            topk_remap_only(*remap_args)
            torch.cuda.synchronize()
            split_topk_args = (
                remapped,
                inputs["dense_kv_indptr"],
                inputs["sparse_kv_indptr"],
                inputs["dense_kv_indices"],
                inputs["sparse_kv_indices"],
                eff_bs, topk_val, args.reserved_bos, args.reserved_eos, pages_per_seg,
            )
            inputs["sparse_kv_indices"].zero_()
            split_topk = bench_kernel(topk_output_sglang, split_topk_args, args.warmup, args.repeat)

            remap_ms = remap_only["mean_ms"]
            topk_after_remap_ms = split_topk["mean_ms"]
            split_total_ms = remap_ms + topk_after_remap_ms
        else:
            remap_ms = None
            topk_after_remap_ms = None
            split_total_ms = None

        # Counter collection is run AFTER all timing measurements for this mode
        # so it cannot affect the timings.
        stats = _collect_threshold_stats(inputs, topk_val, pages_per_seg, args, mode, power)

        row = {
            "mode": mode,
            "mode_name": MAPPING_MODE_NAMES.get(mode, f"m{mode}"),
            "power": power,
            "remap_ms": remap_ms,
            "topk_after_remap_ms": topk_after_remap_ms,
            "split_total_ms": split_total_ms,
            "fused_ms": fused["mean_ms"],
            "parallel_ms": parallel_ms,
            "parallel_splits": parallel_splits_used,
            **stats,
        }
        config["modes"].append(row)

    return config


# Stage-2 working-set cap, matches SMEM_INPUT_SIZE in fast_topk_clean_fused
# (32 KB dynamic smem / 2 ping-pong buffers / 4 bytes per int = 4096).
_STAGE2_SMEM_CAP = 4096


def _print_remap_table(results: List[dict]) -> None:
    # The printed table only carries metrics that participate in the
    # fused-kernel cost model. All purely-informational columns
    # (thr_bin / sel_thr / abv_bins / pg/bin) were dropped — they're
    # still in the JSON for downstream tools, just not in the table.
    header = (
        f"{'mode':<14s}  {'remap_ms':>9s}  {'topk_ms':>9s}  {'split_ms':>9s}  "
        f"{'fused_ms':>9s}  {'par_ms':>9s}  {'splits':>6s}  {'base_ms':>9s}  "
        f"{'s1p2_load':>9s}  {'eff_thr':>7s}  {'rounds':>6s}  {'s2_work':>8s}"
    )
    for cfg in results:
        banner = (
            f"\n[batch={cfg['batch_size']} heads={cfg['num_kv_heads']} "
            f"seq_len={cfg['seq_len']} topk={cfg['topk_val']} "
            f"dist={cfg['distribution']} pages_per_seg={cfg['pages_per_seg']} "
            f"head={cfg.get('head', 'all')}]"
        )
        print(banner)
        extra_notes = []
        if cfg.get("naive_ms") is not None:
            extra_notes.append("Naive row = topk.cu (CUB full sort)")
        if cfg.get("sglang_ori_ms") is not None:
            extra_notes.append("sglang_ori row = topk_sglang_ori.cu")
        notes_str = ""
        if extra_notes:
            notes_str = "  |  " + "  |  ".join(extra_notes)
        print(f"  Baseline: topk_sglang.cu (CUB two-stage){notes_str}")
        print(
            f"  s1p2_load = thr_size (uncapped global re-reads in Stage-1 pass 2)   "
            f"eff_thr = min(thr_size, {_STAGE2_SMEM_CAP})   "
            f"rounds = stage-2 passes (1..4)   "
            f"s2_work = rounds * eff_thr"
        )
        print(header)
        print("-" * len(header))
        base_ms = cfg["baseline_ms"]
        for row in cfg["modes"]:
            if row["mode"] == 0:
                label = "None"
            elif row["mode"] == -1:
                label = row.get("mode_name", "sglang_ori")
            elif row["mode"] == -2:
                label = row.get("mode_name", "Naive")
            else:
                label = f"{row['mode_name']}(p={row['power']})"
            def _fmt(v):
                return f"{v:9.4f}" if v is not None else f"{'N/A':>9s}"
            fused_str = _fmt(row.get("fused_ms"))
            par_str   = _fmt(row.get("parallel_ms"))
            splits    = row.get("parallel_splits")
            splits_str = f"{splits:>6d}" if splits is not None else f"{'N/A':>6s}"
            thr_size  = row.get("threshold_bin_size_mean", 0.0)
            rounds    = row.get("refine_rounds_mean", 0.0)
            eff_thr   = min(thr_size, float(_STAGE2_SMEM_CAP))
            s2_work   = rounds * eff_thr
            s1p2_load = thr_size  # alias: same number, named for the cost-model role
            print(
                f"{label:<14s}  "
                f"{_fmt(row['remap_ms'])}  "
                f"{_fmt(row['topk_after_remap_ms'])}  "
                f"{_fmt(row['split_total_ms'])}  "
                f"{fused_str}  "
                f"{par_str}  "
                f"{splits_str}  "
                f"{base_ms:9.4f}  "
                f"{s1p2_load:9.0f}  "
                f"{eff_thr:7.0f}  "
                f"{rounds:6.2f}  "
                f"{s2_work:8.0f}"
            )


def _combine_per_head_cfgs(per_head_cfgs: List[dict]) -> dict:
    """Combine a list of per-head cfg dicts (same shape, head='0','1',...)
    into a single aggregated cfg tagged head='all', by averaging every
    numeric field. This is used when --per-head-bench is on so the
    aggregated row reflects the realistic per-head behaviour rather than
    a separate kernel launch on an averaged histogram.

    Assumes every cfg has the same `modes` list in the same order — which
    holds because all per-head sub-runs use identical (batch, heads, seq,
    topk, page_size, reserved, mapping_modes) parameters and therefore
    take the same code paths through `_remap_bench_one_config`.
    """
    assert per_head_cfgs, "_combine_per_head_cfgs called with empty list"
    base = per_head_cfgs[0]
    n_modes = len(base["modes"])
    # Sanity: same shape.
    for c in per_head_cfgs[1:]:
        assert len(c["modes"]) == n_modes, (
            f"per-head cfgs disagree on mode count: {n_modes} vs {len(c['modes'])}"
        )

    def _mean_or_none(vals):
        vs = [v for v in vals if v is not None]
        return (sum(vs) / len(vs)) if vs else None

    combined: Dict = {
        "batch_size":   base["batch_size"],
        "num_kv_heads": base["num_kv_heads"],
        "seq_len":      base["seq_len"],
        "topk_val":     base["topk_val"],
        "distribution": base["distribution"],
        "pages_per_seg": base["pages_per_seg"],
        "head":         "all",
        "baseline_ms":  _mean_or_none([c.get("baseline_ms")  for c in per_head_cfgs]),
        "naive_ms":     _mean_or_none([c.get("naive_ms")     for c in per_head_cfgs]),
        "sglang_ori_ms": _mean_or_none([c.get("sglang_ori_ms") for c in per_head_cfgs]),
        "modes": [],
    }

    # Numeric fields per mode row that we average; non-numeric fields (mode,
    # mode_name, power) are copied from the first cfg since they're identical
    # across heads by construction.
    NUMERIC_KEYS = (
        "remap_ms", "topk_after_remap_ms", "split_total_ms", "fused_ms",
        "parallel_ms",
        "threshold_bin_mean", "threshold_bin_max",
        "num_above_mean",
        "threshold_bin_size_mean", "threshold_bin_size_max",
        "selected_from_thr_mean", "selected_from_thr_max",
        "refine_rounds_mean",
        "above_bins_mean", "pages_per_above_bin_mean",
    )
    for mi in range(n_modes):
        sample = base["modes"][mi]
        merged = {
            "mode":      sample["mode"],
            "mode_name": sample["mode_name"],
            "power":     sample["power"],
        }
        for key in NUMERIC_KEYS:
            merged[key] = _mean_or_none([c["modes"][mi].get(key) for c in per_head_cfgs])
        combined["modes"].append(merged)
    return combined


def _run_remap_bench(args) -> None:
    modes = [int(m) for m in args.mapping_modes]
    # Mode 0 is emitted as the "None" row from _remap_bench_one_config
    # itself (pass-through to the ori baseline). Drop any user-supplied 0
    # to avoid a duplicate row.
    modes = [m for m in modes if m != 0]

    distributions = list(args.distributions)
    if getattr(args, "_real_histogram", None) is not None:
        if "real" not in distributions:
            distributions.append("real")
        print(f"[remap-bench] 'real' distribution enabled "
              f"(histogram total count = {int(args._real_histogram.sum())})")

    if getattr(args, "per_head_bench", False):
        if getattr(args, "_real_histograms_raw", None) is None:
            raise SystemExit(
                "[bench-remap] --per-head-bench requires --real-histograms with a 2D raw file."
            )
        if not args.num_kv_heads or any(h <= 0 for h in args.num_kv_heads):
            raise SystemExit("[bench-remap] --per-head-bench requires --num-kv-heads > 0.")
        # When the user passes multiple --num-kv-heads values we slice by the
        # first one (the others are degenerate for per-head reporting since
        # the histogram file has a fixed head count).
        per_head_count = int(args.num_kv_heads[0])

    results = []
    # When --per-head-bench is on, each "real"-distribution aggregate is
    # built by averaging the 8 per-head measurements (NOT by running an
    # extra kernel on an averaged histogram). This grouping keeps the
    # per-head cfgs that should fold into each (bs, heads, seq, topk)
    # aggregate point.
    per_head_groups: dict = {}

    # ---- Per-head tables (printed first) ----
    if getattr(args, "per_head_bench", False):
        raw = args._real_histograms_raw
        saved_agg = args._real_histogram
        try:
            for h in range(per_head_count):
                # Slice rows belonging to head `h`. Rows are interleaved as
                # row_idx % num_kv_heads = head_idx, so this strided slice
                # collects all (call, batch, h) triples across the file.
                args._real_histogram = raw[h::per_head_count].sum(axis=0)
                for bs in args.batch_sizes:
                    for heads in args.num_kv_heads:
                        for seq_len in args.seq_lens:
                            for topk_val in args.topk_vals:
                                cfg = _remap_bench_one_config(
                                    args, bs, heads, seq_len, topk_val, "real", modes,
                                    head_label=str(h),
                                )
                                results.append(cfg)
                                per_head_groups.setdefault(
                                    (bs, heads, seq_len, topk_val), []
                                ).append(cfg)
        finally:
            args._real_histogram = saved_agg

    # ---- Aggregated tables (printed last) ----
    for bs in args.batch_sizes:
        for heads in args.num_kv_heads:
            for seq_len in args.seq_lens:
                for topk_val in args.topk_vals:
                    for dist in distributions:
                        if dist == "real" and getattr(args, "per_head_bench", False):
                            cfgs = per_head_groups.get((bs, heads, seq_len, topk_val), [])
                            if cfgs:
                                # Combine the per-head cfgs into a single
                                # aggregated row — no extra kernel launch.
                                cfg = _combine_per_head_cfgs(cfgs)
                                results.append(cfg)
                                continue
                        cfg = _remap_bench_one_config(
                            args, bs, heads, seq_len, topk_val, dist, modes,
                            head_label="all",
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
    distributions = list(args.distributions)
    if getattr(args, "_real_histogram", None) is not None and "real" not in distributions:
        distributions.append("real")
    results = []
    for bs in args.batch_sizes:
        for heads in args.num_kv_heads:
            for seq_len in args.seq_lens:
                for topk_val in args.topk_vals:
                    for dist in distributions:
                        real_hist = args._real_histogram if dist == "real" else None
                        inputs = make_topk_inputs(
                            batch_size=bs, num_kv_heads=heads, seq_len=seq_len,
                            page_size=args.page_size, topk_val=topk_val,
                            reserved_bos=args.reserved_bos, reserved_eos=args.reserved_eos,
                            score_dtype=torch.bfloat16,
                            distribution=dist if dist != "real" else "normal",
                            real_histogram=real_hist,
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
                   choices=["normal", "lognormal", "uniform", "bucket_uniform", "real"],
                   help="Synthetic distributions. Use 'real' (or --real-histograms) to "
                        "sample scores from a calibrated raw_histograms.npy.")
    p.add_argument("--real-histograms", type=str, default=None,
                   help="Path to raw_histograms.npy from calibrate_topk.py. When set, a "
                        "'real' distribution is appended to the sweep so every "
                        "(mode, hparam) combo is also timed on the calibrated score "
                        "distribution.")
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
    p.add_argument("--bench-parallel", action="store_true",
                   help="Also time topk_output_sglang_parallel (multi-CTA split+merge).")
    p.add_argument("--num-splits", type=int, default=-1,
                   help="Partitions per batch for the parallel kernel. -1 = auto "
                        "(sm_count / eff_batch_size, clamped to pages_per_seg/topk_val).")
    p.add_argument("--per-head-bench", action="store_true",
                   help="In addition to the aggregated 'real'-distribution table, also "
                        "run the remap-bench once per KV head: slice the calibrated "
                        "histogram into one sub-histogram per head (using "
                        "row_idx %% num_kv_heads = head_idx), bench each, and print one "
                        "table per head followed by the aggregated table. Requires "
                        "--real-histograms (with a 2D raw file) and --num-kv-heads.")
    args = p.parse_args()

    args._autotune_hparams = {}
    if args.autotune_json:
        args._autotune_hparams = _load_autotune_hparams(args.autotune_json)
        print(f"[autotune] using best-latency hyperparameters from {args.autotune_json}:")
        for m, v in sorted(args._autotune_hparams.items()):
            print(f"  mode {m:>2d} -> {v}")

    args._real_histogram = None
    args._real_histograms_raw = None
    if args.real_histograms:
        # mmap_mode='r' keeps the (potentially 20+ GB) raw file off-heap; we
        # only materialise per-head sums when --per-head-bench is set.
        raw = np.load(args.real_histograms, mmap_mode='r')
        args._real_histogram = raw.sum(axis=0) if raw.ndim > 1 else raw
        if raw.ndim > 1:
            args._real_histograms_raw = raw
        print(f"[real] loaded calibrated histogram from {args.real_histograms} "
              f"(shape={raw.shape} → [256] aggregate)")

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
