"""
TopK kernel benchmarking suite.

Measures kernel-level latency for the three topk variants (naive/CUB,
sglang with mapping modes) across configurable grid of batch sizes,
sequence lengths, topk values, and KV head counts.

Usage:
    python benchmarking/bench_topk.py --batch-sizes 4 8 --seq-lens 2048 4096 --topk-vals 30 --num-kv-heads 2 --repeat 50
"""

import argparse
import json
import math
import statistics
from typing import Dict, List, Optional

import numpy as np
import torch

from vortex_torch_C import (
    topk_output, topk_output_sglang, topk_output_sglang_ori, topk_profile_histogram,
    topk_profile_stage1, topk_profile_counters,
)

# Canonical mapping mode names — used in logs, tables, and plots
MAPPING_MODE_NAMES = {
    0: "None",
    1: "LUT CDF",
    2: "Quantile",
    3: "Power",
    4: "Log",
    5: "Index Cache",
    6: "Asinh",
    7: "Log1p",
    8: "Trunc8",
    9: "Erf",
    10: "Tanh",
    11: "Subtract",
    13: "ExpStretch",
    14: "TopkWindow",
}

MAPPING_MODE_FORMULAS = {
    0: "None (fp16 bucketing)",
    1: "LUT CDF (calibrated)",
    2: "Quantile (calibrated)",
    3: "Power: sign(x)*|x|^p",
    4: "Log: sign(x)*log(|x|+1)",
    5: "Index Cache",
    6: "Asinh: asinh(beta*x)",
    7: "Log1p: sign(x)*log1p(alpha*|x|)",
    8: "Trunc8: bf16 upper-8-bit bucketing",
    9: "Erf: erf(alpha*x)",
    10: "Tanh: tanh(alpha*x)",
    11: "Subtract: x - pivot (RadiK-style)",
    13: "ExpStretch: exp(alpha*x)",
    14: "TopkWindow: k-aware linear windowing",
}


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
    """Synthesize realistic CSR-formatted paged attention inputs."""
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

    # Generate scores with the requested distribution
    if distribution == "normal":
        x = torch.randn(total_dense_pages, 1, 1, device=device)
    elif distribution == "lognormal":
        x = torch.randn(total_dense_pages, 1, 1, device=device).exp()
    elif distribution == "uniform":
        x = torch.rand(total_dense_pages, 1, 1, device=device)
    elif distribution == "bucket_uniform":
        # Uniform across all 256 fp16 radix buckets.
        # Random uint16 bit patterns → interpret as fp16.
        # Bucket = upper 8 bits of sign-flipped fp16, so random bits → uniform buckets.
        raw_bits = torch.randint(0, 65536, (total_dense_pages,), dtype=torch.int32, device=device)
        # Exclude fp16 NaN/Inf (exponent=31, i.e. |bits| >= 0x7C00)
        abs_bits = raw_bits & 0x7FFF
        raw_bits[abs_bits >= 0x7C00] = raw_bits[abs_bits >= 0x7C00] & 0x8000  # → ±0
        # Reinterpret int16 bits as fp16, then widen to float32
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
    """Time a kernel with CUDA events, return latency stats in ms."""
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
    """Compute bin distribution statistics from histogram tensor [B, 256]."""
    h = histograms.float()
    # Aggregate across batch dimension
    h_sum = h.sum(dim=0)  # [256]
    nonzero_bins = h_sum[h_sum > 0]
    if len(nonzero_bins) == 0:
        return {
            "max_mean_ratio": 0.0, "std": 0.0, "gini": 0.0,
            "num_nonzero_bins": 0, "entropy": 0.0, "effective_bins": 0.0,
        }

    mean_val = nonzero_bins.mean().item()
    max_val = nonzero_bins.max().item()
    std_val = nonzero_bins.std().item() if len(nonzero_bins) > 1 else 0.0

    # Gini coefficient
    sorted_bins = nonzero_bins.sort().values
    n = len(sorted_bins)
    index = torch.arange(1, n + 1, device=sorted_bins.device, dtype=torch.float32)
    gini = (2.0 * (index * sorted_bins).sum() / (n * sorted_bins.sum()) - (n + 1) / n).item()

    # Shannon entropy (base-2)
    p = nonzero_bins / nonzero_bins.sum()
    entropy = -(p * p.log2()).sum().item()
    # Effective number of bins: 2^entropy
    effective_bins = 2 ** entropy

    return {
        "max_mean_ratio": max_val / mean_val if mean_val > 0 else 0.0,
        "std": std_val,
        "gini": max(0.0, gini),
        "num_nonzero_bins": int(len(nonzero_bins)),
        "entropy": entropy,
        "effective_bins": effective_bins,
    }


NUM_HISTOGRAM_BINS = 256


def _histogram_target_pages(pages_per_seg: int, min_samples_per_bin: int = 512) -> int:
    """Compute adaptive page count for statistically reliable histograms.

    With 256 radix bins, each bin needs enough samples for stable gini /
    max-mean statistics.  Returns a total page count rounded up to a full
    segment boundary so every segment contributes equally.
    """
    min_pages = min_samples_per_bin * NUM_HISTOGRAM_BINS
    return math.ceil(min_pages / pages_per_seg) * pages_per_seg


def _load_autotune_powers(path: str) -> Dict[int, float]:
    """Extract best per-mode power from autotune JSON.

    Ranks by res_rate_mean (higher=better) if present, else by gini (lower=better).
    Returns {mode: best_power}, e.g. {3: 0.25, 6: 1.0, 7: 2.0}.
    """
    with open(path) as f:
        data = json.load(f)

    has_res_rate = any("res_rate_mean" in r for r in data)

    best: Dict[int, dict] = {}
    for r in data:
        m = r.get("mode")
        if m not in (3, 6, 7, 9, 10, 13, 14):
            continue
        if has_res_rate:
            score = r.get("res_rate_mean", 0.0)
            is_better = m not in best or score > best[m]["_score"]
        else:
            score = r.get("gini", 1.0)
            is_better = m not in best or score < best[m]["_score"]
        if is_better:
            best[m] = {"param": r["param"], "_score": score}

    return {m: v["param"] for m, v in best.items()}


def _resolve_mode_power(args, mode: int) -> float:
    """Return the power/beta/alpha for a parametric mapping mode.

    Priority: per-mode CLI flag > autotune JSON > global --mapping-power.
    """
    per_mode_flag = {3: args.mapping_power_3, 6: args.mapping_power_6, 7: args.mapping_power_7,
                     9: getattr(args, 'mapping_power_9', None), 10: getattr(args, 'mapping_power_10', None),
                     13: getattr(args, 'mapping_power_13', None), 14: getattr(args, 'mapping_power_14', None)}
    if mode in per_mode_flag and per_mode_flag[mode] is not None:
        return per_mode_flag[mode]
    if hasattr(args, "_autotune_powers") and mode in args._autotune_powers:
        return args._autotune_powers[mode]
    return args.mapping_power


def run_benchmark(args) -> List[dict]:
    """Run the full benchmark sweep and return results."""
    # Load autotune results if provided
    if args.autotune_json:
        args._autotune_powers = _load_autotune_powers(args.autotune_json)
        print(f"Loaded autotune best powers: {args._autotune_powers}")
    else:
        args._autotune_powers = {}

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    score_dtype = dtype_map[args.score_dtype]

    # Load real histogram if provided
    real_histogram = None
    _scores_from_histogram = None
    if args.real_histograms:
        from autotune_topk_mapping import scores_from_histogram
        _scores_from_histogram = scores_from_histogram
        raw = np.load(args.real_histograms)
        real_histogram = raw.sum(axis=0) if raw.ndim > 1 else raw

    # Extend distributions with "real" if calibration data is provided
    distributions = list(args.distributions)
    if real_histogram is not None:
        distributions.append("real")
    args.distributions = distributions

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"TopK Kernel Benchmark Results")
    print(f"GPU: {gpu_name} | SM count: {gpu_props.multi_processor_count}")
    print(f"Score dtype: {args.score_dtype} | Warmup: {args.warmup} | Repeat: {args.repeat}")
    print("=" * 90)

    # Load optional LUT / quantiles
    mapping_lut = None
    mapping_quantiles = None
    if args.lut_path:
        lut_np = np.load(args.lut_path).astype(np.uint8)
        mapping_lut = torch.from_numpy(lut_np).cuda()
    if args.quantiles_path:
        q_np = np.load(args.quantiles_path).astype(np.float32)
        mapping_quantiles = torch.from_numpy(q_np).cuda()

    # Build kernel list
    all_kernels = {
        "naive": "naive",
        "sglang_ori": "sglang_ori",
        "sglang_m0": "sglang_m0",
        "sglang_scale": "sglang_scale",  # mode 3 with p=1.0 (identity + linear auto-range scaling)
        "sglang_m3": "sglang_m3",
        "sglang_m3_noscale": "sglang_m3_noscale",
        "sglang_m4": "sglang_m4",
        "sglang_m6": "sglang_m6",
        "sglang_m6_noscale": "sglang_m6_noscale",
        "sglang_m7": "sglang_m7",
        "sglang_m7_noscale": "sglang_m7_noscale",
        "sglang_m8": "sglang_m8",
        "sglang_m9": "sglang_m9",
        "sglang_m9_noscale": "sglang_m9_noscale",
        "sglang_m10": "sglang_m10",
        "sglang_m10_noscale": "sglang_m10_noscale",
        "sglang_m11": "sglang_m11",
        "sglang_m13": "sglang_m13",
        "sglang_m13_noscale": "sglang_m13_noscale",
        "sglang_m14": "sglang_m14",
    }
    if mapping_lut is not None:
        all_kernels["sglang_m1"] = "sglang_m1"
    if mapping_quantiles is not None:
        all_kernels["sglang_m2"] = "sglang_m2"

    if args.filter_kernels:
        # Validate: if the user explicitly requested sglang_m1 or sglang_m2 but
        # the required calibration file was not provided, fail loudly instead of
        # silently skipping these modes.
        if "sglang_m1" in args.filter_kernels and "sglang_m1" not in all_kernels:
            raise RuntimeError(
                "sglang_m1 (LUT CDF) was requested in --filter-kernels but no "
                "--lut-path was provided.  Mode 1 requires a calibrated LUT file "
                "(lut.npy from calibrate_topk.py).  Either supply --lut-path or "
                "remove sglang_m1 from --filter-kernels."
            )
        if "sglang_m2" in args.filter_kernels and "sglang_m2" not in all_kernels:
            raise RuntimeError(
                "sglang_m2 (Quantile) was requested in --filter-kernels but no "
                "--quantiles-path was provided.  Mode 2 requires a calibrated "
                "quantiles file (quantiles.npy from calibrate_topk.py).  Either "
                "supply --quantiles-path or remove sglang_m2 from --filter-kernels."
            )
        all_kernels = {k: v for k, v in all_kernels.items() if k in args.filter_kernels}

    # Naive kernel only supports bf16
    if score_dtype != torch.bfloat16 and "naive" in all_kernels:
        print(f"Note: naive kernel only supports bfloat16, skipping for {args.score_dtype}")
        del all_kernels["naive"]

    all_results = []

    for bs in args.batch_sizes:
        for seq_len in args.seq_lens:
            for topk_val in args.topk_vals:
                for num_kv_heads in args.num_kv_heads:
                    for dist in args.distributions:
                        if dist == "real" and real_histogram is not None:
                            inputs = make_topk_inputs(
                                batch_size=bs,
                                num_kv_heads=num_kv_heads,
                                seq_len=seq_len,
                                page_size=args.page_size,
                                topk_val=topk_val,
                                reserved_bos=args.reserved_bos,
                                reserved_eos=args.reserved_eos,
                                score_dtype=score_dtype,
                                distribution="normal",
                            )
                            # Replace scores with real-distribution scores
                            total_dense = inputs["eff_batch_size"] * inputs["num_pages_per_seg"]
                            inputs["x"] = _scores_from_histogram(
                                real_histogram, total_dense, device="cuda",
                            )
                        else:
                            inputs = make_topk_inputs(
                                batch_size=bs,
                                num_kv_heads=num_kv_heads,
                                seq_len=seq_len,
                                page_size=args.page_size,
                                topk_val=topk_val,
                                reserved_bos=args.reserved_bos,
                                reserved_eos=args.reserved_eos,
                                score_dtype=score_dtype,
                                distribution=dist,
                            )

                        eff_bs = inputs["eff_batch_size"]
                        pages_per_seg = inputs["num_pages_per_seg"]

                        config_str = (
                            f"bs={bs} | seq={seq_len} | topk={topk_val} | "
                            f"heads={num_kv_heads} | pages/seg={pages_per_seg} | dist={dist}"
                        )
                        print(f"\n{config_str}")

                        config_results = {
                            "batch_size": bs,
                            "seq_len": seq_len,
                            "topk_val": topk_val,
                            "num_kv_heads": num_kv_heads,
                            "distribution": dist,
                            "eff_batch_size": eff_bs,
                            "pages_per_seg": pages_per_seg,
                            "kernels": {},
                        }

                        # Collect all kernel results first, then print sorted by latency
                        kernel_entries = []  # [(label, kernel_name, result)]

                        for kernel_name in all_kernels:
                            # Reset sparse indices each run
                            inputs["sparse_kv_indices"].zero_()

                            if kernel_name == "naive":
                                call_args = (
                                    inputs["x"],
                                    inputs["dense_kv_indptr"],
                                    inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indptr"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs,
                                    topk_val,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    pages_per_seg,
                                )
                                result = bench_kernel(topk_output, call_args, args.warmup, args.repeat)
                            elif kernel_name == "sglang_ori":
                                call_args = (
                                    inputs["x"],
                                    inputs["dense_kv_indptr"],
                                    inputs["sparse_kv_indptr"],
                                    inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs,
                                    topk_val,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    pages_per_seg,
                                )
                                result = bench_kernel(topk_output_sglang_ori, call_args, args.warmup, args.repeat)
                            elif kernel_name == "sglang_scale":
                                call_args = (
                                    inputs["x"],
                                    inputs["dense_kv_indptr"],
                                    inputs["sparse_kv_indptr"],
                                    inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs,
                                    topk_val,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    pages_per_seg,
                                    3,    # mode 3 (power)
                                    1.0,  # p=1.0 → identity
                                    None,
                                    None,
                                )
                                result = bench_kernel(topk_output_sglang, call_args, args.warmup, args.repeat)
                            else:
                                mode_str = kernel_name.split("_m")[1]
                                mode = int(mode_str.split("_")[0])
                                is_noscale = kernel_name.endswith("_noscale")
                                extra_kwargs = {}
                                if mode == 1:
                                    extra_kwargs["mapping_lut"] = mapping_lut
                                elif mode == 2:
                                    extra_kwargs["mapping_quantiles"] = mapping_quantiles

                                if mode in (3, 6, 7, 9, 10, 13, 14):
                                    power = _resolve_mode_power(args, mode)
                                else:
                                    power = 0.5

                                call_args = (
                                    inputs["x"],
                                    inputs["dense_kv_indptr"],
                                    inputs["sparse_kv_indptr"],
                                    inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs,
                                    topk_val,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    pages_per_seg,
                                    mode,
                                    power,
                                    extra_kwargs.get("mapping_lut", None),
                                    extra_kwargs.get("mapping_quantiles", None),
                                    is_noscale,
                                )
                                result = bench_kernel(topk_output_sglang, call_args, args.warmup, args.repeat)

                            # Build label
                            if kernel_name == "naive":
                                label = "naive"
                            elif kernel_name == "sglang_ori":
                                label = "sglang Ori (no remap)"
                            elif kernel_name == "sglang_scale":
                                label = "sglang Scale Only (p=1.0)"
                            else:
                                m_str = kernel_name.split("_m")[1]
                                m = int(m_str.split("_")[0])
                                noscale_suffix = " noscale" if kernel_name.endswith("_noscale") else ""
                                mname = MAPPING_MODE_NAMES.get(m, f'm{m}')
                                if m in (3, 6, 7, 9, 10, 13, 14):
                                    pname = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha", 13: "alpha", 14: "rho"}[m]
                                    label = f"sglang {mname} ({pname}={_resolve_mode_power(args, m)}){noscale_suffix}"
                                else:
                                    label = f"sglang {mname}{noscale_suffix}"

                            # Sub-phase profiling for sglang kernels (skip ori baseline)
                            if kernel_name not in ("naive", "sglang_ori"):
                                if kernel_name == "sglang_scale":
                                    s1_mode, s1_power = 3, 1.0
                                    s1_lut, s1_q = None, None
                                    s1_noscale = False
                                else:
                                    s1_mode_str = kernel_name.split("_m")[1]
                                    s1_mode = int(s1_mode_str.split("_")[0])
                                    s1_noscale = kernel_name.endswith("_noscale")
                                    if s1_mode in (3, 6, 7, 9, 10, 13, 14):
                                        s1_power = _resolve_mode_power(args, s1_mode)
                                    else:
                                        s1_power = 0.5
                                    s1_lut = mapping_lut if s1_mode == 1 else None
                                    s1_q = mapping_quantiles if s1_mode == 2 else None

                                # Histogram only: pre-pass + histogram build
                                hist_buf = torch.zeros(eff_bs, 256, dtype=torch.int32, device="cuda")
                                hist_args = (
                                    inputs["x"],
                                    inputs["dense_kv_indptr"],
                                    hist_buf,
                                    eff_bs,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    s1_mode,
                                    s1_power,
                                    s1_lut,
                                    s1_q,
                                    s1_noscale,
                                )
                                hist_result = bench_kernel(topk_profile_histogram, hist_args, args.warmup, args.repeat)

                                # Stage1 full: pre-pass + hist + cumsum + route/filter
                                inputs["sparse_kv_indices"].zero_()
                                stage1_args = (
                                    inputs["x"],
                                    inputs["dense_kv_indptr"],
                                    inputs["sparse_kv_indptr"],
                                    inputs["dense_kv_indices"],
                                    inputs["sparse_kv_indices"],
                                    eff_bs,
                                    topk_val,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    pages_per_seg,
                                    s1_mode,
                                    s1_power,
                                    s1_lut,
                                    s1_q,
                                    s1_noscale,
                                )
                                stage1_result = bench_kernel(topk_profile_stage1, stage1_args, args.warmup, args.repeat)

                                result['histogram_only_mean_ms'] = hist_result['mean_ms']
                                result['histogram_only_median_ms'] = hist_result['median_ms']
                                result['stage1_full_mean_ms'] = stage1_result['mean_ms']
                                result['stage1_full_median_ms'] = stage1_result['median_ms']
                                result['route_overhead_mean_ms'] = stage1_result['mean_ms'] - hist_result['mean_ms']
                                result['route_overhead_median_ms'] = stage1_result['median_ms'] - hist_result['median_ms']
                                result['stage2_refine_mean_ms'] = result['mean_ms'] - stage1_result['mean_ms']
                                result['stage2_refine_median_ms'] = result['median_ms'] - stage1_result['median_ms']

                                # Optional counter collection
                                if args.counters:
                                    inputs["sparse_kv_indices"].zero_()
                                    counter_buf = torch.zeros(eff_bs, 6, dtype=torch.int32, device="cuda")
                                    counter_args = (
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
                                        s1_mode,
                                        s1_power,
                                        s1_lut,
                                        s1_q,
                                        s1_noscale,
                                    )
                                    topk_profile_counters(*counter_args)
                                    torch.cuda.synchronize()
                                    c = counter_buf.float()
                                    result['counters'] = {
                                        'threshold_bin_mean': c[:, 0].mean().item(),
                                        'num_above_mean': c[:, 1].mean().item(),
                                        'num_equal_mean': c[:, 2].mean().item(),
                                        'remaining_k_mean': c[:, 3].mean().item(),
                                        'refine_rounds_mean': c[:, 4].mean().item(),
                                        'stage2_input_mean': c[:, 5].mean().item(),
                                        'threshold_bin_max': c[:, 0].max().item(),
                                        'num_above_max': c[:, 1].max().item(),
                                        'num_equal_max': c[:, 2].max().item(),
                                        'remaining_k_max': c[:, 3].max().item(),
                                        'refine_rounds_max': c[:, 4].max().item(),
                                        'stage2_input_max': c[:, 5].max().item(),
                                    }

                            # Counter collection for kernels skipped by sub-phase profiling
                            if kernel_name in ("sglang_ori",) and args.counters:
                                inputs["sparse_kv_indices"].zero_()
                                counter_buf = torch.zeros(eff_bs, 6, dtype=torch.int32, device="cuda")
                                counter_args = (
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
                                    0,     # mode 0 (no mapping) — matches ori behavior
                                    0.5,
                                    None,
                                    None,
                                    False,
                                )
                                topk_profile_counters(*counter_args)
                                torch.cuda.synchronize()
                                c = counter_buf.float()
                                result['counters'] = {
                                    'threshold_bin_mean': c[:, 0].mean().item(),
                                    'num_above_mean': c[:, 1].mean().item(),
                                    'num_equal_mean': c[:, 2].mean().item(),
                                    'remaining_k_mean': c[:, 3].mean().item(),
                                    'refine_rounds_mean': c[:, 4].mean().item(),
                                    'stage2_input_mean': c[:, 5].mean().item(),
                                    'threshold_bin_max': c[:, 0].max().item(),
                                    'num_above_max': c[:, 1].max().item(),
                                    'num_equal_max': c[:, 2].max().item(),
                                    'remaining_k_max': c[:, 3].max().item(),
                                    'refine_rounds_max': c[:, 4].max().item(),
                                    'stage2_input_max': c[:, 5].max().item(),
                                }

                            kernel_entries.append((label, kernel_name, result))
                            config_results["kernels"][kernel_name] = result

                        # Print kernel results sorted by mean latency (ascending)
                        kernel_entries.sort(key=lambda e: e[2]['mean_ms'])
                        print(f"  --- kernel latency (sorted by mean, ascending) ---")
                        for label, kernel_name, result in kernel_entries:
                            print(
                                f"  {label:<40s}: "
                                f"mean={result['mean_ms']:.4f}ms  "
                                f"median={result['median_ms']:.4f}ms  "
                                f"\u00b1 {result['std_ms']:.4f}ms  "
                                f"[min={result['min_ms']:.4f}, max={result['max_ms']:.4f}]"
                            )
                            if 'stage1_full_mean_ms' in result:
                                print(
                                    f"    {'Histogram only (map+hist)':<36s}: "
                                    f"mean={result['histogram_only_mean_ms']:.4f}ms  "
                                    f"median={result['histogram_only_median_ms']:.4f}ms"
                                )
                                print(
                                    f"    {'Stage1 full (hist+cumsum+route)':<36s}: "
                                    f"mean={result['stage1_full_mean_ms']:.4f}ms  "
                                    f"median={result['stage1_full_median_ms']:.4f}ms"
                                )
                                print(
                                    f"    {'Route overhead (cumsum+route)':<36s}: "
                                    f"mean={result['route_overhead_mean_ms']:.4f}ms  "
                                    f"median={result['route_overhead_median_ms']:.4f}ms"
                                )
                                print(
                                    f"    {'Stage2 (refine)':<36s}: "
                                    f"mean={result['stage2_refine_mean_ms']:.4f}ms  "
                                    f"median={result['stage2_refine_median_ms']:.4f}ms"
                                )
                            if 'counters' in result:
                                c = result['counters']
                                print(
                                    f"    Counters: threshold_bin={c['threshold_bin_mean']:.0f}  "
                                    f"above={c['num_above_mean']:.0f}  "
                                    f"equal={c['num_equal_mean']:.0f}  "
                                    f"remaining_k={c['remaining_k_mean']:.0f}  "
                                    f"refine_rounds={c['refine_rounds_mean']:.1f}  "
                                    f"stage2_input={c['stage2_input_mean']:.0f}"
                                )

                        # Histogram analysis
                        if args.histogram:
                            # Build a separate (potentially larger) dataset for histogram profiling
                            target_pages = (args.histogram_pages
                                            if args.histogram_pages is not None
                                            else _histogram_target_pages(pages_per_seg))
                            current_pages = eff_bs * pages_per_seg
                            if target_pages > current_pages:
                                hist_bs = math.ceil(target_pages / (num_kv_heads * pages_per_seg))
                                if dist == "real" and real_histogram is not None:
                                    hist_inputs = make_topk_inputs(
                                        batch_size=hist_bs, num_kv_heads=num_kv_heads,
                                        seq_len=seq_len, page_size=args.page_size,
                                        topk_val=topk_val, reserved_bos=args.reserved_bos,
                                        reserved_eos=args.reserved_eos, score_dtype=score_dtype,
                                        distribution="normal",
                                    )
                                    total_hist_dense = hist_inputs["eff_batch_size"] * hist_inputs["num_pages_per_seg"]
                                    hist_inputs["x"] = _scores_from_histogram(real_histogram, total_hist_dense, device="cuda")
                                else:
                                    hist_inputs = make_topk_inputs(
                                        batch_size=hist_bs, num_kv_heads=num_kv_heads,
                                        seq_len=seq_len, page_size=args.page_size,
                                        topk_val=topk_val, reserved_bos=args.reserved_bos,
                                        reserved_eos=args.reserved_eos, score_dtype=score_dtype,
                                        distribution=dist,
                                    )
                                hist_eff_bs = hist_inputs["eff_batch_size"]
                                actual_pages = hist_eff_bs * pages_per_seg
                                print(
                                    f"  histogram dataset  : {actual_pages} pages "
                                    f"(upscaled from {current_pages} for statistical reliability)"
                                )
                            else:
                                hist_inputs = inputs
                                hist_eff_bs = eff_bs
                                actual_pages = current_pages
                                print(f"  histogram dataset  : {actual_pages} pages")

                            # Raw unmapped histogram
                            histograms = torch.zeros(hist_eff_bs, 256, dtype=torch.int32, device="cuda")
                            topk_profile_histogram(
                                hist_inputs["x"],
                                hist_inputs["dense_kv_indptr"],
                                histograms,
                                hist_eff_bs,
                                args.reserved_bos,
                                args.reserved_eos,
                            )
                            hstats = compute_histogram_stats(histograms)
                            hstats["raw_counts"] = histograms.sum(dim=0).tolist()  # [256] ints
                            config_results["histogram"] = hstats
                            print(
                                f"  histogram stats    : max/mean={hstats['max_mean_ratio']:.2f}  "
                                f"gini={hstats['gini']:.3f}  "
                                f"nonzero_bins={hstats['num_nonzero_bins']}/256"
                            )

                            # Collect all histogram entries, then print sorted by gini
                            # Each entry: (display_name, key, mode_stats)
                            hist_entries = []
                            histograms_results = {}

                            # Per-mode histogram analysis (scaled)
                            modes_to_test = [0, 3, 4, 6, 7, 8, 9, 10, 11]
                            if mapping_lut is not None:
                                modes_to_test.append(1)
                            if mapping_quantiles is not None:
                                modes_to_test.append(2)
                            modes_to_test.sort()

                            for mode in modes_to_test:
                                mode_hists = torch.zeros(hist_eff_bs, 256, dtype=torch.int32, device="cuda")

                                extra_lut = mapping_lut if mode == 1 else None
                                extra_q = mapping_quantiles if mode == 2 else None
                                power = _resolve_mode_power(args, mode) if mode in (3, 6, 7, 9, 10, 13, 14) else 0.5

                                topk_profile_histogram(
                                    hist_inputs["x"],
                                    hist_inputs["dense_kv_indptr"],
                                    mode_hists,
                                    hist_eff_bs,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    mode,
                                    power,
                                    extra_lut,
                                    extra_q,
                                    False,      # mapping_noscale
                                    topk_val,   # needed for mode 12/14 (tail/topk window)
                                )
                                torch.cuda.synchronize()

                                mode_stats = compute_histogram_stats(mode_hists)
                                mode_stats["raw_counts"] = mode_hists.sum(dim=0).tolist()
                                mname = MAPPING_MODE_NAMES.get(mode, f"m{mode}")
                                mformula = MAPPING_MODE_FORMULAS.get(mode, mname)
                                mode_stats["name"] = mname
                                mode_stats["formula"] = mformula
                                if mode in (3, 6, 7, 9, 10, 13, 14):
                                    pname = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha", 13: "alpha", 14: "rho"}[mode]
                                    mode_stats["param"] = f"{pname}={power}"
                                    display_name = f"{mname} ({pname}={power})"
                                else:
                                    display_name = mname
                                key = f"mode_{mode}_{mname}"
                                histograms_results[key] = mode_stats
                                hist_entries.append((display_name, f"mode {mode:2d}", mode_stats))

                            # Noscale histogram analysis for parametric transform modes
                            noscale_modes = [m for m in (3, 6, 7, 9, 10, 13) if m in modes_to_test]
                            for mode in noscale_modes:
                                ns_hists = torch.zeros(hist_eff_bs, 256, dtype=torch.int32, device="cuda")
                                power = _resolve_mode_power(args, mode)
                                topk_profile_histogram(
                                    hist_inputs["x"],
                                    hist_inputs["dense_kv_indptr"],
                                    ns_hists,
                                    hist_eff_bs,
                                    args.reserved_bos,
                                    args.reserved_eos,
                                    mode,
                                    power,
                                    None,
                                    None,
                                    True,  # mapping_noscale=True
                                )
                                torch.cuda.synchronize()
                                ns_stats = compute_histogram_stats(ns_hists)
                                ns_stats["raw_counts"] = ns_hists.sum(dim=0).tolist()
                                mname = MAPPING_MODE_NAMES.get(mode, f"m{mode}")
                                mformula = MAPPING_MODE_FORMULAS.get(mode, mname)
                                pname = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha", 13: "alpha"}[mode]
                                ns_stats["name"] = f"{mname} noscale"
                                ns_stats["formula"] = mformula
                                ns_stats["param"] = f"{pname}={power}"
                                display_name = f"{mname} noscale ({pname}={power})"
                                key = f"mode_{mode}_{mname}_noscale"
                                histograms_results[key] = ns_stats
                                hist_entries.append((display_name, f"m{mode:2d} ns", ns_stats))

                            # Scale Only baseline: mode 3 with p=1.0 (identity + linear scaling)
                            scale_hists = torch.zeros(hist_eff_bs, 256, dtype=torch.int32, device="cuda")
                            topk_profile_histogram(
                                hist_inputs["x"],
                                hist_inputs["dense_kv_indptr"],
                                scale_hists,
                                hist_eff_bs,
                                args.reserved_bos,
                                args.reserved_eos,
                                3,    # mode 3 (power)
                                1.0,  # p=1.0 → identity transform
                                None,
                                None,
                            )
                            torch.cuda.synchronize()
                            scale_stats = compute_histogram_stats(scale_hists)
                            scale_stats["raw_counts"] = scale_hists.sum(dim=0).tolist()
                            scale_stats["name"] = "Scale Only"
                            scale_stats["formula"] = "Identity + linear scaling to [0,255]"
                            scale_stats["param"] = "p=1.0"
                            histograms_results["mode_scale_Scale Only"] = scale_stats
                            hist_entries.append(("Scale Only (p=1.0)", "scale  ", scale_stats))

                            # Print all histogram entries sorted by gini (ascending = more uniform = better)
                            hist_entries.sort(key=lambda e: e[2]['gini'])
                            print(f"  --- histogram by gini (sorted, lower=better) ---")
                            for rank, (display_name, mode_tag, stats) in enumerate(hist_entries, 1):
                                print(
                                    f"  {rank:2d}. {display_name:<32s} ({mode_tag}): "
                                    f"gini={stats['gini']:.3f}  "
                                    f"max/mean={stats['max_mean_ratio']:.2f}  "
                                    f"nonzero_bins={stats['num_nonzero_bins']}/256  "
                                    f"eff_bins={stats['effective_bins']:.1f}  "
                                    f"entropy={stats['entropy']:.2f}"
                                )

                            config_results["histograms"] = histograms_results

                        all_results.append(config_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="TopK kernel benchmark suite")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 2048, 4096, 8192])
    parser.add_argument("--topk-vals", nargs="+", type=int, default=[16, 30, 64])
    parser.add_argument("--num-kv-heads", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--reserved-bos", type=int, default=1)
    parser.add_argument("--reserved-eos", type=int, default=2)
    parser.add_argument("--score-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--distributions", nargs="+", default=["normal", "lognormal", "uniform"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--mapping-power", type=float, default=0.5,
                        help="Global fallback power parameter for parametric modes (default: 0.5)")
    parser.add_argument("--mapping-power-3", type=float, default=None,
                        help="Power exponent p for mode 3 (overrides --mapping-power)")
    parser.add_argument("--mapping-power-6", type=float, default=None,
                        help="Beta for mode 6 asinh (overrides --mapping-power)")
    parser.add_argument("--mapping-power-7", type=float, default=None,
                        help="Alpha for mode 7 log1p (overrides --mapping-power)")
    parser.add_argument("--mapping-power-13", type=float, default=None,
                        help="Alpha for mode 13 exp_stretch (overrides --mapping-power)")
    parser.add_argument("--mapping-power-14", type=float, default=None,
                        help="Rho for mode 14 topk_window (overrides --mapping-power)")
    parser.add_argument("--autotune-json", type=str, default=None,
                        help="Path to autotune_results.json — extracts best per-mode hyperparameters "
                             "(overrides --mapping-power for modes 3/6/7/13/14)")
    parser.add_argument("--lut-path", type=str, default=None, help="Path to .npy uint8[256] LUT for mode=1")
    parser.add_argument("--quantiles-path", type=str, default=None, help="Path to .npy float32[256] for mode=2")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--filter-kernels", nargs="+", default=None,
                        help="Only run specific kernels: naive, sglang_m0, sglang_m3, sglang_m4")
    parser.add_argument("--histogram", action="store_true", help="Collect and report bin distribution statistics")
    parser.add_argument("--histogram-pages", type=int, default=None,
                        help="Total pages for histogram profiling. Default: adaptive "
                             "(512 samples/bin × 256 bins, rounded to segment boundary). "
                             "Only used when --histogram is set.")
    parser.add_argument("--real-histograms", type=str, default=None,
                        help="Path to .npy raw_histograms from calibration (adds 'real' distribution)")
    parser.add_argument("--counters", action="store_true",
                        help="Collect diagnostic counters (threshold_bin, num_above, num_equal, "
                             "remaining_k, refine_rounds, stage2_input) for each sglang kernel")

    args = parser.parse_args()
    results = run_benchmark(args)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
