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

from vortex_torch_C import topk_output, topk_output_sglang, topk_profile_histogram

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
        if m not in (3, 6, 7):
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
    per_mode_flag = {3: args.mapping_power_3, 6: args.mapping_power_6, 7: args.mapping_power_7}
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
        "sglang_m0": "sglang_m0",
        "sglang_m3": "sglang_m3",
        "sglang_m4": "sglang_m4",
        "sglang_m6": "sglang_m6",
        "sglang_m7": "sglang_m7",
        "sglang_m8": "sglang_m8",
    }
    if mapping_lut is not None:
        all_kernels["sglang_m1"] = "sglang_m1"
    if mapping_quantiles is not None:
        all_kernels["sglang_m2"] = "sglang_m2"

    if args.filter_kernels:
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

                        for kernel_name in all_kernels:
                            # Reset sparse indices each run
                            inputs["sparse_kv_indices"].zero_()

                            if kernel_name == "naive":
                                # topk_output: (x, dense_indptr, dense_indices, sparse_indptr, sparse_indices, ...)
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
                            else:
                                # Parse mapping mode from kernel name
                                mode = int(kernel_name.split("_m")[1])
                                extra_kwargs = {}
                                if mode == 1:
                                    extra_kwargs["mapping_lut"] = mapping_lut
                                elif mode == 2:
                                    extra_kwargs["mapping_quantiles"] = mapping_quantiles

                                power = _resolve_mode_power(args, mode) if mode in (3, 6, 7) else 0.5

                                # topk_output_sglang: (x, dense_indptr, sparse_indptr, dense_indices, sparse_indices, ...)
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
                                )
                                result = bench_kernel(topk_output_sglang, call_args, args.warmup, args.repeat)

                            if kernel_name == "naive":
                                label = "naive"
                            else:
                                m = int(kernel_name.split("_m")[1])
                                mname = MAPPING_MODE_NAMES.get(m, f'm{m}')
                                if m in (3, 6, 7):
                                    pname = {3: "p", 6: "beta", 7: "alpha"}[m]
                                    label = f"sglang {mname} ({pname}={_resolve_mode_power(args, m)})"
                                else:
                                    label = f"sglang {mname}"
                            print(
                                f"  {label:<30s}: {result['median_ms']:.4f}ms (median) "
                                f"\u00b1 {result['std_ms']:.4f}ms  "
                                f"[min={result['min_ms']:.4f}, max={result['max_ms']:.4f}]"
                            )
                            config_results["kernels"][kernel_name] = result

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

                            # Per-mode histogram analysis
                            modes_to_test = [0, 3, 4, 6, 7, 8]
                            if mapping_lut is not None:
                                modes_to_test.append(1)
                            if mapping_quantiles is not None:
                                modes_to_test.append(2)
                            modes_to_test.sort()

                            histograms_results = {}
                            print(f"  --- histogram by mapping mode ---")
                            for mode in modes_to_test:
                                mode_hists = torch.zeros(hist_eff_bs, 256, dtype=torch.int32, device="cuda")

                                extra_lut = mapping_lut if mode == 1 else None
                                extra_q = mapping_quantiles if mode == 2 else None
                                power = _resolve_mode_power(args, mode) if mode in (3, 6, 7) else 0.5

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
                                )
                                torch.cuda.synchronize()

                                mode_stats = compute_histogram_stats(mode_hists)
                                mode_stats["raw_counts"] = mode_hists.sum(dim=0).tolist()
                                mname = MAPPING_MODE_NAMES.get(mode, f"m{mode}")
                                mformula = MAPPING_MODE_FORMULAS.get(mode, mname)
                                mode_stats["name"] = mname
                                mode_stats["formula"] = mformula
                                if mode in (3, 6, 7):
                                    pname = {3: "p", 6: "beta", 7: "alpha"}[mode]
                                    mode_stats["param"] = f"{pname}={power}"
                                histograms_results[f"mode_{mode}_{mname}"] = mode_stats
                                if mode in (3, 6, 7):
                                    pname = {3: "p", 6: "beta", 7: "alpha"}[mode]
                                    display_name = f"{mname} ({pname}={power})"
                                else:
                                    display_name = mname
                                print(
                                    f"  {display_name:<22s} (mode {mode}): "
                                    f"gini={mode_stats['gini']:.3f}  "
                                    f"max/mean={mode_stats['max_mean_ratio']:.2f}  "
                                    f"nonzero_bins={mode_stats['num_nonzero_bins']}/256  "
                                    f"eff_bins={mode_stats['effective_bins']:.1f}  "
                                    f"entropy={mode_stats['entropy']:.2f}"
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
    parser.add_argument("--autotune-json", type=str, default=None,
                        help="Path to autotune_results.json — extracts best per-mode hyperparameters "
                             "(overrides --mapping-power for modes 3/6/7)")
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

    args = parser.parse_args()
    results = run_benchmark(args)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
