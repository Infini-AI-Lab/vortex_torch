#!/usr/bin/env python3
"""
Offline calibration for TopK mapping modes 1 (LUT CDF) and 2 (quantile).

Runs the model on real data with hit-rate profiling enabled, collects score
histograms from the topk_sglang kernel, and generates:
  - lut.npy      : uint8[256]   CDF-equalized LUT for mapping mode 1
  - quantiles.npy: float32[256] quantile breakpoints for mapping mode 2

Usage:
    python benchmarks/calibrate_topk.py \
        --model-name Qwen/Qwen3-1.7B \
        --topk-val 30 --mem 0.7 \
        --output-dir calibration_output/
"""

import argparse
import json
import os
import sys

import numpy as np

# Add project root to path so we can import from benchmarks/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from benchmarks.profile_topk_distribution import (
    compute_lut_from_histogram,
    generate_tables_from_histograms,
)


def main():
    parser = argparse.ArgumentParser(
        description="Offline calibration for TopK mapping modes 1 & 2"
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--topk-val", type=int, default=30)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--mem", type=float, default=0.7)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=1048576,
        help="Hard cap on KV pool token slots (ServerArgs.max_total_tokens). "
        "Block-sparse profiling uses a small bytes/token estimate, so the auto "
        "budget can be huge on large GPUs; VTXGraphAttnBackend then allocates "
        "dense bf16 sparse_prefill K/V buffers proportional to this cap (~4 KiB per "
        "token per buffer). For offline calibration, a few hundred K1M tokens "
        "is usually enough.",
    )
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--topk-type", type=str, default="sglang")
    parser.add_argument("--num-prompts", type=int, default=16,
                        help="Number of calibration prompts to use (default: 16)")
    parser.add_argument("--output-dir", type=str, default="calibration_output/")
    parser.add_argument("--vortex-module-name", type=str, default="block_sparse_attention")
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help="SGLang scheduler watchdog (seconds). Forward batches must complete within this time. "
        "Default: engine default (300). Use 0 to disable when using this repo's SGLang fork.",
    )
    args = parser.parse_args()

    # Lazy imports to avoid slow startup when just checking --help
    import sglang as sgl
    import torch
    import vortex_torch

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[calibrate] Launching engine with hit-rate profiling enabled...")
    engine_kwargs = dict(
        model_path=args.model_name,
        disable_cuda_graph=True,
        page_size=args.page_size,
        vortex_topk_val=args.topk_val,
        disable_overlap_schedule=True,
        attention_backend="flashinfer",
        enable_vortex_sparsity=True,
        vortex_page_reserved_bos=1,
        vortex_page_reserved_eos=2,
        vortex_layers_skip=list(range(1)),
        vortex_module_name=args.vortex_module_name,
        vortex_max_seq_lens=12288,
        mem_fraction_static=args.mem,
        max_total_tokens=args.max_total_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
        vortex_topk_type=args.topk_type,
        vortex_topk_mapping_mode=0,  # Use mode 0 during calibration
        vortex_topk_histogram=True,  # Enable histogram collection
    )
    if args.watchdog_timeout is not None:
        engine_kwargs["watchdog_timeout"] = args.watchdog_timeout
    llm = sgl.Engine(**engine_kwargs)

    # Clear any residual histograms in the worker process
    llm.clear_topk_histograms()

    # Load calibration prompts
    prompts_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "examples", "amc23.jsonl"
    )
    with open(prompts_path, "r", encoding="utf-8") as f:
        all_requests = [json.loads(line) for line in f]

    # Use up to num_prompts
    requests = all_requests[:args.num_prompts]
    prompts = [req["prompt"] for req in requests]

    print(f"[calibrate] Running {len(prompts)} calibration prompts...")
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 8192,
    }
    llm.generate(prompts, sampling_params)

    # Collect histograms via RPC from worker process
    histograms = llm.get_topk_histograms()
    print(f"[calibrate] Collected {len(histograms)} histogram batches")

    if len(histograms) == 0:
        print("[calibrate] ERROR: No histograms collected. "
              "Ensure topk_type='sglang' and vortex_topk_histogram=True.",
              file=sys.stderr)
        llm.shutdown()
        sys.exit(1)

    # Stack all histograms: each is [eff_bs, 256], concatenate along batch dim
    all_hists = torch.cat(histograms, dim=0).numpy()  # [total_samples, 256]
    print(f"[calibrate] Total histogram samples: {all_hists.shape[0]}")

    # --- Generate LUT (mode 1) ---
    # Aggregate histogram across all samples
    avg_histogram = all_hists.mean(axis=0)
    lut = compute_lut_from_histogram(avg_histogram)
    lut_path = os.path.join(args.output_dir, "lut.npy")
    np.save(lut_path, lut)
    print(f"[calibrate] Saved LUT to {lut_path} (shape={lut.shape}, dtype={lut.dtype})")

    # --- Generate quantiles (mode 2) ---
    # Use bin centers as proxy scores weighted by histogram counts
    bin_centers = np.arange(256, dtype=np.float32)
    # Expand histogram counts into a weighted score distribution
    total_counts = avg_histogram.astype(np.float64)
    total = total_counts.sum()
    if total > 0:
        cdf = np.cumsum(total_counts) / total
        # Invert CDF to get quantile breakpoints in [0, 255] space
        percentiles = np.linspace(0, 1, 256)
        quantiles = np.interp(percentiles, cdf, bin_centers).astype(np.float32)
    else:
        quantiles = bin_centers.copy()

    quantiles_path = os.path.join(args.output_dir, "quantiles.npy")
    np.save(quantiles_path, quantiles)
    print(f"[calibrate] Saved quantiles to {quantiles_path} (shape={quantiles.shape}, dtype={quantiles.dtype})")

    # Save raw histograms for debugging
    raw_path = os.path.join(args.output_dir, "raw_histograms.npy")
    np.save(raw_path, all_hists)
    print(f"[calibrate] Saved raw histograms to {raw_path} (shape={all_hists.shape})")

    # Cleanup
    llm.clear_topk_histograms()
    llm.shutdown()
    print(f"[calibrate] Done. Output files in {args.output_dir}/")


if __name__ == "__main__":
    main()
