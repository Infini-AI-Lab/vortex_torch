#!/usr/bin/env python3
"""
Profile TopK bin distribution and generate mapping tables.

This script collects Stage 1 (8-bit coarse histogram) distributions from
the topk_sglang kernel and generates LUT/quantile mapping tables that
can be used to equalize the bin distribution for improved sorting efficiency.

Usage:
    python scripts/profile_topk_distribution.py \
        --model-name Qwen/Qwen3-1.7B \
        --output mapping_tables.npz \
        --num-prompts 32 \
        --mem 0.7

Output (.npz):
    lut_tables:      [num_collected, 256] uint8  - CDF-equalized LUT per sample
    quantile_tables: [num_collected, 256] float32 - quantile breakpoints per sample
    raw_histograms:  [num_collected, 256] int32   - raw bin histograms
"""

import argparse
import numpy as np
import torch


def compute_lut_from_histogram(histogram: np.ndarray) -> np.ndarray:
    """Compute CDF-equalized LUT from a 256-bin histogram.

    Args:
        histogram: [256] int array of bin counts

    Returns:
        lut: [256] uint8 array where lut[i] = floor(CDF(i) * 255)
    """
    cdf = np.cumsum(histogram).astype(np.float64)
    total = cdf[-1]
    if total == 0:
        return np.arange(256, dtype=np.uint8)
    cdf_normalized = cdf / total
    lut = np.floor(cdf_normalized * 255).astype(np.uint8)
    return lut


def compute_quantiles_from_scores(scores: np.ndarray, num_quantiles: int = 256) -> np.ndarray:
    """Compute quantile breakpoints from raw float scores.

    Args:
        scores: 1D array of float scores
        num_quantiles: number of quantile bins (default 256)

    Returns:
        quantiles: [num_quantiles] float32 array of sorted breakpoints
    """
    if len(scores) == 0:
        return np.zeros(num_quantiles, dtype=np.float32)
    percentiles = np.linspace(0, 100, num_quantiles)
    quantiles = np.percentile(scores, percentiles).astype(np.float32)
    return quantiles


def generate_tables_from_histograms(histograms: np.ndarray) -> dict:
    """Generate LUT and quantile tables from collected histograms.

    Args:
        histograms: [N, 256] int32 array of bin histograms

    Returns:
        dict with 'lut_tables' and 'aggregate_lut'
    """
    N = histograms.shape[0]
    lut_tables = np.zeros((N, 256), dtype=np.uint8)

    for i in range(N):
        lut_tables[i] = compute_lut_from_histogram(histograms[i])

    # Aggregate: average histogram across all samples
    avg_histogram = histograms.mean(axis=0)
    aggregate_lut = compute_lut_from_histogram(avg_histogram)

    return {
        'lut_tables': lut_tables,
        'aggregate_lut': aggregate_lut,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile TopK bin distribution and generate mapping tables")
    parser.add_argument("--output", type=str, default="mapping_tables.npz",
                        help="Output .npz file path")
    parser.add_argument("--histograms-input", type=str, default=None,
                        help="Load pre-collected histograms from .npy file instead of running inference")
    parser.add_argument("--scores-input", type=str, default=None,
                        help="Load pre-collected raw scores from .npy for quantile computation")
    args = parser.parse_args()

    results = {}

    if args.histograms_input:
        print(f"Loading histograms from {args.histograms_input}")
        histograms = np.load(args.histograms_input)
        if histograms.ndim == 1:
            histograms = histograms.reshape(1, -1)
        results['raw_histograms'] = histograms

        tables = generate_tables_from_histograms(histograms)
        results.update(tables)

    if args.scores_input:
        print(f"Loading scores from {args.scores_input}")
        scores = np.load(args.scores_input)
        quantiles = compute_quantiles_from_scores(scores.flatten())
        results['quantile_table'] = quantiles

    if not results:
        print("No input provided. Use --histograms-input or --scores-input.")
        print("\nTo collect histograms, use the topk_profile_histogram() function from vortex_torch_C:")
        print("  from vortex_torch_C import topk_profile_histogram")
        print("  histograms = torch.zeros(eff_batch_size, 256, dtype=torch.int32, device='cuda')")
        print("  topk_profile_histogram(scores, dense_kv_indptr, histograms, eff_batch_size, bos, eos)")
        print("  np.save('histograms.npy', histograms.cpu().numpy())")
        return

    np.savez(args.output, **results)
    print(f"Saved mapping tables to {args.output}")
    for key, val in results.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")


if __name__ == "__main__":
    main()
