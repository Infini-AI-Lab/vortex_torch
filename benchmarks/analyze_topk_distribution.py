"""
TopK distribution analysis and visualization.

Loads profiling data from:
  - profile_topk_distribution.py output (.npz): raw histograms, LUT tables
  - bench_topk.py output (.json): benchmark results + per-mode histogram data

Produces visualization plots for evaluating mapping mode effectiveness.

Usage:
    python scripts/analyze_topk_distribution.py \
        --bench-json bench_hitrate.json \
        --output-dir plots/

    python scripts/analyze_topk_distribution.py \
        --profile-npz profile_output.npz \
        --bench-json bench_hitrate.json \
        --output-dir plots/ --max-segments 8
"""

import argparse
import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Canonical mapping mode names — shared across all profiling/analysis tools
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


def _mode_key_to_display(mode_key: str) -> str:
    """Convert a mode key like 'mode_3' or 'mode_3_Power' to a display name."""
    # Handle new format: "mode_3_Power"
    parts = mode_key.split("_", 2)
    if len(parts) >= 3:
        return parts[2]  # e.g. "Power"
    # Handle old format: "mode_3"
    try:
        mode_num = int(parts[1])
        return MAPPING_MODE_NAMES.get(mode_num, mode_key)
    except (IndexError, ValueError):
        return mode_key


def _mode_key_to_number(mode_key: str) -> int:
    """Extract the mode number from a key like 'mode_3' or 'mode_3_Power'."""
    parts = mode_key.split("_")
    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return -1


def compute_per_segment_stats(histograms: np.ndarray) -> dict:
    """Compute per-row Gini coefficient and max/mean ratio.

    Args:
        histograms: [num_segments, 256] array of bin counts

    Returns:
        dict with 'gini' and 'max_mean' arrays of shape [num_segments]
    """
    num_seg = histograms.shape[0]
    ginis = np.zeros(num_seg)
    max_means = np.zeros(num_seg)

    for i in range(num_seg):
        row = histograms[i].astype(np.float64)
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            continue

        max_means[i] = nonzero.max() / nonzero.mean()

        # Gini coefficient
        sorted_vals = np.sort(nonzero)
        n = len(sorted_vals)
        index = np.arange(1, n + 1, dtype=np.float64)
        ginis[i] = (2.0 * (index * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n)
        ginis[i] = max(0.0, ginis[i])

    return {"gini": ginis, "max_mean": max_means}


def plot_bin_distribution(histograms: np.ndarray, output_dir: str, max_segments: int = 4):
    """Plot 256-bin bar chart per segment (first N segments)."""
    num_seg = min(histograms.shape[0], max_segments)
    for i in range(num_seg):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(256), histograms[i], width=1.0, color="steelblue", edgecolor="none")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Count")
        ax.set_title(f"Segment {i}: 256-bin histogram")
        ax.set_xlim(-1, 256)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"bin_dist_seg_{i}.png"), dpi=150)
        plt.close(fig)
    print(f"  Saved {num_seg} bin distribution plots")


def plot_bin_heatmap(histograms: np.ndarray, output_dir: str):
    """Heatmap: segments x bins, LogNorm colormap."""
    fig, ax = plt.subplots(figsize=(14, max(4, histograms.shape[0] * 0.15 + 1)))
    # Add 1 to avoid log(0)
    data = histograms.astype(np.float64) + 1
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=1, vmax=data.max()),
        interpolation="nearest",
    )
    ax.set_xlabel("Bin")
    ax.set_ylabel("Segment")
    ax.set_title("Bin distribution heatmap (log scale)")
    fig.colorbar(im, ax=ax, label="Count + 1")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bin_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved bin_heatmap.png")


def plot_before_after_mapping(
    raw_histograms: np.ndarray,
    lut_table: np.ndarray,
    output_dir: str,
    max_segments: int = 4,
):
    """Side-by-side: raw histogram vs. LUT-remapped histogram."""
    num_seg = min(raw_histograms.shape[0], max_segments)
    for i in range(num_seg):
        raw = raw_histograms[i]
        # Remap: redistribute counts through LUT
        remapped = np.zeros(256, dtype=np.float64)
        for bin_idx in range(256):
            new_bin = int(lut_table[bin_idx])
            remapped[new_bin] += raw[bin_idx]

        fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
        axes[0].bar(range(256), raw, width=1.0, color="steelblue", edgecolor="none")
        axes[0].set_title(f"Segment {i}: Raw (mode=0)")
        axes[0].set_xlabel("Bin")
        axes[0].set_ylabel("Count")

        axes[1].bar(range(256), remapped, width=1.0, color="darkorange", edgecolor="none")
        axes[1].set_title(f"Segment {i}: After LUT remap")
        axes[1].set_xlabel("Bin")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"mapping_comparison_{i}.png"), dpi=150)
        plt.close(fig)
    print(f"  Saved {num_seg} mapping comparison plots")


def plot_summary_table(
    histograms: np.ndarray,
    mode_stats_data: Optional[dict],
    output_dir: str,
):
    """Per-segment stats table: Gini, max/mean, resolution rate."""
    stats = compute_per_segment_stats(histograms)
    num_seg = histograms.shape[0]

    col_labels = ["Segment", "Gini", "Max/Mean"]
    cell_data = []
    for i in range(num_seg):
        cell_data.append([str(i), f"{stats['gini'][i]:.3f}", f"{stats['max_mean'][i]:.2f}"])

    fig, ax = plt.subplots(figsize=(6, max(2, num_seg * 0.4 + 1)))
    ax.axis("off")
    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax.set_title("Per-segment distribution stats", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved summary_table.png")


def plot_distribution_comparison(dist_histograms: dict, output_dir: str, suffix: str = "", title: str = ""):
    """Overlay 256-bin distributions for different data sources (uniform, normal, real).

    Args:
        dist_histograms: {"uniform": [256], "normal": [256], "real": [256], ...}
        output_dir: output directory for the plot
        suffix: optional suffix for output filename (e.g. "_m0")
        title: optional custom title for the plot
    """
    names = list(dist_histograms.keys())
    n = len(names)
    if n == 0:
        print("  No distribution histograms to compare")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    axes = axes[0]

    for idx, name in enumerate(names):
        counts = np.array(dist_histograms[name], dtype=np.float64)
        ax = axes[idx]
        ax.bar(range(256), counts, width=1.0, color="steelblue", edgecolor="none")
        ax.set_xlabel("Bucket")
        ax.set_ylabel("Count")
        ax.set_xlim(-1, 256)
        ax.set_title(name)

        # Annotate with stats
        nonzero = counts[counts > 0]
        if len(nonzero) > 0:
            mean_val = nonzero.mean()
            max_val = nonzero.max()
            max_mean = max_val / mean_val if mean_val > 0 else 0.0
            sorted_vals = np.sort(nonzero)
            nn = len(sorted_vals)
            index = np.arange(1, nn + 1, dtype=np.float64)
            gini = max(0.0, 2.0 * (index * sorted_vals).sum() / (nn * sorted_vals.sum()) - (nn + 1) / nn)
            nz_bins = int(len(nonzero))
        else:
            max_mean = gini = 0.0
            nz_bins = 0

        stats_text = f"gini={gini:.3f}\nmax/mean={max_mean:.2f}\nbins={nz_bins}/256"
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    fig.suptitle(title or "Bucket Distribution Comparison", fontsize=13)
    fig.tight_layout()
    fname = f"distribution_comparison{suffix}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def save_bucket_table(dist_histograms: dict, output_dir: str, filename: str = "bucket_counts.csv"):
    """Write a CSV table listing the count per bucket for each distribution.

    Columns: bucket, dist1, dist2, ...  (256 rows, one per bucket).
    """
    import csv

    names = list(dist_histograms.keys())
    if not names:
        return

    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket"] + names)
        for b in range(256):
            row = [b] + [int(dist_histograms[n][b]) for n in names]
            writer.writerow(row)

    # Also print a compact summary to stdout (top-20 hottest buckets per dist)
    print(f"  Saved {path}")
    for name in names:
        counts = np.array(dist_histograms[name], dtype=np.int64)
        total = counts.sum()
        top_idx = np.argsort(counts)[::-1][:20]
        print(f"  [{name}] total={total}  top-20 hottest buckets:")
        for rank, idx in enumerate(top_idx):
            if counts[idx] == 0:
                break
            pct = counts[idx] / total * 100 if total > 0 else 0
            print(f"    #{rank+1:2d}  bucket {idx:3d}: {counts[idx]:>10d}  ({pct:5.1f}%)")


def plot_mapping_mode_comparison(mode_stats_data: dict, output_dir: str):
    """Grouped bar chart comparing modes on gini and max/mean."""
    modes = sorted(mode_stats_data.keys())
    if not modes:
        print("  No histogram data to plot mode comparison")
        return

    mode_labels = []
    for m in modes:
        label = _mode_key_to_display(m)
        param = mode_stats_data[m].get("param")
        if param:
            label = f"{label} ({param})"
        mode_labels.append(label)
    ginis = [mode_stats_data[m]["gini"] for m in modes]
    max_means = [mode_stats_data[m]["max_mean_ratio"] for m in modes]

    x = np.arange(len(modes))
    width = 0.3

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, ginis, width, label="Gini", color="darkorange")
    bars2 = ax2.bar(x + width / 2, max_means, width, label="Max/Mean", color="seagreen", alpha=0.7)

    ax1.set_xlabel("Mapping Mode")
    ax1.set_ylabel("Gini")
    ax2.set_ylabel("Max/Mean Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mode_labels, rotation=15, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Mapping Mode Comparison")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mode_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved mode_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze TopK bucket sort distribution")
    parser.add_argument("--profile-npz", type=str, default=None,
                        help="Path to .npz from profile_topk_distribution.py")
    parser.add_argument("--bench-json", type=str, default=None,
                        help="Path to JSON from bench_topk.py")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory for output plots")
    parser.add_argument("--max-segments", type=int, default=4,
                        help="Max segments for per-segment plots")
    parser.add_argument("--real-histograms", type=str, default=None,
                        help="Path to .npy raw_histograms from calibrate_topk.py (real-data bucket counts)")
    args = parser.parse_args()

    if args.profile_npz is None and args.bench_json is None and args.real_histograms is None:
        parser.error("At least one of --profile-npz, --bench-json, or --real-histograms is required")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    raw_histograms = None
    lut_table = None
    mode_stats_data = None

    # Load profile data
    if args.profile_npz:
        print(f"\nLoading profile data from {args.profile_npz}")
        data = np.load(args.profile_npz, allow_pickle=True)
        if "raw_histograms" in data:
            raw_histograms = data["raw_histograms"]
            print(f"  raw_histograms: {raw_histograms.shape}")
        if "aggregate_lut" in data:
            lut_table = data["aggregate_lut"]
            print(f"  aggregate_lut: {lut_table.shape}")
        elif "lut_tables" in data:
            # Use first LUT if aggregate not available
            lut_table = data["lut_tables"]
            if lut_table.ndim > 1:
                lut_table = lut_table[0]
            print(f"  lut_table: {lut_table.shape}")

    # Load bench data
    dist_histograms = {}  # {distribution_name: [256] counts} for comparison plot
    mode_histograms = {}  # {mode_key: {dist_name: [256]}} for per-mode plots

    if args.bench_json:
        print(f"\nLoading benchmark data from {args.bench_json}")
        with open(args.bench_json) as f:
            bench_data = json.load(f)

        if bench_data and isinstance(bench_data, list):
            # Use first config entry for histogram mode visualization
            entry = bench_data[0]
            if "histograms" in entry:
                mode_stats_data = entry["histograms"]
                print(f"  Histogram modes: {list(mode_stats_data.keys())}")

            # Extract raw_counts per distribution from bench entries
            for entry in bench_data:
                dist_name = entry.get("distribution", "unknown")
                hist_data = entry.get("histogram", {})
                if "raw_counts" in hist_data and dist_name not in dist_histograms:
                    dist_histograms[dist_name] = hist_data["raw_counts"]
                    print(f"  Loaded histogram for distribution: {dist_name}")

            # Extract per-mode histograms from histograms data
            mode_histograms = {}  # {mode_key: {dist_name: [256]}}
            for entry in bench_data:
                dist_name = entry.get("distribution", "unknown")
                histograms_data = entry.get("histograms", {})
                for mode_key, mode_data in histograms_data.items():
                    if isinstance(mode_data, dict) and "raw_counts" in mode_data:
                        if mode_key not in mode_histograms:
                            mode_histograms[mode_key] = {}
                        if dist_name not in mode_histograms[mode_key]:
                            mode_histograms[mode_key][dist_name] = mode_data["raw_counts"]
            if mode_histograms:
                print(f"  Loaded per-mode histograms for: {sorted(mode_histograms.keys())}")

    # Load real-data histograms from .npy (calibrate_topk.py output)
    real_counts = None
    if args.real_histograms:
        print(f"\nLoading real-data histograms from {args.real_histograms}")
        real_hists = np.load(args.real_histograms)  # [num_samples, 256]
        real_counts = real_hists.sum(axis=0).tolist()  # aggregate across samples
        dist_histograms["real"] = real_counts
        print(f"  real_histograms shape: {real_hists.shape}, aggregated to [256]")

    # Generate plots
    if raw_histograms is not None:
        print("\nGenerating histogram plots...")
        plot_bin_distribution(raw_histograms, args.output_dir, args.max_segments)
        plot_bin_heatmap(raw_histograms, args.output_dir)
        plot_summary_table(raw_histograms, mode_stats_data, args.output_dir)

        if lut_table is not None:
            print("\nGenerating before/after mapping comparison...")
            plot_before_after_mapping(raw_histograms, lut_table, args.output_dir, args.max_segments)

    if mode_stats_data is not None:
        print("\nGenerating mode comparison plot...")
        plot_mapping_mode_comparison(mode_stats_data, args.output_dir)

    if dist_histograms:
        print("\nGenerating distribution comparison plot (raw/unmapped)...")
        plot_distribution_comparison(dist_histograms, args.output_dir)
        print("\nSaving bucket count table (raw/unmapped)...")
        save_bucket_table(dist_histograms, args.output_dir)

    # Per-mode distribution plots and tables
    if mode_histograms:
        print("\nGenerating per-mode distribution plots and tables...")
        for mode_key in sorted(mode_histograms):
            mname = _mode_key_to_display(mode_key)
            mode_num = _mode_key_to_number(mode_key)
            mformula = MAPPING_MODE_FORMULAS.get(mode_num, mname)
            # Include hyperparameter value in title if available
            param_str = ""
            if mode_stats_data and mode_key in mode_stats_data:
                param = mode_stats_data[mode_key].get("param")
                if param:
                    param_str = f" [{param}]"
            mode_suffix = mname.lower().replace(" ", "_")
            plot_distribution_comparison(
                mode_histograms[mode_key], args.output_dir,
                suffix=f"_{mode_suffix}",
                title=f"Bucket Distribution — {mname}{param_str} ({mformula})",
            )
            save_bucket_table(
                mode_histograms[mode_key], args.output_dir,
                filename=f"bucket_counts_{mode_suffix}.csv",
            )

    print(f"\nDone. All outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
