#!/usr/bin/env python3
"""
Analyze remap-function ablation sweeps.

Reads one or more sweep directories produced by
  ablation_remap_function_block_size.sh
  ablation_remap_function_topk_val.sh
  ablation_remap_function_model.sh
  ablation_remap_function_topk_benchmark.sh

and emits, for each sweep:
  - tidy CSV of every (axis_value, mapping_mode, distribution, head) row
  - wide CSV tables: latency, speedup vs baseline, chosen hparam
  - LaTeX version of the chosen-hparam table
  - markdown summary including the screenshot-style "Selected mapping
    functions" line per axis value
  - matplotlib PDF plots: latency vs axis, speedup vs axis, threshold
    bin size vs axis (one curve per mapping mode)

Usage:
  python examples/analyze_ablation_remap.py \
      --sweep-dir results/ablation_remap_block_size_<ts> \
      [--sweep-dir results/ablation_remap_model_<ts> ...] \
      --output-dir results/ablation_remap_analysis_<ts>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull mode metadata from the autotune script so we don't duplicate it.
SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))
try:
    from autotune_topk_mapping import MODE_NAMES, PARAM_NAME  # type: ignore
except Exception:
    MODE_NAMES = {0: "none", 3: "power", 4: "log", 6: "asinh", 7: "log1p",
                  8: "trunc8", 9: "erf", 10: "tanh", 11: "subtract",
                  13: "exp_stretch", 15: "shift_pow2", 16: "shift_pow3",
                  17: "linear_steep"}
    PARAM_NAME = {3: "p", 6: "beta", 7: "alpha", 9: "alpha", 10: "alpha",
                  11: "pivot", 13: "alpha", 15: "pivot", 16: "pivot", 17: "k"}

DISPLAY_NAME = {
    0: "None", 3: "Power", 6: "Asinh", 7: "Log1p", 9: "Erf",
    10: "Tanh", 11: "Subtract", 13: "ExpStretch",
    15: "ShiftPow2", 16: "ShiftPow3", 17: "LinearSteep",
}


# ---------- Loading ----------

def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _best_per_mode_from_autotune(autotune_results: List[dict]) -> Dict[int, dict]:
    best: Dict[int, dict] = {}
    for r in autotune_results:
        m = int(r["mode"])
        if m not in best or r["latency_ms"] < best[m]["latency_ms"]:
            best[m] = r
    return best


def _flatten_remap_bench(remap_results: List[dict]) -> pd.DataFrame:
    """Flatten bench_topk.py --remap-bench output into one row per
    (cfg, mode_row). Drops per-head sub-rows; keeps head='all' so each
    cell contributes a single point per (mapping_mode, distribution)."""
    rows = []
    for cfg in remap_results:
        if cfg.get("head", "all") != "all":
            continue
        baseline = cfg.get("baseline_ms")
        for mr in cfg.get("modes", []):
            mode = int(mr["mode"])
            rows.append({
                "distribution": cfg.get("distribution"),
                "batch_size": cfg.get("batch_size"),
                "num_kv_heads": cfg.get("num_kv_heads"),
                "seq_len": cfg.get("seq_len"),
                "topk_val": cfg.get("topk_val"),
                "pages_per_seg": cfg.get("pages_per_seg"),
                "mode": mode,
                "mode_name": mr.get("mode_name", MODE_NAMES.get(mode, str(mode))),
                "param_value": mr.get("power"),
                "fused_ms": mr.get("fused_ms"),
                "remap_ms": mr.get("remap_ms"),
                "topk_after_remap_ms": mr.get("topk_after_remap_ms"),
                "split_total_ms": mr.get("split_total_ms"),
                "baseline_ms": baseline,
                "threshold_bin_size_mean": mr.get("threshold_bin_size_mean"),
                "threshold_bin_size_max": mr.get("threshold_bin_size_max"),
                "refine_rounds_mean": mr.get("refine_rounds_mean"),
            })
    return pd.DataFrame(rows)


def load_sweep(sweep_dir: Path) -> Dict[str, Any]:
    idx_path = sweep_dir / "sweep_index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"missing sweep_index.json in {sweep_dir}")
    idx = _load_json(idx_path)
    axis_name = idx["axis_name"]

    rows: List[pd.DataFrame] = []
    chosen_hparams: List[dict] = []
    for cell in idx["cells"]:
        axis_value = cell["axis_value"]
        autotune_results = _load_json(cell["autotune_json"])
        best = _best_per_mode_from_autotune(autotune_results)
        for mode, r in best.items():
            chosen_hparams.append({
                "axis_value": axis_value,
                "mode": int(mode),
                "mode_name": r.get("mode_name", MODE_NAMES.get(int(mode), str(mode))),
                "param_name": r.get("param_name") or PARAM_NAME.get(int(mode), "p"),
                "param_value": r.get("param"),
                "autotune_latency_ms": r.get("latency_ms"),
            })

        remap_results = _load_json(cell["remap_bench_json"])
        df = _flatten_remap_bench(remap_results)
        df.insert(0, "axis_value", axis_value)
        df.insert(0, "axis_name", axis_name)
        rows.append(df)

    tidy = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    chosen = pd.DataFrame(chosen_hparams)
    return {
        "axis_name": axis_name,
        "axis_type": idx.get("axis_type", "kernel"),
        "index": idx,
        "tidy": tidy,
        "chosen": chosen,
    }


# ---------- Tables ----------

def _wide_latency(tidy: pd.DataFrame, axis_name: str, distribution: Optional[str] = None) -> pd.DataFrame:
    df = tidy.copy()
    if distribution is not None and "distribution" in df.columns:
        df = df[df["distribution"] == distribution]
    # Best fused latency per (axis_value, mode) — collapse over distribution
    # if no filter was applied.
    g = df.groupby(["axis_value", "mode", "mode_name"], dropna=False)["fused_ms"].min().reset_index()
    wide = g.pivot(index="axis_value", columns="mode", values="fused_ms")
    # Also pivot mode_name → label for column header.
    return wide.rename(columns=lambda m: f"{m}:{MODE_NAMES.get(int(m), '?')}")


def _wide_baseline(tidy: pd.DataFrame, distribution: Optional[str] = None) -> pd.Series:
    df = tidy.copy()
    if distribution is not None and "distribution" in df.columns:
        df = df[df["distribution"] == distribution]
    return df.groupby("axis_value")["baseline_ms"].min()


def _wide_speedup(tidy: pd.DataFrame, axis_name: str, distribution: Optional[str] = None) -> pd.DataFrame:
    lat = _wide_latency(tidy, axis_name, distribution=distribution)
    base = _wide_baseline(tidy, distribution=distribution)
    return lat.rdiv(base, axis=0)  # baseline / fused


def _wide_chosen_hparam(chosen: pd.DataFrame) -> pd.DataFrame:
    if chosen.empty:
        return pd.DataFrame()
    chosen = chosen.copy()
    chosen["label"] = chosen.apply(
        lambda r: f"{DISPLAY_NAME.get(int(r['mode']), r['mode_name'])}({r['param_name']}={r['param_value']})",
        axis=1,
    )
    wide = chosen.pivot(index="axis_value", columns="mode", values="label")
    return wide.rename(columns=lambda m: f"{m}:{MODE_NAMES.get(int(m), '?')}")


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        return f"% empty table for {label}\n"
    try:
        return df.to_latex(
            float_format=lambda v: "" if pd.isna(v) else f"{v:.4f}",
            na_rep="",
            caption=caption,
            label=label,
        )
    except Exception:
        return df.to_string()


# ---------- Plots ----------

def _axis_x(values: List[Any]) -> List[float]:
    """Convert axis values (which may be strings or ints) to numeric x
    coordinates. Strings are mapped to 0..N-1; numerics keep their value."""
    out = []
    for i, v in enumerate(values):
        if isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(float(i))
    return out


def _plot_metric_vs_axis(tidy: pd.DataFrame, axis_name: str, metric: str,
                         out_path: Path, ylabel: str, title: str,
                         baseline_series: Optional[pd.Series] = None,
                         logy: bool = False) -> None:
    if tidy.empty:
        return
    g = tidy.groupby(["axis_value", "mode", "mode_name"], dropna=False)[metric].min().reset_index()
    axis_values = sorted(g["axis_value"].unique(),
                         key=lambda v: (not isinstance(v, (int, float)), v))
    x = _axis_x(axis_values)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.get_cmap("tab10")
    for i, mode in enumerate(sorted(g["mode"].unique())):
        sub = g[g["mode"] == mode].set_index("axis_value").reindex(axis_values)
        ax.plot(x, sub[metric].values,
                marker="o", color=cmap(i % 10),
                label=f"{mode}:{MODE_NAMES.get(int(mode), '?')}")

    if baseline_series is not None and not baseline_series.empty:
        bx = baseline_series.reindex(axis_values).values
        ax.plot(x, bx, "k--", linewidth=2, label="baseline (unmapped)")

    ax.set_xlabel(axis_name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if all(isinstance(v, (int, float)) for v in axis_values):
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in axis_values])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in axis_values], rotation=20, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------- Per-sweep emitters ----------

def emit_sweep(sweep: Dict[str, Any], out_root: Path) -> None:
    axis_name = sweep["axis_name"]
    out_dir = out_root / axis_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy: pd.DataFrame = sweep["tidy"]
    chosen: pd.DataFrame = sweep["chosen"]

    if tidy.empty:
        print(f"[{axis_name}] no data, skipping")
        return

    tidy.to_csv(out_dir / "tidy.csv", index=False)
    chosen.to_csv(out_dir / "chosen_hparams_long.csv", index=False)

    distributions = sorted([d for d in tidy["distribution"].dropna().unique()])

    # Per-distribution wide tables + plots.
    for dist in distributions + [None]:
        suffix = f"_{dist}" if dist else "_all"
        lat_wide = _wide_latency(tidy, axis_name, distribution=dist)
        spd_wide = _wide_speedup(tidy, axis_name, distribution=dist)
        base = _wide_baseline(tidy, distribution=dist)

        lat_wide.to_csv(out_dir / f"table_latency_ms{suffix}.csv")
        spd_wide.to_csv(out_dir / f"table_speedup_vs_baseline{suffix}.csv")
        base.to_frame("baseline_ms").to_csv(out_dir / f"table_baseline_ms{suffix}.csv")

        with open(out_dir / f"table_latency_ms{suffix}.tex", "w") as f:
            f.write(_df_to_latex(lat_wide,
                                 caption=f"Best fused-kernel latency (ms) on {axis_name} sweep ({dist or 'all dists'})",
                                 label=f"tab:lat-{axis_name}{suffix}"))
        with open(out_dir / f"table_speedup_vs_baseline{suffix}.tex", "w") as f:
            f.write(_df_to_latex(spd_wide,
                                 caption=f"Speedup over unmapped baseline on {axis_name} sweep ({dist or 'all dists'})",
                                 label=f"tab:spd-{axis_name}{suffix}"))

        _plot_metric_vs_axis(
            tidy if dist is None else tidy[tidy["distribution"] == dist],
            axis_name, "fused_ms",
            out_dir / f"plot_latency_vs_{axis_name}{suffix}.pdf",
            ylabel="fused TopK kernel latency (ms)",
            title=f"TopK kernel latency vs {axis_name} ({dist or 'all dists'})",
            baseline_series=base,
        )
        # Speedup plot.
        spd_long = tidy.copy()
        if dist:
            spd_long = spd_long[spd_long["distribution"] == dist]
        spd_long = spd_long.assign(
            speedup=spd_long["baseline_ms"] / spd_long["fused_ms"]
        )
        _plot_metric_vs_axis(
            spd_long, axis_name, "speedup",
            out_dir / f"plot_speedup_vs_{axis_name}{suffix}.pdf",
            ylabel="speedup over unmapped baseline",
            title=f"Speedup vs {axis_name} ({dist or 'all dists'})",
        )
        # Threshold bin size diagnostic.
        _plot_metric_vs_axis(
            tidy if dist is None else tidy[tidy["distribution"] == dist],
            axis_name, "threshold_bin_size_mean",
            out_dir / f"plot_threshold_bin_size_vs_{axis_name}{suffix}.pdf",
            ylabel="mean threshold-bin size (entries)",
            title=f"Stage-1 threshold bin size vs {axis_name} ({dist or 'all dists'})",
        )

    # Chosen-hparam wide table (axis-independent of distribution: autotune
    # picks one hparam per mode per axis cell).
    chosen_wide = _wide_chosen_hparam(chosen)
    chosen_wide.to_csv(out_dir / "table_chosen_hparams.csv")
    with open(out_dir / "table_chosen_hparams.tex", "w") as f:
        f.write(_df_to_latex(chosen_wide,
                             caption=f"Autotuned remap-function hyperparameters per {axis_name} cell",
                             label=f"tab:hparam-{axis_name}"))

    # Markdown summary.
    md_lines: List[str] = []
    md_lines.append(f"# Ablation: remap function vs `{axis_name}`\n")
    md_lines.append(f"Source: `{sweep['index'].get('cells', [{}])[0].get('cell_dir', '')}/...`\n")

    md_lines.append("\n## Selected mapping functions (autotuned)\n")
    md_lines.append("```")
    for v in chosen_wide.index.tolist():
        parts = []
        for col in chosen_wide.columns:
            label = chosen_wide.loc[v, col]
            if isinstance(label, str) and label:
                parts.append(label)
        md_lines.append(f"[{axis_name}={v}] " + "  ".join(parts))
    md_lines.append("```\n")

    md_lines.append("\n## Latency (ms) — best fused, all distributions\n")
    md_lines.append(_wide_latency(tidy, axis_name).to_markdown())
    md_lines.append("\n\n## Speedup over unmapped baseline\n")
    md_lines.append(_wide_speedup(tidy, axis_name).to_markdown())
    md_lines.append("\n\n## Chosen hyperparameters\n")
    md_lines.append(chosen_wide.to_markdown())
    md_lines.append("\n\n## Plots\n")
    for p in sorted(out_dir.glob("plot_*.pdf")):
        md_lines.append(f"- `{p.name}`")

    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[{axis_name}] wrote artifacts to {out_dir}")


# ---------- Top-level ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate ablation_remap_function_*.sh sweep outputs.")
    ap.add_argument("--sweep-dir", action="append", required=True,
                    help="A sweep directory containing sweep_index.json. Repeat for multiple sweeps.")
    ap.add_argument("--output-dir", type=str, required=True,
                    help="Where to write tables, plots, and summary.")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sweeps: List[Dict[str, Any]] = []
    for sd in args.sweep_dir:
        sweep = load_sweep(Path(sd))
        emit_sweep(sweep, out_root)
        sweeps.append(sweep)

    # Cross-axis recommended hparams: for every mode, pick the param value
    # that was selected most often across all axis cells of all sweeps.
    all_chosen = pd.concat([s["chosen"] for s in sweeps if not s["chosen"].empty],
                           ignore_index=True) if sweeps else pd.DataFrame()
    rec_lines: List[str] = []
    if not all_chosen.empty:
        rec = (all_chosen.groupby(["mode", "mode_name", "param_name"])["param_value"]
               .agg(lambda s: s.value_counts().idxmax())
               .reset_index().rename(columns={"param_value": "recommended"}))
        rec.to_csv(out_root / "recommended_hparams.csv", index=False)
        rec_lines.append("## Cross-axis recommended hparams (mode of selections)\n")
        rec_lines.append(rec.to_markdown(index=False))

    index_lines = ["# Remap-function ablation summary\n"]
    for s in sweeps:
        axis = s["axis_name"]
        index_lines.append(f"- [`{axis}`]({axis}/summary.md)")
    if rec_lines:
        index_lines.append("")
        index_lines.extend(rec_lines)
    with open(out_root / "index.md", "w") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"[index] {out_root}/index.md")


if __name__ == "__main__":
    main()
