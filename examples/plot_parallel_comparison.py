#!/usr/bin/env python3
"""Aggregate baseline / fused / adaptive (split) TopK latencies across remap
functions and emit CSV tables + matplotlib bar plots.

Reads one remap_bench_*.json per (topk_val, num_splits) tag from the
directories produced by remap_function_bench_topk_parallel.sh and writes:

  results.csv            long-form per-(K, splits, batch, mode, dist) rows
  summary_topk<K>.csv    wide table per K (averaged across batch sizes)
  summary_all.csv        single combined wide table covering every K
  comparison_topk<K>.png bar plot per K (one bar group per mode)
  comparison_all.png     side-by-side per-K plots

Input format:
  --input "K=2048,splits=ns2=path/to/remap_bench_ns2.json"
  --input "K=30,splits=auto=path/to/remap_bench_auto.json"

The legacy "tag=path" input is also accepted; it lands in the all-K combined
plot but won't fill in the K/splits columns of results.csv.

Usage:
  python plot_parallel_comparison.py \
      --input "K=2048,splits=ns2=.../remap_bench_ns2.json" \
      --input "K=30,splits=auto=.../remap_bench_auto.json" \
      --output-dir <run_dir>/analysis [--emit-csv] [--emit-png]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_DISPLAY = {
    0: "None",
    3: "Power",
    4: "Log",
    6: "Asinh",
    7: "Log1p",
    9: "Erf",
    10: "Tanh",
    11: "Subtract",
    13: "ExpStretch",
    15: "ShiftPow2",
    16: "ShiftPow3",
    17: "LinearSteep",
    18: "HalfSquare",
    19: "HalfCube",
}


# --- input parsing -----------------------------------------------------------


def _parse_input_spec(spec: str) -> Tuple[str, dict, Path]:
    """Parse "K=2048,splits=ns2=path" -> ("K=2048,splits=ns2", {"K": "2048",
    "splits": "ns2"}, Path("path")).

    Falls back to the legacy "tag=path" format if no comma-separated
    key=value attrs precede the trailing "=path" segment.
    """
    if "=" not in spec:
        raise SystemExit(f"--input expects tag=path, got {spec!r}")
    # Split on the LAST '=' that doesn't follow a comma (the path delim).
    # Handle by scanning from the right.
    eq_positions = [i for i, ch in enumerate(spec) if ch == "="]
    path: str = ""
    label: str = spec
    for idx in reversed(eq_positions):
        candidate_path = spec[idx + 1:]
        if "/" in candidate_path or candidate_path.endswith(".json"):
            label = spec[:idx]
            path = candidate_path
            break
    if not path:
        # last-resort: split on the rightmost '='
        label, path = spec.rsplit("=", 1)

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{p} not found (input spec: {spec!r})")

    attrs: Dict[str, str] = {}
    for segment in label.split(","):
        segment = segment.strip()
        if not segment or "=" not in segment:
            continue
        k, v = segment.split("=", 1)
        attrs[k.strip()] = v.strip()
    return label, attrs, p


def _load_rows(json_path: Path) -> List[dict]:
    with open(json_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("results", [])


# --- aggregation ------------------------------------------------------------


def _per_mode_rows(rows: List[dict], distribution: str | None = None):
    """Yield one dict per (config, mode) pair so we can flatten to CSV."""
    for cfg in rows:
        if distribution is not None and cfg.get("distribution") != distribution:
            continue
        cfg_keys = {
            "batch_size":    cfg.get("batch_size"),
            "num_kv_heads":  cfg.get("num_kv_heads"),
            "seq_len":       cfg.get("seq_len"),
            "topk_val":      cfg.get("topk_val"),
            "distribution":  cfg.get("distribution"),
            "pages_per_seg": cfg.get("pages_per_seg"),
            "head":          cfg.get("head", "all"),
            "baseline_ms":   cfg.get("baseline_ms"),
        }
        for m in cfg.get("modes", []):
            mode_id = m.get("mode")
            if mode_id is None or mode_id < 0:
                continue
            yield {
                **cfg_keys,
                "mode":         mode_id,
                "mode_name":    MODE_DISPLAY.get(mode_id, m.get("mode_name", f"m{mode_id}")),
                "power":        m.get("power"),
                "fused_ms":     m.get("fused_ms")
                                    if m.get("fused_ms") is not None
                                    else (m.get("topk_after_remap_ms")
                                          if mode_id == 0 else None),
                "parallel_ms":  m.get("parallel_ms"),
                "parallel_splits": m.get("parallel_splits"),
                "remap_ms":     m.get("remap_ms"),
                "split_total_ms": m.get("split_total_ms"),
            }


def _aggregate_per_mode(rows: List[dict], distribution: str = "real"):
    """Return { mode -> {baseline, fused, parallel} } averaged across configs.
    Falls back to all distributions if `distribution` is empty for these rows.
    """
    used_dist = distribution
    flat = list(_per_mode_rows(rows, distribution))
    if not flat:
        used_dist = None
        flat = list(_per_mode_rows(rows, None))
    out: Dict[int, Dict[str, List[float]]] = {}
    for r in flat:
        bucket = out.setdefault(r["mode"],
                                {"baseline_ms": [], "fused_ms": [], "parallel_ms": []})
        if r.get("baseline_ms") is not None: bucket["baseline_ms"].append(r["baseline_ms"])
        if r.get("fused_ms")    is not None: bucket["fused_ms"].append(r["fused_ms"])
        if r.get("parallel_ms") is not None: bucket["parallel_ms"].append(r["parallel_ms"])
    summary = {
        m: {k: (sum(v) / len(v) if v else float("nan")) for k, v in sub.items()}
        for m, sub in out.items()
    }
    return summary, used_dist


# --- CSV writers ------------------------------------------------------------


def _write_results_csv(records: List[dict], out_path: Path) -> None:
    """Long-form per-(K, splits, batch, mode, dist) records → results.csv."""
    if not records:
        out_path.write_text("")
        return
    fields = list({k for r in records for k in r.keys()})
    # Stable column order: identifiers first.
    preferred = [
        "label", "K", "splits", "batch_size", "num_kv_heads", "seq_len",
        "topk_val", "distribution", "pages_per_seg", "head",
        "mode", "mode_name", "power",
        "baseline_ms", "fused_ms", "parallel_ms", "parallel_splits",
        "remap_ms", "split_total_ms",
    ]
    head = [c for c in preferred if c in fields]
    tail = sorted(c for c in fields if c not in preferred)
    cols = head + tail
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"  wrote {out_path}  ({len(records)} rows)")


def _write_summary_csv(tag: str, summary: Dict[int, Dict[str, float]],
                       out_path: Path, *, attrs: Dict[str, str] | None = None) -> None:
    attrs = attrs or {}
    cols = ["K", "splits", "tag", "mode", "mode_name",
            "baseline_ms", "fused_ms", "parallel_ms",
            "fused_speedup", "parallel_speedup"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for mode_id in sorted(summary):
            s = summary[mode_id]
            base  = s.get("baseline_ms",  float("nan"))
            fused = s.get("fused_ms",     float("nan"))
            par   = s.get("parallel_ms",  float("nan"))
            fs = (base / fused) if fused and not math.isnan(fused) else float("nan")
            ps = (base / par)   if par   and not math.isnan(par)   else float("nan")
            w.writerow([
                attrs.get("K", ""),
                attrs.get("splits", ""),
                tag,
                mode_id,
                MODE_DISPLAY.get(mode_id, f"m{mode_id}"),
                _csv_num(base), _csv_num(fused), _csv_num(par),
                _csv_num(fs),   _csv_num(ps),
            ])
    print(f"  wrote {out_path}")


def _write_summary_all_csv(summaries, attrs_by_tag, out_path: Path) -> None:
    cols = ["tag", "K", "splits", "mode", "mode_name",
            "baseline_ms", "fused_ms", "parallel_ms",
            "fused_speedup", "parallel_speedup"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tag, summary in summaries.items():
            attrs = attrs_by_tag.get(tag, {})
            for mode_id in sorted(summary):
                s = summary[mode_id]
                base  = s.get("baseline_ms",  float("nan"))
                fused = s.get("fused_ms",     float("nan"))
                par   = s.get("parallel_ms",  float("nan"))
                fs = (base / fused) if fused and not math.isnan(fused) else float("nan")
                ps = (base / par)   if par   and not math.isnan(par)   else float("nan")
                w.writerow([
                    tag,
                    attrs.get("K", ""),
                    attrs.get("splits", ""),
                    mode_id,
                    MODE_DISPLAY.get(mode_id, f"m{mode_id}"),
                    _csv_num(base), _csv_num(fused), _csv_num(par),
                    _csv_num(fs),   _csv_num(ps),
                ])
    print(f"  wrote {out_path}")


def _csv_num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.6f}"


# --- plotting ---------------------------------------------------------------


def _plot_bars(tag: str, summary: Dict[int, Dict[str, float]], out_path: Path) -> None:
    modes = sorted(summary.keys())
    labels = [MODE_DISPLAY.get(m, f"m{m}") for m in modes]
    base = [summary[m].get("baseline_ms", float("nan")) for m in modes]
    fused = [summary[m].get("fused_ms", float("nan")) for m in modes]
    par = [summary[m].get("parallel_ms", float("nan")) for m in modes]

    x = np.arange(len(modes))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(modes)), 5))
    ax.bar(x - w, base, w, label="Baseline (sglang)",          color="#888888")
    ax.bar(x,     fused, w, label="Fused (sglang_fused)",       color="#4C72B0")
    ax.bar(x + w, par,   w, label="Adaptive (output_adaptive)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms, lower is better)")
    ax.set_title(f"TopK kernel latency — {tag}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_combined(summaries: Dict[str, Dict[int, Dict[str, float]]], out_path: Path) -> None:
    if not summaries:
        return
    tags = list(summaries.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(max(8, 7 * len(tags)), 5), sharey=False)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        summary = summaries[tag]
        modes = sorted(summary.keys())
        labels = [MODE_DISPLAY.get(m, f"m{m}") for m in modes]
        base = [summary[m].get("baseline_ms", float("nan")) for m in modes]
        fused = [summary[m].get("fused_ms", float("nan")) for m in modes]
        par = [summary[m].get("parallel_ms", float("nan")) for m in modes]
        x = np.arange(len(modes))
        w = 0.27
        ax.bar(x - w, base, w, label="Baseline", color="#888888")
        ax.bar(x,     fused, w, label="Fused",    color="#4C72B0")
        ax.bar(x + w, par,   w, label="Adaptive", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(tag)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Adaptive (split) vs Fused vs Baseline TopK", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- main -------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True,
                   help="<tag-or-K=...,splits=...>=path/to/remap_bench_*.json (repeatable).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--distribution", default="real",
                   help="Distribution column to aggregate (falls back to all).")
    p.add_argument("--emit-csv", action="store_true",
                   help="Write CSV tables (always on; flag kept for explicitness).")
    p.add_argument("--emit-png", action="store_true",
                   help="Write PNG plots (always on; flag kept for explicitness).")
    args = p.parse_args()
    # Always emit both — flags are kept so the shell wrapper can document intent.
    args.emit_csv = True
    args.emit_png = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[int, Dict[str, float]]] = {}
    attrs_by_tag: Dict[str, Dict[str, str]] = {}
    long_form: List[dict] = []

    for spec in args.input:
        tag, attrs, path = _parse_input_spec(spec)
        rows = _load_rows(path)

        # Long-form rows for results.csv
        for r in _per_mode_rows(rows, args.distribution) or _per_mode_rows(rows, None):
            long_form.append({
                "label":  tag,
                "K":      attrs.get("K", ""),
                "splits": attrs.get("splits", ""),
                **r,
            })

        summary, _used_dist = _aggregate_per_mode(rows, distribution=args.distribution)
        summaries[tag] = summary
        attrs_by_tag[tag] = attrs

        if args.emit_csv:
            _write_summary_csv(tag, summary,
                               out_dir / f"summary_{_safe(tag)}.csv",
                               attrs=attrs)
        if args.emit_png:
            _plot_bars(tag, summary, out_dir / f"comparison_{_safe(tag)}.png")

    if args.emit_csv:
        _write_results_csv(long_form, out_dir / "results.csv")
        _write_summary_all_csv(summaries, attrs_by_tag, out_dir / "summary_all.csv")

    if args.emit_png and len(summaries) > 1:
        _plot_combined(summaries, out_dir / "comparison_all.png")


def _safe(tag: str) -> str:
    """Make a tag safe for use in a filename (strip ',', '=', '/')."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


if __name__ == "__main__":
    main()
