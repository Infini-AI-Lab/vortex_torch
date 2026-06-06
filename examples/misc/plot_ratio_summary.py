"""Plot accuracy (mean@largest-trials) and throughput (max across trials)
vs topk_ratio for each (model, dataset), comparing block_sparse vs gqa_quest.

Reads summary JSONs under summary_ratio/ and writes PNGs to figures_ratio/.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

MODEL_ORDER = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
DATASETS = ["examples/math/amc23.jsonl", "examples/math/aime24.jsonl"]
MODULES = [
    "block_sparse_attention",
    "gqa_quest_sparse_attention",
    "gqa_block_sparse_attention",
]
MODULE_LABEL = {
    "block_sparse_attention": "block_sparse",
    "gqa_quest_sparse_attention": "gqa_quest",
    "gqa_block_sparse_attention": "gqa_block_sparse",
    "full_attention": "full_attention",
}
MODULE_STYLE = {
    "block_sparse_attention": dict(color="#2E86DE", marker="o", linewidth=2.2, markersize=7),
    "gqa_quest_sparse_attention": dict(color="#EE5253", marker="s", linewidth=2.2, markersize=7),
    "gqa_block_sparse_attention": dict(color="#8E44AD", marker="^", linewidth=2.2, markersize=8),
    "full_attention": dict(color="#10AC84"),
}

ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json$")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#CCCCCC",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
    })


def style_axes(ax):
    ax.grid(True, which="major", alpha=0.55)
    ax.grid(True, which="minor", alpha=0.25, linestyle=":")
    ax.minorticks_on()


def load_latest(summary_dirs):
    """Merge latest run per config across one or more summary directories."""
    if isinstance(summary_dirs, str):
        summary_dirs = [summary_dirs]
    best = {}
    for summary_dir in summary_dirs:
        for f in sorted(glob.glob(os.path.join(summary_dir, "*.json"))):
            with open(f) as fp:
                d = json.load(fp)
            a = d["args"]
            key = (
                os.path.basename(a["model_name"]),
                a["vortex_module_name"],
                a["trials"],
                a["topk_ratio"],
                a["data_path"],
            )
            ts = ts_re.search(f).group(1)
            if key not in best or ts > best[key][0]:
                best[key] = (ts, d)
    return best


def aggregate(best):
    bucket = defaultdict(dict)
    for (model, module, trials, tk, ds), (_, d) in best.items():
        mean_v = d.get(f"mean@{trials}")
        tput = d.get("throughput")
        bucket[(model, ds, module, tk)][trials] = (mean_v, tput)

    agg = {}
    for key, per_trial in bucket.items():
        max_t = max(per_trial.keys())
        acc = per_trial[max_t][0]
        tputs = [v[1] for v in per_trial.values() if v[1] is not None]
        tput = max(tputs) if tputs else None
        agg[key] = {"acc": acc, "tput": tput, "acc_trials": max_t}
    return agg


def load_baseline(baseline_dir):
    """Load full_attention runs from baseline_dir.

    Returns: {(model, dataset): {"acc": float, "tput": float}} using the same
    rule as figures_val: accuracy = mean@max-trials, throughput = max across trials.
    """
    if not os.path.isdir(baseline_dir):
        return {}
    best = {}
    for f in sorted(glob.glob(os.path.join(baseline_dir, "*.json"))):
        with open(f) as fp:
            d = json.load(fp)
        a = d["args"]
        if a.get("vortex_module_name") != "full_attention":
            continue
        key = (
            os.path.basename(a["model_name"]),
            a["trials"],
            a["data_path"],
        )
        ts = ts_re.search(f).group(1)
        if key not in best or ts > best[key][0]:
            best[key] = (ts, d)

    bucket = defaultdict(dict)  # (model, ds) -> {trials: (mean, tput)}
    for (model, trials, ds), (_, d) in best.items():
        mean_v = d.get(f"mean@{trials}")
        tput = d.get("throughput")
        bucket[(model, ds)][trials] = (mean_v, tput)

    out = {}
    for key, per_trial in bucket.items():
        max_t = max(per_trial.keys())
        acc = per_trial[max_t][0]
        tputs = [v[1] for v in per_trial.values() if v[1] is not None]
        tput = max(tputs) if tputs else None
        out[key] = {"acc": acc, "tput": tput}
    return out


def plot_side_by_side(agg, baseline, model, ds, title_suffix, savepath):
    fig, (ax_acc, ax_tp) = plt.subplots(1, 2, figsize=(12, 4.3))

    for module in MODULES:
        pts = [(tk, v) for (m, d, mo, tk), v in agg.items()
               if m == model and d == ds and mo == module]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs = [p[0] for p in pts]
        accs = [p[1]["acc"] for p in pts]
        tps = [p[1]["tput"] for p in pts]
        st = MODULE_STYLE[module]
        ax_acc.plot(xs, accs, label=MODULE_LABEL[module],
                    markerfacecolor="white", markeredgewidth=1.8, **st)
        ax_tp.plot(xs, tps, label=MODULE_LABEL[module],
                   markerfacecolor="white", markeredgewidth=1.8, **st)

    fa = baseline.get((model, ds))
    fa_color = MODULE_STYLE["full_attention"]["color"]
    if fa is not None:
        if fa["acc"] is not None:
            ax_acc.axhline(fa["acc"], linestyle="--", linewidth=1.8, color=fa_color,
                           label=f"full_attention ({fa['acc']:.3f})")
        if fa["tput"] is not None:
            ax_tp.axhline(fa["tput"], linestyle="--", linewidth=1.8, color=fa_color,
                          label=f"full_attention ({fa['tput']:.0f})")

    ds_name = os.path.basename(ds).replace(".jsonl", "")
    suptitle = f"{model} — {ds_name}"
    if title_suffix:
        suptitle += f"  [{title_suffix}]"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    style_axes(ax_acc)
    ax_acc.set_xlabel("topk_ratio")
    ax_acc.set_ylabel("accuracy (mean @ max trials)")
    ax_acc.set_title("Accuracy vs topk_ratio")
    ax_acc.margins(x=0.05, y=0.10)
    ax_acc.legend(loc="best")

    style_axes(ax_tp)
    ax_tp.set_xlabel("topk_ratio")
    ax_tp.set_ylabel("throughput (tokens/s)")
    ax_tp.set_title("Best throughput vs topk_ratio")
    ax_tp.margins(x=0.05, y=0.10)
    ax_tp.legend(loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(agg, baseline, model, ds, title_suffix, savepath):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for module in MODULES:
        pts = [(tk, v) for (m, d, mo, tk), v in agg.items()
               if m == model and d == ds and mo == module]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        xs = [p[1]["tput"] for p in pts]
        ys = [p[1]["acc"] for p in pts]
        tks = [p[0] for p in pts]
        st = MODULE_STYLE[module]
        ax.plot(xs, ys, label=MODULE_LABEL[module],
                markerfacecolor="white", markeredgewidth=1.8, **st)
        for x, y, tk in zip(xs, ys, tks):
            ax.annotate(f"{tk:g}", (x, y), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points",
                        color=st["color"])

    fa = baseline.get((model, ds))
    if fa is not None and fa["acc"] is not None and fa["tput"] is not None:
        ax.scatter([fa["tput"]], [fa["acc"]], marker="*", s=260,
                   color=MODULE_STYLE["full_attention"]["color"],
                   edgecolors="white", linewidths=1.2,
                   label="full_attention", zorder=5)

    ds_name = os.path.basename(ds).replace(".jsonl", "")
    title = f"{model} — {ds_name}  (labels: topk_ratio)"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("throughput (tokens/s)")
    ax.set_ylabel("accuracy (mean @ max trials)")
    style_axes(ax)
    ax.margins(x=0.08, y=0.12)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-dir", default=["summary_ratio"], nargs="+",
                   help="One or more directories of topk_ratio sweep JSONs "
                        "(default: summary_ratio). Runs are deduped by "
                        "(model, module, trials, topk_ratio, dataset), keeping "
                        "the latest timestamp across all dirs.")
    p.add_argument("--baseline-dir", default="summary",
                   help="Directory to read full_attention baseline from "
                        "(default: summary)")
    p.add_argument("--out-dir", default="figures_ratio",
                   help="Directory to write PNGs (default: figures_ratio)")
    p.add_argument("--title-suffix", default="",
                   help="Text appended in brackets to each figure title "
                        "(e.g. 'topk_val=61').")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    apply_style()
    best = load_latest(args.summary_dir)
    agg = aggregate(best)
    baseline = load_baseline(args.baseline_dir)

    for model in MODEL_ORDER:
        for ds in DATASETS:
            has_data = any(k[0] == model and k[1] == ds for k in agg.keys())
            if not has_data:
                continue
            ds_name = os.path.basename(ds).replace(".jsonl", "")
            plot_side_by_side(
                agg, baseline, model, ds, args.title_suffix,
                os.path.join(args.out_dir, f"{model}_{ds_name}_acc_tput.png"))
            plot_pareto(
                agg, baseline, model, ds, args.title_suffix,
                os.path.join(args.out_dir, f"{model}_{ds_name}_pareto.png"))

    print(f"Wrote figures to {args.out_dir}/")


if __name__ == "__main__":
    main()
