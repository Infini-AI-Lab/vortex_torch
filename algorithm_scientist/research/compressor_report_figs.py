#!/usr/bin/env python
"""Figures for the MHA-compressor research report (PDF, matplotlib).

  fig_rope.pdf      — predicted alpha_i vs measured channel coherence rho_c vs
                      trained-Wk channel energy (the H2/H3 verification)
  fig_frontier.pdf  — arch sweep: accuracy vs descriptor bytes/block frontier
  fig_gran.pdf      — centroid/quest granularity control @ fixed token budget
  fig_longrun.pdf   — long-run eval curves vs tokens seen

Inputs: result/compressor/math_verify/math_verify.json, sweep logs
(logs/compressor/sweep16k_*.log), long-run log. Robust to missing inputs
(skips the figure)."""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "result/compressor/report_figs"
os.makedirs(OUT, exist_ok=True)

EVAL_RE = re.compile(
    r"\[EVAL (?P<set>\w+) n=(?P<n>\d+)\] comp pooled p-cov=(?P<pcov>[\d.]+) "
    r"r@16=(?P<r16>[\d.]+) r@64=(?P<r64>[\d.]+)(?: r@128=(?P<r128>[\d.]+))?.*?"
    r"centroid pooled p-cov=(?P<cent>[\d.]+).*?quest pooled p-cov=(?P<quest>[\d.]+)")


def parse_evals(path):
    """{tag: {set: dict}} for tag in baseline|t..|final, keeping last of each."""
    out = {}
    if not os.path.isfile(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        m = EVAL_RE.search(line)
        if not m:
            continue
        tag = "baseline" if "baseline" in line else ("final" if "final" in line else "mid")
        d = {k: (float(v) if k not in ("set", "n") else v) for k, v in m.groupdict().items()
             if v is not None}
        out.setdefault(tag, {})[m.group("set")] = d
    return out


# ---------------- fig_rope ----------------
mv_path = "result/compressor/math_verify/math_verify.json"
if os.path.isfile(mv_path):
    mv = json.load(open(mv_path))
    d = len(mv["alpha"])
    half = d // 2
    pairs = list(range(half))
    alpha = mv["alpha"][:half]
    rho = mv["h2_rho"]
    rho_pair = [(rho[i] + rho[i + half]) / 2 for i in range(half)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(pairs, alpha, "k-", lw=2, label=r"predicted $\alpha_i$ (Dirichlet)")
    ax.plot(pairs, rho_pair, "C0o", ms=3.5,
            label=r"measured $\rho_i=\mathbb{E}|c|/\mathbb{E}|k|$")
    if mv.get("h3"):
        e = mv["h3"]["wk_pair_energy"]
        emax = max(e)
        ax.plot(pairs, [x / emax for x in e], "C3s", ms=3,
                label=f"trained $W_k$ energy (norm.; corr="
                      f"{mv['h3']['corr_energy_alpha']:.2f})")
    ax.set_xlabel("RoPE pair index $i$ (high freq → low freq)")
    ax.set_ylabel("coherence / energy")
    ax.set_title(f"Mean-pooling decoheres high-frequency RoPE channels (B=64); "
                 f"corr($\\rho,\\alpha$)={mv['h2_corr_rho_alpha']:.2f}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_rope.pdf"); plt.close(fig)
    print("fig_rope.pdf")

    # ---------------- fig_gran ----------------
    g = mv["ctrl_granularity"]
    bss = sorted(g, key=int)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = range(len(bss)); w = 0.35
    ax.bar([i - w / 2 for i in x], [g[b]["centroid_pcov"] for b in bss], w,
           label="centroid", color="C0")
    ax.bar([i + w / 2 for i in x], [g[b]["quest_pcov"] for b in bss], w,
           label="quest", color="C3")
    ax.set_xticks(list(x), [f"block {b}\n({960 // int(b)} blocks)" for b in bss])
    ax.set_ylabel("pooled p-coverage")
    ax.set_title("Baselines vs granularity @ fixed 960-token budget", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_gran.pdf"); plt.close(fig)
    print("fig_gran.pdf")

# ---------------- fig_frontier ----------------
SWEEPS = {  # name -> (bytes per block per kv head @bf16, marker)
    "b1d128": (1 * 128 * 2, "o"), "b2d128": (2 * 128 * 2, "s"),
    "b1d64": (1 * 64 * 2, "^"), "b2d64": (2 * 64 * 2, "v"),
    "lowpass128": (1 * 128 * 2, "P"), "mlp128": (1 * 128 * 2, "X"),
}
rows = []
for name, (nbytes, mk) in SWEEPS.items():
    ev = parse_evals(f"logs/compressor/sweep16k_{name}.log")
    if "final" in ev and "hf_holdout" in ev["final"]:
        rows.append((name, nbytes, mk, ev["final"]))
if rows:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for ax, st, ttl in [(axes[0], "hf_holdout", "held-out (16K, in-distribution)"),
                        (axes[1], "ruler32k", "RULER-32K (length+dataset shift)")]:
        for name, nbytes, mk, fin in rows:
            if st in fin:
                ax.scatter(nbytes, fin[st]["pcov"], marker=mk, s=55, label=name)
        ref = rows[0][3].get(st, {})
        if ref:
            ax.axhline(ref["cent"], color="C0", ls=":", lw=1.2, label="centroid (256B)")
            ax.axhline(ref["quest"], color="C3", ls="--", lw=1.2, label="quest (512B)")
        ax.set_xlabel("descriptor bytes / block / kv-head")
        ax.set_ylabel("pooled p-coverage"); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f"{OUT}/fig_frontier.pdf"); plt.close(fig)
    print("fig_frontier.pdf")

# ---------------- fig_wt ----------------
wt_path = "result/compressor/math_verify/wt_profile.json"
if os.path.isfile(wt_path):
    prof = json.load(open(wt_path))["profile"]            # [bs][m]
    bs = len(prof); m = len(prof[0])
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for j in range(m):
        ax.plot(range(bs), [row[j] for row in prof], "o-", ms=2.5,
                label=f"landmark {j}")
    ax.axhline(1 / bs, color="gray", ls=":", lw=1.2, label="uniform init $1/B$")
    ax.set_xlabel("within-block position $s$")
    ax.set_ylabel(r"learned token weight $\bar{W}_t[s]$")
    ax.set_title("Trained token-mixing: block-leading tokens dominate", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_wt.pdf"); plt.close(fig)
    print("fig_wt.pdf")

# ---------------- fig_longrun ----------------
LR_LOG = "logs/compressor/longrun16k.log"
if os.path.isfile(LR_LOG):
    pts = {}   # set -> [(prompts, pcov)]
    cur_p = [0]
    for line in open(LR_LOG, errors="replace"):
        mm = re.search(r"\[t\s*[\d.]+m p(\d+)\]", line)
        if mm:
            cur_p[0] = int(mm.group(1))
        m = EVAL_RE.search(line)
        if m and "baseline" not in line:
            pts.setdefault(m.group("set"), []).append((cur_p[0], float(m.group("pcov"))))
    if pts:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        for st, v in sorted(pts.items()):
            v = sorted(v)
            ax.plot([x / 1000 for x, _ in v], [y for _, y in v], "o-", ms=3, label=st)
        ax.set_xlabel("training sequences seen (×1000, 16K tokens each)")
        ax.set_ylabel("pooled p-coverage"); ax.grid(alpha=0.3)
        ax.set_title("Long-run scaling", fontsize=10); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_longrun.pdf"); plt.close(fig)
        print("fig_longrun.pdf")

print("done ->", OUT)
