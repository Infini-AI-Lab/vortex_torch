#!/usr/bin/env python
"""Report figures from result/compressor/results_summary.json:
  fig_beatquest.pdf — envelope vs Quest/centroid across the 5 eval sets
  fig_frontier2.pdf — cost (bytes) vs quality, all archs, with the Quest line
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "result/compressor/report_figs"
os.makedirs(OUT, exist_ok=True)
R = json.load(open("result/compressor/results_summary.json"))
SETS = ["hf_holdout", "ruler8k", "ruler16k", "ruler32k", "longbench"]
SLAB = ["holdout\n16K", "RULER\n8K", "RULER\n16K", "RULER\n32K", "Long\nBench"]
any_r = next(iter(R.values()))
quest = [any_r["evals"][s]["q"] for s in SETS]
cent = [any_r["evals"][s]["ce"] for s in SETS]

# ---- fig_beatquest: grouped bars envelope vs quest vs centroid ----
env = R.get("envelope")
if env:
    e = [env["evals"][s]["c"] for s in SETS]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = range(len(SETS)); w = 0.26
    ax.bar([i - w for i in x], cent, w, label="centroid (256 B)", color="#9ecae1")
    ax.bar(list(x), quest, w, label="Quest (512 B)", color="#fc9272")
    ax.bar([i + w for i in x], e, w, label="envelope d128 (512 B)", color="#31a354")
    ax.set_xticks(list(x), SLAB); ax.set_ylabel("pooled p-coverage")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Learned envelope beats Quest on every distribution (iso-cost)", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="upper center"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_beatquest.pdf"); plt.close(fig)
    print("fig_beatquest.pdf")

# ---- fig_frontier2: bytes vs mean-over-sets p-cov ----
def mean_pcov(r):
    vs = [r["evals"][s]["c"] for s in SETS if s in r["evals"]]
    return sum(vs) / len(vs) if vs else 0.0

pts = []
for stem, r in R.items():
    if "trunc" in r["label"]:
        continue  # broken control, shown separately in text
    pts.append((r["bytes"], mean_pcov(r), r["label"]))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
qm = sum(quest) / len(quest); cm = sum(cent) / len(cent)
ax.axhline(qm, color="#fc9272", ls="--", lw=1.3, label=f"Quest (512 B, mean {qm:.3f})")
ax.axhline(cm, color="#9ecae1", ls=":", lw=1.3, label=f"centroid (mean {cm:.3f})")
for b, m, lab in pts:
    mk = "*" if lab.startswith("envelope") else "o"
    sz = 180 if lab.startswith("envelope") else 70
    col = "#31a354" if lab.startswith("envelope") else "#756bb1"
    ax.scatter(b, m, marker=mk, s=sz, color=col, zorder=3)
    ax.annotate(lab.replace("envelope ", "env "), (b, m), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.axvline(512, color="gray", lw=0.6, alpha=0.5)
ax.set_xlabel("descriptor bytes / block / kv-head (bf16)")
ax.set_ylabel("mean pooled p-coverage (5 sets)")
ax.set_xscale("log", base=2); ax.set_xticks([128, 256, 512, 1024])
ax.set_xticklabels(["128", "256", "512", "1024"])
ax.set_title("Cost–quality frontier: envelope dominates at every budget", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_frontier2.pdf"); plt.close(fig)
print("fig_frontier2.pdf  ->", OUT)
