#!/usr/bin/env python
"""NeurIPS-paper figures from result/compressor/results_summary.json."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "result/compressor/report_figs"
os.makedirs(OUT, exist_ok=True)
R = json.load(open("result/compressor/results_summary.json"))
SETS = ["hf_holdout", "ruler8k", "ruler16k", "ruler32k", "longbench"]
SLAB = ["holdout", "RULER-8K", "RULER-16K", "RULER-32K", "LongBench"]
any_r = next(iter(R.values()))
quest = [any_r["evals"][s]["q"] for s in SETS]
cent = [any_r["evals"][s]["ce"] for s in SETS]
def ev(stem, s, key="c"):
    r = R.get(stem)
    return r["evals"][s][key] if r and s in r["evals"] else None

# ---- fig1: beats-quest grouped bars (envelope d128 + d32lowpass vs quest/centroid)
fig, ax = plt.subplots(figsize=(7.0, 3.0))
x = range(len(SETS)); w = 0.2
ax.bar([i-1.5*w for i in x], cent, w, label="centroid (256B)", color="#9ecae1")
ax.bar([i-0.5*w for i in x], quest, w, label="Quest (512B)", color="#fc9272")
ax.bar([i+0.5*w for i in x], [ev("envelope",s) for s in SETS], w, label="envelope d128 (512B)", color="#31a354")
d32=[ev("envelope_d32lowpass",s) for s in SETS]
ax.bar([i+1.5*w for i in x], d32, w, label="envelope d32-lowpass (128B)", color="#006d2c")
ax.set_xticks(list(x), SLAB, fontsize=8); ax.set_ylabel("pooled p-coverage"); ax.set_ylim(0.4,0.95)
ax.legend(fontsize=7.5, ncol=2, loc="upper center"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_beatquest.pdf"); plt.close(fig); print("fig_beatquest.pdf")

# ---- fig2: cost-quality frontier (mean over 5 sets vs bytes)
def meanp(stem):
    vs=[ev(stem,s) for s in SETS]; vs=[v for v in vs if v]; return sum(vs)/len(vs) if vs else None
FRONT=[("envelope_d128_x","envelope d128",512,"envelope"),
       ("envelope_d64","envelope d64",256,"envelope_d64"),
       ("envelope_d32lowpass","env d32-lp",128,"envelope_d32lowpass"),
       ("envelope_d24lowpass","env d24-lp",96,"sprint_d24lowpass"),
       ("envelope_d16lowpass","env d16-lp",64,"sprint_d16lowpass"),
       ("b4","b4d128",1024,"b4d128_0of2"),("b2","b2d128",512,"s0"),("b1","b1d128",256,"b1d128_0of2")]
fig, ax = plt.subplots(figsize=(6.2,3.6))
qm=sum(quest)/5; cm=sum(cent)/5
ax.axhline(qm,color="#fc9272",ls="--",lw=1.3,label=f"Quest 512B ({qm:.3f})")
ax.axhline(cm,color="#9ecae1",ls=":",lw=1.3,label=f"centroid 256B ({cm:.3f})")
for _,lab,b,stem in FRONT:
    m=meanp(stem)
    if m is None: continue
    env=lab.startswith("env")
    ax.scatter(b,m,marker=("*" if env else "o"),s=(190 if env else 60),
               color=("#006d2c" if env else "#756bb1"),zorder=3)
    ax.annotate(lab,(b,m),fontsize=6.8,xytext=(4,3),textcoords="offset points")
ax.set_xscale("log",base=2); ax.set_xticks([64,128,256,512,1024]); ax.set_xticklabels(["64","128","256","512","1024"])
ax.set_xlabel("descriptor bytes / block / KV-head (bf16)"); ax.set_ylabel("mean pooled p-cov (5 sets)")
ax.legend(fontsize=7.5,loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_frontier2.pdf"); plt.close(fig); print("fig_frontier2.pdf")

# ---- fig3: length generalization (16K-train vs mixlen, across eval lengths)
LSET=["ruler8k","ruler16k","ruler32k"]; LX=[8,16,32]
fig, ax = plt.subplots(figsize=(5.0,3.3))
ax.plot(LX,[ev("envelope",s) for s in LSET],"o-",label="envelope (16K-train)",color="#31a354")
ax.plot(LX,[ev("sprint_mixlen",s) for s in LSET],"s-",label="envelope (mixed 8/16/32K)",color="#006d2c")
ax.plot(LX,[any_r["evals"][s]["q"] for s in LSET],"^--",label="Quest",color="#fc9272")
ax.plot(LX,[any_r["evals"][s]["ce"] for s in LSET],"v:",label="centroid",color="#9ecae1")
ax.set_xticks(LX,["8K","16K","32K"]); ax.set_xlabel("evaluation context length"); ax.set_ylabel("pooled p-coverage")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("Length generalization",fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_lengen.pdf"); plt.close(fig); print("fig_lengen.pdf ->",OUT)
