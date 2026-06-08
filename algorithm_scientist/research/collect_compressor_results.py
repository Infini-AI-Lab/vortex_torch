#!/usr/bin/env python
"""Scan logs/compressor/longrun_*.log and emit the final (or latest) held-out
eval per run as a JSON + markdown table. Keeps the report numbers parsed, not
hand-copied. Baselines (centroid/quest) are read from the same lines."""
from __future__ import annotations
import glob, json, os, re, sys

LOGDIR = "logs/compressor"
SETS = ["hf_holdout", "ruler8k", "ruler16k", "ruler32k", "longbench"]
# run stem -> (label, bytes/block/kv-head @bf16)
RUNS = {
    "s0":                  ("b2d128 (m2,d128)", 512),
    "b1d128_0of2":         ("b1d128 (m1,d128)", 256),
    "b4d128_0of2":         ("b4d128 (m4,d128)", 1024),
    "envelope":            ("envelope d128",    512),
    "envelope_d64":        ("envelope d64",     256),
    "envelope_d32":        ("envelope d32 trunc", 128),
    "envelope_d32lowpass": ("envelope d32 lowpass", 128),
    "envelope_topkloss":   ("envelope d128 +DSAloss", 512),
    # research sprint
    "sprint_perchan_tau":  ("envelope +perchan-tau", 512),
    "sprint_grouptarget":  ("envelope +grouptarget", 512),
    "sprint_cov_loss":     ("envelope coverage-loss", 512),
    "sprint_klcov":        ("envelope kl+coverage", 512),
    "sprint_lightning":    ("lightning (DSA ReLU)", 256),
    "sprint_mixlen":       ("envelope d128 mixlen", 512),
    "sprint_d16lowpass":   ("envelope d16 lowpass", 64),
    "sprint_d24lowpass":   ("envelope d24 lowpass", 96),
    "sprint_mixlen_d32lp": ("env mixlen+d32lp", 128),
    "sprint_mixlen_d64lp": ("env mixlen+d64lp", 256),
    "sprint_perchan_d32lp":("env perchan+d32lp", 128),
    # sprint 2: generalization across scale / page size / budget
    "sprint2_champion":    ("CHAMPION (all combined)", 128),
    "sprint2_block32":     ("envelope B=32", 512),
    "sprint2_block128":    ("envelope B=128", 512),
    "sprint2_qwen0p6b":    ("Qwen3-0.6B envelope", 512),
    "sprint2_qwen1p7b":    ("Qwen3-1.7B envelope", 512),
    "sprint2_qwen8b":      ("Qwen3-8B envelope", 512),
    "sprint2_qwen14b":     ("Qwen3-14B envelope", 512),
    "sprint2_budget8":     ("envelope budget=8", 512),
    "sprint2_budget30":    ("envelope budget=30", 512),
    "sprint2_block16":     ("envelope B=16", 512),
    "sprint2_bc3":         ("envelope b_c=3", 768),
    "sprint2_tieqk":       ("envelope tie-qk", 512),
    "sprint2_bc1env":      ("envelope b_c=1", 256),
}
EVAL = re.compile(
    r"EVAL (?P<set>\w+) n=\d+\] comp pooled p-cov=(?P<c>[\d.]+) r@16=(?P<c16>[\d.]+) "
    r"r@64=(?P<c64>[\d.]+) r@128=(?P<c128>[\d.]+).*?centroid pooled p-cov=(?P<ce>[\d.]+) "
    r"r@16=(?P<ce16>[\d.]+) r@64=(?P<ce64>[\d.]+).*?quest pooled p-cov=(?P<q>[\d.]+) "
    r"r@16=(?P<q16>[\d.]+) r@64=(?P<q64>[\d.]+)")


def latest(stem):
    f = os.path.join(LOGDIR, f"longrun_{stem}.log")
    if not os.path.isfile(f):
        f = os.path.join(LOGDIR, f"{stem}.log")   # sprint_* logs have no longrun_ prefix
    if not os.path.isfile(f):
        return None
    final, last, prompts, done = {}, {}, 0, False
    for line in open(f, errors="replace"):
        if "saved compressor" in line:
            m = re.search(r"prompts=(\d+)", line); prompts = int(m.group(1)) if m else prompts
            done = True
        m = EVAL.search(line)
        if not m:
            continue
        d = m.groupdict(); st = d.pop("set")
        rec = {k: float(v) for k, v in d.items()}
        last[st] = rec
        if "final" in line:
            final[st] = rec
    use = final if final else last
    return {"prompts": prompts, "done": done, "final": bool(final), "evals": use}


def main():
    out = {}
    for stem, (label, nbytes) in RUNS.items():
        r = latest(stem)
        if r:
            r["label"] = label; r["bytes"] = nbytes
            out[stem] = r
    os.makedirs("result/compressor", exist_ok=True)
    json.dump(out, open("result/compressor/results_summary.json", "w"), indent=2)

    # quest/centroid are identical across runs (same eval sets); pull from any.
    any_r = next(iter(out.values()))
    print(f"{'arch':<24} {'B':>5} {'tok(M)':>7} " + " ".join(f"{s[:8]:>8}" for s in SETS))
    for tag, base in [("quest", "q"), ("centroid", "ce")]:
        row = []
        for s in SETS:
            e = any_r["evals"].get(s)
            row.append(f"{e[base]:.3f}" if e else "  -  ")
        print(f"{tag:<24} {'':>5} {'':>7} " + " ".join(f"{v:>8}" for v in row))
    print("-" * 92)
    for stem, r in out.items():
        flag = "" if r["final"] else "~"
        row = []
        for s in SETS:
            e = r["evals"].get(s)
            row.append(f"{e['c']:.3f}" if e else "  -  ")
        print(f"{r['label']:<24} {r['bytes']:>5} {r['prompts']*16/1000:>6.0f}{flag} "
              + " ".join(f"{v:>8}" for v in row))
    print("\nwrote result/compressor/results_summary.json")


if __name__ == "__main__":
    main()
