#!/usr/bin/env python
"""Aggregate verify_algo.py summary JSONs into a single markdown report.

Reads every ``*.json`` in --summary-dir, builds a results table (full_attention
baseline first, then sparse modules by topk), and embeds the raw summaries.
Re-runnable: regenerates --out from whatever summaries currently exist, keeping
the newest summary per (module, backend, topk, block, tensor-core, skip) key.

Usage:
    python examples/misc/collect_algo2_results.py \
        --summary-dir summary-glm4.7-flash --out examples/misc/algo2_results.md
"""
import argparse, json, os, glob, datetime

PLAN = """\
# GLM-4.7-Flash · MLA sparse-attention experiments (AIME26, mean@16)

**Driver:** `examples/misc/algo2.sh` → `examples/math/verify_algo.py`
&nbsp;·&nbsp; **Data:** `examples/math/aime26_glm.jsonl`
&nbsp;·&nbsp; **Model:** `zai-org/GLM-4.7-Flash`
&nbsp;·&nbsp; **trials:** 16 &nbsp;·&nbsp; **block=page=16** &nbsp;·&nbsp; **gen ≤ 32768 tok** &nbsp;·&nbsp; **tp=1**

## Plan

| # | module | backend | sparsity | runs |
|---|--------|---------|----------|------|
| 1 | `full_attention` | `flashinfer` (dense MLA) | — (dense baseline) | **1** (topk ignored) |
| 2 | `rope_aware_block_sparse_mla` | `cuda_mla` decode + Triton **tensor-core** indexer, **no layer skip** | topk ∈ {61,93,125,157,253} | 5 |
| 3 | `lserve_centroid_mla` | `cuda_mla` decode + Triton **tensor-core** indexer, **no layer skip** | topk ∈ {61,93,125,157,253} | 5 |

*Why these settings.* `full_attention` is dense, so it cannot use a vortex sparse
backend (those require `enable_vortex_sparsity=True`) — it runs on the dense
flashinfer MLA path as the accuracy ceiling / throughput-floor reference, once.
The two sparse modules run on the geometry-agnostic **`cuda_mla`** hand-CUDA decode;
**`--vortex-use-tensor-core`** turns on the bf16-MMA Triton indexer (hence
`--vortex-impl-backend triton`); **`--vortex-layers-skip`** with no values keeps
*every* layer sparse. The topk sweep traces the accuracy/throughput Pareto.
"""

METRIC_KEYS = ["mean", "pass_trials", "pass@8", "pass@4", "throughput", "total_tokens", "e2e_time"]


def load(summary_dir):
    rows = []
    for path in glob.glob(os.path.join(summary_dir, "*.json")):
        try:
            with open(path) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        a = s.get("args", {})
        trials = a.get("trials")
        mean = s.get(f"mean@{trials}")
        rows.append({
            "path": path, "mtime": os.path.getmtime(path), "raw": s, "args": a,
            "module": a.get("vortex_module_name", "?"),
            "backend": a.get("attention_backend", "?"),
            "topk": a.get("topk_val"), "block": a.get("block_size"),
            "tc": a.get("vortex_use_tensor_core", False),
            "skip": a.get("vortex_layers_skip"),
            "mean": mean, "pass_trials": s.get(f"pass@{trials}"),
            "pass@8": s.get("pass@8"), "pass@4": s.get("pass@4"),
            "throughput": s.get("throughput"), "total_tokens": s.get("total_tokens"),
            "e2e_time": s.get("e2e_time"), "trials": trials,
        })
    # keep newest per config key
    best = {}
    for r in rows:
        key = (r["module"], r["backend"], r["topk"], r["block"], r["tc"], str(r["skip"]))
        if key not in best or r["mtime"] > best[key]["mtime"]:
            best[key] = r
    return list(best.values())


def fnum(x, p=1):
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", default="summary-glm4.7-flash")
    ap.add_argument("--out", default="examples/misc/algo2_results.md")
    args = ap.parse_args()

    rows = load(args.summary_dir)
    # order: full_attention baseline first, then module, then topk
    def sort_key(r):
        is_full = 0 if r["module"] == "full_attention" else 1
        return (is_full, r["module"], r["topk"] if r["topk"] is not None else -1)
    rows.sort(key=sort_key)

    out = [PLAN, "\n## Results\n",
           f"*Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
           f"from `{args.summary_dir}/` ({len(rows)} run(s)).*\n",
           "| module | backend | topk | blk | TC | skip | mean@16 | pass@16 | pass@8 | pass@4 | tok/s | out_tok | e2e_s |",
           "|---|---|--:|--:|:--:|:--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        skip = r["skip"]
        skip_s = "none" if skip == [] else ("[0]" if skip is None else str(skip))
        out.append(
            f"| `{r['module']}` | {r['backend']} | {r['topk'] if r['topk'] is not None else '—'} "
            f"| {r['block']} | {'✓' if r['tc'] else '·'} | {skip_s} "
            f"| {fnum(r['mean'],4)} | {fnum(r['pass_trials'],4)} | {fnum(r['pass@8'],4)} "
            f"| {fnum(r['pass@4'],4)} | {fnum(r['throughput'])} | {r['total_tokens'] or '—'} "
            f"| {fnum(r['e2e_time'])} |")
    if not rows:
        out.append("| _(no summaries yet — run `examples/misc/algo2.sh`)_ |||||||||||||")

    # raw summaries
    out.append("\n## Original summaries (raw)\n")
    for r in rows:
        out.append(f"<details><summary><code>{os.path.basename(r['path'])}</code></summary>\n")
        out.append("```json")
        out.append(json.dumps(r["raw"], indent=2, ensure_ascii=False))
        out.append("```\n</details>\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {args.out}  ({len(rows)} run(s))")
    # also echo the table to stdout
    print("\n".join(l for l in out if l.startswith("|")))


if __name__ == "__main__":
    main()
