---
description: Reproduce a paper's sparse-attention algorithm as a vortex submission — read it (arXiv id / local PDF / papers entry), map the mechanism onto vortex ops, scaffold + preflight, and call /add-ops when an op is missing.
argument-hint: <arxiv-id | path/to.pdf | papers/<name>> [--model <hf-id>]
---

Compose the algorithm from a paper as a runnable vortex submission. Reference:
[AI/workflows/paper.md](../../AI/workflows/paper.md) (Reproduce mode). Driven by
the `vortex-paper-writer` subagent for reading, and `vortex-op-author` when an
op is missing.

## Step 0 — acquire + read the paper

`$1` is one of:
- **arXiv id** (e.g. `2406.10774`) → fetch (WebFetch `arxiv.org/abs/<id>` /
  `/pdf/<id>`).
- **local PDF path** → Read it.
- **`papers/<name>`** → a curated entry; `papers/guide.md` already summarizes
  the ten bundled papers (sinks, heavy-hitters, QUEST, channel sparsity, low-rank
  K, LSH, dual-band centroids…).

Extract two things precisely: the **sparsity mechanism** (what each query
attends to) and the **scoring** (how pages/blocks are ranked/selected).

## Step 1 — map onto vortex ops

Express the score in `forward_indexer` (ending in `topK`/`approxTopK`, ragged
`[S,1,1]`); put per-block summaries in `create_cache` / `forward_cache`. Check
each needed op exists in `vortex_torch/{indexer,cache}/` (see the op tutorials
and `vortex_torch/flow/algorithms.py` for patterns). Write the mapping table:
`paper concept → vortex op(s) / cache field`.

## Step 2 — missing op? → /add-ops

If the mechanism needs an op vortex lacks (or one too slow to be faithful),
invoke the `vortex-op-author` subagent (the `/add-ops` workflow) to author +
verify it, then continue. Don't approximate the algorithm away just to avoid
adding an op — note any deliberate simplification explicitly.

## Step 3 — scaffold, preflight, gate

Use `/new-submission` to scaffold `submissions/<tag>/<paper>_v0.{py,json}` (set
`"model_path": "<model>"`, default `Qwen/Qwen3-1.7B`; MLA paper ⇒ `vFlowMLA`).
Pre-flight (`check_engine_config`) and RULER-gate (≥0.85). Fix until it compiles
and passes.

## Step 4 — (optional) evaluate the tradeoff

Hand the submission(s) to `/iterate --model <model> --task <task>` (or
`/batch-benchmark` in groups of 4) to map the accuracy–throughput tradeoff
against full attention and the running best.

## Output

`paper | mechanism (1 sentence) | op mapping | new ops added | files | preflight
| RULER`. Record the reproduction (paper→flow mapping, any new ops, deliberate
simplifications) in `algorithm_scientist/memory.md`.
