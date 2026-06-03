---
description: Draft N novel sparse-attention submissions in one shot for a chosen model — algorithm-innovation focus, all must compile, no benchmark loop, no memory.md mutation. May call /add-ops when an idea needs an op vortex lacks.
argument-hint: <N> [theme-hint] [--model <hf-id>]
---

Generate **N novel submissions** in a single shot — the *innovation-draft* mode,
complement to `/iterate`. No benchmark loop, no memory.md mutation, no batch-of-4
rule (`N` is whatever the user passes). Every variant must be (1) a genuinely
novel algorithm and (2) compile (`check_engine_config`).

## Step 0 — parse args

```bash
NARGS=($ARGUMENTS)
N=${NARGS[0]:-}; [ -z "$N" ] && { echo "usage: /innovate <N> [theme] [--model <hf-id>]"; exit 1; }
case "$N" in *[!0-9]*) echo "N must be a positive integer"; exit 1 ;; esac
[ "$N" -lt 1 ] && { echo "N must be >= 1"; exit 1; }
```
- Optional `--model <hf-id>` — the **target model** every variant must compile
  against (default `Qwen/Qwen3-1.7B`). Each generated `.json` sets
  `"model_path": "<model>"`. MLA models (DeepSeek/GLM) ⇒ `vFlowMLA` flows; run
  `/support-model <model>` first if unsure.
- The remaining non-flag words are a free-form theme hint.

Establish the env first (`python algorithm_scientist/detect_env.py`; `/setup-env` builds one) and use its run prefix as `$RUN` (GLM = transformers-5 env). `TAG=<sanitized model name>`;
`X=$(ls submissions/${TAG}/innovate_*_id0.json 2>/dev/null | wc -l)`.

## Step 1 — read context (and optionally research like a human)

[AI/AGENTS.md](../../AI/AGENTS.md) §1-§5, the op/program tutorials,
[papers/guide.md](../../papers/guide.md) §16 (off-catalog prompts),
[vortex_torch/flow/algorithms.py](../../vortex_torch/flow/algorithms.py).

Optional research aids ([AI/workflows/research_toolkit.md](../../AI/workflows/research_toolkit.md)):
run `/research-ideas "<theme>"` to survey prior art for genuinely-novel angles,
and — for any variant whose novelty is a new *scoring/selection* rule — **screen
it offline first** with the recall harness (`capture_trace.py` →
`eval_recall.py` with your method in `research/methods/`). A scoring idea that
loses to the centroid/quest/h2o baselines on recall is a cheap kill before you
even write the submission.

## Step 2 — per variant `y ∈ {0..N-1}`, state in one paragraph

The **novelty hypothesis** (one sentence, naming the specific framework op or
behaviour exploited — not "combine paper A + B"), the §16 sub-bucket or op-set
thread, and the cache fields + indexer ops it will use. With a theme hint, all
variants connect to it; without, they must be orthogonal.

## Step 3 — need an op vortex lacks? → call /add-ops

If a variant's mechanism needs an op that doesn't exist (or an existing op is
too slow for the idea to be worth testing), **invoke the `vortex-op-author`
subagent** (the `/add-ops` workflow, mode `new`/`accelerate`/`fuse`) to author
and verify it *before* writing the variant. Then write the variant against the
now-available op. Do not fake an op or call one that isn't registered.

## Step 4 — write each variant

`submissions/${TAG}/innovate_${X}_id${y}.{py,json}`:
- `@register("${TAG}_innovate_${X}_id${y}_cls")` (globally unique).
- Only `vortex_torch.indexer.*` / `vortex_torch.cache.*` ops; no native torch.
- `forward_indexer` ends in `topK`/`approxTopK` (ragged `[S,1,1]` score).
- `.json` sets `vortex_module_path`, `vortex_module_name`,
  `"model_path": "<model>"`, and `"disable_radix_cache": true` iff the indexer
  uses `Save(...)`.

## Step 5 — mandatory pre-flight gate

```bash
for y in $(seq 0 $((N-1))); do
  $RUN -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/innovate_${X}_id${y}.json')" \
    && echo "ok id$y" || echo "FAIL id$y"
done
```
Fix every failure in place (don't delete the variant). If a variant truly can't
compile after a fix attempt, surface the residual error and stop — no partial
run reported as complete.

## Step 6 — report + stop

Table: `y | file | bucket | hypothesis | new op? | preflight`. Then **stop**:
do NOT benchmark, do NOT touch `memory.md`. Hand the user the N paths; they feed
them into `/iterate` or `/batch-benchmark` (groups of 4) when ready.
