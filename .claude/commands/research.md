---
description: Run the full human-researcher loop on a question — survey → analyze real attention → hypothesize → offline recall screen → RULER gate → AIME/iterate → journal + paper-ready summary. Ties every research tool together.
argument-hint: <research question> [--model <hf-id>] [--task aime24|aime25|aime26|amc23] [--max-iterations N]
---

Conduct sparse-attention research the way a person does: understand the problem
empirically, form hypotheses, screen them cheaply, then spend GPU only on
survivors. Reference:
[AI/workflows/research_toolkit.md](../../AI/workflows/research_toolkit.md). Record
everything in `algorithm_scientist/research/journal.md`.

Defaults: `--model Qwen/Qwen3-1.7B`, `--task aime24`. Establish the env first (`python algorithm_scientist/detect_env.py`; `/setup-env`) and adopt its run prefix as `$RUN`
(GLM = the transformers-5 env). Pick a `<tag>`.

## 1 — Survey (prior art)

Run `/research-ideas "<question>"` (add `--deep` for a `deep-research` sweep) →
a brief under `research/briefs/`. Pull candidate mechanisms + their vortex
mapping.

## 2 — Analyze the model's real attention (where to be sparse)

**You choose the calibration dataset — this matters.** Attention is
workload-dependent, so the trace must match what you're optimizing for or the
recall screen will mispredict:
- Optimizing a **task** (e.g. `--task aime24`) → capture on that task's prompts
  (`--data examples/math/aime24.jsonl`, or the `make_task.py` jsonl for a non-default
  model) and add `--generate <N>` so the captured query is a real
  **mid-generation** decode step (AIME generates long outputs — the
  throughput-relevant attention is late in generation, not the prompt's last token).
- Hunting **long-context retrieval heads** → `--data examples/ruler/validation_4k.jsonl`
  (RULER/NIAH), large `--max-ctx`.
- Mix a few sources/samples for robustness. Record why you picked it in the journal.

**Run capture on a GPU for long context** (`--device cuda` — memory-efficient:
SDPA forward + cheap stores). `capture_trace.py` is a *reference* for HF models
on the unified attention interface; if your model/path isn't covered, or you need
the *real* vortex page summaries, you may have to instrument HF or sglang
yourself to save tensors (for sglang set `"disable_cuda_graph": true` and dump in
`forward_indexer`/`forward_cache`). Don't assume one script fits every case.

**Detect a free GPU at launch** (the set is dynamic/shared — never hardcode 0;
with several free, run multiple captures/RULER in parallel, one per free GPU):
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; exit 1; }
CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $RUN algorithm_scientist/research/capture_trace.py \
  --model <model> --data <dataset you chose> --num-samples 3 --max-ctx 8192 \
  [--generate 1024] --layers even8 --device cuda \
  --out algorithm_scientist/research/traces/<slug>.pt
$RUN algorithm_scientist/research/analyze_attention.py \
  --trace algorithm_scientist/research/traces/<slug>.pt   # CPU, no GPU needed
```
Note sink/local mass, effective context, and which heads are retrieval-y — this
tells you whether a sink+window flow suffices or query-aware retrieval is needed.

## 3 — Hypothesize + pre-register

Turn the survey + profile into 3–5 concrete hypotheses, each naming the vortex
op/behaviour exploited. Append a `journal.md` row per hypothesis (stage `idea`,
with the prediction). Flag which are offline-recall-testable.

## 4 — Offline recall screen (cheap, no engine)

For each recall-testable hypothesis, write a scoring method in
`algorithm_scientist/research/methods/<name>.py` (contract in `methods/_base.py`)
and screen it against the baselines (centroid/quest/h2o/streaming/random):
```bash
$RUN algorithm_scientist/research/eval_recall.py \
  --trace algorithm_scientist/research/traces/<modelslug>.pt \
  --method algorithm_scientist/research/methods/<name>.py
```
Promote only methods that beat the relevant baseline on **mass/recall** at a
sensible budget with acceptable **out_err**. Kill the rest in the journal (stage
`offline`, verdict). If a promising idea needs a new vortex op/kernel, call
**`/add-ops`**.

## 5 — Build + RULER-gate the survivors

Turn each promoted scoring method into a submission (`/new-submission`; the
indexer scoring mirrors your offline method), preflight, and RULER-gate (≥0.85).

## 6 — End-to-end + iterate

First predict efficiency: `python algorithm_scientist/research/efficiency.py
--config <variant> --seq-len <ctx>` → the speedup **ceiling** vs full attention
(don't bother running a variant whose ceiling is ~1×). Then run the survivors on
the task via `/iterate --model <model> --task <task> --max-iterations N` (or
`/batch-benchmark` in 4s; agent decides `TIMEOUT_MIN`). If measured throughput is
far below the ceiling → `/add-ops investigate`. Check standing via `/leaderboard
--task <task>`. Update the journal (stage `e2e`) and `memory.md §2/§5`.

## 7 — Synthesize (and survive the critic)

Before claiming anything, run the **`vortex-research-critic`** subagent on each
surviving result — it checks sample size, baselines, calibration-vs-task match,
SWA/leakage pitfalls, and ceiling-vs-measured overclaiming, and returns the
honest, narrower claim. Then summarize: which hypotheses survived offline→e2e→
critic, the Pareto-best variant vs full attention, and the design insight. Offer
`/write-paper` to compile it.

## Output

`question | survey brief | attention profile takeaway | hypotheses (n) |
offline survivors | e2e Pareto-best (mean@16, tput) | journal rows added`.
