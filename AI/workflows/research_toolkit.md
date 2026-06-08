# The research toolkit — discover algorithms like a human

Tools that let an agent research sparse attention the way a person does: read the
literature, look at real attention, form hypotheses, screen them cheaply, and
spend GPU only on survivors. Orchestrated by `/research`; each piece is usable
alone. State lives in `algorithm_scientist/research/`.

## 1. Survey the literature — `/research-ideas <topic> [--deep]`

Web (`WebSearch`/`WebFetch`, or the `deep-research` skill with `--deep`) + the
bundled `papers/` → a brief in `research/briefs/` (methods, scoring, vortex
mapping, hypotheses). The survey-first step `/innovate` and `/reproduce-paper`
can call.

## 2. Look at real attention — `analyze_attention.py`

Profile a captured trace: per-layer sink mass, locality, effective context
(`exp(entropy)`), top-p block coverage, and which heads are retrieval-y. Tells
you *where to be sparse* before designing a flow.

## 3. Capture intermediate tensors — `capture_trace.py`

> **Reminder — saving tensors is often something YOU set up.** `capture_trace.py`
> is a working **reference** for HuggingFace models on the unified attention
> interface (and it runs at long context on GPU). But studying a micro-benchmark
> may require you to instrument the source of truth yourself: add hooks to **HF**
> (a custom layer/attention, a model not on the unified interface) or to
> **sglang/vortex** (dump the real page summaries from `forward_indexer`/
> `forward_cache` or the attention backend — set `"disable_cuda_graph": true`
> first, since cudagraph freezes the forward and skips Python-side saves). Extend
> or replace the capture for what your hypothesis actually needs; don't assume one
> script covers every model and code path.


Grabs, via HuggingFace's unified attention interface (no CUDA graph — tensor
saving just works), each chosen layer's **q**, full-context **K/V**, softmax
scaling, and accumulated recent-query attention (for the H2O baseline).
Ground-truth attention is recomputed exactly from (q, K) downstream.

**You choose `--data` — it matters.** Attention is workload-dependent, so the
trace must match what you're optimizing for or the recall screen mispredicts:
- a **task** (e.g. aime24) → capture on that task's prompts (`--data
  examples/math/aime24.jsonl`, or the `make_task.py` jsonl for a non-default model)
  and `--generate <N>` so the captured query is a real **mid-generation** decode
  step (reasoning tasks emit long outputs; the throughput-relevant attention is
  late in generation, not the prompt's last token);
- **long-context retrieval heads** → `--data examples/ruler/validation_4k.jsonl`
  (RULER/NIAH), large `--max-ctx`.
The trace records `calibration_data`/`generate` so results stay interpretable.

Run it on a **detected free GPU** (the free set is dynamic/shared — never
hardcode a device; with several free, run captures/RULER in parallel one per GPU):
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; exit 1; }
CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} python algorithm_scientist/research/capture_trace.py \
  --model <hf-id> --data <dataset matching your workload> --num-samples 3 \
  --max-ctx 8192 [--generate 1024] --layers even8 --device cuda \
  --out algorithm_scientist/research/traces/<slug>.pt
```

> **Vortex/sglang capture (optional, more faithful).** To capture the *real page
> summaries* the indexer computes instead of HF tensors, run the flow with
> `"disable_cuda_graph": true` (cudagraph freezes the forward and skips
> Python-side dumps) and insert a dump in `forward_indexer`/`forward_cache` or
> the attention backend. Heavier and coupled to a live flow — use it to validate
> an HF-screened idea, not for first-pass iteration.

## 4. Screen retrieval/recall offline — `eval_recall.py` (seconds, no engine)

The core question: **under a token budget, how well does an algorithm identify
the real top-k attended tokens — in each attention head?** Headline metric is
**head_rec** (mean per-head token recall@budget) and **worst** (the worst head,
which gates quality); also reported: mass captured, block-mass recall vs ideal,
and the **output proxy** (`‖attn_sparse − attn_full‖` rel-L2 + `within%` ≤ tol).

**This is a framework — agents test whatever algorithm they want.** Write a
method file (contract in `methods/_base.py`) and screen it; the included
`methods/` are just **reference baselines** (centroid = vortex block_sparse,
quest min-max, quest_hw = per-head, h2o, streaming, random) — not a library to
complete. A method exposes either `block_scores(ctx) -> [nb]` (one shared page
set, like vortex) or `block_scores_headwise(ctx) -> [Hkv, nb]` (per-head
selection, its ceiling). Pooled stats (`q_bar`, `kmean/kmin/kmax`, per-head
`kmean_h/kmin_h/kmax_h`, `accum`) are precomputed so a method is a one-liner.

```bash
python algorithm_scientist/research/eval_recall.py \
  --trace algorithm_scientist/research/traces/<slug>.pt \
  --method algorithm_scientist/research/methods/<your_idea>.py
```
An idea that loses on per-head recall offline is a **cheap kill** before any GPU
run; one that wins becomes the indexer scoring in a real submission.

## 5. Record it — `research/journal.md`

Pre-register hypothesis → prediction → offline recall → e2e result → verdict.
The evidence trail; append a row when you form a hypothesis and at each stage.

## How it fits the loops

`/research` runs 1→5 then hands survivors to `/iterate`. `/innovate` may survey
(step 1) for novelty and recall-screen (step 4) before committing a variant.
A new op an idea needs → `/add-ops`. The Pareto-best result → `/write-paper`.
