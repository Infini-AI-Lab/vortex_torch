---
description: Survey the literature/web for sparse-attention ideas on a topic and write a brief that seeds /innovate — methods, key tricks, citations, and vortex-mappable hypotheses.
argument-hint: <topic or question> [--deep]
---

Do what a human researcher does first: **survey prior art** before designing.
Gather ideas from the web + the bundled `papers/`, distil the mechanisms, and
write a brief that feeds `/innovate` / `/reproduce-paper`.

## Step 1 — gather

- **Web:** use `WebSearch` / `WebFetch` for the topic (recent sparse-/efficient-
  attention, KV-cache retrieval, page/block selection, etc.). With `--deep`,
  invoke the **`deep-research`** skill for a fan-out, fact-checked sweep.
- **Local:** scan [papers/guide.md](../../papers/guide.md) (§14 catalog, §16
  off-catalog prompts) and any matching `papers/<name>` for what's already known.

## Step 2 — distil

For each relevant method capture: the **sparsity mechanism** (what each query
attends to), the **scoring** (how blocks/pages are ranked), its claimed
win/cost, and — critically — **how it maps onto vortex ops** (`forward_indexer`
scoring + `create_cache`/`forward_cache` summaries; which op exists, which would
need `/add-ops`). Prefer primary sources; note anything you couldn't verify.

## Step 3 — write the brief

Write `algorithm_scientist/research/briefs/<topic-slug>__<YYYY-MM-DD>.md`:

```
# Survey: <topic>            (date, sources)
## Methods            — table: method | mechanism | scoring | win/cost | cite
## Vortex mapping     — method → vortex ops / cache fields / missing op
## Hypotheses to try  — 3–5 one-liners, each naming the op/behaviour exploited,
                        flagged novel (§16/op-set) vs catalog, and whether an
                        offline-recall screen (eval_recall.py) can pre-test it
## Open questions / unverified claims
```

## Step 4 — hand off

Point the user (or the calling loop) at the brief. The hypotheses become
`/innovate` variants or `journal.md` rows; recall-testable ones should go through
the offline harness ([AI/workflows/research_toolkit.md](../../AI/workflows/research_toolkit.md))
before any GPU. Do not benchmark here — this is the survey step only.
