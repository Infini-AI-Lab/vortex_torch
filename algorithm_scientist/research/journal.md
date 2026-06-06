# Research journal — sparse-attention discovery

The spine of the research record: pre-register each hypothesis, its prediction,
the cheap offline-recall screen, and the end-to-end result, then a verdict. Read
it at the start of a research session; append a row the moment you form a
hypothesis and again when each stage lands. Richer than `memory.md §3` (which
holds the running batch state) — this is the *why* and the *evidence trail*.

## Legend

- **stage**: `idea` → `offline` (recall screen) → `e2e` (RULER/AIME) → `done`.
- **head_rec@b / worst@b**: per-head top-k token recall@budget b from
  `eval_recall.py` — mean over heads and the worst head — vs. the included
  baselines (centroid/quest/quest_hw/h2o/streaming/random).
- **calib**: which dataset the trace was captured on (attention is
  workload-dependent — record it).
- **verdict**: `promote` / `kill` / `iterate`, one-line reason.

## Hypotheses

| id | date | hypothesis (op/behaviour exploited) | predicted | calib | stage | head_rec@.25 | worst@.25 | mean@16 | tput | verdict |
|----|------|-------------------------------------|-----------|-------|-------|--------------|-----------|---------|------|---------|
| H001 | 2026-06-02 | _example: dual-band centroid (mean+max) beats mean-only centroid on retrieval heads_ | head_rec↑ esp. worst-head vs centroid | aime24+gen | idea | — | — | — | — | — |

## Notes / dead ends

- _Record negative results here too — a method that loses on recall offline is a
  cheap kill before any GPU run, and worth remembering._
