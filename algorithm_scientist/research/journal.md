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
| H002 | 2026-07-28 | `approxTopK(tolerate_ratio)` on trtllm — new bf16 2-pass radix leaf; loosening the single-pass gate trades top-k recall for radix rounds | tol↑ ⇒ fewer rounds ⇒ higher throughput, some quality loss | ruler16k (Qwen3-4B, prompt-last-token) | done | see notes | — | 0.690–0.727 (flat) | 5777–6104 tok/s (flat) | **kill as a throughput knob; use tol=0.0 (exact, free)** |
| H002b | 2026-07-29 | same, second model family (Llama-3.1-8B-Instruct) — is the tol non-lever model-specific? | if H002 holds, RULER flat across tol | ruler16k (Llama-3.1-8B) | done | mass-rec 24.6% (vs Qwen 11.4%) | — | — | — | **confirms H002; tol=1.0 is a CLIFF 98%→0%** |

## Notes / dead ends

- _Record negative results here too — a method that loses on recall offline is a
  cheap kill before any GPU run, and worth remembering._

### H002 — `approxTopK` tolerate_ratio (2026-07-28/29)

**Setup.** Qwen3-4B, `block_sparse_attention` (page key-centroid · mean query),
block=page=32, topk=29, bos=1/eos=2, trtllm indexer + triton impl + tensor core.
Offline analysis `research/approx_topk_rounds.py` on a real RULER-16K trace
(4 samples × 8 layers × 8 kv-heads = 256 selection events).

**Result — tol is not a throughput knob and not a quality knob (until 1.0).**

| tol | E[rounds] | recall@k | mass-recall (BOS/EOS excl.) | RULER-16K | AIME24 mean@16 |
|-----|-----------|----------|------------------------------|-----------|----------------|
| 0.0 | 1.980 | 1.0000 | 11.40% | 98.1 (9-flow mean) | 0.7271 |
| 0.15| 1.914 | 0.9946 | 11.45% | 98.3 | 0.7229 |
| 0.45| 1.766 | 0.9623 | 11.40% | 98.2 | 0.7021 |
| 0.75| 1.621 | 0.8917 | 11.31% | 98.1 | 0.7167 |
| 0.95| 1.492 | 0.8167 | 11.77% | 97.0 | 0.7083 |
| 1.0 | 1.000 | 0.5326 | 9.65%  | — | — |

**Why nothing moved.**
1. Attention mass is **sink-dominated**: the always-kept BOS/EOS pages carry
   0.671 of *all* softmax mass — 94.7% of the 0.709 that exact selection achieves.
   The 29 scored pages contribute only ~3.7 pp.
2. The gate rarely fires: only ~26% of the budget are pass-1 strict winners
   (`floor`=0.2554), so E[rounds] falls just 1.98→1.49 even at tol=0.95. A ≤25%
   cut on a small kernel ⇒ 0% measured AIME24 throughput change (r=+0.33, p=0.36).
3. Losing 14–18% of top-k blocks costs ~0% mass — the swapped blocks are
   score-ties, and mass is exchangeable among ties. Only tol=1.0 breaks (−15.4%).

**Methodological trap found.** Modelling the threshold-bin fill in ascending page
index makes the approximation look like it *gains* mass (+1…+5.6%) because early
pages sit nearer the BOS sink. Descending inflates it to +12.6%; random collapses
it to ~0. Always randomise the tie-fill (`--fill-order random`, multi-seed).

**Also.** p-coverage is a weak proxy for NIAH correctness — the needle lives in
the 3.7 pp tail, so a flow can lose ~no mass and still answer wrong.

**Real headroom.** Exact top-k captures only **11.4%** of the mass available on
candidate pages. The scoring rule, not the top-k approximation, is the lever:
per-head queries instead of one mean query, min/max envelopes (QUEST), or more
reserved recency pages.

### H002b — same sweep on Llama-3.1-8B-Instruct (2026-07-29)

Second model family, identical setup (`block_sparse_attention`, trtllm, block=page=32,
topk=29, bos=1/eos=2). Model lives on a docker volume (see the docker-volume note);
trace = `/models/traces/llama31_8b_ruler16k.pt`, 4 samples × 8 layers × 8 kv-heads.
mass-recall = mass on the 29 scored pages / mass available on candidate pages
(BOS/EOS excluded), random tie-fill, mean±sd over 3 seeds.

| tol | mass-recall % | sd | recall@k | E[rounds] | RULER 16K |
|-----|---------------|-----|----------|-----------|-----------|
| 0.0 | 24.58 | 0.00 | 1.0000 | 1.969 | 98.0% |
| 0.05| 24.58 | 0.00 | 0.9997 | 1.961 | 98.0% |
| 0.15| 24.58 | 0.01 | 0.9947 | 1.906 | 98.0% |
| 0.25| 24.57 | 0.00 | 0.9869 | 1.855 | 98.0% |
| 0.35| 24.74 | 0.14 | 0.9716 | 1.797 | 98.0% |
| 0.45| 24.81 | 0.39 | 0.9586 | 1.738 | 98.0% |
| 0.55| 24.02 | 0.29 | 0.9428 | 1.695 | 98.0% |
| 0.65| 23.91 | 0.44 | 0.9071 | 1.617 | 98.0% |
| 0.75| 23.90 | 0.35 | 0.8778 | 1.555 | 98.0% |
| 0.85| 23.19 | 0.17 | 0.8256 | 1.469 | 97.0% |
| 0.95| 22.90 | 0.21 | 0.7701 | 1.391 | 98.0% |
| 1.0 | 20.24 | 0.26 | 0.4627 | 1.000 | **0.0%** |

**Confirms H002 across model families**, with two new facts:

1. **Llama's scoring rule is 2x better than Qwen's.** Exact top-k mass-recall
   24.6% (Llama) vs 11.4% (Qwen3-4B) on candidate pages; total exact p-coverage
   0.841 vs 0.709. Sinks still dominate (BOS/EOS = 0.790 = 93.8% of coverage).
2. **Llama's mass DOES decay monotonically with tol** (24.6 -> 22.9 by tol .95,
   beyond seed noise), unlike Qwen where it was flat. Yet RULER holds at 98%
   throughout — mass loss of that size is not what NIAH is sensitive to.

**tol=1.0 is a cliff, not a slope: 98% -> 0%.** Not a kernel bug. `--dump` shows
the model still locates the needle page but misreads it:
expected `1ca35cfb…` got `1ca3d5c6…`; expected `fa5d3100-11b7-4948-90e6-…` got
`fa5d3110-11b7-490e-90e6-…`. RULER scores by substring match, so a near-miss on a
36-char UUID is a 0. Never ship tol=1.0; the gate then never refines and recall
collapses to 0.46.
