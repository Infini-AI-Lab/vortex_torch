# Channel-group importance for block-sparse routing (Qwen3-8B)

Replication of the Qwen3-4B channel study (`CHANNEL_STUDY_REPORT.md`) on
**Qwen/Qwen3-8B**. Same question: how is routing-relevant information distributed
across the `head_dim=128` channels (8 groups of 16), and how many / which can be
masked, for three routing families.

**Setup.** Qwen3-8B (GQA: 32 q-heads, 8 kv-heads, head_dim 128 — *identical head
geometry to 4B*, 36 layers, hidden 4096), `attention_backend=flashinfer`, vortex
planner flashinfer, block=page=16, greedy, **RULER** 100 examples, tight
**topk=8** (to discriminate channels), **CUDA graph ON**, N=100. Mechanism =
per-channel query mask (see 4B report §1); same `examples/misc/channel_study_flows.py`.

Full baselines: **block 95%, gqa_block 94%, quest 99%** (quest is a near-perfect
retriever on 8B, vs 86% on 4B).

## 1. Per-group importance (leave-one-out, all 3 families)

Mask exactly one group (keep the other 7):

| masked | block | gqa_block | quest |
|---|--:|--:|--:|
| g0 | 95 | 94 | 99 |
| g1 | 95 | 95 | 100 |
| g2 | 95 | 95 | 99 |
| **g3** | **91** | 93 | **30** |
| g4 | 95 | 94 | 98 |
| g5 | 95 | 92 | 99 |
| g6 | 94 | 94 | 99 |
| **g7** | **93** | 92 | **24** |
| (full) | 95 | 94 | 99 |

**g3 and g7 are the critical groups — the same as Qwen3-4B.** Concentration again
varies by method: **quest extreme** (g3 −69, g7 −75; masking noise groups is
harmless / +1), **block** mild (g3 −4, g7 −2), **gqa_block** nearly **flat**
(max single drop −2 — flatter than 4B's −9, i.e. more channel-redundant at 8B).

## 2. How many groups can be masked? (cumulative, gqa_block, full=94)

| keep | 6 | 5 | 4 | 3 | 2 | 1 |
|---|--:|--:|--:|--:|--:|--:|
| RULER | 95 | 94 | **94** | 92 | 90 | 80 |

**4 of 8 groups maskable losslessly** (keep {1,3,5,7}); graceful below that
(keep-1=g7 still 80%). Slightly more robust than 4B (4B keep-3=90/keep-1=75).

## 3. Cross-module: keep important half vs wrong half

| keep-set | block | gqa_block | quest |
|---|--:|--:|--:|
| full (8) | 95 | 94 | 99 |
| **odd {1,3,5,7}** (keep g3,g7) | 92 | 94 | **100** |
| **even {0,2,4,6}** (mask g3,g7) | **4** | **4** | **0** |

Same dramatic asymmetry as 4B and identical across methods: keep the important
half → lossless (quest *improves* to 100); mask it → collapse to ~random.

## 4. Keep-4 subset ablation — which combination (gqa_block)

| keep-4 | g3 | g7 | RULER |
|---|:--:|:--:|--:|
| odd {1,3,5,7} | ✓ | ✓ | 94 |
| {0,2,3,7} | ✓ | ✓ | 91 |
| hi4 {4,5,6,7} | ✗ | ✓ | 86 |
| lo4 {0,1,2,3} | ✓ | ✗ | 78 |
| even {0,2,4,6} | ✗ | ✗ | 4 |
| {0,1,4,5} | ✗ | ✗ | 0 |

Same hierarchy as 4B: **both g3,g7 → full (~94, robust to the other two); g7-only
→ 86; g3-only → 78; neither → collapse.** Contiguous halves cross-module:

| | lo4 {0,1,2,3} | hi4 {4,5,6,7} |
|---|--:|--:|
| block | 81 | 90 |
| gqa_block | 78 | 86 |
| quest | 65 | 67 |

hi4 > lo4 everywhere (g7-half beats g3-half). quest's halves (65/67) are far
higher than 4B's (27/43) — 8B quest is much more robust — but still well below its
99% full and its 100% interleaved (odd), so importance-aware selection still wins.

## 5. Masking BOTH critical groups (g3 & g7), keeping the other six

| family | full | mask {g3,g7} (keep 6) | drop |
|---|--:|--:|--:|
| block | 95 | **3** | −92 |
| gqa_block | 94 | **4** | −90 |
| quest | 99 | **0** | −99 |

All three collapse with 6/8 groups kept — *which* groups matter, not how many.
This also explains the flat single-group leave-one-out for block/gqa (§1):
g3,g7 are **redundant** (either alone suffices, §4 lo4/hi4), but masking both
removes the signal entirely.

## 6. Control: mask 2 RANDOM non-critical groups (g3,g7 kept), 8 pairs

| family | random-pair masks (8) mean | full | mask {g3,g7} |
|---|--:|--:|--:|
| block | 94.5 (93–96) | 95 | 3 |
| gqa_block | 93.0 (92–94) | 94 | 4 |
| quest | **99.5** (98–100) | 99 | 0 |

Masking any 2 non-critical groups is harmless (block/gqa within ~1 of full) or
beneficial (**quest +0.5, every pair ≥98**), across all 8 pairs × 3 families —
opposite of masking {g3,g7}. Confirms g3,g7 are **specifically** critical, not
"any two groups."

## 7. Cross-model comparison: Qwen3-4B vs Qwen3-8B

| finding | Qwen3-4B | Qwen3-8B |
|---|---|---|
| critical groups | **g3, g7** | **g3, g7** (same) |
| keep-4 (odd) lossless? | yes (94 vs 94) | yes (94 vs 94) |
| mask {g3,g7} → collapse? | yes (0–5) | yes (0–4) |
| random non-crit pair benign? | yes | yes |
| quest full RULER | 86% | **99%** |
| quest g3/g7 dominance | −55 / −76 | −69 / −75 |
| gqa_block per-group peak | g7 −9 (peaky) | −2 (flat, more redundant) |
| quest noise-masking helps | yes (odd 90>86) | yes (odd 100>99) |

**Main cross-model conclusion.** The critical routing channels (g3, g7) and all
structural conclusions (half maskable, keep ≥1 of {g3,g7}, mask-both collapses,
non-critical groups redundant/noise) are **identical across Qwen3-4B and
Qwen3-8B** — strong evidence this is a **stable Qwen3-family property of the K/Q
channel geometry**, not size-specific. Differences are quantitative: the larger
model is a stronger retriever (quest 86→99) and more channel-redundant
(gqa_block leave-one-out flattens), but the *location* of the routing signal is
unchanged.

## 8. Notes

- **CUDA graph ON** for every run (the channel flows are graph-compatible; see
  4B report §6). N=100, topk=8, block=16, greedy.
- All variants are the same `examples/misc/channel_study_flows.py` used for 4B
  (model-agnostic); only `MODEL=Qwen/Qwen3-8B` differs.
- **AIME24** reasoning validation (full vs keep-odd) remains the open follow-up,
  as for 4B.

## 9. Files
- Flows: `examples/misc/channel_study_flows.py`; driver: `marks/mla/test_config_refactor.py`.
- Raw logs: `marks/mla/q8b_loo_*/`, `marks/mla/q8b_p2a_*/`, `marks/mla/q8b_p2b_*/`.
