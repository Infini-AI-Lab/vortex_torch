# Channel-group importance for block-sparse routing (Qwen3-4B)

**Question.** In sparse-attention *routing* (block-top-k page selection), how is the
routing-relevant information distributed across the head-dimension channels, and
**how many channels can be masked** before routing degrades? We group the
`head_dim = 128` into **8 groups of 16 channels** and study which groups matter,
for three routing families.

**Model / setup.** `Qwen/Qwen3-4B` (GQA: 32 q-heads, 8 kv-heads, head_dim 128),
`attention_backend=flashinfer`, vortex planner `flashinfer`, `block=page=16`,
greedy (temp 0), **RULER** `examples/ruler/validation.jsonl` (100 examples, substring
match). To expose channel effects we route at a **tight budget `topk=8`** (128
attended tokens) — at a loose budget every variant saturates and the channels
are indistinguishable. All results below are with **CUDA graph ON** (the full
22-config set was re-run graph-ON, §8); the channel flows are graph-compatible
(§6) and graph-vs-eager makes no accuracy difference.

## 1. Mechanism

All three routing scores are a **sum of per-channel contributions** over the
head dim:

| family | page score |
|---|---|
| `block_sparse` | `⟨q̄, c_p⟩` |
| `gqa_block_sparse` | `softmax_p(⟨q_h, c_p⟩)` then `max_h` |
| `gqa_quest_sparse` | `Σ_d max(q_{h,d}·M_{p,d}, q_{h,d}·m_{p,d})` then `max_h` |

So masking a channel group = **zeroing those query channels** removes their term
from the sum, uniformly across families. We realise it with a per-channel query
mask: a sum of disjoint per-group `MaskSlice`s (1.0 on active groups, 0.0
elsewhere) multiplied into `q` before scoring. Implemented in
`examples/misc/channel_study_flows.py` (compiles + passes `check_engine_config`
preflight for all variants); driven via `marks/mla/test_config_refactor.py`.

## 2. E1 — per-group importance (leave-one-out, gqa_block)

Mask exactly one group (route on the other 7). Drop vs the full-8 baseline (94%):

| masked group | RULER | Δ |
|---|---|--:|
| g7 | 85% | **−9** |
| g3 | 89% | −5 |
| g5 | 92% | −2 |
| g1 | 93% | −1 |
| g0 / g4 / g6 | 94% | 0 |
| g2 | 95% | +1 |

Importance is **highly skewed**: **g7 and g3 dominate**; g0/g2/g4/g6 are
individually free to drop. Removal order (least→most important):
`g2, g0, g4, g6, g1, g5, g3, g7`.

## 3. E2 — how many groups can be masked? (cumulative, gqa_block)

Remove groups in the E1 order, keep the rest:

| keep | masked | RULER |
|--:|---|--:|
| 8 | — | 94% |
| 7 | g2 | 95% |
| 6 | +g0 | 94% |
| 5 | +g4 | 95% |
| **4** | +g6 → keep {1,3,5,7} | **94%** |
| 3 | +g1 → keep {3,5,7} | 90% |
| 2 | +g5 → keep {3,7} | 90% |
| 1 | keep {7} | 75% |

**You can mask 4 of 8 groups (drop half the channels, 64 dims) with zero loss**,
provided the important groups are kept. The cliff is at **keep-3** (−4 pts);
keep-1 (g7 alone) still gets 75% (≫ random), confirming g7 is the single most
informative group.

## 4. E3 — does channel importance generalise across routing methods?

Compare keeping the **odd** groups `{1,3,5,7}` (the important half) vs the
**even** groups `{0,2,4,6}` (which masks the critical g3 & g7), per family:

| keep-set | block_sparse | gqa_block | gqa_quest |
|---|--:|--:|--:|
| full (8) | 95% | 94% | 86% |
| **odd {1,3,5,7}** | 93% | 94% | **90%** |
| **even {0,2,4,6}** | **5%** | **5%** | **0%** |

The asymmetry is **dramatic and identical across all three methods**: keeping the
important half preserves (or for quest *improves*: 86→90, the noisy even channels
were loosening its upper bound) accuracy; masking the important half **collapses
routing to ≈random (0–5%)**.

**Main finding.** Routing-relevant information lives in a *specific, sparse
subset* of channel groups (here g3 & g7 are critical, the even groups are
near-redundant), and **this subset is shared across block / gqa-block / quest** —
i.e. it is a property of the model's K/Q channel geometry, not of the routing
algorithm. Consequences: (a) routing can use **half the channels losslessly**
(a cheaper indexer); (b) a fixed *good* channel subset transfers across routing
methods; (c) choosing the *wrong* channels is catastrophic — so any channel
sparsification must be importance-aware.

## 5. AIME24 phase (filtered — ready, pending a usable GPU)

RULER filters the space sharply: **even-half is eliminated** (collapse); the
survivor to validate on reasoning is **keep-odd (4 groups) vs full (8)** for each
family. The harness is ready (`examples/math/verify_algo.py` now takes
`--vortex-module-path` + `--disable-cuda-graph`; `examples/math/aime24.jsonl` is
Qwen-formatted with thinking enabled). Pending command (one per family, mean@16):

```bash
# full baseline (built-in) — repeat with vortex_module_name in
#   {block_sparse_attention, gqa_block_sparse_attention, gqa_quest_sparse_attention}
python examples/math/verify_algo.py --trials 16 --model-name Qwen/Qwen3-4B \
  --data-path examples/math/aime24.jsonl --page-size 16 --block-size 16 \
  --workload-chunk-size 64 --topk-val 32 --topk-ratio 0 --mem 0.9 \
  --generation-max-new-tokens 32768 --attention-backend flashinfer \
  --vortex-attention-backend flashinfer --vortex-impl-backend triton \
  --vortex-layers-skip --disable-cuda-graph --summary-dir summary-chanstudy-aime24 \
  --vortex-module-name gqa_block_sparse_attention
# keep-odd — add: --vortex-module-name chan_<fam>_odd \
#                  --vortex-module-path examples/misc/channel_study_flows.py
```

> Not yet run: at report time only GPU 0 was free, which is excluded (hardware
> bug); GPUs 1–7 were held by other users. Launch the 6 configs in waves on the
> next free non-0 GPUs.

## 6. Notes / caveats

- **CUDA-graph capture (corrected).** The channel-masked flows are **cuda-graph
  compatible** — `--disable-cuda-graph` is NOT required. Confirmed: (a) the base
  block/gqa/quest modules ran graph-ON at 100% RULER (config-refactor matrix);
  (b) the generated channel `CompiledFunc` is structurally identical to the base
  one — all intermediates preallocated in `__init__`, written into fixed buffers
  in `forward`, with zero runtime allocation/host-sync (the capture-safe
  pattern). The eager runs here used `--disable-cuda-graph` only because two
  early runs were misread: one hit `CUDA device busy` (host contention) and one
  was killed by a `pkill` mid-capture; no genuine capture error was ever logged.
  Eager vs graph give identical accuracy, so these RULER numbers stand; future
  runs should use graph-ON (faster), GPU permitting.
  **Empirically confirmed (2026-05-28):** `chan_gqablock_odd` run with
  `disable_cuda_graph=False` captured to `Capturing batches … 100%` and returned
  20/20 — graph-ON works for the channel flows. (It took 199 retries across
  GPUs 0–5, each failing at `torch.cuda.set_device` with `device busy/
  unavailable` — a host device-acquisition fault, not capture — until GPU 7
  freed.) **Drop `--disable-cuda-graph`; it is not needed.**
- **topk choice.** `topk=8` is deliberately tight to *discriminate* channels; at
  loose budgets the channel effect is masked by saturation. The maskability
  fraction (½) may shift with budget — worth a budget×keep sweep.
- **Per-method importance.** The E1 ranking was measured on gqa_block; E3 shows
  the *odd/even* split transfers, but the exact per-group ranking for block /
  quest was not separately measured (a cheap follow-up).

## 7. Files

- `examples/misc/channel_study_flows.py` — channel-mask flows (3 families × single /
  subset / leave-one-out / cumulative variants), explicit idempotent `@register`.
- `marks/mla/test_config_refactor.py` — RULER driver (env: MODULE, MODULE_PATH,
  ATTN, VORTEX_ATTN, BLOCK, TOPK, N, DISABLE_CG).
- `examples/math/verify_algo.py` — AIME mean@k harness (+`--vortex-module-path`,
  `--disable-cuda-graph`).
- Raw logs under `marks/mla/chan{E1,LOO,KEEP,XMOD}_*/`.

---

# 8. Full rerun with CUDA graph ON (N=100) — verified reproduction

All 22 configs were re-run with **CUDA graph ON** (`disable_cuda_graph=False`),
N=100 (full `validation.jsonl`), Qwen3-4B, flashinfer, block=page=16, topk=8,
greedy. Every channel flow **captured cleanly** (`Capturing batches … 100%`) and
the numbers are **identical to the eager runs** — confirming graph-vs-eager makes
no accuracy difference and that `--disable-cuda-graph` is not needed (§6).

**gqa_block — leave-one-out (mask one group):**

| masked | full | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| RULER | 94 | 94 | 93 | 95 | 89 | 94 | 92 | 94 | 85 |

**gqa_block — cumulative keep (importance-ordered removal):**

| keep | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| RULER | 94 | 95 | 94 | 95 | **94** | 90 | 90 | 75 |

→ **4 of 8 groups maskable losslessly** (keep {1,3,5,7}); cliff at keep-3.

**Cross-module — keep important half vs wrong half:**

| keep-set | block | gqa_block | quest |
|---|--:|--:|--:|
| full (8) | 95 | 94 | 86 |
| odd {1,3,5,7} (important) | 93 | 94 | 90 |
| even {0,2,4,6} (masks g3,g7) | **5** | **5** | **0** |

Identical conclusion: the same channel groups (g3, g7 critical) matter across all
three routing methods; keep the important half → lossless, mask it → collapse.

*(Run: `marks/mla/chan_graphon_full_*/`, 22/22 configs, graph-ON, completed in 22
pool cycles across the GPUs that initialized; flaky GPUs 0–5 were re-queued.)*

---

# 9. Keep-4 subset ablation — *which* combination matters (graph-ON, N=100)

Beyond odd/even, we vary the kept 4-group subset and categorize by whether it
contains the two critical groups g3, g7 (gqa_block):

| keep-4 | g3 | g7 | RULER |
|---|:--:|:--:|--:|
| odd {1,3,5,7} | ✓ | ✓ | 94% |
| {0,2,3,7} | ✓ | ✓ | 93% |
| hi4 {4,5,6,7} | ✗ | ✓ | 86% |
| lo4 {0,1,2,3} | ✓ | ✗ | 79% |
| even {0,2,4,6} | ✗ | ✗ | 5% |
| {0,1,4,5} | ✗ | ✗ | 0% |

**Refined finding (a hierarchy, not a binary):** keep **both** g3+g7 → full quality
(~94%, and robust to *which* other two groups — two independent "both" subsets
give 94/93); **g7 alone** recovers most (86%); **g3 alone** is weaker (79%);
**neither** collapses (0–5%, two independent "neither" subsets agree). This
matches the leave-one-out ranking (g7 −9 > g3 −5). It is the *content* of the
kept set, not the partition shape, that determines accuracy.

**Cross-module — contiguous halves (mask {0,1,2,3} vs {4,5,6,7}):**

| keep-4 | block | gqa_block | quest |
|---|--:|--:|--:|
| lo4 {0,1,2,3} | 83% | 79% | 27% |
| hi4 {4,5,6,7} | 88% | 86% | 43% |

The hi4>lo4 ordering (g7-half beats g3-half) holds across all three methods.
**quest is far more channel-sensitive**: contiguous halves crater it (27/43%) even
though the *interleaved* important set works (odd = 90%, §8). I.e. quest's
min/max-envelope upper bound depends on keeping the right channels distributed,
not just any half — so importance-aware channel selection matters most for quest.

*(Runs: `marks/mla/chan_subset_*/`, graph-ON, N=100.)*

---

# 10. Native per-group importance on block & quest (graph-ON, N=100)

Leave-one-out measured *natively* for each family (complements §2, which was
gqa_block-only). Full baselines: block 95%, quest 86%.

| masked group | block_sparse | gqa_block (§2) | gqa_quest |
|---|--:|--:|--:|
| g0 | 93 | 94 | 87 |
| g1 | 93 | 93 | 89 |
| g2 | 94 | 95 | 84 |
| g3 | 94 | 89 | **31** |
| g4 | 95 | 94 | 86 |
| g5 | 94 | 92 | 92 |
| g6 | 95 | 94 | 91 |
| g7 | 92 | 85 | **10** |
| (full) | 95 | 94 | 86 |

**Same critical groups, very different concentration.** g3 & g7 are the important
groups in every method, but how sharply:

- **quest — extreme.** Masking g7 → 10%, g3 → 31% (the routing signal is almost
  entirely those two). Conversely masking *noise* groups **improves** quest
  (g5 +6, g6 +5, g1 +3): quest's `Σ_d max(q·M, q·m)` upper bound is loosened by
  noisy channels, so pruning them tightens it — this is why quest_odd (90) >
  quest_full (86) in §4.
- **gqa_block — moderate.** g7 −9, g3 −5; other groups free.
- **block_sparse — flat.** Max single-group drop is −3 (g7). The head-averaged
  query `q̄` smooths per-channel peaks, so no single group is decisive — yet
  losing the whole *wrong* half still collapses it (block_even = 5%, §4), i.e.
  importance is diffuse but still concentrated in the g3/g7-bearing half.

**Takeaway.** Which channels carry routing signal (g3, g7) is a model property
shared across routing methods; the *sensitivity* to losing them is set by how the
method aggregates channels — peaky for an upper-bound method (quest), diffuse for
a head-averaged dot (block), in between for per-head softmax-max (gqa_block).
Practical corollary: channel pruning is safest/most beneficial for **quest**
(prune noise → faster *and* more accurate), must be importance-aware for
**gqa_block**, and is least impactful per-group for **block**.

*(Runs: `marks/mla/chan_loo_bq_*/`, graph-ON, N=100.)*

---

# 11. Masking BOTH critical groups (g3 & g7), keeping the other six

Keep {0,1,2,4,5,6}, mask only g3 and g7 (graph-ON, N=100, topk=8):

| family | full | mask {g3,g7} (keep 6) | drop |
|---|--:|--:|--:|
| block_sparse | 95 | **2** | −93 |
| gqa_block | 94 | **5** | −89 |
| quest | 86 | **0** | −86 |

**All three collapse to ≈random even with 6 of 8 groups kept.** The decisive
factor is *which* groups survive, not how many: keeping 6 *wrong* groups is
worthless, while keeping 4 *right* groups (odd ⊇ {g3,g7}) holds 94% (§3, §9).

This also resolves the §10 puzzle for **block_sparse**, whose single-group
leave-one-out looked flat (mask g3 −1, mask g7 −3): g3 and g7 are **redundant**
there — either alone carries the routing signal, so masking one leaves the other
(hence block_lo4=83 with g3-only and block_hi4=88 with g7-only, §9), but masking
**both** removes the signal entirely (−93). quest has the least redundancy (g7
alone −76, g3 alone −55 in §10), gqa_block is intermediate.

**Unifying rule (all three methods):** routing requires **at least one of
{g3, g7}**; masking both is catastrophic regardless of the other six groups. The
g3/g7-bearing channels are a sparse, model-level routing signal; everything else
is largely redundant (block/gqa) or even noise (quest, §10).

*(Runs: `marks/mla/chan_no37_*/`, graph-ON, N=100.)*

---

# 12. Control: mask 2 RANDOM non-critical groups (g3,g7 kept)

To check that §11's collapse is specific to {g3,g7} and not a generic
"mask-any-2-groups" effect, we masked 8 random pairs drawn from the non-critical
groups {0,1,2,4,5,6} (g3,g7 always kept; seed 0), each on all 3 families
(graph-ON, N=100, topk=8). Pairs: (0,1)(0,6)(1,4)(1,5)(1,6)(4,5)(4,6)(5,6).

| pair masked | block | gqa_block | quest |
|---|--:|--:|--:|
| {0,1} | 93 | 93 | 88 |
| {0,6} | 95 | 95 | 91 |
| {1,4} | 94 | 95 | 90 |
| {1,5} | 92 | 90 | 94 |
| {1,6} | 94 | 95 | 93 |
| {4,5} | 94 | 90 | 86 |
| {4,6} | 94 | 94 | 91 |
| {5,6} | 94 | 95 | 91 |
| **mean (full)** | **93.8 (95)** | **93.4 (94)** | **90.5 (86)** |
| §11 mask {g3,g7} | 2 | 5 | 0 |

**Masking any 2 non-critical groups is harmless to beneficial** — block/gqa lose
≤1.5 on average (within noise of full), and **quest gains +4.5** (every pair
removes some envelope-loosening noise, consistent with §10/§4). This is the exact
opposite of masking {g3,g7} (0–5% collapse, §11). The control confirms the
routing signal is concentrated **specifically** in g3 & g7, not in "any two
groups": the partition that matters is critical-vs-noise, and the non-critical
six are mutually redundant.

*(Runs: `marks/mla/chan_randpair_*/`, graph-ON, N=100, 24 configs.)*
