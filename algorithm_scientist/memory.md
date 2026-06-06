# Algorithm-scientist memory

Persistent working notebook for the vortex_torch sparse-attention
submission workflow. **Every agent run reads this file at start and
updates it before stopping.** It survives across sessions; the
conversation does not.

**Hard constraints** (restate every time you open this file):

- **Batch size is always 4 variants.** Parallelism = `min(N, 4)`
  where `N` = number of free GPUs detected by
  `algorithm_scientist/free_gpus.sh` (space-separated indices of
  GPUs with no compute process and memory.used < 1024 MiB).
  - `N >= 4` → all 4 variants run in parallel, one per GPU.
  - `0 < N < 4` → run the 4 variants in **waves of N**
    (sequential fallback). With `N = 1` this is fully serial;
    `N = 2` runs 2 + 2; `N = 3` runs 3 + 1.
  - `N == 0` → **hard wait**, do not launch.
  If you have only one core idea, fill the other 3 slots with
  orthogonal knob sweeps.
- **RULER pre-filter before AIME24.** Run
  `algorithm_scientist/run_ruler.py` on each variant
  sequentially on one free GPU before launching AIME24. Any
  variant scoring **< 0.85 accuracy** on
  `examples/ruler/validation.jsonl` has structurally broken attention
  — fix or replace it before launching AIME24.
- **One batch at a time** on the free GPUs. Concurrent batches
  contend for memory and OOM/thrash. The cadence is: detect free
  → run RULER filter → launch 4-variant batch → `wait` →
  analyse → re-detect → next batch.
- **At least one *genuinely novel* variant per batch** — aim for
  two when slots allow. "Genuinely novel" excludes both paper
  replications **and** combinations of two papers (combinations
  are catalog-adjacent — see `papers/guide.md` §16.1). Acceptable
  origins: `papers/guide.md` §16.2 (untried knobs), §16.3
  (inversions), §16.4 (first-principles), or — best — ideas
  derived from the framework's op set itself that don't fit any
  §16 sub-bucket. Defend each novelty in one sentence naming the
  specific framework op or behaviour exploited. Pre-register each
  novelty hypothesis in §3 the moment you launch.
- **Remaining slots use `papers/guide.md` §16.5 techniques.**
  The 2–3 non-novelty slots per batch should be filled with
  catalog-adjacent parameter sweeps (`vortex_topk_val`,
  `approxTopK`, layer-skip patterns, fp8/bf16 KV, etc.) that
  map the Pareto frontier around the novel variant. These are
  explicitly encouraged for non-novelty slots.
- **File layout.** Submissions live under `submissions/<tag>/`,
  where `<tag>` is your sanitized model name (e.g.
  `claude_opus_4_7`). Batched runs use
  `submissions/<tag>/batch_<x>_id<y>.{py,json}` (`<x>` = batch
  index, `<y>` = variant slot 0…3). Summaries land at
  `summary_submissions/<tag>/batch_<x>_id<y>/latest.json`; RULER
  results at
  `summary_ruler_submissions/<tag>/batch_<x>_id<y>/latest.json`.
- **Objective**: strike the best tradeoff between AIME24 `mean@16`
  and `throughput`. Both are objectives — there is no fixed quality
  floor. Pick winners by where they sit on the
  `(throughput, mean@16)` Pareto frontier in §5, not by clearing a
  fixed bar.

---

## §1. In-flight batches  (at most 1 — every targeted GPU is consumed)

> Append one row the moment you launch a batch; remove it once all
> 4 finished rows land in §2. A batch counts as "in-flight" from
> the first wave's launch until the last child writes its
> `latest.json`.

| tag | batch_x | launched_at | logdir | gpus | submissions | off-cat hyp § | status |
|---|---|---|---|---|---|---|---|
| _none_ |  |  |  |  |  |  |  |

**Model/task:** zai-org/GLM-4.7-Flash (MLA, glm4_moe_lite, vortex_glm env, HF_HOME=/raid/catalyst/models/) · task=examples/math/aime26_glm.jsonl · TP=1 MAX_GPUS=4.

> **STRATEGIC FINDING (after batch_2).** 3 batches / 6 novel routing rules — NONE
> pushes the (throughput, mean@16) frontier; it is owned entirely by plain
> `rope_aware` mean-centroid swept over topk. Reason: (1) the MLA absorbed query
> makes ⟨q̄, centroid⟩ already the true averaged decode logit, so the mean captures
> the signal; (2) every "smarter" rule adds cache fields → wider decode-time index
> read → lower throughput, faster than it adds recall. **Accuracy is capped by the
> topk budget, not the scoring rule, and throughput is capped by summary width.**
> Pivot (batch_3+): stop iterating the scoring RULE; attack the THROUGHPUT axis —
> make the routing summary NARROWER (channel/low-rank reduction of the centroid)
> to cut index bandwidth at equal recall. Also worth: a real-trace recall screen
> (capture_trace→eval_recall) to confirm before more AIME spend; and the deferred
> freq-band k_pe (needs indexer L2NormInterleave op) only if narrow-centroid shows
> k_pe is droppable.

---

## §2. Completed batches

> Oldest first. When a batch completes, copy headline metrics from
> `summary_submissions/<tag>/<stem>/latest.json` for each variant
> and distil 1-3 sentences of takeaway — what moved, what didn't,
> why. Always cite the off-catalog variant and what its result said
> about the §3 hypothesis it pre-registered.

### claude_opus_4_8/batch_0 — GLM-4.7-Flash MLA: routing-rule novelty vs budget  (2026-06-06, GPUs 0-3, task=aime26_glm)

| variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|---|
| batch_0_id0 | 4391f5e4d35d | 1.00 | **0.7125** | 0.90 | 3874 | 2202 | rope_aware mean-centroid, topk=61 — **accuracy anchor (Pareto)** |
| batch_0_id1 | fb42df9fd8fd | 1.00 | 0.70625 | 0.867 | 3405 | 2441 | Quest sub-block bound (NOVEL #1) — **dominated** by id0 |
| batch_0_id2 | 94322f0f73ae | 1.00 | 0.6458 | 0.867 | 3641 | 2389 | self-gated RoPE (NOVEL #2) — **dominated**, accuracy hit |
| batch_0_id3 | 8890c099b07b | 0.99 | 0.70 | 0.833 | **4586** | 1841 | rope_aware topk=40 — **throughput anchor (Pareto)** |

**Knob matrix:** routing rule held-vs-varied at topk=61 (id0 mean-centroid / id1 sub-block channel-extrema bound / id2 NoPE+SiLU(RoPE)); id3 = budget sweep (topk 61→40) on the id0 rule.

**Off-catalog hypotheses tested:** §3 #1 (sub-block Quest bound) → **REFUTED at topk=61**: equal recall to mean-centroid (0.706 vs 0.7125) but −12% throughput (extra CMin/CMax fields + 2nd GeMM). §3 #2 (self-gated RoPE) → **REFUTED**: SiLU-gating the RoPE term *lowered* mean@16 to 0.646 — the symmetric linear NoPE+RoPE add (rope_aware) carries signal the one-sided gate discards.

**What moved:** the only Pareto motion came from the **budget** knob — topk 61→40 bought +18% throughput (3874→4586) for −1.75% mean (0.7125→0.70). Neither novel routing rule beat the plain mean-centroid at topk=61.

**Takeaway:** at topk=61 the mean-centroid is recall-saturated on AIME26 — routing cleverness has no headroom there. The open question (→ batch_1): does a tighter bound (sub-block Quest) help in the **tight-budget** regime (topk≈32) where page selection actually bites? Frontier so far: **id0 (3874, 0.7125)** and **id3 (4586, 0.70)**.

### claude_opus_4_8/batch_1 — GLM MLA: routing rule at TIGHT budget (topk=32)  (2026-06-06, task=aime26_glm)

| variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|---|
| batch_1_id0 | 3a3cff64a220 | 0.99 | 0.6646 | 0.867 | 4075 | 2169 | rope_aware mean-centroid topk=32 (control) |
| batch_1_id1 | 6a97b59535b0 | 1.00 | 0.6833 | 0.90 | 3483 | 2366 | sub-block Quest bound (NOVEL §3#3) — beats control +0.019 |
| batch_1_id2 | 3742a47dab69 | 1.00 | **0.6875** | **0.933** | 3791 | 2259 | mean+Quest blend λ=0.5 (NOVEL §3#4) — **best rule** at topk32 |
| batch_1_id3 | 39aeff7e0d5f | 0.98 | 0.6729 | 0.867 | 4318 | 2101 | rope_aware topk=24 (cheap sweep) |

**Knob matrix:** routing rule varied at fixed topk=32 (id0 mean / id1 sub-block-Quest / id2 mean+Quest-blend); id3 = budget topk=24 on id0 rule.

**Off-catalog hypotheses tested:** §3 #3 (sub-block Quest tighter at tight budget) → **CONFIRMED**: 0.6833 > 0.6646 control (the bound that bought nothing at topk=61 helps at topk=32). §3 #4 (mean/Quest convex blend) → **CONFIRMED & best**: 0.6875 mean, 0.933 pass@16 — blending expectation + optimistic bound beats either alone when budget bites.

**What moved:** routing-rule ORDERING is now resolved — at a fixed tight budget, blend > Quest > mean-centroid. BUT all topk≤32 points are **Pareto-dominated** by batch_0 id3 (topk=40: 0.70@4586 beats every topk≤32 on both axes). The rule gain (+0.023 mean) doesn't overcome the budget penalty of dropping from 40→32.

**Takeaway:** a smarter rule is only useful if applied at a budget that isn't already dominated. The blend/Quest improvement at topk=32 says the rule has headroom — so the play (→ batch_2) is to run the best/new rules at a **productive budget (topk≈44)**, where +0.02-0.03 mean over rope_aware could land ABOVE the frontier (mean>0.70 at thr>3874 would dominate both batch_0 anchors). Frontier still: batch_0 id0 (3874,0.7125) + batch_0 id3 (4586,0.70).

### claude_opus_4_8/batch_2 — GLM MLA: 3 new algorithms vs proven blend at productive budget (topk=44)  (2026-06-06, task=aime26_glm)

| variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|---|
| batch_2_id0 | 6b7781319239 | 1.00 | 0.6833 | 0.933 | 3704 | 2281 | mean+Quest blend (proven bar) — dominated by batch_0 anchors |
| batch_2_id1 | 274036a1d657 | 1.00 | 0.6688 | 0.90 | 3641 | 2366 | split mean-kvc/Quest-kpe (NOVEL §3#5) — dominated |
| batch_2_id2 | d809c1cd4032 | 1.00 | 0.6479 | 0.80 | 3826 | 2253 | mass-aware LSE (NOVEL §3#7) — **worst**, variance term hurt |
| batch_2_id3 | 34edec90d91e | 1.00 | 0.7063 | 0.90 | 3513 | 2385 | asym granularity (NOVEL §3#6) — best of batch but dominated (low thr) |

**Knob matrix:** routing ALGORITHM varied at fixed topk=44 (id0 blend / id1 kv_c-mean+k_pe-Quest split / id2 mass-aware variance / id3 kv_c-coarse+k_pe-subblock); a topk44 rope_aware reference ≈0.70-0.703 (interp. batch_0 topk40=0.70, topk61=0.7125).

**Off-catalog hypotheses tested:** §3 #5 (split kv_c/k_pe estimators) → **REFUTED** (0.6688 < rope_aware ref ~0.70). §3 #6 (asym granularity) → **best of batch (0.7063 ≈ rope_aware) but dominated on throughput** (sub-block centroids cost width). §3 #7 (mass-aware LSE) → **REFUTED** (0.6479, pass 0.80) — exactly the math-predicted failure: MLA low-rank kv_c → correlated channels → diagonal-variance term misleads ranking.

**What moved:** nothing on the frontier — every point dominated by batch_0 rope_aware (3874,0.7125)/(4586,0.70). Special k_pe handling tops out at rope_aware-equivalent accuracy and ALWAYS costs throughput (extra cache width). The variance/mass idea actively hurt.

**Takeaway:** routing-rule cleverness is exhausted on this task — rope_aware mean-centroid is the frontier. The pivot is throughput-side: NARROW the summary (low-rank centroid) so rope_aware-quality recall costs less index bandwidth. See §1 strategic finding.

<!--
### <tag>/batch_<x> — <one-line theme>  (launched YYYY-MM-DD HH:MM, free GPUs: 0,3,5,7)

| variant       | content_hash | RULER acc | mean@16 | pass@16 | throughput (tok/s) | e2e_time (s) | notes |
|---|---|---|---|---|---|---|---|
| batch_<x>_id0 |  |  |  |  |  |  |  |
| batch_<x>_id1 |  |  |  |  |  |  |  |
| batch_<x>_id2 |  |  |  |  |  |  |  |
| batch_<x>_id3 |  |  |  |  |  |  | ← off-catalog (§3 hyp #?) |

**Knob matrix:** id0=…, id1=…, id2=…, id3=… (off-catalog: …)

**Off-catalog hypothesis tested:** §3 row #? — verdict: confirmed/refuted/inconclusive.

**What moved:** …

**Takeaway:** …
-->

---

## §3. Design hypotheses

> Pre-register each novelty hypothesis here the moment its batch
> launches. Record the specific op or behaviour exploited, the
> predicted direction of effect, and the verdict after results land.

| # | hypothesis (one sentence, names op/behaviour) | §16 bucket | batch | verdict |
|---|---|---|---|---|
| 1 | Quest channel min/max bound at SUB-BLOCK granularity (CMaxInterleave/CMinInterleave, k=16) then Max-pool over sub-blocks gives a tighter in-page upper bound than whole-page Quest or sub-block means → better recall at fixed topk=61 on MLA fused latent. | §16.4 first-principles | batch_0 | **REFUTED @topk61** (≈ recall, −12% thr); re-test @topk32 in batch_1 |
| 2 | Routing the RoPE term through a SiLU self-gate (Silu indexer op) — score = ⟨q_nope,c_kv⟩ + λ·SiLU(⟨q_pe,c_pe⟩) — lets positional alignment only *add* confidence (never penalize a content-strong page), re-ordering vs rope_aware's linear 1:1 add. | §16.2 untried knob | batch_0 | **REFUTED** (mean 0.7125→0.646; don't one-side-gate RoPE) |
| 3 | At TIGHT budget (topk=32) the sub-block Quest bound preserves accuracy better than the mean-centroid, because page selection actually bites when few pages are kept → it lands on the frontier where it didn't at topk=61. | §16.4 first-principles | batch_1 | **CONFIRMED** (0.6833 > 0.6646 @topk32; but topk32 itself dominated by topk40) |
| 4 | Spiky-page rescue / mean+Quest convex blend: score = (1−λ)⟨q̄,c_mean⟩ + λ·quest_bound credits pages whose optimistic bound exceeds their mean. | §16.3 inversion | batch_1 | **CONFIRMED, best rule** (0.6875 mean / 0.933 pass @topk32) → carry to topk44 in batch_2 |
| 5 | **(user steer) Treat kv_c and k_pe differently.** content kv_c via ⟨q_nope, mean⟩ + position k_pe via Quest channel-extrema bound. | §16.4 first-principles | batch_2 (=id1) | **REFUTED** (0.6688 < rope_aware ref ~0.70; k_pe-Quest no gain, costs width) |
| 6 | **(user steer, granularity)** block-mean kv_c + per-sub-block k_pe centroids, max over sub-blocks. | §16.4 first-principles | batch_2 (=id3) | **REFUTED on frontier** (0.7063 ≈ rope_aware but dominated — sub-block width kills throughput) |
| 7 | **Mass-aware / log-sum-exp routing** score = ⟨q̄,μ⟩ + β⟨q̄²,σ²⟩ routes by softmax-denominator MASS. | §16.4 first-principles | batch_2 (=id2) | **REFUTED** (0.6479, worst) — MLA low-rank → correlated channels → diagonal variance misleads, as math-researcher warned |
| 8b | **(PIVOT) Low-rank / channel-reduced centroid.** rope_aware ranking on a NARROWER centroid (CMeanInterleave dim=2 to 288/144-d, or drop k_pe → kv_c-only) cuts decode index bandwidth at ~equal recall → pushes the THROUGHPUT axis where rope_aware owns the frontier. | §16.2 untried knob (channel sparsity, Double Sparsity/ShadowKV for MLA) | batch_3 | pending |
| 8 | **Phase-robust freq-band k_pe** (math-verified, GLM RoPE = interleaved (2i,2i+1) → L2NormInterleave(k=2)): score = ⟨q̄_nope,c_kv⟩ + μ·Σ_band ‖q_band‖·mean‖k_band‖. Provable (loose 3×) phase-robust bound; only wins when band-energy structure exists (else ≈ mean-pool). **BLOCKED**: indexer Reshape to (288,2) needs pow2 inner dims → needs an indexer-side `L2NormInterleave` op. Deferred to batch_3, gated on whether k_pe-special-handling (id1/id3) shows signal in batch_2 + a real-trace band-energy check. | §16.4 first-principles | batch_3 | deferred (needs add-op) |

---

## §4. Anti-patterns

> One-liners of things that didn't work, so future iterations don't
> retry them. Include WHY it failed when known.

- **Self-gating the RoPE term (SiLU(⟨q_pe,c_pe⟩)) on GLM MLA** — mean@16
  0.7125→0.646. One-sided gating discards the negative-RoPE signal that
  rope_aware's symmetric linear add uses for page ranking. Don't asymmetrically
  gate RoPE; keep the linear NoPE+RoPE dot.
- **A `(block_size, D)` per-page SCRATCH cache field OOMs the MLA aux pool.** The
  mass-aware flow's naive E[latent²] used a `(32,576)` scratch; `memory_pool_mla`
  allocates every aux field across the WHOLE page pool, so a block-wide field is
  ~block× a normal `(1,D)` field → +2 GiB → CUDA OOM at boot (`_create_aux_buffers`).
  Preflight does NOT catch it (compile-only, no full-pool alloc). **Rule:** aux
  cache fields must be `(small, D)`; never `(block_size, D)`. For per-token squared
  moments, read `Σx²` off `CL2Norm(dim=1)` (no per-token materialization) and do
  the squaring on the indexer side. Also avoid intra-pass cache-op chained reads
  (one op reading another op's freshly-written aux field) — untested; keep each
  cache reduction reading `latent` directly.
- **Sub-block Quest extrema bound at GENEROUS budget (topk=61)** — no recall gain
  over the plain mean-centroid (0.706 vs 0.7125) and −12% throughput (extra
  CMin/CMax cache fields + 2nd GeMM). At topk=61 the mean-centroid is already
  recall-saturated, so a tighter bound buys nothing. (Open: may still help at
  tight budget — §3 #3.)

---

## §5. Patterns that worked (Pareto frontier)

> Confirmed winners worth carrying forward. Each entry should give
> the submission name, its `(throughput, mean@16)` coordinates, and
> why it works. A new entry replaces an older one only if it
> dominates on both axes; otherwise both stay (they are on different
> parts of the frontier).

| submission | mean@16 | throughput (tok/s) | RULER acc | notes |
|---|---|---|---|---|
| batch_0_id0 (rope_aware, block=32, topk=61) | 0.7125 | 3874 | 1.00 | GLM-4.7-Flash accuracy anchor; plain mean-centroid full-dot |
| batch_0_id3 (rope_aware, block=32, topk=40) | 0.70 | 4586 | 0.99 | GLM-4.7-Flash throughput anchor; topk=40 buys +18% thr for −1.75% mean |

---

## §6. Open questions / backlog

> Ideas and experiments that didn't fit the current batch but are
> worth trying. Pick from the top when a slot opens.

**COST MODEL (user steer — estimate cost; add ops/kernels for promising ideas).**
Decode per-step cost ≈ (a) KV attention `topk·block·576·2B` + (b) indexer reads
*every page's* summary `num_pages·summary_bytes`. Cache-field WIDTH is a
throughput knob: rope_aware = 1×576 centroid (1152 B/page); any Quest/3-field
flow = 3×576 (3456 B/page) → at long context (b) rivals (a), which is why
batch_0 id1 lost −12% thr. **Plan:** validate an idea's *accuracy* with existing
(possibly wide/slow) ops first; if it lands on/above the frontier, invest in a
fused/sliced op (→ /add-ops, vortex-op-author/kernel-expert) to cut its footprint
and make it Pareto-dominant.
- **Concrete add-op target:** a *sliced reduction* (mean over latent[0:512],
  max/min over latent[512:576]) so the split-kv_c/k_pe flow (hyp §3 #5) needs only
  512+64+64 = 640 dims (~1.1× baseline) instead of 3×576 = 1728. Build it only if
  batch_2 id1 shows the accuracy win.

**Queued for batch_1 (steer by batch_0 winners):**
- **Granularity sweep on sub-block Quest (id1 follow-up).** If id1 wins, push
  SUB_BLOCK_SIZE 16→8 (finer extrema box, n_b=4 at block=32) and test whether
  the tighter bound lets topk drop (61→45) at equal mean@16 → throughput win.
- **LAMBDA sweep on self-gated RoPE (id2 follow-up).** λ∈{0.5,1.0,2.0} on the
  `Silu(⟨q_pe,c_pe⟩)` bonus; λ→0 recovers rope_unaware, large λ ≈ pure-RoPE.
- **NOVEL — spiky-page rescue (disagreement routing).** Keep mean centroid AND a
  Quest channel envelope; score = ⟨q̄,c_mean⟩ + λ·Relu(quest_bound − ⟨q̄,c_mean⟩):
  add a bonus only when a page's max-achievable logit greatly exceeds its mean
  (a concentrated sharp key the mean smooths away). Ops: GeMM×3 + Add + Relu.
- **NOVEL — latent-norm (value-mass) temperature.** In MLA the latent kv_c IS
  the value proxy (kv_c→v absorbed); weight content score by per-page latent
  L2 norm via indexer `L2Norm` on the centroid. Small λ tie-breaker (large λ
  is query-blind → risky). §16.4 "v-signal no paper uses".
- **NOVEL — low-frequency RoPE routing.** k_pe (64 dims) is multi-frequency;
  MaskSlice-isolate only the low-freq RoPE dims (coarse position) for routing —
  test whether high-freq RoPE is noise for page-level selection.

---

## §7. Reading log

> Timestamped notes from reading tutorials, developer guides, source.
> One bullet per file, recording the single most useful insight.

- _none yet_

---

## §8. Session notes

> Per-session freeform, append-only. Record the date, what was
> attempted, and any context that doesn't fit the structured sections.

- **2026-06-06 — GLM-4.7-Flash iterate session start (tag claude_opus_4_8).**
  Model is MLA (glm4_moe_lite, kv_lora_rank=512, qk_rope_head_dim=64,
  qk_nope=192, v_head=256). Env = `conda run -n vortex_glm python`,
  `HF_HOME=/raid/catalyst/models/`. Built-in MLA flows live in
  `vortex_torch/flow/algorithms_mla.py` (rope_aware / lserve / quest /
  rope_unaware / rope_weighted_b*). MLA JSON keys:
  `attention_backend="cuda_mla"`, `vortex_attention_backend="trtllm"`,
  `vortex_impl_backend="triton"`, block=topk-page=32, known-good topk=61.
- **FRAMEWORK FIX (this session):** `check_engine_config` /
  `verify_flow_compilable` were **not MLA-aware** — `_read_hf_model_shapes`
  tried `head_dim = hidden_size // num_attention_heads` (fails: GLM
  2048/20 non-integer, no `head_dim` field), and `verify_flow_compilable`
  called `flow.initialize(head_dim=D)` which MLA's
  `initialize(block, kv_lora_rank, qk_rope_head_dim, ...)` rejects. Fixed
  additively: (a) `_read_hf_model_shapes` detects `kv_lora_rank` and returns
  `{mla, kv_lora_rank, qk_rope_head_dim, num_q_heads, latent_dim}`;
  (b) `_check_compilable` synthesizes q as [B, num_q_heads, latent_dim],
  num_kv_heads=1, D=latent_dim, passes `mla_dims=(kv_lora_rank,
  qk_rope_head_dim)`, sweeps block∈{16,32}; (c) `verify_flow_compilable`
  gained `mla_dims` kwarg that switches the `initialize` signature. MHA path
  byte-unchanged. All 4 batch_0 variants now preflight OK.
