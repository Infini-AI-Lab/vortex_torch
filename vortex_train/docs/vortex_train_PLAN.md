# vortex_train — build plan

Companion to `vortex_train_DESIGN.md`. Read the design for *what* and *why*; this
is *in what order*, *with what gate*, and *what kills it*.

---

## The replan, up front

The design originally had us writing the sparse attention kernels (fwd, bwd_dq,
bwd_dkv) in Triton, informed by Flash-Sparse-Attention. Two corrections changed
that:

1. **FSA is a reference to read, not a dependency.** We build on **torch 2.11 +
   triton 3.6 + flash-attn-4**, the stack the cluster already runs. No separate
   env, no `flash-attn==2.6.3` pin, no vendored `fsa/`.
2. **FA4 already ships block-sparse attention with a backward.** `flash_attn.cute`
   (CuTeDSL, sm90/sm100/sm120 — i.e. Blackwell-native) contains
   `block_sparsity.py`, `flash_fwd_sm100.py`, `flash_bwd_sm100.py`,
   `topk_gather_kv.py`, `compute_block_sparsity.py`.

| we were going to build | FA4 already has |
|---|---|
| block-sparse `attn_fwd` | `flash_fwd_sm100` + `block_sparsity` |
| `attn_bwd_dq` / `attn_bwd_dkv` | `flash_bwd_sm100` |
| `sel [Q_blk, topk]` q-major | `mask_block_cnt [B,H,M]` + `mask_block_idx [B,H,M,N]` |
| the CSR transpose for dK/dV | backward's Q-direction cnt/idx (same structure) |
| the atomics/work-queue fix | deterministic semaphore `dq_write_order` |
| varlen index plumbing | `cu_total_m_blocks`, `cu_block_idx_offsets` |

**So the project's differentiator is not the attention kernels.** It is exactly
the part vortex is good at and FA4 deliberately leaves to the caller:

> FA4's block sparsity is "utilities for FlexAttention" — the *user* supplies the
> mask. There is **no scorer**. We supply: a DSL for expressing selection
> criteria, a compiler that fuses a scorer into one kernel, and the top-k that
> emits FA4's `mask_block_cnt/idx`.

This is a better shape: we own the expressive layer, and consume a maintained,
tuned, Blackwell-native kernel for the pure engineering.

---

## What sparsity buys — and what it does not

Stated plainly because an earlier draft of this plan got it wrong (it claimed
"memory < dense" as a Phase 4 gate).

**Sparse attention does not reduce memory. It reduces compute.** In *training*,
`dK`/`dV` are needed for **every** KV token, so nothing can be dropped from the
saved tensors. Per layer at T=128k, Hq=32, Hkv=8, D=128, bf16:

| tensor | dense FA | block-sparse |
|---|---|---|
| q / k / v | 1.000 / 0.250 / 0.250 GB | identical |
| o | 1.000 GB | identical |
| lse | 0.016 GB | identical |
| pattern `cnt`+`idx` | — | +0.002 GB |
| per-block state | — | +0.004 GB |
| **total** | **2.516 GB** | **2.522 GB (1.002x)** |

So sparse is *marginally worse* on memory. What it buys, at topk=32, blk=64:

| | dense causal | block-sparse |
|---|---|---|
| attention FLOPs / layer | ~140.7 TFLOP | ~4.4 TFLOP (**~32x less**) |

The win is compute (and therefore time), plus the fact that compute stops growing
quadratically in T while memory grows only linearly.

### The fusion requirement (the real memory story)

Because memory parity is the *baseline*, the failure mode is a naive
implementation that is much **worse** than dense by materializing intermediates.
Two must never reach HBM:

1. **The attention probabilities `p`.** `T x (topk*blk) x Hq` = **16 GB per
   layer** at 128k. Non-negotiable: FlashAttention-style online softmax, tiles
   in SRAM. FA4 already does this; our job is to not defeat it.
2. **The block-score matrix** the scorer produces, `nq x nkv x Hkv`. This one
   grows O(T^2) and is the argument for fusing `score` + `select_topk` into one
   kernel so only the survivors leave SRAM:

   | T | score matrix | emitted `idx` | ratio |
   |---|---|---|---|
   | 32k | 0.004 GB | 0.0005 GB | 8x |
   | 128k | 0.06 GB | 0.0020 GB | 32x |
   | 256k | 0.25 GB | 0.0039 GB | 64x |
   | 1M | **4.00 GB** | 0.0156 GB | 256x |

   At 128k it is merely wasteful; past ~256k (x N layers) it is fatal. Fusing is
   also a pure latency win regardless — one kernel, no HBM round-trip.

**Restated gate:** memory ~= dense (within ~1%), and peak memory must be
**independent of `topk`** — a peak that scales with `topk*blk` is the signature
of a materialized `p` or an unfused scorer.

---

## Phase 0 — resolve the one blocking unknown (do this first, ~1 day)

Two independent reads of `flash_attn/cute/interface.py` **disagreed** on whether
varlen + block-sparsity is supported in the *backward*:

- one reported an assert in `_flash_attn_bwd` that varlen backward with block
  sparsity "is not yet supported";
- the other could not find it (the file is ~4100 lines and got truncated).

Our target regime is **varlen + block-sparse + backward**. If that combination is
unsupported, the plan changes shape. This is not resolvable by more doc-reading —
it needs the installed package.

```bash
# in a scratch env on a B200 node
pip install "flash-attn-4[cu13]"
python - <<'PY'
import inspect, torch
from flash_attn.cute import interface as I
print(inspect.signature(I.flash_attn_varlen_func))
src = inspect.getsource(I._flash_attn_bwd)
for i, l in enumerate(src.splitlines()):
    if "block_spars" in l and ("assert" in l or "support" in l): print(i, l.strip())
PY
```

Then the empirical test, which is what actually decides it: build a tiny
block-sparse varlen case, call backward, see whether it raises.

**Outcomes and branches:**

| outcome | consequence |
|---|---|
| **A. varlen + sparse + bwd works** | best case. Proceed as below; we write *no* attention kernels. |
| **B. works for fixed-length only** | Phase 1–4 target fixed-length (pad-and-mask, one seq per batch row). Varlen becomes a later milestone: either wait for upstream, contribute the kernel, or fall back to our own Triton bwd for the varlen path only. |
| **C. sparse bwd broken generally** | we write `bwd_dq`/`bwd_dkv` in Triton after all, per the original design (FSA's structure). Adds ~3–4 weeks and makes kernels the critical path again. |

Also settle in Phase 0, same session (all cheap, all shape-fixing):
- `pack_gqa` requires block-sparse head dim == 1 (broadcast) — does that force
  one selection shared across *all* heads, or per-KV-head? Our design promises
  per-query-*group* selection; confirm the two are compatible.
- Block sparsity disables the 2-CTA Blackwell fast path (`not use_block_sparsity`)
  → quantify the cost: dense-with-2CTA vs dense-without, to know the ceiling
  sparsity must beat.
- `block_size` constraints: KV block must be a multiple of kernel `tile_n`, Q
  block a multiple of `q_stage * tile_m`. These become validated frontend limits.

**Do not write any vortex_train code before Phase 0 reports.** Everything below
assumes outcome A or B.

---

## Phase 1 — the pattern contract + oracle (foundation, no kernels)

Goal: be able to *state* a sparse pattern, convert it to FA4's format, and know
what the right answer is — before any selection logic exists.

```
vortex_train/
  pattern.py     SparsePattern  <-> BlockSparseTensorsTorch  (ours <-> FA4)
  reference/
    dense.py     flash_attn_varlen_func passthrough (bf16 baseline)
    oracle.py    fp32 masked SDPA from an explicit block mask
```

- `SparsePattern` is our stable internal form: `(cnt, idx, block_size, causal,
  reserve)`. It converts to FA4's `BlockSparseTensorsTorch` in one function, so
  an upstream layout change touches one file (the `compat/` lesson from vortex).
- Oracle expands a `SparsePattern` to a dense boolean mask and runs fp32 SDPA.
  Precision policy per design §3: **bf16** system under test, **fp32** oracle,
  **fp64** only for `gradcheck` at tiny shapes.

**Gates**
1. Oracle vs bf16 dense `flash_attn_varlen_func`, full-mask: match within bf16 tol.
2. Round-trip `SparsePattern → FA4 tensors → dense mask` is the identity.
3. Pattern invariants: counts match packed-index lengths; `idx` valid range;
   causal respected; `reserve_bos/eos` never double-counted.

---

## Phase 2 — sparse attention on a *given* pattern (no scorer yet)

Feed FA4 a pattern we construct by hand and verify forward **and backward**.
This isolates "can we drive FA4 correctly" from "is our scorer right".

Patterns to test, chosen because each breaks a different assumption:
`full` (degeneracy), `causal-diagonal-only`, `strided`, `random-topk`,
`BOS+local` (the realistic shape), `topk > n_kv_blk` (clamping).

**Gates**
1. **Dense degeneracy**: full mask ⇒ equals `flash_attn_varlen_func`, fwd + dq/dk/dv, bf16 tol.
2. Every pattern: `out, dq, dk, dv` vs fp32 oracle.
3. `gradcheck` (fp64, tiny) on q/k/v.
4. **Determinism**: same inputs twice ⇒ bit-identical `dk/dv`. (FA4 has a
   deterministic path via `dq_write_order`; confirm we're on it. Non-determinism
   here would make every later comparison unfalsifiable.)

Deliverable: `nn.SparseAttention` usable with a hand-written pattern. Already
useful — someone can train with a fixed pattern at this point.

---

## Phase 3 — the scorer: frontend + compiler + top-k (the real work)

This is where the project earns its existence. Everything before was plumbing.

```
flow/       Selection base, @register, Field/Budget specs
ops/        Mean, GeMM, Add, L2Norm, Softmax... (the scorer op set)
compiler/   trace on zero-sized dummies -> graph -> fuse -> one Triton kernel
kernels/    build_state, score, select_topk  (+ pattern build)
```

Three kernels, and no others:

| kernel | parallelism | job |
|---|---|---|
| `build_state` | one program per (KV block, field) | fused reduction over the block; all `Field`s in one launch |
| `score` | one program per Q block | the fused scorer subgraph; scores stay in SRAM |
| `select_topk` | one program per Q block | causal mask + force-include reserved + top-k, in registers → emits FA4 `cnt/idx` directly |

Deliberately **no separate transpose kernel** — FA4 builds the backward index
itself (`compute_dq_write_order`). That deletes the primitive I'd called "the
single most important" in the design; worth stating plainly, because it was the
thing I was proudest of and it turned out to be someone else's problem.

**Gates**
1. `block_topk` end-to-end vs fp32 oracle (fwd + all grads).
2. **Index equivalence**: compiler-emitted `cnt/idx` is *bit-identical* to a
   hand-written numpy selector on the same scores. This is the anti-silent-bug
   test — a subtly wrong pattern still trains, just to a different objective.
3. **No host work per step**: assert no `.item()/.cpu()/.tolist()` and no
   Python loop over batch/blocks/heads in the step path (grep + a profile check
   that the step issues a constant number of launches independent of seqlen).
4. Fusion actually happened: scorer chain is 1 kernel, not N. Check the launch
   count, not the source.

---

## Phase 4 — training integration

`nn/`: HF attention-module drop-in, FSDP compatibility, a real finetune script.

**Gates**
1. A Qwen3-4B-scale finetune step runs at 32k, loss decreases over ~200 steps.
2. **Memory at parity with dense, not below it** — and no materialized
   intermediates (see "What sparsity buys" below). Assert saved-for-backward is
   within ~1% of dense at the same seqlen, and that peak memory does **not**
   scale with `topk*blk` (which would mean `p` got materialized).
3. Checkpoint save/load round-trips.
4. bf16 GQA-shared-head reduction precision (design §5): measure whether the
   low-precision accumulation over `num_share_q_heads` degrades gradients at
   long seqlen. Escalate to fp32 staging if so.

---

## Phase 5 — recipes and performance

Port vortex's selection catalogue: quest-style min/max envelopes, lserve
centroids, `venergy`-style gating. Each is a `Selection` subclass; if the op set
is right, each is ~15 lines and needs no new kernel. **That is the test of
whether the frontend design was correct** — if a recipe needs a new op, the op
set was wrong.

Then performance, and here the honest framing: **speedup must be reported against
dense with its fast paths enabled**, including the 2-CTA path that block sparsity
disables. Sweep GQA group size (the FSA finding: group size, not sparsity, is the
binding variable) and seqlen. A number quoted without group size and without the
dense baseline's configuration is meaningless.

---

## Risks, in order of what would hurt most

1. **Phase 0 outcome C** (sparse bwd unusable) — adds a kernel project. Mitigated
   by doing Phase 0 first and by the design already containing the Triton plan.
2. **`pack_gqa` head-dim-1 constraint** conflicts with per-group selection. Would
   force one selection shared across heads — a semantic change to the frontend
   promise, so it must be settled in Phase 0.
3. **FA4 is young and moving.** CuTeDSL, no versioned block-sparse API, README
   doesn't even document varlen/bwd. Mitigation: `pattern.py` is the *only* file
   that knows FA4's layout; pin a commit; keep the oracle so upstream drift shows
   up as a test failure rather than silent wrongness.
4. **Sparse finetune may not converge** for a given recipe. That is a research
   outcome, not a bug — the system's job is a correct gradient and a cheap
   comparison. Phase 4's gate is "loss decreases", not "matches dense".
5. **The 2-CTA loss may eat the sparsity win** at moderate seqlen. Quantified in
   Phase 0 so we know the crossover before building on the assumption.

---

## Sequencing summary

| phase | output | hard gate |
|---|---|---|
| 0 | feasibility report | varlen+sparse+bwd works, or a chosen branch |
| 1 | `SparsePattern`, oracle | oracle == dense; round-trip identity |
| 2 | `nn.SparseAttention` on given patterns | dense degeneracy + oracle grads + determinism |
| 3 | Selection DSL + compiler + 3 kernels | index equivalence; 1 fused kernel; no host work |
| 4 | finetune integration | loss decreases at 32k; memory ~= dense; attention FLOPs down ~topk*blk/(T/2) |
| 5 | recipe catalogue + perf | new recipe needs no new kernel |

Phases 1–2 are useful on their own (fixed-pattern training). Phase 3 is the
contribution. Do not start 3 before 2's gates pass — a scorer debugged against an
unverified kernel is unfalsifiable.
