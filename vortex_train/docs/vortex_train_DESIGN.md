# vortex_train — a training system for user-specified sparse attention

**Goal.** Let a user express *how each query group selects its KV* — the same
way `vortex_torch` submissions do (selection criteria, BOS/EOS reservation,
block size) — and train with it. Dense reference is
`flash_attn_varlen_func`; sparse forward **and backward** are ours.

**Scope decisions (settled up front, they shape everything):**

| decision | choice | consequence |
|---|---|---|
| selection gradient | **hard / straight-through** | top-k is a *constant mask* per step. Backward is a sparse-pattern FlashAttention, not a differentiable router. No grad flows into the scorer. |
| first regime | **long-context finetune of a dense model** | the selection recipe is fixed, the *weights* learn. Numerical parity with dense matters but pretrained weights absorb small error. Must be fast at 32k–256k, batch small. |

Non-goals for v1: differentiable/relaxed top-k, training the selector itself,
MLA (start MHA/GQA), pipeline/tensor parallel beyond what FSDP gives free.

---

## 0. The three facts that determine the architecture

Measured/derived before designing, because each kills an obvious approach.

**(a) Forward is balanced, backward is not — but the imbalance is tolerable.**
With causal masking and a fixed top-k budget, every query block attends exactly
`min(qb+1, topk)` blocks, so the forward is load-balanced per query block. A *KV*
block, however, is selected by an unpredictable number of query blocks —
max/mean work per KV block:

| config | kv blocks | max/mean |
|---|---|---|
| 32k, blk 64, topk 32 | 512 | 3.97× |
| 128k, blk 64, topk 32 | 2048 | 5.82× |
| 64k, blk 64, topk 16 (FSA's bench) | 1024 | 5.56× |

My first instinct was that this forces a **work queue** with split accumulation.
Re-checking at realistic shapes says otherwise: the ratio is roughly
topk-independent, and `kv_blocks` (512–2048) massively oversubscribes the SM
count, so the scheduler hides the variance. **FSA confirms this empirically** —
its `backward_dkdv` runs `grid = (batch, q_heads, kv_blocks)`, one program per KV
block, and just loops that block's selectors. No work queue, no atomics.

> Revised: one program per KV block, fp32 register accumulators, single store.
> Keep the work queue as a documented fallback for pathological patterns (very
> short sequences, or a scorer that concentrates selection on few blocks).

**(b) The transpose must be built on device.** Selection comes out q-major
(`sel[q_block] -> kv ids`); `dK/dV` needs kv-major. That's a counting sort:
`bincount → cumsum → scatter`, i.e. a CSR build. Device-only, no host
round-trip, no `.item()`. This is the single most important primitive in the
project — and **exactly what FSA does**: `torch.bincount(topk_idx...)` in the
forward, `torch.cumsum → cu_topk_q_count`, then a `reorder_topk_idx` scatter
kernel, with the per-block segment read as `cu_topk_q_count[k] : [k+1]` inside
`backward_dkdv`. Independent arrival at the same structure is good evidence it
is the right one.

**(c) Sparsity buys COMPUTE, not memory — so fusion is the whole game.**
This is worth being blunt about, because it is the most common misconception
about sparse *training*. In training `dK`/`dV` are needed for **every** KV token,
so nothing can be dropped from the saved tensors. Per layer at T=128k, Hq=32,
Hkv=8, D=128, bf16: dense FA saves 2.516 GB (q/k/v/o/lse); block-sparse saves the
**same tensors** plus the pattern (+0.002 GB) and per-block state (+0.004 GB) —
**2.522 GB, i.e. 1.002x dense.** Marginally worse.

What it buys: attention FLOPs drop from ~140.7 to ~4.4 TFLOP/layer (**~32x**) at
topk=32, blk=64. Compute stops growing quadratically in T; memory keeps growing
linearly.

Because memory parity is the *baseline*, the way to lose is a naive
implementation that materializes intermediates and ends up far **worse** than
dense. Two must never reach HBM:

| intermediate | size at 128k/layer | verdict |
|---|---|---|
| attention probs `p` = `T x (topk*blk) x Hq` | **16 GB** | always fatal — FlashAttention-style online softmax in SRAM, non-negotiable |
| block-score matrix `nq x nkv x Hkv` | 0.06 GB (but **4 GB at 1M**, O(T²)) | fuse `score`+`top-k` so only survivors leave SRAM |

So the backward recomputes `p = softmax(qk)` per block from `q,k` and the saved
`logsumexp`, exactly as FlashAttention does, and the scorer fuses down to its
top-k output. We store per step: `sel` (`[Q_blk, topk]` int32) + `lse`
(`[T, H]` fp32). Nothing else.

**Testable consequence:** peak memory must be **independent of `topk`**. A peak
that scales with `topk*blk` is the signature of a materialized `p` or an unfused
scorer, and is the single most important performance regression to guard.

---

## 1. Layer map

Mirrors vortex's frontend/compiler/backend split, which already works and which
users of `vortex_torch` will recognize.

```
vortex_train/
  flow/          FRONTEND  — user writes a Selection (the "flow"), registry
  compiler/      COMPILER  — op graph -> fused Triton/CUDA, autograd wiring
  ops/           op set    — the primitives a Selection composes (scorer side)
  kernels/       BACKEND   — fwd / bwd attention + the index primitives
  nn/            integration — SparseAttention module, HF/FSDP glue
  reference/     dense flash_attn_varlen + a masked-SDPA oracle for tests
```

### 1.1 Frontend — what the user writes

Deliberately the *same shape* as a vortex submission, so knowledge transfers.
A `Selection` declares its per-block state and how to score it:

```python
@register("block_topk")
class BlockTopK(Selection):
    """Score each KV block by <q̄, centroid>, keep the top-k."""

    # per-KV-block state, built once per forward from k (and optionally v).
    state = {"centroid": Field(reduce="mean", src="k")}     # [n_blk, D]

    budget = Budget(topk=32, reserve_bos=1, reserve_eos=1)  # blocks, not tokens

    def score(self, q, state, ctx):        # -> [n_q_blk, n_kv_blk]
        qbar = ops.Mean(dim="head")(q, ctx=ctx)            # GQA: group mean
        return ops.GeMM()(qbar, state["centroid"], ctx=ctx)
```

Notes on the contract, each earning its place:

- **`state` is declarative**, not a `forward_cache` method. In training there is
  no incremental KV cache to update — the whole sequence is present — so
  "build per-block state" is a *reduction over the block axis*, expressible as
  a spec. The compiler emits one fused reduction kernel for all fields. (This is
  the main frontend simplification versus `vortex_torch`, which needs
  `create_cache`/`forward_cache` because it updates state per decode step.)
- **`budget` is explicit and separate from `score`.** `reserve_bos`/`reserve_eos`
  are *not* the user's job to implement — getting "always keep the sink and the
  local window" right inside every scorer is exactly the bug farm vortex avoids
  by making it a config. The compiler force-includes those blocks and shrinks
  the top-k accordingly.
- **`score` returns a block-score matrix**, and returning it (rather than calling
  a terminal `topK` op as vortex does) is what lets the compiler own selection —
  including the causal constraint, dedup against reserved blocks, and the
  `sel`→CSR transpose. A user cannot get those wrong.
- **Per query *group***, not per head: `q` is presented as `[n_q_blk, G, D]` so
  GQA groups share a selection, which is what makes the sparse kernel coalesced.

### 1.2 Compiler — three phases, plus autograd

Phase structure copies `vortex_torch` (`profile` → graph → codegen); the new
part is that the graph is compiled **twice**, forward and backward.

```
Selection.score traced on zero-sized dummies   (profile; shapes only, no alloc)
        │
        ├─ scorer graph ──► fuse ──► one Triton kernel: state -> block scores
        │                             (elementwise/reduce/GeMM chains fuse; the
        │                              same subgraph fusion vortex already does)
        │
        └─ budget spec ──► selection kernel: causal mask + reserve + top-k
                                     └─► sel [Q_blk, topk] int32   (q-major)
                                     └─► CSR transpose (§0b)      (kv-major)
```

The compiled artifact is a `SparsePattern`: `(sel, csr_indices, csr_offsets,
lse_slot)`. It is a plain tensor bundle, so it crosses the autograd boundary as a
**non-differentiable constant** — which is precisely what "hard selection" means
and why the backward stays a normal FlashAttention variant.

**Autograd wiring** is one `torch.autograd.Function` at the attention boundary:

```
forward (T tokens, varlen cu_seqlens):
    pattern = compiled_selection(q, k, v, cu_seqlens)   # no grad
    out, lse = sparse_attn_fwd(q, k, v, pattern)
    save_for_backward(q, k, v, lse, pattern)            # NOT p, NOT the mask
backward:
    dq        = sparse_attn_bwd_dq(do, q, k, v, lse, pattern.sel)   # q-major
    dk, dv    = sparse_attn_bwd_dkv(do, q, k, v, lse, pattern.csr)  # kv-major
```

Two passes, not one, on purpose: `dq` is naturally q-major and balanced;
`dk/dv` is kv-major and imbalanced (§0a). Fusing them would force one traversal
order and pay the imbalance twice.

### 1.3 Backend — the kernel set

Five kernels. Everything else is composition.

| kernel | shape of parallelism | notes |
|---|---|---|
| `build_state` | one program per (KV block, field) | fused reduction over block; all fields in one launch |
| `score_and_select` | one program per query block | scorer subgraph inlined + causal + reserve + top-k in registers; writes `sel` |
| `transpose_csr` | bincount/cumsum/scatter | §0b; the only place atomics build an index |
| `attn_fwd` | one program per (q block, head) | loop over that block's `topk` KV blocks; online softmax; emits `out`, `lse` |
| `attn_bwd_dq` | one program per (q block, head) | same traversal as fwd, recompute `p` from `lse` |
| `attn_bwd_dkv` | one program per **(kv block, selector chunk)** | work-queue split; `atomic_add` into `dk/dv`, or fp32 staging + reduce |

`attn_bwd_dkv`'s split is the load-balance fix: long selector lists are chopped
into fixed-size chunks so every program does equal work, and the 4× imbalance
becomes a scheduling detail instead of a tail.

---

## 2. How the three constraints are honoured

**No Python/CPU loop.** The only host-side work is *tracing* (once, at
`compile()` time) and kernel launches. Per step the entire pipeline —
state build, scoring, top-k, transpose, fwd, bwd — is device kernels. Concretely
banned in the per-step path and checkable by grep: `.item()`, `.tolist()`,
`.cpu()`, `for` over batch/blocks/heads. The block counts a launch grid needs
(`n_q_blk`, `nnz = Q_blk*topk`) are known from shapes, never from a device read.

**No long sequential torch op chains.** A scorer like
`Mean → GeMM → Add → Softmax` is *not* executed as four torch ops; the compiler
fuses the subgraph into one kernel, which is the entire reason for having a
compiler rather than eager op objects. Same for `build_state` (all fields, one
launch) and for select (`causal + reserve + top-k` in registers, no intermediate
score matrix written to HBM when it fits in SRAM).

**Graceful.** The user writes ~10 lines (`state`, `budget`, `score`) and never
sees a kernel, an index transpose, or an autograd function. Everything
version- or hardware-specific lives in `kernels/` behind one dispatch, mirroring
how `vortex_torch/engine/sgl/compat/` isolates upstream churn.

---

## 3. Correctness plan

This is where a sparse-training project usually fails silently, so it is
designed in rather than added later.

**Precision policy.** The system under test is **bf16** end to end — q/k/v,
weights, and the kernels (bf16 math, fp32 accumulators, which is what `tl.dot`
plus an fp32 accumulator already gives). That matches FSA, flash-attn, and the
finetune itself. The *oracle* is a separate question: it must be precise enough
that a failure means a bug rather than noise.

| role | dtype | why |
|---|---|---|
| training / kernels | **bf16** (fp32 accum) | the real regime; nothing else is worth testing |
| masked-SDPA oracle | **fp32** | own error ~4e-6, ~5 orders below the bf16 gate (~2.5e-1 at S=4096) — so it resolves real bugs. A bf16 oracle would be indistinguishable from the thing it is checking. |
| `gradcheck` only | fp64, tiny shapes | finite differences bottom out at ~eps^(2/3): 1.5e-5 in fp32 (false failures on a softmax chain) vs 2.3e-11 in fp64 |

1. **Oracle equivalence.** For small shapes, build the dense boolean mask the
   `SparsePattern` implies and run masked SDPA in **fp32**. Assert
   `out`, `dq`, `dk`, `dv` all match the bf16 kernels within bf16 tolerance. This
   is the ground truth — it tests the *pattern semantics* (causal, BOS/EOS
   reserve, dedup) independently of kernels.
2. **Dense degeneracy.** With `topk >= n_kv_blk`, sparse must equal
   `flash_attn_varlen_func` in bf16, forward and backward. Any drift here is a
   kernel bug, not an approximation.
3. **Gradcheck** in fp64 at tiny shapes, for the non-selection inputs only —
   the one place double precision is actually load-bearing (see table).
4. **The transpose is an invariant, not a guess:** assert
   `set(csr[kv]) == {q : kv in sel[q]}` and `csr_offsets[-1] == sel.numel()`
   on random patterns. A wrong transpose silently drops gradient — the worst
   failure mode in the system, because loss still goes down.
5. **Reserved-block accounting:** `reserve_bos + reserve_eos + topk` must never
   double-count; assert exact selected-block counts per query block.

Only after (1)–(5) pass does throughput matter.

---

## 3.5 Relationship to Flash-Sparse-Attention (FSA)

[Relaxed-System-Lab/Flash-Sparse-Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)
(Apache-2.0) is a re-engineered kernel for **NSA**'s selected-attention branch.
It is the closest existing implementation to our backend layer, and it should be
the **starting point for `kernels/`** rather than something to reimplement.

**What it already solves (adopt):**

| problem | FSA's answer |
|---|---|
| dK/dV accumulation | one program per KV block, fp32 **register** accumulators, single store — no atomics, no staging |
| the inverted index | `bincount → cumsum → reorder_topk_idx` scatter; segment per block via `cu_topk_q_count` |
| dQ | separate 2-kernel path: `dq_compute_kernel` → bf16 staging buffers → `dq_reduce_kernel` |
| small GQA groups | the real problem NSA has — tiny tiles force padding (MMA needs ≥8, Triton tile ≥16). FSA inverts the loop nest (KV outer, query inner) to batch query tokens sharing a KV block |
| varlen | `cu_seqlens` throughout |

That last row is the insight I did not have: **the binding constraint for
*training* sparse attention is GQA group size, not the sparsity pattern.** With
group < 8 the natural tile is too small and you pay for padded math; FSA's loop
inversion is what recovers it. FSA even falls back to reference NSA at group ≥ 8,
where the tile is big enough that the inversion stops paying. Our
`score`-returns-a-block-matrix / per-query-*group* contract (§1.1) is compatible
with this, which is fortunate — but the kernel choice should now be
**group-size-dependent**, and that belongs in the backend dispatch, not the
frontend.

**What our project must still add (this is the actual delta):**

1. **Generality of selection.** FSA implements *NSA's* selection (compressed-score
   top-k + fixed init/local blocks). We want the selection to be **user-specified
   and compiled** — arbitrary scorer over declarative per-block state. FSA's
   kernels take `topk_idx` as an input, so this composes cleanly: our compiler
   produces `topk_idx`, their kernels consume it.
2. **The frontend + compiler.** FSA has no scorer DSL, no tracing, no fusion —
   its selection is hand-written for one algorithm. That layer is ours, and it is
   where the vortex lineage actually applies.
3. **BOS/EOS reservation as config.** NSA hardcodes `init_blocks`/`local_blocks`;
   we make reserve counts a `Budget`, which the selection kernel force-includes.
4. **Correctness harness.** FSA's README documents a fwd/bwd correctness compare
   vs NSA reference; we want the fp32 masked-SDPA **oracle** (§3) so pattern
   semantics are tested independently of any kernel.

**Constraints inherited from FSA** (worth stating because they bound v1):
head_dim ≤ 256 and equal across Q/K/V; fp16/bf16; Ampere/Hopper (its tested
envelope is A100/H20/H100/H200 — **note it does not list Blackwell/B200**, which
is what our cluster has, so step 3's first job is confirming it builds and is
fast on sm100); `flash-attn == 2.6.3` pin; block/topk pairs validated at
(64,16) and (128,8).

**License/attribution:** Apache-2.0 permits reuse with attribution. If we vendor
or adapt kernels, they go in `kernels/thirdparty/fsa/` with the license retained
and modifications noted — the same discipline as `third_party/sglang/` in
vortex_torch.

## 4. Build order

Each step ends somewhere testable; nothing depends on a later step.

Revised now that FSA supplies the backend: the kernel work becomes *port and
verify* rather than *invent*, and the risk moves to the compiler.

| # | deliverable | gate |
|---|---|---|
| 1 | `reference/`: dense varlen + fp32 masked-SDPA oracle | oracle matches bf16 flash_attn dense within bf16 tol |
| 2 | Stand up FSA as-is on our hardware; run its own unit tests | **builds and is correct on B200/sm100** (untested upstream) |
| 3 | `SparsePattern` contract + our `transpose_csr`, cross-checked against FSA's `cu_topk_q_count` | invariant tests §3.4; identical to FSA's index on the same `topk_idx` |
| 4 | Drive FSA's fwd/bwd from a **synthetic** `topk_idx` (no scorer) | vs fp32 oracle, fwd + dq/dk/dv; dense degeneracy §3.2 |
| 5 | Frontend + compiler: trace `score`, fuse, emit `topk_idx` + reserve | one recipe (`block_topk`) end-to-end vs oracle |
| 6 | `nn.SparseAttention` + FSDP/HF glue | a real finetune step runs, loss decreases |
| 7 | More recipes; backend dispatch on GQA group size; perf | throughput vs dense at 32k/128k |

**The risk moved.** Step 4 was the hard part when we owned the kernels; FSA
answers it. The new risks are (a) **step 2** — FSA's tested envelope is
Ampere/Hopper and our cluster is B200/sm100, so this must be settled before
building on it; and (b) **step 5** — a compiler that emits a `topk_idx`
*bit-identical* to what a hand-written selector would produce, including the
causal edge cases and reserve dedup. Step 3 exists to make (b) checkable in
isolation: same input, same index, before any kernel is involved.

---

## 5. Honest risks

- ~~`atomic_add` on bf16 `dk/dv`~~ — **retired.** FSA shows one-program-per-KV-block
  with fp32 register accumulators avoids atomics entirely. Note FSA's remaining
  cross-program sum is over GQA-shared query heads, done in **bf16** then
  `.sum(0)` in PyTorch; that reduction's precision is worth measuring at long
  seqlen, since it accumulates over `num_share_q_heads` terms in low precision.
- **B200/sm100 is outside FSA's tested envelope** (A100/H20/H100/H200). It pins
  `flash-attn==2.6.3`, which may not have sm100 wheels. This is now the first
  gate (step 2), not a late surprise: if it needs porting, that changes the
  project's size materially.
- **Sparse training may simply not converge** for a given recipe. That is a
  research outcome, not a bug in this system; the system's job is to make the
  comparison cheap and the gradient correct. Step 6's gate is "loss decreases",
  not "matches dense loss".
- **Block size interacts with the kernel's tile size.** A user-facing
  `block_size` that differs from the kernel's `BLOCK_N` forces either a gather
  or a constraint. v1: require `block_size % BLOCK_N == 0` and say so loudly,
  rather than silently gathering.
- **Recompute cost.** Backward recomputes `p`; at `topk` small this is cheap
  relative to dense, but the *scorer* also reruns unless cached. Save
  `sel` (tiny) — never re-derive it in backward, or a nondeterministic scorer
  would corrupt gradients.
- **GQA group size is the real performance variable**, not sparsity (§3.5). FSA
  wins below group 8 and falls back above it. So "is this faster than dense?" has
  no single answer — the benchmark must sweep group size, and the backend must
  dispatch on it. Any headline speedup quoted without the group size is
  meaningless.
- **A compiler that emits a subtly different `topk_idx`** than a hand-written
  selector is the nastiest failure mode: training still converges, just to a
  slightly different objective, so nothing looks broken. Step 3's
  index-equivalence test is the only real defence.

---

## 6. Why not just reuse `vortex_torch`

Worth stating, since the ask is "based on vortex":

**Reused wholesale (conceptually, and largely portable as code):** the
three-phase op/compiler design, subgraph fusion, the `FORMAT`/vTensor
abstraction, the registry pattern, and the discipline of resolving everything at
trace time so per-step paths are pure device work.

**Genuinely different, and why:**
1. `vortex_torch` is inference-only — there is no backward anywhere in it. The
   backward, its transposed index, and its work queue are the bulk of the new
   engineering.
2. Its cache ops exist to *incrementally update* per-page state across decode
   steps (`Load`/`Save`, `CFill` zeroing). Training sees the full sequence at
   once, so that whole machinery collapses into one declarative reduction — a
   large simplification, and the reason `state` replaces
   `create_cache`/`forward_cache`.
3. Selection is per-*step* there and per-*block-row* here (`[n_q_blk, topk]`,
   not `[1, topk]`), so the top-k kernel and metadata layout are new.
4. It plugs into sglang's serving loop; this plugs into an autograd graph and
   FSDP. No shared integration surface.
