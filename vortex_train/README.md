# vortex_train

Training-side block-sparse attention: users **declare how each query group selects
its KV** — selection criteria, BOS/EOS/local reservations, block size, the same
contract as `vortex_torch` — and the system compiles that into fused Triton kernels
and trains through it, forward and backward.

The dense reference is flash-attention-varlen. Sparsity buys **compute**, not
memory: peak memory matches dense because nothing large is materialized — not the
attention probabilities, not the block-score matrix.

## What a user writes

```python
from vortex_train import Budget, Field, Selection, SparseAttention, register
from vortex_train.flow import ops

@register("my_policy")
class MyPolicy(Selection):
    # per-KV-block state, built once per forward in one fused reduction
    state  = {"centroid": Field(reduce="mean", src="k")}
    # total block budget; reservations are config, not user code
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)
    # block_q = query tokens sharing one selection. 64 amortises the scorer; 1 gives
    # every token its own selection (no averaging) at ~7x the step -- see the block_q
    # table below.
    block_q = block_kv = 64

    def __init__(self):
        self.qbar = ops.QSummary(how="mean")
        self.dot  = ops.Dot(group_reduce="max")

    def score(self, q, state, ctx):
        return self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)

attn = SparseAttention("my_policy", num_kv_heads=8)
out = attn(q, k, v)          # backward is exact over the selected blocks
out.sum().backward()
```

`score` is **traced, not executed**. The compiler lowers it to generated Triton
source, so the scorer chain runs on a register-resident score vector and the
O(T²) block-score matrix never reaches HBM (4 GB per layer at 1M tokens if it
did). The user never sees the top-k, the causal edge, reservation dedup, the CSR
transpose, or the autograd wiring — the parts that fail *silently* when
hand-written.

Selection is a **hard mask** (straight-through): no gradient flows into the
scorer, so the pattern crosses the autograd boundary as a constant.

## Layout

```
vortex_train/
  flow/            FRONTEND  — Selection / Field / Budget, the op set, recipes
    spec.py          the contract users subclass
    ops.py           symbolic scorer ops (QSummary, Dot, Envelope, Norm, Scale, Distance)
    recipes.py       9 registered policies, none needing a new kernel
  compiler/        COMPILER  — trace score() -> graph -> lowered tape
  kernels/         BACKEND
    state.py         per-block state: all Fields in ONE fused reduction
    select.py        generated fused score + reserve + top-k, one launch
    score_ops.py     device fns the generated kernel calls
    fwd.py           online-softmax forward (never materializes p)
    transpose.py     q-major -> kv-major CSR (count/scan/scatter, 3 kernels)
    bwd.py           dq q-major + dk/dv kv-major + fused GQA group reduce
  nn/              integration — autograd.Function + SparseAttention module
  reference/       fp32 masked-SDPA oracle (ground truth)
tests/             150 tests
benchmarks/        attention speed; selection overhead; e2e breakdown; recall
```

Design rules held throughout: **no python/CPU loop and no long sequential torch op
chains** on any per-step path. The pattern transpose, the GQA group reduction, and
the multi-field state build are each one custom kernel rather than
`bincount`/`cumsum`/`view+sum`/per-field chains. `test_no_host_work_in_step_path`
enforces this by AST inspection — no `.item()`/`.tolist()`/`.cpu()` on the step
path, so nothing silently syncs.

Per step: state (1 launch) → score+select (1) → transpose (3) → attention (1),
plus a matching backward set. No launch count depends on sequence length.

## Test

```bash
python -m pytest tests/ -q          # 150 passed, 1 skipped
```

The load-bearing test is `test_index_equivalence`: for every registered policy and
GQA group, the kernel-emitted `cnt`/`idx` must be **identical** to an independent
torch selector written from the spec — not close, identical. A pattern that differs
by one block still trains and still converges, to a slightly different objective,
so nothing looks broken; this is the only real defence against a silently wrong
compiler.

Precision policy: bf16 system under test (fp32 accumulators), fp32 oracle, fp64
only for `gradcheck`. Tolerances scale with the reference's dynamic range, since
bf16 reduction error tracks the largest term summed rather than each output
element. `test_no_worse_than_dense_bf16` calibrates against a real bf16 SDPA kernel
instead of a hand-picked constant.

## Measured: `block_q` — selection granularity (B200, bf16, Hq=32, Hkv=8, D=128, group 4)

`block_q` is how many query tokens share one KV selection. `block_q=64` amortises the
scorer over 64 tokens; **`block_q=1` gives every token its own selection**, with no
averaging of queries into a block — the accuracy reference, and the expensive end of the
tradeoff. Budget: topk 16 + 1 BOS + 1 local = 18 blocks × 64 = **1152 KV tokens**.

Whole-step fwd+bwd (selection + transpose + attention + backward), ms:

| seqlen | dense fb | `bq=1` | `bq=4` | `bq=16` | `bq=64` |
|-------:|---------:|-------:|-------:|--------:|--------:|
|   4096 |     2.00 |   7.55 |   3.76 |    2.81 |    1.80 |
|  16384 |    23.87 |  35.59 |  17.83 |   10.95 |    6.26 |
|  32768 |    90.30 |  77.47 |  39.07 |   22.88 |   12.58 |
|  65536 |   353.43 | 188.13 |  87.25 |   48.45 |   25.67 |

Speedup vs dense SDPA-flash:

| seqlen | `bq=1` | `bq=4` | `bq=16` | `bq=64` |
|-------:|-------:|-------:|--------:|--------:|
|   4096 |  0.26× |  0.53× |   0.71× |   1.11× |
|  16384 |  0.67× |  1.34× |   2.18× |   3.81× |
|  32768 | 1.17× |  2.31× |   3.95× |   7.18× |
|  65536 | **1.88×** |  4.05× |   7.29× |  13.77× |

`block_q=1` costs ~7× `block_q=64` and only beats dense past ~32k. It is a correctness
reference and an accuracy option, not a throughput setting. The whole step is timed
deliberately: selection cost scales as `1/block_q`, so timing the attention kernel alone
would flatter small `block_q`.

Peak memory (MB) — note it *drops* below `block_q=16`:

| seqlen | `bq=1` | `bq=4` | `bq=16` | `bq=64` |
|-------:|-------:|-------:|--------:|--------:|
|  16384 |    663 |    650 |    1159 |    1158 |
|  32768 |   1325 |   1300 |    2318 |    2317 |
|  65536 |   2650 |   2601 |    4636 |    4633 |

That is a dispatch effect, not a `block_q` one: `block_q < 16` uses the packed dk/dv
kernel, which folds the GQA group into the MMA contraction and writes `dk`/`dv` directly,
where the general path stages `[B, Hq, Skv, D]` fp32 buffers and reduces them in a second
pass. The 496 MB drop at 16k matches the 512 MB those two buffers occupy.

### Two optimisations that made `block_q=1` viable

It started at 80.78 ms (seqlen 16k, 0.30× dense). Both wins came from `nsys` profiling,
and both were invisible at `block_q=64`:

| | before | after |
|---|---:|---:|
| CSR segment sort | 37.19 ms (45% of step) | **0** — removed |
| forward kernel | 14.2 ms | 8.16 ms |
| dq kernel | 16.3 ms | 12.30 ms |
| **step** | **80.78 ms** | **35.59 ms** (2.27×) |

1. **An O(len²) segment sort was the single largest cost.** Segment length scales as
   `1/block_q`, so a sort that is genuinely free at `block_q=64` (0.1 ms over 8-entry
   segments) turned quadratic at `block_q=1` (1152-entry segments). It had only ever been
   benchmarked at `block_q=64`. Replaced by a chunked ordered scatter — count per
   (kv block, query chunk), prefix over chunks, each chunk fills its reserved slice — so
   segments come out ascending with no atomic and no sort.
2. **`num_warps` keyed off `head_dim` rather than tile rows.** At `block_q=1` the query
   tile is 16 rows, so 8 warps left most of the MMA idle. Measured optima: the forward
   wants 1 warp at a 16-row tile and 4 at 64 rows; dk/dv is the exception and keeps 8,
   because its long gathered segment loop needs the warps to hide latency.

The transpose is now 2.67 ms at `block_q=1` / 16k (7.6% of the step) and 0.36 ms at
`block_q=64` — up from 0.08 ms, the price of the extra chunk-prefix pass that buys
determinism without a sort. `BLOCK_M=64` is fixed rather than tuned because its two
consumers want opposite things: a larger value shrinks the `[Nkv, chunks]` counts tensor
but makes the scatter's rank computation O(BLOCK_M²). Scaling it to bound the chunk axis
was measured at **32.4 ms** — 12× worse — and reverted.

Two designs were built, measured, and **rejected**, recorded in the kernel docstrings so
they are not retried:

* **Unioning neighbouring query blocks** to fill the tile. The union of 16 consecutive
  tokens' selections measured 102.7 blocks, not the ~18 that strong overlap would imply,
  making total work **1.47× worse**. An initial estimate said 0.44× *better* — it had
  sampled only the first 4096 tokens, which are causally limited to few KV blocks and so
  have artificially small unions.
* **A KV-major forward** (as Flash-Sparse-Attention and flash-moba do, gathering scattered
  query rows into a dense tile). It removes padding entirely but cannot finish the online
  softmax in one pass, so it needs `nnz × D` fp32 partials — ~4 GB at seqlen 16k across 32
  heads.

Reproduce with `python benchmarks/bench_block_q.py`.

## Measured: `block_q=64` (B200, torch 2.11.0+cu130, bf16, Hkv=8, D=128, block=64, topk=16)

Forward+backward vs SDPA flash backend, `fb` = fwd+bwd:

| seqlen | group | dense fb (ms) | sparse fb (ms) | speedup | FLOP ratio |
|-------:|------:|--------------:|---------------:|--------:|-----------:|
|   4096 |     4 |         1.990 |          1.878 |   1.06× |         2× |
|  16384 |     4 |        23.855 |          6.592 |   3.62× |         8× |
|  32768 |     4 |        90.287 |         13.050 |   6.92× |        16× |
|  65536 |     4 |       353.512 |         26.350 |  13.42× |        32× |
| 131072 |     4 |      1404.720 |         53.615 |  26.20× |        64× |

The gap between speedup and FLOP ratio is kernel efficiency left on the table.

Re-measured after the `block_q=1` work, which cost `block_q=64` a few percent (3.78× →
3.62× at 16k): the transpose went 0.08 → 0.36 ms for the chunk-prefix pass that replaced
the sort. That is a deliberate trade — it removed 37 ms at `block_q=1` — but it is not
free at the default, and quoting the old numbers would hide it.

**Memory is independent of topk** (S=16384, group 4) — the fusion check:

| topk | sparse fb (ms) | peak MB |
|-----:|---------------:|--------:|
|    8 |          4.313 |    1158 |
|   16 |          6.293 |    1158 |
|   32 |         10.164 |    1159 |
|   64 |         17.128 |    1159 |

Latency scales 5.2× while peak memory moves by 1 MB.

**Selection-path overhead** (group 4, `benchmarks/bench_selection.py`) — the price
the frontend adds on top of the attention win:

| policy | seqlen | state | select | transpose | total | % of step | peak MB | unfused score matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block_topk |  16384 | 0.028 | 0.150 | 0.081 | 0.258 | 4.3% | 1159 |   2 MB |
| block_topk |  65536 | 0.051 | 1.386 | 0.083 | 1.519 | 6.0% | 4638 |  32 MB |
| block_topk | 131072 | 0.096 | 5.010 | 0.080 | 5.186 | 9.6% | 9275 | 128 MB |
| quest      |  65536 | 0.096 | 1.856 | 0.081 | 2.032 | 7.8% | 4642 |  32 MB |
| streaming  |  65536 | 0.007 | 0.046 | 0.078 | 0.130 | 0.8% | 4633 |  32 MB |

Peak memory tracks the attention tensors, not the score matrix, which is the
evidence that scoring stayed in registers.

**The tiling that made that true.** The first working version sized the scorer's
state tile at the padded `Nkv`, i.e. `[Nkv, D]` fp32 = 512 KB per program at 64k —
far past a register file, so every program spilled to local memory. Selection cost
18 ms at 64k (43% of the step) for `block_topk` and 60 ms (72%) for `quest`, while
`streaming` — the one policy with no `Dot`, hence no state tile — stayed flat at
0.04 ms, which is what localized it. Tiling the KV axis at `TILE_N=64` cut select
by 13× (`block_topk`) and 32× (`quest`). Guarded by
`test_select_state_tile_is_bounded` and `test_select_cost_grows_sublinearly_in_seqlen`.

Worth noting the tests could not have caught this: the pattern was bit-identical
either way. Only the benchmark saw it.

## The three algorithms

| algorithm | state per KV block | scoring |
|---|---|---|
| `block_topk` | 1 centroid | `<q̄, centroid>` |
| `quest` | whole-block key max + min | exact per-channel envelope bound |
| `lserve` | max + min **per 16-token sub-block** | envelope bound, max over sub-blocks |

`lserve` is the sub-block refinement: a whole-block envelope widens with every
unrelated token in the block, so the bound degrades exactly where the block is
heterogeneous. Sub-block envelopes each cover fewer tokens and are therefore tighter,
and taking the max over runs keeps the property that a block needs only **one** good
region to survive. `lserve_centroid` is the controlled comparison — same sub-block
granularity, centroid instead of envelope — so a difference between them is
attributable to the summary rather than to the granularity.

Two ops were added for this and are exact rather than approximations:

* **`Envelope`** — `sum_d max(q_d·M_d, q_d·m_d)`. The per-channel endpoint choice must
  happen *before* the reduction over `D`; `Dot(kmax) + Dot(kmin)` collapses `D` first
  and computes a different, looser quantity. `quest` used that proxy until this op
  existed, and is now exact.
* **`Field(sub_block=…)`** — keep one summary per run of N tokens instead of one per
  block.

## Measured (B200, torch 2.11.0+cu130, bf16, B=1, Hq=32, Hkv=8, D=128, group 4, topk=16)

Full step, selection included. `speedup` is versus SDPA-flash fwd+bwd:

| seqlen | algorithm | fwd ms | bwd ms | e2e ms | state | select | tpose | attn | fwd MB | e2e MB | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32768 | dense (FA4) | 11.13 | n/a | n/a | – | – | – | 11.13 | 896 | n/a | – |
| 32768 | dense (SDPA) | 50.93 | 153.46 | 204.39 | – | – | – | 50.93 | 2184 | 3980 | 1.00× |
| 32768 | block_topk | 5.47 | 25.31 | 30.78 | 0.03 | 0.44 | 0.05 | 4.94 | 1163 | 2579 | **6.64×** |
| 32768 | quest | 5.64 | 25.27 | 30.91 | 0.05 | 0.59 | 0.05 | 4.94 | 1165 | 2581 | 6.61× |
| 32768 | lserve | 9.60 | 25.32 | 34.93 | 0.16 | 1.94 | 0.05 | 7.45 | 1177 | 2593 | 5.85× |
| 32768 | lserve_centroid | 5.98 | 25.33 | 31.31 | 0.09 | 0.88 | 0.05 | 4.96 | 1169 | 2585 | 6.53× |
| 65536 | dense (FA4) | 55.08 | n/a | n/a | – | – | – | 55.08 | 1792 | n/a | – |
| 65536 | dense (SDPA) | 200.28 | 605.28 | 805.56 | – | – | – | 200.28 | 4368 | 7960 | 1.00× |
| 65536 | block_topk | 14.10 | 48.88 | 62.98 | 0.06 | 1.40 | 0.05 | 12.59 | 2327 | 5159 | **12.79×** |
| 65536 | quest | 14.52 | 49.12 | 63.64 | 0.10 | 2.02 | 0.05 | 12.34 | 2331 | 5163 | 12.66× |
| 65536 | lserve | 28.80 | 48.86 | 77.66 | 0.32 | 18.43 | 0.05 | 10.00 | 2355 | 5187 | 10.37× |
| 65536 | lserve_centroid | 18.57 | 48.85 | 67.43 | 0.17 | 5.77 | 0.05 | 12.59 | 2339 | 5171 | 11.95× |
| 131072 | dense (FA4) | 219.55 | n/a | n/a | – | – | – | 219.55 | 3584 | n/a | – |
| 131072 | dense (SDPA) | 793.68 | 2403.26 | 3196.94 | – | – | – | 793.68 | 8736 | 15920 | 1.00× |
| 131072 | block_topk | 32.80 | 98.86 | 131.66 | 0.10 | 9.95 | 0.06 | 22.69 | 4653 | 10317 | **24.28×** |
| 131072 | quest | 38.16 | 102.13 | 140.29 | 0.18 | 15.16 | 0.06 | 22.75 | 4661 | 10325 | 22.79× |
| 131072 | lserve | 105.40 | 99.05 | 204.44 | 0.62 | 81.48 | 0.06 | 23.24 | 4709 | 10373 | 15.64× |
| 131072 | lserve_centroid | 55.27 | 99.29 | 154.55 | 0.32 | 31.98 | 0.06 | 22.91 | 4677 | 10341 | 20.69× |

**Forward-only vs flash-attention-4**, the fastest dense forward available here:

| seqlen | block_topk | quest | lserve | lserve_centroid | SDPA |
|---:|---:|---:|---:|---:|---:|
| 32768 | 2.03× | 1.97× | 1.16× | 1.86× | 0.22× |
| 65536 | 3.91× | 3.79× | 1.91× | 2.97× | 0.28× |
| 131072 | 6.69× | 5.75× | 2.08× | 3.97× | 0.28× |

Memory is at parity-to-below dense and **independent of `topk`** — the correct
result. Sparsity reduces compute, not memory: `dK`/`dV` are needed for every KV
token, so nothing can be dropped from the saved tensors.

### Why FA4 carries no backward number

FA4's *forward* compiles in ~5 s and is the fastest dense forward here. Its
*backward* compiles three CuTeDSL kernels with no persistent on-disk cache and was
measured burning **>25 min of CPU without finishing** on a 1024-token case. That is a
property of the beta's compiler, not its runtime speed, so SDPA-flash carries the
fwd+bwd comparison and FA4 is quoted forward-only. `--fa4-bwd` attempts it anyway.

### Selection quality

Speed alone is meaningless — the cheapest policy is one that picks at random. At
seqlen 4096 with needle blocks planted outside the local window (3 seeds):

| topk | policy | mass recall | top-block recall |
|---:|---|---:|---:|
| 8 | streaming | 0.370 | 0.309 |
| 8 | quest | 0.397 | 0.531 |
| 8 | **lserve** | 0.397 | **0.545** |
| 8 | lserve_centroid | 0.398 | 0.624 |
| 8 | block_topk | 0.398 | 0.744 |

`lserve` beats `quest` at every budget, and the mechanism is measurable: the
sub-block envelope bound is **24.8% tighter** (mean gap 15.01 vs 19.95).

Two honest caveats. First, on i.i.d. Gaussian q/k *every* policy — including
`streaming`, which scores nothing — recalls identically (0.3688 at topk=8), because
attention is near-uniform there and block choice cannot matter; that null control is
`--data gaussian`, and it is why the table above uses planted needles. Second, on this
synthetic data plain `block_topk` has the best top-block recall, so the envelope
family is not vindicated on quality here — a real-model trace is needed before
claiming otherwise.

### The loop-order fix the benchmark caught

`lserve` first measured 328 ms forward at 128k — 4× `quest` for 4× the state, with
`select` alone at 305 ms against quest's 15 ms. The breakdown localized it to the
scorer, and the cause was loop nesting: with the GQA-group loop *outside* the
sub-block loop, each `kmax`/`kmin` tile was reloaded once per group member — 32 loads
where 8 suffice, since this kernel is bound by state traffic rather than arithmetic.
Inverting to sub-block-outer cut `select` 8× at 32k and the whole step 3.1× at 128k
(428 → 204 ms). Guarded by `test_subblock_state_loaded_once_per_tile`.

As with the earlier tiling bug, the tests could not have caught this: the emitted
pattern was bit-identical either way.

## Note on Triton 3.6

`kernels/bwd.py` pins its dk/dv loop with `tl.range(..., num_stages=1)`. This is
load-bearing, not tuning: Triton 3.6 miscompiles a loop that has both a
data-dependent trip count and two fp32 accumulators, silently zeroing `dk` for
programs whose CSR segment is shorter than the pipeline depth. Guarded by
`test_dkv_short_segments`.

## Status

Attention core, the frontend/compiler/backend selection path, and three algorithms
(`block_topk`, `quest`, `lserve`) are complete, tested, and measured at 32K/64K/128K.
Not yet built: HF/FSDP integration and a real finetune script (design Phase 4),
varlen packing, and quality evaluation on real model traces rather than synthetic
retrieval data — see `docs/vortex_train_PLAN.md`.
