# vortex_train

> Vendored into `vortex_torch` as a sibling of the inference stack. `vortex_torch` serves
> a sparse-attention flow; `vortex_train` **trains through one**, forward and backward,
> with the same selection contract. The two meet at the budget convention documented in
> `application/sparse_finetune_qwen3/README.md` — vortex_torch's reservations are additive
> (`selected = topk_val + bos + eos`) while `vortex_train`'s `Budget.topk` is the total,
> and getting that wrong silently trains and serves at different sparsity.

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

## Measured (B200, torch 2.11.0+cu130, bf16, Hkv=8, D=128, block=64, topk=16)

Forward+backward vs SDPA flash backend, `fb` = fwd+bwd:

| seqlen | group | dense fb (ms) | sparse fb (ms) | speedup | FLOP ratio |
|-------:|------:|--------------:|---------------:|--------:|-----------:|
|   4096 |     4 |         1.989 |          1.581 |   1.26× |         2× |
|  16384 |     4 |        23.834 |          6.312 |   3.78× |         8× |
|  32768 |     4 |        90.179 |         12.808 |   7.04× |        16× |
|  65536 |     4 |       353.201 |         26.055 |  13.56× |        32× |
| 131072 |     4 |      1401.809 |         53.412 |  26.25× |        64× |

The gap between speedup and FLOP ratio is kernel efficiency left on the table.
The pattern transpose is ~0.08 ms and flat in sequence length.

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
