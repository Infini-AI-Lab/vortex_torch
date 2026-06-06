# Adding, accelerating, fusing, and investigating ops

`add-ops` is **full-stack** and covers **four modes**. The authoritative
reference is
[`AI/developer_guides/developer_guide.md`](../developer_guides/developer_guide.md):
read **§5 (ops & dispatch)**, **§8 (codegen)**, **§9 (kernel template)**, **§16
(extending)**, **§20 (worked example — adding an op end-to-end)** before writing.
This file is the checklist; the developer guide is the law.

> Side rules: cache-side reductions are `dim ∈ {1,2}` only; cross-block
> (`dim=0`) is indexer-side. `forward_indexer` ends in `topK`/`approxTopK`.

---

## Mode 1 — author a NEW op (the op an agent needs doesn't exist yet)

A vortex op is callable from a vFlow only if **all four** pieces exist. Mirror
the closest existing op (`vortex_torch/indexer/reduce.py`, `elementwise.py`, or
the cache mirror).

1. **Op class** — `vOp` subclass in `vortex_torch/indexer/<file>.py` or
   `vortex_torch/cache/<file>.py`; declares its output `vTensor` shape/format at
   profile time.
2. **Export** from that side's `__init__.py`.
3. **Codegen** — `generate_<op>_impl(...)` in `.../compiler/triton_impl/<op>.py`
   + a `(op_class, Schedule)` entry in `compiler/triton_impl/register.py`; use
   `dtype_cast.py` (clamp-before-cast for fp8).
4. **Verify** — a tiny flow using it passes `verify_flow_compilable`
   (`vortex_torch/flow/verify.py`) with torch-parity. No perf claim before parity.

## Mode 2 — accelerate an existing op (an agent thinks it's slow)

Keep the op's **semantics and parity** fixed; change only its kernel.

1. Establish a baseline: micro-benchmark the current kernel (and, if it's
   kernel-backed, the reference) on representative shapes.
2. Profile to find the bottleneck (see Mode 4 — don't guess).
3. Rewrite/tune the codegen or the hand kernel (tiling, vectorized loads,
   fewer gathers, fp8, better occupancy). For `topk`/`centroid-score` use the
   `kernels/<name>/` loop below.
4. Re-verify parity, then report the latency delta and *why* it improved.

## Mode 3 — fuse several ops into one faster op

Fusion removes intermediate memory traffic by emitting one kernel for a chain
that today emits several. Either confirm the chain lands in one subgraph
(developer_guide §7.3/§8) or author an explicit fused op whose codegen inlines
the chain (like `Add_Mul`). Always show before/after: **parity preserved** +
kernel-count or latency win.

## Mode 4 — investigate theory-vs-practice gaps (looks efficient, isn't)

An algorithm can be FLOP/IO-optimal on paper yet slow end-to-end. **Diagnose
before changing anything**; the fix usually loops back to Mode 1/2/3.

Common culprits (with this repo's precedents):
- **Gather-bound** — the kernel is memory-bound on scattered loads, not compute
  (the centroid-score kernel; cross-workload caching helped ~6%).
- **Wrong phase dominates** — a different stage eats the budget (Triton dense
  prefill was the ~4× RULER bottleneck, fixed by flashinfer MLAPrefill, ~14×).
- **Conversion / launch overhead** — fp16-scale conversion or per-call kernel
  launches dwarf the math (cf. int8 GEMV scale handling).
- **Occupancy / tail effects** — low SM occupancy, ragged tails, unbalanced
  thread-block scheduling (the workload planner exists for this).

Procedure:
1. Measure end-to-end (`run_submission.py` / RULER / `examples/misc/server_launch.sh`)
   and confirm the *expected* speedup is missing.
2. Localize: time each phase; isolate the kernel. For GPU kernel detail use
   Nsight Compute (the `ncu-report-skill`) — memory vs compute bound, occupancy,
   stalls; Blackwell/Hopper specifics in the `KernelWiki` skill.
3. State the root cause in one sentence with the measurement that proves it.
4. Fix via Mode 1/2/3, re-measure, and confirm the gap closed. Record the
   finding (even a negative result) in the relevant `memory_*.md`.

---

## Kernel-backed ops (topk / centroid-score) — the perf-tuning sub-loop

Some ops dispatch to hand kernels under `vortex_torch/kernels/<name>/`, each
with `benchmark.py`, `dispatcher.py` (JIT compile/cache), `configs/`,
`reports/`, and a persistent `memory_*.md`. Triton or CUDA-C (CUDA via
`cuda_loader`); JIT via the side's `dispatcher.py`; benchmark vs the reference
kernels (`--k` for topk, `--num-q-heads` for centroid-score); reports under
`kernels/<name>/<axis>/reports/<tag>/`; keep only kernels that improve
`(geomean speedup)` while **preserving recall**. This is the loop the retired
`iterate_topk` / `iterate_centroids_score` commands ran — `add-ops` owns it now.

## Profiling & benchmarking toolbox

Use real measurements — never guess. Pick the right tool for the layer:

- **Op / kernel micro-bench:** the side's `vortex_torch/kernels/<name>/benchmark.py`
  (`--k`, `--num-q-heads`) for topk/centroid; for a generic op, a tiny harness
  comparing the generated kernel vs a torch reference on representative shapes.
- **Kernel internals — `ncu` (Nsight Compute):** memory-vs-compute bound,
  occupancy, warp stalls, DRAM/L2 traffic. `ncu --set full -o rep python <repro>`,
  then read it — or use the **`ncu-report-skill`** (B200/sm_100). This is how you
  prove a kernel is gather-bound / launch-bound (Mode 4).
- **Timeline / end-to-end — `nsys` (Nsight Systems):** `nsys profile -o tl
  python <repro>` to see which phase/kernel dominates, kernel-launch gaps, and
  CPU↔GPU stalls — the right tool when "the op is fast but the run isn't."
- **sglang internal benchmarks:** `python -m sglang.bench_one_batch`
  (single-batch latency/throughput), `python -m sglang.bench_serving`
  (serving throughput), and `python -m sglang.bench_latency` where available —
  drive these through a vortex flow to measure end-to-end decode with sparsity on.
- **Repo end-to-end:** `algorithm_scientist/run_submission.py` (task `mean@16`/
  `throughput`), `algorithm_scientist/run_ruler.py` (quality gate),
  `examples/misc/server_launch.sh` (a live server). Blackwell/Hopper kernel
  techniques: the **`KernelWiki`** skill.

Report the tool + the number behind every claim.

## Done = all of

- [ ] (new/fused) op class + export + codegen + register entry
- [ ] verify harness shows torch-parity
- [ ] (accelerate/fuse) measured latency or kernel-count win, parity held
- [ ] (investigate) root cause proven by a measurement + the gap closed
- [ ] a flow uses it and passes `check_engine_config`
- [ ] one-line note in the relevant `memory_*.md`
