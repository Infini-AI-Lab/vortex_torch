---
description: Full-stack op work — add a new op, accelerate a slow op, fuse ops, or investigate a theory-vs-practice gap. The shared sub-skill other commands call when they need an op vortex lacks or one that's too slow.
argument-hint: [new|accelerate|fuse|investigate] <description>   (mode optional — inferred from the request)
---

`add-ops` changes the **framework itself** — a new op needs both a Python op
class *and* its kernel, not just a kernel. It is **full-stack** and is the
shared mechanism that `/innovate`, `/iterate`, and `/reproduce-paper` invoke
whenever they hit a missing or slow op. Standalone or called, the work is the
same and is driven by the **`vortex-op-author`** subagent.

## Pick the mode (infer from `$ARGUMENTS` if not given)

1. **new** — the op an agent wants doesn't exist yet. Author it end-to-end.
2. **accelerate** — an existing op is thought to be slow. Speed up its kernel,
   semantics unchanged.
3. **fuse** — replace a chain of ops with one fused op to cut memory traffic.
4. **investigate** — an algorithm looks efficient on paper but isn't fast in
   practice. Profile, find the root cause, then fix (loops back to 1/2/3).

See [AI/workflows/add_op.md](../../AI/workflows/add_op.md) for the per-mode
checklist and [the developer guide](../../AI/developer_guides/developer_guide.md)
§5/§8/§9/§16/§20 for the law.

## Steps

0. Establish the env (`python algorithm_scientist/detect_env.py`; `/setup-env`); adopt its run prefix as `$RUN`. State the mode + a one-sentence goal.
1. **Delegate to the op author.** Invoke
   `Task(subagent_type="vortex-op-author", ...)` with: the mode, the op math /
   the chain to fuse / the algorithm to investigate, the target side
   (indexer/cache), and the model/flow context if called from another loop.
2. **Gate on the subagent's result.** It must return:
   - **new/fuse:** op class + export + codegen + `register.py` entry, and a
     `verify_flow_compilable` torch-parity pass.
   - **accelerate/fuse:** a measured latency or kernel-count win with parity held.
   - **investigate:** a root cause proven by a measurement, and the gap closed
     (or a precise statement of what blocks it).
   For kernel-backed ops (`topk`/`centroid-score`) it benchmarks under
   `vortex_torch/kernels/<name>/` and keeps only recall-preserving speedups.
3. **Prove it's usable.** A real flow that calls the new/fused op passes
   `check_engine_config`. Record a one-line note in the relevant `memory_*.md`.

## When called from another command

`/innovate`, `/iterate`, `/reproduce-paper` invoke the same `vortex-op-author`
subagent mid-loop the moment they discover a needed op is missing or too slow,
then resume their own loop with the new op available. Return control with a
crisp summary (op name, parity, perf delta, flow snippet) so the caller can
continue without re-deriving context.

## Specialist subagents

- **`vortex-math-researcher`** — for a *new* op (Mode 1) or a smarter scoring
  rule: derive the math/bound from first principles, verify it numerically, and
  return an implementable recipe (cache fields + indexer ops).
- **`vortex-kernel-expert`** — for the kernel (Modes 1-4): writes/optimizes the
  Triton/CUDA kernel using the bundled FlashInfer / FlashAttention source as
  reference + the KernelWiki/ncu skills; returns a measured kernel.
- **`algorithm_scientist/research/efficiency.py`** — compute the analytical
  speedup **ceiling** for the config; if a built op's measured throughput is far
  below its ceiling, that's the Mode-4 signal to investigate.

## Profiling note

For Mode 4 (and Mode 2) use the `ncu-report-skill` to profile CUDA kernels
(memory vs compute bound, occupancy, stalls) and `KernelWiki` for
Blackwell/Hopper kernel techniques. Never claim a speedup you didn't measure.
