---
description: Catch-all diagnostic — preflight/compile failures, OOM or stalled benchmark children, wrong-env import errors, server-launch issues, missing summaries, anything else. Find the root cause from logs and propose/apply the minimal fix.
argument-hint: <what's wrong + optional path/log/config>
---

The catch-all for everything the other six commands don't own. Diagnose from
evidence (logs, tracebacks, configs), state the root cause, then propose or (if
the user wants) apply the **minimal** fix. Don't guess — read the actual error.

## Step 0 — environment sanity (the most common failure)

Don't assume `vortex_v1` exists — detect the right env first:
```bash
python algorithm_scientist/detect_env.py     # recommends a run prefix
RUN="<recommended>"; $RUN -c "import sys, vortex_torch; print(sys.executable)"
```
A wrong/missing env (no such conda env, uv/venv/docker host, GLM needing
transformers≥5, or a broken C-extension build) makes every preflight/benchmark
error — rule it out first. If no env works or the build is broken, that's
**`/setup-env`** (build/repair). GLM-family ⇒ the transformers-5 env.

## Step 1 — classify the symptom and go to the matching playbook

- **Preflight / compile error** (`check_engine_config` raises): read the
  traceback against [AI/AGENTS.md](../../AI/AGENTS.md) hard rules — native torch
  op in a vFlow method, shared op instance, `k`/`v` declared, indexer not ending
  in `topK`/`approxTopK`, cache reduction on `dim=0`, missing `CFill(0.0)` for a
  Save/Load field, or `Save(...)` without `"disable_radix_cache": true`. Suggest
  the one-line fix; `/preflight <name>` to re-check.
- **Benchmark child failed / missing summary:** open the per-child log under
  `logs/submission/<...>/gpu<i>_<stem>.{err,out}`. No new `latest.json` ⇒ it
  crashed; the `.err` has the traceback. OOM ⇒ lower `mem_fraction_static`,
  smaller `vortex_max_seq_lens`, or fp8 KV.
- **Stalled / timed-out child:** the launch loop wraps each child in
  `timeout ${TIMEOUT_MIN}m` (agent-decided per model+task); a child that exits
  **124** / leaves no `latest.json` timed out. `kill %<job>` any straggler,
  record as failed in `memory.md §4`, and check the `.out` for a deadlock/compile
  hang — or just a too-tight `TIMEOUT_MIN` for this model+task (raise it).
- **Server-launch issue:** check `examples/misc/server_launch.sh` — the vortex knobs
  go through a single `--vortex-config '<json>'` (folded into a `VortexConfig`
  in the parent only if `vortex_torch` is imported before `ServerArgs` is built),
  `page_size % vortex_block_size == 0`, and the port/GPU are free.
- **Op / kernel correctness or perf:** this is `/add-ops` territory
  (mode `investigate`); use `ncu`/`nsys`/the side's `benchmark.py`. Hand off to
  the `vortex-op-author` subagent if a framework change is needed.
- **Model won't load / boots wrong:** hand off to `/support-model <hf-id>`.

## Step 2 — reproduce minimally, fix, verify

Build the smallest repro (one preflight call, one short generation, one kernel
micro-bench), confirm the failure, apply the minimal fix, and re-run the same
repro to prove it's resolved. Profiling toolbox (ncu/nsys/sglang
`bench_one_batch`/`bench_serving`, repo `run_submission.py`/`run_ruler.py`) is in
[AI/workflows/add_op.md](../../AI/workflows/add_op.md).

## Output

`symptom | root cause (with the log line that proves it) | fix applied/proposed
| verification`. Record non-obvious failures in `algorithm_scientist/memory.md`
§4 so the next run avoids them.
