---
description: Launch the 4-variant batch on the currently-free local GPUs (parallel when N>=4, otherwise sequential waves of N). The only sanctioned benchmark form. Pass exactly 4 submission names.
argument-hint: <name1> <name2> <name3> <name4>   (always 4 variants; parallelism = min(N_free_gpus, 4))
---

Run the **4-variant batch** against the AIME24 benchmark. **Batch
size is fixed at 4.** Parallelism depends on free-GPU count `N`:

- `N >= 4` → all 4 variants run in parallel, one per GPU.
- `0 < N < 4` → variants run in **waves of `N`** on the available
  GPUs (sequential fallback). With `N = 1` this is fully serial;
  `N = 2` runs `2 + 2`; `N = 3` runs `3 + 1`.
- `N == 0` → wait, do not launch.

This is the *only* benchmark command sanctioned by the protocol;
single-variant runs are debug-only.

**Task / model.** Defaults to AIME24 with whatever `model_path` each JSON
declares. To target another task set `TASK_ARG` before the launch loop:
`TASK_ARG="--task aime25"` (built-ins: aime24/25/26, amc23) or
`TASK_ARG="--data examples/<task>__<modelslug>.jsonl"` for a non-default model
(regenerate the tokenizer-bound jsonl first with `examples/misc/make_task.py --task
<t> --model <m>` — see [AI/workflows/run_tasks.md](../../AI/workflows/run_tasks.md)).

The user passes **4 submission names** (not JSON paths); this
command expands each `<nameI>` into `submissions/<tag>/<nameI>.json`,
where `<tag>` is the session's agent identifier. For the standard
iterate-loop layout, the names look like
`batch_<x>_id0 batch_<x>_id1 batch_<x>_id2 batch_<x>_id3`.

Step 0 — resolve `<tag>`, detect *free* GPUs (not the physical
count — other users may be sharing this host), then count
arguments:
```bash
python algorithm_scientist/detect_env.py >/dev/null 2>&1   # establish the env (don't assume vortex_v1)
RUN="conda run -n vortex_v1 python"        # ← detect_env.py's recommended prefix; substitute if different
TAG=<your_agent_tag>           # sanitized model name, set once per session
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || {
    echo "no free GPUs — wait, do not launch" >&2; exit 1
}
N=${#FREE_GPUS[@]}
BATCH_SIZE=4
PARALLEL=$N
[ "$PARALLEL" -gt "$BATCH_SIZE" ] && PARALLEL=$BATCH_SIZE
NAMES=($ARGUMENTS)
if [ "${#NAMES[@]}" -ne "$BATCH_SIZE" ]; then
    echo "expected $BATCH_SIZE variants, got ${#NAMES[@]}" >&2
    exit 1
fi
echo "free GPUs: ${FREE_GPUS[*]}  (N=$N, parallel=$PARALLEL, waves=$(( (BATCH_SIZE + PARALLEL - 1) / PARALLEL )))"
```
Refuse cases:
  - `N == 0` (helper exited non-zero): "No free GPUs. Wait, do
    not launch."
  - `${#NAMES[@]} != 4`: "Batches must contain exactly 4
    variants. Got ${#NAMES[@]} — re-design and re-invoke."
Do not silently shrink or pad the batch.

Step 1 — concurrency cap. Only **one** batch may run at a time
on the GPUs you target:
```bash
jobs -r | wc -l
```
If background jobs from a previous batch are still alive, refuse
to launch — tell the user to wait, or to use the wait-time
activities in `algorithm_scientist/memory.md`. (Other users'
processes are already filtered out by `free_gpus.sh`.)

Step 2 — pre-flight every config locally first (cheap, no GPU):
```bash
for name in "${NAMES[@]}"; do
    $RUN -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/${name}.json')" \
        || echo "[preflight] FAILED: ${TAG}/${name}"
done
```
Refuse to launch any submission whose preflight failed — fix the
failing variant first.

Step 3 — RULER pre-filter (quick quality gate, ≥ 0.85). Run
`algorithm_scientist/run_ruler.py` on each variant sequentially on
one free GPU. Any variant scoring below **0.85 accuracy** on
`examples/ruler/validation_4k.jsonl` has structurally broken attention — fix
it (widen `vortex_topk_val`/`vortex_topk_ratio` or revise the
indexer), re-pre-flight, and re-run RULER until all 4 pass before
launching AIME24.
```bash
for name in "${NAMES[@]}"; do
    CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
        $RUN algorithm_scientist/run_ruler.py --config "submissions/${TAG}/${name}.json"
done
```
Results land in `summary_ruler_submissions/<tag>/<name>/latest.json`.
Refuse to launch any variant whose RULER accuracy < 0.85.

Step 4 — launch the 4 variants in waves of `PARALLEL = min(N, 4)`,
each pinned to a *free* GPU index (NOT 0…N-1), with `wait`
between waves so a wave's GPUs are free before the next reuses
them:
**Decide a per-run timeout — you choose it; there is no fixed limit.**
Wall-clock ≈ (questions × 16 trials × output tokens) / throughput, so it scales
with model size, MLA, and harder/longer tasks (and with the 32768 max-new-tokens
budget). Set `TIMEOUT_MIN` to ~1.5× your estimate so it fires only on a genuine
stall; the `timeout` wrapper **enforces** it (you don't hand-kill).
```bash
TIMEOUT_MIN=<your estimate, minutes>      # agent-decided per model+task
LOGDIR="logs/submission/${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
for start in $(seq 0 $PARALLEL $((BATCH_SIZE - 1))); do
    end=$((start + PARALLEL))
    [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
    for y in $(seq $start $((end - 1))); do
        name="${NAMES[$y]}"
        gpu="${FREE_GPUS[$((y - start))]}"
        CUDA_VISIBLE_DEVICES=$gpu timeout ${TIMEOUT_MIN}m \
            $RUN algorithm_scientist/run_submission.py ${TASK_ARG:---task aime24} --config "submissions/${TAG}/${name}.json" \
            > "$LOGDIR/gpu${gpu}_${name}.out" \
            2> "$LOGDIR/gpu${gpu}_${name}.err" &
    done
    wait
done
```
A child `timeout` killed exits **124** and writes no `latest.json` — treat it as
a timed-out/failed variant.
When `N >= 4` this is one wave of 4 (fully parallel — same
behaviour as the old protocol). When `N < 4` it's
`ceil(4/N)` sequential waves on the available GPUs.
Each child writes its result into
`summary_submissions/<tag>/<name>/<timestamp>__<hash>.json` and
updates `summary_submissions/<tag>/<name>/latest.json` itself.
(The runner mirrors the config's path under `submissions/` into
`summary_submissions/`, so `submissions/<tag>/batch_<x>_id<y>.json`
becomes `summary_submissions/<tag>/batch_<x>_id<y>/...` — per-agent
isolation, no collisions across agents that pick the same stem.)

Step 5 — append a row to `algorithm_scientist/memory.md` §1
*In-flight batches* the moment you launch:
`| <tag> | <batch_id> | <UTC time> | <LOGDIR> | <name1>,…,<name4> | RUNNING |`

Step 6 — while waiting (runtime depends on model + task; the `TIMEOUT_MIN` you
set enforces the cap, longer with `N < 4` due to sequential waves; a child that
exits 124 / leaves no `latest.json` timed out — log it in memory.md §4 and treat
that variant as failed), do NOT idle. Spend the time
reading tutorials / developer guides / source, or designing the
next batch (don't launch — concurrent batches OOM the shared
GPUs), or analysing children that have already produced their
`latest.json`. See AGENTS.md §5d.

Step 7 — once the outer `wait` loop returns (all waves done):

- **Success** → for each `<nameI>`, read
  `summary_submissions/<tag>/<nameI>/latest.json`. Produce a comparison
  table with columns: `name | content_hash | mean@16 | pass@16 |
  throughput | e2e_time`. Identify the **Pareto-non-dominated**
  variants for this batch on the `(throughput, mean@16)` plane —
  a variant is dominated iff some other variant has *both* equal-
  or-higher `throughput` *and* equal-or-higher `mean@16`. Append
  the table + a 1-3 sentence takeaway naming each frontier
  variant and where it sits relative to the running §5 best, to
  `algorithm_scientist/memory.md` §2 *Completed batches*; remove
  the row you added in step 5 from §1.
- **Failure** → any child whose process exited non-zero (or
  whose `latest.json` is missing / older than `$LOGDIR`'s
  timestamp) wrote its traceback to
  `$LOGDIR/gpu<i>_<name>.err`. Open the failing child's `.err`,
  summarise the error, and record the lesson in
  `algorithm_scientist/memory.md` §4 *Anti-patterns*.
