---
description: Autonomous iterate loop over a model + task — design 4 variants, preflight, RULER, run the task, wait, analyse, repeat. Model and task are inputs.
argument-hint: [--model <hf-id>] [--task aime24|aime25|aime26|amc23|<file.jsonl>] [--max-iterations N] [--max-gpus N] [--tp K] [--hint "free-form steer"]
---

You are running the **vortex_torch iterate loop** autonomously. Execute each
step in sequence; do not ask for confirmation between steps.

## Step 0 — parse args, activate env

Parse `$ARGUMENTS`:
- `--model <hf-id>` (default: the bundled `Qwen/Qwen3-1.7B`).
- `--task <name|file>` (default: `aime24`). Built-in: aime24, aime25, aime26,
  amc23. A `*.jsonl` value is treated as a custom math dataset.
- `--max-iterations N` (default: 3).
- `--tp K` — **tensor-parallel degree = GPUs per single run** (default: `1`,
  or what `/support-model` reports for the model). Large models need `K>1`
  (e.g. MiniMax-M2.7 229B ⇒ `4`); the model simply won't fit on one GPU.
- `--max-gpus N` — **the maximum number of GPUs THIS agent may use at the same
  time** (default: as many as are free right now). This is *your* concurrency
  budget, set by the user — **not** a count of all GPUs and **not** including
  jobs other people (or other jobs under your own username) are running.
- `--hint "..."` (optional) — **free-form steer for the variant design**. Anything
  after `--hint` (quote it) biases what the batch explores: a direction
  (`"channel sparsity, g3/g7 heavy heads"`), a knob to push (`"fp8 KV + tighter
  topk for throughput"`), an op/paper to try (`"LServe sub-block centroids"`), or
  a constraint (`"stay ≥ full-attn accuracy"`). It shapes Step 3 only; it does
  **not** relax the contract (still 4 variants, ≥1 genuinely novel, RULER ≥0.85,
  every JSON valid). If omitted, design as usual from memory.md + papers/guide.md.

**Shared-cluster rules — internalize these:**
- The cluster is shared. Other users *and* other processes under your own
  username may be using GPUs. `algorithm_scientist/free_gpus.sh` already
  excludes any GPU with a running compute process or `memory.used ≥ 1024 MiB`,
  so it returns only GPUs free for *anyone* right now.
- `--max-gpus` caps how many of those free GPUs **you** occupy concurrently.
  Never run more than `--max-gpus` GPUs' worth of work at once across all your
  children. Each run consumes `--tp` GPUs, so **at most `floor(max_gpus / tp)`
  runs in flight at a time**; the rest go in later waves.
- Re-detect free GPUs at the start of **every** wave (the free set shifts as
  others start/stop). If fewer than `tp` GPUs are usable, wait and retry — do
  not shrink `tp`.

Set these once for the session:
```bash
TP=<--tp value, default 1>          # GPUs per run
MAX_GPUS=<--max-gpus value>         # your simultaneous-GPU budget (default: all free)
```

**Establish the env first** — don't assume `vortex_v1` (see `/setup-env`). Adopt
the recommended **run prefix** as `$RUN` and use it for every python call below:
```bash
python algorithm_scientist/detect_env.py            # recommends a run prefix (GLM ⇒ vortex_glm env)
RUN="conda run -n vortex_v1 python"                 # ← the recommended prefix; substitute if different
$RUN -c "import sys, vortex_torch; print(sys.executable)"
```

## Step 1 — verify the model is supported, prepare task data

1. **Model support** (once): `$RUN algorithm_scientist/support_model.py <model>`.
   If exit ≠ 0, stop and tell the user to run `/support-model <model>` first.
   MLA models (DeepSeek/GLM) require `vFlowMLA` flows.
2. **Task data is tokenizer-bound** — see
   [AI/workflows/run_tasks.md](../../AI/workflows/run_tasks.md). For the default
   model + a built-in task, `examples/<task>.jsonl` already exists. For any
   other model, regenerate it:
   ```bash
   $RUN examples/misc/make_task.py --task <task> --model <model> \
       --output examples/<task>__<modelslug>.jsonl
   ```
   Remember `DATA=examples/<task>__<modelslug>.jsonl` (or the built-in path);
   the runner below takes it via `--data`, or `--task <task>` for the default.

## Step 2 — pick tag, read context

`<tag>` = sanitized model name (e.g. `claude_opus_4_8`); `mkdir -p submissions/$TAG`.
Read (skip if already loaded): [AI/AGENTS.md](../../AI/AGENTS.md),
[AI/tutorials/overview.md](../../AI/tutorials/overview.md) + the five op/program
tutorials, [vortex_torch/flow/algorithms.py](../../vortex_torch/flow/algorithms.py),
[papers/guide.md](../../papers/guide.md) §14/§16, and
[algorithm_scientist/memory.md](../../algorithm_scientist/memory.md). If §1 shows
a batch RUNNING, jump to Step 6 wait-work.

## Step 3 — design the 4-variant batch

Every batch is exactly 4 ORTHOGONAL variants. **≥1 (aim 2) genuinely novel**
(papers/guide.md §16.2/§16.3/§16.4 or an op-set idea — not §16.1 combos, not a
sweep). id2–id3 may be §16.5 sweeps. Each `.json` must set
`"model_path": "<model>"`. Pre-register novelty hypotheses in memory.md §3.

**If `--hint` was given, let it drive this design** — center the batch on the
hinted direction/knob/op/constraint (e.g. spend the novel slot(s) on the hinted
idea and the sweep slots mapping the Pareto curve *around* it). The hint sets the
theme; it does not waive the rules above (still 4 orthogonal variants, ≥1
genuinely novel, valid JSON, RULER ≥0.85). Note in memory.md §3 that the batch
was hint-steered and how.

For variants whose idea is a new *scoring/selection* rule, consider an **offline
recall screen first** (cheap, no GPU) — capture a trace and test the rule against
the baselines with the recall harness before committing a slot
([AI/workflows/research_toolkit.md](../../AI/workflows/research_toolkit.md)). The
full survey→analyze→screen→iterate loop is `/research`.

## Step 4 — write 8 files + preflight (CPU)

`submissions/$TAG/batch_${BATCH}_id{0..3}.{py,json}`, `@register` globally unique,
`model_path` = the chosen model, **and `"tp_size": <TP>`** in each JSON (so RULER
and the task runner both boot the model on the right number of GPUs).
`BATCH=$(ls submissions/$TAG/batch_*_id0.json 2>/dev/null | wc -l)`.
```bash
for y in 0 1 2 3; do
  $RUN -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')" && echo "ok id$y" || echo "FAIL id$y"
done
```
Fix every failure before continuing.

## Step 5 — RULER gate (≥0.85), then launch the task

**Warm the shared JIT first (once, serial).** The decode-planner and prefill
kernels are shared across all 4 variants; if the variants compile them in
parallel they race on torch's build lock (slow, can wedge). Compile them once in
a single process before any parallel launch — children then hit a warm cache:
```bash
$RUN -c "import vortex_torch; print('jit warmup:', vortex_torch.warmup_jit())"
```

Both RULER and the task allocate **`TP` GPUs per variant**, run at most
**`floor(min(free, MAX_GPUS) / TP)` variants at once**, and **re-detect free
GPUs at the start of every wave**. The reusable allocator (Bash):

```bash
# Launch the 4 variants, TP GPUs each, capped at MAX_GPUS concurrent, in waves.
# $1 = a shell function name that takes "<id> <comma-separated-gpus>" and starts
#      one backgrounded child pinned to those GPUs.
run_batch_tp () {
  local launch_one="$1" BATCH_SIZE=4 y=0
  while [ "$y" -lt "$BATCH_SIZE" ]; do
    local FREE USABLE NU PAR launched gpus
    FREE=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; sleep 60; continue; }
    USABLE=( "${FREE[@]:0:$MAX_GPUS}" ); NU=${#USABLE[@]}; PAR=$(( NU / TP ))
    if [ "$PAR" -lt 1 ]; then echo "need $TP GPUs for one run; only $NU usable now — wait"; sleep 60; continue; fi
    launched=0
    while [ "$launched" -lt "$PAR" ] && [ "$y" -lt "$BATCH_SIZE" ]; do
      gpus=$(IFS=,; echo "${USABLE[*]:$((launched*TP)):$TP}")   # TP indices for this variant
      "$launch_one" "$y" "$gpus"
      launched=$((launched+1)); y=$((y+1))
    done
    wait   # finish this wave before re-detecting for the next
  done
}
```

**RULER gate** — `run_ruler.py` reads `tp_size` from each variant's JSON:
```bash
ruler_one () { CUDA_VISIBLE_DEVICES="$2" $RUN algorithm_scientist/run_ruler.py \
    --config "submissions/${TAG}/batch_${BATCH}_id${1}.json" \
    > "logs/ruler_id${1}.out" 2>&1 & }
run_batch_tp ruler_one
```
Any variant < 0.85 has broken attention — fix, re-preflight, re-RULER.

**Decide a per-run timeout — YOU choose it; there is no fixed limit.** Wall-clock
≈ (questions × 16 trials × expected output tokens) / throughput, so it grows with
model size, MLA, and harder/longer tasks. Estimate it (calibrate from the RULER
gate's observed speed, or a 1-trial probe if unsure), then set `TIMEOUT_MIN` to
~1.5× your estimate for headroom so it fires only on a genuine stall — e.g. a
small (≤2B) model on aime24 ≈ 60–90 min; larger models / aime25/26 / amc23 / MLA
≈ 2–4×. The `timeout` wrapper **enforces** it (you don't hand-kill).

**Launch the task** — pass `--tp $TP` (overrides the JSON), `--task <task>` for
the default model/built-in jsonl or `--data $DATA` for a regenerated one:
```bash
TIMEOUT_MIN=<your estimate, minutes>        # agent-decided per model+task
RUN_DATA_ARG="--task <task>"                 # or: RUN_DATA_ARG="--data $DATA"
LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"
task_one () {
  local id="$1" gpus="$2" cfg="submissions/${TAG}/batch_${BATCH}_id${1}.json"
  local stem; stem=$(basename "$cfg" .json)
  CUDA_VISIBLE_DEVICES="$gpus" timeout ${TIMEOUT_MIN}m \
      $RUN algorithm_scientist/run_submission.py --tp $TP $RUN_DATA_ARG --config "$cfg" \
      > "$LOGDIR/gpu${gpus//,/_}_${stem}.out" 2> "$LOGDIR/gpu${gpus//,/_}_${stem}.err" &
}
run_batch_tp task_one
```
Add a memory.md §1 RUNNING row (with your `TIMEOUT_MIN`, `TP`, `MAX_GPUS`) the moment you launch.
A child that `timeout` killed exits **124** and writes no `latest.json` — treat
it as a timed-out/failed variant (record in §4; consider a larger `TIMEOUT_MIN`,
fewer trials, or a lighter flow next time).

## Step 6 — wait and do productive work (the `timeout` you set enforces the cap)

On each poll do ONE of: (a) read the next priority file → memory.md §7; (b)
invent two §16 hypotheses; (c) design+preflight the next batch (don't launch —
concurrent batches OOM); (d) analyse children whose `latest.json` landed.

## Step 7 — analyse, update memory.md, check budget

Read all 4 summaries under the task's summary dir
(`<summary_dir>/$TAG/batch_${BATCH}_id<y>/latest.json`). Table:
`variant | hash | RULER | mean@16 | pass@16 | throughput | e2e`. Mark
Pareto-non-dominated variants on `(throughput, mean@16)`. Append to memory.md
§2; clear the §1 row; update §3/§4/§5. If batches launched this session
≥ `MAX_ITER`, write a final summary to §8 and **stop**; else go to Step 3.
