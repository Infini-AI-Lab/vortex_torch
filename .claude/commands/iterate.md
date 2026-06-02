---
description: Autonomous iterate loop over a model + task — design 4 variants, preflight, RULER, run the task, wait, analyse, repeat. Model and task are inputs.
argument-hint: [--model <hf-id>] [--task aime24|aime25|aime26|amc23|<file.jsonl>] [--max-iterations N]
---

You are running the **vortex_torch iterate loop** autonomously. Execute each
step in sequence; do not ask for confirmation between steps.

## Step 0 — parse args, activate env

Parse `$ARGUMENTS`:
- `--model <hf-id>` (default: the bundled `Qwen/Qwen3-1.7B`).
- `--task <name|file>` (default: `aime24`). Built-in: aime24, aime25, aime26,
  amc23. A `*.jsonl` value is treated as a custom math dataset.
- `--max-iterations N` (default: 3).

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate vortex_v1
python -c "import sys; print(sys.executable)"
```
If the model is GLM-family, use `vortex_glm` instead (see
[AI/workflows/support_model.md](../../AI/workflows/support_model.md)).

## Step 1 — verify the model is supported, prepare task data

1. **Model support** (once): `python algorithm_scientist/support_model.py <model>`.
   If exit ≠ 0, stop and tell the user to run `/support-model <model>` first.
   MLA models (DeepSeek/GLM) require `vFlowMLA` flows.
2. **Task data is tokenizer-bound** — see
   [AI/workflows/run_tasks.md](../../AI/workflows/run_tasks.md). For the default
   model + a built-in task, `examples/<task>.jsonl` already exists. For any
   other model, regenerate it:
   ```bash
   python examples/make_task.py --task <task> --model <model> \
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

For variants whose idea is a new *scoring/selection* rule, consider an **offline
recall screen first** (cheap, no GPU) — capture a trace and test the rule against
the baselines with the recall harness before committing a slot
([AI/workflows/research_toolkit.md](../../AI/workflows/research_toolkit.md)). The
full survey→analyze→screen→iterate loop is `/research`.

## Step 4 — write 8 files + preflight (CPU)

`submissions/$TAG/batch_${BATCH}_id{0..3}.{py,json}`, `@register` globally unique,
`model_path` = the chosen model. `BATCH=$(ls submissions/$TAG/batch_*_id0.json 2>/dev/null | wc -l)`.
```bash
for y in 0 1 2 3; do
  python -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/${TAG}/batch_${BATCH}_id${y}.json')" && echo "ok id$y" || echo "FAIL id$y"
done
```
Fix every failure before continuing.

## Step 5 — RULER gate (≥0.85), then launch the task

Detect free GPUs and **spread the 4 RULER runs across however many are free**
(re-detect here; don't hardcode):
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPUs — wait"; exit 1; }
N=${#FREE_GPUS[@]}
for y in 0 1 2 3; do
  gpu=${FREE_GPUS[$((y % N))]}
  CUDA_VISIBLE_DEVICES=$gpu python algorithm_scientist/run_ruler.py \
    --config "submissions/${TAG}/batch_${BATCH}_id${y}.json" &
  (( (y+1) % N == 0 )) && wait
done; wait
```
Any variant < 0.85 has broken attention — fix, re-preflight, re-RULER.

**Decide a per-run timeout — YOU choose it; there is no fixed limit.** Wall-clock
≈ (questions × 16 trials × expected output tokens) / throughput, so it grows with
model size, MLA, and harder/longer tasks. Estimate it (calibrate from the RULER
gate's observed speed, or a 1-trial probe if unsure), then set `TIMEOUT_MIN` to
~1.5× your estimate for headroom so it fires only on a genuine stall — e.g. a
small (≤2B) model on aime24 ≈ 60–90 min; larger models / aime25/26 / amc23 / MLA
≈ 2–4×. The `timeout` wrapper **enforces** it (you don't hand-kill).

Re-detect free GPUs, then run the task in waves of `PARALLEL=min(N,4)`. Use
`--task <task>` for the default model/built-in jsonl, or `--data $DATA` for a
regenerated one:
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "hard wait"; exit 1; }
N=${#FREE_GPUS[@]}; BATCH_SIZE=4; PARALLEL=$N; [ "$PARALLEL" -gt 4 ] && PARALLEL=4
TIMEOUT_MIN=<your estimate, minutes>        # agent-decided per model+task
LOGDIR="logs/submission/${TAG}_batch_${BATCH}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$LOGDIR"
RUN_DATA_ARG="--task <task>"        # or: RUN_DATA_ARG="--data $DATA"
for start in $(seq 0 $PARALLEL $((BATCH_SIZE-1))); do
  end=$((start+PARALLEL)); [ "$end" -gt "$BATCH_SIZE" ] && end=$BATCH_SIZE
  for y in $(seq $start $((end-1))); do
    cfg="submissions/${TAG}/batch_${BATCH}_id${y}.json"; gpu="${FREE_GPUS[$((y-start))]}"; stem=$(basename "$cfg" .json)
    CUDA_VISIBLE_DEVICES=$gpu timeout ${TIMEOUT_MIN}m \
        python algorithm_scientist/run_submission.py $RUN_DATA_ARG --config "$cfg" \
        > "$LOGDIR/gpu${gpu}_${stem}.out" 2> "$LOGDIR/gpu${gpu}_${stem}.err" &
  done
  wait
done
```
Add a memory.md §1 RUNNING row (with your `TIMEOUT_MIN`) the moment you launch.
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
