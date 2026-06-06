---
description: Controlled single-knob ablation — hold a flow fixed, vary ONE knob across values, and report the quality/efficiency/throughput curve. The scientific isolation /iterate doesn't do.
argument-hint: <base-submission.json> <knob> <v1,v2,v3,...> [--task aime24] [--full]
---

Isolate the effect of **one** variable. Take a base submission, sweep a single
knob across the given values (everything else fixed), and produce the curve.
Unlike `/iterate` (orthogonal variants) this is a clean controlled experiment.

## Step 0 — parse + env

`$1` = base config path; `$2` = knob (a JSON field, e.g. `vortex_topk_val`,
`vortex_block_size`, `vortex_layers_skip`, `kv_cache_dtype`,
`vortex_attention_backend`); `$3` = comma-separated values. `--task` (default
aime24); `--full` also runs the end-to-end task. Establish the env first (`python algorithm_scientist/detect_env.py` / `/setup-env`); use its run prefix as `$RUN`.

## Step 1 — generate the variants (one knob changed)

For each value `v`, write `submissions/<tag>/ablate_<knob>_<v>.json` = the base
config with `"<knob>": <v>` (and a unique `vortex_module_name`/path if the knob
is structural — reuse the base `.py` otherwise). Keep the same model, block-size
(unless that's the knob), reserved blocks, etc. Preflight each:
```bash
for v in <values>; do
  $RUN -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('submissions/<tag>/ablate_<knob>_'$v'.json')" && echo "ok $v" || echo "FAIL $v"
done
```

## Step 2 — cheap screen (always)

For each variant:
- **Predicted efficiency** (CPU, no GPU): `python
  algorithm_scientist/research/efficiency.py --config <variant> --seq-len <ctx>
  --batch <B>` → token ratio + end-to-end ceiling.
- **Quality gate (GPU)**: RULER. **Detect free GPUs and spread the variants
  across them** (never hardcode a device); re-detect each launch:
  ```bash
  FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; exit 1; }
  N=${#FREE_GPUS[@]}; i=0
  for v in <values>; do
      gpu=${FREE_GPUS[$((i % N))]}; i=$((i+1))
      CUDA_VISIBLE_DEVICES=$gpu $RUN algorithm_scientist/run_ruler.py \
          --config "submissions/<tag>/ablate_<knob>_${v}.json" &
      (( i % N == 0 )) && wait    # fill all free GPUs, then drain the wave
  done; wait
  ```

Tabulate: `knob value | token_ratio | speedup_ceiling | RULER acc`. Often the
curve already tells the story (e.g. RULER flat until topk drops below X, then
cliffs) — you may stop here.

## Step 3 — end-to-end (only with `--full`, in waves of 4)

If `--full`, run the variants on the task via the `/batch-benchmark` machinery
(groups of 4, agent-decided `TIMEOUT_MIN`, free-GPU waves) to get real
`(mean@16, throughput)`. Add columns to the table.

## Step 4 — report

Print the full curve `knob → token_ratio | ceiling | RULER | [mean@16 | tput]`,
name the **knee** (where quality starts to fall) and the **efficiency sweet
spot**, and append a one-line finding to `algorithm_scientist/memory.md` §2 (and
the research `journal.md` if part of a hypothesis). One knob, one clean curve —
that's the deliverable.
