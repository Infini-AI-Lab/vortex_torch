---
description: (DEBUG ONLY) Run a single-variant AIME24 benchmark directly via python. Forbidden in the standard agent workflow — use /batch-benchmark instead.
argument-hint: <submission-name>
---

> **Debug-only command.** The sanctioned benchmark protocol runs
> exclusively as a batch that fills every local GPU, via
> `/batch-benchmark`. Use this single-variant form ONLY for human
> debugging (e.g. confirming a new flow boots end-to-end). Do
> not chain it as part of an automated workflow.

Resolve `$1` to a config path: try
`submissions/<tag>/$1.json` first, then `submissions/$1.json`,
or treat `$1` as a literal path if it ends in `.json`. The
runner **mirrors** the config's location under `submissions/`
into `summary_submissions/`, so
`submissions/<tag>/batch_3_id5.json` produces
`summary_submissions/<tag>/batch_3_id5/`.

Step 1 — establish the env + pre-flight (CPU-only):
```bash
python algorithm_scientist/detect_env.py >/dev/null 2>&1   # don't assume vortex_v1
RUN="conda run -n vortex_v1 python"        # ← detect_env.py's recommended prefix; substitute if different
CFG=<resolved path to submissions/.../$1.json>
$RUN -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('$CFG')"
```
Refuse to run the benchmark if pre-flight fails.

Step 2 — launch directly. **Detect a free GPU** (never hardcode 0) and pin to
it; capture stdout/stderr to a log file:
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; exit 1; }
STEM=$(basename "$CFG" .json)                             # filename stem only
SUMMARY_REL="${CFG#submissions/}"; SUMMARY_REL="${SUMMARY_REL%.json}"   # tag/stem (or just stem for top-level configs)
LOGDIR="logs/submission/single_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} \
    $RUN algorithm_scientist/run_submission.py --task aime24 --config "$CFG" \
    > "$LOGDIR/${STEM}.out" \
    2> "$LOGDIR/${STEM}.err"
```
The runner writes its summary into
`summary_submissions/${SUMMARY_REL}/<timestamp>__<hash>.json`
and updates `summary_submissions/${SUMMARY_REL}/latest.json`
itself; no further polling needed.

Step 3 — read the result:

- **Success** → Open `summary_submissions/${SUMMARY_REL}/latest.json`
  and print `mean@16`, `pass@16`, `e2e_time`, `total_tokens`,
  `throughput`, and `content_hash`. For run-to-run comparisons:
  `cat summary_submissions/${SUMMARY_REL}/INDEX.jsonl | jq -c
  '{finished_at, content_hash, "mean@16", throughput}'`.
- **Failure** (non-zero exit, or no new `latest.json` was
  written) → Open `$LOGDIR/${STEM}.err` (and `.out` if needed),
  summarise the error in 1-2 sentences, and recommend a fix. Do
  NOT re-launch without pre-flight passing first.
