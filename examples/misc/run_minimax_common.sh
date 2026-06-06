#!/usr/bin/env bash
# Shared infrastructure for the run_minimax_{0..4}.sh splits. SOURCE THIS — do
# not execute directly. Sourcing performs setup AND builds the full master $JOBS
# list (1 dense baseline + 18 sparse = 19 entries, identical order to the all-in-
# one examples/misc/run_minimax.sh). Each split script then runs a contiguous chunk
# via mm_run_chunk, so the 5 scripts together cover the same 19 jobs as the
# monolithic run.
#
# Calling pattern (mirrored by each split script):
#     source "$(dirname "${BASH_SOURCE[0]}")/run_minimax_common.sh"
#     mm_run_chunk <start> <count>
#
# Positional $1 of the SOURCING script is the summary dir (default
# summary-<model-slug>); $SUMMARY_DIR env is an equivalent override. Positional
# $2 is the HF model id (default MiniMaxAI/MiniMax-M2.7); $MODEL env is an
# equivalent override. Positional $3 is the data path (default
# examples/math/aime26_minimax.jsonl); $DATA env is an equivalent override. Other
# env overrides: $RESULTS_MD, $GPUS, $EXCLUDE_GPUS. Wave width is 2 (8 GPUs /
# tp=4). See examples/misc/run_minimax.sh for the all-in-one version and full design
# rationale.

set -uo pipefail

# $2 (model) wins, then $MODEL env, then the original MiniMax default.
MODEL="${2:-${MODEL:-MiniMaxAI/MiniMax-M2.7}}"
# $3 (data path) wins, then $DATA env, then the default AIME26 set.
DATA="${3:-${DATA:-examples/math/aime26_minimax.jsonl}}"

# Per-model output paths so different models don't clobber each other.
MODEL_SLUG="$(echo "$MODEL" | tr '[:upper:]/' '[:lower:]-')"
# $1 (summary dir) wins, then $SUMMARY_DIR env, then the per-model default.
SUMMARY_DIR="${1:-${SUMMARY_DIR:-summary-${MODEL_SLUG}}}"
RESULTS_MD="${RESULTS_MD:-examples/run_minimax_results_${MODEL_SLUG}.md}"

TRIALS=32
TP_SIZE=4
JOBS_PER_WAVE=2                    # 8 GPUs / TP_SIZE = 2 parallel jobs per wave
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"

# block/page are set per job (they vary), so NOT in COMMON.
COMMON=(
  --trials "$TRIALS"
  --workload-chunk-size 64 --topk-ratio 0.00
  --model-name "$MODEL" --data-path "$DATA" --mem 0.9
  --generation-max-new-tokens 32768 --max-input-length 4096 --tp-size "$TP_SIZE"
  --summary-dir "$SUMMARY_DIR"
)

SPARSE_FLAGS="--attention-backend flashinfer --vortex-attention-backend trtllm --vortex-impl-backend triton --vortex-layers-skip --vortex-use-tensor-core"

# --- resolve the GPU pool -----------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${GPUS:-}" ]; then
  read -r -a GPU_POOL <<< "$GPUS"
else
  if FREE=$("$HERE/algorithm_scientist/free_gpus.sh" 2>/dev/null); then
    read -r -a DETECTED <<< "$FREE"
  else
    DETECTED=()
  fi
  GPU_POOL=()
  for g in "${DETECTED[@]}"; do
    skip=0
    for x in $EXCLUDE_GPUS; do [ "$g" = "$x" ] && skip=1; done
    [ "$skip" -eq 0 ] && GPU_POOL+=("$g")
  done
  if [ "${#GPU_POOL[@]}" -eq 0 ]; then
    echo "error: no free GPUs detected — set \$GPUS to run on specific indices" >&2
    return 1 2>/dev/null || exit 1
  fi
fi

if [ "${#GPU_POOL[@]}" -lt "$TP_SIZE" ]; then
  echo "error: tp_size=$TP_SIZE needed — only ${#GPU_POOL[@]} GPU(s) in pool: ${GPU_POOL[*]}" >&2
  return 1 2>/dev/null || exit 1
fi

PARALLEL=$(( ${#GPU_POOL[@]} / TP_SIZE ))
[ "$PARALLEL" -gt "$JOBS_PER_WAVE" ] && PARALLEL="$JOBS_PER_WAVE"
echo "=== model=$MODEL ; GPU pool: ${GPU_POOL[*]}  (tp=$TP_SIZE, parallel=$PARALLEL jobs/wave) ==="

# --- build the FULL master JOBS list (same order as examples/misc/run_minimax.sh) -
# Index layout (for the split scripts' chunk choices):
#   0           : full-attention dense baseline (trtllm_mha)
#   1..5        : sweep1 block_sparse_attention topk {61,93,125,157,253}, block=16
#   6..10       : sweep1 gqa_quest_sparse_attention topk {61,93,125,157,253}, block=16
#   11..14      : sweep2 block_sparse_attention (block,budget) {(32,2048),(32,4096),(64,2048),(64,4096)}
#   15..18      : sweep2 gqa_quest_sparse_attention (block,budget) {(32,2048),(32,4096),(64,2048),(64,4096)}
# Total: 19 jobs.
JOBS=()
# Job 0: dense baseline. verify_algo.py special-cases module_name=full_attention
# to enable_vortex_sparsity=False; runs sglang's plain trtllm_mha kernel.
JOBS+=("--vortex-module-name full_attention --attention-backend trtllm_mha --topk-val 253 --block-size 32 --page-size 32")

TOPK_VAL_BLK16=(61 93 125 157 253)
SPARSE_MODULES=(block_sparse_attention gqa_quest_sparse_attention)
BLOCK_SIZES=(32 64)
BUDGETS=(2048 4096)
RESERVED=3                         # bos(1) + eos(2)

for algo in "${SPARSE_MODULES[@]}"; do
  for k in "${TOPK_VAL_BLK16[@]}"; do
    JOBS+=("--vortex-module-name $algo --topk-val $k --block-size 16 --page-size 16 $SPARSE_FLAGS")
  done
done
for algo in "${SPARSE_MODULES[@]}"; do
  for bs in "${BLOCK_SIZES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      (( budget % bs == 0 )) || { echo "skip: budget $budget not divisible by block $bs" >&2; continue; }
      topk=$(( budget / bs - RESERVED ))
      (( topk >= 1 )) || { echo "skip: block=$bs budget=$budget -> topk=$topk < 1" >&2; continue; }
      JOBS+=("--vortex-module-name $algo --topk-val $topk --block-size $bs --page-size $bs $SPARSE_FLAGS")
    done
  done
done

# --- runner + wave loop -------------------------------------------------------
run_job() {
  local gpus="$1"; shift
  echo ">>> [GPUs $gpus] $*"
  CUDA_VISIBLE_DEVICES="$gpus" python examples/math/verify_algo.py "${COMMON[@]}" "$@" \
    || echo "!!! FAILED [GPUs $gpus]: $*"
}

# Run a contiguous slice of $JOBS in waves of $PARALLEL. Renders the summary MD
# at the end (incremental: each chunk re-runs the collector against SUMMARY_DIR,
# so the MD grows as more chunks finish).
mm_run_chunk() {
  local start="$1" count="$2"
  local NJOBS=${#JOBS[@]}
  local stop=$(( start + count ))
  (( stop > NJOBS )) && stop=$NJOBS
  if (( start >= stop )); then
    echo "mm_run_chunk: empty slice ($start, count=$count, total=$NJOBS)" >&2; return 0
  fi
  echo "=== chunk: jobs $start..$((stop - 1)) of $NJOBS (this chunk: $((stop - start))) ==="
  for ((idx=start; idx<stop; idx++)); do echo "  [$idx] ${JOBS[$idx]}"; done

  local wave=1 s i k slot_start gpus end
  for ((s=start; s<stop; s+=PARALLEL)); do
    end=$((s + PARALLEL - 1))
    (( end >= stop )) && end=$((stop - 1))
    echo "=== wave $wave: jobs $s..$end ==="
    for ((i=0; i<PARALLEL && s+i<stop; i++)); do
      slot_start=$((i * TP_SIZE))
      gpus="${GPU_POOL[$slot_start]}"
      for ((k=1; k<TP_SIZE; k++)); do
        gpus+=",${GPU_POOL[$((slot_start + k))]}"
      done
      # shellcheck disable=SC2086 — intentional word-split of the job arg string
      run_job "$gpus" ${JOBS[$((s+i))]} &
    done
    wait
    wave=$((wave + 1))
  done

  python examples/misc/collect_algo2_results.py --summary-dir "$SUMMARY_DIR" --out "$RESULTS_MD"
  echo "=== chunk done; results so far -> $RESULTS_MD ==="
}
