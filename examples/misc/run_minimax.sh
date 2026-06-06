#!/usr/bin/env bash
# Experiment driver for MiniMax-M2.7 (GQA, non-MLA) sparse attention on AIME26.
#
# MiniMax-M2.7 requires tp_size=4, so with 8 GPUs/node we run TWO jobs per wave,
# each pinned to a contiguous 4-GPU slot (e.g. {0,1,2,3} + {4,5,6,7}).
#
# Two algorithms, two sweep styles (combined in one matrix):
#
#   Algorithms (both registered in vortex_torch/flow/algorithms.py):
#     - block_sparse_attention       (per-head page-mean centroid routing)
#     - gqa_quest_sparse_attention   (per-kv-group Quest-style max routing)
#
#   Sweep 1 (algo2.sh-style): TOPK sweep at fixed block=16.
#     topk ∈ {61, 93, 125, 157, 253}.
#
#   Sweep 2 (algo2_blocksize.sh-style): block × budget at constant attended budget
#     (topk + RESERVED) * block_size = budget,  RESERVED = bos(1) + eos(2) = 3.
#     block ∈ {32, 64}, budget ∈ {2048, 4096}:
#       block=32: budget 2048 -> topk 61 ; budget 4096 -> topk 125
#       block=64: budget 2048 -> topk 29 ; budget 4096 -> topk  61
#
#   Total: 1 full-attn baseline + 2 modules × (5 topk + 4 (block,budget))
#        = 1 dense + 18 sparse = 19 runs. The baseline runs FIRST so it shares
#        wave 1 with the first sparse job.
#
# Sparse runs use sglang's flashinfer attention backend (MiniMax is NOT MLA) +
# the vortex trtllm indexer (2D block-table); NO layer skip (every layer sparse).
# The dense baseline uses sglang's trtllm_mha kernel (vortex sparsity disabled).
#
# Results -> examples/run_minimax_results.md (raw summaries embedded).
#
# GPU selection (priority order):
#   1. $GPUS env var (space-separated indices), e.g. GPUS="0 1 2 3 4 5 6 7" ./run_minimax.sh
#   2. algorithm_scientist/free_gpus.sh auto-detection, minus $EXCLUDE_GPUS.
# Need >= TP_SIZE (=4) free GPUs to launch anything; with 8 GPUs the script runs
# 2 parallel jobs per wave, with 4 it falls back to 1 job/wave (serial waves).
#
# Usage:
#   ./run_minimax.sh                                  # default model
#   ./run_minimax.sh <hf-model-id>                    # override model
#   DATA=examples/foo.jsonl ./run_minimax.sh <model>  # override data file too
set -uo pipefail
#export HF_HOME=/raid/catalyst/models/

# $1: HF model id (positional, optional). Default keeps the original MiniMax run.
MODEL="${1:-MiniMaxAI/MiniMax-M2.7}"
# DATA defaults match the original MiniMax run; override with $DATA env var if
# you're pointing at a different model (e.g. DATA=examples/math/aime26_glm.jsonl).
DATA="${DATA:-examples/math/aime26_minimax.jsonl}"

# Derive per-model output paths so different models don't share files. Slug =
# model id with '/' -> '-' and lowercased (e.g. MiniMaxAI/MiniMax-M2.7 ->
# minimaxai-minimax-m2.7).
MODEL_SLUG="$(echo "$MODEL" | tr '[:upper:]/' '[:lower:]-')"
SUMMARY_DIR="${SUMMARY_DIR:-summary-${MODEL_SLUG}}"
RESULTS_MD="${RESULTS_MD:-examples/run_minimax_results_${MODEL_SLUG}.md}"
TRIALS=16
SPARSE_MODULES=(block_sparse_attention gqa_quest_sparse_attention)
TP_SIZE=4
JOBS_PER_WAVE=2                    # 8 GPUs / TP_SIZE = 2 parallel jobs per wave

# Sweep 1: fixed block=16, topk sweep (algo2.sh style)
TOPK_VAL_BLK16=(61 93 125 157 253)
# Sweep 2: block × budget (algo2_blocksize.sh style)
BLOCK_SIZES=(32 64)
BUDGETS=(2048 4096)
RESERVED=3                         # bos(1) + eos(2); the +3 in the budget formula

EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"

# block/page are set per job (they vary across sweeps), so NOT in COMMON.
COMMON=(
  --trials "$TRIALS"
  --workload-chunk-size 64 --topk-ratio 0.00
  --model-name "$MODEL" --data-path "$DATA" --mem 0.9
  --generation-max-new-tokens 32768 --max-input-length 4096 --tp-size "$TP_SIZE"
  --summary-dir "$SUMMARY_DIR"
)

# Fixed per-job knobs for MiniMax sparse runs:
#   --attention-backend flashinfer    sglang full attention (MiniMax is GQA, not MLA)
#   --vortex-attention-backend trtllm 2D block-table indexer (per-request page lists)
#   --vortex-impl-backend triton      tensor-core friendly indexer GeMM
#   --vortex-layers-skip              no values => skip none, every layer sparse
SPARSE_FLAGS="--attention-backend flashinfer --vortex-attention-backend trtllm --vortex-impl-backend triton --vortex-layers-skip"

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
    exit 1
  fi
fi

if [ "${#GPU_POOL[@]}" -lt "$TP_SIZE" ]; then
  echo "error: MiniMax needs tp_size=$TP_SIZE — only ${#GPU_POOL[@]} GPU(s) in pool: ${GPU_POOL[*]}" >&2
  exit 1
fi

PARALLEL=$(( ${#GPU_POOL[@]} / TP_SIZE ))   # how many tp=4 jobs fit in one wave
[ "$PARALLEL" -gt "$JOBS_PER_WAVE" ] && PARALLEL="$JOBS_PER_WAVE"
echo "=== GPU pool: ${GPU_POOL[*]}  (tp=$TP_SIZE, parallel=$PARALLEL jobs/wave) ==="

# --- build the job list -------------------------------------------------------
JOBS=()
# Job 0: full-attention dense baseline (run ONCE; topk ignored). verify_algo.py
# special-cases `--vortex-module-name full_attention` to enable_vortex_sparsity=
# False, so this runs sglang's plain trtllm_mha dense path — the reference
# accuracy against which every sparse run below is judged. Prepended so it
# always lands in the first wave alongside the first sparse job (the only ones
# truly "prioritised" here, since the wave width is 2). block=page=32 for trtllm
# kernel compatibility; the sparse SPARSE_FLAGS (vortex-impl-backend, layers-skip,
# vortex-attention-backend, tensor-core) don't apply and are intentionally omitted.
JOBS+=("--vortex-module-name full_attention --attention-backend trtllm_mha --topk-val 253 --block-size 32 --page-size 32")
# Sweep 1: fixed block=16, topk sweep
for algo in "${SPARSE_MODULES[@]}"; do
  for k in "${TOPK_VAL_BLK16[@]}"; do
    JOBS+=("--vortex-module-name $algo --topk-val $k --block-size 16 --page-size 16 $SPARSE_FLAGS")
  done
done
# Sweep 2: block × budget at constant attended budget
for algo in "${SPARSE_MODULES[@]}"; do
  for bs in "${BLOCK_SIZES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
      if (( budget % bs != 0 )); then
        echo "skip: budget $budget not divisible by block $bs" >&2; continue
      fi
      topk=$(( budget / bs - RESERVED ))
      if (( topk < 1 )); then
        echo "skip: block=$bs budget=$budget -> topk=$topk < 1" >&2; continue
      fi
      JOBS+=("--vortex-module-name $algo --topk-val $topk --block-size $bs --page-size $bs $SPARSE_FLAGS")
    done
  done
done

echo "=== plan (${#JOBS[@]} jobs, ~$(( (${#JOBS[@]} + PARALLEL - 1) / PARALLEL )) waves) ==="
for j in "${JOBS[@]}"; do echo "  $j"; done

# --- runner: $1=comma-joined gpu list (4 indices), $2..=verify_algo args ------
run_job() {
  local gpus="$1"; shift
  echo ">>> [GPUs $gpus] $*"
  CUDA_VISIBLE_DEVICES="$gpus" python examples/math/verify_algo.py "${COMMON[@]}" "$@" \
    || echo "!!! FAILED [GPUs $gpus]: $*"
}

# --- launch in waves of $PARALLEL jobs; each job pins TP_SIZE GPUs ------------
NJOBS=${#JOBS[@]}
wave=1
for ((start=0; start<NJOBS; start+=PARALLEL)); do
  end=$((start + PARALLEL - 1))
  (( end >= NJOBS )) && end=$((NJOBS - 1))
  echo "=== wave $wave: jobs $start..$end ==="
  for ((i=0; i<PARALLEL && start+i<NJOBS; i++)); do
    # Slot i of this wave takes GPU_POOL[i*TP_SIZE .. (i+1)*TP_SIZE - 1].
    slot_start=$((i * TP_SIZE))
    gpus="${GPU_POOL[$slot_start]}"
    for ((k=1; k<TP_SIZE; k++)); do
      gpus+=",${GPU_POOL[$((slot_start + k))]}"
    done
    # shellcheck disable=SC2086 — intentional word-split of the job arg string
    run_job "$gpus" ${JOBS[$((start+i))]} &
  done
  wait
  wave=$((wave + 1))
done

# --- render results markdown --------------------------------------------------
python examples/misc/collect_algo2_results.py --summary-dir "$SUMMARY_DIR" --out "$RESULTS_MD"
echo "=== done; see $RESULTS_MD ==="
