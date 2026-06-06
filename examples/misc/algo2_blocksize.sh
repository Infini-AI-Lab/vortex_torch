#!/usr/bin/env bash
# Block-size study for GLM-4.7-Flash MLA sparse attention (AIME26, mean@16),
# cuda_mla decode. Sibling of algo2_cuda_mla.sh; NO full_attention baseline.
#
# Question: at a FIXED attended-token budget, how do larger blocks trade off
# accuracy vs throughput? We hold the effective budget constant across block
# sizes via
#
#       (topk_val + RESERVED) * block_size = budget          RESERVED = 3
#
# where RESERVED = vortex_block_reserved_bos(1) + vortex_block_reserved_eos(2),
# i.e. the always-attended BOS/EOS blocks. block_size == page_size throughout.
#
# Matrix (2 modules x 2 block sizes x 2 budgets = 8 runs):
#   modules : rope_aware_block_sparse_mla , lserve_centroid_mla
#   block   : 32 , 64           (block_size == page_size)
#   budget  : 2048 , 4096       => topk = budget/block - 3
#       block=32: budget 2048 -> topk 61 ; budget 4096 -> topk 125
#       block=64: budget 2048 -> topk 29 ; budget 4096 -> topk  61
# block_size=16 is already covered in $SUMMARY_DIR (topk 125 / 253), so it is
# not re-run here; the collector shows those rows alongside these for comparison.
#
# Each run uses cuda_mla decode + Triton tensor-core indexer, no layer skip
# (every layer sparse) -- identical to algo2_cuda_mla.sh except block/page/topk.
#
# Parallelism: up to 4 runs at a time, one per GPU, in sequential waves.
# Results -> examples/misc/algo2_blocksize_results.md (raw summaries embedded).
#
# GPU selection (in priority order):
#   1. $GPUS env var (space-separated indices), e.g.  GPUS="1 5 6 7" ./algo2_blocksize.sh
#   2. algorithm_scientist/free_gpus.sh auto-detection, minus $EXCLUDE_GPUS.
# If neither yields a usable GPU the script errors out (it never guesses).
# No GPUs are excluded by default; set $EXCLUDE_GPUS (space-separated indices)
# to skip known-bad/busy ones. Parallelism is capped at 4 even if more are free.
set -uo pipefail
#export HF_HOME=/raid/catalyst/models/

MODEL="GLM-4.7-Flash"
DATA="examples/math/aime26_glm.jsonl"
SUMMARY_DIR="summary-glm4.7-flash"
RESULTS_MD="examples/misc/algo2_blocksize_results.md"
TRIALS=16
SPARSE_MODULES=(rope_aware_block_sparse_mla)
BLOCK_SIZES=(32 64)
BUDGETS=(2048 4096)
RESERVED=3                 # vortex_block_reserved_bos(1) + eos(2); the +3 in the formula
MAX_PARALLEL=1
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"

# block/page are set per job (they vary), so they are NOT in COMMON.
COMMON=(
  --trials "$TRIALS"
  --workload-chunk-size 64 --topk-ratio 0.00
  --model-name "$MODEL" --data-path "$DATA" --mem 0.9
  --generation-max-new-tokens 32768 --max-input-length 4096 --tp-size 1
  --summary-dir "$SUMMARY_DIR" --skip-already-finished-check
)

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
  # no fallback: refuse to guess GPU indices when detection finds nothing usable
  if [ "${#GPU_POOL[@]}" -eq 0 ]; then
    echo "error: no free GPUs detected — set \$GPUS to run on specific indices" >&2
    exit 1
  fi
fi

PARALLEL=${#GPU_POOL[@]}
[ "$PARALLEL" -gt "$MAX_PARALLEL" ] && PARALLEL="$MAX_PARALLEL"
echo "=== GPU pool: ${GPU_POOL[*]}  (parallel=$PARALLEL) ==="

# --- build the job list: module x block_size x budget -------------------------
# topk derived so (topk + RESERVED) * block == budget (constant attended budget).
JOBS=()
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
      JOBS+=("--vortex-module-name $algo --topk-val $topk --block-size $bs --page-size $bs --attention-backend cuda_mla --vortex-impl-backend triton --vortex-use-tensor-core --vortex-layers-skip")
    done
  done
done

echo "=== plan (${#JOBS[@]} jobs) ==="
for j in "${JOBS[@]}"; do echo "  $j"; done

run_job() {  # $1=gpu, $2..=verify_algo args; do not abort the sweep on failure
  local gpu="$1"; shift
  echo ">>> [GPU $gpu] $*"
  CUDA_VISIBLE_DEVICES="$gpu" python examples/math/verify_algo.py "${COMMON[@]}" "$@" \
    || echo "!!! FAILED [GPU $gpu]: $*"
}

# --- launch in waves of $PARALLEL, one job per GPU ----------------------------
NJOBS=${#JOBS[@]}
wave=1
for ((start=0; start<NJOBS; start+=PARALLEL)); do
  echo "=== wave $wave: jobs $start..$((start + PARALLEL - 1 < NJOBS - 1 ? start + PARALLEL - 1 : NJOBS - 1)) ==="
  for ((i=0; i<PARALLEL && start+i<NJOBS; i++)); do
    # shellcheck disable=SC2086 — intentional word-split of the job arg string
    run_job "${GPU_POOL[$i]}" ${JOBS[$((start+i))]} &
  done
  wait
  wave=$((wave + 1))
done

# --- render results markdown --------------------------------------------------
python examples/misc/collect_algo2_results.py --summary-dir "$SUMMARY_DIR" --out "$RESULTS_MD"
echo "=== done; see $RESULTS_MD ==="
