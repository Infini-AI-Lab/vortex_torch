#!/usr/bin/env bash
# Experiment driver for GLM-4.7-Flash MLA sparse attention (AIME26, mean@16).
#
# Matrix (11 runs total):
#   1. full_attention  — dense baseline, run ONCE (topk is ignored). Uses the
#      dense flashinfer MLA path because the vortex sparse backends require
#      enable_vortex_sparsity=True (so full attention can't use cuda_mla).
#   2. rope_aware_block_sparse_mla  } tritondecode + Triton tensor-core indexer,
#   3. lserve_centroid_mla          } NO layer skip (all layers sparse), topk sweep.
#
# Each sparse run uses:
#   --attention-backend triton    hand-CUDA block-table decode (geometry-agnostic)
#   --vortex-impl-backend triton     tensor-core is a Triton-indexer feature ...
#   --vortex-use-tensor-core         ... and requires impl-backend=triton
#   --vortex-layers-skip             (no values) => skip none, all layers sparse
#
# Parallelism: up to 4 runs at a time, one per GPU, in sequential waves. The
# full_attention baseline is job 0 so it always lands in the FIRST wave
# (prioritised — its result is the reference point for every sparse run).
#
# Results land in $SUMMARY_DIR/*.json; collect_algo2_results.py renders them to
# examples/misc/algo2_results.md (with the raw summaries embedded).
#
# GPU selection (in priority order):
#   1. $GPUS env var (space-separated indices), e.g.  GPUS="1 3 4 5" ./algo2.sh
#   2. algorithm_scientist/free_gpus.sh auto-detection, minus $EXCLUDE_GPUS.
# If neither yields a usable GPU the script errors out (it never guesses).
# No GPUs are excluded by default; set $EXCLUDE_GPUS (space-separated indices)
# to skip known-bad/busy ones, e.g. EXCLUDE_GPUS="0 2" ./algo2.sh. Parallelism
# is capped at 4 even if more GPUs are free.
set -uo pipefail
#export HF_HOME=/raid/catalyst/models/

MODEL="GLM-4.7-Flash"
DATA="examples/math/aime26_glm.jsonl"
SUMMARY_DIR="summary-glm4.7-flash"
TRIALS=16
TOPK_VAL=(61 93 125 157 253)
SPARSE_MODULES=(rope_aware_block_sparse_mla)
MAX_PARALLEL=1
EXCLUDE_GPUS="${EXCLUDE_GPUS:-}"

COMMON=(
  --trials "$TRIALS"
  --page-size 16 --block-size 16 --workload-chunk-size 64 --topk-ratio 0.00
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

# --- build the job list (full_attention first => prioritised, first wave) -----
JOBS=()
# 1. full-attention dense baseline (run once; topk ignored).
#    Use sglang's plain Triton backend for dense: the flashinfer dense-MLA path
#    has an illegal-memory-access / page-index bug on this pool. block/page=32
#    (dense isn't compatible with block_size=16).
#JOBS+=("--vortex-module-name full_attention --attention-backend triton --topk-val 253 --block-size 32 --page-size 32")
# 2/3. sparse modules: triton + tensor-core indexer, all layers sparse.
for algo in "${SPARSE_MODULES[@]}"; do
  for k in "${TOPK_VAL[@]}"; do
    JOBS+=("--vortex-module-name $algo --topk-val $k --attention-backend cuda_mla  --vortex-impl-backend triton --vortex-use-tensor-core --vortex-layers-skip")
  done
done

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
python examples/misc/collect_algo2_results.py --summary-dir "$SUMMARY_DIR" --out examples/misc/algo2_results.md
echo "=== done; see examples/misc/algo2_results.md ==="
