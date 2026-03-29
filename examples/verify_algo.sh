#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=7

sparse_algos=(
  "block_sparse_attention"
  "nsa"
  "fsa"
  "flash_moba"
)

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

 for algo in "${sparse_algos[@]}"; do
   OUTFILE="${RESULTS_DIR}/${algo}_bf16_${TIMESTAMP}.log"
   echo ">>> Running verify_algo.py with --vortex-module-name ${algo} --kv-cache-dtype bf16"
   echo ">>> Saving results to ${OUTFILE}"
   { time python verify_algo.py \
     --trials 8 \
     --topk-val 30 \
     --vortex-module-name "${algo}" \
     --model-name Qwen/Qwen3-1.7B \
     --topk-type naive \
     --mem 0.7 ; } \
     2>&1 | tee "${OUTFILE}"
 done