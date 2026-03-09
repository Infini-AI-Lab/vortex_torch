#!/usr/bin/env bash
set -e
# export CUDA_VISIBLE_DEVICES=0

sparse_algos=(
  "block_sparse_attention"
)

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

 for algo in "${sparse_algos[@]}"; do
   OUTFILE="${RESULTS_DIR}/${algo}_fp8_${TIMESTAMP}.log"
   echo ">>> Running verify_algo.py with --vortex-module-name ${algo} --kv-cache-dtype fp8_e4m3"
   echo ">>> Saving results to ${OUTFILE}"
   { time python verify_algo.py \
     --trials 8 \
     --topk-val 30 \
     --vortex-module-name "${algo}" \
     --model-name Qwen/Qwen3-1.7B \
     --kv-cache-dtype fp8_e4m3 \
     --mem 0.7 ; } \
     2>&1 | tee "${OUTFILE}"
 done
