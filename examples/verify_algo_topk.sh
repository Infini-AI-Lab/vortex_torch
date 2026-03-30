#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=5

sparse_algos=(
  "block_sparse_attention"
)

RESULTS_DIR="results"
REPEAT_COUNT="${REPEAT_COUNT:-3}"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for repeat_idx in $(seq 1 "${REPEAT_COUNT}"); do
  for algo in "${sparse_algos[@]}"; do
    OUTFILE="${RESULTS_DIR}/${algo}_naive_${TIMESTAMP}_run${repeat_idx}.log"
    echo ">>> Run ${repeat_idx}/${REPEAT_COUNT}: verify_algo.py with --vortex-module-name ${algo} --topk-type naive"
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
done

for repeat_idx in $(seq 1 "${REPEAT_COUNT}"); do
  for algo in "${sparse_algos[@]}"; do
    OUTFILE="${RESULTS_DIR}/${algo}_sglang_${TIMESTAMP}_run${repeat_idx}.log"
    echo ">>> Run ${repeat_idx}/${REPEAT_COUNT}: verify_algo.py with --vortex-module-name ${algo} --topk-type sglang"
    echo ">>> Saving results to ${OUTFILE}"
    { time python verify_algo.py \
      --trials 8 \
      --topk-val 30 \
      --vortex-module-name "${algo}" \
      --model-name Qwen/Qwen3-1.7B \
      --topk-type sglang \
      --mem 0.7 ; } \
      2>&1 | tee "${OUTFILE}"
  done
done