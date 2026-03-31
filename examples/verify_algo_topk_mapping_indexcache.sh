#!/usr/bin/env bash
set -e
# use CUDA_VISIBLE_DEVICES to set the GPU id you want to use
export CUDA_VISIBLE_DEVICES=5

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

sparse_algos=(
  "block_sparse_attention"
)

# --- Mode 5: Index Cache (default even-layer pattern) ---
for algo in "${sparse_algos[@]}"; do
  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_mode5_index_cache_${TIMESTAMP}.log"
  echo ">>> Running verify_algo.py with --vortex-module-name ${algo} --topk-type sglang --topk-mapping-mode 5 (index cache)"
  echo ">>> Saving results to ${OUTFILE}"
  { time python verify_algo.py \
    --trials 8 \
    --topk-val 30 \
    --vortex-module-name "${algo}" \
    --model-name Qwen/Qwen3-1.7B \
    --topk-type sglang \
    --topk-mapping-mode 5 \
    --index-cache-shared-layers 2 4 6 8 10 12 14 16 18 20 22 24 26 \
    --mem 0.7 ; } \
    2>&1 | tee "${OUTFILE}"
done

# --- Mode 6: Greedy layer selection ---
# for algo in "${sparse_algos[@]}"; do
#  OUTFILE="${RESULTS_DIR}/topk_mapping_${algo}_sglang_mode6_greedy_${TIMESTAMP}.log"
#  echo ">>> Running verify_algo.py with --vortex-module-name ${algo} --topk-type sglang --topk-mapping-mode 6 (greedy)"
#  echo ">>> Saving results to ${OUTFILE}"
#  { time python verify_algo.py \
#    --trials 8 \
#    --topk-val 30 \
#    --vortex-module-name "${algo}" \
#    --model-name Qwen/Qwen3-1.7B \
#    --topk-type sglang \
#    --topk-mapping-mode 6 \
#    --mem 0.7 ; } \
#    2>&1 | tee "${OUTFILE}"
#done
