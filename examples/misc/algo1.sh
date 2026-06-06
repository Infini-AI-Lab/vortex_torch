#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
set -e
sparse_algos=(
block_sparse_attention
)

models=(
Qwen/Qwen3-4B
)
trials=(
4
)
topk_val=(
61
)
for algo in "${sparse_algos[@]}"; do
  for model in "${models[@]}"; do
    for trial in "${trials[@]}"; do
      for k_val in "${topk_val[@]}"; do
        echo ">>> Running verify_algo.py with --vortex-module-name ${algo} and --model-name ${model} for ${trial} trials"
        python examples/math/verify_algo.py \
            --trials ${trial} \
            --topk-val ${k_val} \
            --page-size 16 \
            --workload-chunk-size 64 \
            --block-size 16 \
            --topk-ratio 0.0625 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.85 \
            --data-path examples/math/aime24.jsonl \
            --generation-max-new-tokens 16384 \
            --max-input-length 4096 \
            --tp-size 1 \
            --summary-dir summary-Qwen3-4B
      done
    done
  done
done