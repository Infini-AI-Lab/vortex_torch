#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5,6
export HF_HOME=/raid/catalyst/models/
set -e
sparse_algos=(
block_sparse_attention
)

models=(
MiniMaxAI/MiniMax-M2.7
)
trials=(
16
)
# Sweep block_size (= page_size) while holding
# (topk_val + 3) * block_size = 2048.
block_sizes=(
16
32
64
)
for algo in "${sparse_algos[@]}"; do
  for model in "${models[@]}"; do
    for trial in "${trials[@]}"; do
      for block_size in "${block_sizes[@]}"; do
        k_val=$((2048 / block_size - 3))
        echo ">>> Running verify_algo.py with --vortex-module-name ${algo} and --model-name ${model} for ${trial} trials (block_size=${block_size}, topk_val=${k_val})"
        python examples/math/verify_algo.py \
            --trials ${trial} \
            --topk-val ${k_val} \
            --page-size ${block_size} \
            --workload-chunk-size 64 \
            --block-size ${block_size} \
            --topk-ratio 0.00 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.9 \
            --data-path examples/math/aime26_minimax.jsonl \
            --generation-max-new-tokens 32768 \
            --max-input-length 4096 \
            --tp-size 2 \
            --vortex-attention-backend trtllm \
            --vortex-impl-backend triton \
            --vortex-use-tensor-core \
            --vortex-layers-skip \
            --summary-dir summary-M2.7 \
            --skip-already-finished-check
      done
    done
  done
done
