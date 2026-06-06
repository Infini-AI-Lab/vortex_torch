#!/usr/bin/env bash

set -e
sparse_algos=(
block_sparse_attention
gqa_quest_sparse_attention
)
models=(
Qwen/Qwen3-30B-A3B-FP8
)
trials=(
32
)
topk_val=(
93
125
157
253
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
            --topk-ratio 0.0 \
            --vortex-module-name "${algo}" \
            --model-name  "${model}" \
            --mem 0.9 \
            --data-path examples/math/aime24.jsonl \
            --generation-max-new-tokens 32768 \
            --max-input-length 4096 \
            --tp-size 1 \
            --vortex-impl-backend triton \
            --vortex-attention-backend trtllm \
            --vortex-use-tensor-core \
            --vortex-layers-skip \
            --summary-dir summary-Qwen3-30B-A3B-FP8
      done
    done
  done
done