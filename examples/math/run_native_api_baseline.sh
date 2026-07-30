#!/usr/bin/env bash
# Native-API-style baseline: padded planner + unfused Quest + torch.topk.
set -euo pipefail

OUT="${OUT:-summary_native_api_qwen3_4b}"
GPU="${GPU:-0}"

for topk in 61 125; do
  echo "native-api baseline topk=$topk"
  CUDA_VISIBLE_DEVICES="$GPU" \
  HF_HOME=/workspace/.cache/huggingface \
  conda run --no-capture-output -n vortex_v1 \
    python examples/math/verify_algo.py \
      --trials 16 \
      --topk-val "$topk" --topk-ratio 0 \
      --block-size 16 --page-size 16 --workload-chunk-size 32 \
      --generation-max-new-tokens 16384 --max-input-length 4096 \
      --vortex-module-name gqa_quest_sparse_attention \
      --model-name Qwen/Qwen3-4B \
      --mem 0.6 --data-path examples/math/aime24.jsonl --tp-size 1 \
      --attention-backend flashinfer \
      --vortex-attention-backend trtllm \
      --vortex-impl-backend triton --vortex-use-tensor-core \
      --vortex-layers-skip 0 \
      --naive-workload-planner --disable-cuda-graph \
      --disable-vortex-fusion --use-torch-topk \
      --summary-dir "$OUT" --skip-already-finished-check
done
