#!/usr/bin/env bash
# Fully optimized Vortex deployment points at a production memory fraction.
set -euo pipefail

OUT="${OUT:-summary_vortex_production_mem085_qwen3_4b}"
GPU="${GPU:-0}"

for topk in 61 125; do
  echo "optimized Vortex mem=0.85 topk=$topk"
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
      --mem 0.85 --data-path examples/math/aime24.jsonl --tp-size 1 \
      --attention-backend flashinfer \
      --vortex-attention-backend trtllm \
      --vortex-impl-backend triton --vortex-use-tensor-core \
      --vortex-layers-skip 0 \
      --summary-dir "$OUT" --skip-already-finished-check
done
