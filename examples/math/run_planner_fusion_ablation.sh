#!/usr/bin/env bash
# Controlled 2x2 workload-planning/kernel-fusion ablation for Quest.
set -euo pipefail

OUT="${OUT:-summary_ablation_planner_fusion_qwen3_4b}"
GPU="${GPU:-0}"

COMMON=(
  --trials 16
  --topk-ratio 0
  --block-size 16
  --page-size 16
  --workload-chunk-size 32
  --generation-max-new-tokens 16384
  --max-input-length 4096
  --vortex-module-name gqa_quest_sparse_attention
  --model-name Qwen/Qwen3-4B
  # The unfused baseline materializes Quest's page-wise intermediates; 0.6
  # leaves room for those buffers on a B200. Identical for all four cells.
  --mem 0.6
  --data-path examples/math/aime24.jsonl
  --tp-size 1
  --attention-backend flashinfer
  --vortex-attention-backend trtllm
  --vortex-impl-backend triton
  --vortex-use-tensor-core
  --vortex-layers-skip 0
  --summary-dir "$OUT"
  --skip-already-finished-check
)

for topk in 61 125; do
  for planner in naive optimized; do
    for fusion in off on; do
      extra=()
      if [[ "$planner" == "naive" ]]; then
        extra+=(--naive-workload-planner --disable-cuda-graph)
      fi
      if [[ "$fusion" == "off" ]]; then
        extra+=(--disable-vortex-fusion)
      fi
      echo "topk=$topk planner=$planner fusion=$fusion"
      CUDA_VISIBLE_DEVICES="$GPU" \
      HF_HOME=/workspace/.cache/huggingface \
      conda run --no-capture-output -n vortex_v1 \
        python examples/math/verify_algo.py \
        "${COMMON[@]}" --topk-val "$topk" "${extra[@]}"
    done
  done
done
