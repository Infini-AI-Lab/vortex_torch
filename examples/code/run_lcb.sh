#!/usr/bin/env bash

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model-name>" >&2
  echo "  e.g.  $0 Qwen/Qwen3-4B" >&2
  exit 1
fi

model="$1"

sparse_algos=(
  block_sparse_attention
  gqa_quest_sparse_attention
)
trials=(
  8
  16
  32
)
topk_val=(
  29
  61
  93
  125
  157
  189
  221
  253
)

NGPU=8
summary_dir="summary-lcbv5-${model##*/}"
logdir="logs/lcb/${model##*/}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$logdir"

idx=0
for algo in "${sparse_algos[@]}"; do
  for trial in "${trials[@]}"; do
    for k_val in "${topk_val[@]}"; do
      gpu=$(( idx % NGPU ))
      stem="${algo}_t${trial}_k${k_val}"
      echo ">>> [gpu${gpu}] ${stem}  (model=${model})"
      CUDA_VISIBLE_DEVICES="${gpu}" \
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
          --data-path examples/code/lcbv5.jsonl \
          --generation-max-new-tokens 16384 \
          --max-input-length 4096 \
          --tp-size 1 \
          --summary-dir "${summary_dir}" \
          > "${logdir}/gpu${gpu}_${stem}.out" \
          2> "${logdir}/gpu${gpu}_${stem}.err" &
      idx=$(( idx + 1 ))
      if (( idx % NGPU == 0 )); then
        wait
      fi
    done
  done
done

wait
echo "all runs finished. logs: ${logdir}  summaries: ${summary_dir}"
