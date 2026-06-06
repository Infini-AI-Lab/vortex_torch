#!/usr/bin/env bash

set -uo pipefail

prefix=""
data_path="examples/code/lcbv5.jsonl"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix) prefix="$2"; shift 2 ;;
    -d|--data)   data_path="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [-p <summary-prefix>] [-d <data-path>] <model-name> [model-name ...]" >&2
      echo "  e.g.  $0 -p runs/expt1 Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B" >&2
      exit 0
      ;;
    -*) echo "unknown flag: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [-p <summary-prefix>] [-d <data-path>] <model-name> [model-name ...]" >&2
  echo "  e.g.  $0 -p runs/expt1 Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B" >&2
  exit 1
fi

models=("$@")
algo="full_attention"
trial=2

parent="${prefix:+${prefix%/}/}"
ts="$(date +%Y%m%d_%H%M%S)"
logdir="${parent}logs/lcb/full_attention_${ts}"
mkdir -p "$logdir"

idx=0
for model in "${models[@]}"; do
  gpu="${idx}"
  summary_dir="${parent}summary-lcbv5-${model##*/}"
  stem="${algo}_${model##*/}_t${trial}"
  echo ">>> [gpu${gpu}] ${stem}  (model=${model})"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python examples/code/run_lcb.py \
      --trials ${trial} \
      --topk-val 29 \
      --page-size 16 \
      --workload-chunk-size 64 \
      --block-size 16 \
      --topk-ratio 0.0 \
      --vortex-module-name "${algo}" \
      --model-name  "${model}" \
      --mem 0.9 \
      --data-path "${data_path}" \
      --generation-max-new-tokens 16384 \
      --max-input-length 4096 \
      --tp-size 1 \
      --summary-dir "${summary_dir}" \
      > "${logdir}/gpu${gpu}_${stem}.out" \
      2> "${logdir}/gpu${gpu}_${stem}.err" &
  idx=$(( idx + 1 ))
done

wait
echo "all runs finished. logs: ${logdir}  summaries under: ${parent:-./}summary-lcbv5-*"
