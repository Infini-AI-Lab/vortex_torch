#!/usr/bin/env bash

set -uo pipefail

prefix=""
data_path="examples/code/lcbv5.jsonl"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prefix) prefix="$2"; shift 2 ;;
    -d|--data)   data_path="$2"; shift 2 ;;
    -h|--help)
      echo "usage: $0 [-p <summary-prefix>] [-d <data-path>] <model-name> [trial ...]" >&2
      echo "  e.g.  $0 -p runs/expt1 -d examples/code/lcbv5.jsonl Qwen/Qwen3-4B 1 2 4" >&2
      exit 0
      ;;
    -*) echo "unknown flag: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [-p <summary-prefix>] [-d <data-path>] <model-name> [trial ...]" >&2
  echo "  e.g.  $0 -p runs/expt1 -d examples/code/lcbv5.jsonl Qwen/Qwen3-4B 1 2 4" >&2
  exit 1
fi

model="$1"
shift
algo="gqa_quest_sparse_attention"

if [[ $# -gt 0 ]]; then
  trials=("$@")
else
  trials=(1 2 4)
fi
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
parent="${prefix:+${prefix%/}/}"
summary_dir="${parent}summary-lcbv5-${model##*/}"
logdir="${parent}logs/lcb/${model##*/}_${algo}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$logdir"

idx=0
for trial in "${trials[@]}"; do
  for k_val in "${topk_val[@]}"; do
    gpu=$(( idx % NGPU ))
    stem="${algo}_t${trial}_k${k_val}"
    echo ">>> [gpu${gpu}] ${stem}  (model=${model})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python examples/code/run_lcb.py \
        --trials ${trial} \
        --topk-val ${k_val} \
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
    if (( idx % NGPU == 0 )); then
      wait
    fi
  done
done

wait
echo "all runs finished. logs: ${logdir}  summaries: ${summary_dir}"
