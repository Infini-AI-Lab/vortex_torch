#!/usr/bin/env bash
# Arch sweep for the MHA block compressor at 16K training length.
# Usage: compressor_sweep.sh <gpu_csv> <name:arch:bc:dc:init[:hid]> [...]
# Each config: ~18-min budget + baseline/final evals on a fixed grid.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"; cd "$REPO"
GPUS="$1"; shift
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
mkdir -p logs/compressor result/compressor

for spec in "$@"; do
  IFS=: read -r name arch bc dc init hid <<< "$spec"
  hid="${hid:-0}"
  echo "[sweep] $name arch=$arch bc=$bc dc=$dc init=$init hid=$hid on GPUs $GPUS"
  CUDA_VISIBLE_DEVICES="$GPUS" timeout 45m conda run --no-capture-output -n vortex_v1 \
  torchrun --standalone --nproc-per-node="$NPROC" -m vortex_torch.compressor.train \
    --model Qwen/Qwen3-4B --attn mha \
    --hf-dataset "Jackrong/GLM-5.1-Reasoning-1M-Cleaned,Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned#General-Distillation" \
    --num-prompts 512 --max-minutes 18 --max-tokens 16384 --batch-size 2 \
    --arch "$arch" --bc "$bc" --dc "$dc" --init "$init" --hidden-dim "$hid" \
    --block-size 64 --budget-blocks 15 \
    --recall-n 16,64,128 \
    --eval-jsonl ruler16k:examples/ruler/validation_16k.jsonl:16384:6 \
    --eval-jsonl ruler32k:examples/ruler/validation_32k.jsonl:32768:4 \
    --eval-num-prompts 6 --eval-every-min 0 \
    --log-every 16 \
    --out "result/compressor/sweep16k_${name}.pt" \
    > "logs/compressor/sweep16k_${name}.log" 2>&1
  echo "[sweep] $name done (exit $?)"
done
echo "[sweep] lane complete: $*"
