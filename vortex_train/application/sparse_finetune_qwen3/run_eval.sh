#!/usr/bin/env bash
# Wait for the 1k-step training to finish, then evaluate trained vs untrained on
# AIME24 + AIME25 under the SAME sparse budget the model was finetuned with.
#
# The four (task, model) combinations are independent, so they run concurrently on
# four GPUs with tp=1 each rather than sequentially in one process. That is ~4x less
# wall clock, and it keeps each engine's KV cache on its own device -- a single
# process cycling through four engines would pay the model load and the vortex
# compile four times over anyway.
#
#   GPU0 aime24/base     GPU1 aime24/trained
#   GPU2 aime25/base     GPU3 aime25/trained
#
# Usage: run_eval.sh [checkpoint_dir] [trials]
set -uo pipefail

CKPT="${1:-/scratch/zhuominc/ckpt_1k}"
TRIALS="${2:-16}"
BASE="Qwen/Qwen3-4B"
ROOT=/scratch/zhuominc/vortex_train
OUT=/scratch/zhuominc/eval_1k
PY=/scratch/zhuominc/venv-0516/bin/python

mkdir -p "$OUT"
cd "$ROOT"

# --- wait for training ------------------------------------------------------
# Gate on the checkpoint's own marker file, not on the process table: a crashed run
# would also leave no process, and evaluating a half-written checkpoint silently
# would be worse than not running.
echo "waiting for $CKPT/vortex_selection.json ..."
while true; do
    if [ -f "$CKPT/vortex_selection.json" ] && [ -f "$CKPT/model.safetensors" ]; then
        echo "checkpoint present"
        break
    fi
    if ! pgrep -f "sparse_finetune_qwen3.train" >/dev/null; then
        echo "training is no longer running and the checkpoint is incomplete -- aborting"
        exit 1
    fi
    sleep 120
done
# let the final barrier + file flush settle
sleep 30

echo "=== budget check (must print MATCHED) ==="
HF_HOME=/scratch/zhuominc/hf $PY -m application.sparse_finetune_qwen3.evaluate \
    --trained "$CKPT" --dry-run || exit 1

# --- fan out ----------------------------------------------------------------
i=0
for task in aime24 aime25; do
    for which in base trained; do
        gpu=$i
        log="$OUT/${task}_${which}.log"
        echo ">>> GPU$gpu  $task/$which  -> $log"
        CUDA_VISIBLE_DEVICES=$gpu HF_HOME=/scratch/zhuominc/hf NCCL_DEBUG=WARN \
        setsid nohup $PY -m application.sparse_finetune_qwen3.evaluate \
            --trained "$CKPT" --base "$BASE" \
            --tasks "$task" --which "$which" \
            --trials "$TRIALS" --tp 1 --mem 0.85 \
            --out "$OUT/${task}_${which}.json" \
            > "$log" 2>&1 < /dev/null &
        i=$((i+1))
    done
done

wait
echo "=== all four finished ==="
for f in "$OUT"/*.json; do echo "--- $f"; done
