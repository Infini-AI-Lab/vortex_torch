#!/usr/bin/env bash
# Score a set of validation checkpoints in parallel, one per idle GPU, at full memory.
#
# The watcher (`watch_validate.py`) is deliberately serial and single-GPU: it is meant to
# trail a *running* trainer without competing for memory. Once training has finished the
# other GPUs are idle, and scoring 10 checkpoints one at a time on a memory-starved
# engine wastes hours -- a 0.35 memory fraction leaves a small KV cache, so 480 sequences
# (30 problems x 16 trials) run with far less concurrency than the GPU allows.
#
# This fans them out instead: one engine per GPU at --mem 0.85, as many at a time as
# there are free GPUs, remaining checkpoints queued behind.
#
# Usage: fanout_validate.sh <val_dir> [trials] [gpu_list_csv]
#   fanout_validate.sh /scratch/zhuominc/ckpt_qwen/val 16 1,2,3,4,5,6,7
set -uo pipefail

VAL_DIR="${1:?usage: fanout_validate.sh <val_dir> [trials] [gpus]}"
TRIALS="${2:-16}"
GPUS="${3:-0,1,2,3,4,5,6,7}"
ROOT=/scratch/zhuominc/vortex_train
PY=/scratch/zhuominc/venv-0516/bin/python
MEM="${MEM:-0.85}"

cd "$ROOT"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# Steps that already have a result (or are being scored by the serial watcher) are
# skipped, so this is safe to run alongside it.
mapfile -t STEPS < <(ls -d "$VAL_DIR"/step*/ 2>/dev/null | sed 's#.*/step##;s#/##' | sort -n)
TODO=()
for s in "${STEPS[@]}"; do
    if [ -f "$VAL_DIR/aime24_step${s}.json" ]; then
        echo "skip step $s (already scored)"
        continue
    fi
    if grep -q "\"step\": $s," "$VAL_DIR/history.jsonl" 2>/dev/null; then
        echo "skip step $s (in history)"
        continue
    fi
    TODO+=("$s")
done
echo "to score: ${TODO[*]:-none}  across ${NGPU} GPU(s) at mem=$MEM"
[ ${#TODO[@]} -eq 0 ] && exit 0

i=0
while [ $i -lt ${#TODO[@]} ]; do
    pids=()
    for ((g=0; g<NGPU && i<${#TODO[@]}; g++, i++)); do
        step="${TODO[$i]}"
        gpu="${GPU_ARR[$g]}"
        log="$VAL_DIR/fanout_step${step}.log"
        echo ">>> GPU$gpu  step $step  -> $log"
        CUDA_VISIBLE_DEVICES=$gpu HF_HOME=/scratch/zhuominc/hf NCCL_DEBUG=WARN \
        $PY -m application.sparse_finetune_qwen3.evaluate \
            --trained "$VAL_DIR/step${step}" \
            --tasks aime24 --which trained \
            --trials "$TRIALS" --mem "$MEM" \
            --out "$VAL_DIR/aime24_step${step}.json" \
            > "$log" 2>&1 &
        pids+=($!)
    done
    # Wait for this wave before starting the next, so no GPU gets two engines.
    for p in "${pids[@]}"; do wait "$p"; done
    echo "--- wave done ---"
done

echo "=== results ==="
for f in "$VAL_DIR"/aime24_step*.json; do
    $PY - "$f" <<'PY'
import json, sys, re
p = sys.argv[1]
try:
    d = json.load(open(p))
    r = d["results"][0]
    k = next(x for x in r if x.startswith("mean@"))
    step = re.search(r"step(\d+)", p).group(1)
    print(f"  step {step:>5}  {k}={r[k]:.4f}  pass@k={r['pass@k']:.4f}")
except Exception as e:
    print(f"  {p}: {e}")
PY
done
