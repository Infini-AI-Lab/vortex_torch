#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RULER MHA flow sweep
#
# Sweeps the built-in vortex MHA flows (vortex_torch/flow/algorithms.py)
# through the RULER needle-in-a-haystack benchmark
# (examples/ruler/validation_4k.jsonl) on every indexer backend in BACKENDS and
# prints an accuracy table — 9 flows x 2 backends = 18 runs by default.
#
# Runs one flow per free GPU, in waves, re-detecting free GPUs each wave via
# algorithm_scientist/free_gpus.sh (shared-cluster friendly).
#
# Usage:
#   examples/ruler/sweep_mha.sh
#
# Environment overrides (unified with sweep_mla.sh):
#   MODEL=Qwen/Qwen3-4B           # HF model id
#   PY="python"                   # interpreter where `import vortex_torch` works
#                                 #   (vortex_v1 env: transformers 4.x)
#   FLOWS="..."                   # space-separated flow list (default: all 9)
#   BACKENDS="flashinfer trtllm"  # vortex indexer backends (--indexer-backend)
#   BLOCK=32                      # block size (page_size == vortex_block_size)
#   TOPK=29                       # vortex_topk_val (selected blocks)
#   LAYERS_SKIP=""                # dense layers, comma-separated (default: none)
#   EXTRA_ARGS=""                 # extra run_ruler_mha.py flags (e.g. "--n 20")
#   OUT=<dir>                     # results dir (default: examples/ruler/sweep_results)
# ---------------------------------------------------------------------------
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
PY="${PY:-python}"
BACKENDS="${BACKENDS:-flashinfer trtllm}"
BLOCK="${BLOCK:-32}"
TOPK="${TOPK:-29}"
LAYERS_SKIP="${LAYERS_SKIP:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT="${OUT:-$HERE/sweep_results}"
FREE="$REPO/algorithm_scientist/free_gpus.sh"
mkdir -p "$OUT/logs"

# 9 MHA flows (vortex_torch/flow/algorithms.py). running_avg_block_sparse uses
# Save(...) in forward_indexer -> gets --disable-radix-cache automatically.
DEFAULT_FLOWS="
  block_sparse_attention
  gqa_block_sparse_attention
  gqa_quest_sparse_attention
  lserve_sparse_attention
  lserve_centroid_sparse_attention
  masked_quest_sparse_attention
  centered_block_sparse_attention
  running_avg_block_sparse
  venergy_gated_centroid
"
FLOWS="${FLOWS:-$DEFAULT_FLOWS}"

# Build the job list as "<flow>|<backend>".
JOBS=()
for be in $BACKENDS; do for f in $FLOWS; do JOBS+=("$f|$be"); done; done
echo "[sweep_mha] model=$MODEL jobs=${#JOBS[@]} block=$BLOCK topk=$TOPK layers_skip=[$LAYERS_SKIP] out=$OUT"

run_one () {
  local flow="$1" be="$2" gpu="$3"
  local log="$OUT/logs/mha_${flow}_${be}.log"
  local radix=""
  [ "$flow" = running_avg_block_sparse ] && radix="--disable-radix-cache"
  CUDA_VISIBLE_DEVICES="$gpu" $PY examples/ruler/run_ruler_mha.py \
    --model "$MODEL" --module "$flow" --indexer-backend "$be" \
    --block "$BLOCK" --topk "$TOPK" --layers-skip "$LAYERS_SKIP" \
    $radix $EXTRA_ARGS > "$log" 2>&1
}

# Launch in waves: one job per free GPU, re-detect each wave.
i=0; N=${#JOBS[@]}
while [ "$i" -lt "$N" ]; do
  GPUS=($("$FREE")) || { echo "[sweep_mha] no free GPU — waiting"; sleep 30; continue; }
  s=0
  while [ "$s" -lt "${#GPUS[@]}" ] && [ "$i" -lt "$N" ]; do
    IFS='|' read -r flow be <<< "${JOBS[$i]}"
    echo "[sweep_mha] gpu ${GPUS[$s]}  $flow  $be"
    run_one "$flow" "$be" "${GPUS[$s]}" &
    s=$((s+1)); i=$((i+1))
  done
  wait
done

# ----- collect + print accuracy table -----
acc_of () { grep -oE "[0-9]+(\.[0-9]+)?%" "$1" 2>/dev/null | tail -1; }
echo
echo "===== RULER MHA flow sweep — accuracy ====="
printf "%-34s %-11s %s\n" "flow" "backend" "accuracy"
for be in $BACKENDS; do for f in $FLOWS; do
  a=$(acc_of "$OUT/logs/mha_${f}_${be}.log"); printf "%-34s %-11s %s\n" "$f" "$be" "${a:-FAIL}"
done; done
echo "logs: $OUT/logs"
