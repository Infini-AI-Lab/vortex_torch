#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RULER MLA flow sweep
#
# Sweeps the built-in vortex MLA flows (vortex_torch/flow/algorithms_mla.py)
# through the RULER needle-in-a-haystack benchmark
# (examples/ruler/validation_4k.jsonl) and prints an accuracy table — 2 runs by
# default (rope_aware_block_sparse_mla + lserve_centroid_mla on cuda_mla).
#
# Runs one flow per free GPU, in waves, re-detecting free GPUs each wave via
# algorithm_scientist/free_gpus.sh (shared-cluster friendly).
#
# Usage:
#   examples/ruler/sweep_mla.sh
#
# Environment overrides (unified with sweep_mha.sh):
#   MODEL=zai-org/GLM-4.7-Flash   # HF model id
#   PY="python"                   # interpreter where `import vortex_torch` works;
#                                 #   GLM needs transformers >= 5 (vortex_glm env),
#                                 #   e.g. PY="conda run -n vortex_glm python"
#   FLOWS="..."                   # space-separated flow list (default: both)
#   BACKENDS="cuda_mla"           # sglang attention backends (--attn-backend)
#   BLOCK=32                      # block size (page_size == vortex_block_size)
#   TOPK=29                       # vortex_topk_val (selected blocks)
#   LAYERS_SKIP=""                # dense layers, comma-separated (default: none)
#   EXTRA_ARGS=""                 # extra run_ruler_mla.py flags (e.g. "--n 20")
#   OUT=<dir>                     # results dir (default: examples/ruler/sweep_results)
# ---------------------------------------------------------------------------
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

MODEL="${MODEL:-zai-org/GLM-4.7-Flash}"
PY="${PY:-python}"
BACKENDS="${BACKENDS:-cuda_mla}"
BLOCK="${BLOCK:-32}"
TOPK="${TOPK:-29}"
LAYERS_SKIP="${LAYERS_SKIP:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OUT="${OUT:-$HERE/sweep_results}"
FREE="$REPO/algorithm_scientist/free_gpus.sh"
mkdir -p "$OUT/logs"

# 2 MLA flows (vortex_torch/flow/algorithms_mla.py).
DEFAULT_FLOWS="
  rope_aware_block_sparse_mla
  lserve_centroid_mla
"
FLOWS="${FLOWS:-$DEFAULT_FLOWS}"

# Build the job list as "<flow>|<backend>".
JOBS=()
for be in $BACKENDS; do for f in $FLOWS; do JOBS+=("$f|$be"); done; done
echo "[sweep_mla] model=$MODEL jobs=${#JOBS[@]} block=$BLOCK topk=$TOPK layers_skip=[$LAYERS_SKIP] out=$OUT"

run_one () {
  local flow="$1" be="$2" gpu="$3"
  local log="$OUT/logs/mla_${flow}_${be}.log"
  CUDA_VISIBLE_DEVICES="$gpu" $PY examples/ruler/run_ruler_mla.py \
    --model "$MODEL" --module "$flow" --attn-backend "$be" \
    --block "$BLOCK" --topk "$TOPK" --layers-skip "$LAYERS_SKIP" \
    $EXTRA_ARGS > "$log" 2>&1
}

# Launch in waves: one job per free GPU, re-detect each wave.
i=0; N=${#JOBS[@]}
while [ "$i" -lt "$N" ]; do
  GPUS=($("$FREE")) || { echo "[sweep_mla] no free GPU — waiting"; sleep 30; continue; }
  s=0
  while [ "$s" -lt "${#GPUS[@]}" ] && [ "$i" -lt "$N" ]; do
    IFS='|' read -r flow be <<< "${JOBS[$i]}"
    echo "[sweep_mla] gpu ${GPUS[$s]}  $flow  $be"
    run_one "$flow" "$be" "${GPUS[$s]}" &
    s=$((s+1)); i=$((i+1))
  done
  wait
done

# ----- collect + print accuracy table -----
acc_of () { grep -oE "[0-9]+(\.[0-9]+)?%" "$1" 2>/dev/null | tail -1; }
echo
echo "===== RULER MLA flow sweep — accuracy ====="
printf "%-34s %-11s %s\n" "flow" "backend" "accuracy"
for be in $BACKENDS; do for f in $FLOWS; do
  a=$(acc_of "$OUT/logs/mla_${f}_${be}.log"); printf "%-34s %-11s %s\n" "$f" "$be" "${a:-FAIL}"
done; done
echo "logs: $OUT/logs"
