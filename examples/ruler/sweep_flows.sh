#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RULER flow sweep
#
# Sweeps the built-in vortex attention flows through the RULER needle-in-a-
# haystack benchmark (examples/ruler/validation.jsonl) and prints an accuracy
# table:
#
#   * MHA: all 9 flows in vortex_torch/flow/algorithms.py, each on BOTH indexer
#     backends (flashinfer + trtllm)              -> 18 runs
#   * MLA: 2 flows in vortex_torch/flow/algorithms_mla.py
#          (rope_aware_block_sparse_mla + lserve_centroid_mla) -> 2 runs
#
# Runs one flow per free GPU, in waves, re-detecting free GPUs each wave via
# algorithm_scientist/free_gpus.sh (shared-cluster friendly).
#
# Usage:
#   examples/ruler/sweep_flows.sh [all|mha|mla]      # default: all
#
# Environment overrides:
#   MODEL=Qwen/Qwen3-4B           # MHA model
#   MLA_MODEL=zai-org/GLM-4.7-Flash
#   MHA_PY="python"               # interpreter where `import vortex_torch` works
#                                 #   (vortex_v1 env: transformers 4.x)
#   MLA_PY="python"               # GLM needs transformers >= 5 (vortex_glm env),
#                                 #   e.g. MLA_PY="conda run -n vortex_glm python"
#   BACKENDS="flashinfer trtllm"  # MHA indexer backends to sweep
#   HF_HOME=/raid/catalyst/models/
#   OUT=<dir>                     # results dir (default: examples/ruler/sweep_results)
# ---------------------------------------------------------------------------
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

MODE="${1:-all}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
MLA_MODEL="${MLA_MODEL:-zai-org/GLM-4.7-Flash}"
MHA_PY="${MHA_PY:-python}"
MLA_PY="${MLA_PY:-python}"
BACKENDS="${BACKENDS:-flashinfer trtllm}"
OUT="${OUT:-$HERE/sweep_results}"
export HF_HOME="${HF_HOME:-/raid/catalyst/models/}"
FREE="$REPO/algorithm_scientist/free_gpus.sh"
mkdir -p "$OUT/logs"

# 9 MHA flows (vortex_torch/flow/algorithms.py). running_avg_block_sparse uses
# Save(...) in forward_indexer -> requires disable_radix_cache.
MHA_FLOWS=(
  block_sparse_attention
  gqa_block_sparse_attention
  gqa_quest_sparse_attention
  lserve_sparse_attention
  lserve_centroid_sparse_attention
  masked_quest_sparse_attention
  centered_block_sparse_attention
  running_avg_block_sparse
  venergy_gated_centroid
)
# 2 MLA flows (vortex_torch/flow/algorithms_mla.py).
MLA_FLOWS=(
  rope_aware_block_sparse_mla
  lserve_centroid_mla
)

# Build the job list as "<kind>|<flow>|<backend>".
JOBS=()
if [ "$MODE" = all ] || [ "$MODE" = mha ]; then
  for be in $BACKENDS; do for f in "${MHA_FLOWS[@]}"; do JOBS+=("mha|$f|$be"); done; done
fi
if [ "$MODE" = all ] || [ "$MODE" = mla ]; then
  for f in "${MLA_FLOWS[@]}"; do JOBS+=("mla|$f|cuda_mla"); done
fi
echo "[sweep] mode=$MODE  jobs=${#JOBS[@]}  out=$OUT"

run_one () {
  local kind="$1" flow="$2" be="$3" gpu="$4"
  local log="$OUT/logs/${kind}_${flow}_${be}.log"
  if [ "$kind" = mha ]; then
    local radix=0; [ "$flow" = running_avg_block_sparse ] && radix=1
    CUDA_VISIBLE_DEVICES="$gpu" VORTEX_MODULE="$flow" \
      VORTEX_ATTENTION_BACKEND="$be" DISABLE_RADIX_CACHE="$radix" \
      $MHA_PY examples/ruler/run_ruler.py "$MODEL" > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" $MLA_PY examples/ruler/run_ruler_mla.py \
      --model "$MLA_MODEL" --module "$flow" --attn-backend "$be" > "$log" 2>&1
  fi
}

# Launch in waves: one job per free GPU, re-detect each wave.
i=0; N=${#JOBS[@]}
while [ "$i" -lt "$N" ]; do
  GPUS=($("$FREE")) || { echo "[sweep] no free GPU — waiting"; sleep 30; continue; }
  s=0
  while [ "$s" -lt "${#GPUS[@]}" ] && [ "$i" -lt "$N" ]; do
    IFS='|' read -r kind flow be <<< "${JOBS[$i]}"
    echo "[sweep] gpu ${GPUS[$s]}  $kind  $flow  $be"
    run_one "$kind" "$flow" "$be" "${GPUS[$s]}" &
    s=$((s+1)); i=$((i+1))
  done
  wait
done

# ----- collect + print accuracy table -----
acc_of () { grep -oE "[0-9]+(\.[0-9]+)?%" "$1" 2>/dev/null | tail -1; }
echo
echo "===== RULER flow sweep — accuracy ====="
printf "%-34s %-11s %s\n" "flow" "backend" "accuracy"
if [ "$MODE" = all ] || [ "$MODE" = mha ]; then
  for be in $BACKENDS; do for f in "${MHA_FLOWS[@]}"; do
    a=$(acc_of "$OUT/logs/mha_${f}_${be}.log"); printf "%-34s %-11s %s\n" "$f" "$be" "${a:-FAIL}"
  done; done
fi
if [ "$MODE" = all ] || [ "$MODE" = mla ]; then
  for f in "${MLA_FLOWS[@]}"; do
    a=$(acc_of "$OUT/logs/mla_${f}_cuda_mla.log"); printf "%-34s %-11s %s\n" "$f" "cuda_mla" "${a:-FAIL}"
  done
fi
echo "logs: $OUT/logs"
