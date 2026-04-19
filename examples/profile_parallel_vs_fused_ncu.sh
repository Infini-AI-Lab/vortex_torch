#!/usr/bin/env bash
# ============================================================
# Nsight Compute profiling script for the parallel vs fused
# TopK kernels.
#
# Profiles both:
#   - TopKOutput_Fused_Kernel    (csrc/topk_sglang.cu)
#   - TopKOutput_Parallel_Kernel (csrc/topk_sglang_parallel.cu)
#
# With both remap functions the user cares about:
#   - mode 15: MAPPING_SHIFT_POW2
#   - mode 16: MAPPING_SHIFT_POW3
#
# And both configs:
#   - A: topk=2048, pages_per_seg=32K (topk=2k from 32k)
#   - B: topk=30,   pages_per_seg=2K  (topk=30 from 2k)
#
# Produces one .ncu-rep per (kernel × mode × config). Open with
# the Nsight Compute GUI for an interactive comparison, or dump on
# the CLI with `ncu --import <file>.ncu-rep --page details`.
#
# Usage:
#   bash examples/profile_parallel_vs_fused.sh                  # defaults
#   GPU=4 EFF_BS=1  bash examples/profile_parallel_vs_fused.sh  # small-batch case
#   GPU=4 EFF_BS=32 bash examples/profile_parallel_vs_fused.sh  # saturated case
#   GPU=4 NUM_SPLITS=2 bash examples/profile_parallel_vs_fused.sh
#
# Requires `ncu` on PATH (part of the CUDA toolkit). On most systems
# accessing performance counters requires either:
#   - root/sudo, or
#   - `echo 1 | sudo tee /proc/driver/nvidia/params`  (temporary), or
#   - setting NVreg_RestrictProfilingToAdminUsers=0 in the nvidia driver.
# If ncu reports "ERR_NVGPUCTRPERM" you'll need one of the above.
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DRIVER="${SCRIPT_DIR}/../benchmarks/profile_parallel_vs_fused.py"

# ── Defaults ──────────────────────────────────────────────────
GPU=${GPU:-7}
EFF_BS=${EFF_BS:-1}                 # eff_batch_size = batch_size * num_kv_heads
NUM_SPLITS=${NUM_SPLITS:-2}          # only used by the parallel kernel
POWER=${POWER:--1.0}                  # pivot p for shift_pow{2,3}
WARMUP=${WARMUP:-20}                 # matching-kernel warmup launches (ncu skips)
ITERS=${ITERS:-1}                    # matching-kernel profiled launches (ncu captures)
SECTION_SET=${SECTION_SET:-full}     # ncu section set: "full", "basic", or named sections

# Profiling robustness knobs for shared GPUs / CUDA 13 systems.
# --replay-mode application: re-run the entire process to collect each
#                            counter pass, instead of replaying individual
#                            kernels. Fixes "Failed to prepare kernel" on
#                            systems where kernel replay hits PMU conflicts.
# --clock-control none     : don't try to lock GPU clocks (requires admin on
#                            shared GPUs; without this, "Unknown error on
#                            device 0" is common).
# --cache-control none     : don't flush L1/L2 between passes (also needs
#                            admin on shared systems).
# Override with NCU_EXTRA_FLAGS="..." if you need a different combination.
NCU_EXTRA_FLAGS=${NCU_EXTRA_FLAGS:-"--replay-mode application --clock-control none --cache-control none"}

# DIAG=1 bash profile_parallel_vs_fused.sh  → run one tiny ncu probe to
# verify profiling works before doing the full sweep.
DIAG=${DIAG:-0}

# ── ncu command ───────────────────────────────────────────────
NCU=${NCU:-ncu}
command -v "${NCU}" >/dev/null 2>&1 || {
  echo "ERROR: '${NCU}' not found on PATH. Install Nsight Compute (part of CUDA Toolkit)"
  echo "       or set NCU=/path/to/ncu and re-run."
  exit 1
}

# The templated kernels end up with mangled names like
#   _Z25TopKOutput_Fused_KernelI13__nv_bfloat16ILi15EEEvPKT_...
# ncu supports --kernel-name regex:<pattern> which matches on the
# demangled signature. Using "TopKOutput_Fused_Kernel" and
# "TopKOutput_Parallel_Kernel" as the regex selects all template
# instantiations of each kernel but nothing else.
FUSED_REGEX="regex:TopKOutput_Fused_Kernel"
PARALLEL_REGEX="regex:TopKOutput_Parallel_Kernel"

# ── Output dir ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${SCRIPT_DIR}/results/ncu_parallel_vs_fused_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Nsight Compute profile: parallel vs fused TopK"
echo "  GPU:            ${GPU}"
echo "  eff_bs:         ${EFF_BS}"
echo "  num_splits:     ${NUM_SPLITS}  (parallel kernel only)"
echo "  power (p):      ${POWER}       (for shift_pow{2,3})"
echo "  warmup:         ${WARMUP}      (matching-kernel launches skipped by ncu)"
echo "  iters:          ${ITERS}       (matching-kernel launches captured)"
echo "  sections:       --set ${SECTION_SET}"
echo "  extra ncu flags:${NCU_EXTRA_FLAGS}"
echo "  output dir:     ${OUT_DIR}"
echo "============================================================"

# ── Diagnostic probe ─────────────────────────────────────────
# Verifies that ncu can attach and collect at least one section on
# this GPU before we burn time on the full sweep. Uses --set basic
# which is the cheapest section set. If this fails, see the
# TROUBLESHOOTING block that the script prints on error.
run_diag() {
    echo ""
    echo ">>> Diagnostic probe: can ncu attach at all?"
    local out="${OUT_DIR}/diag.ncu-rep"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" "${NCU}" \
        --force-overwrite \
        --target-processes all \
        --kernel-name "${FUSED_REGEX}" \
        --launch-skip "${WARMUP}" \
        --launch-count 1 \
        --set basic \
        ${NCU_EXTRA_FLAGS} \
        --export "${out}" \
        python "${PY_DRIVER}" \
            --config A --eff-bs 1 --mode 15 --power "${POWER}" \
            --num-splits "${NUM_SPLITS}" --kernel fused \
            --warmup "${WARMUP}" --iters 1
    local rc=$?
    set -e
    if [ ${rc} -ne 0 ]; then
        cat <<'EOF'

============================================================
TROUBLESHOOTING "Failed to prepare kernel for profiling"
============================================================
  1) Is another process using GPU ${GPU}? Check:
       nvidia-smi
     If yes, pick an idle GPU:
       GPU=0 bash examples/profile_parallel_vs_fused.sh

  2) Perf counters may be locked to admin. Try as root:
       sudo -E bash examples/profile_parallel_vs_fused.sh

     Or permanently unlock (admin, persists until reboot):
       sudo sh -c 'echo 1 > /proc/driver/nvidia/params'

     Or permanently in the driver (needs reboot):
       Add NVreg_RestrictProfilingToAdminUsers=0 to
       /etc/modprobe.d/nvidia.conf

  3) MPS or another profiler (CUPTI, Nsight Systems, etc.)
     may be running. Kill with:
       echo quit | nvidia-cuda-mps-control
     and verify nothing else is profiling.

  4) On H100 with MIG: profiling across MIG slices is
     restricted. Use a full-device GPU.

  5) Try a smaller ncu configuration first:
       NCU_EXTRA_FLAGS="--replay-mode application --clock-control none --cache-control none --metrics sm__cycles_elapsed.avg" \
         bash examples/profile_parallel_vs_fused.sh

  6) CUDA 13.2 vs PyTorch-13.0 mismatch is sometimes flagged
     by ncu. Update ncu to match CUDA 13.2, or use the ncu
     shipped with CUDA 13.2:
       NCU=/usr/local/cuda-13.2/bin/ncu bash ...

============================================================
EOF
        echo "Diagnostic probe failed (exit ${rc}). See troubleshooting above."
        exit ${rc}
    fi
    echo ">>> Diagnostic probe OK. Proceeding with full sweep."
}

if [ "${DIAG}" = "1" ]; then
    run_diag
    exit 0
fi

# Always run a cheap probe first so full-sweep failures are caught early
# before we've spent minutes on the heavy --set full passes.
run_diag

# ── Helper: run one ncu profile ──────────────────────────────
#   tag    : name used for the output file
#   kernel : "fused" or "parallel" (drives Python driver dispatch)
#   regex  : ncu --kernel-name filter
#   config : "A" or "B"
#   mode   : 15 or 16
run_ncu() {
    local tag="$1"
    local kernel="$2"
    local regex="$3"
    local config="$4"
    local mode="$5"

    local out="${
    
    
    
    }/${tag}.ncu-rep"

    echo ""
    echo ">>> ${tag}"

    # --launch-skip/--launch-count count ONLY kernels matching
    # --kernel-name, so setup kernels (torch.randn, etc.) don't
    # pollute the offsets. With --launch-skip=${WARMUP} and the
    # Python driver doing ${WARMUP} warmup + ${ITERS} profiled
    # calls, ncu captures exactly the profiled ones.
    CUDA_VISIBLE_DEVICES="${GPU}" "${NCU}" \
        --force-overwrite \
        --target-processes all \
        --kernel-name "${regex}" \
        --launch-skip "${WARMUP}" \
        --launch-count "${ITERS}" \
        --set "${SECTION_SET}" \
        ${NCU_EXTRA_FLAGS} \
        --export "${out}" \
        python "${PY_DRIVER}" \
            --config "${config}" \
            --eff-bs "${EFF_BS}" \
            --mode "${mode}" \
            --power "${POWER}" \
            --num-splits "${NUM_SPLITS}" \
            --kernel "${kernel}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}"

    echo "    report: ${out}"
}

# ── Sweep ────────────────────────────────────────────────────
for MODE in 15 16; do
    if [ "${MODE}" -eq 15 ]; then MODE_TAG="SP2"; else MODE_TAG="SP3"; fi
    for CONFIG in A B; do
        run_ncu "fused_${MODE_TAG}_cfg${CONFIG}_eff${EFF_BS}" \
                "fused"    "${FUSED_REGEX}"    "${CONFIG}" "${MODE}"
        run_ncu "parallel_${MODE_TAG}_cfg${CONFIG}_eff${EFF_BS}_ns${NUM_SPLITS}" \
                "parallel" "${PARALLEL_REGEX}" "${CONFIG}" "${MODE}"
    done
done

echo ""
echo "============================================================"
echo "All profiles done. Reports saved under:"
echo "  ${OUT_DIR}"
echo ""
echo "Interactive analysis (recommended):"
echo "  ncu-ui ${OUT_DIR}/parallel_SP2_cfgA_eff${EFF_BS}_ns${NUM_SPLITS}.ncu-rep"
echo ""
echo "CLI summary, one kernel at a time:"
echo "  ncu --import ${OUT_DIR}/fused_SP2_cfgA_eff${EFF_BS}.ncu-rep --page details"
echo ""
echo "Side-by-side diff (CLI):"
echo "  ncu --import ${OUT_DIR}/fused_SP2_cfgA_eff${EFF_BS}.ncu-rep \\"
echo "      --import ${OUT_DIR}/parallel_SP2_cfgA_eff${EFF_BS}_ns${NUM_SPLITS}.ncu-rep \\"
echo "      --page details --csv > ${OUT_DIR}/compare_SP2_cfgA.csv"
echo ""
echo "What to look at (to pinpoint the overhead vs fused):"
echo "  * Section 'GPU Speed Of Light Throughput'"
echo "       → SM %, Memory %, which one is the bound?"
echo "  * Section 'Launch Statistics'"
echo "       → Grid/Block size, Dynamic Shared Mem per block"
echo "  * Section 'Occupancy'"
echo "       → Theoretical vs achieved; limit (smem / regs / blocks/SM)"
echo "  * Section 'Warp State Statistics'"
echo "       → Stall breakdown: Stall Barrier (__syncthreads/__threadfence),"
echo "         Stall Long Scoreboard (global memory), Stall Short Scoreboard"
echo "         (smem/atomic)"
echo "  * Section 'Memory Workload Analysis'"
echo "       → L2/Device throughput, atomic traffic, smem bank conflicts"
echo "  * Section 'Compute Workload Analysis'"
echo "       → Pipe utilisation (FMA / ALU / FP64)"
echo ""
echo "Likely suspects for the parallel-vs-fused gap:"
echo "  - Occupancy limited by the large dynamic smem (kSmem + chunk_bytes)"
echo "  - Stall Barrier dominating due to the __threadfence before atomicInc"
echo "  - Phase 1 CTAs repeat Stage-2 refinement that fused does only once"
echo "    → visible as 'Pipe Utilisation ALU / Special' for integer radix ops"
echo "============================================================"
