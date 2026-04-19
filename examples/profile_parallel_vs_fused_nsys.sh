#!/usr/bin/env bash
# ============================================================
# Nsight Systems (nsys) profiling — timeline view of the parallel
# vs fused TopK kernels.
#
# Why nsys and not ncu here:
#   ncu needs SM-level perf counters (sm__*), which on this box are
#   gated by the nvidia driver's RmProfilingAdminOnly flag — and we
#   have no sudo. nsys uses CUPTI API/activity tracing and kernel
#   timing, which do NOT require admin. That's enough to answer the
#   "where does the 6-8us overhead come from" question, because we
#   get per-kernel durations, gaps on the stream, memcpy/memset
#   traffic, and NVTX range timing.
#
# Profiles both:
#   - TopKOutput_Fused_Kernel    (csrc/topk_sglang.cu)
#   - TopKOutput_Parallel_Kernel (csrc/topk_sglang_parallel.cu)
#
# For each of mode 15 (SHIFT_POW2), mode 16 (SHIFT_POW3) and both
# configs A (topk=2048 pages=32K) and B (topk=30 pages=2K).
#
# Produces one .nsys-rep per (kernel × mode × config). Open with:
#   nsys-ui <file>.nsys-rep
# or dump CLI summaries with:
#   nsys stats <file>.nsys-rep
#
# Usage:
#   bash examples/profile_parallel_vs_fused_nsys.sh          # defaults
#   GPU=7 NUM_SPLITS=2 bash examples/profile_parallel_vs_fused_nsys.sh
#   ITERS=50 bash examples/profile_parallel_vs_fused_nsys.sh # more samples
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DRIVER="${SCRIPT_DIR}/../benchmarks/profile_parallel_vs_fused.py"

# ── Defaults ──────────────────────────────────────────────────
GPU=${GPU:-7}
EFF_BS=${EFF_BS:-1}
NUM_SPLITS=${NUM_SPLITS:-2}
POWER=${POWER:--1.0}
WARMUP=${WARMUP:-20}
# For nsys we want *many* iterations so the per-kernel timing is
# statistically meaningful and the timeline is readable.
ITERS=${ITERS:-50}

# Prefer the CUDA-13 toolchain's nsys (matches the torch CUDA ABI).
NSYS=${NSYS:-$(command -v nsys || echo /usr/local/cuda/bin/nsys)}
if [ ! -x "${NSYS}" ]; then
    echo "ERROR: nsys not found. Tried: ${NSYS}"
    echo "       Set NSYS=/path/to/nsys manually."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${SCRIPT_DIR}/results/nsys_parallel_vs_fused_${TIMESTAMP}"
mkdir -p "${OUT_DIR}"

# nsys writes intermediate files under $TMPDIR/nvidia/nsight_systems.
# On shared systems /tmp/nvidia is often owned by another user who
# created it first, and we can't write there. Redirect to a
# user-writable cache dir.
export TMPDIR="${TMPDIR:-${HOME}/.cache/nsys_tmp}"
mkdir -p "${TMPDIR}"

echo "============================================================"
echo "Nsight Systems profile: parallel vs fused TopK"
echo "  GPU:          ${GPU}"
echo "  eff_bs:       ${EFF_BS}"
echo "  num_splits:   ${NUM_SPLITS}"
echo "  power (p):    ${POWER}"
echo "  warmup:       ${WARMUP}"
echo "  iters:        ${ITERS}  (profiled launches)"
echo "  nsys binary:  ${NSYS}"
echo "  output dir:   ${OUT_DIR}"
echo "============================================================"
"${NSYS}" --version 2>&1 | head -2

# ── Helper: run one nsys profile ─────────────────────────────
run_nsys() {
    local tag="$1"
    local kernel="$2"
    local config="$3"
    local mode="$4"

    local out="${OUT_DIR}/${tag}"

    echo ""
    echo ">>> ${tag}"

    # --trace cuda,nvtx       : CUDA API/runtime + NVTX ranges. NVTX
    #                           stays on so the timeline still shows
    #                           where the profiled region begins.
    # --sample none / --cpuctxsw none: skip CPU callstack sampling and
    #                           context-switch tracing — both admin-gated
    #                           on this box and we don't need them.
    # --cuda-memory-usage true: log cudaMalloc/cudaFree/cudaMemset so we
    #                           can see if at::empty / at::zeros costs
    #                           anything on the hot path.
    #
    # Capture-range flags intentionally OMITTED. On some nsys builds
    # --capture-range=nvtx silently yields "No reports were generated"
    # when the ranges don't line up exactly; profiling the whole run
    # is more robust and the warmup is easy to filter out later
    # (NVTX range "profile-*" tags the profiled region in nsys stats).
    CUDA_VISIBLE_DEVICES="${GPU}" "${NSYS}" profile \
        --output "${out}" \
        --force-overwrite true \
        --trace cuda,nvtx \
        --sample none \
        --cpuctxsw none \
        --cuda-memory-usage true \
        python "${PY_DRIVER}" \
            --config "${config}" \
            --eff-bs "${EFF_BS}" \
            --mode "${mode}" \
            --power "${POWER}" \
            --num-splits "${NUM_SPLITS}" \
            --kernel "${kernel}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}"

    echo "    report: ${out}.nsys-rep"
}

# ── Sweep ────────────────────────────────────────────────────
for MODE in 15 16; do
    if [ "${MODE}" -eq 15 ]; then MODE_TAG="SP2"; else MODE_TAG="SP3"; fi
    for CONFIG in A B; do
        run_nsys "fused_${MODE_TAG}_cfg${CONFIG}_eff${EFF_BS}" \
                 "fused"    "${CONFIG}" "${MODE}"
        run_nsys "parallel_${MODE_TAG}_cfg${CONFIG}_eff${EFF_BS}_ns${NUM_SPLITS}" \
                 "parallel" "${CONFIG}" "${MODE}"
    done
done

# ── Auto-dump CLI summaries for every report ─────────────────
# `nsys stats` produces text tables that are immediately readable
# and answer most "where did the time go" questions without needing
# the GUI. We dump the most useful ones for every report and stash
# them alongside.
echo ""
echo "============================================================"
echo "Dumping text summaries ('nsys stats') for every report..."
echo "============================================================"
for rep in "${OUT_DIR}"/*.nsys-rep; do
    name="$(basename "${rep}" .nsys-rep)"
    echo ""
    echo ">>> summary for ${name}"
    summary="${OUT_DIR}/${name}.summary.txt"
    {
        echo "### ${name}"
        echo ""
        echo "## cuda_api_sum: CUDA runtime API call distribution"
        echo "##   (count, avg, med, min, max of cudaLaunchKernel / cudaMalloc / etc.)"
        "${NSYS}" stats --report cuda_api_sum --format table "${rep}" 2>&1 || true
        echo ""
        echo "## cuda_gpu_kern_sum: per-kernel GPU duration stats"
        echo "##   (mean/median/std/min/max duration per kernel name, with instance count)"
        "${NSYS}" stats --report cuda_gpu_kern_sum --format table "${rep}" 2>&1 || true
        echo ""
        echo "## cuda_gpu_mem_size_sum: memcpy / memset by size"
        echo "##   (expect 0 memset entries for parallel — no at::zeros on the hot path)"
        "${NSYS}" stats --report cuda_gpu_mem_size_sum --format table "${rep}" 2>&1 || true
        echo ""
        echo "## cuda_gpu_mem_time_sum: memcpy / memset by time"
        "${NSYS}" stats --report cuda_gpu_mem_time_sum --format table "${rep}" 2>&1 || true
        echo ""
        echo "## cuda_kern_exec_sum: kernel launch→exec latency"
        echo "##   (host-side cudaLaunchKernel cost separated from GPU exec cost)"
        "${NSYS}" stats --report cuda_kern_exec_sum --format table "${rep}" 2>&1 || true
        echo ""
        echo "## nvtx_pushpop_sum: NVTX ranges (the 'profile-*' wrapped region)"
        "${NSYS}" stats --report nvtx_pushpop_sum --format table "${rep}" 2>&1 || true
    } > "${summary}" 2>&1
    echo "    saved: ${summary}"
done

echo ""
echo "============================================================"
echo "Reports saved to: ${OUT_DIR}"
echo ""
echo "Quick read — compare fused vs parallel summaries side-by-side:"
echo ""
echo "  diff -y --width=200 \\"
echo "    ${OUT_DIR}/fused_SP2_cfgA_eff${EFF_BS}.summary.txt \\"
echo "    ${OUT_DIR}/parallel_SP2_cfgA_eff${EFF_BS}_ns${NUM_SPLITS}.summary.txt \\"
echo "    | less"
echo ""
echo "Interactive timeline (if you have X11/SSH forwarding):"
echo "  nsys-ui ${OUT_DIR}/parallel_SP2_cfgA_eff${EFF_BS}_ns${NUM_SPLITS}.nsys-rep"
echo ""
echo "What to look for (to nail the overhead vs fused):"
echo "  * 'cuda_gpu_kern_sum' mean duration for each kernel"
echo "      → fused is one kernel × (WARMUP+ITERS), parallel is one kernel × (WARMUP+ITERS)"
echo "        (single-kernel design). Mean duration difference = the GPU work"
echo "        gap (Stage-1 savings minus merge cost)."
echo "  * 'cuda_api_sum' cudaLaunchKernel / cudaMalloc / cudaFree counts"
echo "      → if parallel shows more launches than fused, there's an unexpected"
echo "        extra kernel. Also watch the time spent in cudaLaunchKernel."
echo "  * 'cuda_gpu_mem_size_sum' cudaMemset entries"
echo "      → should be zero for parallel now (__device__ counter removed"
echo "        at::zeros). Any memset here IS overhead we need to explain."
echo "  * 'cuda_kern_exec_sum'"
echo "      → separates host-side cudaLaunchKernel latency from GPU kernel time."
echo "  * 'nvtx_pushpop_sum' profile-* range duration / ${ITERS}"
echo "      → wall-clock per-call including CPU-side overhead."
echo ""
echo "Timeline view (nsys-ui) additionally shows *gaps* between kernels"
echo "on the GPU stream — the cost of __threadfence + atomicInc barrier"
echo "shows up as a visible pause between Phase-1 work and the merge."
echo "============================================================"
