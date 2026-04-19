#!/usr/bin/env bash
# ============================================================
# Run examples/profile_parallel_vs_fused.sh inside an NVIDIA
# CUDA devel container so we can enable profiling without
# touching the host's RmProfilingAdminOnly=1 setting.
#
# Key idea:
#   - The container has `ncu` bundled with the CUDA toolkit.
#   - --cap-add=SYS_ADMIN gives the container the capability
#     CUPTI needs to access perf counters, so ncu works
#     regardless of the host's nvidia-driver profiling restriction.
#   - We mount the host's uv venv and the project, so there's
#     no Python/pytorch install inside the container — the host
#     venv's python is used directly.
#
# Image:
#   Defaults to an NGC public CUDA devel image. For B200 (Blackwell /
#   sm_100) you need CUDA ≥ 12.8 and ncu ≥ 2024.3; CUDA 13.0+ covers
#   that. Override with NCU_IMAGE if you prefer a specific tag.
#
# Usage:
#   bash examples/profile_in_docker.sh                    # defaults
#   GPU=2 NUM_SPLITS=2 bash examples/profile_in_docker.sh
#   NCU_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3 bash examples/profile_in_docker.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Host venv to reuse. uv venvs have their own python binary under
# $VENV/bin/python3.x that is glibc/libstdc++-compatible with the
# container when using NGC Ubuntu 22.04 / 24.04 images.
VENV_DIR="${VENV_DIR:-/home/zhuominc/xinrui_projects/uv_env/vortex}"

# NGC CUDA devel image on Ubuntu. Has /usr/local/cuda/bin/ncu bundled.
# 13.0.1-devel-ubuntu22.04 is public (no NGC login needed), supports
# B200, and matches the host's CUDA 13.x driver ABI.
#
# Alternatives:
#   nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04    # newer base
#   nvcr.io/nvidia/pytorch:25.03-py3                # if you don't want to
#                                                   #   reuse the host venv
# Host is Ubuntu 24.04 + Python 3.12 (the uv venv points to /usr/bin/python3.12).
# Match the container to that so the venv's symlinked python resolves to a
# compatible interpreter inside the container.
NCU_IMAGE="${NCU_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}"

# Pass-through env vars for the inner profile script. Defaults match
# examples/profile_parallel_vs_fused.sh.
GPU="${GPU:-7}"
EFF_BS="${EFF_BS:-1}"
NUM_SPLITS="${NUM_SPLITS:-2}"
POWER="${POWER:--1.0}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-1}"
SECTION_SET="${SECTION_SET:-full}"

# Inside the container, these mount points give the profile script the
# same absolute paths it sees on the host (so the script doesn't need
# to be container-aware).
MOUNT_ROOT="/home/zhuominc/xinrui_projects"

if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: VENV_DIR not found: ${VENV_DIR}"
    echo "       Set VENV_DIR=/path/to/venv or install the venv."
    exit 1
fi

VENV_PY="$(ls "${VENV_DIR}"/bin/python* 2>/dev/null | head -1 || true)"
if [ -z "${VENV_PY}" ]; then
    echo "ERROR: no python found under ${VENV_DIR}/bin/"
    exit 1
fi

echo "============================================================"
echo "Docker-wrapped ncu profiling"
echo "  image:        ${NCU_IMAGE}"
echo "  venv:         ${VENV_DIR}  (python=${VENV_PY##*/})"
echo "  project:      ${PROJECT_DIR}"
echo "  GPU:          ${GPU}"
echo "  eff_bs:       ${EFF_BS}"
echo "  num_splits:   ${NUM_SPLITS}"
echo "  power:        ${POWER}"
echo "  warmup/iters: ${WARMUP}/${ITERS}"
echo "  section set:  ${SECTION_SET}"
echo "============================================================"

# Pull the image up-front (so the output during the run isn't
# interleaved with pull progress). `|| true` — pull is optional;
# if the image is already local, docker run will use the cached copy.
docker pull "${NCU_IMAGE}" || true

# Run the profile script inside the container.
#
#   --gpus all            : give the container access to all GPUs
#                           (CUDA_VISIBLE_DEVICES inside the script
#                           narrows it down to GPU ${GPU}).
#   --cap-add=SYS_ADMIN   : lets CUPTI access perf counters without
#                           touching host profiling restrictions.
#   --security-opt seccomp=unconfined : CUPTI needs a few syscalls
#                           the default seccomp profile blocks.
#   --network host        : not strictly required, but keeps pip/uv
#                           network access working if you ever add
#                           pip-install steps.
#   --user $(id -u):$(id -g)
#                         : write output files owned by your user,
#                           not root.
#   -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
#                         : so the uid inside resolves to a real
#                           user (helps some tools, harmless otherwise).
#   -v ${MOUNT_ROOT}:${MOUNT_ROOT}
#                         : mount the whole xinrui_projects tree so
#                           both the project and the venv are visible
#                           at their host paths.
#   -e PYTHONPATH=...     : add the venv's site-packages explicitly
#                           so `python3 -c 'import vortex_torch_C'`
#                           resolves even without activate.
#   -e PATH=...           : put the venv's bin ahead of /usr/local/cuda/bin
#                           so `python` is the venv python, and keep ncu
#                           reachable.
# When invoked via `sudo`, `id -u` returns 0 (root). Prefer SUDO_UID/
# SUDO_GID so the final chown hands results back to the real user,
# not root. Fall back to the effective uid/gid otherwise.
HOST_UID="${SUDO_UID:-$(id -u)}"
HOST_GID="${SUDO_GID:-$(id -g)}"

docker run --rm \
    --gpus all \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --network host \
    --ipc=host \
    -e DISPLAY="${DISPLAY:-}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${MOUNT_ROOT}:${MOUNT_ROOT}" \
    -w "${PROJECT_DIR}" \
    -e GPU="${GPU}" \
    -e EFF_BS="${EFF_BS}" \
    -e NUM_SPLITS="${NUM_SPLITS}" \
    -e POWER="${POWER}" \
    -e WARMUP="${WARMUP}" \
    -e ITERS="${ITERS}" \
    -e SECTION_SET="${SECTION_SET}" \
    -e NCU="/usr/local/cuda/bin/ncu" \
    -e HOST_UID="${HOST_UID}" \
    -e HOST_GID="${HOST_GID}" \
    -e PATH="${VENV_DIR}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
    "${NCU_IMAGE}" \
    bash -lc '
        set -e
        # Ubuntu 24.04 base may not ship python3.12 in the CUDA devel image.
        # Install it idempotently; this is ~2s if missing and skipped otherwise.
        if [ ! -x /usr/bin/python3.12 ]; then
            echo "--- installing python3.12 in container ---"
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq
            apt-get install -y --no-install-recommends python3.12 >/dev/null
        fi
        echo "--- container environment ---"
        echo "python:  $(readlink -f "$(which python)") ($(python --version 2>&1))"
        echo "ncu:     $(which ncu)"
        ncu --version 2>&1 | head -2
        nvidia-smi -L
        python -c "import torch; print(\"torch: \", torch.__version__, \"cuda:\", torch.version.cuda)"
        python -c "import vortex_torch_C; print(\"vortex_torch_C import OK\")"
        echo "-----------------------------"
        bash examples/profile_parallel_vs_fused.sh
        # Hand output files back to the host user (we ran as root so apt
        # could install python3.12).
        chown -R "${HOST_UID}:${HOST_GID}" examples/results 2>/dev/null || true
    '

echo ""
echo "============================================================"
echo "Docker profiling run complete."
echo "Reports are under: ${PROJECT_DIR}/examples/results/"
echo "(same path as the direct script — you own the files since we"
echo " ran the container as your uid)."
echo "============================================================"
