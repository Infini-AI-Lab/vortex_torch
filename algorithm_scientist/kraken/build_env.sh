#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the vortex_torch + sglang 0.5.16 venv INSIDE a Leviathan container,
# then (optionally) run the RULER MHA / MLA sweeps against it.
#
# Runs on a GPU node via `kraken jobs create`. The cluster has internet, so the
# whole stack comes from PyPI in ONE pip resolve — sglang, sglang-kernel, torch,
# flashinfer and transformers are mutually version-locked in 0.5.16, so they
# must be solved together. Never pre-pin torch or upgrade a single package after.
#
# Env in:
#   REPO      path to the vortex_torch checkout (default: $PWD)
#   VENV      where to build the venv (default: $REPO/.venv-0516)
#   SWEEP     "mha" | "mla" | "both" | "none"  (default: none = build only)
#   MODEL_MHA HF id for the MHA sweep (default: Qwen/Qwen3-4B)
#   MODEL_MLA HF id for the MLA sweep (default: zai-org/GLM-4.7-Flash)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-$PWD}"
VENV="${VENV:-$REPO/.venv-0516}"
SWEEP="${SWEEP:-none}"
# vortex_torch requires python >= 3.12; the Leviathan toolkit image defaults to
# 3.10 but ships 3.12 alongside it, so pick 3.12 explicitly.
PYBIN="${PYBIN:-python3.12}"
MODEL_MHA="${MODEL_MHA:-Qwen/Qwen3-4B}"
MODEL_MLA="${MODEL_MLA:-zai-org/GLM-4.7-Flash}"
SGLANG_DIR="$REPO/third_party/sglang/v0.5.16/sglang/python"

echo "===== ENV BUILD START ====="
date -u
echo "REPO=$REPO  VENV=$VENV  SWEEP=$SWEEP  PYBIN=$PYBIN"
[ -d "$SGLANG_DIR" ] || { echo "FATAL: vendored sglang missing at $SGLANG_DIR"; exit 1; }

echo "===== HOST ====="
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true
command -v "$PYBIN" >/dev/null || { echo "FATAL: $PYBIN not found"; exit 1; }
"$PYBIN" -VV

# ---- venv ----------------------------------------------------------------
# --system-site-packages is deliberately NOT used: the base image ships its own
# torch, and we need pip to install 0.5.16's exact pinned stack cleanly.
if [ ! -x "$VENV/bin/python" ]; then
    echo "===== creating venv at $VENV ====="
    "$PYBIN" -m venv "$VENV" || { echo "FATAL: venv creation failed"; exit 1; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel || exit 1

# ---- ONE resolve: sglang + vortex_torch (and everything they pin) ---------
# sglang 0.5.16 added two PyO3 Rust extensions to its build (sglang.srt.grpc._core
# and sglang.srt.multimodal._core) and setuptools-rust to build-requires, so the
# install needs a Rust toolchain it does not otherwise need. Neither extension is
# used by vortex (gRPC serving / multimodal), and upstream's own setup.py gates
# them on SGLANG_BUILD_RUST_EXTS -- "none" skips both. Set BUILD_RUST_EXTS=all if
# you do need them (then a rustc must be on PATH).
export SGLANG_BUILD_RUST_EXTS="${BUILD_RUST_EXTS:-none}"

echo "===== installing sglang 0.5.16 + vortex_torch (single pip resolve) ====="
echo "      SGLANG_BUILD_RUST_EXTS=$SGLANG_BUILD_RUST_EXTS"
date -u
pip install -e "$SGLANG_DIR" -e "$REPO" > "${PIP_LOG:-/tmp/pip_resolve.log}" 2>&1
rc=$?
tail -25 "${PIP_LOG:-/tmp/pip_resolve.log}"
date -u
if [ "$rc" -ne 0 ]; then
    echo "FATAL: pip resolve failed (rc=$rc) -- full log: ${PIP_LOG:-/tmp/pip_resolve.log}"
    exit 1
fi

# ---- verify --------------------------------------------------------------
echo "===== VERIFY ====="
python - <<'PY'
import sys, traceback
def show(mod, attr="__version__"):
    try:
        m = __import__(mod)
        print(f"  {mod:16s}: {getattr(m, attr, '?')}")
        return m
    except Exception as e:
        print(f"  {mod:16s}: IMPORT FAILED {type(e).__name__}: {e}")
        return None

print(f"  python          : {sys.version.split()[0]}")
t = show("torch")
if t is not None:
    print(f"  torch cuda      : {t.version.cuda}  available={t.cuda.is_available()}")
show("transformers"); show("flashinfer"); show("sglang"); show("vortex_torch")

ok = True
try:
    from sglang.srt.server_args import ServerArgs
    assert hasattr(ServerArgs, "_VORTEX_LEGACY_DEFAULTS")
    print("  vortex hooks    : present (ServerArgs._VORTEX_LEGACY_DEFAULTS)")
    sa = ServerArgs.__new__(ServerArgs)
    sa.__dict__["vortex"] = None
    assert sa.enable_vortex_sparsity is False
    assert sa.vortex_block_size == 16
    print("  vortex shim     : legacy vortex_* reads OK")
except Exception:
    ok = False
    traceback.print_exc()

# The moved-API shim must resolve on 0.5.16.
try:
    from vortex_torch.engine.sgl.compat import get_attention_tp_size
    print("  compat shim     : importable")
except Exception:
    ok = False
    traceback.print_exc()

# Attention-backend registration must include vortex's cuda_mla.
try:
    import vortex_torch
    from sglang.srt.layers.attention import attention_registry as AR
    vortex_torch.integration.integrate()
    for name in ("cuda_mla", "cuda_mla_profile", "flashinfer", "trtllm_mla", "triton"):
        print(f"  backend {name:18s}: {'registered' if name in AR.ATTENTION_BACKENDS else 'MISSING'}")
except Exception:
    ok = False
    traceback.print_exc()

print("VERIFY_OK" if ok else "VERIFY_FAILED")
sys.exit(0 if ok else 1)
PY
vrc=$?
echo "===== verify rc=$vrc ====="
[ "$vrc" -ne 0 ] && { echo "FATAL: verification failed — not running sweeps"; exit 1; }

# ---- sweeps --------------------------------------------------------------
cd "$REPO"
export PY="$VENV/bin/python"

run_mha() {
    echo "===== SWEEP MHA ($MODEL_MHA) ====="; date -u
    MODEL="$MODEL_MHA" PY="$PY" bash examples/ruler/sweep_mha.sh
    echo "===== SWEEP MHA rc=$? ====="; date -u
}
run_mla() {
    echo "===== SWEEP MLA ($MODEL_MLA) ====="; date -u
    MODEL="$MODEL_MLA" PY="$PY" bash examples/ruler/sweep_mla.sh
    echo "===== SWEEP MLA rc=$? ====="; date -u
}

case "$SWEEP" in
    mha)  run_mha ;;
    mla)  run_mla ;;
    both) run_mha; run_mla ;;
    none) echo "===== SWEEP=none: build+verify only =====" ;;
    *)    echo "unknown SWEEP=$SWEEP" ;;
esac

echo "===== ALL DONE ====="
date -u
