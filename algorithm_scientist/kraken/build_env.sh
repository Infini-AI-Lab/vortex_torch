#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the vortex_torch + sglang 0.5.16 venv INSIDE a Leviathan container,
# verify the integration, then optionally run the RULER MHA / MLA sweeps.
#
# Runs on a GPU node submitted via `kraken jobs create` (see README.md). The
# cluster has internet, so the whole stack comes from PyPI in ONE pip resolve:
# sglang, sglang-kernel, torch, flashinfer and transformers are mutually
# version-locked in 0.5.16. Never pre-pin torch or upgrade a single package
# afterwards — piecewise installs yield a silently ABI-mismatched env.
#
# Env in:
#   REPO             vortex_torch checkout                (default: $PWD)
#   VENV             where to build the venv               (default: $REPO/.venv-0516)
#   SWEEP            none | mha | mla | both               (default: none = build only)
#   PYBIN            interpreter to build the venv from    (default: python3.12)
#   PIP_LOG          full pip output goes here             (default: $VENV/pip-resolve.log)
#   BUILD_RUST_EXTS  none | all — sglang's PyO3 extensions (default: none)
#   MODEL_MHA        HF id for the MHA sweep               (default: Qwen/Qwen3-4B)
#   MODEL_MLA        HF id for the MLA sweep               (default: zai-org/GLM-4.7-Flash)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="${REPO:-$PWD}"
VENV="${VENV:-$REPO/.venv-0516}"
SWEEP="${SWEEP:-none}"
# vortex_torch needs python >= 3.12; the toolkit image defaults to 3.10 but
# ships 3.12 alongside it, so name it explicitly.
PYBIN="${PYBIN:-python3.12}"
PIP_LOG="${PIP_LOG:-$VENV/pip-resolve.log}"
MODEL_MHA="${MODEL_MHA:-Qwen/Qwen3-4B}"
MODEL_MLA="${MODEL_MLA:-zai-org/GLM-4.7-Flash}"
SGLANG_DIR="$REPO/third_party/sglang/v0.5.16/sglang/python"

die() { echo "FATAL: $*" >&2; exit 1; }
step() { echo; echo "===== $* ====="; date -u; }

show_host() {
    step "HOST"
    hostname
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || true
    command -v "$PYBIN" >/dev/null || die "$PYBIN not found"
    "$PYBIN" -VV
}

create_venv() {
    # --system-site-packages is deliberately NOT used: the base image ships its
    # own torch, and pip must install 0.5.16's pinned stack cleanly.
    if [ ! -x "$VENV/bin/python" ]; then
        step "CREATING VENV at $VENV"
        "$PYBIN" -m venv "$VENV" || die "venv creation failed"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    # Only pip and wheel: torch 2.11 pins setuptools<82, so upgrading setuptools
    # here just forces the resolve below to walk it back down again.
    python -m pip install --upgrade pip wheel >/dev/null || die "pip bootstrap failed"
}

install_stack() {
    # sglang 0.5.16 added two PyO3 Rust extensions (sglang.srt.grpc._core,
    # sglang.srt.multimodal._core) and setuptools-rust to build-requires, so the
    # install wants a Rust toolchain it otherwise doesn't need. vortex uses
    # neither (gRPC serving / multimodal), and upstream's setup.py gates them on
    # SGLANG_BUILD_RUST_EXTS: "none" skips both. Use "all" only with rustc on PATH.
    export SGLANG_BUILD_RUST_EXTS="${BUILD_RUST_EXTS:-none}"

    step "INSTALLING sglang 0.5.16 + vortex_torch (single pip resolve)"
    echo "SGLANG_BUILD_RUST_EXTS=$SGLANG_BUILD_RUST_EXTS  log=$PIP_LOG"
    if ! pip install -e "$SGLANG_DIR" -e "$REPO" > "$PIP_LOG" 2>&1; then
        tail -30 "$PIP_LOG"
        die "pip resolve failed — full log: $PIP_LOG"
    fi
    tail -1 "$PIP_LOG"
}

# Fails loudly if the env imports but the vortex integration is not actually
# live — a broken hook otherwise shows up much later as a wrong RULER score.
verify_env() {
    step "VERIFY"
    python - <<'PY'
import sys
import traceback

failures = []


def check(label, fn):
    try:
        print(f"  {label:22s}: {fn()}")
    except Exception:
        failures.append(label)
        print(f"  {label:22s}: FAILED")
        traceback.print_exc()


def versions():
    import torch, transformers, flashinfer, sglang, vortex_torch
    return (
        f"torch {torch.__version__} (cuda {torch.version.cuda}, "
        f"available={torch.cuda.is_available()}) | "
        f"transformers {transformers.__version__} | "
        f"flashinfer {flashinfer.__version__} | "
        f"sglang {sglang.__version__} | vortex {vortex_torch.__version__}"
    )


def vortex_hooks():
    from sglang.srt.server_args import ServerArgs
    assert hasattr(ServerArgs, "_VORTEX_LEGACY_DEFAULTS"), (
        "sglang imported but the vortex patch is missing — is SGLANG_DIR the "
        "vendored third_party/sglang/v0.5.16 tree?"
    )
    sa = ServerArgs.__new__(ServerArgs)
    sa.__dict__["vortex"] = None
    assert sa.enable_vortex_sparsity is False
    assert sa.vortex_block_size == 16
    return "ServerArgs vortex shim serves legacy vortex_* reads"


def backends():
    import vortex_torch
    from sglang.srt.layers.attention import attention_registry as AR
    vortex_torch.integration.integrate()
    want = ("cuda_mla", "cuda_mla_profile", "flashinfer", "trtllm_mla", "triton")
    missing = [n for n in want if n not in AR.ATTENTION_BACKENDS]
    assert not missing, f"unregistered attention backends: {missing}"
    return f"{len(want)} attention backends registered"


def mla_dispatch():
    # cuda_mla must resolve to vortex's own handler, not upstream's triton
    # fallback (which picks the absorb path on a prefix hit and breaks prefill).
    from sglang.srt.models.deepseek_common.attention_backend_handler import (
        AttentionBackendRegistry,
    )
    h = AttentionBackendRegistry.get_handler("cuda_mla")
    assert h.__name__ == "handle_attention_vortex_mla", (
        f"cuda_mla resolves to {h.__name__}, expected vortex's handler"
    )
    return "cuda_mla -> handle_attention_vortex_mla"


print(f"  {'python':22s}: {sys.version.split()[0]}")
check("versions", versions)
check("vortex hooks", vortex_hooks)
check("attention backends", backends)
check("MLA extend dispatch", mla_dispatch)

print("VERIFY_FAILED: " + ", ".join(failures) if failures else "VERIFY_OK")
sys.exit(1 if failures else 0)
PY
}

run_sweep() {
    local kind="$1" model="$2"
    step "SWEEP ${kind^^} ($model)"
    MODEL="$model" PY="$VENV/bin/python" OUT="${OUT:-$REPO/examples/ruler/sweep_results}" \
        bash "examples/ruler/sweep_${kind}.sh"
    echo "===== SWEEP ${kind^^} rc=$? ====="
}

# --------------------------------------------------------------------------- #
main() {
    step "ENV BUILD START"
    echo "REPO=$REPO  VENV=$VENV  SWEEP=$SWEEP  PYBIN=$PYBIN"
    [ -d "$SGLANG_DIR" ] || die "vendored sglang missing at $SGLANG_DIR"

    show_host
    create_venv
    install_stack
    verify_env || die "verification failed — not running sweeps"

    cd "$REPO" || die "cannot cd $REPO"
    case "$SWEEP" in
        mha)  run_sweep mha "$MODEL_MHA" ;;
        mla)  run_sweep mla "$MODEL_MLA" ;;
        both) run_sweep mha "$MODEL_MHA"; run_sweep mla "$MODEL_MLA" ;;
        none) echo "SWEEP=none: build + verify only" ;;
        *)    die "unknown SWEEP=$SWEEP (want none|mha|mla|both)" ;;
    esac

    step "ALL DONE"
}

main "$@"
