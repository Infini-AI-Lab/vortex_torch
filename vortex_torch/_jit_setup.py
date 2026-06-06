"""One-time JIT build-environment setup for all of vortex's runtime kernels.

Vortex JIT-compiles a *lot* of CUDA at runtime — the decode/prefill planners
(``indexer/planner_sglang.py``, ``indexer/prefill_sglang.py``), the indexer/cache
custom-op kernels, and the top-k kernels (``kernels/topk/*``) — all via
``torch.utils.cpp_extension.load_inline``.

``load_inline`` decides which GPU gencodes to build from ``TORCH_CUDA_ARCH_LIST``.
When that env var is **unset**, torch emits BOTH a SASS gencode
(``code=sm_XX``) *and* a PTX gencode (``code=compute_XX``) for the detected arch
— i.e. two nvcc passes per kernel, and the PTX pass on register-heavy kernels is
the dominant cost. That is why cold compiles take many minutes across every
vortex entrypoint (not just one script).

Pinning ``TORCH_CUDA_ARCH_LIST`` to the current GPU's compute capability (one
SASS gencode, no PTX) roughly halves every vortex JIT compile. We do it once, at
``import vortex_torch``, so *all* JIT sites inherit it. It:

  * respects an explicit user/env ``TORCH_CUDA_ARCH_LIST`` (no-op if set),
  * detects the arch via ``nvidia-smi`` (no CUDA context is created at import),
  * is a silent no-op on hosts without an NVIDIA GPU.
"""
import os
import subprocess


def _detect_compute_cap() -> str | None:
    """First visible GPU's compute capability as a ``"MAJOR.MINOR"`` string
    (e.g. ``"10.0"``), or ``None`` if it can't be determined. Uses nvidia-smi
    (no CUDA init); a node is homogeneous in practice so the first cap applies.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        caps = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return caps[0] if caps else None
    except Exception:
        return None


def configure_jit_env() -> None:
    """Pin ``TORCH_CUDA_ARCH_LIST`` to this GPU's arch if the user hasn't set it.

    Idempotent and best-effort: any failure leaves the environment untouched
    (torch falls back to its default multi-gencode behaviour).
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    cap = _detect_compute_cap()
    if cap:
        os.environ["TORCH_CUDA_ARCH_LIST"] = cap
