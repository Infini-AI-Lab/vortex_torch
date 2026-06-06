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


def clear_stale_jit_locks(max_age_s: float = 300.0) -> int:
    """Remove torch-extension baton ``lock`` files older than ``max_age_s``.

    torch serializes concurrent builds of the same extension with a ``FileBaton``
    — a ``lock`` file the builder creates and deletes on completion. If that
    builder is **SIGKILL'd** (timeout / OOM / manual kill) before deleting it,
    the file persists and every subsequent build spins forever in
    ``FileBaton.wait()`` ("stuck on JIT"). A lock older than ``max_age_s`` cannot
    belong to a live build (even a cold single-arch compile finishes well under
    5 min), so removing it is safe and unwedges JIT. Returns #files removed.
    """
    import glob
    import time

    root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser(
        "~/.cache/torch_extensions"
    )
    now = time.time()
    removed = 0
    for lock in glob.glob(os.path.join(root, "**", "lock"), recursive=True):
        try:
            if now - os.path.getmtime(lock) > max_age_s:
                os.remove(lock)
                removed += 1
        except OSError:
            pass
    return removed


def warmup_jit(schedule_policy: "str | None" = None, verbose: bool = False) -> dict:
    """Serially compile vortex's **shared** JIT extensions, ONCE, up front.

    The decode planner (``sglang_plan_decode_v2_ext``) and the prefill kernel
    (``sglang_prefill_ext``) are model-independent and shared by every variant,
    so when N variants boot in parallel they all try to build the *same*
    extension into the *same* ``TORCH_EXTENSIONS_DIR`` at once and serialize (or
    wedge) on torch's per-extension build lock. Call this once in a single
    process **before** launching a parallel batch (e.g. the 4-variant iterate
    batch); the children then hit a warm cache and never race.

    The flow-specific kernels (indexer/cache custom-ops, top-k) have per-flow
    cache names, so they don't collide across variants and aren't warmed here.

    Best-effort: a failure to warm one extension is recorded, not raised.
    Returns ``{"plan_decode": ..., "prefill": ...}`` status strings.
    """
    results: dict = {}
    # Self-heal: drop stale baton locks left by previously killed builds so this
    # warmup (and the parallel children after it) don't wedge in FileBaton.wait().
    results["stale_locks_cleared"] = clear_stale_jit_locks()
    try:
        from .indexer.planner_sglang import get_sglang_plan_decode_v2_module
        get_sglang_plan_decode_v2_module(policy_body=schedule_policy, verbose=verbose)
        results["plan_decode"] = "ok"
    except Exception as e:  # noqa: BLE001 — best-effort warmup
        results["plan_decode"] = f"skip: {type(e).__name__}: {e}"
    try:
        from .indexer.prefill_sglang import get_sglang_prefill_module
        get_sglang_prefill_module(verbose=verbose)
        results["prefill"] = "ok"
    except Exception as e:  # noqa: BLE001
        results["prefill"] = f"skip: {type(e).__name__}: {e}"
    return results
