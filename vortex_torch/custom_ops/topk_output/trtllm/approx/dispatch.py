"""approxTopK (trtllm/block-table) leaf dispatch.

Accepts a per-call ``tolerate_ratio`` knob — baked into the kernel as a
compile-time substitution (``__TOLERATE_RATIO__``), so each distinct
ratio yields a separately-cached compiled module.
"""
import json
from pathlib import Path

from vortex_torch.custom_ops._jit import get_kernel
from .._abi import CPP_SOURCE

_LEAF_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _LEAF_DIR / "config.json"


def dispatch(tolerate_ratio: float = 0.0, verbose: bool = True):
    """Return the JIT-compiled approxTopK callable for ``tolerate_ratio``."""
    cfg = json.loads(_CONFIG_PATH.read_text())
    subs = dict(cfg.get("substitutions", {}))
    subs["__TOLERATE_RATIO__"] = float(tolerate_ratio)
    return get_kernel(
        _LEAF_DIR / cfg["file"],
        CPP_SOURCE,
        substitutions=subs,
        extra_cuda_cflags=tuple(cfg.get("extra_cuda_cflags", [])),
        verbose=verbose,
    ).topk
