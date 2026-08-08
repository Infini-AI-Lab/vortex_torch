"""Independent vortex configuration object.

All vortex hyper-parameters live here, in one dataclass owned by vortex_torch,
instead of as ~18 scattered ``vortex_*`` fields on sglang's ``ServerArgs``.
``ServerArgs`` keeps a single ``vortex: Optional[VortexConfig]`` field (the
spawn-safe channel: sglang pickles ``ServerArgs`` to its worker), plus a small
backward-compatible ``__getattr__`` shim so the many existing
``server_args.vortex_*`` / ``server_args.enable_vortex_sparsity`` read sites keep
working unchanged.

Two entry points populate it:
  * Python: ``sgl.Engine(vortex_topk_val=..., enable_vortex_sparsity=True, ...)``
    still works — :func:`install_serverargs_adapter` folds those flat kwargs into
    a ``VortexConfig`` at the ``ServerArgs`` boundary.
  * Explicit: ``sgl.Engine(vortex=VortexConfig(topk_val=..., ...))``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VortexConfig:
    """All vortex sparse-attention hyper-parameters (defaults mirror the former
    ``ServerArgs.vortex_*`` defaults exactly, so behaviour is unchanged)."""

    topk_val: int = 30
    max_topk_val: Optional[int] = None
    layers_skip: Optional[List[int]] = None
    block_reserved_bos: int = 1
    block_reserved_eos: int = 1
    max_seq_lens: int = -1
    workload_chunk_size: int = 32
    dtype: str = "bfloat16"
    module_path: Optional[str] = None
    module_name: Optional[str] = None
    block_size: int = 16
    topk_ratio: float = 0.0
    compilation_cache_dir: Optional[str] = None
    schedule_policy: Optional[str] = None
    attention_backend: str = "flashinfer"
    impl_backend: str = "triton"
    use_tensor_core: bool = False

    @classmethod
    def from_flat(cls, flat: Dict[str, Any]) -> "VortexConfig":
        """Build from a dict of ``vortex_<name>`` keys (prefix stripped)."""
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in flat.items():
            key = k[len("vortex_"):] if k.startswith("vortex_") else k
            if key in names:
                kw[key] = v
        return cls(**kw)


# The legacy ``vortex_<name>`` -> default map, used by the ServerArgs shim when
# vortex is disabled (so a stray read returns the historical default). Kept in
# sync with the dataclass defaults above; duplicated into server_args.py as a
# plain literal to avoid sglang importing vortex_torch.
def legacy_defaults() -> Dict[str, Any]:
    return {f.name: f.default for f in fields(VortexConfig)}


def split_flat_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[VortexConfig], Dict[str, Any]]:
    """Pop ``enable_vortex_sparsity`` + ``vortex_*`` from ``kwargs``.

    Returns ``(config_or_None, remaining_kwargs)``. The config is built iff
    ``enable_vortex_sparsity`` is truthy; otherwise the vortex_* keys are simply
    dropped (vortex stays off).
    """
    enabled = bool(kwargs.pop("enable_vortex_sparsity", False))
    flat = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith("vortex_") and k != "vortex"}
    cfg = VortexConfig.from_flat(flat) if enabled else None
    return cfg, kwargs


#: Which sglang ``attention_backend`` name vortex should register under, per
#: vortex backend. These are *sglang creator names* — the slot vortex's shim
#: hijacks — not the vortex backend itself, which ``VortexConfig.attention_backend``
#: selects. They are separate concepts that users kept having to reconcile by
#: hand (asking for vortex's "trtllm" meant typing sglang's "flashinfer"), so
#: vortex now fills the sglang name in whenever the caller left it unset.
_SGLANG_BACKEND_FOR: Dict[str, str] = {
    # MHA/GQA: both vortex MHA backends are registered on sglang's "flashinfer"
    # slot (and on "trtllm_mha", which hybrid-GDN models require on Blackwell).
    "flashinfer": "flashinfer",
    "trtllm": "flashinfer",
}


def _default_sglang_backend(kwargs: Dict[str, Any]) -> None:
    """Fill in ``attention_backend`` for a vortex run when the caller omitted it.

    Without this, asking for vortex's ``trtllm`` indexer path also required
    passing sglang's ``attention_backend="flashinfer"`` — the name of the
    registry slot vortex's shim replaces, which has nothing to do with trtllm and
    reads like a mistake. An explicit ``attention_backend`` is always respected;
    MLA runs are left alone because their sglang name (``cuda_mla`` /
    ``triton`` / ``trtllm_mla``) genuinely selects a different decode kernel.
    """
    cfg = kwargs.get("vortex")
    if not isinstance(cfg, VortexConfig):
        return
    if kwargs.get("attention_backend") is not None:
        return
    name = _SGLANG_BACKEND_FOR.get(cfg.attention_backend)
    if name is not None:
        kwargs["attention_backend"] = name


def install_serverargs_adapter() -> bool:
    """Wrap ``ServerArgs.__init__`` so flat ``vortex_*`` kwargs fold into the
    single ``vortex`` field. Idempotent; parent-process only (the spawned worker
    unpickles ``ServerArgs`` and never re-runs ``__init__``). Returns False if
    sglang is unavailable.
    """
    try:
        from sglang.srt.server_args import ServerArgs
    except Exception:
        return False
    if getattr(ServerArgs, "_vortex_adapter_installed", False):
        return True

    _orig_init = ServerArgs.__init__

    def __init__(self, *args, **kwargs):
        v = kwargs.get("vortex")
        if isinstance(v, VortexConfig):
            # Explicit object wins; drop any stray flat vortex_* / enable flag.
            for k in [k for k in kwargs if k.startswith("vortex_")]:
                kwargs.pop(k)
            kwargs.pop("enable_vortex_sparsity", None)
        elif isinstance(v, str):
            # CLI path: --vortex-config '<json>' arrives as a JSON string.
            import json
            kwargs["vortex"] = VortexConfig.from_flat(json.loads(v))
        else:
            # Python path: fold flat vortex_* kwargs (gated by enable flag).
            cfg, kwargs = split_flat_kwargs(kwargs)
            kwargs["vortex"] = cfg
        _default_sglang_backend(kwargs)
        _orig_init(self, *args, **kwargs)

    ServerArgs.__init__ = __init__
    ServerArgs._vortex_adapter_installed = True
    return True


def cfg(model_runner_or_server_args) -> Optional[VortexConfig]:
    """Accessor: return the VortexConfig from a ModelRunner or ServerArgs."""
    sa = getattr(model_runner_or_server_args, "server_args", model_runner_or_server_args)
    return getattr(sa, "vortex", None)
