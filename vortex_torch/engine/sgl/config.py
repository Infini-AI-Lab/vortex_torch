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
    # Which vortex MHA/GQA backend to build. "trtllm" is the default: its
    # block-table decode is the faster path and is what the current sweeps use.
    # "flashinfer" remains fully supported (it is the historical default, and the
    # RULER sweeps cover both), so switching back is a one-flag change.
    attention_backend: str = "trtllm"
    impl_backend: str = "triton"
    use_tensor_core: bool = False
    #: Host (pinned) KV cache size in GiB. ``0`` (default) keeps KV on the GPU.
    #:
    #: When set, the **KV blocks** live in pinned host memory and only the
    #: *selected* blocks are fetched to a small GPU staging pool by a Triton
    #: kernel; the vortex auxiliary cache (centroids / envelopes / Save state)
    #: always stays on the GPU, since the indexer scores every block every step
    #: and streaming that would defeat the purpose. This buys context length —
    #: host memory is both larger and far cheaper than HBM — at the cost of PCIe
    #: traffic on cache misses. See :mod:`vortex_torch.engine.sgl.host_kv`.
    #:
    #: The value sizes the *host* buffer, which is what bounds context. The GPU
    #: staging pool is sized from the per-step selection budget instead (see
    #: ``host_kv_pool_blocks``), because that is what determines how much must be
    #: resident at once.
    host_kv_gb: float = 0.0
    #: GPU staging blocks. ``0`` = derive from the worst-case per-step demand
    #: (recommended). Larger raises the hit rate and the HBM cost.
    host_kv_pool_blocks: int = 0
    #: Eviction policy for the GPU cache over the host KV tier — one of
    #: ``"lru"`` (default), ``"fifo"``, ``"full"``, ``"none"``. See
    #: :mod:`vortex_torch.engine.sgl.cache_policy`. ``lru`` suits the usual case
    #: (a working set larger than the pool, with the sinks and local window
    #: re-selected nearly every step); ``fifo`` is cheaper and cannot be flushed by
    #: one scan-heavy step; ``full`` never evicts and asserts the pool holds
    #: everything, turning the cache into a pure prefetch buffer; ``none`` disables
    #: caching (re-copies every selected block every step) and exists as the control
    #: for measuring what the cache is worth.
    host_kv_policy: str = "lru"
    #: Store K/V as packed INT4 (two 4-bit values per byte along ``head_dim``), with a per-CHANNEL
    #: scale for K and a per-TOKEN scale for V. ``False`` (default) keeps the ``dtype`` above.
    #:
    #: This is a **capacity** knob, not a speed one, and on pre-Blackwell parts it costs a little
    #: throughput. flashinfer's native 4-bit paged-KV decode is Blackwell-only, so on A100 attention
    #: still runs in bf16 and the selected blocks are dequantized on the way in. That is affordable
    #: only because sparse attention dequantizes the SELECTION (topk x block) rather than the whole
    #: context. Measured payload compression is **3.46x** including the fp32 scales, not the nominal
    #: 4x. See :mod:`vortex_torch.engine.sgl.int4_store`.
    #:
    #: Orthogonal to ``host_kv_gb``: the same packed format serves GPU-only, host-KV and the
    #: GPU-cache-over-host tier, which differ only in where the tensors live.
    #:
    #: Accuracy, measured on RULER: **dense INT4 costs 0-3 points** (4K 100%, 16K 97%, 32K 99%),
    #: while under sparsity the gap is 8-15. The difference is the indexer's SELECTION being
    #: perturbed by quantized scoring, not payload error -- so a flow that scores from pre-quant K
    #: should recover most of it.
    kv_int4: bool = False

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


#: vortex MHA/GQA backends, i.e. the ``VortexConfig.attention_backend`` values
#: for which vortex owns an sglang registry slot. Both are registered on the same
#: two slots (see ``integration._make_mha_shim``), so the *sglang* name does not
#: choose between them — ``VortexConfig.attention_backend`` does.
_VORTEX_MHA_BACKENDS = frozenset({"flashinfer", "trtllm"})

#: sglang slot to claim for a vortex MHA/GQA run. ``flashinfer`` is the historical
#: default; hybrid models (some layers linear-attention / RNN) must use
#: ``trtllm_mha`` because upstream restricts their full-attention backend to
#: ``{triton, trtllm_mha, fa4}`` on Blackwell and *asserts* on ``flashinfer``.
_SGLANG_SLOT_DEFAULT = "flashinfer"
_SGLANG_SLOT_HYBRID = "trtllm_mha"


def _default_sglang_backend(kwargs: Dict[str, Any]) -> None:
    """Fill in ``attention_backend`` for a vortex run when the caller omitted it.

    Without this, asking for vortex's ``trtllm`` indexer path also required
    passing sglang's ``attention_backend="flashinfer"`` — the name of the
    registry slot vortex's shim replaces, which has nothing to do with trtllm and
    reads like a mistake. The two are orthogonal: ``VortexConfig.attention_backend``
    picks the vortex backend, this picks which upstream slot it is reached through.

    An explicit ``attention_backend`` is always respected. MLA runs are left
    alone: their sglang name (``cuda_mla`` / ``triton`` / ``trtllm_mla``)
    genuinely selects a different decode kernel, so there is nothing to infer.
    """
    cfg = kwargs.get("vortex")
    if not isinstance(cfg, VortexConfig):
        return
    if kwargs.get("attention_backend") is not None:
        return
    if cfg.attention_backend not in _VORTEX_MHA_BACKENDS:
        return
    kwargs["attention_backend"] = (
        _SGLANG_SLOT_HYBRID
        if _is_hybrid_model(kwargs.get("model_path"))
        else _SGLANG_SLOT_DEFAULT
    )


def _is_hybrid_model(model_path: Optional[str]) -> bool:
    """True when ``model_path`` interleaves full attention with linear/RNN layers.

    Best-effort and deliberately quiet: this only chooses a *default*, and an
    explicit ``attention_backend`` bypasses it entirely. A wrong answer here is
    not silent — picking ``flashinfer`` for a hybrid model trips upstream's own
    assertion with a clear message.

    Reads the HF config only (no weights, no ``ModelConfig``), because this runs
    inside ``ServerArgs.__init__`` before sglang has built either.
    """
    if not model_path:
        return False
    try:
        from transformers import AutoConfig

        from vortex_torch.engine.sgl.compat import full_attention_layer_ids

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return full_attention_layer_ids(hf_config) is not None
    except Exception:
        return False


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
