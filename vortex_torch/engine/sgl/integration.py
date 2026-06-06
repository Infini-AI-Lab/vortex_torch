"""Single point of integration between vortex_torch and stock sglang.

Design goal: keep sglang **as close to upstream as possible** so a new sglang
release (e.g. 0.5.12) can be adopted by re-applying a tiny, well-understood set
of hooks rather than a scattered patch. Everything that *can* live outside
sglang lives here; the few things that genuinely cannot (see the report) become
one-line ``# [VORTEX HOOK]`` calls into the functions below.

What this module owns
---------------------
1. :func:`integrate` — registers vortex's attention backends into sglang's
   **public** ``ATTENTION_BACKENDS`` dict (wraps ``flashinfer`` / ``trtllm_mla``
   / ``triton`` with flag-aware shims, and adds ``cuda_mla``). Zero edits to
   ``attention_registry.py``. Called automatically from ``vortex_torch/__init__``
   so merely ``import vortex_torch`` wires sglang — and because sglang spawns its
   scheduler worker (``mp.set_start_method("spawn")``), the in-worker
   ``import vortex_torch`` performed by :func:`build_sparse_flow` re-applies the
   registration in that fresh process, before the backend is selected.
2. :func:`build_sparse_flow` — constructs ``ModelRunner.sparse_attention``.
3. :func:`make_kv_pool` — constructs the vortex KV pool (MLA or MHA).
4. :func:`kv_cell_size` — vortex's KV-cache cell-size for the memory estimate.

These four are the *entire* runtime surface vortex needs from sglang's hot init
path. ``ServerArgs`` fields stay in-source (they must be real dataclass fields:
``Engine(**kwargs)`` builds ``ServerArgs(**kwargs)`` and spawn pickles the
instance), as do two already-duck-typed hooks (``supports_fused_set_kv_buffer``,
``rebuild_aux``) and the int32->int64 ``input_buffers`` upstream bug fix.
"""
from __future__ import annotations

from typing import Optional, Any

_INTEGRATED = False


# ---------------------------------------------------------------------------
# 1. Attention-backend registration (replaces all attention_registry.py edits)
# ---------------------------------------------------------------------------
def _make_flashinfer_shim(orig):
    def create(runner):
        sa = runner.server_args
        # Only the non-MLA + sparsity case is vortex's; everything else (dense
        # flashinfer, dense MLA) is upstream's original creator.
        if (not runner.use_mla_backend) and sa.enable_vortex_sparsity:
            b = sa.vortex_attention_backend
            if b == "flashinfer":
                from .attention_backend import VortexFlashInferBackend
                return VortexFlashInferBackend(runner)
            if b == "trtllm":
                from .attention_backend import VortexTRTLLMBackend
                return VortexTRTLLMBackend(runner)
            raise ValueError(
                f"Unsupported vortex attention backend {b} for sparse attention. "
                "Supported backends are: flashinfer, trtllm."
            )
        return orig(runner)

    return create


def _make_trtllm_mla_shim(orig):
    def create(runner):
        sa = runner.server_args
        if runner.use_mla_backend and sa.enable_vortex_sparsity:
            # trtllm_mla decode path (DeepSeek geometry); prefill via MHA.
            from .attention_backend import VortexTRTLLMMLABackend
            return VortexTRTLLMMLABackend(runner)
        return orig(runner)

    return create


def _make_triton_shim(orig):
    def create(runner):
        sa = runner.server_args
        if runner.use_mla_backend and sa.enable_vortex_sparsity:
            # Geometry-agnostic Triton MLA decode (GLM-4.7-Flash etc.).
            from .attention_backend import VortexTritonMLABackend
            return VortexTritonMLABackend(runner)
        return orig(runner)

    return create


def _create_cuda_mla_backend(runner):
    # New backend name; the hand-written CUDA block-table MLA decode kernel.
    sa = runner.server_args
    if not runner.use_mla_backend or not sa.enable_vortex_sparsity:
        raise ValueError(
            "cuda_mla backend requires an MLA model with enable_vortex_sparsity=True."
        )
    from .attention_backend import VortexCudaMLABackend
    return VortexCudaMLABackend(runner)


def _create_cuda_mla_profile_backend(runner):
    # Profiling twin of cuda_mla: identical decode + per-token per-head
    # p-coverage / recall@N stats. Importing the module self-registers its
    # MHA-prefill dispatch handler. Not cuda-graph compatible (run eager).
    sa = runner.server_args
    if not runner.use_mla_backend or not sa.enable_vortex_sparsity:
        raise ValueError(
            "cuda_mla_profile backend requires an MLA model with "
            "enable_vortex_sparsity=True."
        )
    from .attention_backend.cuda_mla_profile import VortexCudaMLAProfileBackend
    return VortexCudaMLAProfileBackend(runner)


def integrate() -> bool:
    """Register vortex attention backends into sglang's public registry.

    Idempotent and safe to call from any process. Returns True if integration
    is in place, False if sglang could not be imported (e.g. CPU-only tooling).
    """
    global _INTEGRATED
    if _INTEGRATED:
        return True
    try:
        from sglang.srt.layers.attention import attention_registry as AR
    except Exception:
        return False

    B = AR.ATTENTION_BACKENDS  # plain dict: name -> creator(runner)
    # Capture upstream creators and install flag-aware shims that delegate back
    # to them when vortex is off. cuda_mla is brand new.
    if "flashinfer" in B:
        B["flashinfer"] = _make_flashinfer_shim(B["flashinfer"])
    if "trtllm_mla" in B:
        B["trtllm_mla"] = _make_trtllm_mla_shim(B["trtllm_mla"])
    if "triton" in B:
        B["triton"] = _make_triton_shim(B["triton"])
    B["cuda_mla"] = _create_cuda_mla_backend
    B["cuda_mla_profile"] = _create_cuda_mla_profile_backend

    _INTEGRATED = True
    return True


# ---------------------------------------------------------------------------
# 2. ModelRunner.sparse_attention construction  (model_runner.py hook)
# ---------------------------------------------------------------------------
def build_sparse_flow(runner) -> Optional[Any]:
    """Build and initialize ``runner.sparse_attention`` (or return None).

    Mirrors the former in-sglang block. Also (re)applies :func:`integrate` so
    the spawned scheduler worker registers vortex backends before the attention
    backend is selected later in ``ModelRunner.initialize``.
    """
    integrate()
    sa = runner.server_args
    if not sa.enable_vortex_sparsity:
        return None

    import vortex_torch

    flow = vortex_torch.flow.build_vflow(
        sa.vortex_module_name, user_file=sa.vortex_module_path
    )
    if isinstance(flow, vortex_torch.flow.vFlowMLA):
        # MLA flow: latent geometry instead of a single head_dim.
        flow.initialize(
            block_size=runner.block_size,
            kv_lora_rank=runner.model_config.kv_lora_rank,
            qk_rope_head_dim=runner.model_config.qk_rope_head_dim,
            kv_cache_dtype=runner.kv_cache_dtype,
            q_data_type=runner.dtype,
            intermediate_dtype=sa.vortex_dtype,
        )
    else:
        flow.initialize(
            block_size=runner.block_size,
            head_dim=runner.model_config.head_dim,
            kv_cache_dtype=runner.kv_cache_dtype,
            q_data_type=runner.dtype,
            intermediate_dtype=sa.vortex_dtype,
        )
    return flow


# ---------------------------------------------------------------------------
# 3. KV-cache pool construction  (model_runner_kv_cache_mixin.py hook)
# ---------------------------------------------------------------------------
def make_kv_pool(runner):
    """Build the vortex KV pool — MLA (fused latent) or MHA — for ``runner``.

    Called only from the ``enable_vortex_sparsity`` branch of the pool-selection
    chain, so the flag is already known to be set here.
    """
    from sglang.srt.layers.dp_attention import get_attention_tp_size

    common = dict(
        size=runner.max_total_num_tokens,
        page_size=runner.page_size,
        dtype=runner.kv_cache_dtype,
        layer_num=runner.num_effective_layers,
        device=runner.device,
        enable_memory_saver=runner.server_args.enable_memory_saver,
        sparse_attention=runner.sparse_attention,
        model_runner=runner,
        start_layer=runner.start_layer,
        end_layer=runner.end_layer,
    )
    if runner.use_mla_backend:
        from .memory_pool_mla import VortexMLACachePool
        return VortexMLACachePool(
            common["size"],
            page_size=common["page_size"],
            dtype=common["dtype"],
            kv_lora_rank=runner.model_config.kv_lora_rank,
            qk_rope_head_dim=runner.model_config.qk_rope_head_dim,
            layer_num=common["layer_num"],
            device=common["device"],
            enable_memory_saver=common["enable_memory_saver"],
            sparse_attention=common["sparse_attention"],
            model_runner=runner,
            start_layer=common["start_layer"],
            end_layer=common["end_layer"],
        )
    from .memory_pool import VortexCachePool
    return VortexCachePool(
        common["size"],
        page_size=common["page_size"],
        dtype=common["dtype"],
        head_num=runner.model_config.get_num_kv_heads(get_attention_tp_size()),
        head_dim=runner.model_config.head_dim,
        layer_num=common["layer_num"],
        device=common["device"],
        enable_memory_saver=common["enable_memory_saver"],
        sparse_attention=common["sparse_attention"],
        model_runner=runner,
        start_layer=common["start_layer"],
        end_layer=common["end_layer"],
    )


def kv_cell_size(runner, num_layers: int, kv_size: int) -> int:
    """Vortex KV-cache bytes-per-token for the available-memory estimate.

    ``get_token_ratio()`` already encodes (all cache fields) / (the bare KV
    base), so we scale the model's bare per-token KV element count by it. The
    base differs by architecture:

      * **MLA**: the single shared latent ``kv_lora_rank + qk_rope_head_dim``
        (matches sglang's dense-MLA cell size, which ``flow_mla``'s token_ratio
        is defined against — base_bytes = block_size·latent_dim·elem).
      * **MHA/GQA**: ``num_kv_heads · head_dim`` (``flow``'s token_ratio base).
    """
    tr = runner.sparse_attention.get_token_ratio()
    if getattr(runner, "use_mla_backend", False):
        base_elems = (
            runner.model_config.kv_lora_rank + runner.model_config.qk_rope_head_dim
        )
    else:
        from sglang.srt.layers.dp_attention import get_attention_tp_size
        base_elems = (
            runner.model_config.get_num_kv_heads(get_attention_tp_size())
            * runner.model_config.head_dim
        )
    return int(base_elems * num_layers * tr * kv_size)
