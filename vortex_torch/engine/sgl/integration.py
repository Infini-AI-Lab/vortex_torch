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
def _make_mha_shim(orig):
    """Route the non-MLA + sparsity case to a vortex backend, else defer.

    Installed on every sglang backend name under which a vortex MHA/GQA run may
    be requested. Which vortex backend is built depends on
    ``vortex_attention_backend`` (the indexer/planner path), NOT on the sglang
    name — the name only decides *where upstream would have gone* if sparsity
    were off, which is what ``orig`` preserves.

    Two names carry this shim:

    * ``flashinfer`` — the historical default for homogeneous MHA/GQA models.
    * ``trtllm_mha`` — needed for **hybrid** models (Qwen3.5 and other
      hybrid-GDN architectures): upstream restricts their full-attention backend
      to ``{triton, trtllm_mha, fa4}`` on Blackwell/sm100 and asserts on
      ``flashinfer``, so that name cannot be used to reach vortex there.
    """
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


def _register_mla_forward_method() -> None:
    """Tell DeepSeek-family models to use the MHA prefill path under vortex MLA.

    ``DeepseekV2AttentionMLA.dispatch_attn_forward_method`` picks the extend
    implementation from a registry keyed by attention-backend name
    (``models/deepseek_common/attention_backend_handler.py``). vortex's MLA
    backends (``cuda_mla`` / ``cuda_mla_profile``) are not upstream names, so
    they fall through to the ``triton`` handler — which returns ``MHA`` only when
    the batch has **no cached prefix** and otherwise returns ``MLA`` (the
    weight-absorbed MQA path).

    That absorb path hands the attention backend the *fused latent* K
    (``kv_lora_rank + qk_rope_head_dim``, e.g. 576) instead of per-head K/V,
    which vortex's MHA prefill wrapper cannot consume — it fails with
    ``shape '[-1, H, qk_head_dim]' is invalid`` as soon as sglang's radix cache
    produces a prefix hit. vortex reconstructs prefix K/V from the latent itself
    (``mla_prefill._reconstruct_prefix_kv``), so MHA is correct for **every**
    extend batch, prefix or not.

    Registering a handler is enough — the registry is a plain public dict, so no
    edit to the vendored sglang is needed. No-op on releases without it.
    """
    try:
        from sglang.srt.models.deepseek_common.attention_backend_handler import (
            AttentionBackendRegistry,
            AttnForwardMethod,
        )
    except Exception:
        return

    def handle_attention_vortex_mla(attn, forward_batch):
        # Decode/idle goes through vortex's own sparse decode kernel, which reads
        # the latent directly; MLA is the right (and only) choice there. Every
        # extend batch uses the MHA prefill wrapper.
        if forward_batch.forward_mode.is_decode_or_idle():
            return AttnForwardMethod.MLA
        return AttnForwardMethod.MHA

    for name in ("cuda_mla", "cuda_mla_profile"):
        AttentionBackendRegistry.register(name, handle_attention_vortex_mla)


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

    _register_mla_forward_method()

    B = AR.ATTENTION_BACKENDS  # plain dict: name -> creator(runner)
    # Capture upstream creators and install flag-aware shims that delegate back
    # to them when vortex is off. cuda_mla is brand new.
    #
    # The MHA/GQA shim goes on two names: `flashinfer` (the default for
    # homogeneous models) and `trtllm_mha` (the route for hybrid models, whose
    # full-attention backend upstream restricts to {triton, trtllm_mha, fa4} on
    # Blackwell). Both build the same vortex backend; see _make_mha_shim.
    for name in ("flashinfer", "trtllm_mha"):
        if name in B:
            B[name] = _make_mha_shim(B[name])
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

    # Import the subpackage directly, not `import vortex_torch` + attribute
    # access: this runs inside vortex_torch's own import chain (sglang's hook
    # imports this module, which the package __init__ has not finished binding
    # submodules for), so `vortex_torch.flow` may not exist yet. That surfaced as
    # a bare `AttributeError: module 'vortex_torch' has no attribute 'flow'`.
    from vortex_torch import flow as vortex_flow

    flow = vortex_flow.build_vflow(
        sa.vortex_module_name, user_file=sa.vortex_module_path
    )
    if isinstance(flow, vortex_flow.vFlowMLA):
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
def make_kv_pool(runner, *, max_total_num_tokens=None, layer_info=None,
                 req_to_token_pool=None):
    """Build the vortex KV pool for ``runner``.

    Called only from the ``enable_vortex_sparsity`` branch of the pool-selection
    chain, so the flag is already known to be set here. Three shapes:

    * **MHA / GQA** — a flat :class:`VortexCachePool` over every layer.
    * **MLA** — a :class:`VortexMLACachePool` (fused latent) over every layer.
    * **Hybrid** (some layers linear-attention / RNN, e.g. Qwen3.5) — a vortex
      pool covering *only* the full-attention layers, wrapped in upstream's
      ``HybridLinearKVPool`` alongside the mamba state pool. See
      :func:`_wrap_hybrid`.

    The keyword arguments supply what the runner cannot answer yet, because
    sglang 0.5.16 moved *when* the pool is built (see
    :func:`vortex_torch.engine.sgl.compat.runner_view` for the details). Omit
    them — the sglang <= 0.5.9 call site — and the runner is used as-is.
    """
    from vortex_torch.engine.sgl.compat import (
        full_attention_layer_ids,
        in_span,
        runner_view,
    )

    runner = runner_view(
        runner,
        max_total_num_tokens=max_total_num_tokens,
        layer_info=layer_info,
        req_to_token_pool=req_to_token_pool,
    )

    full_ids = full_attention_layer_ids(runner.model_config)
    if full_ids is None:
        # Homogeneous: every layer attends, vortex owns the whole pool.
        return _make_vortex_pool(runner, layer_num=runner.num_effective_layers)

    # Hybrid: size to this rank's full-attention layers only. Sizing to the total
    # layer count over-allocates by the interleave factor (4x on Qwen3.5, which
    # OOMs a B200 on a 4B model) AND mis-indexes, since vortex addresses its
    # per-layer cache as `layer_id - start_layer`, valid only for a contiguous span.
    full_ids = in_span(full_ids, runner.start_layer, runner.end_layer)
    inner = _make_vortex_pool(runner, layer_num=len(full_ids), start_layer=0)
    return _wrap_hybrid(runner, inner, full_ids)


def _make_vortex_pool(runner, *, layer_num: int, start_layer: Optional[int] = None):
    """The vortex pool itself — MLA (fused latent) or MHA — over ``layer_num`` layers.

    ``start_layer`` defaults to the runner's; hybrid passes 0 because the wrapping
    ``HybridLinearKVPool`` translates global layer ids to dense full-attention
    indices before delegating, so the inner pool only ever sees 0..layer_num-1.
    """
    from vortex_torch.engine.sgl.compat import get_attention_tp_size

    if start_layer is None:
        start_layer = runner.start_layer
    common = dict(
        page_size=runner.page_size,
        dtype=runner.kv_cache_dtype,
        layer_num=layer_num,
        device=runner.device,
        enable_memory_saver=runner.server_args.enable_memory_saver,
        sparse_attention=runner.sparse_attention,
        model_runner=runner,
        start_layer=start_layer,
        end_layer=start_layer + layer_num,
    )
    if runner.use_mla_backend:
        from .memory_pool_mla import VortexMLACachePool
        return VortexMLACachePool(
            runner.max_total_num_tokens,
            kv_lora_rank=runner.model_config.kv_lora_rank,
            qk_rope_head_dim=runner.model_config.qk_rope_head_dim,
            **common,
        )
    from .memory_pool import VortexCachePool
    return VortexCachePool(
        runner.max_total_num_tokens,
        head_num=runner.model_config.get_num_kv_heads(get_attention_tp_size()),
        head_dim=runner.model_config.head_dim,
        **common,
    )


def _wrap_hybrid(runner, inner_pool, full_attention_layer_ids):
    """Compose ``inner_pool`` with the mamba state pool via HybridLinearKVPool.

    Upstream's pool already does exactly what a hybrid model needs — hold a
    full-attention KV pool next to a mamba state pool and remap global layer ids
    onto dense full-attention indices — and accepts an injected ``full_kv_pool``.
    So vortex supplies the full-attention half and inherits the composition,
    rather than reimplementing the interleave.
    """
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    mamba_pool = getattr(runner.req_to_token_pool, "mamba_pool", None)
    if mamba_pool is None:
        raise ValueError(
            "vortex sparsity on a hybrid model requires the hybrid "
            "req-to-token pool (which owns the mamba state pool), but "
            f"{type(runner.req_to_token_pool).__name__} has no `mamba_pool`. "
            "This usually means sglang did not recognise the model as hybrid "
            "while vortex did."
        )
    return HybridLinearKVPool(
        size=runner.max_total_num_tokens,
        page_size=runner.page_size,
        dtype=runner.kv_cache_dtype,
        head_num=inner_pool.head_num,
        head_dim=inner_pool.head_dim,
        full_attention_layer_ids=full_attention_layer_ids,
        device=runner.device,
        mamba_pool=mamba_pool,
        enable_memory_saver=runner.server_args.enable_memory_saver,
        use_mla=runner.use_mla_backend,
        start_layer=runner.start_layer,
        # The load-bearing argument: use vortex's sparse pool for the
        # full-attention layers instead of building a dense MHA one.
        full_kv_pool=inner_pool,
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

    ``num_layers`` is supplied by the caller and is already hybrid-correct: for a
    mambaish model ``DefaultPoolConfigurator`` counts only the full-attention
    layers in this rank's span (8 of 32 on Qwen3.5), which is exactly the set
    vortex allocates for. The mamba state cache is sized separately by upstream
    (it is per-request, not per-token), so it does not belong in this per-token
    figure.
    """
    # Host-resident KV (``vortex_host_kv_gb``): K/V live in pinned host memory, so
    # only the auxiliary fields consume HBM and the *device* cell size shrinks
    # accordingly. That alone is not enough, though — sglang computes
    # ``tokens = hbm_budget // cell_size``, and aux is only ~1.5% of the cache
    # (token_ratio 2.031 vs aux 0.031 for gqa_block_sparse), so a cell size
    # counting aux alone grants ~65x more tokens than the host buffer can back.
    # Measured: it OOMs allocating aux for a context nothing holds the KV for.
    #
    # So the token count is *also* capped by ``host_kv_gb`` — see
    # :func:`host_kv_token_cap`, applied where sglang applies its other external
    # token limits so the allocator and req_to_token_pool are re-derived from the
    # capped value rather than desynced from it.
    from vortex_torch.engine.sgl.config import cfg as _vcfg
    if float(getattr(_vcfg(runner), "host_kv_gb", 0.0) or 0.0) > 0.0:
        tr = runner.sparse_attention.get_aux_token_ratio()
    else:
        tr = runner.sparse_attention.get_token_ratio()
    if getattr(runner, "use_mla_backend", False):
        base_elems = (
            runner.model_config.kv_lora_rank + runner.model_config.qk_rope_head_dim
        )
    else:
        from vortex_torch.engine.sgl.compat import get_attention_tp_size
        base_elems = (
            runner.model_config.get_num_kv_heads(get_attention_tp_size())
            * runner.model_config.head_dim
        )
    return int(base_elems * num_layers * tr * kv_size)


def host_kv_token_cap(runner, num_layers: int, kv_size: int) -> Optional[int]:
    """Max tokens the user's ``vortex_host_kv_gb`` of pinned host memory can hold.

    ``None`` when host KV is off, so the caller leaves the budget untouched.

    This is a **second, independent** bound on the token count. ``kv_cell_size``
    reports only the HBM-resident (auxiliary) bytes under host KV, because that is
    what actually occupies the device — but sglang derives tokens by dividing the
    HBM budget by the cell size, so that alone would grant a context far larger
    than the host buffer can back (~65x for gqa_block_sparse: aux is 1.5% of the
    cache). The host buffer is the other resource, and it is the one the user
    sized, so it caps the result.

    Only K/V (or the fused MLA latent) are charged here: the aux fields stay in
    HBM and are already accounted for by the cell size. Counting them twice would
    silently shrink the context the user paid host memory for.
    """
    from vortex_torch.engine.sgl.config import cfg as _vcfg

    host_gb = float(getattr(_vcfg(runner), "host_kv_gb", 0.0) or 0.0)
    if host_gb <= 0.0:
        return None
    if getattr(runner, "use_mla_backend", False):
        # One fused latent per token per layer (no separate K and V).
        per_token = (
            runner.model_config.kv_lora_rank + runner.model_config.qk_rope_head_dim
        ) * num_layers * kv_size
    else:
        from vortex_torch.engine.sgl.compat import get_attention_tp_size
        per_token = 2 * (
            runner.model_config.get_num_kv_heads(get_attention_tp_size())
            * runner.model_config.head_dim
        ) * num_layers * kv_size
    return max(1, int(host_gb * (1024 ** 3)) // per_token)
