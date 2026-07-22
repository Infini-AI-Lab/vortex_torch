"""Out-of-tree Vortex integration for SGLang 0.5.12.

SGLang discovers :func:`register` through the ``sglang.srt.plugins`` entry
point. Every hook is dormant unless ``ServerArgs.vortex`` is populated, so a
normal SGLang launch is unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import fields, make_dataclass
from typing import Any, Optional

import torch
from sglang.srt.server_args import ServerArgs as _BaseServerArgs

from .config import (
    _REQUIRED_SGLANG_HOOKS,
    VORTEX_SGLANG_ABI,
    VORTEX_SGLANG_PLUGIN_TARGETS,
    VortexConfig,
    legacy_defaults,
    normalize_vortex_config,
    split_flat_kwargs,
    validate_sglang_runtime_contract,
)

logger = logging.getLogger(__name__)

_VORTEX_FIELD = fields(
    make_dataclass(
        "_VortexFieldHolder", [("vortex", Optional[VortexConfig], None)]
    )
)[0]
_VORTEX_LEGACY_DEFAULTS = legacy_defaults()


def _server_args_getattr(self, name: str):
    if name == "enable_vortex_sparsity":
        return self.__dict__.get("vortex") is not None
    if name.startswith("vortex_"):
        key = name[len("vortex_") :]
        vortex = self.__dict__.get("vortex")
        if vortex is not None and hasattr(vortex, key):
            return getattr(vortex, key)
        if key in _VORTEX_LEGACY_DEFAULTS:
            return _VORTEX_LEGACY_DEFAULTS[key]
    raise AttributeError(
        f"{type(self).__name__!r} object has no attribute {name!r}"
    )


def _install_server_args_surface() -> None:
    """Add one dynamic dataclass field without replacing SGLang's class.

    The generated upstream ``__init__`` remains intact and is wrapped below.
    Adding the Field to ``__dataclass_fields__`` makes ``fields()``, ``asdict()``,
    ``replace()``, CLI reconstruction, and multiprocessing all preserve the
    Vortex config.
    """
    if "vortex" not in _BaseServerArgs.__dataclass_fields__:
        _BaseServerArgs.__dataclass_fields__["vortex"] = _VORTEX_FIELD
    _BaseServerArgs.vortex = None
    _BaseServerArgs.__getattr__ = _server_args_getattr
    _BaseServerArgs._vortex_sglang_abi = VORTEX_SGLANG_ABI
    _BaseServerArgs._vortex_sglang_hooks = _REQUIRED_SGLANG_HOOKS
    _BaseServerArgs._vortex_sglang_targets = VORTEX_SGLANG_PLUGIN_TARGETS


def _around_server_args_init(original, self, *args, **kwargs):
    vortex = kwargs.pop("vortex", None)
    if vortex is not None:
        for key in list(kwargs):
            if key.startswith("vortex_"):
                kwargs.pop(key)
        kwargs.pop("enable_vortex_sparsity", None)
        vortex = normalize_vortex_config(vortex)
    else:
        vortex, kwargs = split_flat_kwargs(kwargs)
    self.vortex = vortex
    return original(self, *args, **kwargs)


def _around_server_args_post_init(original, self):
    self.vortex = normalize_vortex_config(self.__dict__.get("vortex"))
    if self.vortex is not None:
        if self.vortex.block_size <= 0:
            raise ValueError("vortex.block_size must be positive")
        if self.vortex.topk_val <= 0:
            raise ValueError("vortex.topk_val must be positive")
        if self.vortex.layers_skip is None:
            self.vortex.layers_skip = []
        if self.page_size is None:
            self.page_size = self.vortex.block_size
        if self.disaggregation_mode in ("prefill", "decode"):
            self.disable_overlap_schedule = True
        if self.vortex.attention_backend in (
            "cuda_mla",
            "cuda_mla_profile",
        ):
            self.attention_backend = self.vortex.attention_backend

    result = original(self)
    if self.vortex is not None and self.page_size % self.vortex.block_size:
        raise ValueError(
            "SGLang page_size must be a multiple of vortex.block_size; "
            f"got page_size={self.page_size}, "
            f"block_size={self.vortex.block_size}"
        )
    return result


def _after_add_cli_args(result, parser):
    if any(
        "--vortex-config" in action.option_strings
        for action in parser._actions
    ):
        return None
    parser.add_argument(
        "--vortex-config",
        dest="vortex",
        type=str,
        default=None,
        help="JSON object configuring the Vortex sparse-attention plugin.",
    )
    return None


def _validate_runner(runner) -> None:
    server_args = runner.server_args
    if not server_args.enable_vortex_sparsity:
        return

    from sglang.srt.configs.model_config import is_deepseek_nsa, is_deepseek_v4
    from sglang.srt.platforms import current_platform

    unsupported = []
    if current_platform.is_out_of_tree():
        unsupported.append("an out-of-tree hardware platform")
    if getattr(runner, "is_hybrid_swa", False):
        unsupported.append("hybrid/sliding-window KV pools")
    if getattr(runner, "mambaish_config", None) is not None:
        unsupported.append("Mamba/hybrid-linear KV pools")
    if is_deepseek_nsa(runner.model_config.hf_config):
        unsupported.append("native NSA KV pools")
    if is_deepseek_v4(runner.model_config.hf_config):
        unsupported.append("DeepSeek-V4 compressed KV pools")
    if getattr(server_args, "prefill_only_disable_kv_cache", False):
        unsupported.append("prefill-only cache elision")
    if getattr(server_args, "cpu_offload_gb", 0):
        unsupported.append("KV CPU offload")
    if not runner.spec_algorithm.is_none():
        unsupported.append("speculative decoding")

    if unsupported:
        raise RuntimeError(
            "Vortex does not yet support " + ", ".join(unsupported) + "."
        )

    config = server_args.vortex
    if not config.module_name:
        raise ValueError("vortex.module_name is required")
    if runner.page_size % config.block_size:
        raise ValueError(
            "runner.page_size must be a multiple of vortex.block_size; "
            f"got {runner.page_size} and {config.block_size}"
        )
    if runner.use_mla_backend:
        if runner.kv_cache_dtype != torch.bfloat16:
            raise ValueError("Vortex MLA currently requires a bf16 KV cache")
    elif runner.model_config.v_head_dim != runner.model_config.head_dim:
        raise ValueError(
            "Vortex MHA/GQA currently requires equal K and V head dimensions"
        )


def _after_configure_kv_cache_dtype(result, runner, *args, **kwargs):
    if not runner.server_args.enable_vortex_sparsity:
        return None
    validate_sglang_runtime_contract()
    _validate_runner(runner)
    runner.block_size = runner.server_args.vortex.block_size
    from .integration import build_sparse_flow

    runner.sparse_attention = build_sparse_flow(runner)
    return None


def _around_compute_cell_size(original, configurator, runner, num_layers):
    if not runner.server_args.enable_vortex_sparsity:
        return original(configurator, runner, num_layers)
    if getattr(runner, "sparse_attention", None) is None:
        raise RuntimeError("Vortex sparse flow was not initialized before KV sizing")
    from .integration import kv_cell_size

    element_size = torch._utils._element_size(runner.kv_cache_dtype)
    return kv_cell_size(runner, num_layers, element_size)


def _around_init_pools(original, runner, *args, **kwargs):
    if not runner.server_args.enable_vortex_sparsity:
        return original(runner, *args, **kwargs)

    # Upstream selects these classes inside one large pool-initialization
    # method. Replace only its local class bindings while that method runs, so
    # the Vortex pool is allocated directly and no duplicate dense pool exists.
    globals_dict = original.__globals__
    original_mha = globals_dict["MHATokenToKVPool"]
    original_mla = globals_dict["MLATokenToKVPool"]

    def make_pool(*_args, **_kwargs):
        from .integration import make_kv_pool

        return make_kv_pool(runner)

    globals_dict["MHATokenToKVPool"] = make_pool
    globals_dict["MLATokenToKVPool"] = make_pool
    try:
        result = original(runner, *args, **kwargs)
    finally:
        globals_dict["MHATokenToKVPool"] = original_mha
        globals_dict["MLATokenToKVPool"] = original_mla

    if not getattr(runner.token_to_kv_pool, "is_vortex_pool", False):
        raise RuntimeError(
            "SGLang selected a KV pool family that the Vortex plugin did not replace"
        )
    return result


def _around_enable_fused_set_kv_buffer(original, forward_batch):
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if getattr(pool, "supports_fused_set_kv_buffer", True) is False:
        return False
    return original(forward_batch)


def _after_pop_transferred(result, queue, *args, **kwargs):
    if not result:
        return None
    pool = queue.scheduler.token_to_kv_pool_allocator.get_kvcache()
    if not hasattr(pool, "rebuild_aux"):
        return None
    table = queue.scheduler.req_to_token_pool.req_to_token
    for req in result:
        loc = table[req.req_pool_idx, : req.kv_committed_len]
        pool.rebuild_aux(loc)
    return None


def register() -> None:
    """Register Vortex's conditional hooks with SGLang."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.server_args import (
        ATTENTION_BACKEND_CHOICES,
        add_attention_backend_choices,
    )

    extra_backends = [
        name
        for name in ("cuda_mla", "cuda_mla_profile")
        if name not in ATTENTION_BACKEND_CHOICES
    ]
    if extra_backends:
        add_attention_backend_choices(extra_backends)

    from .transformers_compat import patch_transformers_560_flash_attention

    patch_transformers_560_flash_attention()
    _install_server_args_surface()

    hooks = (
        (
            "sglang.srt.server_args.ServerArgs.__init__",
            _around_server_args_init,
            HookType.AROUND,
        ),
        (
            "sglang.srt.server_args.ServerArgs.__post_init__",
            _around_server_args_post_init,
            HookType.AROUND,
        ),
        (
            "sglang.srt.server_args.ServerArgs.add_cli_args",
            _after_add_cli_args,
            HookType.AFTER,
        ),
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.configure_kv_cache_dtype",
            _after_configure_kv_cache_dtype,
            HookType.AFTER,
        ),
        (
            "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size",
            _around_compute_cell_size,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._init_pools",
            _around_init_pools,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.utils.enable_fused_set_kv_buffer",
            _around_enable_fused_set_kv_buffer,
            HookType.AROUND,
        ),
        (
            "sglang.srt.disaggregation.decode.DecodeTransferQueue.pop_transferred",
            _after_pop_transferred,
            HookType.AFTER,
        ),
    )
    if {target for target, _, _ in hooks} != VORTEX_SGLANG_PLUGIN_TARGETS:
        raise RuntimeError("Vortex SGLang hook target list is out of sync")
    for target, hook, hook_type in hooks:
        HookRegistry.register(target, hook, hook_type)

    logger.info("Registered Vortex hooks for SGLang 0.5.12")
