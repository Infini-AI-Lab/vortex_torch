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
    still works — the official SGLang plugin folds those flat kwargs into a
    ``VortexConfig`` at the ``ServerArgs`` boundary.
  * Explicit: ``sgl.Engine(vortex=VortexConfig(topk_val=..., ...))``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple


VORTEX_SGLANG_ABI = 1
_REQUIRED_SGLANG_HOOKS = frozenset(
    {
        "server_args.vortex_config",
        "model_runner.build_sparse_flow",
        "kv_cache.kv_cell_size",
        "kv_cache.make_kv_pool",
        "model_utils.fused_kv_opt_out",
        "disaggregation.rebuild_aux",
    }
)
VORTEX_SGLANG_PLUGIN_TARGETS = frozenset(
    {
        "sglang.srt.server_args.ServerArgs.__init__",
        "sglang.srt.server_args.ServerArgs.__post_init__",
        "sglang.srt.server_args.ServerArgs.add_cli_args",
        "sglang.srt.model_executor.model_runner.ModelRunner.configure_kv_cache_dtype",
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_cell_size",
        "sglang.srt.model_executor.model_runner_kv_cache_mixin.ModelRunnerKVCacheMixin._init_pools",
        "sglang.srt.models.utils.enable_fused_set_kv_buffer",
        "sglang.srt.disaggregation.decode.DecodeTransferQueue.pop_transferred",
    }
)


def validate_sglang_runtime_contract() -> None:
    """Fail fast unless the matching Vortex plugin and hooks are active."""

    try:
        from sglang.srt.plugins import load_plugins

        load_plugins()
        from sglang.srt.server_args import ServerArgs
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Vortex's SGLang plugin is not active. Install vortex_torch "
            "with its 'sglang' extra (sglang==0.5.12.post1)."
        ) from exc

    actual_abi = getattr(ServerArgs, "_vortex_sglang_abi", None)
    actual_hooks = frozenset(getattr(ServerArgs, "_vortex_sglang_hooks", ()))

    missing_hooks = sorted(_REQUIRED_SGLANG_HOOKS - actual_hooks)
    server_arg_fields = {field.name for field in fields(ServerArgs)}

    errors = []
    if actual_abi != VORTEX_SGLANG_ABI:
        errors.append(
            f"ABI {actual_abi!r} is installed; ABI {VORTEX_SGLANG_ABI} is required"
        )
    if missing_hooks:
        errors.append(f"missing hooks: {', '.join(missing_hooks)}")
    if "vortex" not in server_arg_fields:
        errors.append("ServerArgs has no 'vortex' field")
    from sglang.srt.plugins.hook_registry import HookRegistry

    declared_targets = frozenset(
        getattr(ServerArgs, "_vortex_sglang_targets", ())
    )
    missing_declarations = sorted(
        VORTEX_SGLANG_PLUGIN_TARGETS - declared_targets
    )
    missing_applied = sorted(declared_targets - HookRegistry._patched)
    if missing_declarations:
        errors.append(
            "plugin did not declare targets: "
            + ", ".join(missing_declarations)
        )
    if missing_applied:
        errors.append(
            "plugin hooks were not applied: " + ", ".join(missing_applied)
        )

    if errors:
        raise RuntimeError(
            "Incompatible official SGLang plugin runtime for Vortex ("
            + "; ".join(errors)
            + "). Reinstall matching vortex_torch and SGLang packages."
        )


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


def normalize_vortex_config(value: Any) -> Optional[VortexConfig]:
    """Normalize JSON, dict, and explicit object inputs to ``VortexConfig``."""
    if value is None or isinstance(value, VortexConfig):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--vortex-config must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(
            "vortex must be a VortexConfig, JSON object, dict, or None; "
            f"got {type(value).__name__}"
        )
    return VortexConfig.from_flat(value)


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


def cfg(model_runner_or_server_args) -> Optional[VortexConfig]:
    """Return the Vortex config from a ModelRunner or ServerArgs object."""
    sa = getattr(model_runner_or_server_args, "server_args", model_runner_or_server_args)
    return getattr(sa, "vortex", None)
