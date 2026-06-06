import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import sglang as sgl


DEFAULT_SCHEDULE_POLICY = r"""
const int static_kv_budget = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);
"""


# Default model. Either a local directory containing ``config.json``
# (treated as the resolved model path) or a HuggingFace repo ID such as
# ``"Qwen/Qwen3-1.7B"`` (in which case ``config.json`` is fetched via
# ``hf_hub_download`` and cached under ``~/.cache/huggingface/hub``).
# Submissions can override via the ``model_path`` key in the engine JSON.
MODEL_PATH = "Qwen/Qwen3-1.7B"


def get_engine(
    *,
    model_path: str = MODEL_PATH,
    mem_fraction_static: float = 0.8,
    vortex_max_seq_lens: int = 20480,
    vortex_block_size: int = 16,
    vortex_topk_val: int = 29,
    vortex_topk_ratio: float | None = None,
    vortex_block_reserved_bos: int = 1,
    vortex_block_reserved_eos: int = 2,
    vortex_workload_chunk_size: int = 32,
    vortex_layers_skip: list[int] | None = None,
    vortex_module_name: str = "example_block_sparse_attention_cls",
    vortex_module_path: str = "submissions/example_block_sparse_attention.py",
    vortex_schedule_policy: str | None = None,
    vortex_impl_backend: str = "triton",
    vortex_use_tensor_core: bool = False,
    kv_cache_dtype: str = "auto",
    **kwargs,
):
    layers_skip = list(range(1)) if vortex_layers_skip is None else vortex_layers_skip
    policy = DEFAULT_SCHEDULE_POLICY if vortex_schedule_policy is None else vortex_schedule_policy

    engine_kwargs = dict(
        model_path=model_path,
        page_size=vortex_block_size,
        vortex_block_size=vortex_block_size,
        vortex_topk_val=vortex_topk_val,
        vortex_max_seq_lens=vortex_max_seq_lens,
        vortex_block_reserved_bos=vortex_block_reserved_bos,
        vortex_block_reserved_eos=vortex_block_reserved_eos,
        vortex_workload_chunk_size=vortex_workload_chunk_size,
        vortex_layers_skip=layers_skip,
        vortex_module_name=vortex_module_name,
        vortex_module_path=vortex_module_path,
        vortex_schedule_policy=policy,
        vortex_impl_backend=vortex_impl_backend,
        vortex_use_tensor_core=vortex_use_tensor_core,
        vortex_dtype="bfloat16",
        vortex_compilation_cache_dir="~/.vortex_compilation_cache",
        enable_vortex_sparsity=True,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend="flashinfer",
        mem_fraction_static=mem_fraction_static,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        tp_size=1,
    )
    if vortex_topk_ratio is not None:
        engine_kwargs["vortex_topk_ratio"] = vortex_topk_ratio

    engine_kwargs.update(kwargs)
    return sgl.Engine(**engine_kwargs)


def get_engine_from_json(config_path: str | Path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Engine config at {config_path} must be a JSON object")
    return get_engine(**config)


# ---------------------------------------------------------------------------
# Pre-flight checks for an engine JSON
# ---------------------------------------------------------------------------

class EngineConfigError(ValueError):
    """Raised when an engine JSON fails any pre-flight validation step."""


def _coerce_int(x: Any, label: str) -> int:
    """Coerce a JSON-loaded number to an ``int``.

    Accepts ints and other numerics that round-trip exactly (e.g. ``2.0``
    is treated as ``2``); rejects ``bool``, strings, ``None``, and floats
    that would silently truncate (e.g. ``1.5``).
    """
    if isinstance(x, bool) or x is None or isinstance(x, str):
        raise EngineConfigError(f"{label} must be an int, got {x!r}")
    try:
        i = int(x)
    except (TypeError, ValueError):
        raise EngineConfigError(f"{label} must be an int, got {x!r}")
    if i != x:
        raise EngineConfigError(f"{label} must be a whole number, got {x!r}")
    return i


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _resolve_module_path(json_path: Path, raw: str) -> Path:
    """``vortex_module_path`` may be absolute, CWD-relative, or relative to
    the JSON file. Try each in that order and return the first that exists."""
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, json_path.parent / p, p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise EngineConfigError(
        f"vortex_module_path={raw!r} not found "
        f"(tried: {[str(c) for c in candidates]})"
    )


def _check_registers_class(module_path: Path, module_name: str) -> None:
    """Heuristic source-level check that the file declares
    ``@register("<module_name>")`` somewhere — catches typos before we run
    the file."""
    src = module_path.read_text(encoding="utf-8")
    pattern = rf"@register\(\s*[\"']{re.escape(module_name)}[\"']\s*\)"
    if re.search(pattern, src) is None:
        raise EngineConfigError(
            f"{module_path} does not contain @register({module_name!r})"
        )


_INDEXER_SAVE_PATTERN = re.compile(r"\bSave\s*\(")


def _flow_uses_indexer_save(module_path: Path) -> bool:
    """Return True iff the submission file contains an indexer-side ``Save(``
    call (the persistent state pattern paired with ``Load`` and
    ``CFill(0.0)``). Cache side has no ``Save`` op, so this textual match
    is unambiguous."""
    try:
        src = module_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(_INDEXER_SAVE_PATTERN.search(src))


def _check_disable_radix_cache(module_path: Path, config: Dict[str, Any]) -> None:
    """If the flow uses ``Save(...)`` in the indexer, the engine config
    must set ``disable_radix_cache: true``. Otherwise sglang's prefix
    radix cache will share the Save'd per-request persistent state
    across requests with matching prompt prefixes — silently corrupting
    Save/Load values across decode batches."""
    if not _flow_uses_indexer_save(module_path):
        return
    if config.get("disable_radix_cache", False) is not True:
        raise EngineConfigError(
            f"{module_path.name} uses `Save(...)` in the indexer "
            f"(persistent per-request state via Save/Load), so the "
            f"engine config must set `\"disable_radix_cache\": true`. "
            f"Without it, sglang's radix cache will share Save'd state "
            f"across requests with matching prompt prefixes and corrupt "
            f"the values. Add `\"disable_radix_cache\": true` to the JSON."
        )


def _read_hf_model_shapes(model_path: str) -> Dict[str, int]:
    """Resolve ``config.json`` for a model and return the GQA shapes
    used to drive the compile sweep.

    ``model_path`` is interpreted as a local directory when that
    directory exists; otherwise it is treated as a HuggingFace repo ID
    and ``config.json`` is fetched via
    :func:`huggingface_hub.hf_hub_download` (cached under the standard
    HF cache so the network is only hit once).

    Returns
    -------
    dict
        ``{"G": int, "num_kv_heads": int, "head_dim": int}`` where

        - ``num_kv_heads`` = ``config.num_key_value_heads``
        - ``G``            = ``num_attention_heads // num_key_value_heads``
        - ``head_dim``     = ``config.head_dim`` if present, else
          ``hidden_size // num_attention_heads``

    Raises
    ------
    EngineConfigError
        If the file can't be located, parsed, or doesn't carry the
        required fields.
    """
    local_dir = Path(model_path).expanduser()
    if local_dir.is_dir():
        cfg_path = local_dir / "config.json"
        if not cfg_path.is_file():
            raise EngineConfigError(
                f"local model directory {local_dir} has no config.json"
            )
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise EngineConfigError(
                "huggingface_hub is required to fetch the model config; "
                "install it with `pip install huggingface_hub`"
            ) from e

        try:
            cfg_path = Path(hf_hub_download(repo_id=model_path, filename="config.json"))
        except Exception as e:
            raise EngineConfigError(
                f"failed to fetch config.json from HuggingFace repo "
                f"{model_path!r}: {e}"
            ) from e

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EngineConfigError(f"{cfg_path} is not valid JSON: {e}") from e
    if not isinstance(cfg, dict):
        raise EngineConfigError(f"{cfg_path} must be a JSON object")

    try:
        nq = _coerce_int(cfg["num_attention_heads"], "config.num_attention_heads")
        nkv = _coerce_int(cfg["num_key_value_heads"], "config.num_key_value_heads")
    except KeyError as e:
        raise EngineConfigError(
            f"{cfg_path} missing required field {e.args[0]!r}"
        ) from e
    if nkv <= 0 or nq % nkv != 0:
        raise EngineConfigError(
            f"{cfg_path}: num_attention_heads={nq} must be a positive "
            f"multiple of num_key_value_heads={nkv}"
        )

    # MLA geometry (DeepSeek-V2/V3, GLM glm4_moe*): a single fused latent KV head
    # replaces per-head k/v, so there is no scalar ``head_dim = hidden/heads``.
    # The fused latent is [ kv_c (kv_lora_rank) | k_pe (qk_rope_head_dim) ] and the
    # absorbed query has matching inner dim. Detect it by the LoRA-rank field and
    # hand the verify sweep the latent dims instead of a GQA head_dim.
    if cfg.get("kv_lora_rank"):
        kv_lora_rank = _coerce_int(cfg["kv_lora_rank"], "config.kv_lora_rank")
        try:
            qk_rope_head_dim = _coerce_int(
                cfg["qk_rope_head_dim"], "config.qk_rope_head_dim"
            )
        except KeyError as e:
            raise EngineConfigError(
                f"{cfg_path} has kv_lora_rank (MLA) but missing {e.args[0]!r}"
            ) from e
        return {
            "mla": True,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "num_q_heads": nq,
            "latent_dim": kv_lora_rank + qk_rope_head_dim,
        }

    if "head_dim" in cfg:
        D = _coerce_int(cfg["head_dim"], "config.head_dim")
    else:
        try:
            hidden = _coerce_int(cfg["hidden_size"], "config.hidden_size")
        except KeyError as e:
            raise EngineConfigError(
                f"{cfg_path} has no head_dim and missing {e.args[0]!r} "
                f"(needed to derive head_dim = hidden_size // num_attention_heads)"
            ) from e
        if hidden % nq != 0:
            raise EngineConfigError(
                f"{cfg_path}: hidden_size={hidden} not divisible by "
                f"num_attention_heads={nq}"
            )
        D = hidden // nq

    return {"G": nq // nkv, "num_kv_heads": nkv, "head_dim": D}


def _check_compilable(
    module_path: Path,
    module_name: str,
    model_path: str,
    config: Dict[str, Any] | None = None,
) -> None:
    """Load the user file, build the registered vFlow, and run a tiny
    compile sweep using the GQA shapes from ``model_path``'s
    ``config.json`` (local file when ``model_path`` is a directory,
    otherwise downloaded via huggingface_hub). Uses CPU-side metadata
    only — no CUDA required."""
    # Imports are local: avoid pulling vortex_torch.flow at module-import
    # time of vortex_torch.engine, since that drags in the whole compiler
    # surface for callers that only want ``get_engine``.
    from ...flow.loader import build_vflow
    from ...flow.verify import verify_flow_compilable
    from ...flow.registry import has as _is_registered

    # Re-executing the file would re-run its top-level ``@register`` and
    # trip the "already exists" guard. Once the name is in the registry we
    # can fetch the class directly — step 7 already verified the file is
    # the source of truth for the @register declaration.
    user_file = None if _is_registered(module_name) else str(module_path)

    try:
        flow = build_vflow(module_name, user_file=user_file)
    except Exception as e:
        raise EngineConfigError(
            f"failed to build vFlow {module_name!r} from {module_path}: {e}"
        ) from e

    shapes = _read_hf_model_shapes(model_path)

    # Honour the JSON-declared attention backend so trtllm-only ops
    # (e.g. ``TopK(k)``) survive verify's profile() asserts.
    vortex_attention_backend = (
        (config or {}).get("vortex_attention_backend") or "flashinfer"
    )
    vortex_impl_backend = (
        (config or {}).get("vortex_impl_backend") or "triton"
    )
    vortex_use_tensor_core = bool((config or {}).get("vortex_use_tensor_core", False))
    # Mirror the runtime guard in indexer.Context.create: tensor-core
    # (bf16-compute + tl.dot) codegen is triton-only.
    if vortex_use_tensor_core and vortex_impl_backend != "triton":
        raise EngineConfigError(
            "vortex_use_tensor_core is only supported with "
            f"vortex_impl_backend='triton'; got '{vortex_impl_backend}'."
        )
    if shapes.get("mla"):
        # MLA: a single fused latent head. Synthesize q as [B, H, latent_dim]
        # (G = num q heads so the flow's head-average over dim=1 is exercised),
        # num_kv_heads=1, D = latent_dim. ``mla_dims`` switches verify to the MLA
        # ``initialize(block, kv_lora_rank, qk_rope_head_dim, ...)`` signature.
        G = shapes["num_q_heads"]
        num_kv_heads = 1
        D = shapes["latent_dim"]
        mla_dims = (shapes["kv_lora_rank"], shapes["qk_rope_head_dim"])
        # block sizes must divide the smallest sub-block granularity an MLA flow
        # might use (e.g. LServe/Quest sub-blocks of 16); sweep 16 and 32.
        block_sizes = (16, 32)
    else:
        G, num_kv_heads, D = shapes["G"], shapes["num_kv_heads"], shapes["head_dim"]
        mla_dims = None
        block_sizes = (16,)

    with tempfile.TemporaryDirectory(prefix="vortex_check_") as cache_dir:
        report = verify_flow_compilable(
            flow,
            B=2, num_kv_heads=num_kv_heads,
            G_values=(G,), D_values=(D,),
            block_sizes=block_sizes, page_block_ratios=(1,),
            pages_per_workload_values=(16, 32),
            max_num_pages_per_request=64,
            max_new_tokens_per_batch=64,
            cache_dir=cache_dir,
            verify_indexer=True, verify_cache=True,
            vortex_attention_backend=vortex_attention_backend,
            vortex_impl_backend=vortex_impl_backend,
            vortex_use_tensor_core=vortex_use_tensor_core,
            mla_dims=mla_dims,
        )
        if not report.ok:
            first = report.failed[0]
            raise EngineConfigError(
                f"vFlow {module_name!r} failed to compile "
                f"(num_kv_heads={num_kv_heads}, G={G}, D={D}, "
                f"{first.phase}, cfg {first.cfg.label()}):\n"
                f"{first.traceback}"
            )


def check_engine_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate an engine JSON config end-to-end.

    Steps:
      1. JSON path exists and parses to a dict.
      2. ``vortex_block_size`` is a positive power of 2.
      3. ``vortex_workload_chunk_size`` is a positive power of 2.
      4. ``vortex_block_reserved_bos`` and ``vortex_block_reserved_eos``
         are ints ``>= 1``.
      5. ``vortex_layers_skip`` is missing/None/empty or a list of ints.
      6. ``vortex_module_path`` resolves to an existing file.
      7. That file declares ``@register("<vortex_module_name>")``.
      8. The registered vFlow compiles on a 1-config smoke sweep.
      9. If the flow uses ``Save(...)`` in the indexer, the JSON sets
         ``"disable_radix_cache": true`` (otherwise sglang's prefix
         cache would share per-request persistent state across
         requests with matching prompt prefixes).

    Returns the parsed config dict on success. Raises
    :class:`EngineConfigError` with a descriptive message on the first
    failing step.
    """
    # 1. JSON path exists
    json_path = Path(config_path).expanduser()
    if not json_path.is_file():
        raise EngineConfigError(f"config not found: {json_path}")
    try:
        config = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EngineConfigError(f"{json_path} is not valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise EngineConfigError(f"{json_path} must be a JSON object")

    # 2. vortex_block_size = 2**n
    block_size = _coerce_int(config.get("vortex_block_size"), "vortex_block_size")
    if not _is_pow2(block_size):
        raise EngineConfigError(
            f"vortex_block_size must be a positive power of 2, got {block_size!r}"
        )

    # 3. vortex_workload_chunk_size = 2**k
    chunk = _coerce_int(
        config.get("vortex_workload_chunk_size"), "vortex_workload_chunk_size"
    )
    if not _is_pow2(chunk):
        raise EngineConfigError(
            f"vortex_workload_chunk_size must be a positive power of 2, got {chunk!r}"
        )

    # 4. vortex_block_reserved_{bos,eos} >= 1, ints
    for key in ("vortex_block_reserved_bos", "vortex_block_reserved_eos"):
        v = _coerce_int(config.get(key), key)
        if v < 1:
            raise EngineConfigError(f"{key} must be an int >= 1, got {v!r}")

    # 5. vortex_layers_skip absent/None/empty or list of ints
    layers_skip = config.get("vortex_layers_skip", None)
    if layers_skip not in (None, []):
        if not isinstance(layers_skip, list):
            raise EngineConfigError(
                f"vortex_layers_skip must be empty or a list of ints, got {layers_skip!r}"
            )
        # Validate each element via _coerce_int so 0.0 / "0" / etc. give a
        # clean per-element error.
        for idx, x in enumerate(layers_skip):
            _coerce_int(x, f"vortex_layers_skip[{idx}]")

    # 6. vortex_module_path exists
    raw_module_path = config.get("vortex_module_path")
    if not isinstance(raw_module_path, str) or not raw_module_path:
        raise EngineConfigError(
            f"vortex_module_path must be a non-empty string, got {raw_module_path!r}"
        )
    module_path = _resolve_module_path(json_path, raw_module_path)

    # 7. file declares @register("<name>")
    module_name = config.get("vortex_module_name")
    if not isinstance(module_name, str) or not module_name:
        raise EngineConfigError(
            f"vortex_module_name must be a non-empty string, got {module_name!r}"
        )
    _check_registers_class(module_path, module_name)

    # 8. registered class compiles. Honor a per-submission ``model_path``
    # override so the preflight reads the same model that will be used
    # at runtime.
    model_path = config.get("model_path", MODEL_PATH)
    if not isinstance(model_path, str) or not model_path:
        raise EngineConfigError(
            f"model_path must be a non-empty string, got {model_path!r}"
        )
    _check_compilable(module_path, module_name, model_path, config=config)

    # 9. Save() in indexer ⇒ disable_radix_cache must be true
    _check_disable_radix_cache(module_path, config)

    return config
