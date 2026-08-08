"""Recognizing hybrid (full-attention + linear/RNN) models.

Some architectures interleave full attention with a linear-attention / RNN layer
type that keeps a fixed-size recurrent state instead of a growing KV cache —
e.g. Qwen3.5 (`layer_types`: 3x `linear_attention` then `full_attention`, so
full attention lives at layers `[3, 7, 11, ..., 31]` of 32).

Vortex applies to the **full-attention layers only**: the linear layers have no
KV to sparsify, and are already O(1) per token in context length.

Two consequences this module exists to serve:

* the vortex KV pool must be sized to the number of full-attention layers, not
  the total layer count (otherwise it over-allocates by the interleave factor —
  4x on Qwen3.5, which OOMs a B200 on a 4B model);
* the per-layer cache must be indexed by *dense* full-attention position, not by
  global layer id. Vortex indexes as ``layer_id - start_layer``, which is only
  correct for a contiguous span. Upstream's ``HybridLinearKVPool`` already owns
  that remapping, so vortex supplies the inner full-attention pool and lets it
  wrap — see ``integration.make_kv_pool``.
"""
from __future__ import annotations

from typing import Any, List, Optional


def full_attention_layer_ids(model_config) -> Optional[List[int]]:
    """Global layer ids that use full attention, or ``None`` if not hybrid.

    ``None`` (not ``[]``) means "every layer attends" — the homogeneous MHA / MLA
    case — and keeps the non-hybrid code path free of special-casing.

    Resolution order, most authoritative first:

    1. ``mambaish_config(model_config).full_attention_layer_ids`` — what sglang
       itself uses to build ``HybridLinearKVPool`` / ``HybridLinearAttnBackend``.
       Preferring it guarantees vortex and upstream agree on the layer set.
    2. ``layer_types`` on the text config — the HF-native description.
    3. ``full_attention_interval`` — the compact form Qwen3.5 also carries.
    """
    ids = _from_sglang(model_config)
    if ids is not None:
        return ids

    text_config = _text_config(model_config)
    if text_config is None:
        return None

    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        ids = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        # An all-full_attention list is a homogeneous model spelled the long way.
        return ids if len(ids) != len(layer_types) else None

    interval = getattr(text_config, "full_attention_interval", None)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if interval and num_layers and interval > 1:
        # Qwen3.5 places full attention at the END of each group of `interval`.
        return [i for i in range(num_layers) if (i + 1) % interval == 0]

    return None


def _from_sglang(model_config) -> Optional[List[int]]:
    try:
        from sglang.srt.configs.hybrid_arch import mambaish_config
    except ImportError:
        return None
    try:
        cfg = mambaish_config(model_config)
    except Exception:
        # mambaish_config reaches for sglang-populated fields
        # (`linear_attn_registry_result`, ...) that a bare HF config object does
        # not carry. Fall through to the config-driven paths below rather than
        # failing: those describe the same interleave.
        return None
    if cfg is None:
        return None
    ids = getattr(cfg, "full_attention_layer_ids", None)
    return list(ids) if ids else None


def _text_config(model_config) -> Optional[Any]:
    """The text sub-config, for multimodal wrappers that nest it."""
    hf_config = getattr(model_config, "hf_config", model_config)
    getter = getattr(hf_config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    return getattr(hf_config, "text_config", hf_config)


def in_span(layer_ids: List[int], start_layer: int, end_layer: int) -> List[int]:
    """Restrict ``layer_ids`` to this rank's pipeline-parallel layer span."""
    return [i for i in layer_ids if start_layer <= i < end_layer]
