"""sglang symbols that moved between releases.

One function per relocated symbol, each resolving the location itself so call
sites read like a plain import. Every shim degrades to the newest known
location, so adding a release means adding a branch here and nothing else.
"""
from __future__ import annotations


def get_attention_tp_size() -> int:
    """Attention tensor-parallel world size.

    * sglang <= 0.5.9 — ``layers.dp_attention.get_attention_tp_size``
    * sglang >= 0.5.16 — removed; the value moved onto the parallel context as
      ``runtime_context.get_parallel().attn_tp_size``.
    """
    try:
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_size as _legacy,
        )
    except ImportError:
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().attn_tp_size
    return _legacy()


def is_draft_extend(forward_mode) -> bool:
    """True when this forward is a speculative draft-extend of either generation.

    vortex supports neither, so its backends assert on this.

    * sglang <= 0.5.9 — ``is_draft_extend(include_v2=False)`` plus a separate
      ``is_draft_extend_v2()``.
    * sglang >= 0.5.16 — ``is_draft_extend`` was removed along with v1
      draft-extend; only ``is_draft_extend_v2`` remains.
    """
    legacy = getattr(forward_mode, "is_draft_extend", None)
    if legacy is not None:
        return bool(legacy())
    v2 = getattr(forward_mode, "is_draft_extend_v2", None)
    return bool(v2()) if v2 is not None else False
