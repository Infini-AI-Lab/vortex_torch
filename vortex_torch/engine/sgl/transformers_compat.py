"""Compatibility fixes required by SGLang's pinned Transformers release."""
from __future__ import annotations

from functools import wraps


class _NullAttentionSink:
    def to(self, *_args, **_kwargs):
        return None


_NULL_ATTENTION_SINK = _NullAttentionSink()


def patch_transformers_560_flash_attention() -> bool:
    """Backport the sole 5.6.1 code fix onto SGLang's pinned 5.6.0.

    Transformers 5.6.0 unconditionally calls ``s_aux.to(...)`` even for
    models without attention sinks. 5.6.1 only adds a None guard. Supplying a
    sentinel whose ``to`` returns None is behaviorally identical and avoids
    copying the upstream attention implementation.
    """
    try:
        import transformers
        from transformers.integrations import flash_attention
    except ImportError:
        return False

    if transformers.__version__ != "5.6.0":
        return False
    original = flash_attention.flash_attention_forward
    if getattr(original, "_vortex_transformers_560_fix", False):
        return True

    @wraps(original)
    def fixed(*args, s_aux=None, **kwargs):
        return original(
            *args,
            s_aux=_NULL_ATTENTION_SINK if s_aux is None else s_aux,
            **kwargs,
        )

    fixed._vortex_transformers_560_fix = True
    flash_attention.flash_attention_forward = fixed

    # modeling_utils copies the function into a class-wide dispatch table.
    # Updating that table covers modules imported both before and after this
    # compatibility hook.
    from transformers.modeling_utils import AttentionInterface

    for name in (
        "flash_attention_2",
        "flash_attention_3",
        "flash_attention_4",
    ):
        AttentionInterface.register(name, fixed)
    return True
