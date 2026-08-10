"""The autograd boundary: ``sparse_attention(q, k, v, pattern)``.

The pattern crosses this boundary as a **non-differentiable constant**. That is
what "hard selection" means: top-k is a routing decision, gradients flow only
through the KV that was selected, and nothing flows into the scorer. It makes the
backward an ordinary FlashAttention variant on a fixed sparse pattern.

What is saved for backward: ``q, k, v, out, lse`` and the pattern (plus its
transpose). Notably **not** the probabilities `p` — those are recomputed from
`lse`. At T=128k, `p` would be ~16 GB per layer; this is the difference between a
system that works at long context and one that OOMs.

The transpose is built in the forward, not the backward, for a reason: it depends
only on the pattern, so building it once keeps the backward free of index
construction, and it fails fast if the pattern is malformed.
"""
from __future__ import annotations

import torch

from ..kernels.bwd import sparse_attn_bwd
from ..kernels.fwd import sparse_attn_fwd
from ..kernels.transpose import transpose_pattern
from ..pattern import SparsePattern


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pattern, causal, softmax_scale):
        out, lse = sparse_attn_fwd(
            q, k, v, pattern, causal=causal, softmax_scale=softmax_scale
        )
        t_offsets, t_indices = transpose_pattern(pattern)
        ctx.save_for_backward(q, k, v, out, lse, t_offsets, t_indices)
        ctx.pattern = pattern          # a frozen dataclass of int32 tensors
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        # Enforce "lse is not differentiable" structurally rather than by checking
        # dlse at run time: autograd now raises on any attempt to backward through
        # lse, at zero per-step cost.
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, do, dlse):
        # `lse` is returned for the backward's own use and for inspection, not as a
        # differentiable output — see the `mark_non_differentiable` call in forward,
        # which makes autograd reject a backward through it before we ever get here.
        #
        # This used to be a runtime check, `if dlse is not None and dlse.abs().any()`.
        # That was a device->host sync on EVERY step (visible in a profile as a
        # reduce_kernel plus a Memcpy DtoH), which is exactly what the no-host-work
        # rule forbids — and it bought nothing that the structural guarantee doesn't
        # already give.
        q, k, v, out, lse, t_offsets, t_indices = ctx.saved_tensors
        dq, dk, dv = sparse_attn_bwd(
            do, q, k, v, out, lse, ctx.pattern, t_offsets, t_indices,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern: SparsePattern,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    return_lse: bool = False,
):
    """Block-sparse attention with a hard (non-differentiable) pattern.

    Shapes: ``q [B, Hq, Sq, D]``, ``k/v [B, Hkv, Skv, D]``, GQA via
    ``Hq % Hkv == 0``. Selection is shared within a GQA group, which is what lets
    one loaded KV tile serve the whole group.
    """
    out, lse = _SparseAttention.apply(q, k, v, pattern, causal, softmax_scale)
    return (out, lse) if return_lse else out
