"""The oracle: masked attention in fp32, written to be obviously correct.

This is the ground truth for every other path. It is deliberately the slowest,
simplest thing that could work — a dense mask, a full score matrix, one softmax —
because its job is to be *auditable*, not fast.

Precision, and why it is fp32 rather than bf16 or fp64:

* the system under test is **bf16** (that is the real training regime);
* a **bf16** oracle would have error the same size as the thing it is checking
  (~2.5e-1 relative at S=4096), so a failure could not be distinguished from
  rounding — the test would not resolve bugs;
* **fp32**'s own error is ~4e-6, about five orders below the bf16 tolerance, so a
  failure means a bug;
* **fp64** is only needed for ``torch.autograd.gradcheck``, where finite
  differences bottom out at ~eps^(2/3): 1.5e-5 in fp32 (false failures on a
  softmax chain) vs 2.3e-11 in fp64. See :func:`oracle_attention` ``dtype``.

The bugs this catches are O(1) relative — a dropped KV block, an off-by-one
causal edge, a mis-counted reserve — not subtle drift, which is why fp32 has
ample margin.
"""
from __future__ import annotations

import math

import torch

from ..pattern import SparsePattern


def oracle_attention(
    q: torch.Tensor,               # [B, Hq, Sq, D]
    k: torch.Tensor,               # [B, Hkv, Skv, D]
    v: torch.Tensor,               # [B, Hkv, Skv, Dv]
    *,
    mask: torch.Tensor | None = None,   # [B, Hkv, Sq, Skv] bool, True = attend
    causal: bool = True,
    softmax_scale: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked attention, computed in ``dtype``. Returns ``(out, lse)``.

    ``out`` is returned in ``q``'s original dtype so it can be compared against a
    kernel directly; ``lse`` stays fp32 (as every attention implementation keeps
    it). GQA is handled by expanding the KV heads, which is wasteful and correct —
    exactly the tradeoff this file is for.
    """
    b, hq, sq, d = q.shape
    hkv = k.shape[1]
    assert hq % hkv == 0, f"Hq {hq} must be divisible by Hkv {hkv}"
    group = hq // hkv
    out_dtype = q.dtype

    qf = q.to(dtype)
    kf = k.to(dtype).repeat_interleave(group, dim=1)      # [B, Hq, Skv, D]
    vf = v.to(dtype).repeat_interleave(group, dim=1)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * scale     # [B, Hq, Sq, Skv]

    neg_inf = torch.finfo(dtype).min
    if causal:
        # token-level causal: query at absolute position (Skv - Sq + i) sees j <= that
        skv = k.shape[2]
        qpos = torch.arange(sq, device=q.device) + (skv - sq)
        kpos = torch.arange(skv, device=q.device)
        scores = scores.masked_fill(kpos[None, None, None, :] > qpos[None, None, :, None], neg_inf)
    if mask is not None:
        assert mask.shape == (b, hkv, sq, k.shape[2]), (
            f"mask {tuple(mask.shape)} != expected {(b, hkv, sq, k.shape[2])}"
        )
        scores = scores.masked_fill(~mask.repeat_interleave(group, dim=1), neg_inf)

    lse = torch.logsumexp(scores.float(), dim=-1)                # [B, Hq, Sq]
    probs = torch.softmax(scores, dim=-1)
    # A fully-masked row (no selected block) softmaxes to uniform garbage; force 0
    # so the comparison is against a defined value rather than NaN-adjacent noise.
    all_masked = torch.isneginf(torch.as_tensor(neg_inf)) | (scores <= neg_inf).all(dim=-1, keepdim=True)
    probs = torch.where(all_masked, torch.zeros_like(probs), probs)

    out = torch.einsum("bhqk,bhkd->bhqd", probs, vf)
    return out.to(out_dtype), lse


def oracle_from_pattern(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern: SparsePattern,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle attention restricted to ``pattern``.

    Expands the pattern to a dense token mask (O(T²) — fine at test shapes) and
    defers to :func:`oracle_attention`. This is the function that makes pattern
    *semantics* testable independently of any kernel: if the causal edge or the
    BOS/EOS reservation is wrong, it shows up here with no kernel involved.
    """
    mask = pattern.to_dense_mask()
    return oracle_attention(
        q, k, v, mask=mask, causal=causal, softmax_scale=softmax_scale, dtype=dtype
    )
