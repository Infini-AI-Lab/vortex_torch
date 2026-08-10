"""Hand-written patterns for tests, each chosen to break a different assumption.

Building these in one place keeps the tests about *behaviour* rather than about
constructing indices, and makes the coverage auditable at a glance.
"""
from __future__ import annotations

import torch

from vortex_train.pattern import SparsePattern, pattern_from_dense_mask


def _blocks(seqlen_q, seqlen_kv, block_q, block_kv):
    mq = (seqlen_q + block_q - 1) // block_q
    nkv = (seqlen_kv + block_kv - 1) // block_kv
    return mq, nkv


def _causal_block_mask(mq, nkv, block_q, block_kv, seqlen_q, seqlen_kv, device):
    """Block is allowed if any of its tokens is causally visible."""
    m = torch.arange(mq, device=device)
    n = torch.arange(nkv, device=device)
    q_end = (m + 1) * block_q - 1 + (seqlen_kv - seqlen_q)
    kv_start = n * block_kv
    return kv_start[None, :] <= q_end[:, None]


def make(kind, *, b, hkv, seqlen_q, seqlen_kv, block_q, block_kv, topk=4, device="cuda",
         seed=0) -> SparsePattern:
    mq, nkv = _blocks(seqlen_q, seqlen_kv, block_q, block_kv)
    causal = _causal_block_mask(mq, nkv, block_q, block_kv, seqlen_q, seqlen_kv, device)
    base = causal[None, None].expand(b, hkv, mq, nkv).clone()

    if kind == "full":
        # dense degeneracy: everything causally allowed. Must equal dense attention.
        mask = base
    elif kind == "diagonal":
        # only the query block's own kv block — the minimal non-trivial pattern
        m = torch.arange(mq, device=device)
        mask = torch.zeros_like(base)
        diag = (m * block_q // block_kv).clamp(max=nkv - 1)
        mask[:, :, m, diag] = True
        mask &= base
    elif kind == "strided":
        n = torch.arange(nkv, device=device)
        mask = base & (n % 2 == 0)[None, None, None, :]
        # never leave a row empty: keep the diagonal too
        m = torch.arange(mq, device=device)
        mask[:, :, m, (m * block_q // block_kv).clamp(max=nkv - 1)] = True
        mask &= base
    elif kind == "bos_local":
        # the realistic recipe: sink + local window
        n = torch.arange(nkv, device=device)
        m = torch.arange(mq, device=device)
        own = (m * block_q // block_kv).clamp(max=nkv - 1)
        mask = torch.zeros_like(base)
        mask[:, :, :, 0] = True                                   # BOS sink
        for off in range(2):                                      # local window
            mask[:, :, m, (own - off).clamp(min=0)] = True
        mask &= base
    elif kind == "random_topk":
        g = torch.Generator(device=device).manual_seed(seed)
        score = torch.rand((b, hkv, mq, nkv), device=device, generator=g)
        score = score.masked_fill(~base, -1.0)
        k = min(topk, nkv)
        sel = score.topk(k, dim=3).indices
        mask = torch.zeros_like(base)
        mask.scatter_(3, sel, True)
        mask &= base
    elif kind == "empty_rows":
        # pathological: some query blocks select nothing. Exercises the cnt==0
        # path, which must give out=0 / lse=-inf rather than NaN.
        mask = base.clone()
        mask[:, :, ::2, :] = False
    else:
        raise ValueError(f"unknown pattern kind {kind!r}")

    return pattern_from_dense_mask(
        mask, block_q=block_q, block_kv=block_kv,
        seqlen_q=seqlen_q, seqlen_kv=seqlen_kv,
    )


ALL_KINDS = ["full", "diagonal", "strided", "bos_local", "random_topk", "empty_rows"]
