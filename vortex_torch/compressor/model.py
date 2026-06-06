"""Trainable per-head block compressor — flexible, pluggable architectures.

A compressor maps a block's KV (the fused MLA latent) to a per-head descriptor
and scores it against the query. Architectures are registered in
:data:`ARCH_REGISTRY`; add a new one by subclassing :class:`BlockScorer` and
decorating it with ``@register_arch("name")`` — no other code changes. All
scorers share one interface::

    block_logits(q[H, d], latent[T, d], block_size, layer_id, scaling) -> [H, n_blocks]

so the trainer/eval are arch-agnostic (they hand over the raw latent; the scorer
does its own block compression). The default ``bilinear`` arch is a strict
low-rank generalization of the centroid scorer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CompressorConfig

ARCH_REGISTRY: dict = {}


def register_arch(name: str):
    def deco(cls):
        ARCH_REGISTRY[name] = cls
        return cls
    return deco


def _sub_centroids(latent: torch.Tensor, block_size: int, m: int) -> torch.Tensor:
    """``latent`` [T, d] → [n_blocks, m, d]: m sub-block means per block."""
    T, d = latent.shape
    nb = (T + block_size - 1) // block_size
    pad = nb * block_size - T
    Lp = F.pad(latent, (0, 0, 0, pad)).view(nb, block_size, d)         # [nb, bs, d]
    sub = block_size // m
    # pad block_size up to a multiple of m if needed
    if block_size % m:
        Lp = F.pad(Lp, (0, 0, 0, m - block_size % m))
        sub = Lp.shape[1] // m
    return Lp.view(nb, m, sub, d).mean(dim=2)                          # [nb, m, d]


def _block_centroids(latent: torch.Tensor, block_size: int) -> torch.Tensor:
    return _sub_centroids(latent, block_size, 1).squeeze(1)            # [nb, d]


# --------------------------------------------------------------------------- #
class BlockScorer(nn.Module):
    """Base class for block-scoring architectures."""

    def __init__(self, cfg: CompressorConfig):
        super().__init__()
        self.cfg = cfg

    def _proj_init(self, lead, d, r) -> torch.Tensor:
        if self.cfg.init == "identity":
            base = torch.zeros(d, r)
            k = min(d, r)
            base[:k, :k] = torch.eye(k)
            w = base.expand(*lead, d, r).clone() + 0.01 * torch.randn(*lead, d, r)
            return w
        w = torch.randn(*lead, d, r)
        return torch.nn.init.orthogonal_(w.reshape(-1, r)).reshape(*lead, d, r)

    def _slice(self, W, layer_id):
        return W[layer_id] if self.cfg.per_layer else W

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        raise NotImplementedError


@register_arch("bilinear")
class BilinearScorer(BlockScorer):
    r"""s[h,b] = scaling · (Wqᵀ q_h) · (Wkᵀ centroid_b). Centroid = mean pool.
    ``Wk=Wq=I, r=d`` recovers the exact centroid scorer (warm start)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.Wk = nn.Parameter(self._proj_init(lead + (H,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        cent = _block_centroids(latent, block_size)                    # [B, d]
        Wk = self._slice(self.Wk, layer_id); Wq = self._slice(self.Wq, layer_id)
        g = torch.einsum("bd,hdr->hbr", cent.to(Wk.dtype), Wk)         # [H, B, r]
        u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)             # [H, r]
        return scaling * torch.einsum("hr,hbr->hb", u, g)              # [H, B]


@register_arch("landmark")
class LandmarkScorer(BlockScorer):
    r"""m sub-block descriptors per block; block score = max over landmarks.
    Learned generalization of LServe sub-block centroids."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.m = max(1, cfg.num_landmarks)
        self.Wk = nn.Parameter(self._proj_init(lead + (H,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        sub = _sub_centroids(latent, block_size, self.m)               # [B, m, d]
        Wk = self._slice(self.Wk, layer_id); Wq = self._slice(self.Wq, layer_id)
        g = torch.einsum("bmd,hdr->hbmr", sub.to(Wk.dtype), Wk)        # [H, B, m, r]
        u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)            # [H, r]
        s = scaling * torch.einsum("hr,hbmr->hbm", u, g)              # [H, B, m]
        return s.amax(dim=-1)                                          # [H, B]


@register_arch("factorized")
class FactorizedScorer(BlockScorer):
    r"""Learn BOTH compressions: a token-mixing ``Wt`` [block_size, m] collapses
    the block's tokens (block_size → m descriptors, instead of fixed mean pooling)
    and ``Wk`` [d, r] compresses channels. Per block, descriptor

        g[m, r] = (Wtᵀ · L_block) · Wk          L_block ∈ R^{block_size × d}

    and block score = max_m scaling·(Wqᵀ q)·g[m]. ``Wt = 1/block_size, m=1, Wk=I``
    recovers the centroid scorer; ``m>1`` learns multiple within-block landmarks
    with learned (not uniform) token weights.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        self.m = max(1, cfg.num_landmarks)
        self.block_size = cfg.block_size
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        # token-mixing: init to uniform mean (warm start = centroid for every landmark)
        Wt = torch.full(lead + (H, cfg.block_size, self.m), 1.0 / cfg.block_size)
        Wt = Wt + 0.01 * torch.randn_like(Wt)
        self.Wt = nn.Parameter(Wt)
        self.Wk = nn.Parameter(self._proj_init(lead + (H,), d, r))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._proj_init(lead + (H,), d, r))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        T, d = latent.shape
        nb = (T + block_size - 1) // block_size
        pad = nb * block_size - T
        Lb = F.pad(latent, (0, 0, 0, pad)).view(nb, block_size, d)      # [nb, bs, d]
        Wt = self._slice(self.Wt, layer_id); Wk = self._slice(self.Wk, layer_id)
        Wq = self._slice(self.Wq, layer_id)
        tok = torch.einsum("nsd,hsm->hnmd", Lb.to(Wt.dtype), Wt)        # [H, nb, m, d]
        g = torch.einsum("hnmd,hdr->hnmr", tok, Wk)                     # [H, nb, m, r]
        u = torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)             # [H, r]
        s = scaling * torch.einsum("hr,hnmr->hnm", u, g)              # [H, nb, m]
        return s.amax(dim=-1)                                          # [H, nb]


@register_arch("mlp")
class MLPScorer(BlockScorer):
    r"""Nonlinear per-head heads: score = MLPq(q)·MLPk(centroid). Mean pool."""

    def __init__(self, cfg):
        super().__init__(cfg)
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        hid = cfg.hidden_dim or max(2 * r, d // 2)
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        # per-(layer,head) two-layer projections (d→hid→r) for q and k
        self.Wk1 = nn.Parameter(self._proj_init(lead + (H,), d, hid))
        self.Wk2 = nn.Parameter(0.02 * torch.randn(*(lead + (H, hid, r))))
        self.Wq1 = nn.Parameter(self._proj_init(lead + (H,), d, hid))
        self.Wq2 = nn.Parameter(0.02 * torch.randn(*(lead + (H, hid, r))))

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        cent = _block_centroids(latent, block_size)                        # [B, d]
        Wk1 = self._slice(self.Wk1, layer_id); Wk2 = self._slice(self.Wk2, layer_id)
        Wq1 = self._slice(self.Wq1, layer_id); Wq2 = self._slice(self.Wq2, layer_id)
        gh = F.gelu(torch.einsum("bd,hde->hbe", cent.to(Wk1.dtype), Wk1))  # [H, B, hid]
        g = torch.einsum("hbe,her->hbr", gh, Wk2)                          # [H, B, r]
        uh = F.gelu(torch.einsum("hd,hde->he", q.to(Wq1.dtype), Wq1))      # [H, hid]
        u = torch.einsum("he,her->hr", uh, Wq2)                            # [H, r]
        return scaling * torch.einsum("hr,hbr->hb", u, g)                  # [H, B]


class BlockCompressor(nn.Module):
    """Wraps the configured :class:`BlockScorer` (``cfg.arch``)."""

    def __init__(self, cfg: CompressorConfig):
        super().__init__()
        if cfg.arch not in ARCH_REGISTRY:
            raise ValueError(f"unknown arch {cfg.arch!r}; have {sorted(ARCH_REGISTRY)}")
        self.cfg = cfg
        self.scorer = ARCH_REGISTRY[cfg.arch](cfg)

    def block_logits(self, q, latent, block_size, layer_id, scaling):
        return self.scorer.block_logits(q, latent, block_size, layer_id, scaling)
