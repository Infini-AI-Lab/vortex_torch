"""Trainable per-head block compressor (low-rank bilinear block scorer)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CompressorConfig


class BlockCompressor(nn.Module):
    r"""Per-(layer,head) learned block scorer.

    For head ``h`` it learns key/query projections ``Wk[h], Wq[h] ∈ R^{d×r}`` and
    scores a block ``b`` (with centroid ``c_b = mean_{t∈b} latent_t``) by

        s[h, b] = scaling · (Wqᵀ q_h) · (Wkᵀ c_b)
                = scaling · q_hᵀ (Wq Wkᵀ) c_b.

    This is a rank-``r`` bilinear generalization of the centroid scorer
    (``Wk = Wq = I, r = d`` ⇒ ``scaling · q_h · c_b``). With ``pool="mean"`` the
    descriptor stored per block is ``Wkᵀ c_b`` — an ``r``-dim vector, i.e. a
    *compressed* centroid — so it is cheaper than the raw ``d``-dim centroid and
    maps directly onto a future vortex op (project latent → mean → gemm).

    Params are indexed by ``layer_id`` when ``cfg.per_layer`` (a contiguous index
    into ``cfg.num_layers``), else shared across layers.
    """

    def __init__(self, cfg: CompressorConfig):
        super().__init__()
        self.cfg = cfg
        H, d, r = cfg.num_q_heads, cfg.latent_dim, cfg.proj_dim
        lead = (cfg.num_layers,) if cfg.per_layer else ()
        self.Wk = nn.Parameter(self._init(lead + (H, d, r)))
        self.Wq = self.Wk if cfg.tie_qk else nn.Parameter(self._init(lead + (H, d, r)))

    def _init(self, shape) -> torch.Tensor:
        *lead, d, r = shape
        if self.cfg.init == "identity":
            # Truncated identity: first r channels → exact centroid on those dims
            # (warm start from the centroid baseline), tiny noise to break symmetry.
            base = torch.zeros(d, r)
            k = min(d, r)
            base[:k, :k] = torch.eye(k)
            w = base.expand(*lead, d, r).clone()
            w += 0.01 * torch.randn_like(w)
            return w
        # orthogonal columns per (layer,head)
        w = torch.randn(*shape)
        return torch.nn.init.orthogonal_(w.reshape(-1, w.shape[-1])).reshape(shape)

    def _slice(self, W: torch.Tensor, layer_id: int) -> torch.Tensor:
        return W[layer_id] if self.cfg.per_layer else W            # [H, d, r]

    def descriptors(self, centroids: torch.Tensor, layer_id: int) -> torch.Tensor:
        """``centroids`` [B, d] → per-head descriptors [H, B, r] = Wkᵀ c_b."""
        Wk = self._slice(self.Wk, layer_id)                        # [H, d, r]
        return torch.einsum("bd,hdr->hbr", centroids.to(Wk.dtype), Wk)

    def query_proj(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """``q`` [H, d] → [H, r] = Wqᵀ q_h."""
        Wq = self._slice(self.Wq, layer_id)                        # [H, d, r]
        return torch.einsum("hd,hdr->hr", q.to(Wq.dtype), Wq)

    def block_logits(
        self,
        q: torch.Tensor,            # [H, d]  absorbed query (one position)
        centroids: torch.Tensor,    # [B, d]  per-block mean latent
        layer_id: int,
        scaling: float,
    ) -> torch.Tensor:
        """Per-head block scores [H, B]."""
        qp = self.query_proj(q, layer_id)                          # [H, r]
        gp = self.descriptors(centroids, layer_id)                 # [H, B, r]
        return scaling * torch.einsum("hr,hbr->hb", qp, gp)        # [H, B]
