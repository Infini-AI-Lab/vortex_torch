r"""Learned per-layer bilinear block-sparse routing for MLA decode.

This module deploys the trained per-layer "block compressor"
(:mod:`vortex_torch.compressor`) as a runnable vortex flow. It reads almost
exactly like :class:`RopeAwareBlockSparseMLA` — one centroid per block
(``CMean``), score by a single dot ``⟨V, centroid⟩``, top-k — except the
query side is replaced by the learned :class:`LearnedQuery` transform that
bakes the trained per-layer weights as compiled-in constants and selects
the active layer's slice at runtime.

**Why a learned query suffices.** The trained bilinear scorer is

.. math::

    \operatorname{score}(b) = \sum_h (W_q[\ell,h]^\top q_h)^\top (W_k[\ell,h]^\top c_b)
        = \Big\langle \sum_h W_k[\ell,h]\,(W_q[\ell,h]^\top q_h),\; c_b \Big\rangle
        = \langle V,\, c_b\rangle ,

so the cache side (centroid via ``CMean``) is identical to the centroid
baseline and only the query becomes ``V``.

**Weight loading.** ``__init__`` loads a checkpoint from the
``VORTEX_COMPRESSOR_CKPT`` env var — a ``torch.save`` dict
``{"state_dict", "config", "layer_ids"}`` produced by
``vortex_torch/compressor/train.py`` (with ``per_layer=True`` the scorer's
``Wq``/``Wk`` have shape ``[L, H, d, r]`` and ``layer_ids`` lists the
trained global layer indices). A ``global layer_id -> row`` lookup is built
from ``layer_ids``. With the env var unset (or a layer absent from the
checkpoint) the op falls back to identity (``V = Σ_h q_h``), so the flow
still compiles and reproduces the plain head-summed centroid scorer.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from .flow_mla import vFlowMLA
from .registry import register
from ..indexer import topK, GeMM, LearnedQuery
from ..cache import Mean as CMean
from ..abs import ContextBase


def _load_compressor_weights() -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]
]:
    """Resolve ``(Wq, Wk, layer_lookup, scaling)`` for :class:`LearnedQuery`.

    Reads ``VORTEX_COMPRESSOR_CKPT`` if set and returns the per-layer weights
    ``[L, H, d, r]`` plus a 1-D long ``layer_lookup`` of length
    ``max(global_layer_id)+1`` mapping ``global layer_id -> row`` (``-1`` =
    identity fallback). Returns ``None`` when the env var is unset, so the
    op uses its lazy-identity default (``V = Σ_h q_h``, == centroid scorer).
    """
    ckpt_path = os.environ.get("VORTEX_COMPRESSOR_CKPT", "").strip()
    if not ckpt_path:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    layer_ids = list(ckpt.get("layer_ids", []))

    if not bool(cfg.get("per_layer", True)):
        raise ValueError(
            "learned_block_sparse_mla requires a per_layer=True compressor "
            f"checkpoint; got per_layer={cfg.get('per_layer')}."
        )

    # BilinearScorer params are stored under scorer.Wq / scorer.Wk.
    def _get(name: str) -> torch.Tensor:
        for key in (f"scorer.{name}", name):
            if key in sd:
                return sd[key].float()
        raise KeyError(
            f"compressor checkpoint missing '{name}' (looked for "
            f"'scorer.{name}'); keys: {sorted(sd.keys())[:8]}..."
        )

    Wq = _get("Wq")  # tie_qk -> Wq is Wk; still present after state_dict()
    Wk = _get("Wk") if any(k.endswith("Wk") for k in sd) else Wq
    assert Wq.dim() == 4, f"expected per-layer Wq [L,H,d,r], got {tuple(Wq.shape)}"
    L, H, d, r = Wq.shape

    # Build global layer_id -> row lookup. layer_ids[i] is the global layer
    # index trained into row i; unknown layers map to -1 (identity fallback).
    if layer_ids:
        max_lid = max(layer_ids)
        layer_lookup = torch.full((max_lid + 1,), -1, dtype=torch.long)
        for row, lid in enumerate(layer_ids):
            if row < L:
                layer_lookup[int(lid)] = row
    else:
        # No layer_ids recorded: assume row == global layer id.
        layer_lookup = torch.arange(L, dtype=torch.long)

    return Wq.contiguous(), Wk.contiguous(), layer_lookup, 1.0


@register("learned_block_sparse_mla")
class LearnedBlockSparseMLA(vFlowMLA):
    r"""
    Per-layer **learned** bilinear block-sparse routing on the fused MLA latent.

    Twin of :class:`RopeAwareBlockSparseMLA`: one centroid per block, score by
    a single dot, top-k — but the head-mean query is replaced by the trained
    per-layer transform :math:`V = \sum_h W_k[\ell,h](W_q[\ell,h]^\top q_h)`,
    giving the request-level bilinear score :math:`\langle V, c_b\rangle`.
    """

    def __init__(self) -> None:
        super().__init__()
        # Trained per-layer weights from VORTEX_COMPRESSOR_CKPT, or None for the
        # lazy-identity default (V = Σ_h q_h == head-summed centroid scorer).
        loaded = _load_compressor_weights()
        if loaded is None:
            Wq = Wk = lookup = None
            scaling = 1.0
        else:
            Wq, Wk, lookup, scaling = loaded

        # Indexer-side ops (run every decode step). One op instance per call site.
        self.learned_query = LearnedQuery(Wq, Wk, lookup, scaling=scaling)
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ → per-block score
        self.output_func = topK()      # terminal: write selected block ids to o

        # Cache-side op (run once per finished block): block-mean latent centroid
        # — identical to RopeAwareBlockSparseMLA.
        self.reduction = CMean(dim=1)

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim] ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        V = self.learned_query(q, ctx=ctx)                          # [B, 1, latent_dim]
        score = self.gemm(V, cache["centroids"], ctx=ctx)          # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        # "latent" is auto-provided — declare only the aux centroid (full width).
        return {
            "centroids": (1, kv_lora_rank + qk_rope_head_dim),
        }
