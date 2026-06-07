r"""Learned per-layer bilinear block-sparse routing for MLA decode.

Deploys the trained per-layer "block compressor" (:mod:`vortex_torch.compressor`)
as a runnable vortex flow, built **entirely from existing ops + a
``Vortex.Parameter``** (no bespoke op). It reads almost like
:class:`RopeAwareBlockSparseMLA` — one centroid per block (``CMean``), score by a
single dot ``⟨V, centroid⟩``, top-k — except the head-mean query is replaced by a
learned per-request transform ``V = W·q_flat``.

**The folding.** The trained bilinear scorer is

.. math::

    \operatorname{score}(b) = \sum_h (W_q[\ell,h]^\top q_h)^\top (W_k[\ell,h]^\top c_b)
        = \Big\langle \underbrace{\sum_h W_k[\ell,h]\,W_q[\ell,h]^\top q_h}_{V},\; c_b \Big\rangle .

The per-head sum folds into a single matrix ``W[ℓ] ∈ R^{d×(H·d)}`` with
``W[ℓ][:, h·d:(h+1)·d] = W_k[ℓ,h] W_q[ℓ,h]^\top``, so ``V = W[ℓ] · q_flat`` where
``q_flat`` is the head-flattened query. The cache side (centroid via ``CMean``)
is identical to the centroid baseline.

**Mechanism.** ``W`` is a :class:`~vortex_torch.indexer.Parameter`
(``FORMAT.PARAMETER`` — a batch-shared constant) declared in ``__init__``. The
``GeMM`` that multiplies ``q_flat`` by it sees the PARAMETER operand and runs as
a standalone ``Schedule.S`` ``torch.matmul`` over the per-layer slice (gathered
by the explicit ``cur_layer`` arg) — so the large weight never enters the fused
kernel. ``Reshape`` flattens/​unflattens the inner dims.

**Weights.** ``__init__`` loads ``VORTEX_COMPRESSOR_CKPT`` (a per-layer
``{state_dict, config, layer_ids}`` from ``vortex_torch/compressor/train.py``),
folds ``Wk·Wqᵀ`` into ``W[Lfull, d, H·d]`` (untrained layers → identity, so they
reduce to the centroid scorer). Env unset ⇒ all-identity ⇒ flow == plain
head-summed centroid scorer.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from .flow_mla import vFlowMLA
from .registry import register
from ..indexer import topK, GeMM, Reshape, Parameter, FactorScore
from ..cache import Mean as CMean, LearnedDescriptor
from ..abs import ContextBase


def _load_folded_W() -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Resolve the folded per-layer weight for the learned flow.

    Reads ``VORTEX_COMPRESSOR_CKPT`` (per-layer bilinear scorer) and returns
    ``(W, layer_lookup)`` where ``W`` is ``[Lfull, d, H·d]`` with
    ``W[ℓ][:, h·d:(h+1)·d] = Wk[ℓ,h] Wq[ℓ,h]^\\top`` (untrained rows = the
    ``H``-fold identity, so they reduce to ``V = Σ_h q_h`` = centroid scorer) and
    ``layer_lookup = arange(Lfull)`` (row == global layer id). Returns ``None``
    when the env var is unset → the flow builds an identity Parameter lazily.
    """
    ckpt_path = os.environ.get("VORTEX_COMPRESSOR_CKPT", "").strip()
    if not ckpt_path:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    layer_ids = [int(x) for x in ckpt.get("layer_ids", [])]
    if not bool(cfg.get("per_layer", True)):
        raise ValueError(
            "learned_block_sparse_mla expects a per_layer=True compressor "
            f"checkpoint; got per_layer={cfg.get('per_layer')}."
        )

    def _get(name: str) -> torch.Tensor:
        for key in (f"scorer.{name}", name):
            if key in sd:
                return sd[key].float()
        raise KeyError(f"checkpoint missing 'scorer.{name}'; keys: "
                       f"{sorted(sd.keys())[:8]}...")

    Wq = _get("Wq")                                    # [L, H, d, r]
    Wk = _get("Wk") if any(k.endswith("Wk") for k in sd) else Wq
    assert Wq.dim() == 4, f"expected per-layer Wq [L,H,d,r], got {tuple(Wq.shape)}"
    L, H, d, r = Wq.shape

    # Fold per head: M[l,h] = Wk[l,h] @ Wq[l,h]^T  ([d,d]); W[l][:, h*d:(h+1)*d] = M[l,h].
    M = torch.einsum("lhdr,lher->lhde", Wk, Wq)        # [L, H, d, d]
    W_trained = M.permute(0, 2, 1, 3).reshape(L, d, H * d).contiguous()  # [L, d, H*d]

    # Build Lfull rows (indexed directly by global layer id); untrained = identity.
    Lfull = (max(layer_ids) + 1) if layer_ids else L
    eye_fold = torch.eye(d).repeat(1, H)               # [d, H*d]  (H identity blocks)
    W = eye_fold.unsqueeze(0).repeat(Lfull, 1, 1).contiguous()           # [Lfull, d, H*d]
    if layer_ids:
        for row, lid in enumerate(layer_ids):
            if row < L:
                W[lid] = W_trained[row]
    else:
        W[:L] = W_trained
    layer_lookup = torch.arange(Lfull, dtype=torch.long)
    return W, layer_lookup


@register("learned_block_sparse_mla")
class LearnedBlockSparseMLA(vFlowMLA):
    r"""Per-layer **learned** bilinear block-sparse routing on the fused MLA latent.

    Twin of :class:`RopeAwareBlockSparseMLA`, but the head-mean query is replaced
    by ``V = W[ℓ]·q_flat`` (a ``Vortex.Parameter`` consumed by ``GeMM`` on its
    ``Schedule.S`` torch.matmul path), giving the request-level bilinear score
    ``⟨V, c_b⟩``.
    """

    # NOTE: query-head count H and latent dim d are hardcoded to GLM-4.7-Flash
    # geometry so every op/Parameter can be built in __init__ (vFlowMLA.initialize
    # currently passes d via kv_lora_rank+qk_rope_head_dim but NOT H). TODO: thread
    # num_q_heads through initialize() and derive these instead of hardcoding.
    _H = 20
    _D = 576

    def __init__(self) -> None:
        super().__init__()
        loaded = _load_folded_W()
        if loaded is not None:
            W, lookup = loaded
            d = int(W.shape[1]); H = int(W.shape[2]) // d
        else:
            # identity fallback (no checkpoint): V = Σ_h q_h == centroid scorer.
            H, d = self._H, self._D
            W = torch.eye(d).repeat(1, H)                   # [d, H*d] (-> Parameter [1,d,H*d])
            lookup = None

        # All ops + the Parameter are defined here (no lazy creation in forward).
        # GeMM stays self-contained (standard K-contract); the flatten/transpose
        # are explicit Reshape ops around it.
        self.W = Parameter(W, lookup)                       # batch-shared learned constant
        self.rq = Reshape(-1, 1, H * d)                     # q [B,H,d] -> [B,1,H*d] (K=H*d)
        self.gV = GeMM()                                    # q_flat × W -> [B,d,1] (Schedule.S)
        self.rv = Reshape(-1, 1, d)                         # V [B,d,1] -> [B,1,d]
        self.gemm = GeMM()                                  # V × centroids -> per-block score
        self.output_func = topK()                          # terminal: block ids -> o
        self.reduction = CMean(dim=1)                      # cache: block-mean latent centroid

    def forward_indexer(
        self,
        q: torch.Tensor,               # [B, H, latent_dim] ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        # Composable, all existing ops + the Vortex.Parameter; GeMM is a plain
        # K-contract, Reshape handles the flatten (K=H*d) and the output transpose.
        q_flat = self.rq(q, ctx=ctx)                        # [B, 1, H*d]
        V = self.gV(q_flat, self.W, ctx=ctx)                # [B, d, 1]  (param -> Schedule.S)
        V = self.rv(V, ctx=ctx)                             # [B, 1, d]
        score = self.gemm(V, cache["centroids"], ctx=ctx)  # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        return {"centroids": (1, kv_lora_rank + qk_rope_head_dim)}


# --------------------------------------------------------------------------- #
# Factorized learned block compressor (cache-side LearnedDescriptor +
# indexer-side FactorScore).
# --------------------------------------------------------------------------- #
def _load_factorized_weights():
    r"""Resolve the factorized per-layer weights for the learned flow.

    Reads ``VORTEX_COMPRESSOR_CKPT`` (a ``factorized`` per-layer compressor:
    ``scorer.Wt [L,H,bs,m]``, ``scorer.Wk [L,H,d,r]``, ``scorer.Wq [L,H,d,r]``,
    ``layer_ids``) and scatters the trained rows into ``[Lfull, ...]`` buffers
    indexed by global layer id (row == layer id). Returns
    ``(Wt, Wk, Wq, layer_lookup, H, m, r)`` or ``None`` when the env var is
    unset → the flow builds an identity fallback (== head-summed centroid scorer).
    """
    ckpt_path = os.environ.get("VORTEX_COMPRESSOR_CKPT", "").strip()
    if not ckpt_path:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    layer_ids = [int(x) for x in ckpt.get("layer_ids", [])]
    if cfg.get("arch") not in (None, "factorized"):
        raise ValueError(
            "learned_factorized_block_sparse_mla expects a factorized compressor "
            f"checkpoint; got arch={cfg.get('arch')}."
        )
    if not bool(cfg.get("per_layer", True)):
        raise ValueError("learned_factorized_block_sparse_mla expects per_layer=True.")

    def _get(name: str) -> torch.Tensor:
        for key in (f"scorer.{name}", name):
            if key in sd:
                return sd[key].float()
        raise KeyError(f"checkpoint missing 'scorer.{name}'")

    Wt = _get("Wt")                       # [L, H, bs, m]
    Wk = _get("Wk")                       # [L, H, d, r]
    Wq = _get("Wq")                       # [L, H, d, r]
    L, H, bs, m = Wt.shape
    _, _, d, r = Wk.shape
    Lfull = (max(layer_ids) + 1) if layer_ids else L

    # Identity defaults for untrained layers: Wt = 1/bs (uniform mean over the
    # first landmark, zeros elsewhere), Wk/Wq = truncated identity (the centroid
    # warm-start the trainer itself uses, init="identity").
    Wt_full = torch.zeros(Lfull, H, bs, m)
    Wt_full[..., 0] = 1.0 / bs
    eye = torch.zeros(d, r)
    eye[:min(d, r), :min(d, r)] = torch.eye(min(d, r))
    Wk_full = eye.view(1, 1, d, r).repeat(Lfull, H, 1, 1).contiguous()
    Wq_full = Wk_full.clone()

    rows = layer_ids if layer_ids else list(range(L))
    for row, lid in enumerate(rows):
        if row < L and 0 <= lid < Lfull:
            Wt_full[lid] = Wt[row]
            Wk_full[lid] = Wk[row]
            Wq_full[lid] = Wq[row]

    layer_lookup = torch.arange(Lfull, dtype=torch.long)
    return Wt_full, Wk_full, Wq_full, layer_lookup, H, m, r


@register("learned_factorized_block_sparse_mla")
class LearnedFactorizedBlockSparseMLA(vFlowMLA):
    r"""Per-layer **learned factorized** block-sparse routing on the MLA latent.

    Both the within-block token mixing (``Wt``) and the channel compression
    (``Wk``) are learned (cache side, :class:`LearnedDescriptor`), producing
    ``m`` landmark descriptors ``g[H*m, r]`` per block; the indexer
    (:class:`FactorScore`) scores each block by ``Σ_h max_m ⟨W_q[h]^T q_h,
    g[h,m]⟩``. Identity weights (no checkpoint) reduce to the head-summed
    centroid scorer.

    Requires ``vortex_attention_backend='trtllm'`` (block-table score layout).
    """

    # GLM-4.7-Flash geometry (hardcoded; matches the bilinear flow's TODO until
    # num_q_heads / block_size are threaded through vFlowMLA.initialize). _BS is
    # the cuda_mla block_size (== page_size); the LearnedDescriptor profile
    # asserts the baked Wt block-size axis matches the runtime block_size.
    _H = 20
    _D = 576
    _BS = 32

    def __init__(self) -> None:
        super().__init__()
        loaded = _load_factorized_weights()
        if loaded is not None:
            Wt, Wk, Wq, lookup, H, m, r = loaded
        else:
            # Identity fallback (no checkpoint): m=1, r=d, Wt=1/bs, Wk=Wq=I
            # → reduces to the head-summed centroid scorer.
            H, d = self._H, self._D
            m, r = 1, d
            bs = self._BS
            Wt = torch.zeros(1, H, bs, m); Wt[..., 0] = 1.0 / bs
            eye = torch.eye(d).view(1, 1, d, r).repeat(1, H, 1, 1).contiguous()
            Wk = eye.clone(); Wq = eye.clone()
            lookup = None

        self._m = int(m)
        self._r = int(r)
        # Cache-side Parameters (Wt [L,H,bs,m], Wk [L,H,d,r]) for LearnedDescriptor.
        self.Wt = Parameter(Wt, lookup)
        self.Wk = Parameter(Wk, lookup)
        # Indexer-side Parameter (Wq) consumed by FactorScore.
        self.Wq = Parameter(Wq, lookup)

        self.descriptor = LearnedDescriptor(self.Wt, self.Wk, H, self._m, self._r)
        self.score = FactorScore(self.Wq, H, self._m, self._r)
        self.output_func = topK()
        self._H_eff = H
        self._lookup = lookup

    def initialize(self, block_size, kv_lora_rank, qk_rope_head_dim, *args, **kwargs):
        r"""Standard MLA initialize, plus rebuild the cache-side token-mixing
        ``Wt`` to the configured ``block_size`` when it differs from the trained
        block axis. The trained ``Wt`` is learned for one block size (32 for the
        shipped checkpoint); for any other block size (e.g. the compile-check
        sweep's 16) it falls back to uniform mean-pool ``1/block_size`` in the
        first landmark so the flow still compiles/runs (trained ``Wk``/``Wq``
        are kept). At the deployment block size matching the checkpoint, the
        trained ``Wt`` is used unchanged."""
        trained_bs = int(self.Wt.value.shape[-2])
        if int(block_size) != trained_bs:
            H = self._H_eff
            Wt = torch.zeros(self.Wt.value.shape[0], H, int(block_size), self._m)
            Wt[..., 0] = 1.0 / float(block_size)
            self.Wt = Parameter(Wt, self._lookup)
            self.descriptor = LearnedDescriptor(
                self.Wt, self.Wk, H, self._m, self._r
            )
        return super().initialize(block_size, kv_lora_rank, qk_rope_head_dim,
                                  *args, **kwargs)

    def forward_indexer(self, q, o, cache, ctx: ContextBase):
        score = self.score(q, cache["descriptors"], ctx=ctx)   # [S, 1, 1] RAGGED
        self.output_func(score, o, ctx=ctx)

    def forward_cache(self, cache, loc, ctx: ContextBase):
        self.descriptor(cache["latent"], cache["descriptors"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        return {"descriptors": (self._H * self._m, self._r)}
