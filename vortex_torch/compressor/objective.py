"""On-the-fly supervision targets, distillation loss, and eval metrics.

Everything is computed from the live tensors (absorbed query + latent) the HF
model produced this step — nothing is read from or written to disk.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def block_centroids(latent: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block mean over tokens (last block may be partial).

    ``latent`` [T, d] (MLA shared latent) → [B, d];
    ``latent`` [T, G, hd] (MHA per-KV-head keys) → [B, G, hd].
    """
    T = latent.shape[0]
    rest = latent.shape[1:]
    nb = (T + block_size - 1) // block_size
    pad = nb * block_size - T
    Lp = F.pad(latent, (0, 0) * (latent.dim() - 1) + (0, pad))
    sums = Lp.view(nb, block_size, *rest).sum(dim=1)         # [nb, *rest]
    counts = torch.full((nb,), block_size, device=latent.device, dtype=latent.dtype)
    if pad:
        counts[-1] = block_size - pad
    return sums / counts.view(-1, *([1] * (latent.dim() - 1)))


def centroid_block_logits(q: torch.Tensor, centroids: torch.Tensor,
                          scaling: float) -> torch.Tensor:
    """The (untrained) centroid baseline scorer → [H, B].
    ``centroids`` [B, d] (MLA) or [B, G, hd] (MHA; head h reads group h // (H/G)).
    Used at eval to report the learned compressor's gain over plain centroids."""
    if centroids.dim() == 3:                                 # MHA per-KV-head
        B, G, d = centroids.shape
        H = q.shape[0]
        qr = q.float().view(G, H // G, d)
        z = torch.einsum("gud,bgd->gub", qr, centroids.float())
        return scaling * z.reshape(H, B)
    return scaling * (q.float() @ centroids.float().transpose(0, 1))


def quest_block_logits(q: torch.Tensor, latent: torch.Tensor, block_size: int,
                       scaling: float) -> torch.Tensor:
    """The (untrained) Quest baseline for MHA: per block, elementwise
    ``Σ_d max(q_d·maxK_d, q_d·minK_d)`` over the block's per-KV-head min/max
    envelopes. ``latent`` [T, G, hd] → [H, B]. Partial-block padding is masked
    out of the min/max."""
    T, G, d = latent.shape
    H = q.shape[0]
    gs = H // G
    nb = (T + block_size - 1) // block_size
    pad = nb * block_size - T
    Lf = F.pad(latent, (0, 0, 0, 0, 0, pad)).float()         # [nb*bs, G, d]
    real = (torch.arange(nb * block_size, device=latent.device) < T).view(-1, 1, 1)
    big = torch.finfo(torch.float32).max
    kmax = torch.where(real, Lf, torch.full_like(Lf, -big)).view(nb, block_size, G, d).amax(1)
    kmin = torch.where(real, Lf, torch.full_like(Lf, big)).view(nb, block_size, G, d).amin(1)
    qr = q.float().view(G, gs, 1, d)                         # [G, gs, 1, d]
    pmax = qr * kmax.permute(1, 0, 2).unsqueeze(1)           # [G, gs, nb, d]
    pmin = qr * kmin.permute(1, 0, 2).unsqueeze(1)
    s = torch.maximum(pmax, pmin).sum(dim=-1)                # [G, gs, nb]
    return scaling * s.reshape(H, nb)


def true_attention(q: torch.Tensor, latent: torch.Tensor, scaling: float) -> torch.Tensor:
    """Dense per-head attention A [H, T] = softmax_t(scaling · q_h · k_t).

    ``latent`` [T, d] (MLA: ⟨q_abs_h, latent_t⟩ equals the per-head logit) or
    [T, G, hd] (MHA: head h reads its GQA group g(h) = h // (H/G)).
    """
    if latent.dim() == 3:                                    # MHA per-KV-head
        T, G, d = latent.shape
        H = q.shape[0]
        qr = q.float().view(G, H // G, d)
        z = scaling * torch.einsum("gud,tgd->gut", qr, latent.float())
        return z.reshape(H, T).softmax(dim=-1)
    z = scaling * (q.float() @ latent.float().transpose(0, 1))   # [H, T]
    return z.softmax(dim=-1)


def block_mass_targets(A: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-head per-block attention mass [H, B] from token attention A [H, T].
    Rows sum to 1 (the distillation target distribution over blocks)."""
    H, T = A.shape
    nb = (T + block_size - 1) // block_size
    pad = nb * block_size - T
    Ap = F.pad(A, (0, pad))                                  # [H, nb*bs]
    return Ap.view(H, nb, block_size).sum(dim=-1)           # [H, nb]


def distill_loss(block_logits: torch.Tensor, targets: torch.Tensor,
                 restrict_topk: int = 0) -> torch.Tensor:
    """Soft cross-entropy (= KL up to the target entropy) of the compressor's
    per-head block distribution against the true block-mass distribution.

    ``restrict_topk=K>0`` reproduces DeepSeek-V3.2's sparse-stage indexer loss
    (DSA, Eq. 4): instead of matching the FULL block distribution, restrict the
    softmax+KL to the **selected set** — here the per-head union of the
    compressor's own top-K blocks and the true top-K-mass blocks. This focuses
    capacity on ranking the blocks that actually get selected (false positives
    via the compressor's top-K, missed positives via the truth's top-K),
    directly targeting selection quality / p-coverage rather than the
    distribution tail. Use AFTER a full-KL warm-up (Eq. 3), as in the paper."""
    if restrict_topk and restrict_topk < block_logits.shape[-1]:
        H, B = block_logits.shape
        k = restrict_topk
        sel = torch.zeros(H, B, dtype=torch.bool, device=block_logits.device)
        sel.scatter_(1, block_logits.detach().topk(k, dim=-1).indices, True)   # false positives
        sel.scatter_(1, targets.topk(k, dim=-1).indices, True)                 # missed positives
        neg = torch.finfo(block_logits.dtype).min
        logits_s = block_logits.masked_fill(~sel, neg)
        logp = logits_s.log_softmax(dim=-1)
        tgt = (targets * sel).clamp_min(0)
        tgt = tgt / tgt.sum(dim=-1, keepdim=True).clamp_min(1e-9)              # renormalize on S
        return -(tgt * logp.masked_fill(~sel, 0.0)).sum(dim=-1).mean()
    logp = block_logits.log_softmax(dim=-1)                  # [H, B]
    return -(targets * logp).sum(dim=-1).mean()


def coverage_loss(block_logits: torch.Tensor, targets: torch.Tensor,
                  budget_blocks: int, group_size: int = 0,
                  temp: float = 0.05) -> torch.Tensor:
    """Differentiable surrogate for the DEPLOYMENT metric (pooled p-coverage):
    the true attention mass captured by the compressor's top-k blocks.

    Relaxed top-k: per (group) score row, a soft selection gate
    ``σ((logit − τ_k)/temp)`` where ``τ_k`` is the k-th largest score (a
    detached threshold), so gates are ~1 for selected blocks, ~0 otherwise, but
    differentiable through the logits. Loss = ``1 − Σ_b gate_b · mass_b`` (mass
    summed over the group, matching the deployed per-KV-head selection). Unlike
    KL, this directly optimizes captured mass at the operating budget rather
    than the full distribution."""
    H, B = block_logits.shape
    k = min(budget_blocks, B)
    if group_size and group_size > 1:
        G = H // group_size
        logits = block_logits.view(G, group_size, B).sum(1)      # group-pooled score
        mass = targets.view(G, group_size, B).sum(1)             # group attention mass
        mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
    else:
        logits, mass = block_logits, targets
    thresh = logits.detach().topk(k, dim=-1).values[:, -1:]      # k-th largest, detached
    gate = torch.sigmoid((logits - thresh) / temp)               # ~top-k soft mask
    captured = (gate * mass).sum(-1)                             # mass in soft-selection
    return (1.0 - captured).mean()


@torch.no_grad()
def coverage_recall(
    block_logits: torch.Tensor,     # [H, B]
    A: torch.Tensor,                # [H, T]
    block_size: int,
    budget_blocks: int,
    recall_N: Sequence[int],
    pooled: bool = False,
    group_size: int = 0,
) -> dict:
    """Selection-quality of the compressor's ranking, mirroring the
    cuda_mla_profile metrics (block→token expansion).

    * **p-coverage** — attention mass on tokens whose block was selected.
    * **recall@N**   — fraction of exact top-N tokens inside selected blocks.

    ``pooled=False`` → per-head selection (each head keeps its own top blocks),
    ``pooled=True``  → deployed-selection pooling: with ``group_size=0`` one
    block set shared by ALL heads (MLA decode: sum over heads); with
    ``group_size=g`` one block set per GQA group of ``g`` query heads (MHA
    decode: each KV head selects for its group).
    """
    H, T = A.shape
    nb = block_logits.shape[-1]
    k = min(budget_blocks, nb)
    tok_block = torch.arange(T, device=A.device) // block_size      # [T] block id per token

    if pooled and group_size:
        G = H // group_size
        gl = block_logits.view(G, group_size, nb).sum(dim=1)        # [G, B]
        sel_blocks = gl.topk(k, dim=-1).indices                     # [G, k]
        block_sel = torch.zeros(G, nb, dtype=torch.bool, device=A.device)
        block_sel.scatter_(1, sel_blocks, True)
        sel_tok = block_sel.repeat_interleave(group_size, 0)[:, tok_block]  # [H, T]
    elif pooled:
        sel_blocks = block_logits.sum(dim=0).topk(k).indices        # [k] shared
        sel_mask = torch.isin(tok_block, sel_blocks)                # [T]
        sel_tok = sel_mask.unsqueeze(0).expand(H, T)                # [H, T]
    else:
        sel_blocks = block_logits.topk(k, dim=-1).indices           # [H, k]
        sel_tok = torch.zeros(H, T, dtype=torch.bool, device=A.device)
        # mark tokens whose block is selected, per head
        block_sel = torch.zeros(H, nb, dtype=torch.bool, device=A.device)
        block_sel.scatter_(1, sel_blocks, True)
        sel_tok = block_sel[:, tok_block]                           # [H, T]

    pcov = (A * sel_tok).sum(dim=-1)                                # [H]
    out = {"p_coverage": float(pcov.mean())}
    for N in recall_N:
        n = min(N, T)
        topi = A.topk(n, dim=-1).indices                            # [H, n]
        hit = torch.gather(sel_tok, 1, topi).float().mean(dim=-1)   # [H]
        out[f"recall@{N}"] = float(hit.mean())
    return out
