"""On-the-fly supervision targets, distillation loss, and eval metrics.

Everything is computed from the live tensors (absorbed query + latent) the HF
model produced this step — nothing is read from or written to disk.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def block_centroids(latent: torch.Tensor, block_size: int) -> torch.Tensor:
    """``latent`` [T, d] → per-block mean [B, d] (last block may be partial)."""
    T, d = latent.shape
    nb = (T + block_size - 1) // block_size
    pad = nb * block_size - T
    Lp = F.pad(latent, (0, 0, 0, pad))                       # [nb*bs, d]
    sums = Lp.view(nb, block_size, d).sum(dim=1)             # [nb, d]
    counts = torch.full((nb,), block_size, device=latent.device, dtype=latent.dtype)
    if pad:
        counts[-1] = block_size - pad
    return sums / counts.unsqueeze(1)


def centroid_block_logits(q: torch.Tensor, centroids: torch.Tensor,
                          scaling: float) -> torch.Tensor:
    """The (untrained) centroid baseline scorer: scaling · q_h · centroid_b → [H, B].
    Used at eval to report the learned compressor's gain over plain centroids."""
    return scaling * (q.float() @ centroids.float().transpose(0, 1))


def true_attention(q: torch.Tensor, latent: torch.Tensor, scaling: float) -> torch.Tensor:
    """Dense per-head attention A [H, T] = softmax_t(scaling · q_h · latent_t).

    Exact for absorbed MLA: ⟨q_abs_h, latent_t⟩ equals the per-head logit.
    """
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


def distill_loss(block_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy (= KL up to the target entropy) of the compressor's
    per-head block distribution against the true block-mass distribution."""
    logp = block_logits.log_softmax(dim=-1)                  # [H, B]
    return -(targets * logp).sum(dim=-1).mean()


@torch.no_grad()
def coverage_recall(
    block_logits: torch.Tensor,     # [H, B]
    A: torch.Tensor,                # [H, T]
    block_size: int,
    budget_blocks: int,
    recall_N: Sequence[int],
    pooled: bool = False,
) -> dict:
    """Selection-quality of the compressor's ranking, mirroring the
    cuda_mla_profile metrics (block→token expansion).

    * **p-coverage** — attention mass on tokens whose block was selected.
    * **recall@N**   — fraction of exact top-N tokens inside selected blocks.

    ``pooled=False`` → per-head selection (each head keeps its own top blocks),
    ``pooled=True``  → request-level selection (sum over heads, one block set for
    all heads — what the deployed MLA decode actually does).
    """
    H, T = A.shape
    nb = block_logits.shape[-1]
    k = min(budget_blocks, nb)
    tok_block = torch.arange(T, device=A.device) // block_size      # [T] block id per token

    if pooled:
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
