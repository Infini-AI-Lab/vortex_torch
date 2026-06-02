"""QUEST min-max channel envelope: an upper bound on max_k q·k per block."""
import torch
NAME = "quest"
def block_scores(ctx):
    lo = ctx.q_bar * ctx.kmin
    hi = ctx.q_bar * ctx.kmax
    return torch.maximum(lo, hi).sum(-1)  # [nb]
