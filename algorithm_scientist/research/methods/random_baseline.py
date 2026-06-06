"""Deterministic pseudo-random scores — the lower-bound baseline."""
import torch
NAME = "random"
def block_scores(ctx):
    i = torch.arange(ctx.num_blocks, dtype=torch.float32)
    return torch.frac(torch.sin(i * 12.9898 + 1.0) * 43758.5453)
