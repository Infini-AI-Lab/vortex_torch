"""QUEST min-max envelope, scored PER kv-head (per-head selection ceiling).
Each kv-head ranks blocks by its own group-pooled query against its own min/max."""
import torch
NAME = "quest_hw"
def block_scores_headwise(ctx):
    G, Hkv, D = ctx.G, ctx.Hkv, ctx.D
    q_kv = ctx.q.reshape(Hkv, G, D).mean(1)              # [Hkv, D] group-pooled query
    lo = q_kv[:, None, :] * ctx.kmin_h                   # [Hkv, nb, D]
    hi = q_kv[:, None, :] * ctx.kmax_h
    return torch.maximum(lo, hi).sum(-1)                 # [Hkv, nb]
