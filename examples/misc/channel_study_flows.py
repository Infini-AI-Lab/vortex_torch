"""Channel-group importance study for block-sparse routing (Qwen3-4B, D=128).

Splits the head dimension into 8 groups of 16 channels and studies how routing
quality depends on *which* groups feed the page score, and how many groups can
be MASKED (dropped) before routing degrades. Mechanism: a per-channel query
mask (1.0 on active groups' channels, 0.0 elsewhere; sum of disjoint per-group
MaskSlices) multiplied into the query before scoring. Zeroing a group removes
its per-channel contribution from the routing score, uniformly for:
  block_sparse    : <q_masked, c>
  gqa_block_sparse: softmax_p(<q_masked_h, c>) then max_h
  gqa_quest_sparse: sum_d max(q_masked*M, q_masked*min) then max_h

Select via vortex_module_name + vortex_module_path=examples/misc/channel_study_flows.py.
Names: chan_<fam>_g<i> (only group i), chan_<fam>_<subset> (only those groups),
<fam> in {block, gqablock, quest}. Variants use explicit @register decorators
(so check_engine_config's preflight text-scan passes) wrapped to be idempotent
(re-exec safe).
"""
import torch
from typing import Dict, List

from vortex_torch.flow.flow import vFlow
from vortex_torch.flow.registry import register as _raw_register, _REGISTRY
from vortex_torch.indexer import (
    topK, GeMM, Softmax, Max, Sum, Multiply, Maximum, Add, Mean, MaskSlice,
)
from vortex_torch.cache import Mean as CMean, Max as CMax, Min as CMin
from vortex_torch.abs import ContextBase

GROUP_SIZE = 16
HEAD_DIM = 128
N_GROUPS = HEAD_DIM // GROUP_SIZE  # 8


def register(name):
    """Idempotent @register: keeps the literal decorator the preflight text-scan
    looks for, but skips re-registration if this file is exec'd more than once."""
    def deco(cls):
        if name not in _REGISTRY:
            _raw_register(name)(cls)
        return cls
    return deco


class _ChannelMaskMixin:
    """Applies a per-channel query mask from ``self.GROUPS`` (active 16-channel
    group indices): sum of disjoint per-group position masks == 1 on active
    channels, 0 elsewhere, multiplied into q."""
    GROUPS: List[int] = list(range(N_GROUPS))

    def _init_channel_ops(self):
        self._mask_ops = [
            MaskSlice(start=g * GROUP_SIZE, end=(g + 1) * GROUP_SIZE, dim=2,
                      alpha=1.0, beta=0.0)
            for g in self.GROUPS
        ]
        self._add_ops = [Add() for _ in range(max(0, len(self.GROUPS) - 1))]
        self._chan_mul = Multiply()

    def _mask_q(self, q, ctx):
        m = self._mask_ops[0](q, ctx=ctx)
        for op, add in zip(self._mask_ops[1:], self._add_ops):
            m = add(m, op(q, ctx=ctx), ctx=ctx)
        return self._chan_mul(q, m, ctx=ctx)


class _ChanBlock(_ChannelMaskMixin, vFlow):
    def __init__(self):
        super().__init__()
        self._init_channel_ops()
        self.mean = Mean(dim=1); self.gemm = GeMM()
        self.output_func = topK(); self.reduction = CMean(dim=1)
    def forward_indexer(self, q, o, cache, ctx):
        q_mean = self.mean(self._mask_q(q, ctx), ctx=ctx)
        self.output_func(self.gemm(q_mean, cache["centroids"], ctx=ctx), o, ctx=ctx)
    def forward_cache(self, cache, loc, ctx):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
    def create_cache(self, block_size, head_dim):
        return {"centroids": (1, head_dim)}


class _ChanGQABlock(_ChannelMaskMixin, vFlow):
    def __init__(self):
        super().__init__()
        self._init_channel_ops()
        self.gemm = GeMM(); self.softmax = Softmax(dim=0, scale=0.09)
        self.max_op = Max(dim=2); self.output_func = topK(); self.reduction = CMean(dim=1)
    def forward_indexer(self, q, o, cache, ctx):
        score = self.gemm(self._mask_q(q, ctx), cache["centroids"], ctx=ctx)
        self.output_func(self.max_op(self.softmax(score, ctx=ctx), ctx=ctx), o, ctx=ctx)
    def forward_cache(self, cache, loc, ctx):
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)
    def create_cache(self, block_size, head_dim):
        return {"centroids": (1, head_dim)}


class _ChanQuest(_ChannelMaskMixin, vFlow):
    def __init__(self):
        super().__init__()
        self._init_channel_ops()
        self.mul_max = Multiply(); self.mul_min = Multiply(); self.maximum_op = Maximum()
        self.sum = Sum(dim=2); self.max_op = Max(dim=1); self.output_func = topK()
        self.reduction_max = CMax(dim=1); self.reduction_min = CMin(dim=1)
    def forward_indexer(self, q, o, cache, ctx):
        qm = self._mask_q(q, ctx)
        s = self.maximum_op(self.mul_max(qm, cache["max"], ctx=ctx),
                            self.mul_min(qm, cache["min"], ctx=ctx), ctx=ctx)
        self.output_func(self.max_op(self.sum(s, ctx=ctx), ctx=ctx), o, ctx=ctx)
    def forward_cache(self, cache, loc, ctx):
        self.reduction_max(cache["k"], cache["max"], loc=loc, ctx=ctx)
        self.reduction_min(cache["k"], cache["min"], loc=loc, ctx=ctx)
    def create_cache(self, block_size, head_dim):
        return {"max": (1, head_dim), "min": (1, head_dim)}


# --- explicit @register variants (text-scan friendly) -----------------------
@register("chan_block_g0")
class Chan_block_g0(_ChanBlock):
    GROUPS = [0]

@register("chan_block_g1")
class Chan_block_g1(_ChanBlock):
    GROUPS = [1]

@register("chan_block_g2")
class Chan_block_g2(_ChanBlock):
    GROUPS = [2]

@register("chan_block_g3")
class Chan_block_g3(_ChanBlock):
    GROUPS = [3]

@register("chan_block_g4")
class Chan_block_g4(_ChanBlock):
    GROUPS = [4]

@register("chan_block_g5")
class Chan_block_g5(_ChanBlock):
    GROUPS = [5]

@register("chan_block_g6")
class Chan_block_g6(_ChanBlock):
    GROUPS = [6]

@register("chan_block_g7")
class Chan_block_g7(_ChanBlock):
    GROUPS = [7]

@register("chan_block_lo4")
class Chan_block_lo4(_ChanBlock):
    GROUPS = [0, 1, 2, 3]

@register("chan_block_hi4")
class Chan_block_hi4(_ChanBlock):
    GROUPS = [4, 5, 6, 7]

@register("chan_block_even")
class Chan_block_even(_ChanBlock):
    GROUPS = [0, 2, 4, 6]

@register("chan_block_odd")
class Chan_block_odd(_ChanBlock):
    GROUPS = [1, 3, 5, 7]

@register("chan_gqablock_g0")
class Chan_gqablock_g0(_ChanGQABlock):
    GROUPS = [0]

@register("chan_gqablock_g1")
class Chan_gqablock_g1(_ChanGQABlock):
    GROUPS = [1]

@register("chan_gqablock_g2")
class Chan_gqablock_g2(_ChanGQABlock):
    GROUPS = [2]

@register("chan_gqablock_g3")
class Chan_gqablock_g3(_ChanGQABlock):
    GROUPS = [3]

@register("chan_gqablock_g4")
class Chan_gqablock_g4(_ChanGQABlock):
    GROUPS = [4]

@register("chan_gqablock_g5")
class Chan_gqablock_g5(_ChanGQABlock):
    GROUPS = [5]

@register("chan_gqablock_g6")
class Chan_gqablock_g6(_ChanGQABlock):
    GROUPS = [6]

@register("chan_gqablock_g7")
class Chan_gqablock_g7(_ChanGQABlock):
    GROUPS = [7]

@register("chan_gqablock_lo4")
class Chan_gqablock_lo4(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3]

@register("chan_gqablock_hi4")
class Chan_gqablock_hi4(_ChanGQABlock):
    GROUPS = [4, 5, 6, 7]

@register("chan_gqablock_even")
class Chan_gqablock_even(_ChanGQABlock):
    GROUPS = [0, 2, 4, 6]

@register("chan_gqablock_odd")
class Chan_gqablock_odd(_ChanGQABlock):
    GROUPS = [1, 3, 5, 7]

@register("chan_quest_g0")
class Chan_quest_g0(_ChanQuest):
    GROUPS = [0]

@register("chan_quest_g1")
class Chan_quest_g1(_ChanQuest):
    GROUPS = [1]

@register("chan_quest_g2")
class Chan_quest_g2(_ChanQuest):
    GROUPS = [2]

@register("chan_quest_g3")
class Chan_quest_g3(_ChanQuest):
    GROUPS = [3]

@register("chan_quest_g4")
class Chan_quest_g4(_ChanQuest):
    GROUPS = [4]

@register("chan_quest_g5")
class Chan_quest_g5(_ChanQuest):
    GROUPS = [5]

@register("chan_quest_g6")
class Chan_quest_g6(_ChanQuest):
    GROUPS = [6]

@register("chan_quest_g7")
class Chan_quest_g7(_ChanQuest):
    GROUPS = [7]

@register("chan_quest_lo4")
class Chan_quest_lo4(_ChanQuest):
    GROUPS = [0, 1, 2, 3]

@register("chan_quest_hi4")
class Chan_quest_hi4(_ChanQuest):
    GROUPS = [4, 5, 6, 7]

@register("chan_quest_even")
class Chan_quest_even(_ChanQuest):
    GROUPS = [0, 2, 4, 6]

@register("chan_quest_odd")
class Chan_quest_odd(_ChanQuest):
    GROUPS = [1, 3, 5, 7]



# --- leave-one-out (keep 7 groups; mask group g) -----------------------------
@register("chan_block_no0")
class Chan_block_no0(_ChanBlock):
    GROUPS = [1, 2, 3, 4, 5, 6, 7]

@register("chan_block_no1")
class Chan_block_no1(_ChanBlock):
    GROUPS = [0, 2, 3, 4, 5, 6, 7]

@register("chan_block_no2")
class Chan_block_no2(_ChanBlock):
    GROUPS = [0, 1, 3, 4, 5, 6, 7]

@register("chan_block_no3")
class Chan_block_no3(_ChanBlock):
    GROUPS = [0, 1, 2, 4, 5, 6, 7]

@register("chan_block_no4")
class Chan_block_no4(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 5, 6, 7]

@register("chan_block_no5")
class Chan_block_no5(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 4, 6, 7]

@register("chan_block_no6")
class Chan_block_no6(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 4, 5, 7]

@register("chan_block_no7")
class Chan_block_no7(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 4, 5, 6]

@register("chan_gqablock_no0")
class Chan_gqablock_no0(_ChanGQABlock):
    GROUPS = [1, 2, 3, 4, 5, 6, 7]

@register("chan_gqablock_no1")
class Chan_gqablock_no1(_ChanGQABlock):
    GROUPS = [0, 2, 3, 4, 5, 6, 7]

@register("chan_gqablock_no2")
class Chan_gqablock_no2(_ChanGQABlock):
    GROUPS = [0, 1, 3, 4, 5, 6, 7]

@register("chan_gqablock_no3")
class Chan_gqablock_no3(_ChanGQABlock):
    GROUPS = [0, 1, 2, 4, 5, 6, 7]

@register("chan_gqablock_no4")
class Chan_gqablock_no4(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 5, 6, 7]

@register("chan_gqablock_no5")
class Chan_gqablock_no5(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 4, 6, 7]

@register("chan_gqablock_no6")
class Chan_gqablock_no6(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 4, 5, 7]

@register("chan_gqablock_no7")
class Chan_gqablock_no7(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 4, 5, 6]

@register("chan_quest_no0")
class Chan_quest_no0(_ChanQuest):
    GROUPS = [1, 2, 3, 4, 5, 6, 7]

@register("chan_quest_no1")
class Chan_quest_no1(_ChanQuest):
    GROUPS = [0, 2, 3, 4, 5, 6, 7]

@register("chan_quest_no2")
class Chan_quest_no2(_ChanQuest):
    GROUPS = [0, 1, 3, 4, 5, 6, 7]

@register("chan_quest_no3")
class Chan_quest_no3(_ChanQuest):
    GROUPS = [0, 1, 2, 4, 5, 6, 7]

@register("chan_quest_no4")
class Chan_quest_no4(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 5, 6, 7]

@register("chan_quest_no5")
class Chan_quest_no5(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 4, 6, 7]

@register("chan_quest_no6")
class Chan_quest_no6(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 4, 5, 7]

@register("chan_quest_no7")
class Chan_quest_no7(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 4, 5, 6]



# --- importance-ordered cumulative keep sets (gqablock) ----------------------
@register("chan_gqablock_keep6")
class Chan_gqablock_keep6(_ChanGQABlock):
    GROUPS = [1, 3, 4, 5, 6, 7]

@register("chan_gqablock_keep5")
class Chan_gqablock_keep5(_ChanGQABlock):
    GROUPS = [1, 3, 5, 6, 7]

@register("chan_gqablock_keep3")
class Chan_gqablock_keep3(_ChanGQABlock):
    GROUPS = [3, 5, 7]

@register("chan_gqablock_keep2")
class Chan_gqablock_keep2(_ChanGQABlock):
    GROUPS = [3, 7]



# --- targeted keep-4 probes (g3/g7 membership test) ------------------------
@register("chan_gqablock_c0237")
class Chan_gqablock_c0237(_ChanGQABlock):
    GROUPS = [0, 2, 3, 7]

@register("chan_gqablock_c0145")
class Chan_gqablock_c0145(_ChanGQABlock):
    GROUPS = [0, 1, 4, 5]



# --- mask BOTH critical groups g3,g7 (keep other 6) -------------------------
@register("chan_block_no37")
class Chan_block_no37(_ChanBlock):
    GROUPS = [0, 1, 2, 4, 5, 6]

@register("chan_gqablock_no37")
class Chan_gqablock_no37(_ChanGQABlock):
    GROUPS = [0, 1, 2, 4, 5, 6]

@register("chan_quest_no37")
class Chan_quest_no37(_ChanQuest):
    GROUPS = [0, 1, 2, 4, 5, 6]



# --- control: mask 2 RANDOM non-critical groups (g3,g7 kept) ---------------
@register("chan_block_m01")
class Chan_block_m01(_ChanBlock):
    GROUPS = [2, 3, 4, 5, 6, 7]

@register("chan_gqablock_m01")
class Chan_gqablock_m01(_ChanGQABlock):
    GROUPS = [2, 3, 4, 5, 6, 7]

@register("chan_quest_m01")
class Chan_quest_m01(_ChanQuest):
    GROUPS = [2, 3, 4, 5, 6, 7]

@register("chan_block_m06")
class Chan_block_m06(_ChanBlock):
    GROUPS = [1, 2, 3, 4, 5, 7]

@register("chan_gqablock_m06")
class Chan_gqablock_m06(_ChanGQABlock):
    GROUPS = [1, 2, 3, 4, 5, 7]

@register("chan_quest_m06")
class Chan_quest_m06(_ChanQuest):
    GROUPS = [1, 2, 3, 4, 5, 7]

@register("chan_block_m14")
class Chan_block_m14(_ChanBlock):
    GROUPS = [0, 2, 3, 5, 6, 7]

@register("chan_gqablock_m14")
class Chan_gqablock_m14(_ChanGQABlock):
    GROUPS = [0, 2, 3, 5, 6, 7]

@register("chan_quest_m14")
class Chan_quest_m14(_ChanQuest):
    GROUPS = [0, 2, 3, 5, 6, 7]

@register("chan_block_m15")
class Chan_block_m15(_ChanBlock):
    GROUPS = [0, 2, 3, 4, 6, 7]

@register("chan_gqablock_m15")
class Chan_gqablock_m15(_ChanGQABlock):
    GROUPS = [0, 2, 3, 4, 6, 7]

@register("chan_quest_m15")
class Chan_quest_m15(_ChanQuest):
    GROUPS = [0, 2, 3, 4, 6, 7]

@register("chan_block_m16")
class Chan_block_m16(_ChanBlock):
    GROUPS = [0, 2, 3, 4, 5, 7]

@register("chan_gqablock_m16")
class Chan_gqablock_m16(_ChanGQABlock):
    GROUPS = [0, 2, 3, 4, 5, 7]

@register("chan_quest_m16")
class Chan_quest_m16(_ChanQuest):
    GROUPS = [0, 2, 3, 4, 5, 7]

@register("chan_block_m45")
class Chan_block_m45(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 6, 7]

@register("chan_gqablock_m45")
class Chan_gqablock_m45(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 6, 7]

@register("chan_quest_m45")
class Chan_quest_m45(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 6, 7]

@register("chan_block_m46")
class Chan_block_m46(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 5, 7]

@register("chan_gqablock_m46")
class Chan_gqablock_m46(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 5, 7]

@register("chan_quest_m46")
class Chan_quest_m46(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 5, 7]

@register("chan_block_m56")
class Chan_block_m56(_ChanBlock):
    GROUPS = [0, 1, 2, 3, 4, 7]

@register("chan_gqablock_m56")
class Chan_gqablock_m56(_ChanGQABlock):
    GROUPS = [0, 1, 2, 3, 4, 7]

@register("chan_quest_m56")
class Chan_quest_m56(_ChanQuest):
    GROUPS = [0, 1, 2, 3, 4, 7]

