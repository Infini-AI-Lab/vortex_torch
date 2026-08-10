"""Selection recipes — the catalogue, and the test of whether the op set is right.

Each policy below is ~5 lines and needs **no new kernel**. That is the design's
own stated acceptance criterion for the frontend: "if a recipe needs a new op, the
op set was wrong."

Importing this module registers every policy, so ``SparseAttention("block_topk")``
works by name.
"""
from __future__ import annotations

import math

from . import ops
from .spec import Budget, Field, Selection, register


@register("block_topk")
class BlockTopK(Selection):
    """Centroid affinity: score each KV block by ``<q̄, mean(k)>``.

    The baseline policy, and the one most sparse-attention papers reduce to. Keeps
    an attention sink and the local window; everything else competes on score.
    """

    state = {"centroid": Field(reduce="mean", src="k")}
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        # One op instance per call site, per the op contract.
        self.qbar = ops.QSummary(how="mean")
        self.dot = ops.Dot(group_reduce="max")

    def score(self, q, state, ctx):
        return self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)


@register("quest")
class Quest(Selection):
    """Quest: the exact per-channel envelope bound on ``max_k <q, k>`` in a block.

    ``Envelope`` computes ``sum_d max(q_d * M_d, q_d * m_d)`` from the block's
    coordinate-wise key max ``M`` and min ``m``. Because ``q_d * k_d`` is monotone in
    ``k_d``, the largest attainable per-channel contribution is at one endpoint, so
    choosing the better endpoint per channel and summing is the tightest bound the
    envelopes support.

    The per-channel max must happen **before** the reduction over ``D``, which is why
    this needs a dedicated op rather than ``Dot(kmax) + Dot(kmin)``: two dots each
    collapse ``D`` first, losing the per-channel choice and computing a different,
    looser quantity. This recipe used that proxy until ``Envelope`` existed.
    """

    state = {
        "kmax": Field(reduce="max", src="k"),
        "kmin": Field(reduce="min", src="k"),
    }
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.env = ops.Envelope(group_reduce="max")

    def score(self, q, state, ctx):
        return self.env(self.qbar(q, ctx=ctx), state["kmax"], state["kmin"], ctx=ctx)


@register("lserve")
class LServe(Selection):
    """LServe: QUEST envelopes at **sub-block** granularity.

    Each KV block is split into runs of ``SUB_BLOCK`` tokens, and a coordinate-wise
    max/min key envelope is kept per run. A block is ranked by its single
    best-matching (group member, sub-block) pair, so **one** relevant sub-region is
    enough to select the block.

    Why that matters, and what it fixes in plain ``quest``: a whole-block envelope
    spans ``block_kv`` tokens, and ``max``/``min`` over a wide span is loose — the
    envelope widens with every unrelated token in the block, so the bound degrades
    exactly where the block is heterogeneous. Sub-block envelopes are tighter
    because each covers fewer tokens, and taking the max over runs keeps the
    property that a block only needs one good region to survive. Same selection
    machinery, strictly more informative state.

    Cost is ``block_kv / SUB_BLOCK`` times the envelope state (here 4x), which is
    small next to ``k`` itself, plus that many more contractions in the scorer.
    """

    SUB_BLOCK = 16

    state = {
        "kmax": Field(reduce="max", src="k", sub_block=SUB_BLOCK),
        "kmin": Field(reduce="min", src="k", sub_block=SUB_BLOCK),
    }
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.env = ops.Envelope(group_reduce="max")

    def score(self, q, state, ctx):
        return self.env(self.qbar(q, ctx=ctx), state["kmax"], state["kmin"], ctx=ctx)


@register("lserve_centroid")
class LServeCentroid(Selection):
    """Centroid routing at sub-block granularity — LServe's cheaper variant.

    One centroid per run of ``SUB_BLOCK`` keys; a block scores as the best match
    against any of its sub-centroids. Same "one good region is enough" property as
    :class:`LServe` at half the state (one field instead of two) and one contraction
    per sub-block instead of an envelope's two loads.

    Included as the controlled comparison against :class:`LServe`: it isolates
    *sub-block granularity* from *envelope-vs-centroid*, so a difference between the
    two is attributable.
    """

    SUB_BLOCK = 16

    state = {"centroid": Field(reduce="mean", src="k", sub_block=SUB_BLOCK)}
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.dot = ops.Dot(group_reduce="max")

    def score(self, q, state, ctx):
        return self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)


@register("streaming")
class Streaming(Selection):
    """Sink + local window only — no scoring, no state (StreamingLLM).

    The cheapest possible policy and a useful floor: any scored policy that cannot
    beat this on quality is not earning its state build. Exercises the stateless
    path, where ``build_state`` produces zero fields.
    """

    budget = Budget(topk=8, reserve_bos=1, reserve_local=7)

    def __init__(self) -> None:
        self.dist = ops.Distance()

    def score(self, q, state, ctx):
        # Reservations do the real work; the distance prior orders whatever budget
        # is left over toward recency.
        return self.dist(ctx=ctx)


@register("centroid_norm")
class CentroidNorm(Selection):
    """Centroid affinity plus a key-magnitude prior.

    A block whose keys are large in norm can produce a large logit for *some*
    query even when its mean direction is unremarkable, which a pure centroid dot
    underrates. Demonstrates composing a contraction with a state-only statistic.
    """

    state = {"centroid": Field(reduce="mean", src="k")}
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.dot = ops.Dot(group_reduce="max")
        self.norm = ops.Norm()
        self.scale = ops.Scale(factor=0.25)

    def score(self, q, state, ctx):
        aff = self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)
        return aff + self.scale(self.norm(state["centroid"], ctx=ctx), ctx=ctx)


@register("recency_biased")
class RecencyBiased(Selection):
    """Centroid affinity with a soft recency prior, scaled like a logit.

    Shows the difference between a *soft* preference (this, via ``Distance``) and a
    *hard* guarantee (``Budget.reserve_local``): the prior nudges ranking, the
    reservation cannot be outvoted.
    """

    state = {"centroid": Field(reduce="mean", src="k")}
    budget = Budget(topk=16, reserve_bos=1, reserve_local=2)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.dot = ops.Dot(group_reduce="max")
        self.decay = ops.Scale(factor=0.05)
        self.dist = ops.Distance()

    def score(self, q, state, ctx):
        aff = self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)
        return aff + self.decay(self.dist(ctx=ctx), ctx=ctx)


@register("value_norm")
class ValueNorm(Selection):
    """Score by how much a block's *values* can move the output.

    Inverts the usual framing: rather than asking "does the query match these
    keys", it asks "if attended, would this block change anything". Exercises
    ``src="v"`` in the state builder, which no other recipe here does.
    """

    state = {
        "centroid": Field(reduce="mean", src="k"),
        "vmean": Field(reduce="mean", src="v"),
    }
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.dot = ops.Dot(group_reduce="max")
        self.vnorm = ops.Norm()
        self.w = ops.Scale(factor=0.5)

    def score(self, q, state, ctx):
        aff = self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)
        return aff * self.w(self.vnorm(state["vmean"], ctx=ctx), ctx=ctx)


@register("scaled_centroid")
class ScaledCentroid(Selection):
    """``block_topk`` with the attention scale applied, so scores are logit-like.

    Ranking is scale-invariant, so this changes no selection — it exists to make
    the scores directly comparable to attention logits when debugging a policy, and
    to exercise ``Scale`` on the output node.
    """

    state = {"centroid": Field(reduce="mean", src="k")}
    budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

    def __init__(self) -> None:
        self.qbar = ops.QSummary(how="mean")
        self.dot = ops.Dot(group_reduce="max")
        self.scale = ops.Scale(factor=1.0 / math.sqrt(128))

    def score(self, q, state, ctx):
        return self.scale(self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx), ctx=ctx)
