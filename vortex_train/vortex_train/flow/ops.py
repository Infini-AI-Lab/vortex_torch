"""The scorer op set — symbolic nodes a ``Selection.score`` composes.

These do **not** compute. Calling one records a node in a graph; the compiler
fuses the graph into a single Triton kernel. That indirection is the whole reason
a compiler exists here rather than eager op objects: a scorer like
``QSummary -> Dot -> Add -> Scale`` executed as four torch ops would round-trip
the O(T²) block-score matrix through HBM four times (4 GB per layer at 1M
tokens), where the fused form keeps it in registers and emits only the surviving
indices.

The type system is deliberately three types, because that is all block scoring
needs:

* ``QVEC``   ``[G, D]``   — the query group's summary vector for one query block
* ``BLOCK``  ``[Nkv, D]`` — a per-KV-block state field
* ``SCORE``  ``[Nkv]``    — one number per KV block; what ``score`` must return

An op that cannot be expressed as a shape-preserving map or a contraction over
``D`` does not belong in a *block scorer*, and the compiler rejects it at trace
time rather than emitting something subtly wrong.

Mirrors ``vortex_torch``'s op contract: **each op instance is one call site.**
Instantiate a separate op per use (``self.dot_a = Dot(); self.dot_b = Dot()``)
rather than reusing one, so the graph has a distinct node per position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

# node kinds
QVEC = "qvec"
BLOCK = "block"
SCORE = "score"


@dataclass
class Node:
    """A traced value. ``kind`` is one of QVEC / BLOCK / SCORE."""

    op: str
    kind: str
    inputs: tuple[Node, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    #: assigned by the tracer; stable order for codegen
    id: int = -1

    def __repr__(self) -> str:  # keeps graph dumps readable in test failures
        return f"%{self.id}:{self.op}[{self.kind}]"

    # Operator sugar. Ops are still explicit objects for anything with attributes;
    # these exist so an arithmetic scorer reads like arithmetic.
    def __add__(self, other):
        return _binary("add", self, other)

    def __sub__(self, other):
        return _binary("sub", self, other)

    def __mul__(self, other):
        return _binary("mul", self, other)


class Graph:
    """Records nodes in construction order. One per traced ``score``."""

    def __init__(self) -> None:
        self.nodes: list[Node] = []

    def add(self, node: Node) -> Node:
        node.id = len(self.nodes)
        self.nodes.append(node)
        return node

    def __len__(self) -> int:
        return len(self.nodes)

    def dump(self) -> str:
        lines = []
        for n in self.nodes:
            args = ", ".join(repr(i) for i in n.inputs)
            at = f" {n.attrs}" if n.attrs else ""
            lines.append(f"  {n!r} = {n.op}({args}){at}")
        return "graph {\n" + "\n".join(lines) + "\n}"


# The tracer is a module-level current-graph, so ops need no explicit graph
# argument and a scorer body stays readable. Tracing is single-threaded and
# happens once at compile time, never per step.
_CURRENT: Graph | None = None


def _graph() -> Graph:
    if _CURRENT is None:
        raise RuntimeError(
            "vortex_train.flow.ops used outside a trace. Selection.score is traced "
            "by the compiler; do not call it directly."
        )
    return _CURRENT


class _Tracing:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def __enter__(self) -> Graph:
        global _CURRENT
        if _CURRENT is not None:
            raise RuntimeError("nested trace")
        _CURRENT = self.graph
        return self.graph

    def __exit__(self, *exc) -> None:
        global _CURRENT
        _CURRENT = None


def trace(graph: Graph) -> _Tracing:
    return _Tracing(graph)


def placeholder(op: str, kind: str, **attrs) -> Node:
    return _graph().add(Node(op=op, kind=kind, attrs=attrs))


def _as_node(x) -> Node:
    if isinstance(x, Node):
        return x
    if isinstance(x, (int, float)):
        return _graph().add(Node(op="const", kind=SCORE, attrs={"value": float(x)}))
    raise TypeError(f"expected a traced Node or a number, got {type(x).__name__}")


def _binary(op: str, a, b) -> Node:
    a, b = _as_node(a), _as_node(b)
    # Broadcasting a const against a SCORE is the only mixed-kind case allowed;
    # everything else must agree, because a silent kind coercion here would
    # produce a scorer that compiles and is wrong.
    if a.kind != b.kind:
        const_pair = {a.op, b.op} & {"const"}
        if not const_pair or {a.kind, b.kind} != {SCORE}:
            raise TypeError(
                f"{op}: operand kinds {a.kind} and {b.kind} do not match. Only "
                f"same-kind arithmetic (or a scalar constant against a score) is "
                f"defined for block scoring."
            )
    return _graph().add(Node(op=op, kind=SCORE if SCORE in (a.kind, b.kind) else a.kind,
                             inputs=(a, b)))


class Op:
    """Base for an op with attributes. Calling it records one node."""

    name: ClassVar[str]

    def __call__(self, *args, ctx=None, **kw) -> Node:  # noqa: ARG002 - ctx is contract
        raise NotImplementedError


@dataclass
class QSummary(Op):
    """Collapse a query block's tokens (and GQA group) into one vector per group.

    ``how="mean"`` is the standard choice: a query block's mean direction is what a
    centroid score is actually comparing against. ``"max"`` is available for
    envelope-style policies where the largest component matters more than the
    average.

    QVEC out: ``[G, D]``.
    """

    how: str = "mean"
    name: ClassVar[str] = "q_summary"

    def __call__(self, q: Node, ctx=None) -> Node:
        if q.kind != QVEC:
            raise TypeError(f"QSummary expects the query placeholder (qvec), got {q.kind}")
        if self.how not in ("mean", "max"):
            raise ValueError(f"QSummary.how must be 'mean' or 'max', got {self.how!r}")
        return _graph().add(Node(op=self.name, kind=QVEC, inputs=(q,), attrs={"how": self.how}))


@dataclass
class Dot(Op):
    """Contract a QVEC against a BLOCK field over ``D`` -> SCORE ``[Nkv]``.

    The workhorse: ``Dot()(qbar, state["centroid"])`` is the centroid score. Over
    a GQA group the per-head scores are reduced by ``group_reduce`` — ``"max"`` by
    default, because selection is *shared* across the group and a block that any
    head in the group wants badly should survive. Averaging instead lets one
    strongly-interested head be outvoted, which is the more common cause of a
    recall cliff at large group sizes.
    """

    group_reduce: str = "max"
    name: ClassVar[str] = "dot"

    def __call__(self, qvec: Node, blk: Node, ctx=None) -> Node:
        if qvec.kind != QVEC or blk.kind != BLOCK:
            raise TypeError(
                f"Dot expects (qvec, block), got ({qvec.kind}, {blk.kind}). To combine "
                f"two scores use arithmetic; to combine two block fields use a field op."
            )
        if self.group_reduce not in ("max", "mean", "sum"):
            raise ValueError(f"Dot.group_reduce must be max/mean/sum, got {self.group_reduce!r}")
        return _graph().add(Node(op=self.name, kind=SCORE, inputs=(qvec, blk),
                                 attrs={"group_reduce": self.group_reduce}))


@dataclass
class Scale(Op):
    """Multiply a score by a constant (e.g. ``1/sqrt(D)``)."""

    factor: float
    name: ClassVar[str] = "scale"

    def __call__(self, s: Node, ctx=None) -> Node:
        if s.kind != SCORE:
            raise TypeError(f"Scale expects a score, got {s.kind}")
        return _graph().add(Node(op=self.name, kind=SCORE, inputs=(s,),
                                 attrs={"factor": float(self.factor)}))


@dataclass
class Norm(Op):
    """L2 norm of a BLOCK field, as a SCORE — magnitude without direction.

    Useful as a tie-breaker or a prior: a block whose keys are large in norm can
    produce a large logit for *some* query even when its centroid direction is
    unremarkable, which a pure centroid dot underrates.
    """

    name: ClassVar[str] = "norm"

    def __call__(self, blk: Node, ctx=None) -> Node:
        if blk.kind != BLOCK:
            raise TypeError(f"Norm expects a block field, got {blk.kind}")
        return _graph().add(Node(op=self.name, kind=SCORE, inputs=(blk,)))


@dataclass
class Distance(Op):
    """Position-based score: ``-(query block - kv block)`` in blocks.

    Expresses recency/decay priors without any state. Combined with ``Scale`` this
    is how a soft local-window preference is written, as opposed to the hard
    guarantee ``Budget.reserve_local`` gives.
    """

    name: ClassVar[str] = "distance"

    def __call__(self, ctx=None) -> Node:
        return _graph().add(Node(op=self.name, kind=SCORE))


@dataclass
class Envelope(Op):
    """The exact QUEST bound: ``sum_d max(q_d * M_d, q_d * m_d)`` -> SCORE.

    Given per-block coordinate-wise key envelopes ``M = max(k)`` and ``m = min(k)``,
    this is the **tightest upper bound** on ``max_{k in block} <q, k>`` obtainable
    from the envelopes alone: for each channel independently, the largest attainable
    contribution of ``q_d * k_d`` over ``k_d in [m_d, M_d]`` is at one endpoint, so
    taking the better endpoint per channel and summing bounds the true maximum.

    Why this is its own op rather than ``Dot(kmax) + Dot(kmin)``: the sum of two
    dots is a *different and looser* quantity, because the max must be taken
    **per channel before** the reduction over ``D``. Once the two dots have each
    collapsed ``D``, the per-channel choice is gone. Approximating it that way was
    the honest-but-loose fallback the ``quest`` recipe used before this op existed;
    it is now exact.

    Requires two fields — the max and the min envelope of the *same* source — so it
    is the one scorer op taking two BLOCK operands.
    """

    group_reduce: str = "max"
    name: ClassVar[str] = "envelope"

    def __call__(self, qvec: Node, kmax: Node, kmin: Node, ctx=None) -> Node:
        if qvec.kind != QVEC:
            raise TypeError(f"Envelope expects the query summary first, got {qvec.kind}")
        for nm, blk in (("kmax", kmax), ("kmin", kmin)):
            if blk.kind != BLOCK:
                raise TypeError(f"Envelope's {nm} must be a state field, got {blk.kind}")
        if self.group_reduce not in ("max", "mean", "sum"):
            raise ValueError(
                f"Envelope.group_reduce must be max/mean/sum, got {self.group_reduce!r}"
            )
        return _graph().add(Node(op=self.name, kind=SCORE, inputs=(qvec, kmax, kmin),
                                 attrs={"group_reduce": self.group_reduce}))
