"""Trace a ``Selection.score`` and lower it to a kernel tape.

Three phases, mirroring ``vortex_torch``'s ``profile -> graph -> codegen``:

1. **trace** — call ``score`` with symbolic nodes. Nothing is allocated and no
   kernel runs; the result is a :class:`~vortex_train.flow.ops.Graph`.
2. **lower** — topologically flatten the graph into parallel int/float arrays
   (the "tape") that the fused select kernel interprets. Fields are resolved to
   integer slots here, so the kernel does no name lookup.
3. **bind** — produce a :class:`CompiledSelection` holding the tape, the field
   list for ``build_state``, and the budget. Everything shape- or
   policy-dependent is resolved at this point, so the per-step path is pure
   kernel launches.

The "tape" is an intermediate representation, not something interpreted at runtime:
:mod:`vortex_train.kernels.select` turns it into **generated Triton source**, one
kernel per policy. An interpreted tape was the first design and is the wrong one —
it would emit every op's body at every node and branch on a loaded value each step,
so the fusion would be nominal (the register pressure of the whole op set at every
position), and Triton cannot express the graph walk anyway (no dynamic list to index
register tensors by node id).

The compile happens **once**, at module construction. Nothing here runs per step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..flow import ops as O
from ..flow.spec import Budget, Selection
from ..kernels import select as S

_OPCODE = {
    "const": S.OP_CONST,
    "dot": S.OP_DOT,
    "scale": S.OP_SCALE,
    "norm": S.OP_NORM,
    "distance": S.OP_DISTANCE,
    "add": S.OP_ADD,
    "sub": S.OP_SUB,
    "mul": S.OP_MUL,
    "envelope": S.OP_ENVELOPE,
}
_GRED = {"max": S.GR_MAX, "mean": S.GR_MEAN, "sum": S.GR_SUM}


@dataclass
class CompiledSelection:
    """A traced, lowered selection policy. Immutable and reusable across steps."""

    name: str
    fields: tuple[tuple[str, str, int], ...]  # ordered (reduce, src, num_sub)
    budget: Budget
    block_q: int
    block_kv: int
    causal: bool
    q_how: str
    tape: dict
    graph: O.Graph                            # kept for dumping in test failures

    @property
    def num_nodes(self) -> int:
        return self.tape["num_nodes"]


def compile_selection(sel: Selection | type[Selection]) -> CompiledSelection:
    """Trace and lower a Selection. Raises on anything the backend can't honour."""
    cls = sel if isinstance(sel, type) else type(sel)
    cls.validate()
    inst = sel() if isinstance(sel, type) else sel

    # ---- phase 1: trace ---------------------------------------------------
    graph = O.Graph()
    field_names = tuple(cls.state.keys())
    with O.trace(graph):
        q = O.placeholder("q", O.QVEC)
        state = {
            name: O.placeholder("field", O.BLOCK, slot=i, field=name,
                                nsub=cls.state[name].num_sub(cls.block_kv))
            for i, name in enumerate(field_names)
        }
        out = inst.score(q, state, ctx=None)

    if not isinstance(out, O.Node):
        raise TypeError(
            f"{cls.__name__}.score must return a traced node, got "
            f"{type(out).__name__}. Build the score with vortex_train.flow.ops."
        )
    if out.kind != O.SCORE:
        raise TypeError(
            f"{cls.__name__}.score must return a SCORE (one number per KV block), "
            f"got {out.kind}. A Dot() of the query summary against a state field "
            f"produces a score."
        )

    # ---- phase 2: lower ---------------------------------------------------
    # `q_summary` and `field` nodes are folded away: the kernel loads q and state
    # directly, so they carry no tape entry. Their attributes are hoisted to the
    # compiled object (q_how) or to the consuming node (field slot).
    q_how = "mean"
    for n in graph.nodes:
        if n.op == "q_summary":
            q_how = n.attrs["how"]

    def field_slot_of(node: O.Node) -> int:
        if node.op != "field":
            raise TypeError(
                f"expected a state field as the block operand, got {node.op!r}. "
                f"Block fields come from the `state` dict passed to score()."
            )
        return node.attrs["slot"]

    def sub_of(node: O.Node) -> int:
        return node.attrs["nsub"]

    tape_nodes: list[O.Node] = [n for n in graph.nodes if n.op in _OPCODE]
    if not tape_nodes:
        raise ValueError(
            f"{cls.__name__}.score produced no computation. A scorer must combine "
            f"the query with state or position (see flow/ops.py)."
        )
    pos = {n.id: i for i, n in enumerate(tape_nodes)}
    if out.id not in pos:
        raise ValueError(f"{cls.__name__}.score returned a node that is not computable")

    n = len(tape_nodes)
    ops_a = [0] * n
    arg0 = [0] * n
    arg1 = [0] * n
    fslot = [0] * n
    fslot2 = [0] * n            # Envelope's second field (the min envelope)
    gred = [0] * n
    cval = [0.0] * n
    nsub = [1] * n              # sub-block count of the field this node reads

    for i, node in enumerate(tape_nodes):
        ops_a[i] = _OPCODE[node.op]
        if node.op == "dot":
            qvec, blk = node.inputs
            if qvec.kind != O.QVEC:
                raise TypeError("Dot's first operand must be the query summary")
            fslot[i] = field_slot_of(blk)
            nsub[i] = sub_of(blk)
            gred[i] = _GRED[node.attrs["group_reduce"]]
        elif node.op == "envelope":
            _, kmax, kmin = node.inputs
            fslot[i] = field_slot_of(kmax)
            fslot2[i] = field_slot_of(kmin)
            if sub_of(kmax) != sub_of(kmin):
                # The bound pairs channel-wise endpoints of the SAME region; if the
                # two envelopes were summarised over different spans the max/min
                # would not bracket a common set of keys and the "bound" would not
                # bound anything.
                raise ValueError(
                    f"Envelope's max and min fields must share sub_block "
                    f"({sub_of(kmax)} vs {sub_of(kmin)})"
                )
            nsub[i] = sub_of(kmax)
            gred[i] = _GRED[node.attrs["group_reduce"]]
        elif node.op == "norm":
            fslot[i] = field_slot_of(node.inputs[0])
            nsub[i] = sub_of(node.inputs[0])
        elif node.op == "scale":
            arg0[i] = pos[node.inputs[0].id]
            cval[i] = float(node.attrs["factor"])
        elif node.op == "const":
            cval[i] = float(node.attrs["value"])
        elif node.op in ("add", "sub", "mul"):
            a, b = node.inputs
            for operand in (a, b):
                if operand.id not in pos:
                    raise TypeError(
                        f"{node.op}: operand {operand!r} is not a computed score. "
                        f"Arithmetic combines scores, not raw fields — wrap a field "
                        f"in Dot() or Norm() first."
                    )
            arg0[i] = pos[a.id]
            arg1[i] = pos[b.id]

    # Tuples, not tensors: the tape is passed to the kernel as `tl.constexpr`, so
    # the op dispatch resolves at Triton compile time and the emitted code is a
    # straight-line chain of only the ops this policy uses. Tensors would force a
    # branch on a loaded value at every node and emit every op's body at every
    # position -- the fusion would be nominal rather than real. Cost: one Triton
    # compile per distinct policy, paid once. Tuples are also hashable, which is
    # what lets Triton cache that compile.
    budget = cls.budget
    tape = {
        "ops": tuple(ops_a), "arg0": tuple(arg0), "arg1": tuple(arg1),
        "fslot": tuple(fslot), "fslot2": tuple(fslot2), "gred": tuple(gred),
        "cval": tuple(cval), "nsub": tuple(nsub),
        "num_nodes": n, "out_node": pos[out.id],
    }
    return CompiledSelection(
        name=cls.__name__,
        fields=tuple((f.reduce, f.src, f.num_sub(cls.block_kv))
                     for f in cls.state.values()),
        budget=budget,
        block_q=cls.block_q,
        block_kv=cls.block_kv,
        causal=cls.causal,
        q_how=q_how,
        tape=tape,
        graph=graph,
    )
