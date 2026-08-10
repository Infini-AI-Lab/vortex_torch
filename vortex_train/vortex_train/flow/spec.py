"""The frontend contract — what a user writes to define a selection policy.

A ``Selection`` declares three things and no more:

* ``state`` — per-KV-block summaries to build once per forward (a *declarative*
  reduction spec, not a method). Training sees the whole sequence at once, so
  "build per-block state" is a reduction over the block axis and can be stated
  rather than coded. This is the main simplification versus ``vortex_torch``,
  which needs ``create_cache``/``forward_cache`` because it updates state
  incrementally per decode step.
* ``budget`` — how many blocks to keep, and which are reserved (BOS sink, local
  window, EOS). Reservations are **config, not user code**: getting "always keep
  the sink and the local window" right inside every scorer is exactly the bug
  farm this avoids. The compiler force-includes them and shrinks the top-k.
* ``score`` — a symbolic expression scoring every KV block for a query group.

What the user never touches: the top-k itself, the causal edge, dedup between
reserved and scored blocks, the CSR transpose, the autograd wiring, or a kernel.
Those are the parts that fail *silently* when hand-written — a subtly wrong
pattern still trains, just to a different objective.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

REDUCTIONS = ("mean", "max", "min", "sum")
SOURCES = ("k", "v")


@dataclass(frozen=True)
class Field:
    """A per-KV-block summary, reduced over the tokens in the block.

    ``reduce="mean", src="k"`` is the centroid recipe; ``max``/``min`` over ``k``
    give the quest-style envelope.

    **``sub_block``** splits each KV block into consecutive runs of that many tokens
    and keeps one summary *per run* — the LServe refinement. It matters because a
    whole-block summary is a lossy average over a region that may be mostly
    irrelevant: one hot sub-region gets diluted by its neighbours and the block
    looks uninteresting. With sub-block state the scorer maxes over the runs, so a
    single relevant sub-region is enough to select the block. Cost is
    ``block_kv/sub_block`` times the state, which is small next to `k` itself.

    ``sub_block=None`` (default) means one summary per block, i.e.
    ``sub_block == block_kv``.
    """

    reduce: str = "mean"
    src: str = "k"
    sub_block: int | None = None

    def __post_init__(self) -> None:
        if self.reduce not in REDUCTIONS:
            raise ValueError(f"Field.reduce must be one of {REDUCTIONS}, got {self.reduce!r}")
        if self.src not in SOURCES:
            raise ValueError(f"Field.src must be one of {SOURCES}, got {self.src!r}")
        if self.sub_block is not None:
            if self.sub_block < 1 or self.sub_block & (self.sub_block - 1):
                raise ValueError(
                    f"Field.sub_block must be a positive power of two, got {self.sub_block}"
                )

    def num_sub(self, block_kv: int) -> int:
        """How many summaries this field keeps per KV block."""
        if self.sub_block is None:
            return 1
        if block_kv % self.sub_block:
            raise ValueError(
                f"sub_block={self.sub_block} does not divide block_kv={block_kv}"
            )
        return block_kv // self.sub_block


@dataclass(frozen=True)
class Budget:
    """How many KV blocks a query block may attend, and which are guaranteed.

    ``topk`` is the **total** block budget, reservations included — not a bonus on
    top. That makes the compute cost of a policy exactly ``topk * block_kv`` tokens
    per query block regardless of how the reservations are set, which is the
    property that makes a benchmark interpretable.

    Reservations are in *blocks*:

    * ``reserve_bos`` — the first blocks of the sequence (attention sink).
    * ``reserve_local`` — the most recent blocks up to and including the query
      block's own diagonal (the local window). Default 1, i.e. a query block
      always sees itself; a policy that can drop its own diagonal is almost
      always a bug rather than a choice.
    * ``reserve_eos`` — the last blocks of the sequence. Under causal masking
      these are only visible to query blocks near the end, so this mostly matters
      for non-causal use; it is here because the selection contract promises it.
    """

    topk: int
    reserve_bos: int = 0
    reserve_local: int = 1
    reserve_eos: int = 0

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")
        for name in ("reserve_bos", "reserve_local", "reserve_eos"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.reserved > self.topk:
            # Otherwise the top-k would silently drop a block the user asked to
            # guarantee — the exact class of quiet miscompile this layer exists to
            # prevent.
            raise ValueError(
                f"reserved blocks ({self.reserved} = bos {self.reserve_bos} + local "
                f"{self.reserve_local} + eos {self.reserve_eos}) exceed the total "
                f"budget topk={self.topk}; raise topk or lower the reservations"
            )

    @property
    def reserved(self) -> int:
        """Upper bound on force-included blocks. Actual count is lower where the
        reserved ranges overlap (e.g. BOS *is* the diagonal for query block 0);
        the selection kernel dedups, so this is a bound, not a promise."""
        return self.reserve_bos + self.reserve_local + self.reserve_eos


class Selection:
    """Base class for a selection policy. Subclass, declare, register.

    Example — score each KV block by its centroid's affinity to the query group,
    keep a sink and a local window::

        @register("block_topk")
        class BlockTopK(Selection):
            state = {"centroid": Field(reduce="mean", src="k")}
            budget = Budget(topk=16, reserve_bos=1, reserve_local=1)

            def score(self, q, state, ctx):
                qbar = ops.QSummary(how="mean")(q, ctx=ctx)
                return ops.Dot()(qbar, state["centroid"], ctx=ctx)

    ``score`` is **traced, not executed** — ``q`` and ``state`` are symbolic
    nodes, and the returned node is compiled into one Triton kernel fused with the
    top-k. So the block-score matrix (O(T²), 4 GB per layer at 1M tokens) never
    reaches HBM; only the surviving indices do.
    """

    #: name -> Field. Built once per forward, shared by every query block.
    state: ClassVar[dict[str, Field]] = {}
    #: block budget and reservations.
    budget: ClassVar[Budget]
    #: user-chosen block sizes, in tokens.
    block_q: ClassVar[int] = 64
    block_kv: ClassVar[int] = 64
    #: causal masking. Selection never returns a block the mask forbids.
    causal: ClassVar[bool] = True

    def score(self, q, state, ctx):  # noqa: D102 - contract documented above
        raise NotImplementedError(
            f"{type(self).__name__} must implement score(self, q, state, ctx) and "
            f"return a score node built from vortex_train.flow.ops"
        )

    # ------------------------------------------------------------------ checks
    @classmethod
    def validate(cls) -> None:
        """Catch a malformed policy at registration/compile time, not mid-training."""
        if not isinstance(getattr(cls, "budget", None), Budget):
            raise TypeError(f"{cls.__name__}.budget must be a Budget instance")
        if not isinstance(cls.state, dict):
            raise TypeError(f"{cls.__name__}.state must be a dict of name -> Field")
        for name, field in cls.state.items():
            if not isinstance(field, Field):
                raise TypeError(
                    f"{cls.__name__}.state[{name!r}] must be a Field, got {type(field).__name__}"
                )
            if field.sub_block is not None and cls.block_kv % field.sub_block:
                raise ValueError(
                    f"{cls.__name__}.state[{name!r}]: sub_block={field.sub_block} does not "
                    f"divide block_kv={cls.block_kv}"
                )
        for attr in ("block_q", "block_kv"):
            val = getattr(cls, attr)
            if val < 1 or val & (val - 1):
                raise ValueError(
                    f"{cls.__name__}.{attr} must be a positive power of two, got {val}"
                )
        # A scorer with no state can only express position-based policies (pure
        # BOS+local). That is legitimate, so it is allowed but worth no warning.


REGISTRY: dict[str, type[Selection]] = {}


def register(name: str):
    """Register a Selection under ``name`` so configs can refer to it by string."""

    def deco(cls: type[Selection]) -> type[Selection]:
        if not issubclass(cls, Selection):
            raise TypeError(f"@register({name!r}) applied to {cls.__name__}, not a Selection")
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise ValueError(f"selection {name!r} already registered to {REGISTRY[name].__name__}")
        cls.validate()
        REGISTRY[name] = cls
        return cls

    return deco


def get_selection(name: str) -> type[Selection]:
    if name not in REGISTRY:
        raise KeyError(f"unknown selection {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]
