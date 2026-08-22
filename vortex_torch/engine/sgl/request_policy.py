"""Retention and overflow policies for the request-bound cache domain.

The page-bound domain has no policy: a field's lifetime is the page's, decided by sglang's
allocator. The request domain does need one, because it holds state for a *unit* (today: a block)
that is being built up over several steps, and two questions have no single right answer:

**Retention** — when the unit is complete, does its slot go back to the pool?

``complete``
    Release on completion. The state was a means to an end: something else consumed it and it will
    never be read again. INT4 staging is exactly this — the staged bf16 exists only until the block
    can be quantized, and the packed bytes are authoritative afterwards.
``keep``
    Never release. The slot belongs to this key until the arena is torn down. For state that is
    accumulated across a whole request rather than per unit.
``residual_n``
    Keep the last ``n`` completed units per key, release older ones. For a flow that wants a short
    history — the previous block's summary as well as the current one.

**Overflow** — a key needs a slot and none is free.

``drop``
    Decline; the caller's write is skipped and counted. **Required for any flow whose unit must be
    complete to be usable**, INT4 included: evicting an incomplete unit means quantizing a block
    that was only partly written, which is silent data loss, whereas dropping is at least a counted
    loss of the newest token.
``evict_lru``
    Displace the least-recently-touched *complete* unit. Only sound when a partial unit is still
    meaningful, and even then it must never target an incomplete one.

Mirrors :mod:`cache_policy`'s shape: integer codes as ``tl.constexpr`` so the policy specialises the
kernel at compile time instead of branching per token.

**``residual_n`` and ``evict_lru`` are declared but NOT implemented, deliberately.** No caller needs
them, and both have failure modes that are silent rather than loud — a wrong retention leaks slots
until the arena starves (which reads as zeros everywhere, see :mod:`int4_arena`), and a wrong
eviction corrupts a unit that was still being written. Declaring them documents the design space;
:func:`resolve` refuses them so nobody gets a plausible-looking no-op. Implement one alongside the
flow that first needs it, with a test that reaches it.
"""

from __future__ import annotations

import triton.language as tl

#: Retention policy names.
RETENTIONS = ("complete", "residual_n", "keep")

#: Overflow policy names.
OVERFLOWS = ("drop", "evict_lru")

#: Kernel constants. Must be the ``tl.constexpr(...)`` CALL form -- a @jit'ed function cannot read a
#: plain module global, and the annotated form (``x: tl.constexpr = 0``) is unsupported. Triton only
#: reports this once the reading kernel compiles, so the error surfaces far from here.
RETAIN_COMPLETE = tl.constexpr(0)
RETAIN_RESIDUAL = tl.constexpr(1)
RETAIN_KEEP = tl.constexpr(2)

OVERFLOW_DROP = tl.constexpr(0)
OVERFLOW_EVICT_LRU = tl.constexpr(1)

#: Plain ints for launch sites (the constexprs above are for kernel code).
_RETENTION_CODES = {"complete": 0, "residual_n": 1, "keep": 2}
_OVERFLOW_CODES = {"drop": 0, "evict_lru": 1}

#: Combinations with no implementation. Kept as data rather than as an ``if`` chain so the message
#: can say *why* each one is refused -- a bare NotImplementedError invites someone to "just add the
#: branch", which is how a silent eviction bug gets written.
_UNIMPLEMENTED = {
    "residual_n": (
        "residual_n retention keeps the last n completed units per key, which needs a per-key ring "
        "of slots and a release rule for the oldest. Nothing needs it yet, and getting the release "
        "wrong leaks slots until the arena starves -- which reads as every counter at zero, not as "
        "an error."
    ),
    "evict_lru": (
        "evict_lru overflow displaces a complete unit to make room. It is only sound when a partial "
        "unit is still meaningful, and it must never target an incomplete one -- for INT4 that "
        "would quantize a partly written block, i.e. silent data loss. Use 'drop', which loses the "
        "same token but counts it."
    ),
}


def resolve(retention: str, overflow: str) -> tuple:
    """Validate a policy pair and return ``(retention_code, overflow_code)`` as plain ints.

    Refuses the declared-but-unimplemented options with the reason, rather than accepting them and
    behaving like the nearest implemented one.
    """
    if retention not in _RETENTION_CODES:
        raise ValueError(
            f"unknown request-cache retention {retention!r}; expected one of {RETENTIONS}"
        )
    if overflow not in _OVERFLOW_CODES:
        raise ValueError(
            f"unknown request-cache overflow {overflow!r}; expected one of {OVERFLOWS}"
        )
    for name in (retention, overflow):
        if name in _UNIMPLEMENTED:
            raise NotImplementedError(f"request-cache policy {name!r}: {_UNIMPLEMENTED[name]}")
    return _RETENTION_CODES[retention], _OVERFLOW_CODES[overflow]


__all__ = ["RETENTIONS", "OVERFLOWS", "RETAIN_COMPLETE", "RETAIN_RESIDUAL", "RETAIN_KEEP",
           "OVERFLOW_DROP", "OVERFLOW_EVICT_LRU", "resolve"]
