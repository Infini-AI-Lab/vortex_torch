"""The request-bound cache domain: buffers, the key->slot map, and the claim/complete/release cycle.

Two domains, one compiler
-------------------------
=============  ==========================  =====================================
               PAGE (existing)             REQUEST (this module)
=============  ==========================  =====================================
declared by    ``create_cache``            ``create_request_cache``
allocated      ``num_blocks x r x c``      ``n_slots x r x c``
addressed by   block id                    device ``slot_of[key]`` map
scales with    context length              **concurrency**
format         ``FORMAT.PAGED``            ``FORMAT.SLOTTED``
=============  ==========================  =====================================

The page domain is right for state summarising *stored* KV -- centroids, envelopes -- because such
state lives exactly as long as the page it describes. It is wrong for state belonging to a request
*in flight*: that state is per-unit-under-construction, so sizing it per block wastes memory
proportional to context (measured: a per-block bf16 mirror cost 21632 B/block against bf16's own
16896, a 1.28x REGRESSION) and its lifetime has nothing to do with the page's.

This is a **domain**, not a set of INT4 kernels. Its fields are ``FORMAT.SLOTTED`` compiler
operands, so a flow reads both domains in ONE fused kernel -- see ``FORMAT.SLOTTED`` in
``abs/tensor.py`` and the addressing it generates.

Binding is by LIFETIME, not by addressing
-----------------------------------------
The obvious design is to index by request slot, and it does not work. ``set_kv_buffer`` is the only
hook where incoming K/V is visible and it receives no request id; ``req_to_token`` maps
``(req_slot, pos) -> kv_index``, so ``loc`` holds that map's *values*. Recovering the slot means
searching the map plus a host sync, which cudagraph capture forbids outright. Keys are therefore
**block ids**, already globally unique from sglang's page pool. "Request-bound" describes the
state's lifetime, not its index.

Fields excluded from ``token_ratio``
-----------------------------------
Those ratios are per-TOKEN budgets. A request-domain field is constant in size per *slot*, so
charging it per token reserves HBM that grows with context for a buffer that does not.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from .request_policy import resolve as resolve_policy

logger = logging.getLogger(__name__)

#: ``slot_of`` sentinels, matching the compiler's SLOTTED addressing: anything negative is a miss.
SLOT_FREE = -1
#: A claim in progress. See ``int4_arena._claim`` for why arbitration is on the KEY and not on the
#: slot pool -- optimistic claiming passes at decode width and drops most tokens at prefill width.
SLOT_CLAIMING = -2


class RequestCacheDomain:
    """Slot allocator + key->slot map for one layer's request-bound fields.

    Owns the **maps only**; the payload tensors live in the cache dict as ordinary fields. That
    split is deliberate and was learned by getting it wrong: an earlier version held the payload
    itself (``arena.k`` / ``arena.v``), so every consumer had to know about the domain in order to
    find the newest tokens, and the read path broke the moment staging became a pool field. Payload
    from the cache dict, maps from here -- which is also what makes the domain reusable by a
    non-INT4 flow.
    """

    def __init__(self, num_keys: int, n_slots: int, *,
                 retention: str = "complete", overflow: str = "drop",
                 device: str = "cuda"):
        self.retention_code, self.overflow_code = resolve_policy(retention, overflow)
        self.retention = retention
        self.overflow = overflow
        self.num_keys = int(num_keys)
        self.n_slots = int(n_slots)
        if self.n_slots <= 0:
            raise ValueError(f"request domain needs at least one slot, got {n_slots}")
        # Allocated up front, all of it. A lazily-grown map would be allocated during the first
        # captured step and the allocation itself becomes part of the graph.
        self.slot_of = torch.full((self.num_keys,), SLOT_FREE, dtype=torch.int32, device=device)
        self.owner_of = torch.full((self.n_slots,), -1, dtype=torch.int32, device=device)

    # -- geometry -------------------------------------------------------------------------------

    def nbytes_maps(self) -> int:
        """Bytes held by the maps. The payload is the caller's; see the class docstring."""
        return sum(t.element_size() * t.numel() for t in (self.slot_of, self.owner_of))

    def payload_bytes(self, meta: Dict[str, Tuple[int, int]], dtype=torch.bfloat16) -> int:
        """Bytes one layer's request-domain payload needs for ``meta`` at this slot count.

        Note what is NOT here: any dependence on ``num_keys``. That independence is the feature --
        it is what makes the domain constant in context length.
        """
        elt = torch._utils._element_size(dtype)
        return sum(self.n_slots * r * c * elt for (r, c) in meta.values())

    def allocate_payload(self, meta: Dict[str, Tuple[int, int]], *,
                         dtype=torch.bfloat16, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Allocate ``{name: [n_slots, r, c]}`` for a ``create_request_cache`` declaration.

        Leading dim is ``n_slots``, which is the entire difference from the page domain's allocation
        and the reason the compiler needs a distinct FORMAT rather than just a different shape: the
        *addressing* has to know to go through the map.
        """
        return {
            name: torch.zeros((self.n_slots, r, c), dtype=dtype, device=device)
            for name, (r, c) in meta.items()
        }

    # -- host-side driving ----------------------------------------------------------------------
    #
    # These exist for setup, tests and diagnostics. The per-step path is device-side (the claim runs
    # inside the update kernel), because a host round trip per token is both far too slow and
    # uncapturable.

    def reset(self) -> None:
        """Return every slot to the pool. Setup / teardown only."""
        self.slot_of.fill_(SLOT_FREE)
        self.owner_of.fill_(-1)

    def occupancy(self) -> int:
        """Slots currently held. Syncs -- diagnostics only, never a captured path."""
        return int((self.owner_of >= 0).sum())

    def slot_for(self, key: int) -> int:
        """This key's slot, or a negative sentinel. Syncs; diagnostics and tests only."""
        return int(self.slot_of[key])

    def release(self, key: int) -> None:
        """Hand a key's slot back. Order matters: clear the map first, then the owner.

        A slot whose ``owner`` is free must never still be reachable through the map, or a claimant
        that takes the slot and a reader that follows the stale map both address the same row.
        """
        slot = int(self.slot_of[key])
        if slot < 0:
            return
        self.slot_of[key] = SLOT_FREE
        self.owner_of[slot] = -1

    def __repr__(self) -> str:
        return (f"RequestCacheDomain(num_keys={self.num_keys}, n_slots={self.n_slots}, "
                f"retention={self.retention!r}, overflow={self.overflow!r})")


def request_domain_for(num_keys: int, concurrency: int, num_kv_heads: int,
                       **kwargs) -> RequestCacheDomain:
    """Build a domain sized by CONCURRENCY: one in-flight unit per (row, kv head).

    ``concurrency`` should be the cudagraph capture ceiling rather than anything derived from the
    live batch: the domain must be allocated before capture, and two derivations that were tried
    (``max_running_requests``, ``req_to_token_pool.size``) both undersized it into silent
    starvation -- 0% accuracy with every counter reading zero, because starvation is a lost claim
    rather than a refusal.
    """
    return RequestCacheDomain(num_keys, max(1, concurrency) * max(1, num_kv_heads), **kwargs)


__all__ = ["SLOT_FREE", "SLOT_CLAIMING", "RequestCacheDomain", "request_domain_for"]
