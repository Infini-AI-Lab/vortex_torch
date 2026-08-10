"""The sparse pattern — which KV blocks each query block attends to.

This is the one data structure the whole system agrees on. The scorer produces
it; the attention kernels consume it; the oracle expands it to a dense mask to
check everything else.

Layout (chosen to match what a block-sparse kernel actually wants to read):

    cnt [B, Hkv, Mq]        int32  how many KV blocks query-block m attends
    idx [B, Hkv, Mq, K]     int32  those KV block ids, packed left, valid = idx[..., :cnt]

Two deliberate choices:

* **counts + packed indices, not a bitmap.** A bitmap is O(Mq*Nkv) and grows
  quadratically in sequence length (4 GB/layer at 1M tokens); packed indices are
  O(Mq*K) with K = topk + reserved, which is what makes long context tractable.
  FlashAttention-4's `block_sparsity` module independently uses the same
  representation, which is reassuring.
* **`Hkv`, not `Hq`.** Selection is shared across a GQA group. That is not a
  simplification — it is what lets the attention kernel load one KV block and use
  it for all `Hq/Hkv` query heads in the group, which is the entire reason
  block-sparse GQA is efficient.

Entries beyond `cnt` are `-1` (INVALID) so a kernel that reads them unmasked
fails loudly rather than silently attending to block 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

INVALID = -1


@dataclass(frozen=True)
class SparsePattern:
    """Which KV blocks each (batch, kv-head, query-block) attends to.

    Frozen: a pattern is a *fact about a step*, produced once and consumed by
    forward and backward. Mutating it between the two would silently corrupt
    gradients, so the dataclass refuses.
    """

    cnt: torch.Tensor          # [B, Hkv, Mq] int32
    idx: torch.Tensor          # [B, Hkv, Mq, K] int32
    block_q: int               # query-block size in tokens
    block_kv: int              # kv-block size in tokens
    seqlen_q: int
    seqlen_kv: int

    # ---------------------------------------------------------------- shape
    @property
    def batch(self) -> int:
        return self.cnt.shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self.cnt.shape[1]

    @property
    def num_q_blocks(self) -> int:
        return self.cnt.shape[2]

    @property
    def max_selected(self) -> int:
        """K — the padded width of ``idx``, not the per-row count."""
        return self.idx.shape[3]

    @property
    def num_kv_blocks(self) -> int:
        return (self.seqlen_kv + self.block_kv - 1) // self.block_kv

    # ------------------------------------------------------------- validate
    def validate(self) -> None:
        """Assert the invariants a kernel is allowed to rely on.

        Called from tests and from the debug path — not per step. Every check
        here corresponds to a way a wrong pattern would otherwise fail silently
        (attending to the future, or to block 0 by accident).
        """
        assert self.cnt.dtype == torch.int32, f"cnt must be int32, got {self.cnt.dtype}"
        assert self.idx.dtype == torch.int32, f"idx must be int32, got {self.idx.dtype}"
        assert self.cnt.device == self.idx.device
        assert self.cnt.is_contiguous() and self.idx.is_contiguous()

        b, h, m = self.cnt.shape
        assert self.idx.shape[:3] == (b, h, m), (
            f"idx {tuple(self.idx.shape)} inconsistent with cnt {tuple(self.cnt.shape)}"
        )
        assert m == (self.seqlen_q + self.block_q - 1) // self.block_q, (
            f"num_q_blocks {m} does not match seqlen_q {self.seqlen_q} / block_q {self.block_q}"
        )

        # counts in range
        assert int(self.cnt.min()) >= 0, "negative count"
        assert int(self.cnt.max()) <= self.max_selected, (
            f"count {int(self.cnt.max())} exceeds idx width {self.max_selected}"
        )

        # valid entries are real block ids; padding is INVALID
        ar = torch.arange(self.max_selected, device=self.idx.device, dtype=torch.int32)
        valid = ar[None, None, None, :] < self.cnt[..., None]
        sel = self.idx[valid]
        if sel.numel():        # a pattern may legitimately select nothing at all
            assert int(sel.min()) >= 0, "valid entry is negative"
            assert int(sel.max()) < self.num_kv_blocks, (
                "valid entry indexes a kv block past the end of the sequence"
            )
        assert bool((self.idx[~valid] == INVALID).all()), (
            "padding entries must be INVALID (-1); a 0 there would silently "
            "attend to the first kv block"
        )

        # No row may name the same KV block twice. Two things break on a duplicate:
        # the attention kernels would count that block's contribution twice (wrong
        # softmax denominator, wrong dk/dv), and the CSR transpose's segment sort
        # ranks by `#{u < v}`, which is only a permutation when the ids are distinct —
        # a duplicate collides two entries onto one slot, leaves another unwritten as
        # -1, and produced `dk = NaN` when measured. Checked here rather than trusted
        # because a scorer bug or a hand-built pattern can both produce it.
        scratch = torch.full_like(self.idx, self.num_kv_blocks)
        safe = torch.where(valid, self.idx, scratch).to(torch.int64)
        hist = torch.zeros(
            self.cnt.shape + (self.num_kv_blocks + 1,), dtype=torch.int32,
            device=self.idx.device,
        )
        hist.scatter_add_(3, safe, valid.to(torch.int32))
        dup = hist[..., : self.num_kv_blocks] > 1
        assert not bool(dup.any()), (
            f"{int(dup.sum())} (row, kv block) pairs are selected more than once; "
            f"a row must name each KV block at most once"
        )

    def assert_causal(self) -> None:
        """No query block may select a KV block that starts after it ends.

        Separate from :meth:`validate` because a non-causal pattern is legitimate
        (encoder attention); only the causal recipes must satisfy this.
        """
        m_ar = torch.arange(self.num_q_blocks, device=self.idx.device, dtype=torch.int32)
        # Last token of query block m, in ABSOLUTE positions. The
        # `seqlen_kv - seqlen_q` offset is what makes `seqlen_q != seqlen_kv` work:
        # query row i sits at absolute position `i + (seqlen_kv - seqlen_q)`, which is
        # the convention the kernels and `oracle_attention` both use. Omitting it here
        # made this checker reject legitimate decode-style patterns (any prefix of KV
        # longer than the query window) while the kernels handled them correctly.
        offset = self.seqlen_kv - self.seqlen_q
        q_end = (m_ar + 1) * self.block_q - 1 + offset
        # first token of kv block n
        kv_start = self.idx.clamp(min=0).to(torch.int64) * self.block_kv
        ar = torch.arange(self.max_selected, device=self.idx.device, dtype=torch.int32)
        valid = ar[None, None, None, :] < self.cnt[..., None]
        violates = valid & (kv_start > q_end[None, None, :, None])
        assert not bool(violates.any()), (
            f"{int(violates.sum())} selected blocks start after their query block ends"
        )

    # ----------------------------------------------------------- densify
    def to_dense_mask(self) -> torch.Tensor:
        """Expand to ``[B, Hkv, seqlen_q, seqlen_kv]`` bool — the oracle's input.

        Deliberately simple and O(T²): this exists to be *obviously correct*, so
        it is the reference the fast paths are checked against. Never call it in
        training; it is what the packed representation exists to avoid.
        """
        b, h, m = self.cnt.shape
        n_kv = self.num_kv_blocks
        dev = self.idx.device

        ar = torch.arange(self.max_selected, device=dev, dtype=torch.int32)
        valid = ar[None, None, None, :] < self.cnt[..., None]        # [B,H,M,K]

        # Scatter selected block ids into a [B,H,M,n_kv] block-level mask.
        #
        # Padding must go to a SCRATCH column, not to column 0. `scatter_` writes
        # every one of the K slots, so mapping padding to index 0 makes a padded
        # slot write `False` over the `True` a real slot wrote for block 0 —
        # silently dropping it. That bug only appears when a row both has padding
        # and selects block 0 (i.e. exactly the BOS-sink recipes), which is why it
        # survived the patterns without padding.
        block_mask = torch.zeros((b, h, m, n_kv + 1), dtype=torch.bool, device=dev)
        scratch = torch.full_like(self.idx, n_kv)
        safe_idx = torch.where(valid, self.idx, scratch).to(torch.int64)
        block_mask.scatter_(3, safe_idx, valid)
        block_mask = block_mask[..., :n_kv]        # drop the scratch column

        # block -> token, then crop the ragged tail
        token_mask = block_mask.repeat_interleave(self.block_q, dim=2)
        token_mask = token_mask.repeat_interleave(self.block_kv, dim=3)
        return token_mask[:, :, : self.seqlen_q, : self.seqlen_kv]


def pattern_from_dense_mask(
    block_mask: torch.Tensor,
    *,
    block_q: int,
    block_kv: int,
    seqlen_q: int,
    seqlen_kv: int,
    max_selected: int | None = None,
) -> SparsePattern:
    """Build a :class:`SparsePattern` from a ``[B, Hkv, Mq, Nkv]`` bool block mask.

    The inverse of :meth:`SparsePattern.to_dense_mask` at block granularity. Used
    by tests to construct hand-written patterns, and to round-trip-check the
    representation. Device-only: no host loop, no ``.item()``.
    """
    assert block_mask.dtype == torch.bool and block_mask.dim() == 4
    b, h, m, n = block_mask.shape

    cnt = block_mask.sum(dim=3).to(torch.int32)
    k = max_selected if max_selected is not None else max(int(cnt.max()), 1)

    # Rank the selected entries per row, then scatter each into its rank slot.
    # argsort on a bool key is stable in the sense we need: True first, and
    # within True the original (ascending block id) order is preserved by using
    # a composite key instead of relying on sort stability.
    ar_n = torch.arange(n, device=block_mask.device, dtype=torch.int32)
    key = torch.where(block_mask, ar_n.expand_as(block_mask), torch.full_like(ar_n.expand_as(block_mask), n))
    order = key.argsort(dim=3)[..., :k]                      # first k = selected, ascending
    idx = torch.gather(key, 3, order)
    idx = torch.where(idx < n, idx, torch.full_like(idx, INVALID)).to(torch.int32)

    return SparsePattern(
        cnt=cnt.contiguous(),
        idx=idx.contiguous(),
        block_q=block_q,
        block_kv=block_kv,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
    )
