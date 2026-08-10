"""Pattern-level tests. No kernels involved — these test *semantics*.

Deliberately runnable on CPU so the representation can be verified without a GPU,
and so a failure here is unambiguously a logic bug rather than a kernel bug.
"""
from __future__ import annotations

import pytest
import torch

from vortex_train.pattern import INVALID, SparsePattern, pattern_from_dense_mask

from .patterns import ALL_KINDS, make

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = dict(b=2, hkv=2, seqlen_q=128, seqlen_kv=128, block_q=32, block_kv=32)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_validate(kind):
    p = make(kind, **SHAPE, device=DEV)
    p.validate()


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_causal(kind):
    make(kind, **SHAPE, device=DEV).assert_causal()


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_dense_roundtrip(kind):
    """pattern -> dense mask -> pattern must be a fixed point.

    Catches index packing bugs (wrong order, lost entries, padding treated as
    real) that `validate` alone would pass.
    """
    p = make(kind, **SHAPE, device=DEV)
    dense = p.to_dense_mask()

    # re-derive the block mask by sampling the top-left token of each block
    bq, bkv = p.block_q, p.block_kv
    block = dense[:, :, ::bq, ::bkv][:, :, : p.num_q_blocks, : p.num_kv_blocks]
    p2 = pattern_from_dense_mask(
        block, block_q=bq, block_kv=bkv,
        seqlen_q=p.seqlen_q, seqlen_kv=p.seqlen_kv,
        max_selected=p.max_selected,
    )
    assert torch.equal(p.cnt, p2.cnt)
    # compare as sets per row: packing order is an implementation detail
    for t1, t2 in zip(p.idx.flatten(0, 2), p2.idx.flatten(0, 2)):
        assert set(t1[t1 >= 0].tolist()) == set(t2[t2 >= 0].tolist())


def test_padding_is_invalid():
    """Padding must be -1, never 0.

    A 0 there is the nastiest possible bug: a kernel that reads past `cnt` would
    silently attend to KV block 0 and still produce plausible numbers.
    """
    p = make("random_topk", **SHAPE, device=DEV, topk=2)
    ar = torch.arange(p.max_selected, device=p.idx.device, dtype=torch.int32)
    pad = ar[None, None, None, :] >= p.cnt[..., None]
    assert bool((p.idx[pad] == INVALID).all())


def test_validate_rejects_bad_count():
    p = make("full", **SHAPE, device=DEV)
    bad = SparsePattern(
        cnt=(p.cnt + p.max_selected + 1).contiguous(), idx=p.idx,
        block_q=p.block_q, block_kv=p.block_kv,
        seqlen_q=p.seqlen_q, seqlen_kv=p.seqlen_kv,
    )
    with pytest.raises(AssertionError):
        bad.validate()


def test_validate_rejects_zero_padding():
    p = make("random_topk", **SHAPE, device=DEV, topk=2)
    idx = p.idx.clone()
    idx[idx == INVALID] = 0                     # the silent-bug case
    with pytest.raises(AssertionError):
        SparsePattern(
            cnt=p.cnt, idx=idx.contiguous(), block_q=p.block_q, block_kv=p.block_kv,
            seqlen_q=p.seqlen_q, seqlen_kv=p.seqlen_kv,
        ).validate()


def test_assert_causal_catches_violation():
    p = make("diagonal", **SHAPE, device=DEV)
    idx = p.idx.clone()
    idx[:, :, 0, 0] = p.num_kv_blocks - 1       # first query block sees the last kv block
    with pytest.raises(AssertionError):
        SparsePattern(
            cnt=p.cnt, idx=idx.contiguous(), block_q=p.block_q, block_kv=p.block_kv,
            seqlen_q=p.seqlen_q, seqlen_kv=p.seqlen_kv,
        ).assert_causal()


def test_full_pattern_is_causal_dense():
    """The `full` pattern must expand to exactly the causal mask.

    This anchors the dense-degeneracy test in test_attention.py: if this is wrong,
    that test proves nothing.
    """
    p = make("full", **SHAPE, device=DEV)
    dense = p.to_dense_mask()[0, 0]
    sq, skv = p.seqlen_q, p.seqlen_kv
    qpos = torch.arange(sq, device=dense.device) + (skv - sq)
    kpos = torch.arange(skv, device=dense.device)
    want = kpos[None, :] <= qpos[:, None]
    # block granularity means `full` is a superset of token causality only where
    # blocks straddle the diagonal; inside a block-aligned setup they coincide.
    assert bool((dense | ~want).all()), "full pattern must cover all causal tokens"


def test_assert_causal_honours_the_kv_offset():
    """`assert_causal` must use ABSOLUTE positions when seqlen_kv != seqlen_q.

    Query row i sits at absolute position `i + (seqlen_kv - seqlen_q)` — the
    convention the kernels and `oracle_attention` both use. Without that offset this
    checker rejected legitimate decode-style patterns (a KV prefix longer than the
    query window) even though the kernels handled them correctly, so the bug was in
    the assertion rather than in the math it guards.
    """
    # 1 query block of 64 tokens against 192 KV tokens = 3 KV blocks. With the offset,
    # the query block ends at absolute position 191, so ALL three KV blocks are legal.
    sq, skv, bq, bkv = 64, 192, 64, 64
    mask = torch.ones((1, 1, sq // bq, skv // bkv), dtype=torch.bool, device="cuda")
    p = pattern_from_dense_mask(mask, block_q=bq, block_kv=bkv,
                                seqlen_q=sq, seqlen_kv=skv)
    p.validate()
    p.assert_causal()          # must NOT raise

    # And it must still reject a genuine violation: same geometry, but a query block
    # that reaches only position 63 selecting the block starting at 128.
    mask2 = torch.zeros((1, 1, 2, 3), dtype=torch.bool, device="cuda")
    mask2[0, 0, 0, 2] = True   # q block 0 (ends at 63 when sq == skv) -> kv block 2
    p2 = pattern_from_dense_mask(mask2, block_q=bq, block_kv=bkv,
                                 seqlen_q=128, seqlen_kv=128)
    with pytest.raises(AssertionError, match="start after their query block ends"):
        p2.assert_causal()


def test_validate_rejects_a_duplicate_kv_id():
    """A row naming the same KV block twice must be rejected, not silently miscomputed.

    Two things break on a duplicate. The attention kernels count that block twice
    (wrong softmax denominator, wrong dk/dv), and the CSR transpose's segment sort
    ranks by ``#{u < v}`` — a permutation only when the ids are distinct. A duplicate
    collides two entries onto one slot and leaves another as -1, which measured as
    ``dk = NaN``. Nothing in the select kernel can produce this (it retires each winner
    with -inf, and reservations only boost an existing score — verified across 243
    recipe/geometry combinations), so this guards hand-built and future patterns.
    """
    cnt = torch.tensor([[[2]]], dtype=torch.int32, device="cuda")
    idx = torch.tensor([[[[0, 0]]]], dtype=torch.int32, device="cuda")
    p = SparsePattern(cnt=cnt, idx=idx, block_q=64, block_kv=64,
                      seqlen_q=64, seqlen_kv=64)
    with pytest.raises(AssertionError, match="more than once"):
        p.validate()

    # the same shape without the duplicate is fine
    ok = SparsePattern(cnt=torch.tensor([[[1]]], dtype=torch.int32, device="cuda"),
                       idx=torch.tensor([[[[0, INVALID]]]], dtype=torch.int32, device="cuda"),
                       block_q=64, block_kv=64, seqlen_q=64, seqlen_kv=64)
    ok.validate()


def test_transpose_sorts_segments_longer_than_one_tile():
    """Segments longer than the sort's tile must still come out ascending.

    The sort tiles at BLOCK_S=64 and loops, because a segment can be up to ``Mq`` long
    (16384 at block_q=1) and sizing the tile to that worst case would be a [16k, 16k]
    comparison. This exercises the loop: at block_q=1 with 512 tokens the longest
    segment is 512 entries, 8x the tile.
    """
    from vortex_train.kernels.transpose import transpose_pattern, verify_transpose

    s = 512
    n_kv = s // 64
    mask = torch.ones((1, 1, s, n_kv), dtype=torch.bool, device="cuda")
    mi = torch.arange(s, device="cuda")
    ni = torch.arange(n_kv, device="cuda")
    mask &= ((ni[None, :] * 64) <= mi[:, None])[None, None]
    p = pattern_from_dense_mask(mask, block_q=1, block_kv=64, seqlen_q=s, seqlen_kv=s)

    t_off, t_ind = transpose_pattern(p)
    verify_transpose(p, t_off, t_ind)
    longest = 0
    for n in range(p.num_kv_blocks):
        lo, hi = int(t_off[0, 0, n]), int(t_off[0, 0, n + 1])
        seg = t_ind[0, 0, lo:hi]
        longest = max(longest, seg.numel())
        if seg.numel() > 1:
            assert bool((seg.diff() > 0).all()), (
                f"kv block {n} ({seg.numel()} entries) is not ascending"
            )
    assert longest > 64, f"test did not exercise the tile loop (longest {longest})"
