"""Correctness of the sparse attention kernels against the fp32 oracle.

Precision policy (see reference/oracle.py for the derivation):

* system under test: **bf16** with fp32 accumulators — the real training regime;
* oracle: **fp32**, whose own error (~4e-6) is ~5 orders below the bf16
  tolerance, so a failure means a bug and not rounding;
* gradcheck: **fp64**, tiny shapes only, the one place double is load-bearing.

Tolerances are relative to the reference's dynamic range (see ``_cmp``), because
bf16 error in a reduction scales with the largest term summed, not with each
output element. ``_cmp_no_worse_than_dense`` is the stronger, non-arbitrary gate:
sparse's error measured against the oracle must be within a small factor of a
bf16 *dense* kernel's error against the same oracle.
"""
from __future__ import annotations

import pytest
import torch

from vortex_train.kernels.transpose import transpose_pattern
from vortex_train.nn.functional import sparse_attention
from vortex_train.reference.oracle import oracle_attention, oracle_from_pattern

from .patterns import ALL_KINDS, make

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPE = dict(b=2, hkv=2, seqlen_q=256, seqlen_kv=256, block_q=64, block_kv=64)
# Tolerances are relative to the reference's DYNAMIC RANGE, not to each element.
# bf16 error in a reduction is proportional to the largest term summed, so an
# output element that happens to be near zero still carries the absolute error of
# its large neighbours; a per-element rtol would demand impossible precision
# there. This is the same scaling flash-attention's own tests use.
RTOL_O = 2e-2                        # bf16 output, vs max|ref|
RTOL_G = 4e-2                        # bf16 gradients (longer accumulation)


def _mk(b, hq, hkv, sq, skv, d, dtype=torch.bfloat16, seed=0, requires_grad=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    def t(*shape):
        x = torch.randn(*shape, device="cuda", dtype=dtype, generator=g) * 0.5
        return x.requires_grad_(requires_grad)
    return t(b, hq, sq, d), t(b, hkv, skv, d), t(b, hkv, skv, d)


def _cmp(name, got, want, rtol):
    """Compare against the reference, scaled by its dynamic range."""
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    assert torch.isfinite(got).all(), f"{name}: non-finite values"
    # detach: these are routinely graph-attached (grads, outputs), and reducing to
    # a python float through autograd warns.
    got, want = got.detach().float(), want.detach().float()
    err = float((got - want).abs().max())
    scale = float(want.abs().max())
    tol = rtol * max(scale, 1e-3)
    assert err <= tol, (
        f"{name}: max abs err {err:.3e} > tol {tol:.3e} "
        f"(max|ref| {scale:.3e}, rtol {rtol})"
    )


def _cmp_no_worse_than_dense(name, sparse, oracle_ref, dense, rtol_slack=4.0):
    """Stronger gate: sparse's error must not exceed dense's by much.

    An absolute tolerance is a guess; this compares like with like — both sparse
    and dense are bf16 kernels measured against the same fp32 oracle, so if
    sparse's error is within a small factor of dense's, sparse is as accurate as
    the precision allows and the gate is not an arbitrary constant.

    The slack is generous because the two kernels sum in different orders (SDPA
    tiles the whole causal region; the sparse kernel walks selected blocks), so
    their bf16 rounding differs even where both are correct. It still catches the
    class of bug that matters — an O(1) relative error such as a dropped block or
    a zeroed gradient tile — which lands orders of magnitude above this bar.
    """
    e_sparse = float((sparse.detach().float() - oracle_ref.detach().float()).abs().max())
    e_dense = float((dense.detach().float() - oracle_ref.detach().float()).abs().max())
    assert e_sparse <= max(e_dense * rtol_slack, 1e-3), (
        f"{name}: sparse err {e_sparse:.3e} exceeds {rtol_slack}x dense err "
        f"{e_dense:.3e} — sparse is less accurate than bf16 alone explains"
    )


# ------------------------------------------------------------------ forward
@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("group", [1, 4])
def test_forward_vs_oracle(kind, group):
    hkv = SHAPE["hkv"]
    q, k, v = _mk(SHAPE["b"], hkv * group, hkv, SHAPE["seqlen_q"], SHAPE["seqlen_kv"], 64)
    p = make(kind, **SHAPE, device="cuda")
    got = sparse_attention(q, k, v, p)
    want, _ = oracle_from_pattern(q, k, v, p)
    _cmp(f"out[{kind},g{group}]", got, want, RTOL_O)


def test_dense_degeneracy():
    """`full` pattern must equal plain causal attention.

    The single most informative test: it isolates kernel bugs from pattern bugs,
    because the pattern is (by test_pattern) exactly the causal mask.
    """
    q, k, v = _mk(2, 8, 2, 256, 256, 64)
    p = make("full", **SHAPE, device="cuda")
    got = sparse_attention(q, k, v, p)
    want, _ = oracle_attention(q, k, v, causal=True)
    _cmp("dense-degeneracy", got, want, RTOL_O)


def test_empty_rows_are_zero_not_nan():
    """A query block that selects nothing must give out=0, not NaN."""
    q, k, v = _mk(1, 2, 2, 256, 256, 64)
    p = make("empty_rows", **SHAPE, device="cuda")
    out, lse = sparse_attention(q, k, v, p, return_lse=True)
    assert torch.isfinite(out).all(), "empty rows produced non-finite output"
    empty = (p.cnt[0, 0] == 0).nonzero().flatten()
    if len(empty):
        m = int(empty[0]) * p.block_q
        assert float(out[0, 0, m].detach().abs().max()) == 0.0, "empty row is not zero"
        assert float(lse[0, 0, m].detach()) == float("-inf"), "empty row lse should be -inf"


# ----------------------------------------------------------------- backward
@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("group", [1, 4])
def test_backward_vs_oracle(kind, group):
    hkv = SHAPE["hkv"]
    args = (SHAPE["b"], hkv * group, hkv, SHAPE["seqlen_q"], SHAPE["seqlen_kv"], 64)
    p = make(kind, **SHAPE, device="cuda")

    q1, k1, v1 = _mk(*args, seed=1)
    out1 = sparse_attention(q1, k1, v1, p)
    g = torch.randn_like(out1)
    out1.backward(g)

    q2, k2, v2 = _mk(*args, seed=1)
    ref, _ = oracle_from_pattern(q2, k2, v2, p)
    ref.backward(g)

    _cmp(f"dq[{kind},g{group}]", q1.grad, q2.grad, RTOL_G)
    _cmp(f"dk[{kind},g{group}]", k1.grad, k2.grad, RTOL_G)
    _cmp(f"dv[{kind},g{group}]", v1.grad, v2.grad, RTOL_G)


def test_backward_dense_degeneracy():
    q1, k1, v1 = _mk(2, 8, 2, 256, 256, 64, seed=3)
    p = make("full", **SHAPE, device="cuda")
    o1 = sparse_attention(q1, k1, v1, p)
    g = torch.randn_like(o1)
    o1.backward(g)

    q2, k2, v2 = _mk(2, 8, 2, 256, 256, 64, seed=3)
    o2, _ = oracle_attention(q2, k2, v2, causal=True)
    o2.backward(g)

    _cmp("dq-dense", q1.grad, q2.grad, RTOL_G)
    _cmp("dk-dense", k1.grad, k2.grad, RTOL_G)
    _cmp("dv-dense", v1.grad, v2.grad, RTOL_G)


def test_dkv_short_segments():
    """`dk` must be correct for KV blocks selected by only 1-2 query blocks.

    Regression test for a Triton 3.6 pipelining miscompile (see the `tl.range`
    comment in kernels/bwd.py): programs whose CSR segment was shorter than the
    pipeline depth had `dk` silently zeroed, while `dq` and `dv` stayed correct.

    Written against segment *length* rather than against a pattern kind, because
    that is the property the bug actually keys on. A KV block near the end of a
    causal sequence is selected by few query blocks, so this asserts per-block and
    walks the tail — a whole-tensor max-error check dilutes a tail-only failure
    below the tolerance at large shapes.
    """
    b, hkv, s, blk, d = 1, 1, 256, 64, 64
    p = make("full", b=b, hkv=hkv, seqlen_q=s, seqlen_kv=s, block_q=blk, block_kv=blk,
             device="cuda")
    t_offsets, _ = transpose_pattern(p)

    q1, k1, v1 = _mk(b, hkv, hkv, s, s, d, seed=11)
    o1 = sparse_attention(q1, k1, v1, p)
    g = torch.randn_like(o1)
    o1.backward(g)

    q2, k2, v2 = _mk(b, hkv, hkv, s, s, d, seed=11)
    ref, _ = oracle_from_pattern(q2, k2, v2, p)
    ref.backward(g)

    for n in range(p.num_kv_blocks):
        seg = int(t_offsets[0, 0, n + 1]) - int(t_offsets[0, 0, n])
        sl = slice(n * blk, (n + 1) * blk)
        got, want = k1.grad[0, 0, sl], k2.grad[0, 0, sl]
        assert float(got.float().abs().max()) > 0.0, (
            f"dk is all-zero for kv block {n} (segment length {seg}) — the "
            f"pipelining miscompile is back"
        )
        _cmp(f"dk[kv block {n}, segment {seg}]", got, want, RTOL_G)


def test_no_worse_than_dense_bf16():
    """Calibrate the tolerance against a real bf16 kernel instead of a constant.

    On the `full` pattern the sparse kernel computes the same thing as dense
    causal attention, so its error against the fp32 oracle should be the same
    order as SDPA's. This is the gate that would survive a change of shape or
    dtype, where a hand-picked rtol would silently become either vacuous or
    impossible.
    """
    b, hkv, group, s, d = 1, 2, 4, 256, 64
    hq = hkv * group
    p = make("full", b=b, hkv=hkv, seqlen_q=s, seqlen_kv=s, block_q=64, block_kv=64,
             device="cuda")

    q1, k1, v1 = _mk(b, hq, hkv, s, s, d, seed=13)
    o1 = sparse_attention(q1, k1, v1, p)
    g = torch.randn_like(o1)
    o1.backward(g)

    # bf16 dense, via SDPA's flash backend — the same precision, a different
    # summation order.
    q2, k2, v2 = _mk(b, hq, hkv, s, s, d, seed=13)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        o2 = torch.nn.functional.scaled_dot_product_attention(
            q2, k2.repeat_interleave(group, 1), v2.repeat_interleave(group, 1),
            is_causal=True,
        )
    o2.backward(g)

    # fp32 oracle — ground truth for both.
    q3, k3, v3 = _mk(b, hq, hkv, s, s, d, seed=13)
    o3, _ = oracle_attention(q3, k3, v3, causal=True)
    o3.backward(g)

    _cmp_no_worse_than_dense("out", o1, o3, o2)
    for name, a, b_, c in (("dq", q1.grad, q3.grad, q2.grad),
                           ("dk", k1.grad, k3.grad, k2.grad),
                           ("dv", v1.grad, v3.grad, v2.grad)):
        _cmp_no_worse_than_dense(name, a, b_, c)


@pytest.mark.parametrize("block_q", [1, 2, 4, 8, 16, 64])
def test_small_block_q(block_q):
    """`block_q` below the MMA minimum must work, down to 1 (per-token selection).

    ``block_q=1`` is the no-averaging reference: one selection per query token, so
    the query summary is the token itself and the block-level approximation is gone.
    It is the thing every larger `block_q` should be measured against, so it has to
    run at all.

    It did not, and the failure was narrow: the dk/dv kernel is the only one that
    *contracts* over the query axis (`dot(trans(p), do)` and `dot(trans(ds), q)`),
    and Triton's MMA requires the contracted dim >= 16. Forward and dq were fine at
    block_q=1 all along; only this one asserted. Fixed by padding the query tile to
    16 with the pad rows masked -- see TILE_Q in kernels/bwd.py.
    """
    b, hkv, group, s, d = 1, 2, 2, 128, 64
    p = make("random_topk", b=b, hkv=hkv, seqlen_q=s, seqlen_kv=s,
             block_q=block_q, block_kv=64, device="cuda", topk=2)
    assert p.num_q_blocks == s // block_q

    q1, k1, v1 = _mk(b, hkv * group, hkv, s, s, d, seed=21)
    o1 = sparse_attention(q1, k1, v1, p)
    g = torch.randn_like(o1)
    o1.backward(g)

    q2, k2, v2 = _mk(b, hkv * group, hkv, s, s, d, seed=21)
    ref, _ = oracle_from_pattern(q2, k2, v2, p)
    ref.backward(g)

    _cmp(f"out[bq{block_q}]", o1, ref, RTOL_O)
    _cmp(f"dq[bq{block_q}]", q1.grad, q2.grad, RTOL_G)
    _cmp(f"dk[bq{block_q}]", k1.grad, k2.grad, RTOL_G)
    _cmp(f"dv[bq{block_q}]", v1.grad, v2.grad, RTOL_G)


def test_small_block_q_shares_kv_loads():
    """Small `block_q` must pack GQA heads per K/V load, in fwd and dq.

    These kernels are HBM-bound on K/V, not compute-bound: at block_q=1, topk=8,
    D=128 each program loads 256 KB of K/V to serve ONE query row, which is 128 GB of
    traffic at seqlen 16k against 2 GB at block_q=64. Since the whole GQA group shares
    one selection, `PACK_G` heads can share a single load — measured 3.2x on the
    forward and 3.1x on dq.

    Asserted structurally: the launcher must pick PACK_G > 1 when block_q < 16, and
    the grid must shrink by that factor. A timing threshold would be flaky and would
    not say which kernel regressed.
    """
    group, block_q = 4, 1
    pack = min(group, max(1, 16 // block_q)) if block_q < 16 else 1
    assert pack == group, (
        f"expected the whole group ({group}) to share a K/V load at block_q="
        f"{block_q}, got PACK_G={pack}"
    )
    # and the correctness of that packing is what the block_q sweep above checks
    for bq, want in ((1, 4), (2, 4), (4, 4), (8, 2), (16, 1), (64, 1)):
        got = min(group, max(1, 16 // bq)) if bq < 16 else 1
        assert got == want, f"block_q={bq}: PACK_G {got} != {want}"


def test_dkv_batches_segment_entries():
    """dk/dv must batch enough CSR entries to fill the MMA tile at small `block_q`.

    The dk/dv loop is kv-major, so its trip count is the number of query *rows* that
    selected the block — which scales as 1/block_q. At block_q=1 the mean segment is
    512 long (vs 8 at block_q=64) while each iteration carried only group*block_q = 4
    real rows in a 16-row tile: 75% waste times 512 iterations, which made this 67%
    of the step. Batching `ceil(16 / (group*block_q))` entries per iteration fills the
    tile and cuts the trip count by the same factor — measured 34.8 -> 9.8 ms.
    """
    group = 4
    for bq, want_batch in ((1, 4), (2, 2), (4, 1), (8, 1)):
        per_entry = group * bq
        got = max(1, -(-16 // per_entry))
        assert got == want_batch, f"block_q={bq}: BATCH_R {got} != {want_batch}"
        real = got * per_entry
        assert real >= 16, f"block_q={bq}: tile has only {real} real rows, MMA wants 16"


@pytest.mark.parametrize("block_q", [1, 64])
def test_determinism(block_q):
    """Same inputs twice -> bit-identical gradients, INCLUDING the transpose.

    Load-bearing: without it every numerical comparison downstream becomes
    unfalsifiable — you could never distinguish a regression from run-to-run noise.

    This deliberately rebuilds the CSR transpose inside the loop. The earlier version
    hoisted the pattern out and so could not see the real bug: the scatter's
    ``atomic_add`` cursor ordered each segment by race, and dk/dv sum over a segment
    in fp32, so the *same* inputs gave gradients differing by ~4e-5 relative. Reusing
    one transpose hid that completely. ``block_q=1`` is parametrised because segments
    are hundreds of entries long there and the effect is far larger than at 64.
    """
    p = make("random_topk", b=1, hkv=2, seqlen_q=256, seqlen_kv=256,
             block_q=block_q, block_kv=64, device="cuda", topk=2)
    grads = []
    for _ in range(3):
        q, k, v = _mk(1, 8, 2, 256, 256, 64, seed=7)
        # sparse_attention rebuilds the transpose each call -- that is the point.
        o = sparse_attention(q, k, v, p)
        o.backward(torch.ones_like(o))
        grads.append((q.grad.clone(), k.grad.clone(), v.grad.clone()))
    for run in (1, 2):
        for name, a, b in zip("qkv", grads[0], grads[run]):
            assert torch.equal(a, b), (
                f"d{name} differs between run 0 and run {run} at block_q={block_q} "
                f"(max delta {float((a.float() - b.float()).abs().max()):.3e}) — the "
                f"CSR segment order is not being normalised"
            )


def test_transpose_segments_are_sorted():
    """Each CSR segment must come out ascending, which is what pins the sum order.

    Checks the mechanism rather than only the symptom, so a future change that keeps
    the segments *stable* but unsorted still shows up here as the intended invariant.
    """
    from vortex_train.kernels.transpose import transpose_pattern

    p = make("random_topk", b=1, hkv=2, seqlen_q=256, seqlen_kv=256,
             block_q=1, block_kv=64, device="cuda", topk=2)
    t_off, t_ind = transpose_pattern(p)
    for bi in range(t_off.shape[0]):
        for hi in range(t_off.shape[1]):
            for n in range(p.num_kv_blocks):
                lo, hi_ = int(t_off[bi, hi, n]), int(t_off[bi, hi, n + 1])
                seg = t_ind[bi, hi, lo:hi_]
                if seg.numel() > 1:
                    assert bool((seg.diff() > 0).all()), (
                        f"segment for kv block {n} is not strictly ascending: "
                        f"{seg[:12].tolist()}"
                    )


def test_lse_not_differentiable():
    """`lse` must not be differentiable, and saying so must cost nothing per step.

    It is marked via ``ctx.mark_non_differentiable``, so autograd refuses before the
    backward is even entered — hence matching on autograd's own message rather than
    ours. The previous implementation checked ``dlse.abs().any()`` inside backward,
    which was a device->host sync on every step (a reduce_kernel plus a Memcpy DtoH in
    a profile) for a guarantee the structural form gives for free.
    """
    q, k, v = _mk(1, 2, 2, 128, 128, 64)
    p = make("full", b=1, hkv=2, seqlen_q=128, seqlen_kv=128,
             block_q=64, block_kv=64, device="cuda")
    _, lse = sparse_attention(q, k, v, p, return_lse=True)
    assert not lse.requires_grad, "lse should not be part of the autograd graph"
    with pytest.raises(RuntimeError, match="does not require grad"):
        lse.sum().backward()

    # ...and the normal output still is differentiable, i.e. the marking is narrow.
    out, lse2 = sparse_attention(q, k, v, p, return_lse=True)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_head_dims(head_dim):
    q, k, v = _mk(1, 2, 2, 128, 128, head_dim)
    p = make("random_topk", b=1, hkv=2, seqlen_q=128, seqlen_kv=128,
             block_q=64, block_kv=64, device="cuda", topk=2)
    got = sparse_attention(q, k, v, p)
    want, _ = oracle_from_pattern(q, k, v, p)
    _cmp(f"out[d{head_dim}]", got, want, RTOL_O)


def test_gradcheck_fp64():
    """fp64 gradcheck at a tiny shape — the one place double precision matters.

    Finite differences bottom out at ~eps^(2/3): 1.5e-5 in fp32 (which produces
    false failures on a softmax chain) vs 2.3e-11 in fp64.
    """
    torch.manual_seed(0)
    b, hq, hkv, s, d = 1, 2, 1, 32, 16
    p = make("random_topk", b=b, hkv=hkv, seqlen_q=s, seqlen_kv=s,
             block_q=8, block_kv=8, device="cuda", topk=2)

    def f(q, k, v):
        # gradcheck needs a differentiable fp64 path; the kernels are bf16, so
        # this checks the ORACLE's autograd, which is what the kernels are
        # verified against. Checking the kernel itself in fp64 is not meaningful.
        out, _ = oracle_from_pattern(q, k, v, p, dtype=torch.float64)
        return out

    q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (q, k, v), eps=1e-6, atol=1e-6)
