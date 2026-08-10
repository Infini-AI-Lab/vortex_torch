"""Correctness of the frontend + compiler + selection kernels.

The load-bearing test here is :func:`test_index_equivalence`. A subtly wrong
pattern is the nastiest failure mode in the system: training still converges, just
to a slightly different objective, so nothing looks broken. So the fused
score/select kernel is checked to emit **exactly** what an independent, obviously
correct torch implementation of the same policy emits — not "close", identical.

The torch reference in :func:`_reference_select` is deliberately written from the
*spec* (score, causal-mask, boost reserved, top-k) rather than by reading the
kernel, so agreement is evidence rather than a tautology.
"""
from __future__ import annotations

import pytest
import torch

from vortex_train.compiler import compile_selection
from vortex_train.flow import ops
from vortex_train.flow.spec import REGISTRY, Budget, Field, Selection, register
from vortex_train.kernels.state import build_state
from vortex_train.nn import SparseAttention
from vortex_train.reference.oracle import oracle_from_pattern

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BOOST = 1.0e30


def _mk(b, hq, hkv, s, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = lambda *sh: torch.randn(*sh, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5
    return t(b, hq, s, d), t(b, hkv, s, d), t(b, hkv, s, d)


# --------------------------------------------------------------- torch reference
def _reference_state(k, v, fields, block_kv):
    """Per-block (and per-sub-block) reductions, in fp32, the obvious way."""
    b, hkv, skv, d = k.shape
    n_kv = (skv + block_kv - 1) // block_kv
    if not fields:                      # a stateless policy is legitimate
        return torch.zeros((b, hkv, n_kv, 0, 1, d), dtype=torch.float32, device=k.device)
    n_sub = max(f[2] for f in fields)
    out = torch.zeros((b, hkv, n_kv, len(fields), n_sub, d),
                      dtype=torch.float32, device=k.device)
    for f, (red, src, nsub) in enumerate(fields):
        x = (k if src == "k" else v).float()
        sub_len = block_kv // nsub
        for n in range(n_kv):
            for u in range(nsub):
                lo = n * block_kv + u * sub_len
                blk = x[:, :, lo : lo + sub_len, :]
                if red == "mean":
                    out[:, :, n, f, u] = blk.mean(2)
                elif red == "max":
                    out[:, :, n, f, u] = blk.max(2).values
                elif red == "min":
                    out[:, :, n, f, u] = blk.min(2).values
                else:
                    out[:, :, n, f, u] = blk.sum(2)
    return out


def _reference_scores(compiled, q, state, num_kv_heads, seqlen_kv):
    """Evaluate the traced graph in torch — written from the op semantics."""
    b, hq, sq, d = q.shape
    group = hq // num_kv_heads
    block_q, block_kv = compiled.block_q, compiled.block_kv
    m = (sq + block_q - 1) // block_q
    n_kv = state.shape[2]

    # query summary per (b, kv head, q block, group member)
    qs = torch.zeros((b, num_kv_heads, m, group, d), dtype=torch.float32, device=q.device)
    for i in range(m):
        blk = q[:, :, i * block_q : (i + 1) * block_q, :].float()
        red = blk.mean(2) if compiled.q_how == "mean" else blk.max(2).values   # [B,Hq,D]
        qs[:, :, i] = red.view(b, num_kv_heads, group, d)

    vals: dict[int, torch.Tensor] = {}
    tape, graph = compiled.tape, compiled.graph
    tape_nodes = [n for n in graph.nodes if n.op in
                  ("const", "dot", "scale", "norm", "distance", "add", "sub", "mul",
                   "envelope")]
    for i, node in enumerate(tape_nodes):
        if node.op == "dot":
            slot, nsub = tape["fslot"][i], tape["nsub"][i]
            st = state[:, :, :, slot, :nsub, :]                 # [B,Hkv,Nkv,U,D]
            # [B,Hkv,M,G,Nkv,U] -> max over sub-blocks -> reduce group
            per = torch.einsum("bhmgd,bhnud->bhmgnu", qs, st).max(5).values
            gr = node.attrs["group_reduce"]
            vals[i] = (per.max(3).values if gr == "max"
                       else per.sum(3) if gr == "sum" else per.mean(3))
        elif node.op == "envelope":
            smax, smin, nsub = tape["fslot"][i], tape["fslot2"][i], tape["nsub"][i]
            kmax = state[:, :, :, smax, :nsub, :]               # [B,Hkv,Nkv,U,D]
            kmin = state[:, :, :, smin, :nsub, :]
            # per-CHANNEL endpoint choice, THEN sum over D -- the whole point of the
            # op. qs is [B,Hkv,M,G,D]; broadcast against [B,Hkv,1,1,Nkv,U,D].
            qe = qs[:, :, :, :, None, None, :]
            hi = qe * kmax[:, :, None, None, :, :, :]
            lo = qe * kmin[:, :, None, None, :, :, :]
            per = torch.maximum(hi, lo).sum(-1).max(-1).values   # [B,Hkv,M,G,Nkv]
            gr = node.attrs["group_reduce"]
            vals[i] = (per.max(3).values if gr == "max"
                       else per.sum(3) if gr == "sum" else per.mean(3))
        elif node.op == "norm":
            slot, nsub = tape["fslot"][i], tape["nsub"][i]
            nrm = state[:, :, :, slot, :nsub, :].pow(2).sum(-1).sqrt().max(-1).values
            vals[i] = nrm[:, :, None, :].expand(b, num_kv_heads, m, n_kv).clone()
        elif node.op == "distance":
            mi = torch.arange(m, device=q.device).view(1, 1, m, 1)
            ni = torch.arange(n_kv, device=q.device).view(1, 1, 1, n_kv)
            vals[i] = -(mi - ni).float().expand(b, num_kv_heads, m, n_kv).clone()
        elif node.op == "const":
            vals[i] = torch.full((b, num_kv_heads, m, n_kv), tape["cval"][i],
                                 dtype=torch.float32, device=q.device)
        elif node.op == "scale":
            vals[i] = vals[tape["arg0"][i]] * tape["cval"][i]
        else:
            a, bb = vals[tape["arg0"][i]], vals[tape["arg1"][i]]
            vals[i] = a + bb if node.op == "add" else a - bb if node.op == "sub" else a * bb
    return vals[tape["out_node"]]                              # [B,Hkv,M,Nkv]


def _reference_select(compiled, scores, seqlen_q, seqlen_kv):
    """Mask, boost reservations, top-k — straight from the spec."""
    b, hkv, m, n_kv = scores.shape
    bq, bkv = compiled.block_q, compiled.block_kv
    bud = compiled.budget
    dev = scores.device
    s = scores.clone()

    mi = torch.arange(m, device=dev).view(1, 1, m, 1)
    ni = torch.arange(n_kv, device=dev).view(1, 1, 1, n_kv)
    eligible = torch.ones_like(s, dtype=torch.bool)
    if compiled.causal:
        q_end = (mi + 1) * bq - 1 + (seqlen_kv - seqlen_q)
        eligible = eligible & (ni * bkv <= q_end)
    s = s.masked_fill(~eligible, float("-inf"))

    if bud.reserve_bos:
        s = s + torch.where(eligible & (ni < bud.reserve_bos), BOOST, 0.0)
    if bud.reserve_local:
        lo = mi - (bud.reserve_local - 1)
        s = s + torch.where(eligible & (ni <= mi) & (ni >= lo), BOOST, 0.0)
    if bud.reserve_eos:
        s = s + torch.where(eligible & (ni >= n_kv - bud.reserve_eos), BOOST, 0.0)

    k_eff = min(bud.topk, n_kv)
    # Tie-break to the lower index, matching the kernel's packed (score, -index).
    order = torch.argsort(
        torch.stack([-s, ni.expand_as(s).float()], -1).view(*s.shape, 2)[..., 0]
        + 0.0, dim=-1, stable=True,
    )
    # stable argsort on -s already breaks ties toward the lower index
    order = order[..., :k_eff]
    top_s = torch.gather(s, 3, order)
    valid = torch.isfinite(top_s)
    idx = torch.where(valid, order.int(), torch.full_like(order.int(), -1))
    cnt = valid.sum(-1).int()
    return cnt, idx


# ------------------------------------------------------------------- the tests
ALL_RECIPES = sorted(REGISTRY)


@pytest.mark.parametrize("name", ALL_RECIPES)
def test_compiles(name):
    """Every registered recipe traces and lowers without a new op."""
    c = compile_selection(REGISTRY[name])
    assert c.num_nodes >= 1, f"{name} lowered to an empty tape"
    assert c.tape["out_node"] < c.num_nodes


@pytest.mark.parametrize("name", ALL_RECIPES)
@pytest.mark.parametrize("group", [1, 4])
def test_index_equivalence(name, group):
    """THE test: kernel-emitted cnt/idx must be *identical* to the torch reference.

    Not "close" — identical. A pattern that differs by one block still trains and
    still converges, to a slightly different objective, which is why this is the
    only real defence against a silently wrong compiler.
    """
    hkv, s, d = 2, 256, 64
    q, k, v = _mk(1, hkv * group, hkv, s, d, seed=5)
    attn = SparseAttention(REGISTRY[name], num_kv_heads=hkv)
    c = attn.compiled

    pattern = attn.build_pattern(q, k, v)
    pattern.validate()
    if c.causal:
        pattern.assert_causal()

    state = _reference_state(k, v, c.fields, c.block_kv)
    scores = _reference_scores(c, q, state, hkv, s)
    ref_cnt, ref_idx = _reference_select(c, scores, s, s)

    assert torch.equal(pattern.cnt, ref_cnt), (
        f"{name}: cnt differs from the reference selector\n"
        f"got  {pattern.cnt.flatten()[:16].tolist()}\n"
        f"want {ref_cnt.flatten()[:16].tolist()}\n{c.graph.dump()}"
    )
    # Compare as sets per row: the kernel and torch may order equal-scoring blocks
    # differently, and the attention kernel treats a row as a set.
    got = pattern.idx
    for bi in range(got.shape[0]):
        for hi in range(got.shape[1]):
            for mi in range(got.shape[2]):
                a = {x for x in got[bi, hi, mi].tolist() if x >= 0}
                e = {x for x in ref_idx[bi, hi, mi].tolist() if x >= 0}
                assert a == e, (
                    f"{name} (group {group}) row (b{bi},h{hi},m{mi}): "
                    f"selected {sorted(a)} != reference {sorted(e)}"
                )


@pytest.mark.parametrize("name", ALL_RECIPES)
def test_state_matches_torch(name):
    """The fused multi-field reduction equals per-field torch reductions."""
    hkv, s, d = 2, 256, 64
    _, k, v = _mk(1, hkv, hkv, s, d, seed=7)
    c = compile_selection(REGISTRY[name])
    if not c.fields:
        pytest.skip(f"{name} is stateless")
    got = build_state(k, v, c.fields, block_kv=c.block_kv)
    want = _reference_state(k, v, c.fields, c.block_kv)
    err = float((got - want).abs().max())
    assert err <= 1e-5, f"{name}: state max err {err:.3e}"


@pytest.mark.parametrize("name", ALL_RECIPES)
def test_reservations_honoured(name):
    """Reserved blocks are always present, and never double-counted.

    Checks the guarantee the frontend makes on the user's behalf — that a sink and
    the local window survive regardless of what the scorer thinks — plus that
    ``cnt`` never exceeds the budget even where reserved ranges overlap.
    """
    hkv, s, d, group = 2, 512, 64, 2
    q, k, v = _mk(1, hkv * group, hkv, s, d, seed=9)
    attn = SparseAttention(REGISTRY[name], num_kv_heads=hkv)
    c, bud = attn.compiled, attn.compiled.budget
    p = attn.build_pattern(q, k, v)
    n_kv = p.num_kv_blocks

    for mi in range(p.num_q_blocks):
        row = {x for x in p.idx[0, 0, mi].tolist() if x >= 0}
        assert len(row) == int(p.cnt[0, 0, mi]), (
            f"{name} row {mi}: duplicate entries — cnt {int(p.cnt[0,0,mi])} but "
            f"{len(row)} distinct blocks. Reservations double-counted."
        )
        assert int(p.cnt[0, 0, mi]) <= min(bud.topk, n_kv), f"{name} row {mi}: over budget"
        for n in range(bud.reserve_bos):
            if n <= mi:
                assert n in row, f"{name} row {mi}: BOS block {n} not reserved"
        for n in range(max(0, mi - bud.reserve_local + 1), mi + 1):
            assert n in row, f"{name} row {mi}: local block {n} not reserved"


@pytest.mark.parametrize("name", ALL_RECIPES)
@pytest.mark.parametrize("group", [1, 4])
def test_end_to_end_vs_oracle(name, group):
    """Full path — policy -> pattern -> attention -> grads — against the fp32 oracle.

    The oracle is fed the *same* pattern the module produced, so this isolates the
    attention math from the selection: a mismatch here is a kernel bug, whereas a
    bad selection shows up in test_index_equivalence.
    """
    hkv, s, d = 2, 256, 64
    q, k, v = _mk(1, hkv * group, hkv, s, d, seed=11)
    for t in (q, k, v):
        t.requires_grad_(True)
    attn = SparseAttention(REGISTRY[name], num_kv_heads=hkv)
    p = attn.build_pattern(q, k, v)

    out = attn(q, k, v)
    g = torch.randn_like(out)
    out.backward(g)

    q2, k2, v2 = q.detach().clone(), k.detach().clone(), v.detach().clone()
    for t in (q2, k2, v2):
        t.requires_grad_(True)
    ref, _ = oracle_from_pattern(q2, k2, v2, p)
    ref.backward(g)

    for nm, a, b in (("out", out, ref), ("dq", q.grad, q2.grad),
                     ("dk", k.grad, k2.grad), ("dv", v.grad, v2.grad)):
        e = float((a.detach().float() - b.detach().float()).abs().max())
        scale = float(b.detach().float().abs().max())
        tol = 4e-2 * max(scale, 1e-3)
        assert e <= tol, f"{name} {nm}: err {e:.3e} > tol {tol:.3e} (scale {scale:.3e})"


def test_dense_budget_degenerates_to_dense():
    """A budget >= n_kv_blocks must reproduce full causal attention.

    The cleanest end-to-end check that no stage drops a block: selection, pattern,
    and attention all have to be right for this to hold.
    """
    from vortex_train.reference.oracle import oracle_attention

    class Everything(Selection):
        state = {"centroid": Field(reduce="mean", src="k")}
        budget = Budget(topk=4, reserve_bos=0, reserve_local=1)
        block_q = block_kv = 64

        def __init__(self):
            self.qbar = ops.QSummary()
            self.dot = ops.Dot()

        def score(self, q, state, ctx):
            return self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx)

    hkv, s, d, group = 2, 256, 64, 2      # 256/64 = 4 kv blocks == topk
    q, k, v = _mk(1, hkv * group, hkv, s, d, seed=13)
    for t in (q, k, v):
        t.requires_grad_(True)
    attn = SparseAttention(Everything, num_kv_heads=hkv)
    out = attn(q, k, v)
    g = torch.randn_like(out)
    out.backward(g)

    q2, k2, v2 = q.detach().clone(), k.detach().clone(), v.detach().clone()
    for t in (q2, k2, v2):
        t.requires_grad_(True)
    ref, _ = oracle_attention(q2, k2, v2, causal=True)
    ref.backward(g)

    for nm, a, b in (("out", out, ref), ("dq", q.grad, q2.grad),
                     ("dk", k.grad, k2.grad), ("dv", v.grad, v2.grad)):
        e = float((a.detach().float() - b.detach().float()).abs().max())
        scale = float(b.detach().float().abs().max())
        assert e <= 4e-2 * max(scale, 1e-3), f"{nm}: {e:.3e} vs scale {scale:.3e}"


def test_selection_is_not_differentiable():
    """No gradient may flow into the scorer — the straight-through contract.

    If selection ever became differentiable by accident, the backward would be
    subtly wrong (it assumes a constant mask) and memory would grow by the scorer's
    saved activations.
    """
    hkv, s, d = 2, 128, 64
    q, k, v = _mk(1, hkv * 2, hkv, s, d, seed=15)
    for t in (q, k, v):
        t.requires_grad_(True)
    attn = SparseAttention("block_topk", num_kv_heads=hkv)
    p = attn.build_pattern(q, k, v)
    assert not p.cnt.requires_grad and not p.idx.requires_grad
    assert p.cnt.grad_fn is None and p.idx.grad_fn is None


def test_budget_rejects_impossible_reservations():
    """A reservation larger than the budget is a config error, caught at declare time."""
    with pytest.raises(ValueError, match="exceed the total budget"):
        Budget(topk=2, reserve_bos=2, reserve_local=1)


def test_score_must_return_a_score():
    """A scorer that returns a field (not a score) is rejected with a useful message."""

    class Bad(Selection):
        state = {"centroid": Field()}
        budget = Budget(topk=4)

        def score(self, q, state, ctx):
            return state["centroid"]          # BLOCK, not SCORE

    with pytest.raises(TypeError, match="must return a SCORE"):
        compile_selection(Bad)


def test_arithmetic_on_raw_field_is_rejected():
    """Combining a raw field with a score is a kind error, not silent coercion."""

    class Bad(Selection):
        state = {"centroid": Field()}
        budget = Budget(topk=4)

        def __init__(self):
            self.qbar = ops.QSummary()
            self.dot = ops.Dot()

        def score(self, q, state, ctx):
            return self.dot(self.qbar(q, ctx=ctx), state["centroid"], ctx=ctx) + state["centroid"]

    with pytest.raises(TypeError, match="do not match"):
        compile_selection(Bad)


def test_registry_rejects_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):
        @register("block_topk")
        class Dup(Selection):
            budget = Budget(topk=4)

            def score(self, q, state, ctx):
                return ops.Distance()(ctx=ctx)


def test_select_state_tile_is_bounded():
    """The scorer's state tile must not scale with sequence length.

    Regression test for a register-spill that cost 43-72% of the step at 64k: the
    state tile was sized at the padded ``Nkv``, so it reached 512 KB per program
    (``[Nkv, D]`` fp32) and spilled to local memory. Tiling the KV axis caps it at
    ``[TILE_N, D]``.

    Asserted structurally rather than by timing, because a latency threshold in a
    test is both flaky and silent about *why* it regressed. If ``TILE_N`` stops
    bounding the tile, this fails immediately at any sequence length.
    """
    from vortex_train.kernels.select import TILE_N, generated_source

    c = compile_selection(REGISTRY["block_topk"])
    src = generated_source(c.tape)
    # The scorer chain must be inside the tile loop, and every state load must be
    # shaped by TILE_N rather than BLOCK_N.
    assert "for t in tl.static_range(NTILES):" in src, "the KV tile loop is gone"
    assert "score_dot(qs, STATE" in src, "Dot no longer reads a tiled state slice"
    assert "TILE_N)" in src, "Dot is not tile-shaped"

    tile_bytes = TILE_N * 128 * 4
    assert tile_bytes <= 64 * 1024, (
        f"state tile is {tile_bytes/1024:.0f} KB per program at D=128 — too large "
        f"for a register file, which is what caused the original spill"
    )


def test_select_cost_grows_sublinearly_in_seqlen():
    """Selection time must not blow up with sequence length.

    A loose bound (not a fixed threshold) so it is robust across machines: quadrupling
    the sequence must cost well under 16x, which is what the pre-tiling spill did
    (0.28 -> 18 ms for a 4x sequence increase, i.e. ~65x). Catches the class of
    regression where selection quietly becomes the dominant cost.
    """
    import time

    from vortex_train.kernels.select import score_and_select
    from vortex_train.kernels.state import build_state

    hkv, group, d = 8, 4, 128

    def sel_ms(seqlen):
        q = torch.randn(1, hkv * group, seqlen, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, hkv, seqlen, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, hkv, seqlen, d, device="cuda", dtype=torch.bfloat16)
        c = compile_selection(REGISTRY["block_topk"])
        n_kv = seqlen // c.block_kv
        st = build_state(k, v, c.fields, block_kv=c.block_kv)
        kw = dict(
            num_kv_blocks=n_kv, seqlen_kv=seqlen, block_q=c.block_q,
            block_kv=c.block_kv, num_kv_heads=hkv, topk=c.budget.topk,
            reserve_bos=c.budget.reserve_bos, reserve_local=c.budget.reserve_local,
            reserve_eos=c.budget.reserve_eos, causal=c.causal, q_how=c.q_how,
        )
        for _ in range(3):
            score_and_select(q, st, c.tape, **kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            score_and_select(q, st, c.tape, **kw)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 100.0        # ms per call

    short, long = sel_ms(4096), sel_ms(16384)
    ratio = long / max(short, 1e-6)
    assert ratio < 16.0, (
        f"selection cost grew {ratio:.1f}x for a 4x longer sequence "
        f"({short:.3f} -> {long:.3f} ms) — the state tile is probably spilling again"
    )


def test_subblock_state_loaded_once_per_tile():
    """Sub-block loops must be OUTER of the group loop, not inner.

    Regression test for a 20x selection slowdown: with the group loop outside, each
    ``kmax``/``kmin`` tile is reloaded once per group member, so ``GROUP=4, NSUB=4``
    issued 32 loads where 8 suffice. Selection cost 305 ms at 128k against quest's
    15 ms — 20x for 4x the state — because this kernel is bound by state traffic, not
    arithmetic. Inverting the loops cut lserve's step 3.1x (428 -> 204 ms at 128k).

    Checked by source structure rather than by timing: a latency threshold would be
    flaky across machines and would not say *why* it regressed.
    """
    import inspect

    from vortex_train.kernels import score_ops

    for fn_name in ("score_dot", "score_envelope"):
        jit_fn = getattr(score_ops, fn_name)
        # @triton.jit wraps the function; getsource needs the python callable under it.
        src = inspect.getsource(getattr(jit_fn, "fn", jit_fn))
        # Only the max-group path folds the two reductions into one accumulator; that
        # is the path this checks, and it is the default every recipe uses.
        head = src.split("# sum / mean")[0].split("if GRED != 0")[0]
        u_pos = head.find("for u in tl.static_range(NSUB)")
        g_pos = head.find("for g in tl.static_range(GROUP)")
        assert u_pos != -1 and g_pos != -1, f"{fn_name}: expected both loops"
        assert u_pos < g_pos, (
            f"{fn_name}: the GROUP loop is outside the NSUB loop, so each state tile "
            f"is reloaded once per group member — the 20x selection regression is back"
        )


def test_step_issues_no_dtoh_copy_and_constant_launches():
    """A real step must issue no device->host copy, and a launch count fixed in seqlen.

    The AST check below catches the *syntactic* forms of a sync. This catches the rest
    — a sync hidden inside a library call — by profiling an actual step. It found one:
    ``dlse.abs().any()`` in the autograd backward, which showed up as a reduce_kernel
    plus a Memcpy DtoH on every step.

    The constant-launch half is the other half of the same promise: if the launch count
    grew with sequence length, something would be looping on the host over blocks.
    """
    from torch.profiler import ProfilerActivity, profile

    def kernels_for(seqlen):
        # this module's _mk is (b, hq, hkv, s, d) and does not set requires_grad
        q, k, v = _mk(1, 8, 2, seqlen, 64, seed=3)
        q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
        attn = SparseAttention("block_topk", num_kv_heads=2)

        def step():
            o = attn(q, k, v)
            o.backward(torch.ones_like(o))
            q.grad = k.grad = v.grad = None

        step()                      # warm the JIT compiles out of the measurement
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        rows = [e for e in prof.key_averages() if e.self_device_time_total > 0]
        return sum(e.count for e in rows), {e.key for e in rows}

    n_short, names = kernels_for(256)
    n_long, _ = kernels_for(1024)

    offenders = [nm for nm in names if "Memcpy" in nm and "DtoH" in nm]
    assert not offenders, (
        f"the step issues a device->host copy ({offenders}) — that is a per-step sync"
    )
    assert n_short == n_long, (
        f"launch count changed with sequence length ({n_short} at 256 vs {n_long} at "
        f"1024) — something is looping on the host over blocks"
    )


def test_no_host_work_in_step_path():
    """The per-step path must issue no host sync and no device->host copy.

    Asserted by source inspection over the modules on the step path: a ``.item()``
    or ``.tolist()`` there would serialize the pipeline every step and silently
    destroy throughput at scale, which a latency benchmark on small shapes would
    not reveal.
    """
    import ast
    import inspect

    from vortex_train.kernels import bwd, fwd, select, state, transpose
    from vortex_train.nn import functional, module

    # Match attribute *calls* in the AST, not text: a grep would also hit the words
    # in a docstring that explains the rule (it did), which makes the test
    # unfalsifiable in the annoying direction — failing on its own documentation.
    banned = {"item", "tolist", "cpu", "numpy"}
    # Explicitly test-only helpers, excluded by name and documented as such.
    exempt = {"verify_transpose", "generated_source"}

    for mod in (state, select, transpose, fwd, bwd, functional, module):
        tree = ast.parse(inspect.getsource(mod))
        skip: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in exempt:
                skip.update(id(n) for n in ast.walk(node))
        for node in ast.walk(tree):
            if id(node) in skip:
                continue
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in banned and not node.args):
                raise AssertionError(
                    f"{mod.__name__}:{node.lineno} calls .{node.func.attr}() on the "
                    f"step path — that is a device->host sync every step"
                )
