"""Torch-parity + plumbing test for the per-layer LearnedQuery op and the
explicit ``cur_layer`` compiled-function argument.

Run (GPU):  python algorithm_scientist/test_learned_query_parity.py

Checks
------
1. ``compute_V`` torch reference == compiled generated kernel output for a
   chosen ``cur_layer`` (per-layer weight gather), bf16 ~1e-3.
2. The explicit ``forward(..., cur_layer=L)`` argument selects different
   layer rows (L=0 vs L=1 give different V).
3. Identity fallback (untrained / -1 row) reproduces V = sum_h q_h, i.e. the
   plain head-summed centroid scorer.
4. Default ``cur_layer`` (forward called without it) == cur_layer=0.
"""
import tempfile
import uuid

import torch

import vortex_torch  # noqa: F401  (registers ops/flows)
from vortex_torch.abs import vTensor, FORMAT
from vortex_torch.utils import Mode
from vortex_torch.indexer import LearnedQuery
from vortex_torch.indexer.compiler.compile import compile as compile_indexer
from vortex_torch.flow.verify import _make_indexer_ctx, Config


def _build_cfg(B, H, d):
    return Config(
        B=B, G=H, D=d, num_kv_heads=1,
        block_size=16, page_size=16, num_blocks_per_page=1,
        num_pages_per_workload=1, workload_chunk_size=1,
    )


def _compile_learned_query(Wq, Wk, lookup, B, H, d, device, cache_dir):
    """Build a 1-op graph V = LearnedQuery(q) and compile it.

    Returns (compiled_func_instance, ctx, op, v_attr_name). The ctx is the
    compile-time ctx (holds ctx.op_list with the baked op + ctx.metadata);
    it must be passed to forward() so the generated launcher can reach the
    op's baked constants.
    """
    cfg = _build_cfg(B, H, d)
    name = f"lqtest_{uuid.uuid4().hex[:8]}"
    ctx = _make_indexer_ctx(
        cfg=cfg, max_num_pages_per_request=8, max_new_tokens_per_batch=64,
        cache_dir=cache_dir, sparse_attention_name=name,
        tensor_device=device,
    )
    ctx.vortex_dtype = torch.bfloat16
    ctx.query_arg_names = ["q"]
    ctx.mode = Mode.profile

    q = vTensor(shape=(B, H, d), dtype=torch.bfloat16, device=device,
                _format=FORMAT.BATCHED, tensor_id=0)
    ctx.tensor_list.append(q)
    ctx.output_tensor_to_op_list.append(None)
    ctx.tensor_id_to_tensor_name_map[0] = "q"

    # LearnedQuery's output becomes the terminal "o" (tensor_id 1) so DCE keeps
    # it and it is caller-provided (we pass our own V buffer named "o").
    op = LearnedQuery(Wq, Wk, lookup)
    V = op.profile(q, ctx)
    v_tid = V.tensor_id  # == 1 (next free id after q)
    assert v_tid == 1, f"expected V tid 1, got {v_tid}"
    ctx.tensor_id_to_tensor_name_map[v_tid] = "o"

    cls = compile_indexer(ctx)
    compiled = cls()
    return compiled, ctx, op, v_tid


def _run(compiled, ctx, q, B, d, *, pass_layer, cur_layer=0):
    # LearnedQuery writes V into the caller-provided "o" buffer (BATCHED
    # leading dim max_bs * num_kv_heads); read back the live [:B] rows.
    leading = ctx.max_bs * ctx.num_kv_heads
    o = torch.zeros(leading, 1, d, dtype=torch.bfloat16, device=q.device)
    if pass_layer:
        compiled.forward(q=q, o=o, cache={}, ctx=ctx, cur_layer=cur_layer)
    else:
        compiled.forward(q=q, o=o, cache={}, ctx=ctx)  # default cur_layer
    return o[:B].clone()


def _err(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    device = "cuda:0"
    torch.manual_seed(0)
    B, H, d, r = 2, 4, 32, 8
    Lfull = 3

    cache_dir = tempfile.mkdtemp(prefix="lqtest_")

    Wq = torch.randn(Lfull, H, d, r) * 0.1
    Wk = torch.randn(Lfull, H, d, r) * 0.1
    lookup = torch.tensor([0, 1, -1], dtype=torch.long)  # layer 2 -> identity

    q = torch.randn(B, H, d, device=device, dtype=torch.bfloat16)

    compiled, ctx, op, _ = _compile_learned_query(
        Wq, Wk, lookup, B, H, d, device, cache_dir
    )
    ctx.metadata.set_batch_size(B)

    tol = 2e-3  # bf16 tolerance
    fails = []

    # Test 1: parity, trained layer 0.
    V0 = _run(compiled, ctx, q, B, d, pass_layer=True, cur_layer=0)
    ref0 = op.compute_V(q, 0).squeeze(1)  # [B, d]
    e0 = _err(V0.squeeze(1), ref0)
    print(f"[T1] layer 0 parity  max|err| = {e0:.2e}  (tol {tol})")
    if e0 > tol:
        fails.append("T1 layer-0 parity")

    # Test 2: parity, trained layer 1 — different weights => different V.
    V1 = _run(compiled, ctx, q, B, d, pass_layer=True, cur_layer=1)
    ref1 = op.compute_V(q, 1).squeeze(1)
    e1 = _err(V1.squeeze(1), ref1)
    diff01 = _err(V0.squeeze(1), V1.squeeze(1))
    print(f"[T2] layer 1 parity  max|err| = {e1:.2e};  "
          f"|V0-V1| = {diff01:.2e} (should be >> 0)")
    if e1 > tol:
        fails.append("T2 layer-1 parity")
    if diff01 < 1e-2:
        fails.append("T2 layers not distinguished")

    # Test 3: identity fallback (layer 2 -> -1 row): V == sum_h q_h.
    # The op casts V back to bf16, so compare against the bf16-rounded sum
    # (a plain bf16 reduction of H terms; ~1e-2 is the dtype floor, not error).
    V2 = _run(compiled, ctx, q, B, d, pass_layer=True, cur_layer=2)
    ident_ref = q.float().sum(dim=1).to(torch.bfloat16).float()
    e2 = _err(V2.squeeze(1).to(torch.bfloat16).float(), ident_ref)
    print(f"[T3] identity fallback (layer 2)  max|err vs bf16 sum_h q_h| = {e2:.2e}")
    if e2 > tol:
        fails.append("T3 identity fallback")

    # Test 4: default cur_layer (omitted) == cur_layer=0.
    Vd = _run(compiled, ctx, q, B, d, pass_layer=False)
    ed = _err(Vd.squeeze(1), V0.squeeze(1))
    print(f"[T4] default-arg == cur_layer=0  max|err| = {ed:.2e}")
    if ed > 1e-4:
        fails.append("T4 default-arg path")

    # Test 5: identity-fallback flow == plain head-summed centroid scorer.
    # An all-identity LearnedQuery (no ckpt) must produce V = sum_h q_h for
    # every layer, matching RopeAwareBlockSparseMLA's head-summed query.
    comp_id, ctx_id, op_id, _ = _compile_learned_query(
        None, None, None, B, H, d, device, tempfile.mkdtemp(prefix="lqid_")
    )
    ctx_id.metadata.set_batch_size(B)
    Vid = _run(comp_id, ctx_id, q, B, d, pass_layer=True, cur_layer=7)  # any layer
    e5 = _err(Vid.squeeze(1).to(torch.bfloat16).float(),
              q.float().sum(dim=1).to(torch.bfloat16).float())
    print(f"[T5] no-ckpt identity flow == sum_h q_h  max|err| = {e5:.2e}")
    if e5 > tol:
        fails.append("T5 no-ckpt identity flow")

    print()
    if fails:
        print("FAILED:", ", ".join(fails))
        raise SystemExit(1)
    print("ALL PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
