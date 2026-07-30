"""Schedule.S topk / approxTopK / block-table TopK / Union codegen.

Backend-agnostic launcher emitters. The underlying kernel is resolved at
runtime via :func:`vortex_torch.custom_ops.find` — these generators just
emit the Python call site against the resolved callable. They have no
dependency on the Schedule.W fused kernel (Triton-emitted or CUDA-emitted),
so they live under ``custom_impl`` and are dispatched from
``indexer/compiler/impl.py`` regardless of which backend handles W.
"""
import os

from ..graph import Graph
from ...context import Context
from ....abs import FORMAT
from ....utils import INDENT
from ..backend import get_backend


def _check_topk_io(graph: Graph, op_id: int) -> tuple[int, int]:
    input_tensor_id  = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    assert t_i._format == FORMAT.RAGGED, f"Expected ragged input tensor for topk, got {t_i._format}"
    assert t_o._format == FORMAT.RAGGED, f"Expected ragged output tensor for topk, got {t_o._format}"
    return input_tensor_id, output_tensor_id


def generate_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    # Pick the best-known top-k kernel at runtime via
    # ``custom_ops.find('topk_output', <backend>, max_topk_val=...)``.
    # The dispatch key is
    # ``(max_topk_val, vortex_attention_backend)`` — neither is known at
    # codegen time, so we emit a tiny per-module memo and look up on the
    # hot path.
    #
    # The two backends use different C ABIs (see ``custom_ops/topk_output``):
    #   flashinfer: (x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
    #                sparse_kv_indices, eff_bs, bos, eos, max_num_pages)
    #   trtllm    : (x, dense_seqlens, sparse_seqlens, dense_block_tables,
    #                sparse_block_tables, eff_bs, bos, eos,
    #                max_blocks_per_seq, block_size)  ← indptr-free
    # All backend-specific tensor/scalar choices come from
    # :mod:`indexer.compiler.backend`. The runtime kernel resolution goes
    # through :func:`vortex_torch.custom_ops.find` — bucket selection
    # picks the right ``k_<N>`` / ``default`` leaf for the JIT-compiled
    # extension.
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        "_vortex_topk_kernel_cache = {}",
        "def _vortex_topk_kernel_for(max_topk, attn_backend):",
        "    key = (max_topk, attn_backend)",
        "    fn = _vortex_topk_kernel_cache.get(key)",
        "    if fn is None:",
        "        fn = _vortex_topk_kernel_cache[key] = _vortex_custom_ops_find(",
        "            'topk_output', attn_backend, max_topk_val=max_topk)()",
        "    return fn",
    ])
    input_tensor_id, output_tensor_id = _check_topk_io(graph, op_id)

    bk = get_backend(ctx)
    if os.environ.get("VORTEX_USE_TORCH_TOPK", "0") == "1":
        assert bk.name == "trtllm", (
            "The vectorized torch.topk ablation currently requires the "
            "TensorRT-LLM block-table layout."
        )
        ctx.compilation_header_lines.append("import torch")
        # Native-API baseline: operate directly on the padded TensorRT-LLM
        # block table with stock PyTorch tensor ops. The planner has already
        # populated BOS/EOS and short-row fallbacks; only eligible long rows
        # have their middle slots replaced.
        return "\n".join([
            "_eff_bs = ctx.metadata.batch_size * ctx.num_kv_heads",
            "_stride = ctx.metadata.dense_block_tables.shape[1]",
            f"_scores = tensor_{input_tensor_id}.reshape(-1)"
            "[:_eff_bs * _stride].view(_eff_bs, _stride).float()",
            "_dense_tokens = ctx.metadata.dense_seqlens[:_eff_bs]",
            "_dense_blocks = torch.div(",
            f"{INDENT}_dense_tokens + ctx.block_size - 1, ctx.block_size,",
            f"{INDENT}rounding_mode='floor')",
            "_candidate = torch.clamp(",
            f"{INDENT}_dense_blocks - ctx.block_reserved_bos - ctx.block_reserved_eos,",
            f"{INDENT}min=0)",
            "_pos = torch.arange(_stride, device=_scores.device)[None, :]",
            "_valid = ((_pos >= ctx.block_reserved_bos) &",
            f"{INDENT * 2}(_pos < (_dense_blocks - ctx.block_reserved_eos)[:, None]))",
            "_masked_scores = _scores.masked_fill(~_valid, float('-inf'))",
            f"_, _top_pos = torch.topk(_masked_scores, k={ctx.topk_val}, dim=1)",
            "_picked = torch.gather(",
            f"{INDENT}ctx.metadata.dense_block_tables[:_eff_bs], 1, _top_pos)",
            f"_dst = tensor_{output_tensor_id}[:_eff_bs, "
            f"ctx.block_reserved_bos:ctx.block_reserved_bos + {ctx.topk_val}]",
            f"_eligible = (_candidate >= {ctx.topk_val})[:, None]",
            "_dst.copy_(torch.where(_eligible, _picked, _dst))",
        ])

    trailing = "".join(
        f"{INDENT * 2}{arg},\n" for arg in bk.topk_trailing_args
    )

    impl_lines = [
        f"_vortex_topk_kernel_for(getattr(ctx, 'max_topk_val', None), "
        f"getattr(ctx, 'vortex_attention_backend', 'flashinfer'))(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}{bk.topk_per_row_args[0]},",
        f"{INDENT * 2}{bk.topk_per_row_args[1]},",
        f"{INDENT * 2}{bk.topk_dense_src},",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.metadata.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{trailing})",
    ]
    return "\n".join(impl_lines)


def generate_block_table_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Codegen for the ``TopK(k)`` op — trtllm-only, multi-output
    (block_table + seqlens). Both outputs are auto-allocated intermediate
    buffers by ``interface.generate_entry_point``'s memory_initialization
    (block_table = RAGGED int32 of leading dim ``ctx.max_num_blocks``,
    seqlens = BATCHED int32 of leading dim ``eff_bs``). The C ABI lives
    in ``custom_ops/topk/trtllm/_abi.py`` and the underlying CUDA kernel
    in ``custom_ops/topk/trtllm/default/kernel.cu``.
    """
    bk = get_backend(ctx)
    assert bk.name == "trtllm", (
        f"TopK(k) codegen requires the trtllm backend; got {bk.name!r}. "
        f"This is enforced at op.profile() too — should not happen."
    )

    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_ids = graph.op_to_output_tensor_list[op_id]
    assert len(output_tensor_ids) == 2, (
        f"TopK(k) codegen: expected 2 outputs (block_table, seqlens), "
        f"got {len(output_tensor_ids)}"
    )
    block_table_id, seqlens_id = output_tensor_ids

    op = graph.op_list[op_id]
    k_val = int(getattr(op, "k", 0))
    assert k_val >= 1, f"TopK(k) codegen: invalid k={k_val} on op instance"

    # One callable per op id — same k always resolves to the same compiled
    # kernel (dispatcher caches by config), so multiple TopK ops in the
    # same flow share the binary even with different k.
    callable_name = f"_vortex_block_table_topk_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        f"{callable_name} = _vortex_custom_ops_find('topk', 'trtllm')()",
    ])

    # The C entry takes (x, dense_seqlens, sparse_seqlens, dense_block_tables,
    # sparse_block_tables, eff_bs, bos, eos, max_blocks_per_seq, block_size, k).
    # Inputs come from ctx.metadata (dense side, written by planner) +
    # ``tensor_<input_id>`` (the scores). Outputs are the two auto-allocated
    # intermediate buffers — written through their ``tensor_<id>`` handles.
    impl_lines = [
        f"{callable_name}(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}ctx.metadata.dense_seqlens,",
        f"{INDENT * 2}tensor_{seqlens_id},",
        f"{INDENT * 2}ctx.metadata.dense_block_tables,",
        f"{INDENT * 2}tensor_{block_table_id},",
        f"{INDENT * 2}ctx.metadata.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{INDENT * 2}ctx.metadata.dense_block_tables.shape[1],",
        f"{INDENT * 2}ctx.block_size,",
        f"{INDENT * 2}{k_val},",
        f")",
    ]
    return "\n".join(impl_lines)


def generate_union_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Codegen for ``Union()`` — merges two (block_table, seqlens) pairs
    and overwrites ``ctx.metadata.sparse_block_tables`` (= ``o``) and
    ``ctx.metadata.sparse_seqlens``. trtllm-only.

    Inputs (in order): ``bt_0, sl_0, bt_1, sl_1``. Single output is the
    framework-provided ``o``. The seqlens side-effect target
    (``ctx.metadata.sparse_seqlens``) is wired directly into the kernel
    call — it's not modeled as a graph output (mirrors how the regular
    ``topK`` writes ``sparse_kv_indices`` without registering it).
    """
    bk = get_backend(ctx)
    assert bk.name == "trtllm", (
        f"Union() codegen requires the trtllm backend; got {bk.name!r}. "
        f"(Op.profile() also enforces this.)"
    )

    input_ids = graph.op_to_input_tensor_list[op_id]
    assert len(input_ids) == 4, (
        f"Union() codegen: expected 4 inputs (bt_0, sl_0, bt_1, sl_1), "
        f"got {len(input_ids)}"
    )
    bt0_id, sl0_id, bt1_id, sl1_id = input_ids
    output_ids = graph.op_to_output_tensor_list[op_id]
    assert len(output_ids) == 1, (
        f"Union() codegen: expected 1 output (o), got {len(output_ids)}"
    )
    o_id = output_ids[0]

    callable_name = f"_vortex_union_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        f"{callable_name} = _vortex_custom_ops_find('union', 'trtllm')()",
    ])

    impl_lines = [
        f"{callable_name}(",
        f"{INDENT * 2}ctx.metadata.dense_seqlens,",
        f"{INDENT * 2}ctx.metadata.sparse_seqlens,",
        f"{INDENT * 2}ctx.metadata.dense_block_tables,",
        f"{INDENT * 2}tensor_{bt0_id},",
        f"{INDENT * 2}tensor_{sl0_id},",
        f"{INDENT * 2}tensor_{bt1_id},",
        f"{INDENT * 2}tensor_{sl1_id},",
        f"{INDENT * 2}tensor_{o_id},",
        f"{INDENT * 2}ctx.metadata.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.metadata.dense_block_tables.shape[1],",
        f"{INDENT * 2}ctx.block_size,",
        f")",
    ]
    return "\n".join(impl_lines)


def generate_approx_topk_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    """Codegen for `approxTopK(tolerate_ratio=...)` — the adaptive 8-bit
    radix variant. Reads `tolerate_ratio` off the op instance at codegen
    time and bakes it into a compile-time substitution for the
    JIT-built kernel under ``custom_ops/topk_output/<backend>/approx/``.

    Both backends ship an approx leaf, and each takes its own backend's
    C ABI (see ``generate_topk_impl`` above for the two signatures), so
    the launcher is emitted from :mod:`indexer.compiler.backend` exactly
    like the exact-topK path — including the full ``topk_trailing_args``
    tuple (flashinfer passes one, trtllm two).
    """
    input_tensor_id, output_tensor_id = _check_topk_io(graph, op_id)

    op = graph.op_list[op_id]
    # Defensive: profile() runs first and the registry already pinned
    # the op type via (approxTopK, Schedule.S). Still guard the attribute.
    tolerate_ratio = float(getattr(op, "tolerate_ratio", 0.0))

    # tolerate_ratio is a compile-time knob; each op-id pins its own
    # callable. The dispatcher de-dupes by (file, substitutions), so
    # two ops with the same tolerate_ratio share a compiled module.
    bk = get_backend(ctx)

    callable_name = f"_vortex_approx_topk_{op_id}"
    ctx.compilation_header_lines.extend([
        "from vortex_torch.custom_ops import find as _vortex_custom_ops_find",
        # The approx leaf accepts ``tolerate_ratio`` at dispatch time and
        # bakes it into ``__TOLERATE_RATIO__`` for this op-id's compiled
        # module (cache key includes substitutions, so two ops with the
        # same ratio share the binary).
        f"{callable_name} = _vortex_custom_ops_find("
        f"'topk_output', {bk.name!r}, approx=True)("
        f"tolerate_ratio={tolerate_ratio!r})",
    ])

    trailing = "".join(
        f"{INDENT * 2}{arg},\n" for arg in bk.topk_trailing_args
    )

    impl_lines = [
        f"{callable_name}(",
        f"{INDENT * 2}tensor_{input_tensor_id},",
        f"{INDENT * 2}{bk.topk_per_row_args[0]},",
        f"{INDENT * 2}{bk.topk_per_row_args[1]},",
        f"{INDENT * 2}{bk.topk_dense_src},",
        f"{INDENT * 2}tensor_{output_tensor_id},",
        f"{INDENT * 2}ctx.metadata.batch_size * ctx.num_kv_heads,",
        f"{INDENT * 2}ctx.block_reserved_bos,",
        f"{INDENT * 2}ctx.block_reserved_eos,",
        f"{trailing})",
    ]
    return "\n".join(impl_lines)
