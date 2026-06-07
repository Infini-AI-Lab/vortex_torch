"""learned_descriptor — Schedule.S cache launcher emitter (backend-agnostic).

Cache mirror of ``indexer.compiler.custom_impl.gemm_param``. The
:class:`~vortex_torch.cache.LearnedDescriptor` op bakes its per-layer
``Wt`` / ``Wk`` :class:`~vortex_torch.abs.Parameter` weights on the op
instance (reached at runtime via ``ctx.op_list[<global_op_id>]``); this
launcher gathers the active layer's slice (explicit ``cur_layer`` arg) and
runs a ``torch.einsum`` over the just-written blocks (``op.compute_g``), so
the weight never enters the fused per-block cache kernel.
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...learned_descriptor import LearnedDescriptor


def generate_learned_descriptor_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]

    assert issubclass(op.__class__, LearnedDescriptor), (
        f"Expected a LearnedDescriptor op, got {op}"
    )
    assert t_i._format == FORMAT.PAGED, (
        f"generate_learned_descriptor_impl: latent must be PAGED, got {t_i._format}"
    )
    assert t_o._format in (FORMAT.PAGED, FORMAT.RAGGED), (
        f"generate_learned_descriptor_impl: descriptors must be PAGED/RAGGED, "
        f"got {t_o._format}"
    )

    global_op_id = ctx.op_list.index(op)
    return "\n".join([
        f"{INDENT}# LearnedDescriptor: per-layer factorized block compressor",
        f"{INDENT}# (baked Wt/Wk Parameters) -> torch.einsum, scattered by block.",
        f"{INDENT}# ``cur_layer`` is the explicit forward()-threaded argument.",
        f"{INDENT}_ld_op = ctx.op_list[{global_op_id}]",
        f"{INDENT}_ld_op.compute_g(",
        f"{INDENT*2}tensor_{input_tensor_id},",
        f"{INDENT*2}tensor_{output_tensor_id},",
        f"{INDENT*2}loc,",
        f"{INDENT*2}ctx.block_size,",
        f"{INDENT*2}ctx.page_size,",
        f"{INDENT*2}ctx.num_blocks_per_page,",
        f"{INDENT*2}cur_layer,",
        f"{INDENT})",
    ])
