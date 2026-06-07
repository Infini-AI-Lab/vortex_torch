"""factor_score — Schedule.S indexer launcher emitter (backend-agnostic).

The :class:`~vortex_torch.indexer.FactorScore` op bakes its per-layer ``Wq``
:class:`~vortex_torch.abs.Parameter` on the op instance (reached at runtime via
``ctx.op_list[<global_op_id>]``); this launcher gathers the active layer's slice
(explicit ``cur_layer`` arg) and runs a ``torch`` scoring pass over the trtllm
block tables (``op.compute_score``), emitting a RAGGED ``[S, 1, 1]`` score in
the exact ``bx * max_blocks_per_seq + col`` layout the ``topK`` kernel reads.

trtllm-only: the score layout is keyed on ``dense_block_tables`` /
``dense_seqlens`` (the indptr-free block-table planner), matching
``TopKOutput_Kernel``.
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...factor_score import FactorScore
from ..backend import get_backend


def generate_factor_score_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    q_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    desc_tensor_id = graph.op_to_input_tensor_list[op_id][1]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_q = graph.tensor_list[q_tensor_id]
    t_d = graph.tensor_list[desc_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]

    assert issubclass(op.__class__, FactorScore), f"Expected a FactorScore op, got {op}"
    assert t_q._format == FORMAT.BATCHED, (
        f"generate_factor_score_impl: q must be BATCHED, got {t_q._format}"
    )
    assert t_d._format == FORMAT.PAGED, (
        f"generate_factor_score_impl: descriptors must be PAGED, got {t_d._format}"
    )
    assert t_o._format == FORMAT.RAGGED, (
        f"generate_factor_score_impl: score must be RAGGED, got {t_o._format}"
    )

    bk = get_backend(ctx)
    assert bk.name == "trtllm", (
        "FactorScore requires the trtllm attention backend (block-table layout); "
        f"got backend {bk.name!r}. Set vortex_attention_backend='trtllm'."
    )

    global_op_id = ctx.op_list.index(op)
    return "\n".join([
        f"{INDENT}# FactorScore: per-layer learned factorized block scorer",
        f"{INDENT}# (baked Wq Parameter) -> torch scoring over block tables.",
        f"{INDENT}# ``cur_layer`` is the explicit forward()-threaded argument.",
        f"{INDENT}_fs_op = ctx.op_list[{global_op_id}]",
        f"{INDENT}_fs_op.compute_score(",
        f"{INDENT*2}tensor_{q_tensor_id},",
        f"{INDENT*2}tensor_{desc_tensor_id},",
        f"{INDENT*2}tensor_{output_tensor_id},",
        f"{INDENT*2}ctx.metadata.dense_seqlens,",
        f"{INDENT*2}ctx.metadata.dense_block_tables,",
        f"{INDENT*2}ctx.metadata.batch_size * ctx.num_kv_heads,",
        f"{INDENT*2}ctx.block_size,",
        f"{INDENT*2}ctx.block_reserved_bos,",
        f"{INDENT*2}ctx.block_reserved_eos,",
        f"{INDENT*2}cur_layer,",
        f"{INDENT})",
    ])
