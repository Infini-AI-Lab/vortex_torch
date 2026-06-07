"""gemm_param — Schedule.S launcher for ``GeMM`` with a ``Vortex.Parameter``.

When ``GeMM``'s ``y`` operand is a :class:`~vortex_torch.indexer.Parameter`
(``FORMAT.PARAMETER``, a batch-shared learned constant), the op is scheduled as
``Schedule.S`` and the big weight is baked on the op instance — reached at
runtime via ``ctx.op_list[<global_op_id>]`` (the same producer-less-constant
trick ``Conv1d`` uses). The launcher gathers the per-layer
slice (explicit ``cur_layer`` arg) and runs ``torch.matmul`` (``op.compute_param``),
so the weight never enters the fused per-workload kernel.

Per-request transform ``x[B, Nx, K] -> O[B, Ny, Nx]`` (both ``BATCHED``); writes
only the live ``[:bs]`` rows.
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...matmul import GeMM


def generate_gemm_param_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]

    assert issubclass(op.__class__, GeMM), f"Expected a GeMM op, got {op}"
    assert getattr(op, "_param", None) is not None, (
        "generate_gemm_param_impl: GeMM op has no Parameter operand"
    )
    assert t_i._format == FORMAT.BATCHED and t_o._format == FORMAT.BATCHED, (
        f"generate_gemm_param_impl: input/output must be BATCHED, "
        f"got {t_i._format}/{t_o._format}"
    )

    global_op_id = ctx.op_list.index(op)
    return "\n".join([
        f"{INDENT}# GeMM with a batch-shared Vortex.Parameter -> torch.matmul.",
        f"{INDENT}# The activation carries one row per (batch, kv_head); for MLA",
        f"{INDENT}# num_kv_heads == 1 so the live rows are [:bs].",
        f"{INDENT}_gp_op = ctx.op_list[{global_op_id}]",
        f"{INDENT}_gp_bs = ctx.metadata.batch_size * ctx.num_kv_heads",
        f"{INDENT}# ``cur_layer`` is the explicit forward()-threaded argument.",
        f"{INDENT}_gp_O = _gp_op.compute_param("
        f"tensor_{input_tensor_id}[:_gp_bs], cur_layer)",
        f"{INDENT}tensor_{output_tensor_id}[:_gp_bs].copy_(_gp_O)",
    ])
