"""learned_query — launcher emitter (Schedule.S, backend-agnostic).

``LearnedQuery`` bakes the trained per-layer compressor weights into the
compiled function as *producer-less constants*: the weights live on the op
instance (``Wq``/``Wk``/``layer_lookup``), and the generated launcher
reaches them back through ``ctx.op_list[<global_op_id>]`` — the same trick
``Conv1d`` uses for its ``weight``. Runtime per-layer selection reads the
EXPLICIT ``cur_layer`` argument threaded into every generated subgraph
``_impl`` from ``CompiledFunc.forward(..., cur_layer=...)`` — NOT a ctx
field and NOT an internal counter. The MLA backend passes
``cur_layer=layer.layer_id`` on each ``forward`` call.

The op is a per-request transform ``q[B, H, d] -> V[B, 1, d]`` (both
``BATCHED``), so the launcher operates on the full tensors and writes only
the live ``[:bs]`` rows of the output buffer.
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...learned_query import LearnedQuery


def generate_learned_query_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    input_tensor_id = graph.op_to_input_tensor_list[op_id][0]
    output_tensor_id = graph.op_to_output_tensor_list[op_id][0]
    t_i = graph.tensor_list[input_tensor_id]
    t_o = graph.tensor_list[output_tensor_id]
    op = graph.op_list[op_id]

    assert issubclass(op.__class__, LearnedQuery), (
        f"Expected a LearnedQuery op, got {op}"
    )
    assert t_i._format == FORMAT.BATCHED, (
        f"generate_learned_query_impl: input must be BATCHED, got {t_i._format}"
    )
    assert t_o._format == FORMAT.BATCHED, (
        f"generate_learned_query_impl: output must be BATCHED, got {t_o._format}"
    )

    # Index of this op in the *global* op list — the live op (with its baked
    # constants) is reachable at runtime via ``ctx.op_list[global_op_id]``.
    global_op_id = ctx.op_list.index(op)

    impl_lines = [
        f"{INDENT}# Learned bilinear query transform with per-layer baked weights.",
        f"{INDENT}# The query buffer carries one row per (batch, kv_head); for MLA",
        f"{INDENT}# num_kv_heads == 1 so the live rows are [:bs].",
        f"{INDENT}_lq_op = ctx.op_list[{global_op_id}]",
        f"{INDENT}_lq_bs = ctx.metadata.batch_size * ctx.num_kv_heads",
        f"{INDENT}# ``cur_layer`` is the explicit forward()-threaded argument.",
        f"{INDENT}_lq_V = _lq_op.compute_V("
        f"tensor_{input_tensor_id}[:_lq_bs], cur_layer)",
        f"{INDENT}tensor_{output_tensor_id}[:_lq_bs].copy_(_lq_V)",
    ]
    return "\n".join(impl_lines)
