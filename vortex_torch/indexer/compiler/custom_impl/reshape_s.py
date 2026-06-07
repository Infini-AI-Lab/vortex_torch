"""reshape_s — Schedule.S launcher for ``Reshape`` under padded (non-pow2) dims.

The fused ``tl.reshape`` (``triton_impl/reshape.py``) needs pow2 inner dims and
an unpadded tile. When the real inner dims aren't pow2 the buffer is padded, and
a tile reshape would fold padding into real positions. This standalone launcher
sidesteps that entirely: it slices the **real** ``[:, :x1, :y1]`` region of the
padded buffer (torch, so "which dim is padding vs data" is just a slice),
``reshape``s it row-major to ``[:, :x2, :y2]`` and writes the real region of the
output buffer. Per-row independent → format-agnostic (BATCHED / RAGGED). The
``copy_``/``reshape`` are capturable, so it stays cuda-graph-safe.
"""
from ..graph import Graph
from ...context import Context
from ....utils import INDENT
from ....abs import FORMAT
from ...reshape import Reshape


def generate_reshape_s_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    in_id = graph.op_to_input_tensor_list[op_id][0]
    out_id = graph.op_to_output_tensor_list[op_id][0]
    op = graph.op_list[op_id]
    assert issubclass(op.__class__, Reshape), f"Expected a Reshape op, got {op}"

    t_x = graph.tensor_list[in_id]
    t_o = graph.tensor_list[out_id]
    x1, y1 = int(t_x.shape[1]), int(t_x.shape[2])     # real input inner dims
    x2, y2 = int(op.x2), int(op.y2)                   # real output inner dims

    # Operate on the REAL row count, not the full buffer (input/output buffers
    # can have different leading allocations). BATCHED → bs (the per-request row
    # count, as the other Schedule.S ops use); RAGGED → the common live prefix.
    if t_o._format == FORMAT.BATCHED:
        n = "ctx.metadata.batch_size * ctx.num_kv_heads"
    else:
        n = f"min(tensor_{in_id}.shape[0], tensor_{out_id}.shape[0])"
    return "\n".join([
        f"{INDENT}# Schedule.S reshape over the REAL (unpadded) inner region.",
        f"{INDENT}_rs_n = {n}",
        f"{INDENT}_rs_in = tensor_{in_id}[:_rs_n, :{x1}, :{y1}]",
        f"{INDENT}tensor_{out_id}[:_rs_n, :{x2}, :{y2}].copy_("
        f"_rs_in.reshape(_rs_n, {x2}, {y2}))",
    ])
