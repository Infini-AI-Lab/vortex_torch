"""Dispatch a sub-graph to its impl codegen.

Two axes:

  * ``Schedule.W`` — the fused per-workload kernel. Codegen is
    backend-specific: ``triton_impl`` emits a ``@triton.jit`` kernel,
    ``cuda_impl`` emits a CUDA ``__global__`` kernel + host-side
    launcher embedded via ``torch.utils.cpp_extension.load_inline``.
    Picked by ``ctx.impl_backend``.

  * ``Schedule.S`` — standalone ops where the underlying kernel is
    resolved at runtime via :func:`vortex_torch.custom_ops.find`. The
    emitted launcher body is plain Python and has no coupling to the
    fused W kernel, so the same code is produced regardless of which
    backend handles ``Schedule.W``. The S codegens live under
    :mod:`custom_impl`.

This module owns the non-W wrapper template — the single op carries all
of its own codegen via :func:`custom_impl.get_impl_func`, and we stitch
the function signature and indent the op-impl body around it. The W path
is delegated to the chosen backend.
"""
from ...utils import Schedule, INDENT, indent_block

from .graph import Graph
from ..context import Context

from .triton_impl import generate_triton_impl
from .cuda_impl import generate_cuda_impl
from . import custom_impl


def _generate_non_w_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Direct op-impl wrapper for non-workload-scheduled (Schedule.S) subgraphs.

    The single op carries all the codegen via
    :func:`custom_impl.get_impl_func`; this layer just stitches the
    function signature and indents the op-impl body. Identical between
    the triton and cuda Schedule.W backends — neither is involved here.
    """
    assert len(sub_graph.op_list) == 1, (
        "Expected exactly one operation in non-workload-scheduled "
        "sub-graph for direct implementation."
    )

    custom_impl.register_headers(ctx)

    arg_list = [f"tensor_{tid}" for tid in sub_graph.input_tensor_ids]
    arg_list += [f"tensor_{tid}" for tid in sub_graph.output_tensor_ids]
    arg_list.append("ctx")
    args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)

    op_impl_str = indent_block(
        custom_impl.get_impl_func(sub_graph.op_list[0])(sub_graph, 0, ctx), 1,
    )

    impl_str = f"""
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def},
{INDENT}cur_layer=0,
):
{op_impl_str}
"""
    return impl_str.strip()


def _make_backend_dispatcher(w_impl_fn):
    """Wrap a Schedule.W backend codegen so the same callable also
    handles Schedule.S subgraphs by routing them through
    :func:`_generate_non_w_impl`.
    """

    def generate_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
        if sub_graph.schedule == Schedule.W:
            return w_impl_fn(sub_graph, sub_graph_id, ctx)
        return _generate_non_w_impl(sub_graph, sub_graph_id, ctx)

    return generate_impl


AVAILABLE_IMPL_BACKENDS = {
    "triton": _make_backend_dispatcher(generate_triton_impl),
    "cuda":   _make_backend_dispatcher(generate_cuda_impl),
}
