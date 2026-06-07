from .graph import Graph
from typing import Dict, Tuple, Callable
from ..context import Context
from ...utils import Schedule, INDENT, indent_block
from ...abs import FORMAT
from .impl import AVAILABLE_IMPL_BACKENDS
import os
def generate_interface(full_graph: Graph, sub_graphs: list[Graph], ctx: Context) -> str:

    cache_dir = ctx.compilation_cache_dir or os.path.dirname(__file__)
    cache_dir = os.path.expanduser(cache_dir)
    cache_dir = os.path.abspath(cache_dir)
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(
        cache_dir,
        f"{ctx.sparse_attention_name}_compiled_func.py"
    )
    print(f"Generating compiled function interface at {dst}")
    
    body_parts: list[str] = []

    for sub_graph_id, sub_graph in enumerate(sub_graphs):
        func_str = generate_subgraph_func(sub_graph, sub_graph_id, ctx)
        body_parts.append(func_str)

    entry_cls_str = generate_entry_point(full_graph, sub_graphs, ctx)
    body_parts.append(entry_cls_str)

    header_lines = list(dict.fromkeys(ctx.compilation_header_lines))
    header_str = "\n".join(header_lines)
    auxilary_func_def_str = "\n".join(ctx.auxilary_func_def_lines)
    body_str = "\n".join(body_parts)

    final_str = header_str + "\n\n" + auxilary_func_def_str + "\n\n" + body_str

    with open(dst, "w") as f:
        f.write(final_str)

    return dst, f"{ctx.sparse_attention_name}_CompiledFunc"


def generate_subgraph_func(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> Tuple[str, str]:
    """
    Generate a Python function interface for a given sub-graph.
    """

    generate_impl = AVAILABLE_IMPL_BACKENDS.get(ctx.impl_backend)
    impl_str = generate_impl(sub_graph, sub_graph_id, ctx)

    arg_list = []

    # A tensor that's both an input and an output (e.g. a cache field
    # Load'd + Save'd in the same subgraph) must only be bound once.
    output_set = set(sub_graph.output_tensor_ids)

    # Collect input tensors (skip any that also appear in outputs)
    for local_tensor_id in sub_graph.input_tensor_ids:
        if local_tensor_id in output_set:
            continue
        arg_list.append(f"tensor_{local_tensor_id}")

    # Collect output tensors
    for local_tensor_id in sub_graph.output_tensor_ids:
        arg_list.append(f"tensor_{local_tensor_id}")

    # Append context argument
    arg_list.append("ctx")

    # Explicit trailing per-layer argument (additive, backward compatible).
    # A vFlow compiles ONCE but is invoked for every decode layer; ops that
    # bake per-layer constants select the active layer's slice
    # from this value. Defaults to 0 so every existing flow keeps working.
    args_with_layer = arg_list + ["cur_layer=0"]
    call_with_layer = arg_list + ["cur_layer=cur_layer"]

    # For function definition (4-space indent)
    args_def = ",\n    ".join(args_with_layer)

    # For function call (8-space indent)
    args_call = ",\n        ".join(call_with_layer)

    func_str = f"""
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_interface(
    {args_def}
):

    {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
        {args_call}
    )
    """


    return impl_str + "\n\n" + func_str + "\n\n\n\n"

def generate_subgraph_entry_point(sub_graph: Graph, sub_graph_id: int, ctx: Context, tensor_id_to_tensor_name_map: dict) -> Tuple[str, str]:
    lines = []
    lines.append(f"{ctx.sparse_attention_name}_subgraph_{sub_graph_id}_interface(")
    # Match the input/output deduplication in ``generate_subgraph_func``
    # so the call-site argument count matches the target signature.
    output_set = set(sub_graph.global_output_tensor_ids)
    for global_tensor_id in sub_graph.global_input_tensor_ids:
        if global_tensor_id in output_set:
            continue
        tensor_name = tensor_id_to_tensor_name_map[global_tensor_id]
        lines.append(f"    {tensor_name},  # global input tensor {global_tensor_id}")
    for global_tensor_id in sub_graph.global_output_tensor_ids:
        tensor_name = tensor_id_to_tensor_name_map[global_tensor_id]
        lines.append(f"    {tensor_name},  # global output tensor {global_tensor_id}")
    lines.append("    ctx,")
    # Thread the explicit per-layer argument from forward() into each subgraph.
    lines.append("    cur_layer=cur_layer,")
    lines.append(")")
    return "\n".join(lines)

def generate_entry_point(full_graph: Graph, sub_graphs: list[Graph], ctx: Context) -> str:

    memory_initiazation_lines = []
    entry_point_arg_list = []
    entry_point_impl_lines = []
    tensor_id_to_tensor_name_map = ctx.tensor_id_to_tensor_name_map

    # Caller-provided tensors (the attention output ``o`` and any
    # side-effect sinks such as ``Save`` outputs pointing into the
    # cache) are already in ``full_graph.global_output_tensor_ids`` —
    # their names are supplied by the user via
    # ``ctx.tensor_id_to_tensor_name_map`` (e.g. ``"o"``, ``"cache['k']"``).
    # We only allocate ``self.tensor_<id>`` buffers for *intermediate*
    # RAGGED tensors that cross subgraph boundaries.
    # A tensor is "caller-provided" if the user already wired a Python name
    # for it (e.g. ``"o"``, ``"cache['k']"``). Side-effect sinks like
    # ``Save`` end up in ``global_output_tensor_ids`` *and* the name map —
    # we skip them. An intermediate that ended up flagged as a
    # global-output (e.g. an unused tensor that survived DCE as a sink)
    # without a user-provided name still needs a real buffer, so allocate.
    for sub_graph_id, sub_graph in enumerate(sub_graphs):
       for local_tensor_id in sub_graph.output_tensor_ids:
           t = sub_graph.tensor_list[local_tensor_id]
           if t.tensor_id in tensor_id_to_tensor_name_map:
                continue  # caller-provided ("o", cache fields, ...)
           if t._format == FORMAT.RAGGED:
               leading = ctx.max_num_blocks
           elif t._format == FORMAT.BATCHED:
               leading = ctx.max_bs * ctx.num_kv_heads
           else:
               raise AssertionError(
                   f"Intermediate tensor must be RAGGED or BATCHED, got {t._format}"
               )
           memory_initiazation_lines.append(f"self.tensor_{t.tensor_id} = torch.empty(({leading}, {t.shape[1]}, {t.shape[2]}), dtype={t.dtype}, device='{t.device}')")
           tensor_id_to_tensor_name_map[t.tensor_id] = f"self.tensor_{t.tensor_id}"
    
    # Python forbids empty function bodies — fall back to ``pass`` when
    # a subgraph pipeline has no intermediate RAGGED tensors to allocate,
    # or when there are no subgraphs to dispatch to.
    if not memory_initiazation_lines:
        memory_initiazation_lines = ["pass"]
    memory_initiazation_str = indent_block("\n".join(memory_initiazation_lines), 2)
    # Query arg name(s): a single "q" for MHA, or the absorbed pair
    # ("q_nope_out", "q_pe") for MLA flows. Driven by ctx so the generated
    # forward() signature matches the names the trace registered.
    query_arg_names = list(getattr(ctx, "query_arg_names", None) or ["q"])
    # ``cur_layer`` is an EXPLICIT trailing forward() argument (default 0):
    # the compiled indexer runs for every decode layer but is compiled once,
    # so the backend passes the active global layer id here and per-layer
    # ops gather their layer slice. The default keeps every
    # existing caller (which passes only q.../o/cache/ctx) working unchanged.
    entry_point_arg_list = query_arg_names + ["o", "cache", "ctx", "cur_layer=0"]
    entry_point_arg_str = ",".join(entry_point_arg_list)

    for sub_graph_id, sub_graph in enumerate(sub_graphs):
        impl_lines = generate_subgraph_entry_point(sub_graph, sub_graph_id, ctx, tensor_id_to_tensor_name_map)
        entry_point_impl_lines.append(impl_lines)
    if not entry_point_impl_lines:
        entry_point_impl_lines = ["pass"]
    entry_point_impl_str = indent_block("\n\n".join(entry_point_impl_lines), 2)
    entry_cls_str = f"""
class {ctx.sparse_attention_name}_CompiledFunc:
{INDENT}def __init__(self):
{memory_initiazation_str}

{INDENT}def forward(self, {entry_point_arg_str}):
{entry_point_impl_str}
""" 
    return entry_cls_str