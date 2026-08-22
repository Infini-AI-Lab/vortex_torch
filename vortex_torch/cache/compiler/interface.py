"""Generate the Python interface that wires compiled cache subgraphs.

Mirrors :mod:`vortex_torch.indexer.compiler.interface`, with two cache-
specific differences:

  * The entry-point class signature is ``forward(loc, cache, ctx)``
    (cache pipelines are driven by a per-token ``loc`` tensor and a
    cache buffer, rather than the indexer's ``q, o, cache, ctx``).
  * Memory initialization for intermediate RAGGED tensors uses
    ``ctx.max_new_tokens_per_batch * ctx.head_num`` for the leading
    dimension (the runtime "B" axis on the cache side).
"""

from .graph import Graph
from typing import Tuple, List
from ..context import Context
from ...utils import INDENT, indent_block
from ...abs import FORMAT
from .impl import AVAILABLE_IMPL_BACKENDS
from .triton_impl.kernel_gen import SLOT_MAP_ARG, has_slotted
import os


def generate_interface(full_graph: Graph, sub_graphs: List[Graph], ctx: Context) -> Tuple[str, str]:
    """Emit the compiled module to disk and return ``(file_path, class_name)``."""

    cache_dir = ctx.compilation_cache_dir or os.path.dirname(__file__)
    cache_dir = os.path.expanduser(cache_dir)
    cache_dir = os.path.abspath(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(
        cache_dir,
        f"{ctx.sparse_attention_name}_compiled_func.py",
    )
    print(f"Generating compiled cache function interface at {dst}")

    body_parts: List[str] = []

    for sub_graph_id, sub_graph in enumerate(sub_graphs):
        body_parts.append(generate_subgraph_func(sub_graph, sub_graph_id, ctx))

    body_parts.append(generate_entry_point(full_graph, sub_graphs, ctx))

    header_lines = list(dict.fromkeys(ctx.compilation_header_lines))
    header_str = "\n".join(header_lines)
    auxilary_func_def_str = "\n".join(ctx.auxilary_func_def_lines)
    body_str = "\n".join(body_parts)

    final_str = header_str + "\n\n" + auxilary_func_def_str + "\n\n" + body_str

    with open(dst, "w") as f:
        f.write(final_str)

    return dst, f"{ctx.sparse_attention_name}_CompiledFunc"


def generate_subgraph_func(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Generate the per-subgraph Python interface (impl + thin wrapper)."""

    generate_impl = AVAILABLE_IMPL_BACKENDS.get(ctx.impl_backend)
    if generate_impl is None:
        raise RuntimeError(f"Unknown impl_backend: {ctx.impl_backend!r}")
    impl_str = generate_impl(sub_graph, sub_graph_id, ctx)

    arg_list: List[str] = []
    for local_tensor_id in sub_graph.input_tensor_ids:
        arg_list.append(f"tensor_{local_tensor_id}")
    for local_tensor_id in sub_graph.output_tensor_ids:
        arg_list.append(f"tensor_{local_tensor_id}")
    arg_list.append("loc")
    arg_list.append("ctx")

    # The request domain's key->slot map, threaded like ``loc``: one extra argument, present only
    # when this subgraph touches a SLOTTED tensor. Note the def and the call need DIFFERENT
    # spellings (``x=None`` vs ``x=x``) -- the two lists below are shared for every other argument,
    # which is why this one cannot just be appended to ``arg_list``.
    slot_def: List[str] = []
    slot_call: List[str] = []
    if has_slotted(sub_graph):
        slot_def.append(f"{SLOT_MAP_ARG}=None")
        slot_call.append(f"{SLOT_MAP_ARG}={SLOT_MAP_ARG}")

    # Explicit trailing per-layer argument (additive, backward compatible).
    # The cache pipeline compiles ONCE but is invoked for every decode layer's
    # aux refresh; ops that bake per-layer constants (e.g. LearnedDescriptor)
    # select the active layer's slice via this value. Defaults to 0 so every
    # existing centroid flow keeps working unchanged. Mirrors the indexer side.
    args_with_layer = arg_list + slot_def + ["cur_layer=0"]
    call_with_layer = arg_list + slot_call + ["cur_layer=cur_layer"]

    args_def = ",\n    ".join(args_with_layer)
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


def generate_subgraph_entry_point(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
    tensor_id_to_tensor_name_map: dict,
) -> str:
    lines: List[str] = []
    lines.append(f"{ctx.sparse_attention_name}_subgraph_{sub_graph_id}_interface(")
    for global_tensor_id in sub_graph.global_input_tensor_ids:
        tensor_name = tensor_id_to_tensor_name_map[global_tensor_id]
        lines.append(f"    {tensor_name},  # global input tensor {global_tensor_id}")
    for global_tensor_id in sub_graph.global_output_tensor_ids:
        tensor_name = tensor_id_to_tensor_name_map[global_tensor_id]
        lines.append(f"    {tensor_name},  # global output tensor {global_tensor_id}")
    lines.append("    loc,")
    lines.append("    ctx,")
    if has_slotted(sub_graph):
        lines.append(f"    {SLOT_MAP_ARG}={SLOT_MAP_ARG},")
    # Thread the explicit per-layer argument from forward() into each subgraph.
    lines.append("    cur_layer=cur_layer,")
    lines.append(")")
    return "\n".join(lines)


def generate_entry_point(full_graph: Graph, sub_graphs: List[Graph], ctx: Context) -> str:
    """Emit a ``CompiledFunc`` class with ``__init__`` (buffer alloc) + ``forward``."""

    memory_initiazation_lines: List[str] = []
    entry_point_impl_lines: List[str] = []
    tensor_id_to_tensor_name_map = ctx.tensor_id_to_tensor_name_map

    final_output_set = set(full_graph.global_output_tensor_ids)

    for sub_graph_id, sub_graph in enumerate(sub_graphs):
        for local_tensor_id in sub_graph.output_tensor_ids:
            t = sub_graph.tensor_list[local_tensor_id]
            # Final outputs are passed in by the caller (mapped to the cache
            # buffers); only allocate buffers for true intermediates.
            if t.tensor_id in final_output_set:
                continue
            # SLOTTED lands here too if a flow ever produces a request-domain intermediate. It
            # must not: the domain owns its slots and their lifetime, so a buffer allocated here
            # would be addressed through the domain's map while being invisible to it.
            assert t._format == FORMAT.RAGGED, (
                f"Expected ragged tensor format for intermediate outputs, "
                f"got {t._format}"
                + (" -- a request-domain (SLOTTED) field must be declared by "
                   "create_request_cache and passed in, never allocated as an intermediate"
                   if t._format == FORMAT.SLOTTED else "")
            )
            B = ctx.max_new_tokens_per_batch * ctx.head_num
            memory_initiazation_lines.append(
                f"self.tensor_{t.tensor_id} = torch.empty("
                f"({B}, {t.shape[1]}, {t.shape[2]}), "
                f"dtype={t.dtype}, device='{t.device}')"
            )
            tensor_id_to_tensor_name_map[t.tensor_id] = f"self.tensor_{t.tensor_id}"

    # Python forbids empty function bodies — fall back to ``pass`` when
    # a subgraph pipeline has no intermediate RAGGED tensors to allocate
    # (e.g., everything reads from and writes to caller-provided buffers)
    # or when there are no subgraphs to dispatch to.
    if not memory_initiazation_lines:
        memory_initiazation_lines = ["pass"]
    memory_initiazation_str = indent_block("\n".join(memory_initiazation_lines), 2)
    # ``cur_layer`` is an EXPLICIT trailing forward() argument (default 0): the
    # compiled cache pipeline runs for every decode layer's aux refresh but is
    # compiled once, so the caller (memory_pool_mla) passes the active global
    # layer id here and per-layer ops gather their layer slice. The default
    # keeps every existing caller (which passes only cache/loc/ctx) working.
    entry_point_arg_str = "cache, loc, ctx, cur_layer=0"
    if any(has_slotted(sg) for sg in sub_graphs):
        entry_point_arg_str += f", {SLOT_MAP_ARG}=None"

    for sub_graph_id, sub_graph in enumerate(sub_graphs):
        entry_point_impl_lines.append(
            generate_subgraph_entry_point(
                sub_graph, sub_graph_id, ctx, tensor_id_to_tensor_name_map,
            )
        )
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
