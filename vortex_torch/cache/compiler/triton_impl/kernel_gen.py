"""Triton kernel generator for fused cache subgraphs.

Cache pipelines schedule on **end-of-block tokens** — each page contains
``NUM_BLOCKS_PER_PAGE`` blocks, and we want one program per block (not
per page). This mirrors :func:`vortex_torch.cache.triton_kernels.reduce_impl.reduce_pp_kernel`:

    grid: (NNZ, NUM_KV_HEAD)
    NNZ = loc.shape[0]
    each program loads token_position = loc[token_id], returns early
    unless (token_position + 1) % BLOCK_SIZE == 0.

Indexing inside the kernel:

  * ``page_id  = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id``
  * ``block_id = page_id * NUM_BLOCKS_PER_PAGE
                  + (token_position % PAGE_SIZE) // BLOCK_SIZE``

Per-format addressing:

  * **PAGED** input/output  → indexed by ``block_id`` (one tile per block)
  * **RAGGED** (token-major) → indexed by ``token_id * NUM_KV_HEAD + head_id``
    (one tile per block-end token)

This contrasts with :mod:`vortex_torch.indexer.compiler.triton_impl.kernel_gen`
which schedules on workload chunks (``winfo_*``).

Schedule.S subgraphs (single op, no fusion) emit just the impl wrapper
returned by the op's codegen — same convention as the indexer side.
"""

from ..graph import Graph
from typing import List
import torch
from ...context import Context
from ....abs import FORMAT
from ....utils import Schedule, INDENT, indent_block
from .register import get_impl_func


# ---------------------------------------------------------------------------
# FP8 helpers: same semantics as
# ``vortex_torch.cache.triton_kernels.reduce_impl.reduce_pp_kernel``.
#
#   * On load  : FP8 tensor is passed to the kernel as ``uint8`` and
#                bitcast back inside (``tl.float8e5`` / ``tl.float8e4nv``)
#                before casting to fp32 for computation.
#   * On store : fp32 → target dtype; FP8 targets go via
#                ``.to(tl.float8eX).to(tl.uint8, bitcast=True)`` so the
#                store writes raw FP8 bits into a uint8-viewed buffer.
#
# The wrapper side ``.view(torch.uint8)`` each FP8 tensor before
# launching, mirroring ``_quant_view`` in the cache kernel helpers.
# ---------------------------------------------------------------------------

_FP8_DTYPES = (torch.float8_e5m2, torch.float8_e4m3fn)


def _is_fp8(t) -> bool:
    return t.dtype in _FP8_DTYPES


def _load_cast_expr(tensor_ptr_expr: str, t) -> str:
    """Return a Triton expression that loads from ``tensor_ptr_expr`` and
    yields a ``tl.float32`` block, decoding FP8 if necessary."""
    if t.dtype == torch.float8_e5m2:
        return (
            f"tl.load({tensor_ptr_expr}).to(tl.float8e5, bitcast=True).to(tl.float32)"
        )
    if t.dtype == torch.float8_e4m3fn:
        return (
            f"tl.load({tensor_ptr_expr}).to(tl.float8e4nv, bitcast=True).to(tl.float32)"
        )
    # bf16 / fp16 / fp32 / ... — the default cast is enough.
    return f"tl.load({tensor_ptr_expr}).to(tl.float32)"


def _load_cast_expr_masked(tensor_ptr_expr: str, mask_expr: str, t) -> str:
    """Variant of :func:`_load_cast_expr` that applies ``mask_expr`` and
    pads out-of-bounds lanes with zero. Used when the tensor's inner
    dims aren't already a power of two.
    """
    if t.dtype == torch.float8_e5m2:
        return (
            f"tl.load({tensor_ptr_expr}, mask={mask_expr}, other=0)"
            f".to(tl.float8e5, bitcast=True).to(tl.float32)"
        )
    if t.dtype == torch.float8_e4m3fn:
        return (
            f"tl.load({tensor_ptr_expr}, mask={mask_expr}, other=0)"
            f".to(tl.float8e4nv, bitcast=True).to(tl.float32)"
        )
    return (
        f"tl.load({tensor_ptr_expr}, mask={mask_expr}, other=0.0).to(tl.float32)"
    )


_TORCH_TO_TL = {
    torch.bfloat16: "tl.bfloat16",
    torch.float16:  "tl.float16",
    torch.float32:  "tl.float32",
}


def _store_cast_expr(block_expr: str, t) -> str:
    """Return the expression to be written by ``tl.store`` so it matches the
    underlying storage dtype of ``t`` (after the wrapper's FP8 → uint8 view).

    FP8 paths clamp to the representable range before the narrow cast to
    avoid saturating to ``inf``. Max finite magnitudes: e5m2 = 57344,
    e4m3fn = 448 (mirrors ``set_kv_buffer_fp8_*``).
    """
    if t.dtype == torch.float8_e5m2:
        clamped = f"tl.minimum(tl.maximum({block_expr}, -57344.0), 57344.0)"
        return f"{clamped}.to(tl.float8e5).to(tl.uint8, bitcast=True)"
    if t.dtype == torch.float8_e4m3fn:
        clamped = f"tl.minimum(tl.maximum({block_expr}, -448.0), 448.0)"
        return f"{clamped}.to(tl.float8e4nv).to(tl.uint8, bitcast=True)"
    tl_name = _TORCH_TO_TL.get(t.dtype)
    if tl_name is None:
        raise NotImplementedError(
            f"cache.kernel_gen: unsupported store dtype {t.dtype!r}"
        )
    return f"{block_expr}.to({tl_name})"


def _padded_inner_mask_expr(local_tensor_id: int, t) -> str:
    """Build ``(dim1_ptr < shape[1]) & (dim2_ptr < shape[2])`` with the
    2D broadcast suffix the cache load/store sites use; ``""`` when no
    masking is needed (both inner dims pow2).
    """
    if not t.needs_padding():
        return ""
    parts: List[str] = []
    if t.padded_shape[1] != t.shape[1]:
        parts.append(f"(tensor_{local_tensor_id}_dim1_ptr[:, None] < {t.shape[1]})")
    if t.padded_shape[2] != t.shape[2]:
        parts.append(f"(tensor_{local_tensor_id}_dim2_ptr[None, :] < {t.shape[2]})")
    return " & ".join(parts) if parts else ""


def generate_initialization_str(sub_graph: Graph, ctx: Context) -> str:
    """Per-tensor index pointer setup (used by load/store snippets).

    Both inner-axis arange constexprs emit ``padded_shape`` so Triton's
    pow2 requirement holds; the load/store helpers attach a mask
    against the real ``shape`` whenever the two differ.
    """
    lines: List[str] = []
    for local_tensor_id in list(sub_graph.input_tensor_ids) + list(sub_graph.output_tensor_ids):
        t = sub_graph.tensor_list[local_tensor_id]
        lines.append(
            f"tensor_{local_tensor_id}_dim1_ptr = tl.arange(0, {t.padded_shape[1]})"
        )
        lines.append(
            f"tensor_{local_tensor_id}_dim2_ptr = tl.arange(0, {t.padded_shape[2]})"
        )
    return "\n".join(lines) if lines else "# No initialization required"


def _block_load_lines(local_tensor_id: int, t, ctx: Context) -> List[str]:
    """Emit lines that load one (D0, D1) block into ``tensor_<id>_block`` (fp32).

    FP8 inputs are bitcast from the uint8-viewed pointer into the matching
    ``tl.float8eX`` dtype before the fp32 cast, matching ``reduce_pp_kernel``.

    When ``t`` needs padding, the load is masked: trailing padded lanes
    of either inner axis read as zero. Memory strides stay anchored to
    the real ``shape``.
    """
    lines: List[str] = []
    if t._format == FORMAT.PAGED:
        # Block-major addressing: read block_id (page_id * NUM_BLOCKS_PER_PAGE + block-in-page).
        lines.append(
            f"tensor_{local_tensor_id}_off = block_id * {t.shape[1] * t.shape[2]}"
        )
    elif t._format == FORMAT.RAGGED:
        # Token-major addressing: read token-tile (token_id, head_id).
        lines.append(
            f"tensor_{local_tensor_id}_off = (token_id * NUM_KV_HEAD + head_id) "
            f"* {t.shape[1] * t.shape[2]}"
        )
    else:
        raise NotImplementedError(
            f"cache.kernel_gen: load for format {t._format} not implemented yet"
        )

    lines.append(
        f"tensor_{local_tensor_id}_ptr_2d = "
        f"tensor_{local_tensor_id}_ptr + tensor_{local_tensor_id}_off "
        f"+ tensor_{local_tensor_id}_dim1_ptr[:, None] * {t.shape[2]} "
        f"+ tensor_{local_tensor_id}_dim2_ptr[None, :]"
    )
    mask_expr = _padded_inner_mask_expr(local_tensor_id, t)
    if mask_expr:
        load_expr = _load_cast_expr_masked(
            f"tensor_{local_tensor_id}_ptr_2d", mask_expr, t,
        )
    else:
        load_expr = _load_cast_expr(f"tensor_{local_tensor_id}_ptr_2d", t)
    lines.append(f"tensor_{local_tensor_id}_block = {load_expr}")
    return lines


def _block_store_lines(local_tensor_id: int, t, ctx: Context) -> List[str]:
    """Emit lines that store ``tensor_<id>_block`` back to the right slot.

    Output dtype selection:

      * ``bf16``              — default; ``.to(tl.bfloat16)``.
      * ``fp8_e5m2``          — ``.to(tl.float8e5).to(tl.uint8, bitcast=True)``
      * ``fp8_e4m3fn``        — ``.to(tl.float8e4nv).to(tl.uint8, bitcast=True)``

    FP8 tensors are passed in viewed as ``uint8`` by the wrapper, so the
    store writes raw FP8 bits into the same backing storage. Padded
    inner dims pick up an explicit mask so the synthetic lanes don't
    bleed into the next slot.
    """
    lines: List[str] = []
    if t._format == FORMAT.PAGED:
        lines.append(
            f"tensor_{local_tensor_id}_off = block_id * {t.shape[1] * t.shape[2]}"
        )
    elif t._format == FORMAT.RAGGED:
        lines.append(
            f"tensor_{local_tensor_id}_off = (token_id * NUM_KV_HEAD + head_id) "
            f"* {t.shape[1] * t.shape[2]}"
        )
    else:
        raise NotImplementedError(
            f"cache.kernel_gen: store for format {t._format} not implemented yet"
        )

    lines.append(
        f"tensor_{local_tensor_id}_ptr_2d = "
        f"tensor_{local_tensor_id}_ptr + tensor_{local_tensor_id}_off "
        f"+ tensor_{local_tensor_id}_dim1_ptr[:, None] * {t.shape[2]} "
        f"+ tensor_{local_tensor_id}_dim2_ptr[None, :]"
    )
    store_expr = _store_cast_expr(f"tensor_{local_tensor_id}_block", t)
    mask_expr = _padded_inner_mask_expr(local_tensor_id, t)
    if mask_expr:
        lines.append(
            f"tl.store(tensor_{local_tensor_id}_ptr_2d, {store_expr}, "
            f"mask={mask_expr})"
        )
    else:
        lines.append(f"tl.store(tensor_{local_tensor_id}_ptr_2d, {store_expr})")
    return lines


def generate_load_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    blocks: List[str] = []
    for local_tensor_id in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        blocks.append("\n".join(_block_load_lines(local_tensor_id, t, ctx)))
    return "\n\n".join(blocks) if blocks else "# No tensor loading required"


def generate_store_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    blocks: List[str] = []
    for local_tensor_id in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[local_tensor_id]
        blocks.append("\n".join(_block_store_lines(local_tensor_id, t, ctx)))
    return "\n\n".join(blocks) if blocks else "# No tensor storing required"


def generate_computation_str(sub_graph: Graph, ctx: Context) -> str:
    lines: List[str] = []
    for op_id, op in enumerate(sub_graph.op_list):
        op_impl_func = get_impl_func(op)
        lines.append(op_impl_func(sub_graph, op_id, ctx))
    return "\n\n".join(lines) if lines else "# No computation required"


def generate_triton_kernel(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Emit the @triton.jit kernel for a Schedule.W cache subgraph."""
    kernel_arg_list: List[str] = ["loc,"]

    for local_tensor_id in sub_graph.input_tensor_ids:
        kernel_arg_list.append(f"tensor_{local_tensor_id}_ptr,")
    for local_tensor_id in sub_graph.output_tensor_ids:
        kernel_arg_list.append(f"tensor_{local_tensor_id}_ptr,")

    kernel_arg_list.extend([
        "NUM_KV_HEAD: tl.constexpr,",
        "PAGE_SIZE: tl.constexpr,",
        "BLOCK_SIZE: tl.constexpr,",
        "NUM_BLOCKS_PER_PAGE: tl.constexpr,",
    ])

    kernel_args = "\n".join(f"{INDENT}{arg}" for arg in kernel_arg_list)

    initialization_str = indent_block(generate_initialization_str(sub_graph, ctx), 1)
    load_tensor_str = indent_block(generate_load_tensor_str(sub_graph, ctx), 1)
    store_tensor_str = indent_block(generate_store_tensor_str(sub_graph, ctx), 1)
    computation_str = indent_block(generate_computation_str(sub_graph, ctx), 1)

    kernel_str = f"""
@triton.jit
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel(
{kernel_args}
):
    # ------------------------------------------------------------
    # Token / head program ids; one program per (token, head). Trigger
    # only when this token is the last token of its block (a page
    # contains NUM_BLOCKS_PER_PAGE blocks of BLOCK_SIZE tokens each).
    # ------------------------------------------------------------
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    token_position = tl.load(loc + token_id)
    if (token_position + 1) % BLOCK_SIZE != 0:
        return

    page_id = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
    block_id = page_id * NUM_BLOCKS_PER_PAGE + (token_position % PAGE_SIZE) // BLOCK_SIZE

{initialization_str}

{load_tensor_str}

{computation_str}

{store_tensor_str}
"""
    return kernel_str.strip()


def generate_triton_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Generate the per-subgraph kernel + thin Python wrapper."""
    ctx.compilation_header_lines.extend([
        "import torch",
        "import triton",
        "import triton.language as tl",
    ])

    if sub_graph.schedule == Schedule.W:
        kernel_str = generate_triton_kernel(sub_graph, sub_graph_id, ctx)

        arg_list: List[str] = []
        kernel_input_list: List[str] = ["loc"]

        # FP8-aware wrapper: any FP8 tensor (input or output) needs to be
        # reinterpreted as uint8 before being passed to the kernel, which
        # bitcasts it back inside. ``fp8_rebind_lines`` holds the rebinding
        # statements inserted at the top of the impl function body.
        fp8_rebind_lines: List[str] = []

        for local_tensor_id in sub_graph.input_tensor_ids:
            tensor_name = f"tensor_{local_tensor_id}"
            arg_list.append(tensor_name)
            kernel_input_list.append(tensor_name)
            if _is_fp8(sub_graph.tensor_list[local_tensor_id]):
                fp8_rebind_lines.append(
                    f"{tensor_name} = {tensor_name}.view(torch.uint8)"
                )
        for local_tensor_id in sub_graph.output_tensor_ids:
            tensor_name = f"tensor_{local_tensor_id}"
            arg_list.append(tensor_name)
            kernel_input_list.append(tensor_name)
            if _is_fp8(sub_graph.tensor_list[local_tensor_id]):
                fp8_rebind_lines.append(
                    f"{tensor_name} = {tensor_name}.view(torch.uint8)"
                )

        arg_list.append("loc")
        arg_list.append("ctx")

        kernel_input_list.extend([
            "NUM_KV_HEAD=ctx.head_num",
            "PAGE_SIZE=ctx.page_size",
            "BLOCK_SIZE=ctx.block_size",
            "NUM_BLOCKS_PER_PAGE=ctx.num_blocks_per_page",
            "num_warps=4",
            "num_stages=1",
        ])

        args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)
        kernel_inputs = ",\n".join(f"{INDENT * 2}{arg}" for arg in kernel_input_list)
        fp8_rebind_str = (
            indent_block("\n".join(fp8_rebind_lines), 1) if fp8_rebind_lines else ""
        )

        impl_str = f"""
{kernel_str}

def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def},
{INDENT}cur_layer=0,
):
{fp8_rebind_str}
    {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel[(loc.shape[0], ctx.head_num)](
{kernel_inputs}
    )
"""
        return impl_str.strip()

    # Schedule.S: standalone op — defer entirely to the op's Schedule.S codegen
    # (cache-side custom_impl registry). The emitted impl takes the explicit
    # trailing ``cur_layer`` arg so per-layer-weight ops can gather their slice.
    from .. import custom_impl

    arg_list = []
    for local_tensor_id in sub_graph.input_tensor_ids:
        arg_list.append(f"tensor_{local_tensor_id}")
    for local_tensor_id in sub_graph.output_tensor_ids:
        arg_list.append(f"tensor_{local_tensor_id}")
    arg_list.append("loc")
    arg_list.append("ctx")
    args_def = ",\n".join(f"{INDENT}{arg}" for arg in arg_list)

    assert len(sub_graph.op_list) == 1, (
        "Expected exactly one operation in non-workload-scheduled cache subgraph."
    )
    custom_impl.register_headers(ctx)
    op_impl_func = custom_impl.get_impl_func(sub_graph.op_list[0])
    op_impl_str = indent_block(op_impl_func(sub_graph, 0, ctx), 1)

    impl_str = f"""
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def},
{INDENT}cur_layer=0,
):
{op_impl_str}
"""
    return impl_str.strip()
