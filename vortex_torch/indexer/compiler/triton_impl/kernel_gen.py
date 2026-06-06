"""Triton kernel generator for Schedule.W indexer subgraphs.

Emits a ``@triton.jit`` kernel + a Python wrapper that launches it
across ``num_sms * 4`` programs. The kernel iterates planner-scheduled
workloads ``[start, end)``, each covering ``workload_chunk_size`` ragged
slots / pages. Generated kernels load inputs by ``FORMAT``
(BATCHED / PAGED / RAGGED), run a chain of op impl strings, then store
outputs by ``FORMAT``. The store body is also gated for BATCHED outputs
that may be written by multiple workloads sharing the same
``(batch, head)`` slot.

Schedule.S codegen lives in :mod:`indexer.compiler.custom_impl` and is
dispatched from :mod:`indexer.compiler.impl` — it has no involvement
with this backend.

Public API: :func:`generate_triton_impl`. The other ``generate_*`` /
helper symbols below are kept exported for in-tree debugging and tests.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

from ..graph import Graph
from ...context import Context
from ....abs import FORMAT
from ....utils import Schedule, INDENT
from .register import get_impl_func
from ..backend import get_backend
from .dtype_cast import (
    is_fp8 as _is_fp8,
    load_cast_expr as _load_cast_expr,
    store_cast_expr as _store_cast_expr,
    compute_tl_dtype as _compute_tl_dtype,
)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def _has_batched_output(sub_graph) -> bool:
    """True iff the subgraph stores any BATCHED tensor.

    Drives ``winfo_is_first_workload_per_batch`` in the kernel signature /
    launcher and the per-workload "first" gate in the store body.
    """
    return any(
        sub_graph.tensor_list[tid]._format == FORMAT.BATCHED
        for tid in sub_graph.output_tensor_ids
    )


def indent_block(text: str, level: int = 1) -> str:
    """Indent every non-blank line in ``text`` by ``level`` ``INDENT`` units."""
    prefix = INDENT * level
    return "\n".join(
        prefix + line if line.strip() else line for line in text.splitlines()
    )


def _join_sections(sections: Iterable[str], fallback: str) -> str:
    """Join non-empty ``sections`` with ``\\n\\n``; return ``fallback`` if all empty."""
    non_empty = [s for s in sections if s]
    return "\n\n".join(non_empty) if non_empty else fallback


def _contig_arange(ctx: Context, n: int) -> str:
    """``tl.arange(0, n)``, optionally wrapped with contiguity / alignment
    hints for backends that guarantee consecutive arange lanes map to
    consecutive (and ``n``-aligned) global addresses.

    Emits ``tl.max_contiguous(tl.multiple_of(tl.arange(0, n), n), n)``
    when ``get_backend(ctx).wants_index_hints`` is set (trtllm). The
    ``multiple_of(_, n)`` reads as "this contiguous run starts on an
    ``n``-aligned boundary"; for a bare ``arange`` the base is ``0`` so
    it is trivially correct, and ``max_contiguous(_, n)`` states the ``n``
    lanes are contiguous — both true for an arange. The hints let Triton
    fold the per-tile pointer math into wide aligned LDG transactions.
    Plain ``tl.arange(0, n)`` otherwise (e.g. flashinfer CSR).
    """
    if get_backend(ctx).wants_index_hints:
        return f"tl.max_contiguous(tl.multiple_of(tl.arange(0, {n}), {n}), {n})"
    return f"tl.arange(0, {n})"


def _iter_kernel_tensors(sub_graph) -> Iterable[Tuple[int, str]]:
    """Yield ``(tensor_id, role)`` for each kernel argument tensor.

    A tensor that's both an input (``Load``) and an output (``Save``)
    appears exactly once, with ``role='output'``. Otherwise role is
    ``'input'`` (input-only) or ``'output'`` (output-only).
    """
    output_set = set(sub_graph.output_tensor_ids)
    for tid in sub_graph.input_tensor_ids:
        if tid in output_set:
            continue
        yield tid, "input"
    for tid in sub_graph.output_tensor_ids:
        yield tid, "output"


def _emit_block_ptr(tensor_id: int, t, offset_var: str, block_dim0_expr) -> str:
    """Emit one ``tl.make_block_ptr(...)`` assignment.

    Used by BATCHED/RAGGED/PAGED loads and the PAGED-single-page store
    **only when ``t`` doesn't require padding** (its inner dims are
    already powers of two). The strides + shape come from ``t.shape``;
    only the offset and block-shape leading dim differ across callers.

    For padded tensors (``t.needs_padding()`` is True) the fused-load /
    fused-store path can't use ``make_block_ptr`` because we'd have to
    fabricate a global shape consistent with a padded inner row stride,
    which doesn't match the real memory layout. Padded paths build
    pointers and masks from raw arange exprs instead — see
    :func:`_load_*_padded` / :func:`_store_*_padded`.
    """
    return "\n".join([
        f"tensor_{tensor_id}_block_ptr = tl.make_block_ptr(",
        f"{INDENT}base=tensor_{tensor_id}_ptr,",
        f"{INDENT}shape=(tensor_{tensor_id}_dim0 * {t.shape[1]}, {t.shape[2]}),",
        f"{INDENT}strides=({t.shape[2]}, 1),",
        f"{INDENT}offsets=({offset_var}, 0),",
        f"{INDENT}block_shape=({block_dim0_expr}, {t.shape[2]}),",
        f"{INDENT}order=(1, 0),",
        f")",
    ])


# --------------------------------------------------------------------------- #
# Padding-aware helpers
# --------------------------------------------------------------------------- #
#
# Triton block-shape constexprs (``tl.zeros``, ``tl.arange``,
# ``tl.reshape``, ``tl.make_block_ptr.block_shape``, ...) must each be a
# power of two. Real models carry non-pow2 inner dims (e.g. Qwen3-14B's
# group_size = 5). To handle both:
#
#   * Tile-side block shapes always emit ``t.padded_shape[i]`` (pow2 by
#     construction). For pow2-shape tensors this equals ``t.shape[i]`` —
#     no overhead.
#   * Memory-addressing strides always emit ``t.shape[i]`` (the real
#     row stride). ``make_block_ptr.shape`` / ``strides`` likewise.
#   * When ``t.needs_padding()`` is True the load/store path switches
#     to a raw arange + explicit mask form (``make_block_ptr`` won't fit
#     because the padded block_shape would over-read into the next row).
#   * When ``t.needs_padding()`` is False the fast path (the original
#     ``make_block_ptr`` + ``boundary_check`` form for loads, the
#     arange-stride form for stores) is preserved verbatim.
#
# The helpers below produce the inner-dim arange + mask snippets so the
# load/store helpers stay readable.


def _inner_dim_arange_names(tid: int) -> Tuple[str, str]:
    """Local names of the two inner-axis arange pointers for tensor ``tid``."""
    return f"tensor_{tid}_dim1_ptr", f"tensor_{tid}_dim2_ptr"


def _inner_dim_arange_decls(tid: int, t) -> List[str]:
    """Emit ``tl.arange`` declarations for the two inner axes at padded length.

    These are shared between load and store on the same tensor, so the
    caller dedupes (the same arange is fine to define once).
    """
    d1, d2 = _inner_dim_arange_names(tid)
    return [
        f"{d1} = tl.arange(0, {t.padded_shape[1]})",
        f"{d2} = tl.arange(0, {t.padded_shape[2]})",
    ]


def _inner_dim_mask_expr(tid: int, t, *, d1_broadcast: str, d2_broadcast: str) -> str:
    """Build the boolean ``(d1 < shape[1]) & (d2 < shape[2])`` expression
    with the broadcast suffixes the caller needs.

    ``d1_broadcast`` / ``d2_broadcast`` are e.g. ``"[None, :, None]"``
    matching the surrounding tile rank.

    Returns ``"1"`` (always-true scalar) when ``t`` doesn't need
    padding — the caller can splice this into a larger ``mask=`` expr
    without a dedicated guard.
    """
    if not t.needs_padding():
        return "1"
    d1, d2 = _inner_dim_arange_names(tid)
    parts: List[str] = []
    if t.padded_shape[1] != t.shape[1]:
        parts.append(f"({d1}{d1_broadcast} < {t.shape[1]})")
    if t.padded_shape[2] != t.shape[2]:
        parts.append(f"({d2}{d2_broadcast} < {t.shape[2]})")
    return " & ".join(parts) if parts else "1"


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #

def generate_initialization_str(sub_graph: Graph, ctx: Context) -> str:
    """Pre-kernel-loop declarations: zero-init BATCHED input slots and
    pre-build ``tl.arange`` index pointers for ragged / paged outputs.

    Block-shape constexprs (``tl.zeros``, ``tl.arange``) always emit the
    **padded** length so Triton's pow2-constexpr rule is satisfied; the
    real ``shape`` only enters memory-addressing math and per-tensor
    masks emitted by the load/store helpers.
    """
    lines: List[str] = []
    arange_emitted: set = set()  # (tid, "d1"|"d2") -> dedupe
    compute_dtype = _compute_tl_dtype(ctx)

    def _emit_arange(tid: int, t) -> None:
        d1, d2 = _inner_dim_arange_names(tid)
        if (tid, "d1") not in arange_emitted:
            lines.append(f"{d1} = {_contig_arange(ctx, t.padded_shape[1])}")
            arange_emitted.add((tid, "d1"))
        if (tid, "d2") not in arange_emitted:
            lines.append(f"{d2} = {_contig_arange(ctx, t.padded_shape[2])}")
            arange_emitted.add((tid, "d2"))

    for tid in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.BATCHED:
            lines.append(f"# Declare variables for tensor_{tid}")
            # Placeholder; the per-workload load reassigns this block in
            # ``compute_dtype``. Match the dtype so the SSA value type is
            # stable across the loop's first iteration.
            lines.append(
                f"tensor_{tid}_block = tl.zeros((1, {t.padded_shape[1]}, "
                f"{t.padded_shape[2]}), dtype={compute_dtype})"
            )
        # Padded input tensors (any FORMAT) need the inner-axis arange
        # pre-built — they reference it in the masked-load body.
        if t.needs_padding():
            _emit_arange(tid, t)

    for tid in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format in (FORMAT.RAGGED, FORMAT.BATCHED):
            _emit_arange(tid, t)

    # Multi-page PAGED scratch index pointers (workload spans >1 page).
    # ``page_idx_i32_ptr`` enumerates pages within a workload;
    # ``block_i32_ptr`` enumerates blocks within a page; ``tensor_X_flat_ptr``
    # is the flattened per-page element index (one page = num_blocks_per_page
    # blocks of padded_shape[1] x padded_shape[2] elements — the per-page
    # tile fed to ``tl.reshape``).
    if ctx.num_pages_per_workload > 1:
        lines.append(f"page_idx_i32_ptr = {_contig_arange(ctx, ctx.num_pages_per_workload)}")
        lines.append(f"block_i32_ptr = {_contig_arange(ctx, ctx.num_blocks_per_page)}")
        for tid in sub_graph.input_tensor_ids:
            t = sub_graph.tensor_list[tid]
            if t._format == FORMAT.PAGED:
                if t.needs_padding():
                    # Padded PAGED load builds a 4D pointer (npw, nbp, d1, d2)
                    # directly; the flat arange is unused. The inner aranges
                    # are emitted above (or here as a fallback).
                    _emit_arange(tid, t)
                else:
                    # Per-page flat run is fully stride-1 contiguous in the
                    # cache pool (trtllm guarantees the page's blocks are
                    # consecutive), so the hint folds it into a wide LDG.
                    lines.append(
                        f"tensor_{tid}_flat_ptr = "
                        f"{_contig_arange(ctx, ctx.num_blocks_per_page * t.shape[1] * t.shape[2])}"
                    )
        for tid in sub_graph.output_tensor_ids:
            t = sub_graph.tensor_list[tid]
            if t._format == FORMAT.PAGED:
                _emit_arange(tid, t)

    return "\n".join(lines) if lines else "# No initialization required"


# --------------------------------------------------------------------------- #
# Load (per FORMAT)
# --------------------------------------------------------------------------- #

def _load_batched(tid: int, t, compute_dtype: str = "tl.float32") -> List[str]:
    if t.needs_padding():
        return _load_batched_padded(tid, t, compute_dtype)
    block_ptr = _emit_block_ptr(
        tid, t,
        offset_var=f"tensor_{tid}_ptr_row_start",
        block_dim0_expr=t.shape[1],
    )
    reshape = (
        f"tl.reshape(tl.load(tensor_{tid}_block_ptr, boundary_check=(0, 1), "
        f'padding_option="zero", cache_modifier=".ca"), '
        f"(1, {t.shape[1]}, {t.shape[2]}))"
    )
    return [
        f"tensor_{tid}_ptr_row_start = new_batch_idx_i32 * {t.shape[1]}",
        block_ptr,
        f"tensor_{tid}_block = {_load_cast_expr(reshape, t, compute_dtype)}",
    ]


def _load_batched_padded(tid: int, t, compute_dtype: str = "tl.float32") -> List[str]:
    """Masked BATCHED load: tile (1, padded_shape[1], padded_shape[2]).

    Memory layout: ``[N_rows, shape[1], shape[2]]`` flattened. A
    ``make_block_ptr`` with padded block_shape would read past the real
    row boundary into the next row's data — incorrect — so build the
    pointer from raw aranges and apply an inner-axis mask.
    """
    d1, d2 = _inner_dim_arange_names(tid)
    mask = _inner_dim_mask_expr(
        tid, t,
        d1_broadcast="[None, :, None]",
        d2_broadcast="[None, None, :]",
    )
    row_base = f"new_batch_idx_i32 * {t.shape[1] * t.shape[2]}"
    ptr_expr = (
        f"tensor_{tid}_block_ptr = (tensor_{tid}_ptr + {row_base} "
        f"+ {d1}[None, :, None] * {t.shape[2]} "
        f"+ {d2}[None, None, :])"
    )
    load_expr = (
        f"tl.load(tensor_{tid}_block_ptr, mask={mask}, other=0.0, "
        f'cache_modifier=".ca")'
    )
    return [
        ptr_expr,
        f"tensor_{tid}_block = {_load_cast_expr(load_expr, t, compute_dtype)}",
    ]


def _load_paged_single_page(tid: int, t, ctx: Context,
                            compute_dtype: str = "tl.float32") -> List[str]:
    if t.needs_padding():
        return _load_paged_single_page_padded(tid, t, ctx, compute_dtype)
    block_ptr = _emit_block_ptr(
        tid, t,
        offset_var=f"tensor_{tid}_ptr_row_start",
        block_dim0_expr=t.shape[1] * ctx.workload_chunk_size,
    )
    reshape = (
        f"tl.reshape(tl.load(tensor_{tid}_block_ptr, boundary_check=(0, 1), "
        f'padding_option="zero", cache_modifier=".cv"), '
        f"({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]}))"
    )
    return [
        f"tensor_{tid}_ptr_row_start = page_idx_i32 * {t.shape[1]}",
        block_ptr,
        f"tensor_{tid}_block = {_load_cast_expr(reshape, t, compute_dtype)}",
    ]


def _load_paged_single_page_padded(tid: int, t, ctx: Context,
                                   compute_dtype: str = "tl.float32") -> List[str]:
    """Masked PAGED single-page load.

    Tile shape ``(workload_chunk_size, padded_shape[1], padded_shape[2])``.
    Memory stride per workload-slot is ``shape[1] * shape[2]``; rows
    inside a slot are spaced by ``shape[2]``.
    """
    d1, d2 = _inner_dim_arange_names(tid)
    mask = _inner_dim_mask_expr(
        tid, t,
        d1_broadcast="[None, :, None]",
        d2_broadcast="[None, None, :]",
    )
    page_base = f"page_idx_i32 * {t.shape[1] * t.shape[2]}"
    ptr_expr = (
        f"tensor_{tid}_block_ptr = (tensor_{tid}_ptr + {page_base} "
        f"+ workload_ptr[:, None, None] * {t.shape[1] * t.shape[2]} "
        f"+ {d1}[None, :, None] * {t.shape[2]} "
        f"+ {d2}[None, None, :])"
    )
    load_expr = (
        f"tl.load(tensor_{tid}_block_ptr, mask={mask}, other=0.0, "
        f'cache_modifier=".cv")'
    )
    return [
        ptr_expr,
        f"tensor_{tid}_block = {_load_cast_expr(load_expr, t, compute_dtype)}",
    ]


def _load_paged_multi_page(tid: int, t, ctx: Context,
                           compute_dtype: str = "tl.float32") -> List[str]:
    if t.needs_padding():
        return _load_paged_multi_page_padded(tid, t, ctx, compute_dtype)
    # ``page_valid`` masks the per-page lanes (length num_pages_per_workload);
    # distinct from ``valid`` which masks the per-block workload lanes.
    reshape = (
        f"tl.reshape(tl.load(tensor_{tid}_block_ptr, mask=page_valid[:, None], "
        f'other=0.0, cache_modifier=".cv"), '
        f"({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]}))"
    )
    return [
        f"_tensor_{tid}_block_ptr = page_indices_i32[:,None] * "
        f"{t.shape[1] * t.shape[2]} + tensor_{tid}_flat_ptr[None,:]",
        f"tensor_{tid}_block_ptr = tensor_{tid}_ptr + _tensor_{tid}_block_ptr",
        f"tensor_{tid}_block = {_load_cast_expr(reshape, t, compute_dtype)}",
    ]


def _load_paged_multi_page_padded(tid: int, t, ctx: Context,
                                  compute_dtype: str = "tl.float32") -> List[str]:
    """Masked PAGED multi-page load.

    Build a 4D pointer ``(npw, nbp, padded_d1, padded_d2)`` and apply
    both the existing ``page_valid`` page-level mask and the inner
    real-shape mask. The flat-arange path used by the non-padded form
    can't represent the inner-dim mask because it collapses the inner
    axes into a single flat index.
    """
    d1, d2 = _inner_dim_arange_names(tid)
    block_elem = t.shape[1] * t.shape[2]
    parts = [
        f"page_valid[:, None, None, None]",
        f"({d1}[None, None, :, None] < {t.shape[1]})" if t.padded_shape[1] != t.shape[1] else None,
        f"({d2}[None, None, None, :] < {t.shape[2]})" if t.padded_shape[2] != t.shape[2] else None,
    ]
    mask = " & ".join(p for p in parts if p)
    ptr_expr = (
        f"tensor_{tid}_block_ptr = (tensor_{tid}_ptr "
        f"+ page_indices_i32[:, None, None, None] * {block_elem} "
        f"+ block_i32_ptr[None, :, None, None] * {block_elem} "
        f"+ {d1}[None, None, :, None] * {t.shape[2]} "
        f"+ {d2}[None, None, None, :])"
    )
    load_expr = (
        f"tl.load(tensor_{tid}_block_ptr, mask={mask}, other=0.0, "
        f'cache_modifier=".cv")'
    )
    reshape_expr = (
        f"tl.reshape({load_expr}, "
        f"({ctx.workload_chunk_size}, {t.padded_shape[1]}, {t.padded_shape[2]}))"
    )
    return [
        ptr_expr,
        f"tensor_{tid}_block = {_load_cast_expr(reshape_expr, t, compute_dtype)}",
    ]


def _load_ragged(tid: int, t, ctx: Context,
                 compute_dtype: str = "tl.float32") -> List[str]:
    if t.needs_padding():
        return _load_ragged_padded(tid, t, ctx, compute_dtype)
    block_ptr = _emit_block_ptr(
        tid, t,
        offset_var=f"tensor_{tid}_ptr_row_start",
        block_dim0_expr=t.shape[1] * ctx.workload_chunk_size,
    )
    reshape = (
        f"tl.reshape(tl.load(tensor_{tid}_block_ptr, boundary_check=(0, 1), "
        f'padding_option="zero",), '
        f"({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]}))"
    )
    return [
        f"tensor_{tid}_ptr_row_start = ragged_idx_i32 * {t.shape[1]}",
        block_ptr,
        f"tensor_{tid}_block = {_load_cast_expr(reshape, t, compute_dtype)}",
    ]


def _load_ragged_padded(tid: int, t, ctx: Context,
                        compute_dtype: str = "tl.float32") -> List[str]:
    """Masked RAGGED load: tile (workload_chunk_size, padded_d1, padded_d2)."""
    d1, d2 = _inner_dim_arange_names(tid)
    mask = _inner_dim_mask_expr(
        tid, t,
        d1_broadcast="[None, :, None]",
        d2_broadcast="[None, None, :]",
    )
    block_elem = t.shape[1] * t.shape[2]
    ptr_expr = (
        f"tensor_{tid}_block_ptr = (tensor_{tid}_ptr "
        f"+ ragged_idx_i32 * {block_elem} "
        f"+ workload_ptr[:, None, None] * {block_elem} "
        f"+ {d1}[None, :, None] * {t.shape[2]} "
        f"+ {d2}[None, None, :])"
    )
    load_expr = f"tl.load(tensor_{tid}_block_ptr, mask={mask}, other=0.0)"
    return [
        ptr_expr,
        f"tensor_{tid}_block = {_load_cast_expr(load_expr, t, compute_dtype)}",
    ]


def generate_load_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """Emit input-tensor load code, grouped by FORMAT.

    Layout: BATCHED (with the ``new_batch_idx_i32`` load prefix) →
    addressing (``ragged_idx_i32`` + paged-index load) → PAGED → RAGGED.
    """
    batched: List[str] = []
    paged: List[str] = []
    ragged: List[str] = []

    compute_dtype = _compute_tl_dtype(ctx)

    for tid in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.BATCHED:
            batched.extend(_load_batched(tid, t, compute_dtype))
        elif t._format == FORMAT.PAGED:
            if ctx.num_pages_per_workload == 1:
                paged.extend(_load_paged_single_page(tid, t, ctx, compute_dtype))
            else:
                paged.extend(_load_paged_multi_page(tid, t, ctx, compute_dtype))
        elif t._format == FORMAT.RAGGED:
            ragged.extend(_load_ragged(tid, t, ctx, compute_dtype))

    batched_str = (
        "\n".join([
            "new_batch_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)",
            "\n".join(batched),
        ])
        if batched else ""
    )

    if ctx.num_pages_per_workload == 1:
        page_idx_line = "page_idx_i32 = tl.load(indices + ragged_idx_i32).to(tl.int32)"
    else:
        # ``page_idx_i32_ptr`` enumerates pages within a workload; multiplying
        # by ``num_blocks_per_page`` yields the block-granular stride into
        # ``indices`` (which is keyed at block-of-page granularity).
        page_idx_line = (
            f"page_indices_i32 = tl.load(indices + ragged_idx_i32 + "
            f"page_idx_i32_ptr * {ctx.num_blocks_per_page}, mask=page_valid, "
            f"other=0).to(tl.int32)"
        )
    addressing_str = "\n".join([
        "ragged_idx_i32 = tl.load(winfo_y_offsets + i).to(tl.int32)",
        page_idx_line,
    ])

    paged_str = "\n".join(paged) if paged else "# No paged tensor loading required"
    ragged_str = "\n".join(ragged) if ragged else "# No ragged tensor loading required"

    return _join_sections(
        [batched_str, addressing_str, paged_str, ragged_str],
        fallback="# No tensor loading required",
    )


# --------------------------------------------------------------------------- #
# Store (per FORMAT)
# --------------------------------------------------------------------------- #

def _and_inner_mask(tid: int, t, *, base_mask: str,
                    d1_broadcast: str, d2_broadcast: str) -> str:
    """Combine ``base_mask`` with the inner-dim masks when ``t`` is padded.

    The base mask carries the format-specific gating (RAGGED's ``valid``
    over the workload axis, PAGED multi-page's ``page_valid``) and is
    always present; the inner-dim conjuncts only fire when at least one
    inner dim was rounded up.
    """
    if not t.needs_padding():
        return base_mask
    extra = _inner_dim_mask_expr(
        tid, t,
        d1_broadcast=d1_broadcast,
        d2_broadcast=d2_broadcast,
    )
    if extra == "1":
        return base_mask
    return f"{base_mask} & {extra}"


def _store_paged_single_page(tid: int, t, ctx: Context) -> List[str]:
    if t.needs_padding():
        return _store_paged_single_page_padded(tid, t, ctx)
    block_ptr = _emit_block_ptr(
        tid, t,
        offset_var=f"tensor_{tid}_ptr_row_start",
        block_dim0_expr=t.shape[1] * ctx.workload_chunk_size,
    )
    reshape = (
        f"tl.reshape(tensor_{tid}_block, "
        f"({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}))"
    )
    return [
        f"tensor_{tid}_ptr_row_start = page_idx_i32 * {t.shape[1]}",
        block_ptr,
        f"tl.store(tensor_{tid}_block_ptr, {_store_cast_expr(reshape, t.dtype)})",
    ]


def _store_paged_single_page_padded(tid: int, t, ctx: Context) -> List[str]:
    """Masked PAGED single-page store.

    ``tensor_<id>_block`` already has padded inner dims (from the
    upstream load / compute). Build a per-element pointer + mask the
    inner padded lanes out of the store.
    """
    d1, d2 = _inner_dim_arange_names(tid)
    block_elem = t.shape[1] * t.shape[2]
    ptr_expr = (
        f"tensor_{tid}_block_ptr = (tensor_{tid}_ptr "
        f"+ page_idx_i32 * {block_elem} "
        f"+ workload_ptr[:, None, None] * {block_elem} "
        f"+ {d1}[None, :, None] * {t.shape[2]} "
        f"+ {d2}[None, None, :])"
    )
    mask = _inner_dim_mask_expr(
        tid, t,
        d1_broadcast="[None, :, None]",
        d2_broadcast="[None, None, :]",
    )
    return [
        ptr_expr,
        f"tl.store(tensor_{tid}_block_ptr, "
        f"{_store_cast_expr(f'tensor_{tid}_block', t.dtype)}, mask={mask})",
    ]


def _store_paged_multi_page(tid: int, t, ctx: Context) -> List[str]:
    # When ``page_size > block_size`` the destination layout has separate
    # page-of-workload and block-of-page axes; the pointer expression and
    # reshape both keep all four (page, block, dim1, dim2).
    ptr_expr = (
        f"tensor_{tid}_block_ptr = tensor_{tid}_ptr "
        f"+ page_indices_i32[:,None,None,None] * {t.shape[1] * t.shape[2]} "
        f"+ block_i32_ptr[None,:,None,None] * {t.shape[1] * t.shape[2]} "
        f"+ tensor_{tid}_dim1_ptr[None,None,:,None] * {t.shape[2]} "
        f"+ tensor_{tid}_dim2_ptr[None,None,None,:]"
    )
    reshape = (
        f"tl.reshape(tensor_{tid}_block, "
        f"({ctx.num_pages_per_workload}, {ctx.num_blocks_per_page}, "
        f"{t.padded_shape[1]}, {t.padded_shape[2]}))"
    )
    mask = _and_inner_mask(
        tid, t,
        base_mask="page_valid[:, None, None, None]",
        d1_broadcast="[None, None, :, None]",
        d2_broadcast="[None, None, None, :]",
    )
    return [
        ptr_expr,
        f"tl.store(tensor_{tid}_block_ptr, "
        f"{_store_cast_expr(reshape, t.dtype)}, "
        f"mask={mask})",
    ]


def _store_ragged(tid: int, t) -> List[str]:
    ptr_expr = (
        f"tensor_{tid}_block_ptr = tensor_{tid}_ptr "
        f"+ ragged_idx_i32 * {t.shape[1] * t.shape[2]} "
        f"+ workload_ptr[:,None,None] * {t.shape[1] * t.shape[2]} "
        f"+ tensor_{tid}_dim1_ptr[None,:,None] * {t.shape[2]} "
        f"+ tensor_{tid}_dim2_ptr[None,None,:]"
    )
    mask = _and_inner_mask(
        tid, t,
        base_mask="valid[:, None, None]",
        d1_broadcast="[None, :, None]",
        d2_broadcast="[None, None, :]",
    )
    return [
        ptr_expr,
        f"tl.store(tensor_{tid}_block_ptr, "
        f"{_store_cast_expr(f'tensor_{tid}_block', t.dtype)}, "
        f"mask={mask})",
    ]


def _store_batched(tid: int, t) -> List[str]:
    """One slot per (batch, head). Multiple workloads can share the same
    (batch, head) when its KV span exceeds workload_chunk_size, but only
    the *first* such workload writes — gated by
    ``winfo_is_first_workload_per_batch[i]`` (computed by the planner).
    The gate load is hoisted by :func:`generate_store_tensor_str` so it
    happens once per workload regardless of how many BATCHED outputs the
    subgraph has.

    When ``t`` needs padding, the per-element store also picks up the
    inner-dim mask so the trailing pow2-padded lanes don't overwrite
    the next batch slot's data.
    """
    ptr_expr = (
        f"tensor_{tid}_block_ptr = tensor_{tid}_ptr "
        f"+ new_batch_idx_i32 * {t.shape[1] * t.shape[2]} "
        f"+ tensor_{tid}_dim1_ptr[None, :, None] * {t.shape[2]} "
        f"+ tensor_{tid}_dim2_ptr[None, None, :]"
    )
    if t.needs_padding():
        mask = _inner_dim_mask_expr(
            tid, t,
            d1_broadcast="[None, :, None]",
            d2_broadcast="[None, None, :]",
        )
        store_call = (
            f"    tl.store(tensor_{tid}_block_ptr, "
            f"{_store_cast_expr(f'tensor_{tid}_block', t.dtype)}, mask={mask})"
        )
    else:
        store_call = (
            f"    tl.store(tensor_{tid}_block_ptr, "
            f"{_store_cast_expr(f'tensor_{tid}_block', t.dtype)})"
        )
    return [
        ptr_expr,
        "if _is_first_workload != 0:",
        store_call,
    ]


def generate_store_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """Emit output-tensor store code, grouped by FORMAT.

    Layout: PAGED → RAGGED (with the workload ``valid`` mask preamble
    when single-page) → BATCHED (with the hoisted ``_is_first_workload``
    gate load).
    """
    paged: List[str] = []
    ragged: List[str] = []
    batched: List[str] = []

    for tid in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.PAGED:
            if ctx.num_pages_per_workload == 1:
                paged.extend(_store_paged_single_page(tid, t, ctx))
            else:
                paged.extend(_store_paged_multi_page(tid, t, ctx))
        elif t._format == FORMAT.RAGGED:
            ragged.extend(_store_ragged(tid, t))
        elif t._format == FORMAT.BATCHED:
            batched.extend(_store_batched(tid, t))

    paged_str = "\n".join(paged) if paged else ""

    if ragged:
        # When ``num_pages_per_workload == 1`` the preamble doesn't define
        # ``_len`` / ``valid`` (the multi-page preamble does); RAGGED stores
        # need both, so emit them here in the single-page case.
        ragged_str = (
            "\n".join([
                "_len = tl.load(winfo_y_lens + i)",
                "valid = workload_ptr < _len",
                "\n".join(ragged),
            ])
            if ctx.num_pages_per_workload == 1 else "\n".join(ragged)
        )
    else:
        ragged_str = ""

    batched_str = (
        "\n".join([
            "_is_first_workload = tl.load(winfo_is_first_workload_per_batch + i)",
            "\n".join(batched),
        ])
        if batched else ""
    )

    # Fallback comment kept verbatim from pre-refactor for byte-equivalent
    # output (the original generated "# No tensor loading required" for
    # the store fallback path; preserved to avoid drift in golden diffs).
    return _join_sections(
        [paged_str, ragged_str, batched_str],
        fallback="# No tensor loading required",
    )


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #

def generate_computation_str(sub_graph: Graph, ctx: Context) -> str:
    lines = [
        get_impl_func(op)(sub_graph, op_id, ctx)
        for op_id, op in enumerate(sub_graph.op_list)
    ]
    return "\n\n".join(lines) if lines else "# No computation required"


# --------------------------------------------------------------------------- #
# Kernel definition (W-scheduled)
# --------------------------------------------------------------------------- #

def _build_kernel_signature(sub_graph) -> List[str]:
    """Kernel-argument lines in declaration order."""
    args = [
        "indices,                            # int32",
        "winfo_x_indices,                    # int32",
    ]
    if _has_batched_output(sub_graph):
        args.append("winfo_is_first_workload_per_batch,  # uint8")
    args += [
        "winfo_y_offsets,                    # int32",
        "winfo_y_lens,                       # int32",
        "winfo_num_workloads,                # int32",
    ]
    for tid, _role in _iter_kernel_tensors(sub_graph):
        args.extend([
            f"tensor_{tid}_ptr,",
            f"tensor_{tid}_dim0: tl.constexpr,",
        ])
    return args


def _build_workload_preamble(ctx: Context) -> str:
    """Per-iteration ``_len`` / ``valid`` / ``_page`` / ``page_valid`` lines
    (only needed when each workload spans multiple pages)."""
    if ctx.num_pages_per_workload <= 1:
        return "# No workload preparation required"
    return "\n".join([
        "_len = tl.load(winfo_y_lens + i)",
        "valid = workload_ptr < _len",
        f"_page = (_len + {ctx.num_blocks_per_page - 1}) // {ctx.num_blocks_per_page}",
        "page_valid = page_idx_i32_ptr < _page",
    ])


def generate_triton_kernel(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Emit the ``@triton.jit`` kernel string for a W-scheduled subgraph."""
    kernel_args = "\n".join(f"{INDENT}{arg}" for arg in _build_kernel_signature(sub_graph))

    initialization_str = indent_block(generate_initialization_str(sub_graph, ctx), 1)
    prepare_workload_str = indent_block(_build_workload_preamble(ctx), 2)
    load_tensor_str = indent_block(generate_load_tensor_str(sub_graph, ctx), 2)
    store_tensor_str = indent_block(generate_store_tensor_str(sub_graph, ctx), 2)
    computation_str = indent_block(generate_computation_str(sub_graph, ctx), 2)

    kernel_str = f"""
@triton.jit
def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel(
{kernel_args}
):
    # ------------------------------------------------------------
    # Program-level partitioning of workloads
    # ------------------------------------------------------------
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    n_workloads = tl.load(winfo_num_workloads)

    per = n_workloads // num_progs
    r = n_workloads % num_progs
    start = pid * per + tl.minimum(pid, r)
    end = start + per + (pid < r)
    workload_ptr = {_contig_arange(ctx, ctx.workload_chunk_size)}

{initialization_str}

    for i in range(start, end):
{prepare_workload_str}
{load_tensor_str}
{computation_str}
{store_tensor_str}
"""
    return kernel_str.strip()


# --------------------------------------------------------------------------- #
# Python launcher / impl wrapper
# --------------------------------------------------------------------------- #

def _build_launcher_args(sub_graph, ctx: Context) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(wrapper_args, kernel_call_inputs, fp8_rebind_lines)``.

    FP8 reinterpret: any FP8 tensor (input or output) is bitcast to
    ``uint8`` before being passed to the kernel, which bitcasts it back
    to the matching ``tl.float8eX`` dtype on load and stores raw fp8
    bits on write. Mirrors the cache-side convention.
    """
    wrapper_args: List[str] = []
    # The Schedule.W kernel preamble does
    #   ``page_idx_i32 = tl.load(indices + ragged_idx_i32)``
    # and ``ragged_idx_i32 * shape[1]`` for RAGGED store/load addresses.
    # Every backend-specific consumer-side detail (which tensor ``indices``
    # binds to, plus the matching ``winfo_kv_offsets`` encoding emitted by
    # the planner) is captured in :mod:`indexer.compiler.backend`.
    kernel_inputs: List[str] = [
        get_backend(ctx).indices_src,
        "ctx.metadata.winfo_q_indices",
    ]
    if _has_batched_output(sub_graph):
        kernel_inputs.append("ctx.metadata.winfo_is_first_workload_per_batch")
    kernel_inputs += [
        "ctx.metadata.winfo_kv_offsets",
        "ctx.metadata.winfo_kv_lens",
        "ctx.metadata.winfo_num_workloads",
    ]
    fp8_rebind: List[str] = []

    for tid, _role in _iter_kernel_tensors(sub_graph):
        name = f"tensor_{tid}"
        wrapper_args.append(name)
        kernel_inputs.extend([name, f"{name}.shape[0]"])
        if _is_fp8(sub_graph.tensor_list[tid]):
            fp8_rebind.append(f"{name} = {name}.view(torch.uint8)")

    wrapper_args.append("ctx")
    kernel_inputs.append("num_warps=8")
    kernel_inputs.append("num_stages=2")

    return wrapper_args, kernel_inputs, fp8_rebind


def _generate_w_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    kernel_str = generate_triton_kernel(sub_graph, sub_graph_id, ctx)
    wrapper_args, kernel_inputs, fp8_rebind = _build_launcher_args(sub_graph, ctx)

    args_def = ",\n".join(f"{INDENT}{arg}" for arg in wrapper_args)
    kernel_inputs_str = ",\n".join(f"{INDENT * 2}{arg}" for arg in kernel_inputs)
    fp8_rebind_str = (
        indent_block("\n".join(fp8_rebind), 1) if fp8_rebind else ""
    )

    impl_str = f"""
{kernel_str}

def {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_impl(
{args_def},
    cur_layer=0,
):
{fp8_rebind_str}
    {ctx.sparse_attention_name}_subgraph_{sub_graph_id}_kernel[({ctx.num_sms * 4},)](
{kernel_inputs_str}
    )
"""
    return impl_str.strip()


def generate_triton_impl(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Generate the Triton Schedule.W kernel and its Python wrapper.

    Schedule.S subgraphs are routed through
    :mod:`indexer.compiler.custom_impl` by
    :mod:`indexer.compiler.impl` and never reach this function.
    """
    assert sub_graph.schedule == Schedule.W, (
        f"generate_triton_impl only handles Schedule.W; got {sub_graph.schedule}. "
        f"Schedule.S subgraphs are dispatched via indexer.compiler.impl."
    )
    ctx.compilation_header_lines.extend([
        "import torch",
        "import triton",
        "import triton.language as tl",
    ])
    return _generate_w_impl(sub_graph, sub_graph_id, ctx)
