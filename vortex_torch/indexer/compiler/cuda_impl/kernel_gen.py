"""CUDA kernel generator for Schedule.W indexer subgraphs.

Emits a CUDA ``__global__`` kernel and a host-side C++ launcher;
embeds both as a Python raw string that
``torch.utils.cpp_extension.load_inline`` JIT-compiles at
module-import time; wraps the compiled launcher in a Python function
that forwards tensor args + scalars pulled from ``ctx``.

Schedule.S codegen lives in :mod:`indexer.compiler.custom_impl` and is
dispatched from :mod:`indexer.compiler.impl` — it has no involvement
with this backend.

Public API: :func:`generate_cuda_impl`. The other ``generate_*`` /
helper symbols are exported for in-tree debugging and tests.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import torch

from ..graph import Graph
from ...context import Context
from ....abs import FORMAT
from ....utils import Schedule, INDENT
from .register import get_impl_func
from .dtype_cast import cast_float_to_smem


# --------------------------------------------------------------------------- #
# Torch dtype -> C++ type tables
# --------------------------------------------------------------------------- #
#
# Each entry is ``(aten_t, cuda_t)``:
#   * ``aten_t`` is the type used for ``tensor.data_ptr<...>()`` (the ATen-side
#     surface; e.g. ``at::Half`` for fp16, ``at::BFloat16`` for bf16). It is
#     the only type torch's ``data_ptr<T>`` template instantiates for those
#     half-precision dtypes.
#   * ``cuda_t`` is the type the generated kernel signature uses, picked so
#     in-kernel arithmetic uses the native CUDA intrinsic types
#     (``__half`` / ``__nv_bfloat16``). Where ``aten_t != cuda_t`` the
#     launcher emits a ``reinterpret_cast<cuda_t*>`` — those types are
#     binary-compatible so the cast is safe.
#
# FP8 dtypes are rebound to ``uint8`` on the Python side via
# ``.view(torch.uint8)`` before the launcher call, mirroring the Triton
# backend convention; their entries here are uint8 for both sides.
_TORCH_TO_CPP_TYPES = {
    torch.float32:       ("float",         "float"),
    torch.float16:       ("at::Half",      "__half"),
    torch.bfloat16:      ("at::BFloat16",  "__nv_bfloat16"),
    torch.int32:         ("int32_t",       "int32_t"),
    torch.int64:         ("int64_t",       "int64_t"),
    torch.uint8:         ("uint8_t",       "uint8_t"),
    torch.int8:          ("int8_t",        "int8_t"),
    torch.float8_e5m2:   ("uint8_t",       "uint8_t"),
    torch.float8_e4m3fn: ("uint8_t",       "uint8_t"),
}


# Byte size of each supported dtype — used to compute dynamic shared-memory
# offsets in :func:`generate_initialization_str`.
_TORCH_DTYPE_BYTES = {
    torch.float32:       4,
    torch.float16:       2,
    torch.bfloat16:      2,
    torch.int32:         4,
    torch.int64:         8,
    torch.uint8:         1,
    torch.int8:          1,
    torch.float8_e5m2:   1,
    torch.float8_e4m3fn: 1,
}


def _dtype_bytes(dtype) -> int:
    try:
        return _TORCH_DTYPE_BYTES[dtype]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )


def _is_fp8(t) -> bool:
    """True iff the tensor's dtype is one of the fp8 variants the framework
    accepts. Matches :func:`triton_impl.dtype_cast.is_fp8` so the cuda and
    triton backends agree on which tensors get the ``.view(torch.uint8)``
    rebind treatment.
    """
    return t.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)


def _cpp_kernel_ptr_type(dtype) -> str:
    """Return the kernel-side pointer type (``cuda_t*``) for a torch dtype."""
    try:
        return f"{_TORCH_TO_CPP_TYPES[dtype][1]}*"
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )


def _cpp_kernel_value_type(dtype) -> str:
    """Return the kernel-side value type (``cuda_t``) for a torch dtype.

    Sibling of :func:`_cpp_kernel_ptr_type`. Used by anything that
    declares a value-typed local / shared-memory tile rather than a
    pointer.
    """
    try:
        return _TORCH_TO_CPP_TYPES[dtype][1]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )




def _cpp_data_ptr_expr(name: str, dtype) -> str:
    """``<name>.data_ptr<aten_t>()``, ``reinterpret_cast``-ed to ``cuda_t*``
    when the two differ (fp16 / bf16). Used by the launcher to forward
    each ``torch::Tensor`` argument as the right pointer type to the
    kernel.
    """
    try:
        aten_t, cuda_t = _TORCH_TO_CPP_TYPES[dtype]
    except KeyError:
        raise NotImplementedError(
            f"cuda_impl.kernel_gen: unsupported tensor dtype {dtype!r}"
        )
    base = f"{name}.data_ptr<{aten_t}>()"
    if aten_t == cuda_t:
        return base
    return f"reinterpret_cast<{cuda_t}*>({base})"


# Includes shared by every JIT-compiled subgraph TU:
#   * ``<torch/extension.h>``        — ``torch::Tensor``, pybind11 glue.
#   * ``<ATen/cuda/CUDAContext.h>``  — ``at::cuda::getCurrentCUDAStream()``
#     used by the launcher emitted by :func:`generate_cuda_launcher`.
#   * ``<cuda_fp16.h>`` / ``<cuda_bf16.h>`` — ``__half`` / ``__nv_bfloat16``
#     names used in the kernel signature and the launcher's
#     ``reinterpret_cast``s.
#   * ``<cuda_pipeline.h>``           — ``__pipeline_memcpy_async`` /
#     ``__pipeline_commit`` / ``__pipeline_wait_prior`` for the
#     async-copy load path. Available on sm_80+ (Ampere) and is the
#     standard headers-only wrapper for ``cp.async.cg`` PTX.
#   * ``<cstdint>``                   — ``int32_t``, ``int64_t``, ``uint8_t``.
_CUDA_KERNEL_PREAMBLE = (
    "#include <torch/extension.h>\n"
    "#include <ATen/cuda/CUDAContext.h>\n"
    "#include <cuda_fp16.h>\n"
    "#include <cuda_bf16.h>\n"
    "#include <cuda_pipeline.h>\n"
    "#include <cstdint>"
)


def _pick_vec_bytes(
    n_elems: int, elem_bytes: int, stride_bytes: int,
) -> int:
    """Largest power-of-two vector size in {16, 8, 4, elem_bytes} that
    satisfies the **two** alignment constraints for an unrolled-vector
    copy:

      * ``(n_elems * elem_bytes) % vec_bytes == 0`` — the span length
        fits the vector lane count.
      * ``stride_bytes % vec_bytes == 0`` — the per-row / per-page
        starting offset baked into the caller's pointer expression
        (e.g. ``tensor_X_ptr + idx * block_elems`` becomes
        ``stride_bytes = block_elems * elem_bytes``) is also
        ``vec_bytes``-aligned. Without this check, a small ``block_elems``
        (e.g. ``1`` for ``[chunk, 1, 1]`` output tiles) lets a non-trivial
        ``idx`` start the load on an unaligned address — ``int4``
        reinterpret_cast triggers ``cudaErrorMisalignedAddress`` at
        runtime even when the span size itself is a clean multiple of 16.

    Smem tiles are 16-byte aligned by
    :func:`generate_initialization_str` and torch's cudaMalloc
    allocations are ≥ 256-byte aligned, so the implicit ``tensor_X_ptr``
    / ``__vortex_dyn_smem`` bases never break the constraint — only the
    runtime-variable index step does, which is what ``stride_bytes``
    encodes. Falls back to a single-element copy
    (``vec_bytes == elem_bytes``) when no wider option fits.
    """
    total_bytes = n_elems * elem_bytes
    for vb in (16, 8, 4):
        if vb <= elem_bytes:
            continue
        if total_bytes % vb == 0 and stride_bytes % vb == 0:
            return vb
    return elem_bytes


_VEC_BYTES_TO_CTYPE = {
    16: "int4",
    8:  "int2",
    4:  "int",
}


def _emit_vector_copy_loop(
    dst_ptr_expr: str, src_ptr_expr: str, n_elems: int,
    elem_bytes: int, indent_level: int, stride_bytes: int,
    async_load: bool = False,
) -> str:
    """Block-cooperative ``n_elems`` element copy from ``src_ptr_expr``
    to ``dst_ptr_expr``, vectorised when the span size *and* the per-row
    stride in bytes both line up with the vector lane count.

    Both pointer expressions are evaluated *inside* the emitted scope so
    each thread sees the same per-iteration offset. ``stride_bytes`` is
    the byte distance between consecutive starting addresses of this
    copy as the surrounding index changes — pass
    ``block_elems * elem_bytes`` for the contig load/store path
    (``tensor_X_ptr + idx * block_elems``) and
    ``page_elems * elem_bytes`` for the multi-page paths
    (``dst + p * page_elems`` / ``src + p * page_elems``). The
    vectorisation choice is the largest power-of-two width that divides
    both ``n_elems * elem_bytes`` and ``stride_bytes``; see
    :func:`_pick_vec_bytes`.

    ``async_load=True`` rewrites the vectorised copy to use
    ``__pipeline_memcpy_async`` (cp.async.cg.shared.global PTX). The
    issuing thread doesn't stall on the load; the caller is responsible
    for emitting one ``__pipeline_commit()`` per load section and one
    ``__pipeline_wait_prior(0)`` before the first consumer of the
    loaded data. cp.async only accepts 4 / 8 / 16-byte payloads, so the
    scalar fallback (``vec_bytes <= elem_bytes``, i.e. < 4 bytes) stays
    on the regular synchronous path even when ``async_load`` is set.

    Scalar fallback uses ``elem_bytes``-wide loads (a single ``dst[j] =
    src[j]`` per thread per stride). The vectorised path reinterprets
    both pointers as ``int4`` (16 B), ``int2`` (8 B) or ``int`` (4 B)
    and shrinks the stride loop count proportionally.
    """
    pre = INDENT * indent_level
    vec_bytes = _pick_vec_bytes(n_elems, elem_bytes, stride_bytes)
    # Parenthesise the pointer expressions so callers can pass arbitrary
    # ``ptr + offset`` arithmetic without it binding looser than the
    # trailing ``[j]`` subscript — e.g. ``src + p * 1[j]`` parses as
    # ``src + p * (1[j])`` and chokes nvcc.
    dst_p = f"({dst_ptr_expr})"
    src_p = f"({src_ptr_expr})"
    if vec_bytes <= elem_bytes:
        # Scalar fallback — synchronous regardless of ``async_load``.
        return (
            f"{pre}for (int j = tid; j < {n_elems}; j += blockDim.x) {{\n"
            f"{pre}{INDENT}{dst_p}[j] = {src_p}[j];\n"
            f"{pre}}}"
        )
    vec_ty = _VEC_BYTES_TO_CTYPE[vec_bytes]
    lane = vec_bytes // elem_bytes
    n_vec = n_elems // lane
    if async_load:
        # cp.async.cg.shared.global — non-blocking from the issuing
        # thread, bound to the current pipeline group. The caller's
        # ``__pipeline_commit()`` + ``__pipeline_wait_prior(0)`` book-
        # ending is what makes the loaded data visible.
        return (
            f"{pre}{{\n"
            f"{pre}{INDENT}{vec_ty}* __dst_v = reinterpret_cast<{vec_ty}*>{dst_p};\n"
            f"{pre}{INDENT}const {vec_ty}* __src_v = reinterpret_cast<const {vec_ty}*>{src_p};\n"
            f"{pre}{INDENT}for (int j = tid; j < {n_vec}; j += blockDim.x) {{\n"
            f"{pre}{INDENT}{INDENT}__pipeline_memcpy_async(__dst_v + j, __src_v + j, {vec_bytes});\n"
            f"{pre}{INDENT}}}\n"
            f"{pre}}}"
        )
    return (
        f"{pre}{{\n"
        f"{pre}{INDENT}{vec_ty}* __dst_v = reinterpret_cast<{vec_ty}*>{dst_p};\n"
        f"{pre}{INDENT}const {vec_ty}* __src_v = reinterpret_cast<const {vec_ty}*>{src_p};\n"
        f"{pre}{INDENT}for (int j = tid; j < {n_vec}; j += blockDim.x) {{\n"
        f"{pre}{INDENT}{INDENT}__dst_v[j] = __src_v[j];\n"
        f"{pre}{INDENT}}}\n"
        f"{pre}}}"
    )


# --------------------------------------------------------------------------- #
# Small utilities (pure, backend-agnostic)
# --------------------------------------------------------------------------- #

def _has_batched_output(sub_graph) -> bool:
    """True iff the subgraph stores any BATCHED tensor.

    Same role as in the Triton backend: drives the
    ``winfo_is_first_workload_per_batch`` kernel argument / launcher
    binding and the per-workload "first" gate in the store body.
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
    pass


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


# --------------------------------------------------------------------------- #
# Initialization / Load / Store / Computation codegen (CUDA C++)
# --------------------------------------------------------------------------- #

def _tensor_smem_bytes(t, chunk: int) -> int:
    """16-byte-aligned smem byte cost of a single tile of tensor ``t``.

    BATCHED tiles collapse the leading dim to 1 — one slot per
    ``(batch, head)`` is all the BATCHED codegens ever touch.
    """
    leading = 1 if t._format == FORMAT.BATCHED else chunk
    elem_bytes = _dtype_bytes(t.dtype)
    size = leading * t.shape[1] * t.shape[2] * elem_bytes
    return (size + 15) & ~15


def _compute_tensor_lifetimes(sub_graph: Graph):
    """Per-tensor ``(first_phase, last_phase)`` over the iteration body.

    Phases ordered as:

      * ``-1`` — ``LOAD_PHASE``. All inputs reach smem here in a single
        scope; multiple inputs cannot overlap with each other in smem
        because they're all written at this phase.
      * ``0 .. len(op_list) - 1`` — compute phases, one per op. A tile
        is read at the start of its consumer's phase and (for the
        producer's output) written at the end of the producer's phase.
        We conservatively bind both ends to the same op id; the
        ``last_phase >= first_phase`` overlap test means an op's
        inputs and outputs are treated as simultaneously live, so a
        single-op in-place reuse stays out of reach. That keeps us
        safe for reductions / broadcasts where multiple threads need
        to see the input slot intact while writing the output slot.
      * ``len(op_list)`` — ``STORE_PHASE``. All outputs drain here in
        a single scope; outputs cannot overlap with each other.

    A Load+Save tensor (sits in both input and output sets) gets
    ``(-1, n_ops)`` — alive across the entire iteration body.
    """
    n_ops = len(sub_graph.op_list)
    LOAD_PHASE = -1
    STORE_PHASE = n_ops
    input_set = set(sub_graph.input_tensor_ids)
    output_set = set(sub_graph.output_tensor_ids)
    lifetimes = {}
    for tid in range(len(sub_graph.tensor_list)):
        is_input = tid in input_set
        is_output = tid in output_set
        read_ops = [
            op_id for op_id, inputs in enumerate(sub_graph.op_to_input_tensor_list)
            if tid in inputs
        ]
        write_ops = [
            op_id for op_id, outputs in enumerate(sub_graph.op_to_output_tensor_list)
            if tid in outputs
        ]
        if is_input:
            first = LOAD_PHASE
        elif write_ops:
            first = min(write_ops)
        else:
            # No producer and not a subgraph input — should not normally
            # happen, but stay safe by pinning to LOAD_PHASE so the
            # tile sticks around until something consumes it.
            first = LOAD_PHASE
        if is_output:
            last = STORE_PHASE
        elif read_ops:
            last = max(read_ops)
        elif write_ops:
            # Written but never read and not an output. Dead immediately
            # after its writing op; keep birth == death.
            last = max(write_ops)
        else:
            last = first
        lifetimes[tid] = (first, last)
    return lifetimes


def _allocate_smem_offsets(sub_graph: Graph, ctx: Context):
    """Greedy lifetime-aware smem allocator.

    Returns ``(offsets, sizes, lifetimes, total_bytes)``. Tensors whose
    lifetimes don't overlap can share the same byte range, dropping
    per-kernel smem from the trivial sum-of-tiles bound. On the
    elementwise-heavy flows in this repo (``masked_quest``,
    ``gqa_quest``, ``lserve``) this typically halves or thirds the
    footprint, unlocking 2-3× the concurrent blocks per SM that the
    previous static layout could host.

    Algorithm: linear-scan in birth order. For each tensor, take the
    lowest 16-byte-aligned offset whose ``[offset, offset+size)``
    doesn't overlap any still-live previously-placed tile. Lifetimes
    use ``>=`` for the overlap test (see :func:`_compute_tensor_lifetimes`)
    so an op's inputs and outputs never overlap — that keeps reduce /
    broadcast op semantics intact even though their codegen does
    technically read-then-write within a single phase.
    """
    chunk = ctx.workload_chunk_size
    n = len(sub_graph.tensor_list)

    sizes = {
        tid: _tensor_smem_bytes(t, chunk)
        for tid, t in enumerate(sub_graph.tensor_list)
    }
    lifetimes = _compute_tensor_lifetimes(sub_graph)

    # Sort by birth ascending; tie-break by size descending so the
    # tallest tile in a birth cohort gets first pick of the floor.
    order = sorted(
        range(n),
        key=lambda t: (lifetimes[t][0], -sizes[t]),
    )

    offsets = {}
    for tid in order:
        first, _ = lifetimes[tid]
        size = sizes[tid]

        # Live conflicts = previously-placed tiles whose death is at
        # or after this tile's birth.
        conflicts = []
        for placed_tid, off in offsets.items():
            _, placed_last = lifetimes[placed_tid]
            if placed_last >= first:
                conflicts.append((off, off + sizes[placed_tid]))
        conflicts.sort()

        # Merge overlapping intervals so the gap-finding loop has
        # disjoint slots to compare against.
        merged = []
        for s, e in conflicts:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        candidate = 0
        for s, e in merged:
            candidate = (candidate + 15) & ~15
            if candidate + size <= s:
                break
            candidate = max(candidate, e)
        candidate = (candidate + 15) & ~15
        offsets[tid] = candidate

    total = max((offsets[t] + sizes[t] for t in offsets), default=0)
    return offsets, sizes, lifetimes, total


def generate_initialization_str(sub_graph: Graph, ctx: Context) -> str:
    """Declare typed-pointer smem aliases for every tensor in the subgraph.

    Every tensor's tile lives inside one ``extern __shared__`` blob
    (``__vortex_dyn_smem``). Per-tensor offsets are picked by
    :func:`_allocate_smem_offsets` from a liveness analysis of the
    subgraph: tiles whose ``(first_phase, last_phase)`` windows don't
    overlap can share the same byte range. The pointer-alias
    declarations preserve the source-level access pattern of every
    op codegen (``tensor_<id>_smem[i][j][k]`` /
    ``&tensor_<id>_smem[0][0][0]``) — only the offset arithmetic baked
    into each ``reinterpret_cast`` changes between layout strategies.

    Each tile is sized by the tensor's format:

      * RAGGED / PAGED:
            ``[workload_chunk_size][t.shape[1]][t.shape[2]]``
        Matches Triton's per-workload tile shape — one chunk of
        ragged blocks / paged blocks per iteration. Intermediates
        from Elementwise / Elementwise_Binary follow this layout when
        they inherit RAGGED format.
      * BATCHED:
            ``[1][t.shape[1]][t.shape[2]]``
        BATCHED tensors are one slot per ``(batch, head)``; the body
        only ever touches index ``0`` of the leading dim. Collapsing
        to ``1`` saves the otherwise-wasted
        ``(workload_chunk_size - 1) * shape[1] * shape[2]`` slots.

    The element type is the tensor's **storage** dtype (``__half`` /
    ``__nv_bfloat16`` / ``float`` / ``uint8_t`` / ``int32_t`` /
    ``int8_t``). Body codegen is responsible for promoting to fp32 in
    registers during arithmetic (see :mod:`cuda_impl.dtype_cast`).

    The launcher pairs this with
    ``cudaFuncSetAttribute(MaxDynamicSharedMemorySize, …)`` when the
    total exceeds 48 KB — see :func:`generate_cuda_launcher`.
    """
    if not sub_graph.tensor_list:
        return "// No initialization required"

    offsets, sizes, lifetimes, total = _allocate_smem_offsets(sub_graph, ctx)

    # Stash the total on the subgraph so the launcher can read it.
    sub_graph._smem_bytes = total  # type: ignore[attr-defined]
    # Stash the lifetimes + offsets so downstream codegens (e.g. a
    # future async-copy pass) can ask "is tile X still live during op
    # Y" without re-running the analysis.
    sub_graph._smem_offsets = offsets    # type: ignore[attr-defined]
    sub_graph._smem_sizes = sizes        # type: ignore[attr-defined]
    sub_graph._tensor_lifetimes = lifetimes  # type: ignore[attr-defined]

    lines: List[str] = ["extern __shared__ unsigned char __vortex_dyn_smem[];"]
    for tid, t in enumerate(sub_graph.tensor_list):
        cuda_t = _cpp_kernel_value_type(t.dtype)
        first, last = lifetimes[tid]
        lines.append(
            f"{cuda_t} (*tensor_{tid}_smem)[{t.shape[1]}][{t.shape[2]}] = "
            f"reinterpret_cast<{cuda_t} (*)[{t.shape[1]}][{t.shape[2]}]>"
            f"(__vortex_dyn_smem + {offsets[tid]});"
            f"  // life=[{first},{last}] sz={sizes[tid]}"
        )
    return "\n".join(lines)


def _emit_contig_load(
    tid: int, cuda_t: str, src_offset_expr: str, n_elems: int,
    elem_bytes: int, block_elems: int,
) -> str:
    """Strided block-cooperative copy of ``n_elems`` contiguous elements
    from ``tensor_<tid>_ptr + src_offset_expr`` into
    ``tensor_<tid>_smem`` (flattened).

    Used for BATCHED, RAGGED, and PAGED single-page loads — all three
    map to a contiguous span in global memory and differ only in
    ``src_offset_expr``. The strided pattern is delegated to
    :func:`_emit_vector_copy_loop`, which picks the widest ``int4`` /
    ``int2`` / ``int`` bulk-copy width compatible with both the span
    length and the per-index stride.

    ``block_elems`` is the per-index element step — the surrounding
    workload index multiplies it to form ``src_offset_expr``. The byte
    form ``block_elems * elem_bytes`` is what bounds the alignment
    available across all index values: for a tile shaped
    ``[chunk, 1, 1]`` the stride is just ``elem_bytes`` and the
    fallback is scalar, even though ``n_elems = chunk`` looks large
    enough to vectorise — without this check, an unaligned
    ``ragged_idx`` triggers ``cudaErrorMisalignedAddress``.
    """
    body = _emit_vector_copy_loop(
        dst_ptr_expr="dst",
        src_ptr_expr="src",
        n_elems=n_elems,
        elem_bytes=elem_bytes,
        indent_level=1,
        stride_bytes=block_elems * elem_bytes,
        async_load=True,
    )
    return (
        f"{{\n"
        f"{INDENT}auto* dst = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}const {cuda_t}* src = tensor_{tid}_ptr + {src_offset_expr};\n"
        f"{body}\n"
        f"}}"
    )


def _emit_multipage_load(
    tid: int, cuda_t: str, npw: int, nbp: int, block_elems: int,
    elem_bytes: int, zero_expr: str,
) -> str:
    """Page-by-page gather for PAGED inputs spanning multiple pages.

    For each ``p`` in ``[0, num_pages_per_workload)``:

      * If ``p < _page``: look up the start-of-page block index via
        ``indices[ragged_idx_i32 + p * num_blocks_per_page]``, then copy
        ``num_blocks_per_page * block_elems`` contiguous elements from
        that page into ``tensor_<tid>_smem[p * num_blocks_per_page ...]``.
      * Otherwise: zero-fill the same smem slot using ``zero_expr`` so
        compute can read uniformly without a per-page mask.
        ``zero_expr`` must already be a value of the storage dtype —
        the caller builds it via :func:`cuda_impl.dtype_cast.cast_float_to_smem`
        so the half-precision types get the right intrinsic
        (``__float2half_rn(0.0f)`` / ``__float2bfloat16_rn(0.0f)``)
        rather than a constructor call, which PyTorch's cpp_extension
        disables via ``__CUDA_NO_BFLOAT16_CONVERSIONS__``.

    Pages are written contiguously in smem (page p → slots
    ``[p*nbp, (p+1)*nbp)`` of the chunk axis), so compute and store
    index by ``chunk_idx`` and don't need to know the page layout. The
    per-page copy goes through :func:`_emit_vector_copy_loop` so the
    valid-page branch picks up the same ``int4`` / ``int2`` widening
    as the contig path. The zero-fill branch stays scalar — the
    ``zero_expr`` is a per-element value of the storage dtype, not a
    vector, and the cost of the cold-page write is dominated by the
    valid-page reads anyway.
    """
    page_elems = nbp * block_elems
    # Both dst (smem at p*page_elems) and src (global at block_idx*block_elems)
    # advance by ``block_elems * elem_bytes`` per index step (p / block_idx).
    # The src stride is the tighter of the two since smem is already 16-aligned
    # and ``page_elems = nbp * block_elems`` is a multiple of ``block_elems``.
    valid_copy = _emit_vector_copy_loop(
        dst_ptr_expr=f"dst + p * {page_elems}",
        src_ptr_expr="src",
        n_elems=page_elems,
        elem_bytes=elem_bytes,
        indent_level=3,
        stride_bytes=block_elems * elem_bytes,
        async_load=True,
    )
    return (
        f"{{\n"
        f"{INDENT}auto* dst = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}for (int p = 0; p < {npw}; ++p) {{\n"
        f"{INDENT}{INDENT}if (p < _page) {{\n"
        f"{INDENT}{INDENT}{INDENT}const int block_idx = "
        f"indices[ragged_idx_i32 + p * {nbp}];\n"
        f"{INDENT}{INDENT}{INDENT}const {cuda_t}* src = "
        f"tensor_{tid}_ptr + block_idx * {block_elems};\n"
        f"{valid_copy}\n"
        f"{INDENT}{INDENT}}} else {{\n"
        f"{INDENT}{INDENT}{INDENT}for (int j = tid; j < {page_elems}; "
        f"j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}{INDENT}{INDENT}dst[p * {page_elems} + j] = {zero_expr};\n"
        f"{INDENT}{INDENT}{INDENT}}}\n"
        f"{INDENT}{INDENT}}}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def generate_load_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """Cooperatively copy every subgraph **input** from global memory
    into its ``__shared__`` tile.

    Iterates :attr:`Graph.input_tensor_ids` directly — a Load+Save
    tensor sits in both ``input_tensor_ids`` and ``output_tensor_ids``,
    and the load side has to fire here regardless of whether the
    store side will later write back to the same tile. (Note:
    :func:`_iter_kernel_tensors` would yield such a tensor with role
    ``'output'`` and skip the load.)

    Output shape per format:

      * **BATCHED** — ``shape[1] * shape[2]`` contiguous elements from
        ``tensor_<id>_ptr + new_batch_idx_i32 * (shape[1] * shape[2])``
        into ``tensor_<id>_smem[0]``. One row per ``(batch, head)``
        slot; BATCHED smem tiles only have a single leading slot per
        the format gating in :func:`generate_initialization_str`.
      * **RAGGED** — ``workload_chunk_size * shape[1] * shape[2]``
        contiguous elements from
        ``tensor_<id>_ptr + ragged_idx_i32 * (shape[1] * shape[2])``
        into the full ``tensor_<id>_smem`` tile. The chunk is
        contiguous in global memory by construction.
      * **PAGED single-page** (``num_pages_per_workload == 1``) —
        identical pattern to RAGGED, but the global offset is
        ``page_idx_i32 * (shape[1] * shape[2])`` where
        ``page_idx_i32 = indices[ragged_idx_i32]`` is the start-of-page
        block index. The single page contains exactly the workload's
        ``num_blocks_per_page == workload_chunk_size`` blocks.
      * **PAGED multi-page** (``num_pages_per_workload > 1``) —
        page-by-page gather: each page p has block index
        ``indices[ragged_idx_i32 + p * num_blocks_per_page]``; valid
        pages (``p < _page``) are copied into smem slots
        ``[p*nbp, (p+1)*nbp)``, invalid pages are zero-filled so the
        compute path can read uniformly. See
        :func:`_emit_multipage_load`.

    Thread mapping: ``blockDim.x = 256`` threads stride through the
    flat element index ``j``. The strided loop ``for (j = tid; j < N;
    j += blockDim.x)`` is uniform across all ``workload_chunk_size``
    values — chunk in [16, 32, 64, 128] only scales ``N`` and therefore
    the per-thread iteration count.

    No ``__syncthreads()`` is emitted at the end of the load section.
    The next consumer of smem syncs itself: every Schedule.W compute
    op (Elementwise / Reduce / GeMM / ...) starts with
    ``__syncthreads()``, and :func:`generate_store_tensor_str` does
    the same. Cross-iteration safety is covered by the store's
    end-sync that runs just before the next iteration's load.
    """
    chunk = ctx.workload_chunk_size
    npw   = ctx.num_pages_per_workload
    nbp   = ctx.num_blocks_per_page

    batched: List[Tuple[int, object]] = []
    paged:   List[Tuple[int, object]] = []
    ragged:  List[Tuple[int, object]] = []
    for tid in sub_graph.input_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.BATCHED:
            batched.append((tid, t))
        elif t._format == FORMAT.PAGED:
            paged.append((tid, t))
        elif t._format == FORMAT.RAGGED:
            ragged.append((tid, t))

    if not (batched or paged or ragged):
        return "// No tensor loading required"

    sections: List[str] = []

    # ---- 1. Per-format index loads (each emitted at most once) -------
    idx_lines: List[str] = []
    if batched:
        idx_lines.append("const int new_batch_idx_i32 = winfo_x_indices[i];")
    if paged or ragged:
        idx_lines.append("const int ragged_idx_i32 = winfo_y_offsets[i];")
    if paged and npw == 1:
        idx_lines.append("const int page_idx_i32 = indices[ragged_idx_i32];")
    if idx_lines:
        sections.append("\n".join(idx_lines))

    # ---- 2. Per-tensor load blocks -----------------------------------
    for tid, t in batched:
        block_elems = t.shape[1] * t.shape[2]
        sections.append(_emit_contig_load(
            tid, _cpp_kernel_value_type(t.dtype),
            src_offset_expr=f"new_batch_idx_i32 * {block_elems}",
            n_elems=block_elems,
            elem_bytes=_dtype_bytes(t.dtype),
            block_elems=block_elems,
        ))

    for tid, t in ragged:
        block_elems = t.shape[1] * t.shape[2]
        sections.append(_emit_contig_load(
            tid, _cpp_kernel_value_type(t.dtype),
            src_offset_expr=f"ragged_idx_i32 * {block_elems}",
            n_elems=chunk * block_elems,
            elem_bytes=_dtype_bytes(t.dtype),
            block_elems=block_elems,
        ))

    for tid, t in paged:
        block_elems = t.shape[1] * t.shape[2]
        cuda_t = _cpp_kernel_value_type(t.dtype)
        if npw == 1:
            sections.append(_emit_contig_load(
                tid, cuda_t,
                src_offset_expr=f"page_idx_i32 * {block_elems}",
                n_elems=chunk * block_elems,
                elem_bytes=_dtype_bytes(t.dtype),
                block_elems=block_elems,
            ))
        else:
            sections.append(_emit_multipage_load(
                tid, cuda_t, npw=npw, nbp=nbp, block_elems=block_elems,
                elem_bytes=_dtype_bytes(t.dtype),
                zero_expr=cast_float_to_smem("0.0f", t.dtype),
            ))

    # Trailing ``__pipeline_commit()`` commits every cp.async issued
    # in the per-tensor blocks above into a single pipeline group.
    # The first compute op pairs this with a ``__pipeline_wait_prior(0)``
    # — see :func:`generate_computation_str`. Sync-load tiles that
    # bypass cp.async (scalar fallback paths or zero-fill in the
    # multi-page invalid branch) sit in the same group; the commit
    # treats them as a no-op for the async pipeline, and their writes
    # are flushed by the same downstream ``__syncthreads()``.
    sections.append("__pipeline_commit();")
    return "\n\n".join(sections)


def _emit_contig_store(
    tid: int, cuda_t: str, dst_offset_expr: str, n_elems_expr: str,
    elem_bytes: int, n_elems_const: int = 0, block_elems: int = 0,
) -> str:
    """Strided block-cooperative copy of ``n_elems_expr`` contiguous
    elements from ``tensor_<tid>_smem`` (flattened) to
    ``tensor_<tid>_ptr + dst_offset_expr``.

    Inverse of :func:`_emit_contig_load`. Used for BATCHED, RAGGED, and
    PAGED single-page stores — all three are a single contiguous span
    in global memory and differ only in ``dst_offset_expr`` and the
    element count (RAGGED uses ``_len * block_elems`` to skip the
    trailing invalid blocks; BATCHED uses ``block_elems`` because each
    batch slot is exactly one row; PAGED single-page uses
    ``chunk * block_elems`` since the page is always fully populated).

    Vectorisation policy:

      * BATCHED / PAGED single-page — ``n_elems_const > 0`` is supplied
        and the loop iterates a compile-time-known number of elements;
        delegate to :func:`_emit_vector_copy_loop`, which folds the
        constant into the ``int4`` / ``int2`` widening decision.
      * RAGGED — ``n_elems_expr`` is ``_len * block_elems``, a runtime
        value. We still vectorise across the chunks (``block_elems``
        boundary is the same per-row constant from the load side, so
        the same alignment guarantees hold), and the loop iterates
        ``_len * (block_elems / lane)`` vector slots.

    ``block_elems`` is required for the RAGGED path so the runtime
    ``_len`` factor can be combined with the compile-time per-row
    vector lane count.
    """
    if n_elems_const > 0:
        # Constant span (BATCHED / PAGED single-page). The dst stride
        # per workload index is ``block_elems * elem_bytes`` — the same
        # per-row stride the load side uses. For the BATCHED case
        # callers pass ``block_elems == n_elems_const`` (block_elems is
        # exactly one row); for PAGED single-page they pass
        # ``block_elems`` itself. Default to the span length when the
        # caller didn't supply a per-index stride (already aligned).
        stride_bytes = (block_elems if block_elems > 0 else n_elems_const) * elem_bytes
        body = _emit_vector_copy_loop(
            dst_ptr_expr="dst",
            src_ptr_expr="src",
            n_elems=n_elems_const,
            elem_bytes=elem_bytes,
            indent_level=1,
            stride_bytes=stride_bytes,
        )
        return (
            f"{{\n"
            f"{INDENT}{cuda_t}* dst = tensor_{tid}_ptr + {dst_offset_expr};\n"
            f"{INDENT}const auto* src = &tensor_{tid}_smem[0][0][0];\n"
            f"{body}\n"
            f"}}"
        )

    # RAGGED — vectorise inside the constant ``block_elems`` factor and
    # leave ``_len`` as the runtime loop bound on the vector count.
    assert block_elems > 0, "RAGGED store requires block_elems for vec lane math"
    vec_bytes = _pick_vec_bytes(block_elems, elem_bytes, block_elems * elem_bytes)
    if vec_bytes <= elem_bytes:
        return (
            f"{{\n"
            f"{INDENT}{cuda_t}* dst = tensor_{tid}_ptr + {dst_offset_expr};\n"
            f"{INDENT}const auto* src = &tensor_{tid}_smem[0][0][0];\n"
            f"{INDENT}for (int j = tid; j < {n_elems_expr}; j += blockDim.x) {{\n"
            f"{INDENT}{INDENT}dst[j] = src[j];\n"
            f"{INDENT}}}\n"
            f"}}"
        )
    vec_ty = _VEC_BYTES_TO_CTYPE[vec_bytes]
    lane = vec_bytes // elem_bytes
    per_row_vec = block_elems // lane
    return (
        f"{{\n"
        f"{INDENT}{cuda_t}* dst = tensor_{tid}_ptr + {dst_offset_expr};\n"
        f"{INDENT}const auto* src = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}{vec_ty}* __dst_v = reinterpret_cast<{vec_ty}*>(dst);\n"
        f"{INDENT}const {vec_ty}* __src_v = reinterpret_cast<const {vec_ty}*>(src);\n"
        f"{INDENT}const int __n_vec = _len * {per_row_vec};\n"
        f"{INDENT}for (int j = tid; j < __n_vec; j += blockDim.x) {{\n"
        f"{INDENT}{INDENT}__dst_v[j] = __src_v[j];\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def _emit_multipage_store(
    tid: int, cuda_t: str, npw: int, nbp: int, block_elems: int,
    elem_bytes: int,
) -> str:
    """Page-by-page scatter for PAGED outputs spanning multiple pages.

    Inverse of :func:`_emit_multipage_load`. For each page ``p`` in
    ``[0, num_pages_per_workload)`` with ``p < _page``: look up the
    start-of-page block index via
    ``indices[ragged_idx_i32 + p * num_blocks_per_page]``, then copy
    ``num_blocks_per_page * block_elems`` contiguous smem elements
    (from chunk slots ``[p*nbp, (p+1)*nbp)``) into that page in global
    memory.

    Invalid pages (``p >= _page``) are simply skipped — unlike the
    load side, there's nothing to zero-fill since global memory there
    is owned by the page allocator and won't be read at this index.
    """
    page_elems = nbp * block_elems
    # Both dst (global at block_idx*block_elems) and src (smem at p*page_elems)
    # advance by ``block_elems * elem_bytes`` per index step. The global side
    # is the tighter constraint since smem is already 16-byte aligned.
    page_copy = _emit_vector_copy_loop(
        dst_ptr_expr="dst",
        src_ptr_expr=f"src + p * {page_elems}",
        n_elems=page_elems,
        elem_bytes=elem_bytes,
        indent_level=3,
        stride_bytes=block_elems * elem_bytes,
    )
    return (
        f"{{\n"
        f"{INDENT}const auto* src = &tensor_{tid}_smem[0][0][0];\n"
        f"{INDENT}for (int p = 0; p < {npw}; ++p) {{\n"
        f"{INDENT}{INDENT}if (p < _page) {{\n"
        f"{INDENT}{INDENT}{INDENT}const int block_idx = "
        f"indices[ragged_idx_i32 + p * {nbp}];\n"
        f"{INDENT}{INDENT}{INDENT}{cuda_t}* dst = "
        f"tensor_{tid}_ptr + block_idx * {block_elems};\n"
        f"{page_copy}\n"
        f"{INDENT}{INDENT}}}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def generate_store_tensor_str(sub_graph: Graph, ctx: Context) -> str:
    """Cooperatively write subgraph **outputs** back from smem to global.

    Iterates :attr:`Graph.output_tensor_ids` — a Load+Save tensor sits
    in both input and output sets; the smem tile holds whatever the
    compute path produced and gets flushed here, possibly via the same
    addressing the load side used to populate it.

    Output shape per format:

      * **BATCHED** — gated by ``_is_first_workload = winfo_is_first_
        workload_per_batch[i]``. Only one workload per ``(batch, head)``
        commits the accumulator to global; the rest of the workloads
        for that slot already added their partial contributions into
        the same smem accumulator. ``shape[1] * shape[2]`` contiguous
        elements written to ``tensor_<id>_ptr + new_batch_idx_i32 *
        (shape[1] * shape[2])``. The gate is block-uniform so all
        threads take the same branch — no warp divergence.
      * **RAGGED** — ``_len * shape[1] * shape[2]`` contiguous elements
        written to ``tensor_<id>_ptr + ragged_idx_i32 *
        (shape[1] * shape[2])``. The bound is ``_len`` (not ``chunk``)
        so the trailing invalid blocks in a last-workload are skipped;
        this replaces Triton's per-lane ``valid`` mask vector.
      * **PAGED single-page** (``num_pages_per_workload == 1``) —
        ``chunk * shape[1] * shape[2]`` contiguous elements written to
        ``tensor_<id>_ptr + page_idx_i32 * (shape[1] * shape[2])``.
        No mask; the page is always fully populated for a single-page
        workload.
      * **PAGED multi-page** (``num_pages_per_workload > 1``) — see
        :func:`_emit_multipage_store`. Per-page scatter with a
        ``p < _page`` skip on invalid pages.

    Scoping: the store body lives in its own ``{ ... }`` block. That
    isolates store's index-load declarations (``new_batch_idx_i32``,
    ``ragged_idx_i32``, ``page_idx_i32``, ``_is_first_workload``) from
    :func:`generate_load_tensor_str`'s identically-named declarations
    at iteration scope. nvcc CSEs the duplicate ``winfo_*[i]`` loads
    so the source-level duplication has no runtime cost. The wrapping
    scope also means store can run even when load is empty (no
    subgraph inputs).

    Syncs: one ``__syncthreads()`` before the scope (compute may have
    written smem, store reads it) and one after (store reads smem,
    next iteration's load will write the same slots). Both are
    correctness-critical for compute paths that do cross-thread smem
    reductions; the cost is two block barriers per workload iteration.
    """
    chunk = ctx.workload_chunk_size
    npw   = ctx.num_pages_per_workload
    nbp   = ctx.num_blocks_per_page

    batched: List[Tuple[int, object]] = []
    paged:   List[Tuple[int, object]] = []
    ragged:  List[Tuple[int, object]] = []
    for tid in sub_graph.output_tensor_ids:
        t = sub_graph.tensor_list[tid]
        if t._format == FORMAT.BATCHED:
            batched.append((tid, t))
        elif t._format == FORMAT.PAGED:
            paged.append((tid, t))
        elif t._format == FORMAT.RAGGED:
            ragged.append((tid, t))

    if not (batched or paged or ragged):
        return "// No tensor storing required"

    sections: List[str] = []

    # ---- 1. Per-format index loads (own scope; isolated from load) ---
    idx_lines: List[str] = []
    if batched:
        idx_lines.append("const int new_batch_idx_i32 = winfo_x_indices[i];")
        idx_lines.append(
            "const uint8_t _is_first_workload = "
            "winfo_is_first_workload_per_batch[i];"
        )
    if paged or ragged:
        idx_lines.append("const int ragged_idx_i32 = winfo_y_offsets[i];")
    if paged and npw == 1:
        idx_lines.append("const int page_idx_i32 = indices[ragged_idx_i32];")
    if idx_lines:
        sections.append("\n".join(idx_lines))

    # ---- 2. BATCHED stores, all gated by _is_first_workload ---------
    if batched:
        batched_blocks: List[str] = []
        for tid, t in batched:
            block_elems = t.shape[1] * t.shape[2]
            batched_blocks.append(_emit_contig_store(
                tid, _cpp_kernel_value_type(t.dtype),
                dst_offset_expr=f"new_batch_idx_i32 * {block_elems}",
                n_elems_expr=str(block_elems),
                elem_bytes=_dtype_bytes(t.dtype),
                n_elems_const=block_elems,
                block_elems=block_elems,
            ))
        sections.append(
            "if (_is_first_workload) {\n"
            f"{indent_block(chr(10).join(batched_blocks), 1)}\n"
            "}"
        )

    # ---- 3. RAGGED stores, bound by _len -----------------------------
    for tid, t in ragged:
        block_elems = t.shape[1] * t.shape[2]
        sections.append(_emit_contig_store(
            tid, _cpp_kernel_value_type(t.dtype),
            dst_offset_expr=f"ragged_idx_i32 * {block_elems}",
            n_elems_expr=f"_len * {block_elems}",
            elem_bytes=_dtype_bytes(t.dtype),
            block_elems=block_elems,
        ))

    # ---- 4. PAGED stores --------------------------------------------
    for tid, t in paged:
        block_elems = t.shape[1] * t.shape[2]
        cuda_t = _cpp_kernel_value_type(t.dtype)
        if npw == 1:
            sections.append(_emit_contig_store(
                tid, cuda_t,
                dst_offset_expr=f"page_idx_i32 * {block_elems}",
                n_elems_expr=str(chunk * block_elems),
                elem_bytes=_dtype_bytes(t.dtype),
                n_elems_const=chunk * block_elems,
                block_elems=block_elems,
            ))
        else:
            sections.append(_emit_multipage_store(
                tid, cuda_t, npw=npw, nbp=nbp, block_elems=block_elems,
                elem_bytes=_dtype_bytes(t.dtype),
            ))

    body = "\n\n".join(sections)
    body_indented = indent_block(body, 1)
    return (
        f"__syncthreads();\n"
        f"{{\n"
        f"{body_indented}\n"
        f"}}\n"
        f"__syncthreads();"
    )


_LEADING_SYNC_RE = re.compile(r"^\s*__syncthreads\s*\(\s*\)\s*;\s*\n")


def generate_computation_str(sub_graph: Graph, ctx: Context) -> str:
    """Emit per-op computation source, with data-dependency-driven
    ``__syncthreads()`` placement.

    Each op codegen emits its own leading ``__syncthreads();\\n`` at the
    top of its body — kept that way so single-op codegens remain
    correct in isolation (for tests / standalone use). Here we strip
    those leading syncs and *re-insert* them only between op pairs
    whose data flow actually requires a barrier.

    Sync placement rule. Maintain ``dirty`` = the set of tensor ids
    written since the last barrier. Before each op, if its inputs
    intersect ``dirty``, emit a sync and clear ``dirty``. Add the op's
    output(s) to ``dirty`` afterwards.

    Initial state. The load section writes inputs across threads
    (different threads fill different elements of each tile), so all
    subgraph inputs start out in ``dirty``. The first compute op that
    reads any input therefore picks up its own leading sync — same as
    the previous always-emit policy. The store section keeps its own
    leading/trailing syncs (see :func:`generate_store_tensor_str`), so
    we don't have to flush ``dirty`` here.

    The optimisation matters on flows like ``masked_quest`` where
    consecutive ``Multiply`` ops share inputs but don't read each
    other's outputs — e.g. ``q*max`` and ``q*min`` are independent and
    only ``Maximum`` later reads both. The old codegen synced before
    every op (8 syncs for masked_quest's 8 W-ops); the new one syncs
    only when needed, typically 4-5.
    """
    if not sub_graph.op_list:
        return "// No computation required"

    op_sources = [
        get_impl_func(op)(sub_graph, op_id, ctx)
        for op_id, op in enumerate(sub_graph.op_list)
    ]
    # Each per-op codegen emits a leading ``__syncthreads();\n`` (kept
    # for standalone correctness). Strip it so we control sync emission
    # centrally below.
    op_bodies = [_LEADING_SYNC_RE.sub("", src, count=1) for src in op_sources]

    # Post-load barrier — unconditional. ``__pipeline_wait_prior(0)``
    # only drains *this* thread's cp.async copies. With smem reuse
    # (see :func:`_allocate_smem_offsets`) a later intermediate may
    # share its byte range with a dead input that another thread is
    # still cp.async-filling — the ``__syncthreads()`` here makes
    # those cross-thread cp.async writes globally visible before any
    # op touches a reused slot.
    out_pieces: List[str] = [
        "__pipeline_wait_prior(0);",
        "__syncthreads();",
    ]
    syncs_kept = 1  # the unconditional post-load sync
    syncs_dropped = 0

    # Pull smem offsets/sizes that the allocator stashed on the
    # subgraph. They let us decide overlap by byte range instead of
    # tensor id — required for correctness under smem reuse where two
    # different tensor ids point at the same offset.
    offsets = sub_graph._smem_offsets  # type: ignore[attr-defined]
    sizes = sub_graph._smem_sizes      # type: ignore[attr-defined]

    def _byte_range(tid: int):
        return (offsets[tid], offsets[tid] + sizes[tid])

    def _ranges_overlap(a, b) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    # ``written_since_sync`` — tids whose smem range was written since
    # the last barrier. Reads of an overlapping range need a sync to
    # see the writes.
    # ``touched_since_sync`` — tids whose smem range was either read
    # or written. Writes to an overlapping range need a sync to avoid
    # races with the in-flight reads/writes from prior ops (this is
    # the case smem reuse makes possible — see the rationale above).
    written_since_sync: set = set()
    touched_since_sync: set = set()

    for op_id, body in enumerate(op_bodies):
        inputs = set(sub_graph.op_to_input_tensor_list[op_id])
        outputs = set(sub_graph.op_to_output_tensor_list[op_id])

        need_sync = False
        # 1. Read-after-write: this op's inputs overlap a tile written
        #    since the last sync.
        for tin in inputs:
            r_in = _byte_range(tin)
            for tdw in written_since_sync:
                if _ranges_overlap(r_in, _byte_range(tdw)):
                    need_sync = True
                    break
            if need_sync:
                break
        # 2. Write-after-anything: this op's outputs overlap a tile
        #    touched (read or written) since the last sync. Catches
        #    the smem-reuse race where a new intermediate shares a
        #    dead input's slot but other threads are still reading
        #    that input.
        if not need_sync:
            for tout in outputs:
                r_out = _byte_range(tout)
                for tt in touched_since_sync:
                    if _ranges_overlap(r_out, _byte_range(tt)):
                        need_sync = True
                        break
                if need_sync:
                    break

        if need_sync:
            out_pieces.append("__syncthreads();")
            written_since_sync.clear()
            touched_since_sync.clear()
            syncs_kept += 1
        else:
            syncs_dropped += 1

        out_pieces.append(body)
        written_since_sync |= outputs
        touched_since_sync |= inputs
        touched_since_sync |= outputs

    # Stash counts so reviewers inspecting the generated TU can see
    # how aggressive the sync placement ended up being.
    sub_graph._syncs_kept = syncs_kept       # type: ignore[attr-defined]
    sub_graph._syncs_dropped = syncs_dropped # type: ignore[attr-defined]
    return "\n\n".join(out_pieces)


# --------------------------------------------------------------------------- #
# Kernel definition (W-scheduled)
# --------------------------------------------------------------------------- #

def _build_kernel_signature(sub_graph) -> List[str]:
    """Return kernel-argument declarations (CUDA C++), in declaration order.

    Order is 1:1 with :func:`generate_cuda_launcher`'s kernel-call
    expression list — that ordering is the ABI contract between the
    host launcher and the device kernel.

    Param qualifiers:

      * Planner pointers (``indices``, ``winfo_*``) are
        ``const T* __restrict__`` — read-only across the kernel, so
        nvcc can route loads through the read-only / texture path.
      * Per-tensor pointers are ``T* __restrict__``. Input-only tensors
        get an extra ``const``; output tensors (and Load+Save tensors,
        which :func:`_iter_kernel_tensors` yields with role
        ``'output'``) drop the ``const`` so writes are allowed.
      * ``tensor_<id>_dim0`` is ``int64_t`` to match the ``int64_t``
        arg type emitted by the launcher — no narrowing at the kernel
        call site.

    Each list entry is a single parameter **without** a trailing
    comma; the kernel template at :func:`generate_cuda_kernel` joins
    with ``",\\n"`` so the last param has no dangling comma.
    """
    args: List[str] = [
        "const int32_t* __restrict__ indices",
        "const int32_t* __restrict__ winfo_x_indices",
    ]
    if _has_batched_output(sub_graph):
        args.append("const uint8_t* __restrict__ winfo_is_first_workload_per_batch")
    args += [
        "const int32_t* __restrict__ winfo_y_offsets",
        "const int32_t* __restrict__ winfo_y_lens",
        "const int32_t* __restrict__ winfo_num_workloads",
    ]
    for tid, role in _iter_kernel_tensors(sub_graph):
        t = sub_graph.tensor_list[tid]
        cuda_t = _cpp_kernel_value_type(t.dtype)
        const_q = "const " if role == "input" else ""
        args.append(f"{const_q}{cuda_t}* __restrict__ tensor_{tid}_ptr")
        # ``tensor_<id>_dim0`` used to land here as an ``int64_t`` mirror
        # of the wrapper's ``tensor.shape[0]``, intended for bounds
        # checks. No op codegen ever read it, so it was just consuming
        # a kernel-arg slot (constant memory + an extra register on the
        # cudaLaunchKernel marshalling path). Dropped.
    return args


def _build_workload_preamble(ctx: Context) -> str:
    """Per-iteration ``_len`` / ``_page`` declarations (CUDA C++).

    Emitted at the top of every iteration of the workload loop in the
    kernel body. Two scalar locals, a deliberate subset of Triton's
    four-line preamble:

      * ``_len``  — number of valid blocks in this workload, loaded
        from ``winfo_y_lens[i]``. Every thread loads the same value
        and nvcc keeps it in a register.
      * ``_page`` — number of pages those blocks span; ceil-divide
        ``(_len + num_blocks_per_page - 1) / num_blocks_per_page`` with
        ``num_blocks_per_page`` baked in as a literal so the divide
        folds at compile time.

    Triton additionally emits two per-lane masks (``valid`` of length
    ``workload_chunk_size`` and ``page_valid`` of length
    ``num_pages_per_workload``). The CUDA backend doesn't precompute
    those — the load / store body codegens evaluate per-block validity
    inline against ``_len`` / ``_page`` when they do the actual
    addressing. Keeping the preamble scalar-only lets every thread run
    it without coordination.
    """
    nbp = ctx.num_blocks_per_page
    return "\n".join([
        "const int _len  = winfo_y_lens[i];",
        f"const int _page = (_len + {nbp - 1}) / {nbp};",
    ])


def generate_cuda_kernel(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Emit the ``__global__`` kernel CUDA C++ source for a W-scheduled
    subgraph (kernel definition only — the host-side launcher comes
    from :func:`generate_cuda_launcher`).

    Template structure (mirrors the Triton kernel one-to-one):

      <preamble: #include directives shared with the launcher>

      __global__ void <name>_subgraph_<id>_kernel(
          <signature from _build_kernel_signature>
      ) {
          // Block-level partitioning of workloads. Each block claims
          // [start, end) — same per/r split Triton uses.
          const int pid       = blockIdx.x;
          const int num_progs = gridDim.x;
          const int tid       = threadIdx.x;

          const int n_workloads = winfo_num_workloads[0];

          const int per   = n_workloads / num_progs;
          const int r     = n_workloads % num_progs;
          const int start = pid * per + (pid < r ? pid : r);
          const int end   = start + per + (pid < r ? 1 : 0);

          <initialization_str>

          for (int i = start; i < end; ++i) {
              <prepare_workload_str>
              <load_tensor_str>
              <computation_str>
              <store_tensor_str>
          }
      }

    Body fragments come from the helpers below (currently ``pass``):

      * :func:`_build_kernel_signature`  — kernel parameter list.
      * :func:`generate_initialization_str` — pre-loop scratch decls.
      * :func:`_build_workload_preamble`    — per-iteration mask setup.
      * :func:`generate_load_tensor_str`    — input loads.
      * :func:`generate_computation_str`    — op computations.
      * :func:`generate_store_tensor_str`   — output stores.

    Until those land, calling this function will raise ``TypeError`` /
    ``AttributeError`` from the ``pass`` returns — the template body is
    here so subsequent steps can wire each helper in independently.
    """
    sa_name = ctx.sparse_attention_name
    kernel_fn = f"{sa_name}_subgraph_{sub_graph_id}_kernel"

    kernel_args = ",\n".join(
        f"{INDENT}{arg}" for arg in _build_kernel_signature(sub_graph)
    )

    initialization_str   = indent_block(generate_initialization_str(sub_graph, ctx), 1)
    prepare_workload_str = indent_block(_build_workload_preamble(ctx),               2)
    load_tensor_str      = indent_block(generate_load_tensor_str(sub_graph, ctx),    2)
    computation_str      = indent_block(generate_computation_str(sub_graph, ctx),    2)
    store_tensor_str     = indent_block(generate_store_tensor_str(sub_graph, ctx),   2)

    # ``__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`` tells
    # nvcc the actual block_size we launch with so it can pick a
    # register budget tuned for that occupancy target instead of the
    # default "assume 1024 threads/block" pessimistic budget. Block
    # size is fixed at 256 in :func:`generate_cuda_launcher`. The
    # minBlocksPerSM hint biases nvcc towards higher occupancy when
    # the smem budget allows (typical case post-Iter-1) and is a
    # no-op when smem alone already throttles occupancy.
    #
    # An Iter-8 attempt to drop minBlocksPerSM (let nvcc decide
    # per-kernel) measured mixed results — wins and losses cancelled
    # to roughly zero across the set, with one clear regression on
    # ``venergy_gated_centroid__trtllm`` (-10 %). Reverted; (256, 2)
    # stays as the verified setting that delivered Pass 2's +17.8 %
    # geomean.
    LAUNCH_BLOCK_SIZE = 256
    MIN_BLOCKS_PER_SM = 2
    return (
        f"{_CUDA_KERNEL_PREAMBLE}\n"
        f"\n"
        f"\n"
        f"__global__ "
        f"__launch_bounds__({LAUNCH_BLOCK_SIZE}, {MIN_BLOCKS_PER_SM}) "
        f"void {kernel_fn}(\n"
        f"{kernel_args}\n"
        f") {{\n"
        f"{INDENT}// ------------------------------------------------------------\n"
        f"{INDENT}// Block-level partitioning of workloads (mirrors the Triton\n"
        f"{INDENT}// kernel's per/r/start/end split: each block claims a\n"
        f"{INDENT}// contiguous span of [start, end) workloads).\n"
        f"{INDENT}// ------------------------------------------------------------\n"
        f"{INDENT}const int pid       = static_cast<int>(blockIdx.x);\n"
        f"{INDENT}const int num_progs = static_cast<int>(gridDim.x);\n"
        f"{INDENT}const int tid       = static_cast<int>(threadIdx.x);\n"
        f"\n"
        f"{INDENT}const int n_workloads = winfo_num_workloads[0];\n"
        f"\n"
        f"{INDENT}const int per   = n_workloads / num_progs;\n"
        f"{INDENT}const int r     = n_workloads % num_progs;\n"
        f"{INDENT}const int start = pid * per + (pid < r ? pid : r);\n"
        f"{INDENT}const int end   = start + per + (pid < r ? 1 : 0);\n"
        f"\n"
        f"{initialization_str}\n"
        f"\n"
        f"{INDENT}for (int i = start; i < end; ++i) {{\n"
        f"{prepare_workload_str}\n"
        f"{load_tensor_str}\n"
        f"{computation_str}\n"
        f"{store_tensor_str}\n"
        f"{INDENT}}}\n"
        f"}}"
    )


def _build_cuda_launcher_params(sub_graph) -> List[str]:
    """Return the C++ parameter list (one entry per param, no commas)
    of the host-side launcher emitted by :func:`generate_cuda_launcher`.

    Factored out so :func:`_generate_w_impl` can re-use the same list to
    emit a matching ``void <fn>(<params>);`` forward declaration in the
    ``cpp_sources`` arg of ``load_inline`` — pybind11 needs the symbol
    visible at link time when it generates the ``m.def(...)`` binding,
    so the declaration has to land in the host-compiled TU, not the
    nvcc-compiled one.
    """
    params: List[str] = [
        "torch::Tensor indices",
        "torch::Tensor winfo_x_indices",
    ]
    if _has_batched_output(sub_graph):
        params.append("torch::Tensor winfo_is_first_workload_per_batch")
    params += [
        "torch::Tensor winfo_y_offsets",
        "torch::Tensor winfo_y_lens",
        "torch::Tensor winfo_num_workloads",
    ]
    for tid, _role in _iter_kernel_tensors(sub_graph):
        params.append(f"torch::Tensor tensor_{tid}")
        # ``int64_t tensor_<id>_dim0`` dropped in lockstep with
        # ``_build_kernel_signature`` — no consumer reads it.
    return params


def generate_cuda_launcher(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Emit the host-side C++ launcher source for a W-scheduled subgraph.

    Signature mirrors the kernel signature emitted by
    :func:`generate_cuda_kernel`:

      * planner pointers — ``indices``, ``winfo_x_indices``,
        (``winfo_is_first_workload_per_batch`` iff BATCHED outputs),
        ``winfo_y_offsets``, ``winfo_y_lens``, ``winfo_num_workloads``.
        All are ``int32`` tensors except the per-batch flag which is
        ``uint8``.
      * per kernel tensor: a ``torch::Tensor`` arg + an ``int64_t`` leading
        dim. The leading dim's role matches Triton's ``tensor_X_dim0``
        ``tl.constexpr`` parameter; in CUDA we just pass it at runtime.

    The body launches the kernel with a 1-D grid of ``num_sms * 4``
    blocks (mirrors the Triton wrapper's ``(ctx.num_sms * 4,)`` grid)
    and 256 threads per block (8 warps × 32 threads, mirroring Triton's
    ``num_warps=8`` tuning). The kernel reads ``winfo_num_workloads[0]``
    itself and partitions the workload range across blocks, so the
    launcher does **not** sync to read the workload count on the host.

    The launcher uses ``at::cuda::getCurrentCUDAStream()`` —
    :func:`generate_cuda_kernel` must include
    ``<ATen/cuda/CUDAContext.h>`` (and the half-precision headers
    ``<cuda_fp16.h>`` / ``<cuda_bf16.h>``) in its preamble.

    The exposed function name matches the entry put in
    ``functions=[...]`` of the ``load_inline`` call emitted by
    :func:`_generate_w_impl`.
    """
    sa_name     = ctx.sparse_attention_name
    kernel_fn   = f"{sa_name}_subgraph_{sub_graph_id}_kernel"
    launcher_fn = f"{sa_name}_subgraph_{sub_graph_id}_launcher"

    # --- launcher parameters (host-side C++ signature) -----------------
    params = _build_cuda_launcher_params(sub_graph)

    # --- kernel-call arguments (device-side pointers + scalars) --------
    kcall: List[str] = [
        "indices.data_ptr<int32_t>()",
        "winfo_x_indices.data_ptr<int32_t>()",
    ]
    if _has_batched_output(sub_graph):
        kcall.append("winfo_is_first_workload_per_batch.data_ptr<uint8_t>()")
    kcall += [
        "winfo_y_offsets.data_ptr<int32_t>()",
        "winfo_y_lens.data_ptr<int32_t>()",
        "winfo_num_workloads.data_ptr<int32_t>()",
    ]
    for tid, _role in _iter_kernel_tensors(sub_graph):
        t = sub_graph.tensor_list[tid]
        kcall.append(_cpp_data_ptr_expr(f"tensor_{tid}", t.dtype))
        # ``tensor_<id>_dim0`` dropped — see :func:`_build_kernel_signature`.

    # Grid + block: bake as ``constexpr int`` literals at codegen time so
    # the compiled SASS specializes on the host's SM count. The kernel
    # body itself reads ``gridDim.x`` / ``blockIdx.x`` to partition work.
    num_progs  = ctx.num_sms * 4
    block_size = 256  # 8 warps * 32 threads — matches Triton ``num_warps=8``

    # Total dynamic shared-memory bytes required by the kernel — set by
    # :func:`generate_initialization_str` when it laid out tiles into
    # ``extern __shared__``. Default to 0 for kernels with no tiles.
    smem_bytes = int(getattr(sub_graph, "_smem_bytes", 0))

    params_str = ",\n".join(f"{INDENT}{p}" for p in params)
    kcall_str  = ",\n".join(f"{INDENT * 2}{a}" for a in kcall)

    # Static __shared__ is capped at 48 KB. For dynamic shared memory we
    # must opt in via ``cudaFuncAttributeMaxDynamicSharedMemorySize`` when
    # the request exceeds that bound — on H100/B200 the per-block ceiling
    # is much higher (~228 KB) but is gated by this attribute.
    #
    # The attribute is a kernel-function property, not a per-launch flag,
    # so it's safe to set it exactly once per process. Wrap the call in
    # a static-init lambda — the first launch pays the host-side
    # ``cudaFuncSetAttribute`` (~1 µs + a CPU/GPU sync barrier) and
    # every subsequent launch reads a single bool. For long decode
    # loops this elides hundreds of redundant runtime calls.
    if smem_bytes > 48 * 1024:
        smem_setattr = (
            f"{INDENT}static const bool _attr_inited = [] {{\n"
            f"{INDENT}{INDENT}cudaFuncSetAttribute(\n"
            f"{INDENT}{INDENT}{INDENT}(const void*)&{kernel_fn},\n"
            f"{INDENT}{INDENT}{INDENT}cudaFuncAttributeMaxDynamicSharedMemorySize,\n"
            f"{INDENT}{INDENT}{INDENT}{smem_bytes});\n"
            f"{INDENT}{INDENT}return true;\n"
            f"{INDENT}}}();\n"
            f"{INDENT}(void)_attr_inited;\n"
        )
    else:
        smem_setattr = ""

    return (
        f"void {launcher_fn}(\n"
        f"{params_str}\n"
        f") {{\n"
        f"{INDENT}const auto stream = at::cuda::getCurrentCUDAStream();\n"
        f"{INDENT}constexpr int num_progs = {num_progs};\n"
        f"{INDENT}constexpr int block_size = {block_size};\n"
        f"{INDENT}constexpr int smem_bytes = {smem_bytes};\n"
        f"{smem_setattr}"
        f"{INDENT}{kernel_fn}<<<num_progs, block_size, smem_bytes, stream>>>(\n"
        f"{kcall_str}\n"
        f"{INDENT});\n"
        f"}}"
    )


# --------------------------------------------------------------------------- #
# Python launcher / impl wrapper
# --------------------------------------------------------------------------- #

def _build_launcher_args(sub_graph, ctx: Context) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(wrapper_args, launcher_call_args, fp8_rebind_lines)``.

    Mirrors :func:`triton_impl.kernel_gen._build_launcher_args` for the
    ``ctx``-side bindings — both go through
    :func:`indexer.compiler.backend.get_backend` so the indices source
    follows the active attention backend (flashinfer CSR vs. trtllm
    block-tables) and every dynamic buffer is read off ``ctx.metadata``.
    The per-tensor (``tensor_<id>``, ``tensor_<id>.shape[0]``) suffix is
    identical to the triton path.

    ``fp8_rebind_lines`` — any FP8 tensor is bitcast to ``uint8`` before
    being forwarded so the launcher's ``uint8_t*`` C++ signature lines up
    with the storage dtype (see ``_TORCH_TO_CPP_TYPES``).
    """
    from ..backend import get_backend

    wrapper_args: List[str] = []
    launcher_call_args: List[str] = [
        get_backend(ctx).indices_src,
        "ctx.metadata.winfo_q_indices",
    ]
    if _has_batched_output(sub_graph):
        launcher_call_args.append("ctx.metadata.winfo_is_first_workload_per_batch")
    launcher_call_args += [
        "ctx.metadata.winfo_kv_offsets",
        "ctx.metadata.winfo_kv_lens",
        "ctx.metadata.winfo_num_workloads",
    ]
    fp8_rebind: List[str] = []

    for tid, _role in _iter_kernel_tensors(sub_graph):
        name = f"tensor_{tid}"
        t = sub_graph.tensor_list[tid]
        wrapper_args.append(name)
        # ``tensor_<id>_dim0`` is no longer in the kernel signature
        # (see ``_build_kernel_signature``) — it was dead weight that
        # cost a kernel-arg slot per tensor. The launcher's C++ params
        # and kernel call drop the int64 in lock-step.
        launcher_call_args.append(name)
        if _is_fp8(t):
            fp8_rebind.append(f"{name} = {name}.view(torch.uint8)")

    wrapper_args.append("ctx")
    return wrapper_args, launcher_call_args, fp8_rebind


def _generate_w_impl(sub_graph: Graph, sub_graph_id: int, ctx: Context) -> str:
    """Generate a W-scheduled subgraph as a JIT-compiled CUDA extension.

    Output structure (concatenated into the per-subgraph module):

        _<name>_subgraph_<id>_cuda_src = r\"\"\"
            <cuda kernel + host launcher source>
        \"\"\"
        _<name>_subgraph_<id>_cuda_mod = load_inline(
            name=...,
            cpp_sources="",
            cuda_sources=_<name>_subgraph_<id>_cuda_src,
            functions=["<name>_subgraph_<id>_launcher"],
        )

        def <name>_subgraph_<id>_impl(tensor_X, ..., ctx):
            <fp8 rebinds>
            _<name>_subgraph_<id>_cuda_mod.<launcher>(
                <forwarded args>
            )

    The ``load_inline`` call runs at module-import time; ``cpp_extension``
    caches by source hash, so repeat imports are fast.
    """
    kernel_src = generate_cuda_kernel(sub_graph, sub_graph_id, ctx)
    launcher_src = generate_cuda_launcher(sub_graph, sub_graph_id, ctx)
    wrapper_args, launcher_call_args, fp8_rebind = _build_launcher_args(sub_graph, ctx)

    sa_name = ctx.sparse_attention_name
    impl_name   = f"{sa_name}_subgraph_{sub_graph_id}_impl"
    launcher_fn = f"{sa_name}_subgraph_{sub_graph_id}_launcher"
    mod_name    = f"{sa_name}_subgraph_{sub_graph_id}_cuda_mod"
    src_var     = f"_{sa_name}_subgraph_{sub_graph_id}_cuda_src"
    decl_var    = f"_{sa_name}_subgraph_{sub_graph_id}_cpp_decl"
    mod_var     = f"_{sa_name}_subgraph_{sub_graph_id}_cuda_mod"

    # Concatenate the kernel and host-launcher sources into a single
    # CUDA TU. Both must be compiled by nvcc (the launcher uses the
    # ``<<<...>>>`` triple-chevron syntax), so both go in
    # ``cuda_sources``.
    cuda_source = "\n\n".join(s for s in (kernel_src, launcher_src) if s)

    # ``cpp_sources`` carries a forward declaration of the launcher.
    # ``load_inline`` generates the pybind11 ``m.def`` binding for each
    # name in ``functions=[...]`` from the host-compiled TU, so the
    # symbol has to be declared *there* (the definition lives in the
    # nvcc-compiled TU and gets linked in via ninja).
    params_str = ",\n".join(f"{INDENT}{p}" for p in _build_cuda_launcher_params(sub_graph))
    cpp_decl = (
        f"#include <torch/extension.h>\n"
        f"#include <cstdint>\n"
        f"void {launcher_fn}(\n"
        f"{params_str}\n"
        f");"
    )

    args_def = ",\n".join(f"{INDENT}{a}" for a in wrapper_args)
    call_args = ",\n".join(f"{INDENT * 2}{a}" for a in launcher_call_args)
    fp8_block = indent_block("\n".join(fp8_rebind), 1) if fp8_rebind else ""

    impl_str = f'''
{src_var} = r"""
{cuda_source}
"""

{decl_var} = r"""
{cpp_decl}
"""

{mod_var} = load_inline(
    name={mod_name!r},
    cpp_sources={decl_var},
    cuda_sources={src_var},
    functions=[{launcher_fn!r}],
    verbose=False,
)


def {impl_name}(
{args_def},
{INDENT}cur_layer=0,
):
{fp8_block}
    {mod_var}.{launcher_fn}(
{call_args}
    )
'''
    return impl_str.strip()


def generate_cuda_impl(
    sub_graph: Graph,
    sub_graph_id: int,
    ctx: Context,
) -> str:
    """Generate the CUDA Schedule.W kernel and its Python wrapper.

    Schedule.S subgraphs are routed through
    :mod:`indexer.compiler.custom_impl` by
    :mod:`indexer.compiler.impl` and never reach this function.
    """
    assert sub_graph.schedule == Schedule.W, (
        f"generate_cuda_impl only handles Schedule.W; got {sub_graph.schedule}. "
        f"Schedule.S subgraphs are dispatched via indexer.compiler.impl."
    )
    ctx.compilation_header_lines.extend([
        "import torch",
        "from torch.utils.cpp_extension import load_inline",
    ])
    return _generate_w_impl(sub_graph, sub_graph_id, ctx)
