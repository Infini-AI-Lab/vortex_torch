"""INT4 KV store: layout metadata, capacity accounting, and the coalesced read.

One storage format, three placements
------------------------------------
The same packed representation serves GPU-only, host-KV and the GPU-cache-over-host tier, because
all three get their buffers from ``cache_meta_info`` in ``memory_pool._create_buffers`` and differ
only in *where* the tensors live and *who reads them*:

============  ==========================  ==============================================
placement     K/V tensors live            consumption
============  ==========================  ==============================================
gpu-only      HBM                         gather the selection into a bf16 scratch
host-kv       pinned host                 dequant the STAGED pool after fetch_kv
gpu-cache     pinned host + HBM pool      same as host-kv, with real eviction
============  ==========================  ==============================================

So this module owns the *format* -- shapes, dtypes, quantize, the read kernels -- and each
placement owns only its plumbing. Splitting it the other way (an INT4 variant per placement)
would triplicate the nibble arithmetic, which is exactly the kind of duplication that lets two
placements silently disagree about a layout.

Why consumption cannot be uniform
---------------------------------
flashinfer 0.6.14 *does* have a native 4-bit paged-KV path (uint8 cache with ``head_dim//2`` plus
``kv_cache_sf`` per-16-channel scales), which would let attention read INT4 directly. It is
**Blackwell-only**: on A100 (sm_80) a uint8 cache falls into the generic decode template and fails
to compile with ``vec_dtypes.cuh(117): no suitable conversion from DTypeKV to float``. Note that
``-DFLASHINFER_ENABLE_FP4_E2M1`` is defined and compiles for sm_80, and the ``fp8_e4m3`` scale
tensor allocates fine, so neither is evidence of support -- it was established by running a
decode. Hence :func:`native_int4_attention_supported`, and hence the dequant path being the
default rather than the fallback.

On pre-Blackwell the win is therefore **capacity and memory bandwidth, not attention
throughput**: attention still runs in bf16 and there is an extra dequant pass. That is the right
trade for enlarging KV, but it is not a decode speedup, and the dequant is affordable only because
sparse attention dequantizes the SELECTION (topk x block) rather than the context.

COALESCED STORES: interleave in registers with ``tl.join``, then write once
--------------------------------------------------------------------------
A read kernel unpacks byte ``d`` into channels ``2d`` and ``2d+1``. Storing those with two
separate stride-2 stores makes every transaction carry half-useful bytes. ``tl.join`` stacks the
halves on a new trailing axis, so ``join(lo, hi).reshape(T, head_dim)`` is exactly channel order
``2d, 2d+1`` -- i.e. NATURAL order, contiguous -- and the store becomes one unit-stride write.
Measured on 11904 entries, K only:

    two strided stores        1.241 ms      79 GB/s
    join + one contiguous     0.086 ms    1133 GB/s      14.4x, output bit-identical

The bit-identical part is what makes this safe, and it is why an earlier attempt failed: that one
got the same 14x by PERMUTING the output to split-half order, which is not allowed because ``q``
never passes through this cache (rel err 1.44 in isolation; and permuting ``q`` does not save it,
since V's channels ARE the output channels). RULER went to 0% on all placements. Here the channel
order never changes -- only the number and shape of the stores does.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from ...cache.triton_kernels.int4_kv import INT4_BIAS, quantize_block

logger = logging.getLogger(__name__)

#: Kernel-readable bias, sourced from the one definition in int4_kv so the pack and unpack sides
#: cannot drift apart. Must be the tl.constexpr(...) CALL form to be readable from a @jit'ed
#: function.
_BIAS = tl.constexpr(INT4_BIAS)

#: Scale field names. They are ordinary auxiliary cache fields, which is what makes the whole
#: scheme fit vortex: ``flow.py`` assigns a dtype PER FIELD, so K/V can be uint8 while their
#: scales are fp32 in the same cache dict -- no "two dtypes in one tensor".
K_SCALE = "k_int4_scale"
V_SCALE = "v_int4_scale"

#: bf16 staging field names, for blocks whose scale is not yet computable.
#:
#: These are REQUEST-domain fields, not page-domain ones, and that distinction is load-bearing: a
#: per-block bf16 mirror costs ``block_tokens x head_dim`` per block, measured at 21632 B/block
#: against bf16's 16896 -- a **1.28x REGRESSION** instead of the intended 3.2x saving. Sized by
#: concurrency instead, the staging area is constant in context length.
STAGE_K = "int4_stage_k"
STAGE_V = "int4_stage_v"


def native_int4_attention_supported() -> bool:
    """Can the attention kernel consume packed INT4 KV directly?

    True only on Blackwell-class parts (capability >= 12.0), where flashinfer's NVFP4 decode
    module exists. Everywhere else the caller must dequantize to bf16 first. Deliberately a
    capability check rather than a try/except around a compile: the failure mode is a multi-second
    ninja build ending in a C++ template error, which is not something to discover on the first
    decode step of a served request.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (12, 0)


def int4_cache_meta(block_tokens: int, head_dim: int) -> Dict[str, Tuple[Tuple[int, int],
                                                                        torch.dtype]]:
    """The PAGE-domain ``cache_meta_info`` entries INT4 needs.

    * ``k``/``v``  -> ``(block_tokens, head_dim // 2)`` uint8, two 4-bit values per byte.
    * ``k_int4_scale`` -> ``(1, head_dim)`` fp32: per-CHANNEL, shared by the block's tokens. This
      is precisely the shape ``Reduce(dim=1)`` produces, i.e. what the Quest flow already computes.
    * ``v_int4_scale`` -> ``(block_tokens, 1)`` fp32: per-TOKEN (``Reduce(dim=2)``).

    The asymmetry is measured, not stylistic -- see ``int4_kv.py``.
    """
    if head_dim % 2:
        raise ValueError(f"INT4 KV needs an even head_dim to pack two per byte, got {head_dim}")
    return {
        "k": ((block_tokens, head_dim // 2), torch.uint8),
        "v": ((block_tokens, head_dim // 2), torch.uint8),
        K_SCALE: ((1, head_dim), torch.float32),
        V_SCALE: ((block_tokens, 1), torch.float32),
    }


def int4_request_cache_meta(block_tokens: int, head_dim: int) -> Dict[str, Tuple[int, int]]:
    """The REQUEST-domain (staging) fields, for ``create_request_cache`` to return.

    Full-width bf16, because the point of the staging area is that these values have never been
    quantized -- the scale that would quantize them does not exist until the block is complete.
    """
    return {STAGE_K: (block_tokens, head_dim), STAGE_V: (block_tokens, head_dim)}


def bytes_per_block(block_tokens: int, head_dim: int) -> int:
    """Stored PAGE-domain bytes per block for K+V including scale overhead.

    Used for capacity accounting and to state the real compression honestly: the nominal 4x is
    diluted by the fp32 scales, and a caller sizing a pool must budget the true number.
    """
    packed = 2 * block_tokens * (head_dim // 2)            # K + V, 1 byte per 2 values
    scales = 4 * head_dim + 4 * block_tokens               # fp32 per-channel + per-token
    return packed + scales


def compression_ratio(block_tokens: int, head_dim: int) -> float:
    """bf16 bytes / INT4 bytes for one block of K+V, scales included."""
    bf16 = 2 * block_tokens * head_dim * 2
    return bf16 / bytes_per_block(block_tokens, head_dim)


def quantize_into(dst_k, dst_v, dst_ks, dst_vs, src_k, src_v) -> None:
    """Quantize one complete block from bf16 into the packed cache, in place.

    ``src_k``/``src_v`` are ``[block_tokens, head_dim]`` bf16 -- normally a staging slot, i.e.
    values that have never been quantized.
    """
    pk, ks = quantize_block(src_k, per_channel=True)      # K: per-channel
    pv, vs = quantize_block(src_v, per_channel=False)     # V: per-token
    dst_ks.copy_(ks)
    dst_vs.copy_(vs)
    dst_k.copy_(pk)
    dst_v.copy_(pv)


@triton.jit
def _gather_unpack_kernel(
    PK, PV, SCALE_K, SCALE_V, TABLE, OUT_TBL,
    STAGE_K_PTR, STAGE_V_PTR, SLOT_OF_BLOCK, OK, OV,
    n_entries, n_blocks,
    HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr, N_TOK: tl.constexpr,
):
    """Per block-table ENTRY: unpack that block into scratch row ``i`` and rewrite the table.

    This is the selective read for GPU-only KV. Unpacking the whole pool instead costs work
    proportional to the POOL rather than the selection -- measured 2.6 ms/layer at a 12288-block
    pool, i.e. ~94 ms per decode step over 36 layers against a budget of a few ms.

    Writing ``OUT_TBL[i] = i`` is what makes it safe: the attention wrapper then reads scratch row
    ``i`` for entry ``i``, so the compaction is invisible to it. Duplicate entries (two rows
    selecting the same block) unpack it twice into different scratch rows -- wasteful by the
    duplication factor (measured 1.65x on a realistic table) but correct. De-duplicating was
    measured and REJECTED: the ceiling is 2.22 ms/step saved while ``torch.unique`` costs 4.75,
    a net loss, and it needs a host sync for the variable output size.

    In-flight blocks are read CONDITIONALLY from the staging area. Loading it unconditionally and
    selecting with ``tl.where`` cost 4.56x (0.170 -> 0.776 ms) to serve the ~0.2% of blocks
    actually in flight; the branch is UNIFORM across the program (every lane shares ``blk``), so it
    is a skipped load rather than divergence.
    """
    i = tl.program_id(0)
    if i >= n_entries:
        return
    blk = tl.load(TABLE + i)
    tl.store(OUT_TBL + i, i)
    if (blk < 0) | (blk >= n_blocks):
        return

    t = tl.arange(0, BLOCK_T)
    dh = tl.arange(0, HALF_DIM)
    tmask = t < N_TOK
    m = tmask[:, None]
    lo = dh * 2
    hi = dh * 2 + 1

    # Unpack from the packed cache. The common case (a completed block), so it is done
    # unconditionally -- and Triton needs every name bound on the fall-through of the branch below.
    psrc = blk * N_TOK * HALF_DIM + t[:, None] * HALF_DIM + dh[None, :]
    bk = tl.load(PK + psrc, mask=m, other=0).to(tl.int32)
    bv = tl.load(PV + psrc, mask=m, other=0).to(tl.int32)
    ks_lo = tl.load(SCALE_K + blk * HEAD_DIM + lo)[None, :]
    ks_hi = tl.load(SCALE_K + blk * HEAD_DIM + hi)[None, :]
    vs = tl.load(SCALE_V + blk * N_TOK + t, mask=tmask, other=0.0)[:, None]
    k_lo = ((bk & 0x0F) - _BIAS).to(tl.float32) * ks_lo
    k_hi = (((bk >> 4) & 0x0F) - _BIAS).to(tl.float32) * ks_hi
    v_lo = ((bv & 0x0F) - _BIAS).to(tl.float32) * vs
    v_hi = (((bv >> 4) & 0x0F) - _BIAS).to(tl.float32) * vs

    a_slot = tl.load(SLOT_OF_BLOCK + blk)
    if a_slot >= 0:
        # Still being written: no final scale exists, so the values above are meaningless and the
        # staging area's exact bf16 is authoritative. These are the NEWEST tokens, which attention
        # weights most heavily -- omitting them cost ALL the accuracy (0/20) when first wired.
        asrc = a_slot * N_TOK * HEAD_DIM
        k_lo = tl.load(STAGE_K_PTR + asrc + t[:, None] * HEAD_DIM + lo[None, :],
                       mask=m, other=0.0).to(tl.float32)
        k_hi = tl.load(STAGE_K_PTR + asrc + t[:, None] * HEAD_DIM + hi[None, :],
                       mask=m, other=0.0).to(tl.float32)
        v_lo = tl.load(STAGE_V_PTR + asrc + t[:, None] * HEAD_DIM + lo[None, :],
                       mask=m, other=0.0).to(tl.float32)
        v_hi = tl.load(STAGE_V_PTR + asrc + t[:, None] * HEAD_DIM + hi[None, :],
                       mask=m, other=0.0).to(tl.float32)

    # ONE contiguous store per tensor; see the module docstring -- this is a 14x difference and it
    # is entirely in these two stores.
    full = tl.arange(0, HEAD_DIM)
    dst = i * N_TOK * HEAD_DIM + t[:, None] * HEAD_DIM + full[None, :]
    tl.store(OK + dst, tl.join(k_lo, k_hi).reshape(BLOCK_T, HEAD_DIM)
             .to(OK.dtype.element_ty), mask=m)
    tl.store(OV + dst, tl.join(v_lo, v_hi).reshape(BLOCK_T, HEAD_DIM)
             .to(OV.dtype.element_ty), mask=m)


def gather_unpack(packed_k, packed_v, scale_k, scale_v, table, out_table,
                  stage_k, stage_v, slot_of_block, out_k, out_v) -> None:
    """Selective INT4 read for GPU-only KV: unpack just the selected blocks, compacted.

    ``table`` is the indexer's block table; ``out_table`` receives the identity mapping so the
    attention wrapper reads scratch row i for entry i. Grid is the number of table entries, so cost
    tracks the SELECTION, not the pool.
    """
    n = table.numel()
    if n == 0:
        return
    block_tokens = out_k.shape[1]
    head_dim = out_k.shape[2]
    _gather_unpack_kernel[(n,)](
        packed_k, packed_v, scale_k, scale_v, table, out_table,
        stage_k, stage_v, slot_of_block, out_k, out_v,
        n, packed_k.shape[0],
        HEAD_DIM=head_dim, HALF_DIM=head_dim // 2,
        BLOCK_T=triton.next_power_of_2(block_tokens), N_TOK=block_tokens,
        num_warps=4,
    )


__all__ = ["K_SCALE", "V_SCALE", "STAGE_K", "STAGE_V",
           "native_int4_attention_supported", "int4_cache_meta", "int4_request_cache_meta",
           "bytes_per_block", "compression_ratio", "quantize_into", "gather_unpack"]
