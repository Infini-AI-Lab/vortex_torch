"""INT4 KV quantization: packed-uint8 storage, per-channel K / per-token V scales.

Why these axes, and why they differ between K and V
---------------------------------------------------
Measured on real captured q/K/V (Qwen3-4B over 192 head-layers, replicated on Qwen3-32B over
128), comparing 8 candidate schemes:

    scheme                       K err    V err   OUT err   sel recall
    per-channel K, per-token V   0.067    0.145     0.147        0.939
    per-token both               0.230    0.147     0.510        0.759
    one scale for everything     0.351    0.380     0.897        0.667   <- the fp8 approach

K has strong per-CHANNEL outliers: a few dimensions with a much larger range. A per-token scale
cannot absorb them -- one fat channel inflates the scale for its whole row and crushes every
other channel of that token to a couple of levels (K err 0.230 vs 0.067, 3.4x). V is flatter,
and there the ordering REVERSES: per-token 0.145 beats per-channel 0.185. So the two tensors
genuinely want different axes, and using one axis for both -- the obvious implementation -- is
what costs the most. Asymmetric (zero-point) quantization was also tested and lost (OUT err
0.438), so this is symmetric-only.

``sel recall`` is the vortex-specific column and the reason this is not a copy of a quantization
paper: the *indexer* scores blocks from stored K, so if quantization reorders the top-k,
quantization error compounds with sparsity error. A scheme with slightly worse reconstruction
but stable selection is the better choice here.

Storage layout
--------------
Two 4-bit values per byte, so K/V keep the cache's ``(pages*page_size*heads, head_dim)``
addressing with the last dim halved. Channel ``d`` lives in byte ``d // 2``, low nibble for even
``d`` and high nibble for odd ``d``. Packing along ``head_dim`` (not along tokens) keeps a
token's channels contiguous and matches what flashinfer's native NVFP4 path expects
(``head_dim // 2`` uint8) should we ever reach it on Blackwell.

Do NOT be tempted to permute channels to make the halves unit-stride (i.e. byte ``d`` -> channels
``d`` and ``d + head_dim/2``). It is 14x faster on the store, and it produces **0% accuracy**:
``q`` never passes through this cache, so it stays in natural channel order and ``q·K``
misaligns (verified in isolation, rel err 1.44). Permuting ``q`` as well does not save it either,
because V's channels ARE the output channels, so the attention output would need un-permuting
too. Take the coalescing from the STORE SHAPE instead -- see ``int4_store``'s read kernels, which
get the same 14x with ``tl.join`` and one contiguous store, bit-identically.

Scales are ordinary fp32 tensors, NOT packed:
  * K scale: ``(1, head_dim)`` -- one per (block, channel), shared by the block's tokens. This is
    exactly what ``Reduce(dim=1)`` produces, i.e. the shape the Quest flow already computes.
  * V scale: ``(block_tokens, 1)`` -- one per (block, token), shared across channels
    (``Reduce(dim=2)``).

Symmetric, 15 levels
--------------------
``q = clamp(round(x / scale), -7, 7)``, stored biased by +7 so the nibble is unsigned:
``[0, 14]``. 15 levels rather than 16 because a symmetric range needs an odd count, and spending
one level buys **exact representation of zero** -- attention K/V are near-zero-mean, so a grid
that cannot represent 0 puts a bias on every block.

Uninitialised memory is NOT self-identifying: a zeroed page decodes to ``-7 * scale``, not to
zero. That is acceptable only because reading an unwritten block is already forbidden --
``slot_of``/``owner_of`` guarantee it for the staging pool, and the frontier staging area keeps a
partially-filled block in bf16 until it is complete, so no consumer ever dequantizes a block that
was not fully written. Do not weaken either guarantee.
"""

import torch
import triton
import triton.language as tl

#: Symmetric int4: 15 levels, [-7, +7]. Not 16 -- a symmetric range needs an odd count, and
#: giving up one level buys exact representation of 0, which matters because attention K/V are
#: near-zero-mean and a scheme that cannot represent 0 biases every block.
INT4_QMAX: int = 7
#: Stored biased so the on-disk byte is unsigned: q + BIAS in [0, 14].
INT4_BIAS: int = 7

#: Kernel-readable copies. A @jit'ed function cannot read a plain module global, and the
#: ANNOTATED form (``x: tl.constexpr = 7``) is explicitly unsupported -- it must be the
#: ``tl.constexpr(...)`` CALL form. Triton only reports this once the kernel compiles, so the
#: failure surfaces far from the definition.
_QMAX = tl.constexpr(INT4_QMAX)
_BIAS = tl.constexpr(INT4_BIAS)


@triton.jit
def _quant_block_kernel(
    SRC,                 # bf16/fp32 source, [n_tok, HEAD_DIM] within one block
    DST,                 # uint8 packed,     [n_tok, HEAD_DIM // 2]
    KSCALE,              # fp32, per-channel: [1, HEAD_DIM]   (K)
    VSCALE,              # fp32, per-token:   [n_tok, 1]      (V)
    n_tok,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,      # HEAD_DIM // 2; a separate constexpr because tl.arange needs
                                 # a constexpr argument and `HEAD_DIM // 2` does not propagate
    PER_CHANNEL: tl.constexpr,   # True -> K (scale varies by channel), False -> V (by token)
    BLOCK_T: tl.constexpr,
):
    """Quantize one block's worth of tokens to packed int4.

    One program per block. ``PER_CHANNEL`` is a constexpr so each variant compiles to its own
    kernel with no per-element branching -- the K and V paths differ only in which axis the scale
    is broadcast along.
    """
    t = tl.arange(0, BLOCK_T)
    tmask = t < n_tok
    m = tmask[:, None]

    # Pack channel pairs (2d, 2d+1) into byte d, loading the even and odd columns directly rather
    # than loading [T, D] and re-gathering: one pass, and the scale for each half is fetched with
    # the same stride as the data it divides.
    half = HALF_DIM
    dh = tl.arange(0, HALF_DIM)
    lo_off = t[:, None] * HEAD_DIM + (dh[None, :] * 2)
    hi_off = t[:, None] * HEAD_DIM + (dh[None, :] * 2 + 1)
    if PER_CHANNEL:
        s_lo = tl.load(KSCALE + dh * 2)[None, :]
        s_hi = tl.load(KSCALE + dh * 2 + 1)[None, :]
    else:
        s_lo = tl.load(VSCALE + t, mask=tmask, other=1.0)[:, None]
        s_hi = s_lo
    x_lo = tl.load(SRC + lo_off, mask=m, other=0.0).to(tl.float32)
    x_hi = tl.load(SRC + hi_off, mask=m, other=0.0).to(tl.float32)
    i_lo = tl.where(s_lo > 0.0, 1.0 / s_lo, 0.0)
    i_hi = tl.where(s_hi > 0.0, 1.0 / s_hi, 0.0)
    q_lo = tl.extra.cuda.libdevice.round(x_lo * i_lo)
    q_hi = tl.extra.cuda.libdevice.round(x_hi * i_hi)
    q_lo = (tl.minimum(tl.maximum(q_lo, -_QMAX), _QMAX).to(tl.int32) + _BIAS)
    q_hi = (tl.minimum(tl.maximum(q_hi, -_QMAX), _QMAX).to(tl.int32) + _BIAS)
    packed = (q_lo | (q_hi << 4)).to(tl.uint8)

    dst_off = t[:, None] * half + dh[None, :]
    tl.store(DST + dst_off, packed, mask=m)


@triton.jit
def _dequant_block_kernel(
    SRC,                 # uint8 packed, [n_tok, HEAD_DIM // 2]
    DST,                 # bf16 out,     [n_tok, HEAD_DIM]
    KSCALE, VSCALE,
    n_tok,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Unpack + rescale one block back to bf16."""
    t = tl.arange(0, BLOCK_T)
    half = HALF_DIM
    dh = tl.arange(0, HALF_DIM)
    tmask = t < n_tok
    m = tmask[:, None]

    src_off = t[:, None] * half + dh[None, :]
    b = tl.load(SRC + src_off, mask=m, other=0).to(tl.int32)
    q_lo = (b & 0x0F) - _BIAS
    q_hi = ((b >> 4) & 0x0F) - _BIAS

    if PER_CHANNEL:
        s_lo = tl.load(KSCALE + dh * 2)[None, :]
        s_hi = tl.load(KSCALE + dh * 2 + 1)[None, :]
    else:
        s = tl.load(VSCALE + t, mask=tmask, other=0.0)[:, None]
        s_lo = s
        s_hi = s

    x_lo = q_lo.to(tl.float32) * s_lo
    x_hi = q_hi.to(tl.float32) * s_hi

    lo_off = t[:, None] * HEAD_DIM + (dh[None, :] * 2)
    hi_off = t[:, None] * HEAD_DIM + (dh[None, :] * 2 + 1)
    tl.store(DST + lo_off, x_lo.to(DST.dtype.element_ty), mask=m)
    tl.store(DST + hi_off, x_hi.to(DST.dtype.element_ty), mask=m)


def quantize_block(src: torch.Tensor, per_channel: bool):
    """Quantize ``src`` [n_tok, head_dim] -> (packed uint8, scale).

    ``per_channel=True`` gives K's axis (one scale per channel, shared across the block's
    tokens); ``False`` gives V's (one per token, shared across channels). Returns the scale in the
    shape the cache field uses: ``(1, head_dim)`` for K, ``(n_tok, 1)`` for V.
    """
    n_tok, head_dim = src.shape
    assert head_dim % 2 == 0, f"head_dim must be even to pack two int4/byte, got {head_dim}"
    x = src.to(torch.float32)
    if per_channel:
        scale = x.abs().amax(dim=0, keepdim=True) / INT4_QMAX      # (1, head_dim)
    else:
        scale = x.abs().amax(dim=1, keepdim=True) / INT4_QMAX      # (n_tok, 1)
    scale = scale.clamp_min(1e-8).contiguous()
    dst = torch.empty((n_tok, head_dim // 2), dtype=torch.uint8, device=src.device)
    _quant_block_kernel[(1,)](
        src.contiguous(), dst, scale, scale, n_tok,
        HEAD_DIM=head_dim, HALF_DIM=head_dim // 2, PER_CHANNEL=per_channel,
        BLOCK_T=triton.next_power_of_2(n_tok),
        num_warps=1,
    )
    return dst, scale


def dequantize_block(packed: torch.Tensor, scale: torch.Tensor, per_channel: bool,
                     n_tok: int, head_dim: int, out_dtype=torch.bfloat16):
    """Inverse of :func:`quantize_block`."""
    dst = torch.empty((n_tok, head_dim), dtype=out_dtype, device=packed.device)
    _dequant_block_kernel[(1,)](
        packed.contiguous(), dst, scale, scale, n_tok,
        HEAD_DIM=head_dim, HALF_DIM=head_dim // 2, PER_CHANNEL=per_channel,
        BLOCK_T=triton.next_power_of_2(n_tok),
        num_warps=1,
    )
    return dst


__all__ = ["INT4_QMAX", "INT4_BIAS", "quantize_block", "dequantize_block",
           "_quant_block_kernel", "_dequant_block_kernel"]
