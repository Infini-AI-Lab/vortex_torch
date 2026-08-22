"""INT4 KV quant/dequant: torch parity, packing correctness, and the measured axis choice.

Three separable things must hold, and a failure in each means something different:

1. **Packing is lossless as a container.** All error must be the QUANTIZER's, none the packer's.
   Compared against an independent torch reference IN UNITS OF THE QUANTIZATION LEVEL, allowing a
   1-level difference on exact rounding ties: ``torch.round`` is half-to-EVEN while libdevice
   round (and the kernel's reciprocal-multiply instead of a true divide) is half-away-from-zero,
   so a value landing exactly on .5 legitimately rounds the other way. Measured: 4/4096 elements
   differ, every one at ``frac(x/scale) == 0.5000`` exactly. A real packing bug instead shows
   many-level errors following byte boundaries, which this rejects separately.

2. **The two axes are wired to the right tensors.** K's scale must vary along channels and be
   shared by the block's tokens; V's the reverse. Getting these backwards still runs, still
   round-trips, and silently costs accuracy -- the offline screen measured 3.4x worse K error.
   Verified structurally: a tensor with ONE fat channel must quantize well per-channel and badly
   per-token, and vice versa for one fat token.

3. **The measured ordering reproduces.** K prefers per-channel, V prefers per-token. If it does
   not, the kernel is not implementing the scheme that was actually chosen.

Run: python examples/misc/test_int4_kv.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_torch.cache.triton_kernels.int4_kv import (  # noqa: E402
    INT4_QMAX, dequantize_block, quantize_block,
)

fails = 0


def ref_quant(x, per_channel):
    """Independent torch reference: the exact arithmetic the kernel should implement."""
    xf = x.float()
    if per_channel:
        scale = xf.abs().amax(dim=0, keepdim=True) / INT4_QMAX
    else:
        scale = xf.abs().amax(dim=1, keepdim=True) / INT4_QMAX
    scale = scale.clamp_min(1e-8)
    q = torch.clamp(torch.round(xf / scale), -INT4_QMAX, INT4_QMAX)
    return q * scale, scale


def rel(a, b):
    d = a.float().norm().item()
    return (a.float() - b.float()).norm().item() / d if d > 0 else 0.0


print("1. torch parity, measured in quantization LEVELS (ties allowed)")
for n_tok, head_dim in ((32, 128), (16, 128), (32, 64), (7, 128), (1, 128)):
    for per_channel in (True, False):
        torch.manual_seed(0)
        x = (torch.randn(n_tok, head_dim, device="cuda") * 0.5).to(torch.bfloat16)
        packed, scale = quantize_block(x, per_channel=per_channel)
        got = dequantize_block(packed, scale, per_channel, n_tok, head_dim,
                               out_dtype=torch.float32)
        want, ref_scale = ref_quant(x, per_channel)

        shape_ok = tuple(packed.shape) == (n_tok, head_dim // 2)
        scale_ok = torch.allclose(scale, ref_scale, rtol=1e-5, atol=1e-8)
        diff_levels = ((got - want) / ref_scale).abs()
        n_off = int((diff_levels > 1e-3).sum())
        max_levels = diff_levels.max().item()
        ties_only = max_levels <= 1.0 + 1e-3
        frac = (x.float() / ref_scale)
        frac = frac - frac.floor()
        on_tie = bool((frac[diff_levels > 1e-3] - 0.5).abs().max() < 1e-3) if n_off else True
        ok = shape_ok and scale_ok and ties_only and on_tie
        axis = "per-channel(K)" if per_channel else "per-token(V)"
        print(f"   {n_tok:>3d}x{head_dim:<4d} {axis:<16s} packed={tuple(packed.shape)} "
              f"off={n_off:>3d}/{x.numel():<5d} max={max_levels:.3f} lvl "
              f"{'ok' if ok else '<-- FAIL'}")
        if not ok:
            fails += 1
            if not shape_ok:
                print(f"      packed shape {tuple(packed.shape)} != {(n_tok, head_dim // 2)}")
            if not ties_only:
                print(f"      {max_levels:.2f} levels off -- too large for a rounding tie, "
                      f"this is a packing bug")
            if not on_tie:
                print("      mismatches are NOT on .5 boundaries -- packing bug, not rounding")

print("\n2. the axes are wired to the right tensors")
torch.manual_seed(0)
n_tok, head_dim = 32, 128
fat_ch = (torch.randn(n_tok, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
fat_ch[:, 7] = 50.0
fat_tok = (torch.randn(n_tok, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
fat_tok[5, :] = 50.0

for name, x, better in (("fat channel", fat_ch, "per-channel"),
                        ("fat token", fat_tok, "per-token")):
    e = {}
    for per_channel in (True, False):
        packed, scale = quantize_block(x, per_channel=per_channel)
        got = dequantize_block(packed, scale, per_channel, n_tok, head_dim,
                               out_dtype=torch.float32)
        e["per-channel" if per_channel else "per-token"] = rel(x, got)
    worse = "per-token" if better == "per-channel" else "per-channel"
    ok = e[better] < e[worse]
    print(f"   {name:<12s}: per-channel={e['per-channel']:.4f} per-token={e['per-token']:.4f}"
          f"  -> {better} should win: {'ok' if ok else '<-- FAIL'}")
    if not ok:
        fails += 1

print("\n3. K prefers per-channel and V prefers per-token, as the offline screen found")
# Emulate the MEASURED structure. K: a few channels with a much larger range. V: no channel
# structure but variation in per-token magnitude. i.i.d. normal V is the wrong model -- with no
# structure on either axis both scales are equivalent and the comparison cannot discriminate.
torch.manual_seed(1)
K = (torch.randn(n_tok, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
cg = torch.ones(head_dim, device="cuda"); cg[::16] = 12.0
K = (K.float() * cg).to(torch.bfloat16)
V = torch.randn(n_tok, head_dim, device="cuda") * 0.1
tg = torch.ones(n_tok, 1, device="cuda"); tg[::8] = 12.0
V = (V * tg).to(torch.bfloat16)

for name, x, expect in (("K (channel outliers)", K, "per-channel"),
                        ("V (token spread)", V, "per-token")):
    e = {}
    for per_channel in (True, False):
        packed, scale = quantize_block(x, per_channel=per_channel)
        got = dequantize_block(packed, scale, per_channel, n_tok, head_dim,
                               out_dtype=torch.float32)
        e["per-channel" if per_channel else "per-token"] = rel(x, got)
    win = min(e, key=e.get)
    ok = (win == expect)
    print(f"   {name:<22s} per-channel={e['per-channel']:.4f} "
          f"per-token={e['per-token']:.4f} -> winner {win} "
          f"({'ok' if ok else 'FAIL, expected ' + expect})")
    if not ok:
        fails += 1

print("\n4. zero and constant blocks (degenerate scales must not divide by zero)")
for name, x in (("all zeros", torch.zeros(n_tok, head_dim, device="cuda",
                                          dtype=torch.bfloat16)),
                ("all ones", torch.ones(n_tok, head_dim, device="cuda",
                                        dtype=torch.bfloat16))):
    for per_channel in (True, False):
        packed, scale = quantize_block(x, per_channel=per_channel)
        got = dequantize_block(packed, scale, per_channel, n_tok, head_dim,
                               out_dtype=torch.float32)
        finite = bool(torch.isfinite(got).all())
        err = rel(x, got)
        ok = finite and err < 1e-2
        print(f"   {name:<10s} {'per-channel' if per_channel else 'per-token':<12s} "
              f"finite={finite} err={err:.2e} {'ok' if ok else '<-- FAIL'}")
        if not ok:
            fails += 1

print()
if fails:
    print(f"*** {fails} FAILURE(S) ***")
    sys.exit(1)
print("INT4 KV KERNEL TESTS PASS")
