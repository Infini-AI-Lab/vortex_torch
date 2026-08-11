"""Sweep ``block_q`` — the selection granularity — against the dense baseline.

``block_q`` is how many query tokens share one KV selection. ``block_q=64`` (the default)
amortises the scorer over 64 tokens; ``block_q=1`` gives **every token its own selection**,
with no averaging of queries into a block. That is the accuracy reference and the expensive
end of the tradeoff, so it deserves its own numbers rather than a footnote.

Reporting rules this follows, because sparse-attention numbers are easy to overstate:

1. the baseline is dense SDPA with its **flash backend explicitly enabled**, not a masked
   reference that would be unfairly slow;
2. the whole step is timed — selection, transpose, attention, backward — because selection
   cost scales with ``1/block_q`` and timing only the attention kernel would flatter small
   ``block_q`` badly;
3. peak memory is reported — and it does **drop** below ``block_q=16``, which is not a
   ``block_q`` effect but a dispatch one: ``block_q < 16`` uses the packed dk/dv kernel,
   which folds the GQA group into the MMA contraction and so writes ``dk``/``dv``
   directly, whereas the general path stages ``[B, Hq, Skv, D]`` fp32 buffers and reduces
   them in a second pass. Measured 1159 -> 663 MB at seqlen 16k, matching the 512 MB
   those two buffers occupy.

Usage:
    python benchmarks/bench_block_q.py
    python benchmarks/bench_block_q.py --seqlens 16384 --block-qs 1 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_train.flow.spec import REGISTRY, Budget  # noqa: E402
from vortex_train.nn import SparseAttention  # noqa: E402


def _time(fn, warmup: int = 3, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def _qkv(seqlen: int, hq: int, hkv: int, d: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    mk = lambda h: (torch.randn(1, h, seqlen, d, device="cuda", dtype=torch.bfloat16,
                                generator=g) * 0.5).requires_grad_(True)
    return mk(hq), mk(hkv), mk(hkv)


def dense_step(seqlen: int, hq: int, hkv: int, d: int, group: int) -> tuple[float, float]:
    q, k, v = _qkv(seqlen, hq, hkv, d)
    backend = torch.nn.attention.SDPBackend.FLASH_ATTENTION

    def f():
        with torch.nn.attention.sdpa_kernel(backend):
            o = torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
                is_causal=True)
        o.backward(torch.ones_like(o))
        q.grad = k.grad = v.grad = None

    torch.cuda.reset_peak_memory_stats()
    ms = _time(f)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    del q, k, v
    torch.cuda.empty_cache()
    return ms, peak


def sparse_step(seqlen: int, block_q: int, a) -> tuple[float, float]:
    policy = type(f"P{block_q}", (REGISTRY[a.algo],), {
        "block_q": block_q, "block_kv": a.block_kv,
        "budget": Budget(topk=a.topk, reserve_bos=a.reserve_bos,
                         reserve_local=a.reserve_local),
    })
    hq = a.hkv * a.group
    q, k, v = _qkv(seqlen, hq, a.hkv, a.head_dim)
    attn = SparseAttention(policy, num_kv_heads=a.hkv)

    def f():
        o = attn(q, k, v)
        o.backward(torch.ones_like(o))
        q.grad = k.grad = v.grad = None

    torch.cuda.reset_peak_memory_stats()
    ms = _time(f)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    del q, k, v, attn
    torch.cuda.empty_cache()
    return ms, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--block-qs", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--block-kv", type=int, default=64)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--reserve-bos", type=int, default=1)
    ap.add_argument("--reserve-local", type=int, default=1)
    ap.add_argument("--algo", default="block_topk")
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    a = ap.parse_args()

    hq = a.hkv * a.group
    total = a.topk + a.reserve_bos + a.reserve_local
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"Hq={hq} Hkv={a.hkv} D={a.head_dim} group={a.group}, bf16")
    print(f"budget: topk {a.topk} + bos {a.reserve_bos} + local {a.reserve_local} = "
          f"{total} blocks x {a.block_kv} = {total * a.block_kv} KV tokens\n")

    hdr = (f"{'seqlen':>7} {'dense fb':>9} "
           + " ".join(f"{'bq=' + str(b):>10}" for b in a.block_qs))
    print(hdr)
    print("-" * len(hdr))
    results: dict[tuple[int, int], tuple[float, float]] = {}
    dense: dict[int, float] = {}
    for s in a.seqlens:
        try:
            d_ms, _ = dense_step(s, hq, a.hkv, a.head_dim, a.group)
        except torch.cuda.OutOfMemoryError:
            d_ms = float("nan")
            torch.cuda.empty_cache()
        dense[s] = d_ms
        cells = []
        for bq in a.block_qs:
            try:
                ms, peak = sparse_step(s, bq, a)
                results[(s, bq)] = (ms, peak)
                cells.append(f"{ms:>10.2f}")
            except torch.cuda.OutOfMemoryError:
                cells.append(f"{'OOM':>10}")
                torch.cuda.empty_cache()
        print(f"{s:>7} {d_ms:>9.2f} " + " ".join(cells))

    print(f"\nspeedup vs dense SDPA-flash")
    print(f"{'seqlen':>7} " + " ".join(f"{'bq=' + str(b):>10}" for b in a.block_qs))
    for s in a.seqlens:
        print(f"{s:>7} " + " ".join(
            f"{dense[s] / results[(s, b)][0]:>9.2f}x" if (s, b) in results else f"{'n/a':>10}"
            for b in a.block_qs))

    print(f"\npeak MB (drops below block_q=16: the packed dk/dv kernel needs no fp32 staging)")
    print(f"{'seqlen':>7} " + " ".join(f"{'bq=' + str(b):>10}" for b in a.block_qs))
    for s in a.seqlens:
        print(f"{s:>7} " + " ".join(
            f"{results[(s, b)][1]:>10.0f}" if (s, b) in results else f"{'n/a':>10}"
            for b in a.block_qs))

    print("\nblock_q=1 gives every query token its own selection (no averaging of")
    print("queries into a block); block_q=64 amortises the scorer over 64 tokens.")
    print("The step includes selection, transpose, attention and backward -- selection")
    print("cost scales as 1/block_q, so timing attention alone would flatter small block_q.")


if __name__ == "__main__":
    main()
