"""Test + understand flash_attn_with_kvcache with a paged KV cache.

flash_attn 2.x supports a paged KV layout via ``block_table``:

    k_cache, v_cache : [num_blocks, page_block_size, nheads_kv, head_dim]
    block_table      : [batch_size, max_num_blocks_per_seq], int32
    cache_seqlens    : [batch_size], int32   (logical KV length per request)

Constraints worth remembering:
    * ``page_block_size`` must be a multiple of 256 (flash_attn requirement).
    * ``q`` layout is ``[B, seqlen_q, nheads_q, head_dim]`` — NOT [B, Hq, D].
    * For GQA, ``nheads_q`` must be divisible by ``nheads_kv``.
    * If ``k``/``v`` are passed, they are appended IN-PLACE into the page at
      positions ``[cache_seqlens, cache_seqlens + seqlen_new)``. Block_table
      must already have enough blocks reserved to cover the new length.

Two subtests:
    1. paged decode, no append — pure attention over the prefilled cache.
    2. paged decode WITH append — flash_attn writes new K/V into the cache
       and attends over (old + new) in a single kernel.

Each subtest is validated against a torch SDPA reference that gathers K and V
from the paged cache via the same block_table.

Run inside conda env ``vortex_v04``:

    /home/zhuominc/anaconda3/envs/vortex_v04/bin/python \
        examples/misc/test_flash_attn_paged_kvcache.py
"""

from __future__ import annotations

import argparse
import math

import torch
from flash_attn import flash_attn_with_kvcache


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def allocate_paged_cache(
    *,
    seq_lens: list[int],
    page_block_size: int,
    nheads_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    extra_blocks_per_seq: int = 0,
):
    """Allocate a paged KV cache and a block_table covering ``seq_lens``.

    Returns
    -------
    k_cache, v_cache : [num_blocks, page_block_size, nheads_kv, head_dim]
    block_table      : [B, max_blocks_per_seq], int32
    pages_per_seq    : list[int], how many pages each request occupies

    Block IDs are assigned greedily — request i gets a contiguous run.
    ``extra_blocks_per_seq`` reserves spare pages per request (needed when
    you intend to append new tokens past the current ``cache_seqlens``).
    """
    B = len(seq_lens)
    pages_per_seq = [
        (s + page_block_size - 1) // page_block_size + extra_blocks_per_seq
        for s in seq_lens
    ]
    total_blocks = sum(pages_per_seq)
    max_blocks_per_seq = max(pages_per_seq)

    k_cache = torch.randn(
        (total_blocks, page_block_size, nheads_kv, head_dim),
        dtype=dtype, device=device,
    ) * 0.1
    v_cache = torch.randn_like(k_cache) * 0.1

    block_table = torch.zeros((B, max_blocks_per_seq), dtype=torch.int32, device=device)
    next_id = 0
    for i, n_pages in enumerate(pages_per_seq):
        for j in range(n_pages):
            block_table[i, j] = next_id
            next_id += 1
    return k_cache, v_cache, block_table, pages_per_seq


def gather_logical_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct dense per-request K, V from the paged cache.

    Returns ``k_dense, v_dense`` of shape ``[B, max_seqlen, Hkv, D]``.
    Tail positions past ``seq_lens[i]`` are zeroed.
    """
    B, max_blocks = block_table.shape
    page_block_size = k_cache.shape[1]
    Hkv, D = k_cache.shape[2], k_cache.shape[3]
    max_seqlen = int(seq_lens.max().item())
    out_k = torch.zeros(
        (B, max_seqlen, Hkv, D), dtype=k_cache.dtype, device=k_cache.device
    )
    out_v = torch.zeros_like(out_k)
    for i in range(B):
        s = int(seq_lens[i].item())
        for j in range(max_blocks):
            t_start = j * page_block_size
            if t_start >= s:
                break
            t_end = min(t_start + page_block_size, s)
            bid = int(block_table[i, j].item())
            n = t_end - t_start
            out_k[i, t_start:t_end] = k_cache[bid, :n]
            out_v[i, t_start:t_end] = v_cache[bid, :n]
    return out_k, out_v


def reference_attention(
    q: torch.Tensor,             # [B, Lq, Hq, D]
    k: torch.Tensor,             # [B, Lk, Hkv, D]
    v: torch.Tensor,             # [B, Lk, Hkv, D]
    seq_lens: torch.Tensor,      # [B], int32 — effective K length per request
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Torch SDPA reference with per-request key truncation and GQA expansion.

    Mirrors flash_attn's bottom-right-aligned causal mask.
    """
    B, Lq, Hq, D = q.shape
    Lk_max = k.shape[1]
    Hkv = k.shape[2]
    assert Hq % Hkv == 0
    group = Hq // Hkv

    out = torch.zeros_like(q)
    for i in range(B):
        s = int(seq_lens[i].item())
        q_i = q[i].float().transpose(0, 1)              # [Hq, Lq, D]
        k_i = k[i, :s].float().transpose(0, 1)          # [Hkv, s, D]
        v_i = v[i, :s].float().transpose(0, 1)
        k_i = k_i.repeat_interleave(group, dim=0)       # [Hq, s, D]
        v_i = v_i.repeat_interleave(group, dim=0)
        logits = torch.einsum("hld,hsd->hls", q_i, k_i) * softmax_scale
        if causal:
            # bottom-right alignment: query l attends up to key index s - Lq + l
            row = torch.arange(Lq, device=q.device).unsqueeze(1)         # [Lq, 1]
            col = torch.arange(s, device=q.device).unsqueeze(0)          # [1, s]
            keep = col <= (s - Lq + row)
            logits = logits.masked_fill(~keep.unsqueeze(0), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        # If a row was entirely -inf, softmax produced NaN; flash_attn outputs 0.
        probs = torch.nan_to_num(probs, nan=0.0)
        out_i = torch.einsum("hls,hsd->hld", probs, v_i)                 # [Hq, Lq, D]
        out[i] = out_i.transpose(0, 1).to(q.dtype)
    return out


def compare(name: str, out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> bool:
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = (diff / ref.float().abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<30s} max|Δ|={max_abs:.3e}  mean|Δ|={mean_abs:.3e}  max rel={rel:.3e}")
    return ok


# --------------------------------------------------------------------------- #
# Subtest 1: paged decode, no append
# --------------------------------------------------------------------------- #

def subtest_decode_no_append(args, device) -> bool:
    print("\n[subtest 1] paged decode, no append (k=v=None)")
    dtype = args.dtype_t
    seq_lens_list = args.seq_lens
    B = len(seq_lens_list)

    k_cache, v_cache, block_table, pages = allocate_paged_cache(
        seq_lens=seq_lens_list,
        page_block_size=args.page_block_size,
        nheads_kv=args.nheads_kv,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    cache_seqlens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    q = torch.randn(
        (B, 1, args.nheads_q, args.head_dim), dtype=dtype, device=device
    ) * 0.1
    sm_scale = 1.0 / math.sqrt(args.head_dim)

    print(
        f"  shapes: q={tuple(q.shape)}  k_cache={tuple(k_cache.shape)}  "
        f"block_table={tuple(block_table.shape)}  cache_seqlens={cache_seqlens.tolist()}"
    )
    print(f"  pages/req: {pages}  (page_block_size={args.page_block_size})")

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=sm_scale,
        causal=False,   # decode w/ seqlen_q=1 — causal flag is moot
    )
    torch.cuda.synchronize()

    k_dense, v_dense = gather_logical_kv(k_cache, v_cache, block_table, cache_seqlens)
    ref = reference_attention(q, k_dense, v_dense, cache_seqlens, sm_scale, causal=False)

    return compare("decode_no_append", out, ref, args.atol, args.rtol)


# --------------------------------------------------------------------------- #
# Subtest 2: paged decode WITH append
# --------------------------------------------------------------------------- #

def subtest_decode_with_append(args, device) -> bool:
    print("\n[subtest 2] paged decode WITH append (k, v provided — in-place cache update)")
    dtype = args.dtype_t
    seq_lens_list = args.seq_lens
    B = len(seq_lens_list)

    # Reserve one extra page per seq so cache_seqlens+1 always fits.
    k_cache, v_cache, block_table, pages = allocate_paged_cache(
        seq_lens=seq_lens_list,
        page_block_size=args.page_block_size,
        nheads_kv=args.nheads_kv,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
        extra_blocks_per_seq=1,
    )
    cache_seqlens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    q = torch.randn(
        (B, 1, args.nheads_q, args.head_dim), dtype=dtype, device=device
    ) * 0.1
    k_new = torch.randn(
        (B, 1, args.nheads_kv, args.head_dim), dtype=dtype, device=device
    ) * 0.1
    v_new = torch.randn_like(k_new)

    # Snapshot the cache BEFORE the call so we can verify in-place mutation.
    k_cache_before = k_cache.clone()

    sm_scale = 1.0 / math.sqrt(args.head_dim)
    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=sm_scale,
        causal=True,
    )
    torch.cuda.synchronize()

    # ---- Verify in-place append landed at the right page slot --------------- #
    pre_existing_diff = 0.0
    for i in range(B):
        s = seq_lens_list[i]
        block_in_page = s // args.page_block_size
        slot = s % args.page_block_size
        bid = int(block_table[i, block_in_page].item())
        written = k_cache[bid, slot]                          # [Hkv, D]
        expected = k_new[i, 0]                                 # [Hkv, D]
        d = (written.float() - expected.float()).abs().max().item()
        if d > 1e-5:
            print(f"  [WARN] req {i}: written K @ block {bid} slot {slot} mismatched (max={d:.3e})")
        pre_existing_diff = max(pre_existing_diff, d)
    print(f"  in-place append check: max|k_cache[slot] - k_new|={pre_existing_diff:.3e}")

    # ---- Reference: gather using the UPDATED cache + new seq_lens ----------- #
    new_seqlens = cache_seqlens + 1
    k_dense, v_dense = gather_logical_kv(k_cache, v_cache, block_table, new_seqlens)
    ref = reference_attention(q, k_dense, v_dense, new_seqlens, sm_scale, causal=True)

    ok = compare("decode_with_append", out, ref, args.atol, args.rtol)

    # Sanity: pages prior to each request's append region should be untouched.
    pre_unchanged = True
    for i in range(B):
        s = seq_lens_list[i]
        full_pages = s // args.page_block_size
        for j in range(full_pages):
            bid = int(block_table[i, j].item())
            if not torch.equal(k_cache[bid], k_cache_before[bid]):
                pre_unchanged = False
                print(f"  [WARN] page {bid} of req {i} was modified unexpectedly")
                break
    print(f"  pre-existing pages untouched: {pre_unchanged}")
    return ok and pre_unchanged


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[37, 300, 1023, 2048],
                   help="logical KV length per request before append")
    p.add_argument("--nheads-q", type=int, default=32)
    p.add_argument("--nheads-kv", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-block-size", type=int, default=256,
                   help="must be a multiple of 256")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=5e-3)
    p.add_argument("--rtol", type=float, default=5e-3)
    args = p.parse_args()

    if args.page_block_size % 256 != 0:
        raise SystemExit("flash_attn requires page_block_size % 256 == 0")
    if args.nheads_q % args.nheads_kv != 0:
        raise SystemExit("nheads_q must be divisible by nheads_kv (GQA constraint)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    args.dtype_t = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    cc = torch.cuda.get_device_capability(device)
    print(
        f"device={torch.cuda.get_device_name(device)} sm_{cc[0]}{cc[1]}  dtype={args.dtype}"
    )

    results = []
    results.append(subtest_decode_no_append(args, device))
    results.append(subtest_decode_with_append(args, device))

    print()
    print("summary:")
    print(f"  no-append: {'PASS' if results[0] else 'FAIL'}")
    print(f"  append   : {'PASS' if results[1] else 'FAIL'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
