"""Selection quality: does each policy pick the blocks that actually matter?

Speed without quality is meaningless — a policy that picks blocks at random is the
fastest of all. This measures, at a fixed block budget, how much of the true
attention mass each policy's selection captures:

* **mass recall** — fraction of the total post-softmax attention weight that lands
  inside the selected blocks. The number that matters, because attention output is a
  weighted average: capturing 99% of the mass means the output is nearly exact even
  if many blocks were dropped.
* **top-block recall** — fraction of the ``topk`` *highest-mass* blocks that were
  selected. A stricter, mass-agnostic view; a policy can score well on mass by
  grabbing the local window alone, and this exposes that.
* **bound tightness** (envelope policies) — the gap between the policy's score and
  the true ``max_k <q,k>`` in the block. This is the mechanism behind any recall
  difference between ``quest`` and ``lserve``, so measuring it turns "lserve is
  better" into "lserve is better *because* its bound is tighter".

Ground truth is exact dense attention in fp32, restricted to the causal region.
Reference is the oracle path, not a kernel, so a bad number here is a *policy*
result rather than a kernel bug.

**The data matters more than the metric.** On i.i.d. Gaussian q/k, attention is
almost uniform: every block holds nearly the same mass, so *every* policy — including
one that only keeps a local window and scores nothing — recalls the same fraction,
and the comparison is a null result by construction. Measured: mass recall 0.3688 for
all five policies at topk=8. So this script defaults to ``--data retrieval``, which
plants a small number of high-affinity "needle" blocks that a policy has to *find*.
Gaussian is still available via ``--data gaussian`` as the negative control: all
policies tying there is the expected outcome and confirms the harness is not
accidentally favouring one.

Usage:
    python benchmarks/bench_recall.py
    python benchmarks/bench_recall.py --seqlen 8192 --topk 8 16 32
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vortex_train.flow.spec import REGISTRY  # noqa: E402
from vortex_train.nn import SparseAttention  # noqa: E402

ALGOS = ["streaming", "block_topk", "quest", "lserve", "lserve_centroid"]


def make_inputs(kind, b, hq, hkv, s, d, *, seed, block_kv=64, needles=4):
    """Synthesise q/k/v. ``retrieval`` plants needle blocks; ``gaussian`` is control.

    The needles are built by giving a few KV blocks keys that align with a shared
    direction which the queries also carry. That produces genuinely concentrated
    attention -- the regime long-context retrieval actually cares about, and the only
    one where block selection can be right or wrong.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda h: torch.randn(b, h, s, d, device="cuda", dtype=torch.float32,
                               generator=g) * 0.5
    q, k, v = mk(hq), mk(hkv), mk(hkv)
    if kind == "retrieval":
        n_kv = s // block_kv
        # one shared query/key direction per kv head
        direction = torch.randn(b, hkv, 1, d, device="cuda", generator=g)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        group = hq // hkv
        qd = direction.repeat_interleave(group, dim=1)
        q = q + 2.0 * qd                                  # queries look for it
        # plant it in `needles` blocks spread across the sequence, avoiding the
        # local window so a streaming policy cannot get them for free
        stride = max(n_kv // (needles + 1), 1)
        for j in range(1, needles + 1):
            blk = min(j * stride, n_kv - 1)
            lo = blk * block_kv
            k[:, :, lo : lo + block_kv, :] += 3.0 * direction
    return (q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16))


def true_block_mass(q, k, block_q, block_kv, *, num_kv_heads):
    """Exact post-softmax attention mass per (query block, kv block), fp32.

    Computed per query block to keep the score matrix bounded; the whole point of
    the sparse path is that this is what you cannot afford at scale.
    """
    b, hq, sq, d = q.shape
    skv = k.shape[2]
    group = hq // num_kv_heads
    m = sq // block_q
    n_kv = skv // block_kv
    scale = 1.0 / math.sqrt(d)

    kf = k.float().repeat_interleave(group, dim=1)                # [B,Hq,Skv,D]
    mass = torch.zeros((b, num_kv_heads, m, n_kv), dtype=torch.float32, device=q.device)
    kpos = torch.arange(skv, device=q.device)

    for i in range(m):
        qb = q[:, :, i * block_q : (i + 1) * block_q, :].float()  # [B,Hq,bq,D]
        s = torch.einsum("bhqd,bhkd->bhqk", qb, kf) * scale
        qpos = torch.arange(i * block_q, (i + 1) * block_q, device=q.device)
        s = s.masked_fill(kpos[None, None, None, :] > qpos[None, None, :, None],
                          float("-inf"))
        p = torch.softmax(s, dim=-1)                              # [B,Hq,bq,Skv]
        # Sum mass over the query tokens in the block and over the GQA group, then
        # fold Skv into kv blocks. Group-summed because selection is shared by the
        # group, so the relevant quantity is what the group as a whole wants.
        pb = p.sum(2).view(b, num_kv_heads, group, n_kv, block_kv).sum(-1).sum(2)
        mass[:, :, i] = pb / (block_q * group)
    return mass


def envelope_gap(q, k, *, block_kv, sub_block, num_kv_heads, block_q):
    """Mean (bound - true_max) over blocks: how loose the envelope bound is.

    Both variants are computed from the same keys, so the difference is purely the
    granularity of the max/min summary.
    """
    b, hq, sq, d = q.shape
    skv = k.shape[2]
    group = hq // num_kv_heads
    n_kv = skv // block_kv
    m = sq // block_q

    kb = k.float().view(b, num_kv_heads, n_kv, block_kv, d)
    # query summary per (b, kv head, q block, group member) -- what the policy uses
    qs = q.float().view(b, num_kv_heads, group, m, block_q, d).mean(4)   # [B,Hkv,G,M,D]

    def bound(nsub):
        sub = block_kv // nsub
        ks = kb.view(b, num_kv_heads, n_kv, nsub, sub, d)
        kmax, kmin = ks.max(4).values, ks.min(4).values              # [B,Hkv,Nkv,U,D]
        qe = qs[:, :, :, :, None, None, :]                           # [B,Hkv,G,M,1,1,D]
        hi = qe * kmax[:, :, None, None, :, :, :]
        lo = qe * kmin[:, :, None, None, :, :, :]
        return torch.maximum(hi, lo).sum(-1).max(-1).values          # [B,Hkv,G,M,Nkv]

    # true max_k <q_summary, k> within each block
    true = torch.einsum("bhgmd,bhnjd->bhgmnj", qs, kb).max(-1).values  # [B,Hkv,G,M,Nkv]

    out = {}
    for nsub, label in ((1, "block"), (block_kv // sub_block, "sub")):
        gap = (bound(nsub) - true).clamp(min=0)
        out[label] = float(gap.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--topk", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data", choices=["retrieval", "gaussian"], default="retrieval",
                    help="retrieval plants needle blocks; gaussian is the null control")
    a = ap.parse_args()

    hkv, d, s = a.hkv, a.head_dim, a.seqlen
    hq = hkv * a.group
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"seqlen {s}, Hq={hq} Hkv={hkv} D={d}, {a.seeds} seeds, bf16 inputs / fp32 truth")
    print(f"data: {a.data}"
          + ("  (needle blocks planted outside the local window)" if a.data == "retrieval"
             else "  (NULL CONTROL -- all policies are expected to tie)") + "\n")

    hdr = f"{'algorithm':>16} {'topk':>5} {'mass recall':>12} {'top-blk recall':>15} {'blocks/row':>11}"
    print(hdr)
    print("-" * len(hdr))

    for topk in a.topk:
        for algo in a.algos:
            mr, tr, cnts = [], [], []
            for seed in range(a.seeds):
                q, k, v = make_inputs(a.data, 1, hq, hkv, s, d, seed=seed)

                cls = REGISTRY[algo]
                # Override the budget for this sweep point without mutating the
                # registered class (which other rows still use).
                from vortex_train.flow.spec import Budget
                # Reservations must fit inside the swept budget. `streaming`
                # reserves 8 blocks by definition, so at topk=4 its local window is
                # clipped -- scaled proportionally rather than skipping the row,
                # since a clipped streaming policy is still the right floor to
                # compare against at that budget.
                b0 = cls.budget
                res = b0.reserve_bos + b0.reserve_local + b0.reserve_eos
                if res > topk:
                    keep_bos = min(b0.reserve_bos, 1 if topk >= 2 else 0)
                    keep_local = max(topk - keep_bos, 1)
                    bud = Budget(topk=topk, reserve_bos=keep_bos,
                                 reserve_local=keep_local, reserve_eos=0)
                else:
                    bud = Budget(topk=topk, reserve_bos=b0.reserve_bos,
                                 reserve_local=b0.reserve_local,
                                 reserve_eos=b0.reserve_eos)
                policy = type(f"{cls.__name__}_k{topk}", (cls,), {"budget": bud})
                attn = SparseAttention(policy, num_kv_heads=hkv)
                c = attn.compiled
                p = attn.build_pattern(q, k, v)

                mass = true_block_mass(q, k, c.block_q, c.block_kv, num_kv_heads=hkv)
                n_kv = mass.shape[3]
                sel = torch.zeros_like(mass, dtype=torch.bool)
                ar = torch.arange(p.idx.shape[3], device=q.device)
                valid = ar[None, None, None, :] < p.cnt[..., None]
                scratch = torch.full_like(p.idx, n_kv)
                safe = torch.where(valid, p.idx, scratch).to(torch.int64)
                padded = torch.zeros(mass.shape[:3] + (n_kv + 1,), dtype=torch.bool,
                                     device=q.device)
                padded.scatter_(3, safe, valid)
                sel = padded[..., :n_kv]

                total = mass.sum(-1, keepdim=True).clamp(min=1e-9)
                mr.append(float(((mass * sel).sum(-1, keepdim=True) / total).mean()))

                kk = min(topk, n_kv)
                top_true = mass.topk(kk, dim=-1).indices
                hit = torch.gather(sel, 3, top_true).float().mean()
                tr.append(float(hit))
                cnts.append(float(p.cnt.float().mean()))

            print(f"{algo:>16} {topk:>5} {sum(mr)/len(mr):>11.4f} "
                  f"{sum(tr)/len(tr):>14.4f} {sum(cnts)/len(cnts):>11.1f}")
        print()

    # Why lserve differs from quest, if it does: the bound's tightness.
    print("Envelope bound looseness -- mean(bound - true max) over blocks.")
    print("Lower is tighter; this is the mechanism behind any quest/lserve gap.")
    q, k, _ = make_inputs(a.data, 1, hq, hkv, s, d, seed=0)
    gaps = envelope_gap(q, k, block_kv=64, sub_block=16, num_kv_heads=hkv, block_q=64)
    print(f"  whole-block envelope (quest):   {gaps['block']:.4f}")
    print(f"  sub-block envelope  (lserve):   {gaps['sub']:.4f}")
    if gaps["block"] > 0:
        print(f"  tightening: {100 * (1 - gaps['sub'] / gaps['block']):.1f}%")


if __name__ == "__main__":
    main()
