"""Offline study of `approxTopK(tolerate_ratio)` on REAL RULER-16K attention.

Answers three questions per tolerate_ratio, using real q/K captured from a
model (``capture_trace.py``) rather than synthetic scores:

  * **expected radix rounds** — does the single-pass gate fire? The trtllm
    leaf histograms the high byte of the bf16 key, then stops if at least
    ``ceil((1-tol)*k)`` blocks are already strict winners; otherwise it runs a
    second round on the low byte. This is the *only* thing tolerate_ratio buys,
    so it bounds any possible speedup.
  * **recall@k** — selected blocks vs the exact top-k of the same score.
  * **attention mass coverage (p-coverage)** — the fraction of the true softmax
    mass that lands on the selected blocks: sum_{t in S} softmax(q.k_t). This is
    what actually determines output quality, and it is bounded above by the
    *scoring rule's* own ceiling, independent of the approximation.

Scoring replicates the `block_sparse_attention` flow exactly: one key centroid
per page (`CMean` over the page's keys), scored by the dot product against the
mean query `q̄ = (1/H_q) sum_h q_h` (`Mean(dim=1)` then `GeMM`), with the score
rounded to bf16 — the dtype the kernel actually keys on.

Usage
-----
::

    python algorithm_scientist/research/approx_topk_rounds.py \\
        --trace algorithm_scientist/research/traces/qwen3_4b_ruler16k.pt \\
        --block-size 32 --topk 29 --bos 1 --eos 2
"""
import argparse
import math
from pathlib import Path

import torch


def bf16_key16(x: torch.Tensor) -> torch.Tensor:
    """bf16 score -> total-order uint16, exactly as the kernel does.

    ``(bits & 0x8000) ? ~bits : (bits | 0x8000)`` — monotone in the score.
    """
    bits = x.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    neg = (bits & 0x8000) != 0
    return torch.where(neg, (~bits) & 0xFFFF, bits | 0x8000)


def simulate_gate(keys: torch.Tensor, k: int, tol: float):
    """Replicate the leaf's pass-1 histogram + gate for one row of candidates.

    Returns (n_strict, single_pass, strict_mask, tbin_mask).
      * strict_mask — high byte strictly above the threshold bin. These are
        provably the top ``n_strict`` elements.
      * tbin_mask   — high byte equal to the threshold bin; the kernel fills the
        remaining ``k - n_strict`` slots from here in atomic-arrival order.
    """
    hi = (keys >> 8) & 0xFF                                   # [P]
    hist = torch.bincount(hi, minlength=256)                  # count per bin
    # rev[b] = #elements with bin >= b   (the kernel's reverse cumsum)
    rev = torch.flip(torch.cumsum(torch.flip(hist, [0]), 0), [0])
    rev = torch.cat([rev, rev.new_zeros(1)])                  # rev[256] = 0
    # threshold bin: rev[b] > k >= rev[b+1]
    cand = ((rev[:256] > k) & (rev[1:] <= k)).nonzero()
    tbin = int(cand[0]) if len(cand) else 0
    n_strict = int(rev[tbin + 1])                             # == #(bin > tbin)
    min_strict = math.ceil((1.0 - tol) * k)
    return n_strict, n_strict >= min_strict, hi > tbin, hi == tbin


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--topk", type=int, default=29)
    ap.add_argument("--bos", type=int, default=1)
    ap.add_argument("--eos", type=int, default=2)
    ap.add_argument("--tols", default="0.0,0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fill-order", default="asc", choices=["asc", "desc", "random"],
                    help="How to model the threshold-bin fill. The real kernel "
                         "fills in nondeterministic atomic-arrival order; 'asc' "
                         "(ascending page index) biases toward early pages, which "
                         "sit nearer the BOS sink and carry more mass. Compare "
                         "orders to separate a real effect from a fill artifact.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tols = [float(x) for x in args.tols.split(",")]
    tr = torch.load(args.trace, map_location="cpu", weights_only=False)
    B, K_, dev = args.block_size, args.topk, args.device
    G = tr["G"]
    print(f"trace: {tr['model']}  data={tr.get('calibration_data')}  "
          f"layers={tr['layers']}  samples={len(tr['samples'])}  G={G}")
    print(f"block={B} topk={K_} bos={args.bos} eos={args.eos}\n")

    # accumulators keyed by tol
    n_units = 0                      # (sample, layer, kv-head) selection events
    rounds = {t: 0.0 for t in tols}
    single = {t: 0 for t in tols}
    recall = {t: 0.0 for t in tols}
    floor = {t: 0.0 for t in tols}   # guaranteed n_strict/k
    cover = {t: 0.0 for t in tols}   # attention mass, averaged over query heads
    cover_exact = 0.0                # exact top-k coverage (the rule's ceiling)
    cover_dense_head = 0             # query-head count for the coverage means
    reserved_mass = 0.0              # mass in the always-kept BOS/EOS pages
    per_layer = {}                   # li -> [n, sum_rounds@mid, sum_cover_exact, sum_G]
    MID = tols[len(tols) // 2]
    # BOS/EOS-excluded accounting: only the mass the SELECTOR is responsible for
    sel_mass = {t: 0.0 for t in tols}    # mass on the k selected pages
    sel_mass_exact = 0.0                 # same, for exact top-k
    avail_mass = 0.0                     # total mass on candidate pages

    for si, s in enumerate(tr["samples"]):
        S = s["seq_len"]
        for li, d in s["layers"].items():
            q = d["q"].to(dev).float()                     # [Hq, D]
            Kt = d["K"].to(dev).float()                    # [Hkv, S, D]
            scaling = d["scaling"]
            Hq, Dh = q.shape
            Hkv = Kt.shape[0]
            P = math.ceil(S / B)
            if P <= args.bos + args.eos + K_:
                continue                                    # nothing to select

            # ---- flow replication: page centroids + mean-query score ----
            pad = P * B - S
            Kp = torch.nn.functional.pad(Kt, (0, 0, 0, pad))          # [Hkv, P*B, D]
            Kp = Kp.view(Hkv, P, B, Dh)
            cnt = torch.full((P,), float(B), device=dev)
            cnt[-1] = B - pad if pad else B
            centroid = Kp.sum(dim=2) / cnt[None, :, None]             # [Hkv, P, D]
            qbar = q.mean(dim=0)                                      # [D]
            score = (centroid @ qbar)                                 # [Hkv, P]

            # ---- true attention distribution (per query head) ----
            attn = torch.softmax(
                (q.view(Hkv, G, Dh) @ Kt.transpose(1, 2)) * scaling, dim=-1)  # [Hkv,G,S]
            # mass per page, per query head
            am = torch.nn.functional.pad(attn, (0, pad)).view(Hkv, G, P, B).sum(-1)

            lo, hi_ = args.bos, P - args.eos                # candidate page range
            reserved = am[:, :, :lo].sum(-1) + am[:, :, hi_:].sum(-1)   # [Hkv,G]

            for h in range(Hkv):
                sc = score[h, lo:hi_]
                keys = bf16_key16(sc)
                nc = sc.numel()
                if nc <= K_:
                    continue
                n_units += 1
                mass = am[h, :, lo:hi_]                     # [G, ncand]

                exact_idx = torch.topk(sc.to(torch.bfloat16).float(), K_).indices
                thresh = sc.to(torch.bfloat16).float()[exact_idx].min()
                ce = float((reserved[h] + mass[:, exact_idx].sum(-1)).sum())
                cover_exact += ce
                cover_dense_head += G
                reserved_mass += float(reserved[h].sum())
                sel_mass_exact += float(mass[:, exact_idx].sum())
                avail_mass += float(mass.sum())
                pl = per_layer.setdefault(li, [0, 0.0, 0.0, 0])
                pl[0] += 1
                pl[2] += ce
                pl[3] += G

                for t in tols:
                    n_strict, ok, strict_m, tbin_m = simulate_gate(keys, K_, t)
                    if ok:
                        # single pass: strict winners + arrival-order fill from
                        # the threshold bin (modelled as ascending index order;
                        # the real fill is nondeterministic among equals).
                        sel = strict_m.nonzero().flatten()
                        need = K_ - sel.numel()
                        if need > 0:
                            pool = tbin_m.nonzero().flatten()
                            if args.fill_order == "desc":
                                pool = torch.flip(pool, [0])
                            elif args.fill_order == "random":
                                pool = pool[torch.randperm(pool.numel(), device=pool.device)]
                            sel = torch.cat([sel, pool[:need]])
                        rounds[t] += 1.0
                        single[t] += 1
                    else:
                        # second round resolves the low byte => exact top-k
                        sel = exact_idx
                        rounds[t] += 2.0
                    sel = sel[:K_]
                    scb = sc.to(torch.bfloat16).float()
                    recall[t] += float((scb[sel] >= thresh).sum()) / K_
                    floor[t] += min(1.0, n_strict / K_)
                    cover[t] += float((reserved[h] + mass[:, sel].sum(-1)).sum())
                    sel_mass[t] += float(mass[:, sel].sum())
                    if t == MID:
                        per_layer[li][1] += 1.0 if ok else 2.0

    print(f"selection events (sample x layer x kv-head): {n_units}")
    print(f"exact top-k attention mass coverage (the SCORING RULE's ceiling): "
          f"{cover_exact / cover_dense_head:.4f}\n")
    print(f"{'tol':>6}{'E[rounds]':>11}{'P(1 pass)':>11}{'recall@k':>10}"
          f"{'floor':>8}{'p-coverage':>12}{'d cover':>10}")
    base_cov = cover[tols[0]] / cover_dense_head
    for t in tols:
        print(f"{t:>6}{rounds[t]/n_units:>11.3f}{single[t]/n_units:>11.3f}"
              f"{recall[t]/n_units:>10.4f}{floor[t]/n_units:>8.4f}"
              f"{cover[t]/cover_dense_head:>12.4f}"
              f"{cover[t]/cover_dense_head - base_cov:>+10.4f}")

    print(f"\nof the exact-selection coverage, the always-kept BOS/EOS pages "
          f"alone carry {reserved_mass/cover_dense_head:.4f} "
          f"({reserved_mass/cover_exact*100:.1f}% of it)")

    # ---- BOS/EOS EXCLUDED: only the mass the selector is responsible for ----
    av = avail_mass / cover_dense_head
    se = sel_mass_exact / cover_dense_head
    print(f"\n=== BOS/EOS EXCLUDED — scored pages only ===")
    print(f"mass available on candidate pages (all {'~'}P-bos-eos of them): {av:.4f}")
    print(f"exact top-k captures {se:.4f} of it => "
          f"{se/av*100:.1f}% mass-recall, the scoring rule's ceiling\n")
    print(f"{'tol':>6}{'sel mass':>10}{'mass-recall':>13}{'d vs exact':>12}"
          f"{'rel loss':>10}{'recall@k':>10}")
    for t in tols:
        sm = sel_mass[t] / cover_dense_head
        print(f"{t:>6}{sm:>10.4f}{sm/av*100:>12.2f}%{sm-se:>+12.4f}"
              f"{(sm/se - 1)*100:>+9.2f}%{recall[t]/n_units:>10.4f}")
    print(f"\nper layer (E[rounds] at tol={MID}):")
    print(f"{'layer':>7}{'E[rounds]':>11}{'exact p-cov':>13}")
    for li in sorted(per_layer):
        n, r, ce, gs = per_layer[li]
        print(f"{li:>7}{r/n:>11.3f}{ce/gs:>13.4f}")


if __name__ == "__main__":
    main()
