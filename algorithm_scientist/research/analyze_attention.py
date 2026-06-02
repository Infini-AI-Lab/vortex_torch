"""Profile a model's real attention to seed sparsity hypotheses.

From a ``capture_trace.py`` trace, recompute exact attention and report, per
layer (and flag notable heads): attention-sink mass, locality (recent-window
mass), how concentrated/sparse it is (top-p block coverage, effective context =
exp(entropy)), and which heads look "retrieval-y" (spread, low sink/local) vs
"local". This is the *where to be sparse* reconnaissance a human does before
designing a flow.

Usage
-----
::

    python algorithm_scientist/research/analyze_attention.py \\
        --trace algorithm_scientist/research/traces/qwen3_1.7b.pt \\
        --sink-tokens 16 --local-window 256 --p 0.9
"""

import argparse
import torch


def _attn(q, K, scaling, G):
    q = q.float(); K = K.float()        # K may be stored fp16 in the trace
    Hq = q.shape[0]
    rows = []
    for h in range(Hq):
        rows.append(torch.softmax((q[h] @ K[h // G].T) * scaling, dim=-1))
    return torch.stack(rows)        # [Hq, S]


def _top_p_blocks(mass_sorted, p):
    c = torch.cumsum(mass_sorted, 0)
    return int((c < p).sum().item()) + 1


def main():
    ap = argparse.ArgumentParser(description="Attention-pattern analyzer.")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--sink-tokens", type=int, default=16)
    ap.add_argument("--local-window", type=int, default=256)
    ap.add_argument("--p", type=float, default=0.9)
    args = ap.parse_args()

    trace = torch.load(args.trace, weights_only=False)
    bs, G = trace["block_size"], trace["G"]
    print(f"# attention profile — {trace['model']}  "
          f"(sink={args.sink_tokens}tok, local={args.local_window}tok, p={args.p})\n")
    print(f"{'layer':>5} {'sink%':>6} {'local%':>7} {'eff_ctx':>8} "
          f"{f'blk@p{int(args.p*100)}':>9} {'retr_heads':>11}")

    # collect per-layer means across samples
    per_layer = {}
    for sample in trace["samples"]:
        for li, d in sample["layers"].items():
            A = _attn(d["q"], d["K"], d["scaling"], G)           # [Hq, S]
            S = A.shape[1]
            sink = A[:, :args.sink_tokens].sum(1)                # [Hq]
            local = A[:, max(0, S - args.local_window):].sum(1)  # [Hq]
            ent = -(A.clamp_min(1e-12) * A.clamp_min(1e-12).log()).sum(1)
            eff = torch.exp(ent)                                  # effective #positions
            nb = (S + bs - 1) // bs
            idx = torch.arange(S) // bs
            bm = torch.zeros(A.shape[0], nb).index_add_(1, idx, A)
            blk_p = [_top_p_blocks(torch.sort(bm[h], descending=True).values, args.p)
                     for h in range(A.shape[0])]
            # "retrieval-y" head: low sink+local but still concentrated (few blocks for p)
            retr = ((sink + local) < 0.5) & (torch.tensor(blk_p, dtype=torch.float) < 0.33 * nb)
            row = {
                "sink": sink.mean().item(),
                "local": local.mean().item(),
                "eff": eff.mean().item(),
                "blkp": sum(blk_p) / len(blk_p),
                "retr": int(retr.sum().item()),
                "Hq": A.shape[0],
            }
            per_layer.setdefault(li, []).append(row)

    for li in sorted(per_layer):
        rs = per_layer[li]
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{li:>5} {100*m('sink'):>5.1f}% {100*m('local'):>6.1f}% "
              f"{m('eff'):>8.1f} {m('blkp'):>9.1f} {m('retr'):>6.1f}/{rs[0]['Hq']:<4}")

    print("\nReading: high sink%/local% ⇒ a sink+window flow suffices; high "
          "retr_heads + low blk@p ⇒ those heads need query-aware retrieval "
          "(centroid/quest/learned scoring). Low eff_ctx ⇒ aggressive sparsity is safe.")


if __name__ == "__main__":
    main()
