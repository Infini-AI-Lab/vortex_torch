"""Offline recall harness — under a token budget, how well does each algorithm
identify the REAL top-k attended tokens, IN EACH ATTENTION HEAD?

Given a trace from ``capture_trace.py``, for each (sample, layer) it recomputes
exact attention from the captured (q, K), then for every method (the baselines in
``methods/`` plus any ``--method path.py``) and every budget reports, per kv-head
selection:

  - **head_rec** : mean over heads of token recall@budget — fraction of a head's
                   true top-(budget×block) attended tokens that land in the
                   selected blocks. THE headline metric.
  - **worst**    : the worst single head's token recall (quality is gated by it).
  - **mass**     : mean over heads of true attention mass captured.
  - **recall**   : mass captured ÷ mass of that head's ideal top-budget blocks.
  - **out_err**  : mean relative L2 error of the per-head attention OUTPUT when
                   restricted to the selected blocks (output proxy); **within%**
                   = fraction of heads with out_err ≤ --tol.

A method may select a single shared page set (``block_scores`` → [nb], like
vortex) or per-head (``block_scores_headwise`` → [Hkv, nb], its per-head ceiling).

Usage
-----
::

    python algorithm_scientist/research/eval_recall.py \\
        --trace algorithm_scientist/research/traces/qwen3_1.7b.pt \\
        --method algorithm_scientist/research/methods/my_idea.py \\
        --budgets 0.125,0.25,0.5 --tol 0.05
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods._base import Ctx   # noqa: E402

METHODS_DIR = Path(__file__).resolve().parent / "methods"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"method_{path.stem}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, "NAME", path.stem), m


def _discover(extra):
    mods = {}
    for p in sorted(METHODS_DIR.glob("*.py")):
        if p.name.startswith("_") or p.name == "__init__.py":
            continue
        name, m = _load_module(p)
        mods[name] = m
    for e in extra or []:
        name, m = _load_module(Path(e))
        mods[name] = m
    return mods


def _ground_truth(ctx):
    """Return A[Hq,S] (exact attention) and bm[Hq,nb] (per-head block mass)."""
    rows = [torch.softmax((ctx.q[h] @ ctx.K[h // ctx.G].T) * ctx.scaling, dim=-1)
            for h in range(ctx.Hq)]
    A = torch.stack(rows)                                   # [Hq, S]
    nb = ctx.num_blocks
    idx = (torch.arange(ctx.S) // ctx.block_size)
    bm = torch.zeros(ctx.Hq, nb).index_add_(1, idx, A)      # [Hq, nb]
    return A, bm


def _sel_mask(blocks, S, bs):
    m = torch.zeros(S, dtype=torch.bool)
    for b in blocks.tolist():
        m[b * bs:(b + 1) * bs] = True
    return m


def _selection(module, ctx, budget):
    """Return a list[Hkv] of selected-block index tensors (shared or per-head)."""
    if hasattr(module, "block_scores_headwise"):
        hs = module.block_scores_headwise(ctx).float()      # [Hkv, nb]
        return [torch.argsort(hs[kv], descending=True)[:budget] for kv in range(ctx.Hkv)]
    s = module.block_scores(ctx).float()                    # [nb]
    shared = torch.argsort(s, descending=True)[:budget]
    return [shared] * ctx.Hkv


def evaluate(trace, mods, budgets, tol):
    bs, G = trace["block_size"], trace["G"]
    agg = {name: {bf: [] for bf in budgets} for name in mods}
    n_ctx = 0
    for sample in trace["samples"]:
        for li, d in sample["layers"].items():
            ctx = Ctx(q=d["q"], K=d["K"], V=d["V"], accum_pos=d["accum"],
                      scaling=d["scaling"], block_size=bs, G=G)
            A, bm = _ground_truth(ctx)
            nb = ctx.num_blocks
            n_ctx += 1
            for name, module in mods.items():
                for bf in budgets:
                    budget = max(1, round(bf * nb))
                    sel_kv = _selection(module, ctx, budget)
                    k_tok = min(ctx.S, budget * bs)
                    hrec, hmass, hrecall, herr, hwithin = [], [], [], [], []
                    for h in range(ctx.Hq):
                        kv = h // G
                        mask = _sel_mask(sel_kv[kv], ctx.S, bs)
                        # per-head token recall of the real top-k tokens
                        truek = torch.argsort(A[h], descending=True)[:k_tok]
                        hrec.append(mask[truek].float().mean().item())
                        # per-head block-mass + recall vs ideal
                        cap = bm[h][sel_kv[kv]].sum().item()
                        ideal = bm[h][torch.argsort(bm[h], descending=True)[:budget]].sum().item()
                        hmass.append(cap)
                        hrecall.append(cap / (ideal + 1e-9))
                        # output proxy
                        a = A[h]; o_full = a @ ctx.V[kv]
                        asel = a * mask; den = asel.sum()
                        o_sel = (asel / den) @ ctx.V[kv] if den > 0 else torch.zeros_like(o_full)
                        e = (torch.norm(o_sel - o_full) / (torch.norm(o_full) + 1e-6)).item()
                        herr.append(e); hwithin.append(1.0 if e <= tol else 0.0)
                    nh = ctx.Hq
                    agg[name][bf].append({
                        "head_rec": sum(hrec) / nh,
                        "worst": min(hrec),
                        "mass": sum(hmass) / nh,
                        "recall": sum(hrecall) / nh,
                        "out_err": sum(herr) / nh,
                        "within": sum(hwithin) / nh,
                    })
    return agg, n_ctx


def _mean(rows, key):
    return sum(r[key] for r in rows) / max(1, len(rows))


def main():
    ap = argparse.ArgumentParser(description="Per-head top-k token recall harness.")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--method", action="append", help="extra method .py (repeatable)")
    ap.add_argument("--budgets", default="0.125,0.25,0.5", help="fractions of blocks kept")
    ap.add_argument("--tol", type=float, default=0.05, help="output-proxy rel-err tolerance")
    args = ap.parse_args()

    trace = torch.load(args.trace, weights_only=False)
    budgets = [float(x) for x in args.budgets.split(",")]
    mods = _discover(args.method)
    agg, n_ctx = evaluate(trace, mods, budgets, args.tol)

    print(f"# per-head top-k recall — {trace['model']}  "
          f"(calib={trace.get('calibration_data','?')}, gen={trace.get('generate',0)}, "
          f"{n_ctx} layer-samples, block={trace['block_size']}, tol={args.tol})\n")
    for bf in budgets:
        print(f"## budget = {bf:.3f} of blocks  (≈{bf*100:.0f}% KV; token budget = {bf:.3f}×ctx)")
        print(f"{'method':<14} {'head_rec':>8} {'worst':>6} {'mass':>6} {'recall':>7} "
              f"{'out_err':>8} {'within%':>8}")
        for name in sorted(mods, key=lambda n: -_mean(agg[n][bf], "head_rec")):
            r = agg[name][bf]
            print(f"{name:<14} {_mean(r,'head_rec'):>8.3f} {_mean(r,'worst'):>6.3f} "
                  f"{_mean(r,'mass'):>6.3f} {_mean(r,'recall'):>7.3f} "
                  f"{_mean(r,'out_err'):>8.3f} {100*_mean(r,'within'):>7.1f}%")
        print()


if __name__ == "__main__":
    main()
