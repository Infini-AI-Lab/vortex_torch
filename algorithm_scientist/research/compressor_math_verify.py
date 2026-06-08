#!/usr/bin/env python
"""Empirical verification of the QKNorm+RoPE block-compression math (Qwen3-4B).

Hypotheses (derived in the report):
  H1  QKNorm ⇒ per-head key norms are nearly constant across tokens
      (CV small), so q·k is angle-driven and the centroid norm measures
      within-block angular coherence.
  H2  RoPE ⇒ mean-pooling over a B-token block attenuates channel pair i by
      the Dirichlet factor alpha_i = |sin(B·th_i/2)/(B·sin(th_i/2))|; measured
      per-channel centroid/key energy ratios should track alpha (with a
      ~1/sqrt(B) incoherent floor).
  H3  A trained Wk (identity-init) should learn to down-weight the
      alpha-dead high-frequency channels: per-channel row energy of Wk
      correlates with alpha.
  CTRL  Centroid/Quest granularity control: block 16/32/64 at a FIXED
      960-token budget (60/30/15 blocks) — is quest@16 better than
      learned@64 territory?

Usage (one GPU, ~10 min):
  python algorithm_scientist/research/compressor_math_verify.py \
      --ckpt result/compressor/sweep16k_b1d128.pt --out result/compressor/math_verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from vortex_torch.compressor.capture_mha import MHASupervision          # noqa: E402
from vortex_torch.compressor.model import rope_coherence               # noqa: E402
from vortex_torch.compressor import objective as O                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--ckpt", default="result/compressor/sweep16k_b1d128.pt")
    ap.add_argument("--data", default="examples/ruler/validation_16k.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--budget-tokens", type=int, default=960)
    ap.add_argument("--out", default="result/compressor/math_verify")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sup = MHASupervision(args.model, layers=None, device="cuda")
    rope_theta = float(getattr(sup.model.config, "rope_theta", 1e6))
    d = sup.head_dim
    alpha = rope_coherence(d, args.block, rope_theta)                   # [d]

    import json as _json
    rows = [
        _json.loads(line)["input"]
        for _, line in zip(range(args.n), open(args.data, encoding="utf-8"))
    ]

    cvs = []                       # H1: per (prompt, layer, kv head) norm CV
    rho_sum = torch.zeros(d)       # H2: measured channel coherence
    rho_n = 0
    gran = {16: [], 32: [], 64: []}   # CTRL: {bs: [(cent_pcov, quest_pcov), ...]}
    group_size = sup.group_size

    for sd in sup.stream(rows, render=True, max_tokens=args.max_tokens):
        for lid, dd in sd.items():
            k = dd["latent"].cuda().float()                            # [T, G, d]
            q = dd["q_abs"][-1].cuda()                                 # [H, d]
            scal = dd["scaling"]
            T = k.shape[0]
            # H1: token-norm CV per kv head
            nrm = k.norm(dim=-1)                                       # [T, G]
            cvs.append((nrm.std(dim=0) / nrm.mean(dim=0)).cpu())
            # H2: per-channel |centroid| / |key| energy ratio at block 64
            cent = O.block_centroids(k, args.block)                    # [nb, G, d]
            rho = cent.abs().mean(dim=(0, 1)) / k.abs().mean(dim=(0, 1)).clamp_min(1e-8)
            rho_sum += rho.cpu()
            rho_n += 1
            # CTRL: centroid vs quest at fixed token budget, 3 granularities
            A = O.true_attention(q, k, scal)
            for bs in (16, 32, 64):
                nb_budget = max(args.budget_tokens // bs, 1)
                cl = O.centroid_block_logits(q, O.block_centroids(k, bs), scal)
                ql = O.quest_block_logits(q, k, bs, scal)
                ce = O.coverage_recall(cl, A, bs, nb_budget, [16], pooled=True,
                                       group_size=group_size)
                qe = O.coverage_recall(ql, A, bs, nb_budget, [16], pooled=True,
                                       group_size=group_size)
                gran[bs].append((ce["p_coverage"], qe["p_coverage"],
                                 ce["recall@16"], qe["recall@16"]))

    cv = torch.stack(cvs)                                              # [N, G]
    rho = (rho_sum / max(rho_n, 1))                                    # [d]

    # H3: trained Wk channel energy (identity-init ckpt) vs alpha
    h3 = None
    if os.path.isfile(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        Wk = ck["state_dict"].get("scorer.Wk")                         # [L, G, d, r]
        if Wk is not None:
            energy = Wk.float().norm(dim=-1).mean(dim=(0, 1))          # [d]
            pair_e = (energy[:d // 2] + energy[d // 2:]) / 2           # [d/2]
            pair_a = alpha[:d // 2]
            pair_r = rho[:d // 2] if rho_n else None
            corr = torch.corrcoef(torch.stack([pair_e, pair_a]))[0, 1]
            h3 = {"wk_pair_energy": pair_e.tolist(),
                  "corr_energy_alpha": float(corr)}

    out = {
        "alpha": alpha.tolist(),
        "h1_norm_cv": {"mean": float(cv.mean()), "p95": float(cv.quantile(0.95)),
                       "max": float(cv.max())},
        "h2_rho": rho.tolist(),
        "h2_corr_rho_alpha": float(torch.corrcoef(
            torch.stack([rho[:d // 2], alpha[:d // 2]]))[0, 1]),
        "h3": h3,
        "ctrl_granularity": {
            str(bs): {
                "centroid_pcov": float(torch.tensor([x[0] for x in v]).mean()),
                "quest_pcov": float(torch.tensor([x[1] for x in v]).mean()),
                "centroid_r16": float(torch.tensor([x[2] for x in v]).mean()),
                "quest_r16": float(torch.tensor([x[3] for x in v]).mean()),
                "n": len(v),
            } for bs, v in gran.items()
        },
        "config": {"model": args.model, "block": args.block, "rope_theta": rope_theta,
                   "budget_tokens": args.budget_tokens, "n_prompts": args.n,
                   "data": args.data, "ckpt": args.ckpt},
    }
    with open(os.path.join(args.out, "math_verify.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"H1 norm CV: mean={out['h1_norm_cv']['mean']:.4f} "
          f"p95={out['h1_norm_cv']['p95']:.4f} (constant-norm shell if << 1)")
    print(f"H2 corr(rho, alpha) over pairs: {out['h2_corr_rho_alpha']:.3f}")
    if h3:
        print(f"H3 corr(trained-Wk energy, alpha): {h3['corr_energy_alpha']:.3f}")
    for bs, v in out["ctrl_granularity"].items():
        print(f"CTRL block={bs}: centroid p-cov={v['centroid_pcov']:.3f} "
              f"r16={v['centroid_r16']:.3f} | quest p-cov={v['quest_pcov']:.3f} "
              f"r16={v['quest_r16']:.3f}  (budget {args.budget_tokens} tok)")
    print(f"saved -> {args.out}/math_verify.json")


if __name__ == "__main__":
    main()
