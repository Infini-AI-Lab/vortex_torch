#!/usr/bin/env python
"""Train the per-head block compressor on-the-fly against a frozen HF MLA model.

No traces are written to disk: each step runs one HF forward, reconstructs the
absorbed query + latent, builds the exact block-mass distillation target, and
updates only the (tiny) compressor parameters. Only the trained compressor
weights + config are saved at the end.

    conda activate vortex_glm          # GLM needs transformers >= 5
    export HF_HOME=/raid/catalyst/models/
    CUDA_VISIBLE_DEVICES=0 python -m vortex_torch.compressor.train \
        --model zai-org/GLM-4.7-Flash --num-prompts 32 --epochs 3 \
        --proj-dim 128 --out result/compressor/glm.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .config import CompressorConfig
from .model import BlockCompressor
from .capture import MLASupervision
from . import objective as O


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    p.add_argument("--data", default="examples/ruler/validation.jsonl",
                   help="local jsonl with prompts (ignored if --hf-dataset is set).")
    p.add_argument("--field", default="input", help="prompt field in the local jsonl.")
    p.add_argument("--hf-dataset", default=None,
                   help="HF dataset id to stream prompts from (e.g. "
                        "Jackrong/GLM-5.1-Reasoning-1M-Cleaned). Overrides --data.")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-user-field", default="input",
                   help="dataset field for the user turn.")
    p.add_argument("--hf-assistant-field", default="output",
                   help="dataset field for the assistant turn (included to build a long "
                        "context; '' to use the user turn only).")
    p.add_argument("--min-tokens", type=int, default=0,
                   help="skip dataset rows whose meta input+output tokens is below this "
                        "(bias toward long contexts; uses row['meta'] when present).")
    p.add_argument("--layers", default=None,
                   help="comma list of layer indices to train (default: all MLA layers).")
    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--num-query-positions", type=int, default=1,
                   help="supervise on the last N real token positions per prompt.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="prompts per HF forward (>1 pads to the longest; pads are "
                        "excluded from supervision and loss).")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="wall-clock training budget; loops over the prompt pool until "
                        "reached (overrides --epochs when > 0).")
    p.add_argument("--log-every", type=int, default=16,
                   help="progress-log cadence in prompts (windowed averages).")
    p.add_argument("--save-every", type=int, default=0,
                   help="checkpoint cadence in prompts (0 = only per-pass + final).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--budget-blocks", type=int, default=64,
                   help="blocks kept at eval (topk_val + reserved); for recall/coverage.")
    p.add_argument("--recall-n", default="16,64,128")
    p.add_argument("--tie-qk", action="store_true")
    p.add_argument("--out", default="result/compressor/compressor.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="auto",
                   help="teacher weight dtype: 'auto' keeps the checkpoint's native "
                        "precision (e.g. bf16); or bfloat16/float16.")
    p.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                   action="store_false",
                   help="disable gradient checkpointing on the teacher (on by default).")
    p.set_defaults(grad_checkpointing=True)
    return p.parse_args()


def load_prompts(args, tokenizer):
    """Return (texts, pre_rendered). For a local jsonl the texts are raw user
    strings (the capture renders the chat template). For an HF dataset we build
    the full user+assistant turn here (a long context) and mark it pre-rendered."""
    n = args.num_prompts
    if not args.hf_dataset:
        rows = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
        return [str(r[args.field]) for r in rows], False

    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    texts = []
    for row in ds:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        tok = int(meta.get("input_tokens", 0) or 0) + int(meta.get("output_tokens", 0) or 0)
        if args.min_tokens and tok and tok < args.min_tokens:
            continue
        user = str(row.get(args.hf_user_field, "") or "")
        msgs = [{"role": "user", "content": user}]
        if args.hf_assistant_field:
            asst = str(row.get(args.hf_assistant_field, "") or "")
            if asst:
                msgs.append({"role": "assistant", "content": asst})
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:
            text = user + ("\n" + asst if args.hf_assistant_field and asst else "")
        texts.append(text)
        if len(texts) >= n:
            break
    return texts, True


def main():
    args = parse_args()
    recall_N = [int(x) for x in args.recall_n.split(",") if x]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    print(f"[train] loading {args.model} ...", flush=True)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    sup = MLASupervision(args.model, layers=layers, device=args.device,
                         dtype=dtype, num_query_positions=args.num_query_positions,
                         gradient_checkpointing=args.grad_checkpointing)
    lid2pos = {lid: i for i, lid in enumerate(sup.layer_ids)}
    print(f"[train] MLA latent_dim={sup.latent_dim} heads={sup.num_q_heads} "
          f"layers={sup.layer_ids}", flush=True)

    cfg = CompressorConfig(
        latent_dim=sup.latent_dim, num_q_heads=sup.num_q_heads,
        proj_dim=args.proj_dim, per_layer=True, num_layers=len(sup.layer_ids),
        tie_qk=args.tie_qk,
    )
    comp = BlockCompressor(cfg).to(args.device)
    opt = torch.optim.Adam(comp.parameters(), lr=args.lr)

    prompts, do_render = load_prompts(args, sup.tokenizer)
    print(f"[train] {len(prompts)} prompts × {args.epochs} epochs, "
          f"proj_dim={args.proj_dim}, lr={args.lr}, batch_size={args.batch_size}, "
          f"source={'HF:'+args.hf_dataset if args.hf_dataset else args.data}", flush=True)

    bs = args.block_size
    keys = ["p_coverage"] + [f"recall@{N}" for N in recall_N]

    def save_ckpt():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({"state_dict": comp.state_dict(), "config": cfg.__dict__,
                    "layer_ids": sup.layer_ids}, args.out)
        cfg.to_json(args.out + ".json")

    def run_one(sup_dict):
        """One optimizer step over a sequence's layers; returns (loss, eval, pooled-eval)."""
        opt.zero_grad()
        loss = 0.0; ev = evp = None
        for lid, d in sup_dict.items():
            latent = d["latent"].to(args.device)                   # [T,dim] fp16
            q_pos = d["q_abs"].to(args.device)                     # [W,H,dim] fp32
            scal = d["scaling"]; layer_pos = lid2pos[lid]
            T = latent.shape[0]; W = q_pos.shape[0]
            for j in range(W):
                Tj = T - W + 1 + j                                  # causal prefix length
                Lj = latent[:Tj].float()
                q = q_pos[j]                                        # [H,dim]
                with torch.no_grad():
                    A = O.true_attention(q, Lj, scal)
                    tgt = O.block_mass_targets(A, bs)
                    cent = O.block_centroids(Lj, bs)
                logits = comp.block_logits(q, cent, layer_pos, scal)
                loss = loss + O.distill_loss(logits, tgt)
                if j == W - 1:                                      # eval on decode query
                    ev = O.coverage_recall(logits.detach(), A, bs,
                                           args.budget_blocks, recall_N, pooled=False)
                    evp = O.coverage_recall(logits.detach(), A, bs,
                                            args.budget_blocks, recall_N, pooled=True)
        loss = loss / max(len(sup_dict), 1)
        loss.backward(); opt.step()
        return float(loss.detach()), ev, evp

    budget_s = args.max_minutes * 60.0
    t0 = time.time()
    seen = passes = 0
    win = {"loss": 0.0, "n": 0, "nev": 0}
    cov = {k: 0.0 for k in keys}; cov_p = {k: 0.0 for k in keys}
    print(f"[train] start: budget={args.max_minutes}m (0=use {args.epochs} epochs)", flush=True)

    stop = False
    while not stop:
        for sup_dict in sup.stream(prompts, render=do_render, max_tokens=args.max_tokens,
                                   batch_size=args.batch_size):
            l, ev, evp = run_one(sup_dict)
            seen += 1; win["loss"] += l; win["n"] += 1
            if ev is not None:
                for k in keys: cov[k] += ev[k]; cov_p[k] += evp[k]
                win["nev"] += 1
            if args.log_every and seen % args.log_every == 0:
                el = (time.time() - t0) / 60.0
                ne = max(win["nev"], 1)
                print(f"[t{el:5.1f}m p{seen}] loss={win['loss']/max(win['n'],1):.4f} | "
                      f"ph p-cov={cov['p_coverage']/ne:.3f} "
                      + " ".join(f"r@{N}={cov[f'recall@{N}']/ne:.3f}" for N in recall_N)
                      + f" | pooled p-cov={cov_p['p_coverage']/ne:.3f} "
                      + " ".join(f"r@{N}={cov_p[f'recall@{N}']/ne:.3f}" for N in recall_N),
                      flush=True)
                win = {"loss": 0.0, "n": 0, "nev": 0}
                cov = {k: 0.0 for k in keys}; cov_p = {k: 0.0 for k in keys}
            if args.save_every and seen % args.save_every == 0:
                save_ckpt()
                print(f"[t{(time.time()-t0)/60:.1f}m] checkpoint saved ({seen} prompts)", flush=True)
            if budget_s and (time.time() - t0) >= budget_s:
                stop = True; break
        passes += 1
        print(f"[pass {passes} complete] seen={seen} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        if not budget_s and passes >= args.epochs:
            stop = True

    save_ckpt()
    print(f"[train] saved compressor → {args.out} "
          f"(prompts={seen}, passes={passes}, elapsed={(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
