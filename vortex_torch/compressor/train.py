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
    # held-out evaluation
    p.add_argument("--eval-num-prompts", type=int, default=256,
                   help="held-out prompts per eval set (Jackrong held-out + LongBench-v2).")
    p.add_argument("--eval-longbench", default="zai-org/LongBench-v2",
                   help="HF dataset for the cross-distribution long-context eval set.")
    p.add_argument("--eval-every-min", type=float, default=45.0,
                   help="run held-out eval this often (minutes); also at start and end.")
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
    p.add_argument("--arch", default="bilinear",
                   help="block-scorer architecture: bilinear | landmark | mlp.")
    p.add_argument("--num-landmarks", type=int, default=1,
                   help="arch=landmark: sub-descriptors per block (max-scored).")
    p.add_argument("--hidden-dim", type=int, default=0,
                   help="arch=mlp: hidden width (0 = auto).")
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


def _jackrong_text(row, args, tok):
    user = str(row.get(args.hf_user_field, "") or "")
    msgs = [{"role": "user", "content": user}]
    asst = ""
    if args.hf_assistant_field:
        asst = str(row.get(args.hf_assistant_field, "") or "")
        if asst:
            msgs.append({"role": "assistant", "content": asst})
    try:
        return tok.apply_chat_template(msgs, tokenize=False)
    except Exception:
        return user + ("\n" + asst if asst else "")


def build_jackrong(args, tok):
    """Stream Jackrong once: first ``eval_num`` rows → held-out eval, next
    ``num_prompts`` rows → train (DISJOINT). Returns (train_texts, eval_texts)."""
    from datasets import load_dataset
    ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    ne, nt = args.eval_num_prompts, args.num_prompts
    ev, tr = [], []
    for row in ds:
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        ntok = int(meta.get("input_tokens", 0) or 0) + int(meta.get("output_tokens", 0) or 0)
        if args.min_tokens and ntok and ntok < args.min_tokens:
            continue
        t = _jackrong_text(row, args, tok)
        if len(ev) < ne:
            ev.append(t)
        elif len(tr) < nt:
            tr.append(t)
        else:
            break
    return tr, ev


def build_longbench(args, tok):
    """``eval_num`` LongBench-v2 prompts (Q + choices first, then the long context,
    so right-truncation keeps the question and as much context as fits)."""
    from datasets import load_dataset
    ds = load_dataset(args.eval_longbench, split="train", streaming=True)
    out = []
    for row in ds:
        q = str(row.get("question", "") or "")
        ch = "\n".join(f"{L}. {row.get('choice_' + L, '')}" for L in "ABCD")
        ctx = str(row.get("context", "") or "")
        text = f"Question: {q}\n{ch}\n\nContext:\n{ctx}"
        try:
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
        out.append(text)
        if len(out) >= args.eval_num_prompts:
            break
    return out


@torch.no_grad()
def evaluate(comp, sup, eval_sets, args, recall_N, lid2pos, bs):
    """Held-out eval: per set, mean compressor (per-head + pooled) and centroid-
    baseline (pooled) p-coverage / recall@N over (prompt × layer) examples."""
    comp.eval()
    keys = ["p_coverage"] + [f"recall@{N}" for N in recall_N]
    report = {}
    for name, texts in eval_sets:
        acc = {f"{m}:{k}": 0.0 for m in ("ph", "pool", "cent") for k in keys}
        n = 0
        for sd in sup.stream(texts, render=False, max_tokens=args.max_tokens,
                             batch_size=args.batch_size):
            for lid, d in sd.items():
                lp = lid2pos.get(lid)
                if lp is None:
                    continue
                latent = d["latent"].to(args.device).float()
                q = d["q_abs"][-1].to(args.device)
                scal = d["scaling"]
                A = O.true_attention(q, latent, scal)
                cent = O.block_centroids(latent, bs)
                ml = comp.block_logits(q, latent, bs, lp, scal)
                cl = O.centroid_block_logits(q, cent, scal)
                ph = O.coverage_recall(ml, A, bs, args.budget_blocks, recall_N, pooled=False)
                po = O.coverage_recall(ml, A, bs, args.budget_blocks, recall_N, pooled=True)
                ce = O.coverage_recall(cl, A, bs, args.budget_blocks, recall_N, pooled=True)
                for k in keys:
                    acc[f"ph:{k}"] += ph[k]; acc[f"pool:{k}"] += po[k]; acc[f"cent:{k}"] += ce[k]
                n += 1
        n = max(n, 1)
        report[name] = {k: v / n for k, v in acc.items()}
        report[name]["_n"] = n
    comp.train()
    return report


def _fmt_eval(name, r, recall_N):
    rr = lambda m: " ".join(f"r@{N}={r[f'{m}:recall@{N}']:.3f}" for N in recall_N)
    return (f"[EVAL {name} n={r['_n']}] "
            f"comp pooled p-cov={r['pool:p_coverage']:.3f} {rr('pool')} | "
            f"comp per-head p-cov={r['ph:p_coverage']:.3f} {rr('ph')} | "
            f"centroid pooled p-cov={r['cent:p_coverage']:.3f} {rr('cent')}")


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
        proj_dim=args.proj_dim, arch=args.arch, num_landmarks=args.num_landmarks,
        hidden_dim=args.hidden_dim, block_size=args.block_size,
        per_layer=True, num_layers=len(sup.layer_ids), tie_qk=args.tie_qk,
    )
    comp = BlockCompressor(cfg).to(args.device)
    opt = torch.optim.Adam(comp.parameters(), lr=args.lr)

    # Build DISJOINT train + held-out eval sets.
    eval_sets = []
    if args.hf_dataset:
        prompts, jackrong_eval = build_jackrong(args, sup.tokenizer)
        do_render = False
        if jackrong_eval:
            eval_sets.append(("jackrong_ho", jackrong_eval))
    else:
        with open(args.data, encoding="utf-8") as f:
            prompts = [json.loads(line)[args.field] for _, line in zip(range(args.num_prompts), f)]
        prompts = [str(p) for p in prompts]
        do_render = True
    if args.eval_longbench:
        print(f"[train] building LongBench-v2 eval ({args.eval_num_prompts}) ...", flush=True)
        eval_sets.append(("longbench", build_longbench(args, sup.tokenizer)))
    print(f"[train] train={len(prompts)} prompts | eval sets: "
          f"{', '.join(f'{n}={len(t)}' for n, t in eval_sets)} | "
          f"proj_dim={args.proj_dim} lr={args.lr} bs={args.batch_size} "
          f"max_tokens={args.max_tokens} source={'HF:'+args.hf_dataset if args.hf_dataset else args.data}",
          flush=True)

    bs = args.block_size
    keys = ["p_coverage"] + [f"recall@{N}" for N in recall_N]

    def run_eval(tag):
        rep = evaluate(comp, sup, eval_sets, args, recall_N, lid2pos, bs)
        for name, r in rep.items():
            print(f"  {tag} " + _fmt_eval(name, r, recall_N), flush=True)

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
                logits = comp.block_logits(q, Lj, bs, layer_pos, scal)
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

    if eval_sets:
        run_eval("[t  0.0m baseline]")
    next_eval = args.eval_every_min

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
            if eval_sets and args.eval_every_min and (time.time() - t0) / 60.0 >= next_eval:
                run_eval(f"[t{(time.time()-t0)/60:5.1f}m p{seen}]")
                next_eval += args.eval_every_min
            if budget_s and (time.time() - t0) >= budget_s:
                stop = True; break
        passes += 1
        print(f"[pass {passes} complete] seen={seen} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        if not budget_s and passes >= args.epochs:
            stop = True

    save_ckpt()
    if eval_sets:
        run_eval(f"[final t{(time.time()-t0)/60:.1f}m]")
    print(f"[train] saved compressor → {args.out} "
          f"(prompts={seen}, passes={passes}, elapsed={(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
