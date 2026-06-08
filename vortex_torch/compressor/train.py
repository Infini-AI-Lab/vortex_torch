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
    p.add_argument("--data", default="examples/ruler/validation_4k.jsonl",
                   help="local jsonl with prompts (ignored if --hf-dataset is set).")
    p.add_argument("--field", default="input", help="prompt field in the local jsonl.")
    p.add_argument("--hf-dataset", default=None,
                   help="comma-separated HF dataset ids to stream prompts from, with "
                        "optional '#config' (e.g. 'Jackrong/GLM-5.1-Reasoning-1M-Cleaned,"
                        "Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned#General-Distillation'). "
                        "Datasets are interleaved round-robin. Overrides --data.")
    p.add_argument("--no-pack", action="store_true",
                   help="HF datasets: do NOT pack consecutive conversations up to "
                        "--max-tokens per training sequence (default: pack, since "
                        "typical rows are ~3K tokens; the tokenizer truncation then "
                        "cuts each packed sequence at exactly --max-tokens).")
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
    p.add_argument("--eval-holdout", type=int, default=0,
                   help="in-distribution held-out eval: this many --data prompts AFTER "
                        "the train slice (0 = off).")
    p.add_argument("--eval-jsonl", action="append", default=[],
                   help="extra eval set 'name:path.jsonl:max_tokens[:num]' (repeatable) — "
                        "e.g. length-generalization sets ruler16k:examples/ruler/"
                        "validation_16k.jsonl:16384:8.")
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
    p.add_argument("--loss-topk", type=int, default=0,
                   help="DeepSeek-V3.2 DSA sparse-stage loss (Eq. 4): after the "
                        "warm-up fraction, restrict the distill KL to the per-head "
                        "union of the compressor's and the truth's top-K blocks "
                        "(0 = always full-distribution KL, Eq. 3). Try ~budget_blocks.")
    p.add_argument("--loss-topk-warmup", type=float, default=0.3,
                   help="fraction of training (by prompts/budget) to run full-KL "
                        "before switching to the top-K-restricted loss.")
    p.add_argument("--loss", default="kl", choices=("kl", "coverage", "kl+coverage"),
                   help="kl=distillation KL (default); coverage=differentiable "
                        "soft-top-k captured-mass surrogate (optimizes deployment "
                        "metric directly); kl+coverage=sum of both.")
    p.add_argument("--loss-group-pool", action="store_true",
                   help="distill the per-KV-GROUP summed logits/mass (matches the "
                        "deployed per-group selection) instead of per-head.")
    p.add_argument("--tau-per-channel", action="store_true",
                   help="gqa_envelope: learn a temperature per channel.")
    p.add_argument("--index-heads", type=int, default=4,
                   help="gqa_lightning: number of ReLU indexer heads.")
    p.add_argument("--train-lengths", default="",
                   help="mixed-length training: comma list of pack targets to cycle "
                        "(e.g. '8192,16384,32768'); empty = single --max-tokens.")
    p.add_argument("--attn", default="auto", choices=("auto", "mla", "mha"),
                   help="teacher attention kind: auto-detect (kv_lora_rank in the HF "
                        "config => mla), or force mla / mha (GQA, e.g. Qwen3-4B).")
    p.add_argument("--proj-dim", "--dc", dest="proj_dim", type=int, default=128,
                   help="d_c — descriptor channel rank (compression of head_dim/latent_dim).")
    p.add_argument("--arch", default=None,
                   help="block-scorer architecture: bilinear | landmark | factorized | "
                        "mlp | gqa_factorized (default: bilinear for MLA, "
                        "gqa_factorized for MHA).")
    p.add_argument("--num-landmarks", "--bc", dest="num_landmarks", type=int, default=1,
                   help="b_c — descriptors per block (max-scored); arch=landmark/"
                        "factorized/gqa_factorized.")
    p.add_argument("--hidden-dim", type=int, default=0,
                   help="arch=mlp/gqa_factorized_mlp: hidden width (0 = auto).")
    p.add_argument("--init", default="identity",
                   help="projection init: identity (centroid warm start) | orthogonal | "
                        "lowpass (gqa_factorized: keep the r most rope-coherent channels, "
                        "alpha-weighted — the QKNorm+RoPE math-derived warm start).")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--budget-blocks", type=int, default=64,
                   help="blocks kept at eval (topk_val + reserved); for recall/coverage.")
    p.add_argument("--recall-n", default="16,64,128")
    p.add_argument("--tie-qk", action="store_true")
    p.add_argument("--out", default="result/compressor/compressor.pt")
    p.add_argument("--shard", default="0/1",
                   help="NCCL-FREE data parallel: 'k/n' trains only on train "
                        "sequences with global index % n == k (disjoint shards "
                        "across independent single-GPU processes). Combine with "
                        "matching --seed so the runs share an init and can be "
                        "model-soup averaged afterward. Ignored under torchrun.")
    p.add_argument("--seed", type=int, default=0,
                   help="torch manual seed for the compressor init (set equal "
                        "across shards so their weights are mergeable).")
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
    """Render one dataset row as a REAL teacher-tokenizer conversation
    (user + assistant turn — including the inline ``<think>`` reasoning, which
    the Qwen3 template preserves verbatim in assistant content)."""
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


def _packed_hf_stream(args, tok):
    """Generator of packed training sequences from one or more HF datasets
    (comma-separated ``id[#config]``, round-robin interleaved). Each row is
    rendered with the teacher's chat template; unless ``--no-pack``,
    consecutive conversations are PACKED into ≈``--max-tokens`` sequences
    (estimated from ``meta`` token counts; the tokenizer truncation at
    supervision time cuts each at exactly the target). Fully lazy — nothing
    is materialized, so 64K+ prompt runs stream straight from the hub."""
    from datasets import load_dataset

    streams = []
    for part in args.hf_dataset.split(","):
        part = part.strip()
        if not part:
            continue
        ds_id, _, cfg = part.partition("#")
        ds = (load_dataset(ds_id, cfg, split=args.hf_split, streaming=True) if cfg
              else load_dataset(ds_id, split=args.hf_split, streaming=True))
        streams.append(iter(ds))

    def rows():                              # round-robin over the dataset streams
        live = list(streams)
        while live:
            nxt = []
            for it in live:
                try:
                    yield next(it)
                    nxt.append(it)
                except StopIteration:
                    pass
            live = nxt

    # mixed-length: cycle a list of pack targets (e.g. 8k/16k/32k); else single.
    targets = ([int(x) for x in args.train_lengths.split(",") if x]
               if getattr(args, "train_lengths", "") else
               ([] if args.no_pack else [args.max_tokens]))
    ti = 0
    target = targets[0] if targets else 0
    buf, buf_tok = [], 0
    for row in rows():
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        ntok = int(meta.get("input_tokens", 0) or 0) + int(meta.get("output_tokens", 0) or 0)
        if args.min_tokens and ntok and ntok < args.min_tokens:
            continue
        text = _jackrong_text(row, args, tok)
        if target:
            buf.append(text)
            buf_tok += (ntok or len(text) // 4) + 32   # + template overhead
            if buf_tok < target:
                continue
            text, buf, buf_tok = "".join(buf), [], 0
            if len(targets) > 1:                        # advance to next pack target
                ti = (ti + 1) % len(targets); target = targets[ti]
        yield text


def build_hf_mix(args, tok):
    """Held-out eval list + lazy train-stream factory over the HF mix.

    The first ``eval_num_prompts`` packed sequences become the held-out eval
    list (every rank builds the same list); ``make_train_stream(rank, world,
    limit)`` then yields up to ``limit`` LATER sequences whose global index
    ≡ rank (mod world) — disjoint from eval, disjoint across ranks, and
    equal-count per rank (keeps the per-step gradient all-reduce in
    lockstep)."""
    ne = args.eval_num_prompts
    ev = []
    for text in _packed_hf_stream(args, tok):
        ev.append(text)
        if len(ev) >= ne:
            break

    def make_train_stream(rank, world, limit):
        # Persistent across passes: each call yields the NEXT ``limit`` of this
        # rank's sequences (fresh data every pass; wraps around at dataset end).
        state = {"it": None}

        def fresh():
            for i, text in enumerate(_packed_hf_stream(args, tok)):
                if i < ne or (i - ne) % world != rank:
                    continue
                yield text

        def gen():
            if state["it"] is None:
                state["it"] = fresh()
            for _ in range(limit):
                try:
                    yield next(state["it"])
                except StopIteration:
                    state["it"] = fresh()
                    yield next(state["it"])
        return gen

    return make_train_stream, ev


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


def _dist_env():
    """torchrun support: returns (rank, world, device-or-None). Single-process
    when RANK/WORLD_SIZE are absent."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, f"cuda:{local}"
    return 0, 1, None


@torch.no_grad()
def evaluate(comp, sup, eval_sets, args, recall_N, lid2pos, bs, group_size=0,
             rank=0, world=1):
    """Held-out eval: per set, mean compressor (per-head + pooled) and baseline
    (pooled centroid; + pooled quest for MHA) p-coverage / recall@N over
    (prompt × layer) examples. ``group_size>0`` = MHA: pooling is per GQA group
    (one block set per KV head), matching the deployed decode. Eval sets are
    ``(name, texts, max_tokens)`` triples; under torchrun the prompts are
    sharded across ranks and the sums all-reduced."""
    comp.eval()
    keys = ["p_coverage"] + [f"recall@{N}" for N in recall_N]
    methods = ("ph", "pool", "cent") + (("quest",) if group_size else ())
    report = {}
    for name, texts, max_tokens in eval_sets:
        acc = {f"{m}:{k}": 0.0 for m in methods for k in keys}
        n = 0
        for sd in sup.stream(texts[rank::world], render=False, max_tokens=max_tokens,
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
                po = O.coverage_recall(ml, A, bs, args.budget_blocks, recall_N,
                                       pooled=True, group_size=group_size)
                ce = O.coverage_recall(cl, A, bs, args.budget_blocks, recall_N,
                                       pooled=True, group_size=group_size)
                for k in keys:
                    acc[f"ph:{k}"] += ph[k]; acc[f"pool:{k}"] += po[k]; acc[f"cent:{k}"] += ce[k]
                if group_size:
                    ql = O.quest_block_logits(q, latent, bs, scal)
                    qe = O.coverage_recall(ql, A, bs, args.budget_blocks, recall_N,
                                           pooled=True, group_size=group_size)
                    for k in keys:
                        acc[f"quest:{k}"] += qe[k]
                n += 1
        if world > 1:                       # all-reduce sums + count across ranks
            import torch.distributed as dist
            akeys = sorted(acc)
            buf = torch.tensor([acc[k] for k in akeys] + [float(n)],
                               device=torch.cuda.current_device())
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            acc = {k: float(buf[i]) for i, k in enumerate(akeys)}
            n = int(buf[-1])
        n = max(n, 1)
        report[name] = {k: v / n for k, v in acc.items()}
        report[name]["_n"] = n
    comp.train()
    return report


def _fmt_eval(name, r, recall_N):
    rr = lambda m: " ".join(f"r@{N}={r[f'{m}:recall@{N}']:.3f}" for N in recall_N)
    msg = (f"[EVAL {name} n={r['_n']}] "
           f"comp pooled p-cov={r['pool:p_coverage']:.3f} {rr('pool')} | "
           f"comp per-head p-cov={r['ph:p_coverage']:.3f} {rr('ph')} | "
           f"centroid pooled p-cov={r['cent:p_coverage']:.3f} {rr('cent')}")
    if "quest:p_coverage" in r:
        msg += f" | quest pooled p-cov={r['quest:p_coverage']:.3f} {rr('quest')}"
    return msg


def main():
    args = parse_args()
    rank, world, dist_dev = _dist_env()
    if dist_dev is not None:
        args.device = dist_dev
    is_main = rank == 0
    # NCCL-free data parallel: when NOT under torchrun (world==1), --shard k/n
    # selects this process's disjoint slice of the train stream. ``world`` (the
    # collective axis) stays 1 — no all-reduce, no barrier — so independent
    # single-GPU processes never touch NCCL. They share an init (--seed) and are
    # merged by model-soup averaging afterward (compressor_merge.py).
    if world == 1:
        data_rank, data_world = (int(x) for x in args.shard.split("/"))
    else:
        data_rank, data_world = rank, world
    # Each independent shard process (world==1) saves its OWN checkpoint to its
    # own --out; under torchrun only rank 0 saves the all-reduced model.
    log = print if (rank == 0) else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    recall_N = [int(x) for x in args.recall_n.split(",") if x]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    # Resolve the teacher attention kind (MLA shared latent vs MHA/GQA).
    attn_kind = args.attn
    if attn_kind == "auto":
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        attn_kind = "mla" if getattr(hf_cfg, "kv_lora_rank", None) else "mha"

    log(f"[train] loading {args.model} (attn={attn_kind}, world={world}) ...", flush=True)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    if attn_kind == "mha":
        from .capture_mha import MHASupervision
        sup = MHASupervision(args.model, layers=layers, device=args.device,
                             dtype=dtype, num_query_positions=args.num_query_positions,
                             gradient_checkpointing=args.grad_checkpointing)
        num_kv_heads = sup.num_kv_heads
        group_size = sup.group_size
        arch = args.arch or "gqa_factorized"
        log(f"[train] MHA head_dim={sup.head_dim} q_heads={sup.num_q_heads} "
            f"kv_heads={num_kv_heads} (group={group_size}) layers={sup.layer_ids}",
            flush=True)
    else:
        sup = MLASupervision(args.model, layers=layers, device=args.device,
                             dtype=dtype, num_query_positions=args.num_query_positions,
                             gradient_checkpointing=args.grad_checkpointing)
        num_kv_heads = 0
        group_size = 0
        arch = args.arch or "bilinear"
        log(f"[train] MLA latent_dim={sup.latent_dim} heads={sup.num_q_heads} "
            f"layers={sup.layer_ids}", flush=True)
    lid2pos = {lid: i for i, lid in enumerate(sup.layer_ids)}

    rope_theta = float(getattr(sup.model.config, "rope_theta", 1e6))
    cfg = CompressorConfig(
        latent_dim=sup.latent_dim, num_q_heads=sup.num_q_heads,
        num_kv_heads=num_kv_heads,
        proj_dim=args.proj_dim, arch=arch, num_landmarks=args.num_landmarks,
        hidden_dim=args.hidden_dim, block_size=args.block_size,
        per_layer=True, num_layers=len(sup.layer_ids), tie_qk=args.tie_qk,
        init=args.init, rope_theta=rope_theta,
        tau_per_channel=args.tau_per_channel, index_heads=args.index_heads,
    )
    comp = BlockCompressor(cfg).to(args.device)
    opt = torch.optim.Adam(comp.parameters(), lr=args.lr)

    # Build DISJOINT train + held-out eval sets ((name, texts, max_tokens) triples).
    eval_sets = []
    if args.hf_dataset:
        log(f"[train] streaming + packing {args.hf_dataset} ...", flush=True)
        make_train_stream, hf_eval = build_hf_mix(args, sup.tokenizer)
        prompts = None                       # lazy: built per pass below
        do_render = False
        if hf_eval:
            eval_sets.append(("hf_holdout", hf_eval, args.max_tokens))
    else:
        with open(args.data, encoding="utf-8") as f:
            rows = [json.loads(line)[args.field]
                    for _, line in zip(range(args.num_prompts + args.eval_holdout), f)]
        prompts = [str(p) for p in rows[:args.num_prompts]]
        do_render = True
        if args.eval_holdout > 0:
            ho = [sup.render(str(p)) for p in rows[args.num_prompts:]]
            if ho:
                eval_sets.append(("holdout", ho, args.max_tokens))
    # Extra jsonl eval sets, e.g. length generalization: "name:path:max_tokens[:num]".
    for spec in args.eval_jsonl:
        parts = spec.split(":")
        name, path, mt = parts[0], parts[1], int(parts[2])
        num = int(parts[3]) if len(parts) > 3 else args.eval_num_prompts
        with open(path, encoding="utf-8") as f:
            texts = [sup.render(str(json.loads(line)[args.field]))
                     for _, line in zip(range(num), f)]
        eval_sets.append((name, texts, mt))
    if args.eval_longbench:
        log(f"[train] building LongBench-v2 eval ({args.eval_num_prompts}) ...", flush=True)
        eval_sets.append(("longbench", build_longbench(args, sup.tokenizer), args.max_tokens))

    # Shard the train prompts across the DATA axis (data_world): torchrun ranks
    # OR independent --shard k/n processes. Each shard sees disjoint sequences
    # (index mod data_world); jsonl path slices the list.
    per_pass = max(args.num_prompts // data_world, 1)
    if prompts is None:
        prompt_iter = make_train_stream(data_rank, data_world, per_pass)
    else:
        if data_world > 1:
            assert len(prompts) >= data_world, f"num_prompts {len(prompts)} < shards {data_world}"
            prompts = prompts[data_rank * per_pass:(data_rank + 1) * per_pass]
        per_pass = len(prompts)
        prompt_iter = (lambda p=prompts: iter(p))
    log(f"[train] shard {data_rank}/{data_world} | train={per_pass} prompts/pass | "
        f"eval sets: {', '.join(f'{n}={len(t)}@{mt}' for n, t, mt in eval_sets)} | "
        f"arch={arch} bc={args.num_landmarks} dc={args.proj_dim} block={args.block_size} "
        f"lr={args.lr} bs={args.batch_size} seed={args.seed} "
        f"max_tokens={args.max_tokens} source={'HF:'+args.hf_dataset if args.hf_dataset else args.data}",
        flush=True)

    bs = args.block_size
    keys = ["p_coverage"] + [f"recall@{N}" for N in recall_N]

    def run_eval(tag):
        # Collective under torchrun: ALL ranks must call (metrics are all-reduced).
        rep = evaluate(comp, sup, eval_sets, args, recall_N, lid2pos, bs,
                       group_size=group_size, rank=rank, world=world)
        for name, r in rep.items():
            log(f"  {tag} " + _fmt_eval(name, r, recall_N), flush=True)

    def save_ckpt():
        if not is_main:
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save({"state_dict": comp.state_dict(), "config": cfg.__dict__,
                    "layer_ids": sup.layer_ids}, args.out)
        cfg.to_json(args.out + ".json")

    # DSA-style loss schedule: full-KL warm-up, then top-K-restricted (Eq.4).
    loss_state = {"restrict": 0}

    def run_one(sup_dict):
        """One optimizer step over a sequence's layers; returns (loss, eval, pooled-eval)."""
        opt.zero_grad()
        loss = 0.0; ev = evp = None
        rk = loss_state["restrict"]
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
                gp = group_size if args.loss_group_pool else 0
                if args.loss in ("kl", "kl+coverage"):
                    if gp:
                        Gn = logits.shape[0] // gp
                        lg = logits.view(Gn, gp, -1).sum(1)
                        tg = tgt.view(Gn, gp, -1).sum(1)
                        tg = tg / tg.sum(-1, keepdim=True).clamp_min(1e-9)
                        loss = loss + O.distill_loss(lg, tg, restrict_topk=rk)
                    else:
                        loss = loss + O.distill_loss(logits, tgt, restrict_topk=rk)
                if args.loss in ("coverage", "kl+coverage"):
                    loss = loss + O.coverage_loss(logits, tgt, args.budget_blocks,
                                                  group_size=group_size)
                if j == W - 1:                                      # eval on decode query
                    ev = O.coverage_recall(logits.detach(), A, bs,
                                           args.budget_blocks, recall_N, pooled=False)
                    evp = O.coverage_recall(logits.detach(), A, bs,
                                            args.budget_blocks, recall_N,
                                            pooled=True, group_size=group_size)
        loss = loss / max(len(sup_dict), 1)
        loss.backward()
        if world > 1:
            # Manual gradient all-reduce (the compressor is tiny, ~MBs) — avoids
            # DDP's one-forward-per-backward constraint (we run one forward per
            # layer before a single backward).
            import torch.distributed as dist
            for p in comp.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt.step()
        return float(loss.detach()), ev, evp

    budget_s = args.max_minutes * 60.0
    t0 = time.time()
    seen = passes = 0
    win = {"loss": 0.0, "n": 0, "nev": 0}
    cov = {k: 0.0 for k in keys}; cov_p = {k: 0.0 for k in keys}
    log(f"[train] start: budget={args.max_minutes}m (0=use {args.epochs} epochs)", flush=True)

    if eval_sets:
        run_eval("[t  0.0m baseline]")
    next_eval = args.eval_every_min

    def sync_flags(do_stop: bool, do_eval: bool):
        """Agree on stop/eval across ranks (MAX) so the collectives that follow
        (eval all-reduce, per-step grad all-reduce) stay in lockstep."""
        if world == 1:
            return do_stop, do_eval
        import torch.distributed as dist
        flag = torch.tensor([float(do_stop), float(do_eval)], device=args.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag[0]), bool(flag[1])

    stop = False
    while not stop:
        for sup_dict in sup.stream(prompt_iter(), render=do_render,
                                   max_tokens=args.max_tokens,
                                   batch_size=args.batch_size):
            # DSA loss schedule: switch to top-K-restricted KL after the warm-up
            # fraction (by budget if set, else by epoch fraction).
            if args.loss_topk and loss_state["restrict"] == 0:
                frac = ((time.time() - t0) / budget_s if budget_s
                        else passes / max(args.epochs, 1))
                if frac >= args.loss_topk_warmup:
                    loss_state["restrict"] = args.loss_topk
                    log(f"[t{(time.time()-t0)/60:.1f}m] switching to top-{args.loss_topk} "
                        f"restricted distill loss (DSA Eq.4)", flush=True)
            l, ev, evp = run_one(sup_dict)
            seen += 1; win["loss"] += l; win["n"] += 1
            if ev is not None:
                for k in keys: cov[k] += ev[k]; cov_p[k] += evp[k]
                win["nev"] += 1
            if args.log_every and seen % args.log_every == 0:
                el = (time.time() - t0) / 60.0
                ne = max(win["nev"], 1)
                log(f"[t{el:5.1f}m p{seen * world}] loss={win['loss']/max(win['n'],1):.4f} | "
                    f"ph p-cov={cov['p_coverage']/ne:.3f} "
                    + " ".join(f"r@{N}={cov[f'recall@{N}']/ne:.3f}" for N in recall_N)
                    + f" | pooled p-cov={cov_p['p_coverage']/ne:.3f} "
                    + " ".join(f"r@{N}={cov_p[f'recall@{N}']/ne:.3f}" for N in recall_N),
                    flush=True)
                win = {"loss": 0.0, "n": 0, "nev": 0}
                cov = {k: 0.0 for k in keys}; cov_p = {k: 0.0 for k in keys}
            if args.save_every and seen % args.save_every == 0:
                save_ckpt()
                log(f"[t{(time.time()-t0)/60:.1f}m] checkpoint saved ({seen * world} prompts)",
                    flush=True)
            el_min = (time.time() - t0) / 60.0
            stop, do_eval = sync_flags(
                bool(budget_s and el_min * 60.0 >= budget_s),
                bool(eval_sets and args.eval_every_min and el_min >= next_eval),
            )
            if do_eval:
                run_eval(f"[t{el_min:5.1f}m p{seen * world}]")
                next_eval += args.eval_every_min
            if stop:
                break
        passes += 1
        log(f"[pass {passes} complete] seen={seen * world} "
            f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        if not budget_s and passes >= args.epochs:
            stop = True

    save_ckpt()
    if eval_sets:
        run_eval(f"[final t{(time.time()-t0)/60:.1f}m]")
    log(f"[train] saved compressor → {args.out} "
        f"(prompts={seen * world}, passes={passes}, elapsed={(time.time()-t0)/60:.1f}m)",
        flush=True)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
