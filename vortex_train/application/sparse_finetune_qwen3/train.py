"""Sparse-attention finetuning of Qwen3-4B on GLM-5.1 reasoning traces.

    torchrun --nproc_per_node 8 -m application.sparse_finetune_qwen3.train --steps 100

Single-GPU still works (``python -m ...``) and is useful for debugging, but the
default target is **8x B200 with FSDP2**.

Configuration this application targets (all overridable):

===================  =========================================================
model                ``Qwen/Qwen3-4B`` (36 layers, Hq=32, Hkv=8, D=128)
data                 ``Jackrong/GLM-5.1-Reasoning-1M-Cleaned``, rows >= 12k tokens
max context          40960 (the model's own ``max_position_embeddings``)
selection            ``block_sparse_attention`` = centroid top-k = ``block_topk``
block_q / block_kv   1 / 64
topk                 32 blocks = **2048 KV tokens per query**, context-independent
===================  =========================================================

``block_q=1`` means **one selection per query token** — no averaging of queries into a
block. That is the expensive end of the tradeoff (measured ~7x the step of block_q=64 on
the attention kernels alone) and the accurate end: no query is forced to share a KV
budget with 63 neighbours.

Why ``min_length`` defaults to 12288: sparsity only bites above ``topk * block_kv`` =
2048 tokens, and this dataset's median row is ~3000 tokens with only ~2.8% above 16k. An
unfiltered stream would spend most steps in the regime where the sparse path degenerates
to dense — the loss would look fine and the experiment would measure nothing. The cost is
a low keep rate (~5%), which the run log reports so it cannot pass unnoticed.

Batch size is 1 **per rank** by design, not by memory: a padded batch would need the
attention mask honoured, and the sparse kernels consume a *pattern* rather than a mask.
One sequence per rank keeps every attention call on the sparse path. The effective batch
is ``world_size * grad_accum`` — 8 sequences per step on one node, which at these lengths
is 100k-300k tokens.

**Parallelism: FSDP2 (``fully_shard``), not DDP.** DDP would replicate all 4B params,
grads, and Adam state on every rank (~64 GB of optimizer state alone in fp32), leaving
little for activations at 40k context. FSDP2 shards all three across the 8 ranks, so the
memory freed goes to sequence length — which is the point of the experiment. Data
parallelism over *sequences* also keeps each rank's attention call on one unpadded
sequence, so the sparse path is never bypassed.

Each rank streams a different slice of the dataset (``seed + rank``, and a rank-strided
skip) so no sequence is trained on twice per step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from application.sparse_finetune_qwen3 import patch
from application.sparse_finetune_qwen3.data import ReasoningTraces, collate


def dist_info() -> tuple[int, int, int]:
    """(rank, local_rank, world_size) from torchrun's env, or single-process defaults."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--datasets", nargs="+",
                    default=["Jackrong/Qwen3.5-reasoning-700x",
                             "r0b0tlab/qwen3.8-max-distillation-50k"],
                    help="both are Qwen-distilled; the earlier GLM-5.1 run regressed "
                         "AIME by ~0.30, consistent with cross-model distillation")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--min-length", type=int, nargs="+", default=[12288, 0],
                    help="per-dataset minimum, in --datasets order. The long set gates "
                         "at 12288 so those steps exercise sparsity; the short set "
                         "gates at 0 because it has NO rows past 12k and would "
                         "otherwise contribute nothing")
    ap.add_argument("--block-q", type=int, default=1)
    ap.add_argument("--block-kv", type=int, default=64)
    ap.add_argument("--topk", type=int, default=16,
                    help="LEARNED top-k in vortex_torch's convention; reservations are "
                         "added on top (see patch.install)")
    ap.add_argument("--reserve-bos", type=int, default=1,
                    help="always-selected first blocks (vortex_torch reserved_bos)")
    ap.add_argument("--reserve-local", type=int, default=1,
                    help="always-selected recent blocks; the training analogue of "
                         "vortex_torch's reserved_eos")
    ap.add_argument("--reserve-eos", type=int, default=0,
                    help="always-selected last blocks of the sequence (rarely useful "
                         "in training; every position has its own recent window)")
    ap.add_argument("--algo", default="block_topk",
                    help="selection policy; block_topk == vortex's block_sparse_attention")
    ap.add_argument("--attn", choices=["vortex_sparse", "sdpa", "flash_attention_2"],
                    default="vortex_sparse", help="sdpa gives the dense baseline")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0,
                    help="advance the stream before training; use on resume so the "
                         "continuation does not re-train the same documents")
    ap.add_argument("--no-checkpointing", action="store_true",
                    help="disable gradient checkpointing (needs far more memory at 40k)")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--out", default=None, help="write a JSON run record here")
    ap.add_argument("--save", default=None,
                    help="directory to save the finetuned checkpoint into")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save every N steps (0 = only at the end)")
    ap.add_argument("--val-every", type=int, default=100,
                    help="run AIME24 validation every N steps (0 = never)")
    ap.add_argument("--val-trials", type=int, default=16)
    ap.add_argument("--dist-timeout", type=int, default=7200,
                    help="NCCL collective timeout in seconds; must exceed the slowest "
                         "validation because the non-zero ranks wait in a barrier")
    ap.add_argument("--val-dir", default=None,
                    help="where validation checkpoints + results go "
                         "(default <save>/val)")
    ap.add_argument("--init-from", default=None,
                    help="resume weights from a checkpoint dir instead of the base "
                         "model (the optimizer state is NOT restored -- see main)")
    ap.add_argument("--no-fsdp", action="store_true",
                    help="keep the model replicated (debug only; OOMs at long context)")
    return ap


def run_validation(model, tokenizer, args, step: int, rank: int, world: int, log,
                   records: list) -> None:
    """Emit a validation checkpoint. Scoring is done by a SEPARATE watcher process.

    This is the check the previous run lacked: its loss fell the whole way (1.14 ->
    0.76) while AIME24 mean@16 actually dropped 0.71 -> 0.37. Loss on distillation
    traces and task accuracy are different quantities, and only the second is the
    objective.

    Why the score is not computed here, having tried it: the sparse *training* kernels
    are not a decode path, so validation needs vortex_torch's serving engine, which
    means a second model in a second process. In-process that failed twice, and both
    failures are worth recording because they look like tuning problems and are not:

      * the seven non-validating ranks wait in a barrier while rank 0 evaluates, and
        NCCL's 600 s watchdog SIGABRTs them as a hung collective;
      * even with a longer timeout, the trainer's ~90 GB of live FSDP parameters are
        still resident -- `empty_cache()` frees cached blocks, not live tensors -- so
        the serving engine's KV allocation was OOM-killed (exit -9).

    Lowering the child's memory fraction would only postpone that: the trainer's peak
    tracks sequence length, so a fraction that fits at step 3 can OOM at step 700.

    So the trainer just writes the checkpoint and moves on. ``watch_validate.py``
    evaluates each one on an idle GPU and appends to a results file. Training never
    pauses, the engine gets clean memory, and a failed validation cannot take the run
    down. The trade is that a score arrives some minutes after its step rather than at
    the step boundary -- which for catching a regression over 1000 steps is immaterial.
    """
    val_root = Path(args.val_dir or ((args.save or "/scratch/zhuominc/ckpt") + "/val"))
    ckpt = val_root / f"step{step}"
    save_checkpoint(model, tokenizer, args, str(ckpt), rank, world, log)
    if rank == 0:
        # A marker the watcher polls for, written last so it cannot see a partial dir.
        (ckpt / "READY").write_text(f"step {step}\n")
        log(f"[val] wrote {ckpt} for the watcher to score")
    if world > 1:
        torch.distributed.barrier()


def save_checkpoint(model, tokenizer, args, path: str, rank: int, world: int, log) -> None:
    """Save a plain HF checkpoint that vortex_torch / sglang can load.

    Under FSDP2 the parameters are ``DTensor`` shards, so a naive ``save_pretrained``
    would write shards rather than tensors. ``get_model_state_dict`` with
    ``full_state_dict=True, cpu_offload=True`` gathers them to unsharded CPU tensors on
    rank 0 — which is what a serving stack expects, and it keeps peak GPU memory flat
    during the save.

    The selection geometry is written alongside as ``vortex_selection.json``, in
    **vortex_torch's convention**, so the evaluation cannot silently use a different
    budget than the model was trained with.
    """
    import json as _json

    out = Path(path)
    if world > 1:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions, get_model_state_dict,
        )
        sd = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        sd = model.state_dict()

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(out, state_dict=sd, safe_serialization=True)
        tokenizer.save_pretrained(out)
        (out / "vortex_selection.json").write_text(_json.dumps({
            "vortex_algo": args.algo,
            "vortex_block_size": args.block_kv,
            "vortex_topk_val": args.topk,
            "vortex_block_reserved_bos": args.reserve_bos,
            "vortex_block_reserved_eos": max(args.reserve_local, args.reserve_eos),
            "train_block_q": args.block_q,
            "train_datasets": args.datasets,
            "train_total_blocks": args.topk + args.reserve_bos
                                  + args.reserve_local + args.reserve_eos,
            "train_kv_tokens": (args.topk + args.reserve_bos + args.reserve_local
                                + args.reserve_eos) * args.block_kv,
            "note": "topk_val is the LEARNED top-k; vortex_torch adds bos+eos on top. "
                    "reserved_eos mirrors the training reserve_local (the recent "
                    "window), which is what eos means during decode.",
        }, indent=2))
        log(f"saved checkpoint -> {out}")
    if world > 1:
        torch.distributed.barrier()


def main() -> None:
    args = build_argparser().parse_args()
    os.environ.setdefault("HF_HOME", "/scratch/zhuominc/hf")
    rank, local_rank, world = dist_info()
    torch.cuda.set_device(local_rank)
    if world > 1:
        # A LONG timeout is load-bearing here, not defensive tuning. Validation runs on
        # rank 0 only (it shells out to a serving engine), so the other seven ranks sit
        # in a barrier for minutes. NCCL's default 600 s watchdog treats that as a hung
        # collective and SIGABRTs them -- measured: the whole run died at the first
        # validation. `timeout` must exceed the slowest validation, engine boot and
        # cudagraph capture included.
        from datetime import timedelta
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(seconds=args.dist_timeout))
    # Same seed for weights (all ranks must agree), different data slice per rank.
    torch.manual_seed(args.seed)
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.attn == "vortex_sparse":
        patch.install(algo=args.algo, topk=args.topk,
                      block_q=args.block_q, block_kv=args.block_kv,
                      reserve_bos=args.reserve_bos,
                      reserve_local=args.reserve_local,
                      reserve_eos=args.reserve_eos)

    log(f"model      {args.model}")
    log(f"attention  {args.attn}")
    if args.attn == "vortex_sparse":
        log(f"budget     {patch.describe()}")
    for d, m in zip(args.datasets, (args.min_length if len(args.min_length) > 1
                                    else args.min_length * len(args.datasets))):
        log(f"data       {d}  ({m} <= tokens <= {args.max_length})")
    log(f"parallel   world_size={world}, "
        + ("FSDP2 (fully_shard)" if world > 1 and not args.no_fsdp else "replicated")
        + f", effective batch = {world * args.grad_accum} seq/step")

    # `--init-from` continues from saved WEIGHTS only; AdamW moments and the LR
    # schedule restart. That is a real difference from a true resume, and it is stated
    # rather than hidden: for a short cosine-decay finetune the moments re-warm within
    # a few tens of steps, but the loss curve will show a small discontinuity at the
    # restart and the effective schedule is two cosines rather than one.
    init_path = args.init_from or args.model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        init_path, dtype=torch.bfloat16, attn_implementation=args.attn,
    ).cuda()
    if args.init_from:
        log(f"resumed weights from {args.init_from} "
            f"(optimizer state and LR schedule restart)")
    model.config.use_cache = False          # incompatible with training + checkpointing
    if world > 1 and not args.no_fsdp:
        from torch.distributed.fsdp import fully_shard
        # Shard each decoder layer separately so parameters are gathered per layer and
        # freed again -- one shard of the whole model would defeat the point.
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
    if not args.no_checkpointing:
        # Recompute activations: at 40k x 36 layers the stored activations dominate, and
        # sparsity does NOT reduce them (dK/dV need every KV token). Checkpointing is
        # what makes the long-context end of this run fit at all.
        model.gradient_checkpointing_enable()
    model.train()

    min_len = args.min_length if len(args.min_length) > 1 else args.min_length[0]
    dataset = ReasoningTraces(
        tokenizer, max_length=args.max_length, min_length=min_len,
        datasets=args.datasets, seed=args.seed,
        rank=rank, world_size=world, skip=args.skip_rows,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.cuda.reset_peak_memory_stats()
    records: list[dict] = []
    stream = iter(dataset)
    t_start = time.perf_counter()

    log(f"\n{'step':>5} {'loss':>8} {'lr':>9} {'tokens':>7} {'ms':>8} "
        f"{'tok/s':>8} {'peak_GB':>8} {'sparse%':>8}")
    log("-" * 74)

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        patch.reset_stats()
        n_tok = 0
        loss_sum = 0.0
        for _ in range(args.grad_accum):
            try:
                ex = next(stream)
            except StopIteration:
                log("dataset exhausted"); break
            batch = collate([ex], pad_id=pad_id, pad_to_multiple_of=args.block_kv)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            # batch of 1 and padded to a block multiple: no real padding to mask, so
            # the mask is dropped and the model uses its causal path. Passing an
            # all-ones mask would push every layer onto the dense fallback.
            batch.pop("attention_mask")
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            loss_sum += out.loss.detach().float().item()
            n_tok += ex.n_total

        if args.grad_clip:
            # FSDP2's DTensor params make this a collective; it must run on all ranks.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if world > 1:
            # Report the TRUE step: mean loss and summed tokens across ranks. A
            # rank-0-only number would understate throughput by world_size and hide
            # a rank whose loss diverged.
            agg = torch.tensor([loss_sum, float(n_tok)], device="cuda")
            torch.distributed.all_reduce(agg)
            loss_sum = agg[0].item() / world
            n_tok = int(agg[1].item())

        rec = {
            "step": step, "loss": loss_sum / args.grad_accum, "lr": lr_at(step),
            "tokens": n_tok, "ms": dt * 1000, "tok_per_s": n_tok / dt,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "sparse_frac": patch.sparse_fraction(),
            "sparse_calls": patch.STATS["sparse_calls"],
            "dense_calls": patch.STATS["dense_calls"],
        }
        records.append(rec)
        if step % args.log_every == 0 or step == args.steps - 1:
            log(f"{step:>5} {rec['loss']:>8.4f} {rec['lr']:>9.2e} {n_tok:>7} "
                f"{rec['ms']:>8.0f} {rec['tok_per_s']:>8.0f} {rec['peak_gb']:>8.1f} "
                f"{100*rec['sparse_frac']:>7.0f}%")

        if args.val_every and (step + 1) % args.val_every == 0:
            run_validation(model, tokenizer, args, step + 1, rank, world, log,
                           records)

        if args.save and args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint(model, tokenizer, args, f"{args.save}/step{step+1}",
                            rank, world, log)

    if args.save:
        save_checkpoint(model, tokenizer, args, args.save, rank, world, log)

    total_s = time.perf_counter() - t_start
    losses = [r["loss"] for r in records]
    n = len(losses)
    log(f"\n{n} steps in {total_s/60:.1f} min")
    if n >= 10:
        first, last = statistics.mean(losses[:5]), statistics.mean(losses[-5:])
        log(f"loss: first-5 mean {first:.4f} -> last-5 mean {last:.4f} "
            f"({'DECREASING' if last < first else 'NOT decreasing'})")
    log(f"data (rank {rank}): {dataset.stats()}")
    if records:
        log(f"attention: {records[-1]['sparse_calls']} sparse / "
            f"{records[-1]['dense_calls']} dense calls per step "
            f"(model has {model.config.num_hidden_layers} layers)")
        if patch.STATS["dense_reason"]:
            log(f"dense fallbacks: {patch.STATS['dense_reason']}")

    if args.out and is_main:
        Path(args.out).write_text(json.dumps(
            {"args": vars(args), "world_size": world,
             "data_rank0": dataset.stats(), "steps": records}, indent=2))
        log(f"wrote {args.out}")

    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
