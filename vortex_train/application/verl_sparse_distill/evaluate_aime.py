"""Score a verl checkpoint on AIME24/AIME25 through vortex_torch + sglang.

This is the *rollout* half of the distillation loop, and deliberately the **full-GPU**
vortex path (KV in HBM, no host-KV tier): the question is whether training under a
restricted context helps a sparsely-served model, so the serving side must be the plain
configuration rather than one that also varies KV placement.

The scoring itself is ``application.sparse_finetune_qwen3.evaluate``, which already drives
``vortex_torch.engine.sgl.api.get_engine``, renders the AIME prompts, computes mean@k, and
— importantly — reads the serving budget from a ``vortex_selection.json`` *written by
training*. That file is the mechanism that stops train/serve budget drift, and it matters
because the two projects count blocks differently:

    vortex_torch:  selected = topk_val + reserved_bos + reserved_eos   (additive)
    vortex_train:  Budget.topk is the TOTAL, reservations included

So training with a total of 18 blocks must be served with ``topk_val=16, bos=1, eos=1``,
**not** ``topk_val=18``. Getting that backwards changes the context the model sees by two
blocks and quietly invalidates the comparison.

verl does not write that file, so this module's job is to **emit it** from the same
``VORTEX_SPARSE_*`` environment that configured training, then delegate. Writing it (rather
than passing flags) keeps a single source of truth: the checkpoint carries its own budget,
so a later evaluation cannot be run against the wrong one.

Usage
-----
    # same shell/env as training, or pass --topk/--block-kv explicitly
    python -m application.verl_sparse_distill.evaluate_aime \\
        --ckpt /scratch/zhuominc/ckpt_verl_sparse/global_step_200 \\
        --tasks aime24 aime25 --trials 16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def resolve_hf_dir(ckpt: Path) -> Path:
    """Find the HF-format weights inside a verl checkpoint directory.

    verl writes ``<default_local_dir>/global_step_<n>/`` with sharded FSDP state, plus a
    ``huggingface/`` subdirectory. That subdirectory always contains the config and
    tokenizer, but the **weights** only appear when ``checkpoint.save_contents`` includes
    ``hf_model`` — so its mere existence is not proof of a loadable model. Check for a
    weight file, or sglang is handed a config-only directory and fails deep in its loader
    with something far less actionable.
    """
    ckpt = ckpt.expanduser()
    if (ckpt / "config.json").is_file() and any(ckpt.glob("*.safetensors")):
        return ckpt                                      # already an HF dir
    hf = ckpt / "huggingface"
    if hf.is_dir():
        if any(hf.glob("*.safetensors")) or (hf / "pytorch_model.bin").is_file():
            return hf
        raise SystemExit(
            f"{hf} has config/tokenizer but no weights — verl writes those even when\n"
            f"`checkpoint.save_contents` excludes 'hf_model'. Either re-run with\n"
            f"    checkpoint.save_contents=[model,optimizer,extra,hf_model]\n"
            f"or merge the FSDP shards using verl's scripts/model_merger.py."
        )
    raise SystemExit(f"no HF-format weights under {ckpt} (expected {ckpt}/huggingface/)")


def write_selection(hf_dir: Path, *, topk: int, block_kv: int,
                    reserve_bos: int, reserve_local: int, reserve_eos: int) -> dict:
    """Write the ``vortex_selection.json`` that ``evaluate.py`` treats as authoritative.

    The conversion is the whole point. Training used a **total** of
    ``topk + bos + local + eos`` blocks (``vortex_train``'s ``Budget.topk`` semantics).
    Serving must select the same number under vortex_torch's additive rule, and its
    ``reserved_eos`` is the *last N blocks of the sequence* — during decode, the recent
    window. Training's ``reserve_local`` is the per-query analogue of exactly that, so it
    maps to ``reserved_eos`` here; ``reserve_eos`` (a trailing reservation during training)
    folds in as well.

        serving topk_val = total - reserved_bos - reserved_eos

    which for the defaults (16 + 1 + 1 = 18) gives ``topk_val=16, bos=1, eos=1`` — 18
    blocks, matching training exactly.
    """
    total = topk + reserve_bos + reserve_local + reserve_eos
    reserved_bos = reserve_bos
    reserved_eos = max(1, reserve_local + reserve_eos)
    topk_val = total - reserved_bos - reserved_eos
    if topk_val < 1:
        raise SystemExit(
            f"budget too small to serve: total={total} blocks minus bos={reserved_bos} "
            f"and eos={reserved_eos} leaves topk_val={topk_val}"
        )
    sel = {
        "vortex_block_size": block_kv,
        "vortex_topk_val": topk_val,
        "vortex_block_reserved_bos": reserved_bos,
        "vortex_block_reserved_eos": reserved_eos,
        "training_total_blocks": total,
        "training_budget": {
            "topk": topk, "reserve_bos": reserve_bos,
            "reserve_local": reserve_local, "reserve_eos": reserve_eos,
        },
    }
    (hf_dir / "vortex_selection.json").write_text(json.dumps(sel, indent=2))
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="verl checkpoint dir (…/global_step_N) or an HF model dir")
    ap.add_argument("--base", default="Qwen/Qwen3-4B",
                    help="untrained reference, scored in the SAME engine so the delta is "
                         "measured rather than compared against a remembered number")
    ap.add_argument("--which", nargs="+", default=["base", "trained"])
    ap.add_argument("--tasks", nargs="+", default=["aime24", "aime25"])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mem", type=float, default=0.85)
    # Training geometry. Defaults read the same env vars run_sft.sh exports, so running
    # this in the training shell needs no flags and cannot disagree with the run.
    ap.add_argument("--topk", type=int,
                    default=int(os.environ.get("VORTEX_SPARSE_TOPK", "16")))
    ap.add_argument("--block-kv", type=int,
                    default=int(os.environ.get("VORTEX_SPARSE_BLOCK_KV", "64")))
    ap.add_argument("--reserve-bos", type=int,
                    default=int(os.environ.get("VORTEX_SPARSE_RESERVE_BOS", "1")))
    ap.add_argument("--reserve-local", type=int,
                    default=int(os.environ.get("VORTEX_SPARSE_RESERVE_LOCAL", "1")))
    ap.add_argument("--reserve-eos", type=int,
                    default=int(os.environ.get("VORTEX_SPARSE_RESERVE_EOS", "0")))
    a, extra = ap.parse_known_args()

    hf_dir = resolve_hf_dir(Path(a.ckpt))
    sel = write_selection(hf_dir, topk=a.topk, block_kv=a.block_kv,
                          reserve_bos=a.reserve_bos, reserve_local=a.reserve_local,
                          reserve_eos=a.reserve_eos)
    print(f"[eval] checkpoint : {hf_dir}", flush=True)
    print(f"[eval] training   : total={sel['training_total_blocks']} blocks "
          f"({sel['training_budget']})", flush=True)
    print(f"[eval] serving    : topk_val={sel['vortex_topk_val']} + "
          f"bos={sel['vortex_block_reserved_bos']} + "
          f"eos={sel['vortex_block_reserved_eos']} = "
          f"{sel['vortex_topk_val'] + sel['vortex_block_reserved_bos'] + sel['vortex_block_reserved_eos']}"
          f" blocks x {sel['vortex_block_size']} = "
          f"{sel['training_total_blocks'] * sel['vortex_block_size']} KV tokens "
          f"(full-GPU KV)", flush=True)
    print(f"[eval] wrote {hf_dir / 'vortex_selection.json'} — evaluate.py reads this as "
          f"the source of truth", flush=True)

    cmd = [
        sys.executable, "-m", "application.sparse_finetune_qwen3.evaluate",
        "--trained", str(hf_dir), "--base", a.base,
        "--which", *a.which, "--tasks", *a.tasks,
        "--trials", str(a.trials), "--mem", str(a.mem),
    ]
    if a.out:
        cmd += ["--out", a.out]
    cmd += extra
    print("[eval] " + " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=dict(os.environ)))


if __name__ == "__main__":
    main()
