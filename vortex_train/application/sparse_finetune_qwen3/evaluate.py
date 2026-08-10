"""Evaluate a finetuned checkpoint in vortex_torch (sglang) on AIME24 / AIME25.

Scores the **trained** and **untrained** models under the *same* sparse-attention budget
the model was finetuned with, so the comparison isolates the effect of training rather
than of the sparsity configuration.

    python -m application.sparse_finetune_qwen3.evaluate \\
        --trained /scratch/zhuominc/ckpt_1k --base Qwen/Qwen3-4B \\
        --tasks aime24 aime25 --trials 16

**Why this drives the engine directly instead of calling ``examples/math/verify_algo.py``:**
that harness hardcodes ``vortex_block_reserved_bos=1, vortex_block_reserved_eos=2`` and
defaults ``vortex_layers_skip=[0]``. Both silently change the budget relative to training
(19 selected blocks instead of 18, and layer 0 left dense), which is exactly the kind of
mismatch that makes a train/serve comparison meaningless. Driving ``sgl.Engine`` here
keeps every budget parameter explicit and printed.

Budget matching, stated once because the two conventions genuinely differ:

* vortex_torch: ``selected = topk_val + reserved_bos + reserved_eos`` — reservations are
  **additive**.
* vortex_train: ``Budget.topk`` is the **total**, reservations included.

So training with ``topk=16, reserve_bos=1, reserve_local=1`` attends 18 blocks, and
serving must use ``topk_val=16, bos=1, eos=1`` — *not* ``topk_val=18``. The checkpoint's
``vortex_selection.json`` is the source of truth and the script refuses to run on a
mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

VORTEX_TORCH = Path(os.environ.get("VORTEX_TORCH", "/scratch/zhuominc/vortex_torch"))


# --------------------------------------------------------------------- answers
def extract_answer(text: str) -> str | None:
    """Pull the final integer answer out of a reasoning trace.

    AIME answers are integers 0-999. ``\\boxed{}`` is the convention the prompts ask
    for; the fallbacks exist because a model that runs out of tokens mid-``</think>``
    still often states its answer, and scoring that as wrong would conflate "bad at
    math" with "hit the token limit".
    """
    if not text:
        return None
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        cand = m[-1]
    else:
        m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*\**\s*(-?\d+)", text, re.I)
        if m:
            cand = m[-1]
        else:
            nums = re.findall(r"-?\d+", text[-400:])
            if not nums:
                return None
            cand = nums[-1]
    digits = re.findall(r"-?\d+", cand.replace(",", ""))
    return digits[-1] if digits else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return int(pred) == int(str(gold).strip())
    except ValueError:
        return str(pred).strip() == str(gold).strip()


# ------------------------------------------------------------------- selection
def load_selection(ckpt: str) -> dict:
    p = Path(ckpt) / "vortex_selection.json"
    return json.loads(p.read_text()) if p.exists() else {}


def build_engine(model_path: str, sel: dict, a):
    """Construct the engine through vortex_torch's own ``get_engine``.

    Deliberately NOT ``sglang.Engine(**vortex_kwargs)``: the ``vortex_*`` names are not
    ``ServerArgs`` fields (checked -- all 15 are rejected). vortex collects them into a
    ``VortexConfig`` at its own wrapper, which also installs the **schedule policy**
    that computes the budget:

        static = topk_val + block_reserved_bos + block_reserved_eos
        return max(static, cached_block_len * topk_ratio)

    With ``topk_val=16, bos=1, eos=1`` the static term is 18 blocks — exactly what
    training attended. ``topk_ratio`` must stay 0, or the dynamic term would exceed 18
    on long contexts and serving would silently attend more than training did.

    ``get_engine`` also sets ``page_size = vortex_block_size``, so the KV page and the
    selection block are the same 64 tokens.
    """
    if a.use_tensor_core and a.vortex_impl_backend != "triton":
        raise SystemExit(
            "--use-tensor-core requires --vortex-impl-backend triton; pass "
            "--no-tensor-core to use the cuda backend's fp32 path instead"
        )
    if a.full_attention:
        import sglang as sgl
        return sgl.Engine(
            model_path=model_path, tp_size=a.tp, mem_fraction_static=a.mem,
            context_length=a.context_length, kv_cache_dtype=a.kv_cache_dtype,
            attention_backend=a.attention_backend, trust_remote_code=True,
            disable_cuda_graph=a.disable_cuda_graph,
        )

    from vortex_torch.engine.sgl.api import get_engine

    return get_engine(
        model_path=model_path,
        vortex_block_size=sel["vortex_block_size"],
        vortex_topk_val=sel["vortex_topk_val"],
        vortex_topk_ratio=a.topk_ratio,
        vortex_block_reserved_bos=sel["vortex_block_reserved_bos"],
        vortex_block_reserved_eos=sel["vortex_block_reserved_eos"],
        # [] not the default [0]: training left NO layer dense, so neither may serving.
        vortex_layers_skip=a.layers_skip,
        vortex_module_name=a.serving_algo,
        # built-in flow -> no submission file to load
        vortex_module_path=None,
        vortex_impl_backend=a.vortex_impl_backend,
        vortex_use_tensor_core=a.use_tensor_core,
        vortex_max_seq_lens=a.context_length,
        vortex_workload_chunk_size=a.workload_chunk_size,
        kv_cache_dtype=a.kv_cache_dtype,
        mem_fraction_static=a.mem,
        # extras forwarded to sgl.Engine
        tp_size=a.tp,
        context_length=a.context_length,
        trust_remote_code=True,
        vortex_attention_backend=a.vortex_attention_backend,
        disable_cuda_graph=a.disable_cuda_graph,
    )


def run_task(engine, tokenizer, rows, a, tag: str) -> dict:
    """Sample ``trials`` completions per problem and score mean@k."""
    prompts, golds = [], []
    for row in rows:
        # The dataset's `prompt` field is tokenizer-bound; re-render from the raw
        # question with THIS model's template so the eval matches how it was trained.
        q = row.get("question") or row.get("problem") or row.get("prompt")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
        )
        for _ in range(a.trials):
            prompts.append(text)
            golds.append(str(row.get("answer", row.get("gt", ""))))

    sampling = {"temperature": a.temperature, "top_p": a.top_p,
                "max_new_tokens": a.max_new_tokens}
    t0 = time.perf_counter()
    outs = engine.generate(prompts, sampling)
    dt = time.perf_counter() - t0

    n_prob = len(rows)
    per_problem = [0] * n_prob
    n_tok = 0
    for i, o in enumerate(outs):
        text = o["text"] if isinstance(o, dict) else str(o)
        meta = o.get("meta_info", {}) if isinstance(o, dict) else {}
        n_tok += meta.get("completion_tokens", 0)
        if is_correct(extract_answer(text), golds[i]):
            per_problem[i // a.trials] += 1

    mean_at_k = sum(c / a.trials for c in per_problem) / n_prob
    solved_any = sum(1 for c in per_problem if c > 0)
    return {
        "tag": tag, "problems": n_prob, "trials": a.trials,
        f"mean@{a.trials}": mean_at_k,
        "pass@k": solved_any / n_prob,
        "per_problem_correct": per_problem,
        "wall_s": dt, "completion_tokens": n_tok,
        "tok_per_s": n_tok / dt if dt else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--tasks", nargs="+", default=["aime24", "aime25"])
    ap.add_argument("--which", nargs="+", default=["base", "trained"],
                    choices=["trained", "base"])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--context-length", type=int, default=40960)
    ap.add_argument("--workload-chunk-size", type=int, default=32,
                    help="vortex indexer chunking; page_size is set by get_engine to "
                         "vortex_block_size, so it is not a separate knob here")
    ap.add_argument("--topk-ratio", type=float, default=0.0,
                    help="0 keeps the budget context-independent, as in training")
    ap.add_argument("--layers-skip", type=int, nargs="*", default=[],
                    help="layers left dense; [] matches training (harness default [0] "
                         "does not)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--attention-backend", default="flashinfer")
    ap.add_argument("--vortex-attention-backend", default="trtllm")
    # `triton` + tensor core: vortex rejects use_tensor_core on the `cuda` backend
    # (tensor-core codegen exists only in the triton W-kernel path; cuda has its own
    # fp32-accumulation path and would silently ignore the flag, so vortex raises).
    # This pairing also matches the configuration the project's earlier AIME sweeps used.
    ap.add_argument("--vortex-impl-backend", default="triton",
                    choices=["triton", "cuda"])
    ap.add_argument("--no-tensor-core", dest="use_tensor_core",
                    action="store_false", default=True,
                    help="disable bf16 tensor-core indexer codegen (required if "
                         "--vortex-impl-backend cuda)")
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--full-attention", action="store_true",
                    help="dense reference; ignores the sparse budget")
    ap.add_argument("--serving-algo", default="block_sparse_attention")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/scratch/zhuominc/hf")
    sys.path.insert(0, str(VORTEX_TORCH))

    sel = load_selection(a.trained)
    if not sel:
        print(f"no vortex_selection.json in {a.trained}; using the goal's config")
        sel = {"vortex_block_size": 64, "vortex_topk_val": 16,
               "vortex_block_reserved_bos": 1, "vortex_block_reserved_eos": 1,
               "train_total_blocks": 18, "train_kv_tokens": 1152, "train_block_q": 1}

    total = (sel["vortex_topk_val"] + sel["vortex_block_reserved_bos"]
             + sel["vortex_block_reserved_eos"])
    print("=" * 78)
    print("BUDGET MATCHING  (vortex_torch: selected = topk_val + bos + eos)")
    print(f"  block_size (= train block_kv) : {sel['vortex_block_size']}")
    print(f"  topk_val (learned)            : {sel['vortex_topk_val']}")
    print(f"  reserved bos / eos            : {sel['vortex_block_reserved_bos']} / "
          f"{sel['vortex_block_reserved_eos']}")
    print(f"  selected blocks               : {total} "
          f"({total * sel['vortex_block_size']} KV tokens)")
    if "train_total_blocks" in sel:
        ok = total == sel["train_total_blocks"]
        print(f"  training attended             : {sel['train_total_blocks']} blocks "
              f"({sel['train_kv_tokens']} tokens)   "
              f"{'MATCHED' if ok else '*** MISMATCH -- refusing to run ***'}")
        if not ok:
            sys.exit(1)
    print(f"  layers left dense             : {a.layers_skip or 'none'}")
    print(f"  topk_ratio                    : {a.topk_ratio} "
          f"(must be 0 so the dynamic floor never exceeds the static {total})")
    print(f"  sparsity                      : "
          f"{'DISABLED (--full-attention)' if a.full_attention else 'enabled'}")
    print("=" * 78, flush=True)
    if a.dry_run:
        return

    from transformers import AutoTokenizer

    all_results = []
    for which in a.which:
        model = a.trained if which == "trained" else a.base
        print(f"\n### loading {which}: {model}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model)
        engine = build_engine(model, sel, a)
        try:
            for task in a.tasks:
                data = VORTEX_TORCH / "examples" / "math" / f"{task}.jsonl"
                if not data.exists():
                    print(f"  skip {task}: {data} missing")
                    continue
                rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
                r = run_task(engine, tokenizer, rows, a, f"{task}/{which}")
                r.update(task=task, model=which, model_path=model)
                all_results.append(r)
                print(f"  {task}/{which}: mean@{a.trials}={r[f'mean@{a.trials}']:.4f} "
                      f"pass@k={r['pass@k']:.4f} "
                      f"({r['completion_tokens']} tok, {r['tok_per_s']:.0f} tok/s)",
                      flush=True)
        finally:
            engine.shutdown()

    print("\n" + "=" * 78)
    print(f"{'task':>8} {'model':>10} {f'mean@{a.trials}':>10} {'pass@k':>8} {'tok/s':>8}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['task']:>8} {r['model']:>10} {r[f'mean@{a.trials}']:>10.4f} "
              f"{r['pass@k']:>8.4f} {r['tok_per_s']:>8.0f}")
    # the delta is the point of the run
    for task in a.tasks:
        got = {r["model"]: r[f"mean@{a.trials}"] for r in all_results if r["task"] == task}
        if "trained" in got and "base" in got:
            d = got["trained"] - got["base"]
            print(f"\n{task}: trained - base = {d:+.4f} "
                  f"({got['base']:.4f} -> {got['trained']:.4f})")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"selection": sel, "args": vars(a), "results": all_results}, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
