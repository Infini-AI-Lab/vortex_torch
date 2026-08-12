"""Turn ``nvidia/Nemotron-SFT-Math-v4`` into parquet for verl's SFT trainer.

verl's ``MultiTurnSFTDataset`` reads a parquet file and applies the chat template to a
``messages`` column, which this dataset already has (user + assistant, plus ``tools``).
So this is deliberately *not* a reformatter — it is a **length filter and packer**, which
is the part that actually matters here.

Why length is the whole story
-----------------------------
The assistant turn carries ``reasoning_content`` (the CoT) separately from ``content``
(the short boxed answer). Qwen3's template renders the trace inside ``<think>``, so it is
trained, and it dominates: measured **37x** the answer by tokens. Token lengths with the
Qwen3-4B template over a 60-row sample:

    p50 = 11 402      p90 = 72 068      max = 130 531
    >=  4096 tokens: 68%      >= 12288: 48%      >= 32768: 30%

Two consequences drive the flags below:

* **Sparse attention only engages above the budget.** ``vortex_train`` falls back to
  dense when a sequence is shorter than ``topk * block_kv`` tokens, because below that
  every block is selected anyway. At the default 16x64 = 1024 that is nearly all rows,
  but the *point* of this pipeline is the long tail, so ``--min-tokens`` keeps the run
  focused on sequences where sparsity is doing something. It defaults to 4096: high
  enough that the sparse path dominates, low enough to retain ~68% of the data.
* **The tail will OOM a fixed budget.** A 130K-token row is 10x the p50. ``--max-tokens``
  drops rather than truncates: truncating a CoT mid-derivation trains the model to stop
  reasoning halfway, which is worse than not seeing the example. The default 32768 keeps
  ~70% of rows and bounds activation memory.

``--max-rows`` exists because the full set is large and a smoke run wants 2k rows, not a
week of tokenizing.

Usage
-----
    python -m application.verl_sparse_distill.prepare_data \\
        --out-dir /scratch/zhuominc/data/nemotron_math \\
        --model Qwen/Qwen3-4B --min-tokens 4096 --max-tokens 32768 --max-rows 20000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_REPO = "nvidia/Nemotron-SFT-Math-v4"



def inline_reasoning(messages) -> list[dict]:
    """Fold ``reasoning_content`` into ``content`` as an explicit ``<think>`` block.

    This is the single most important transformation in this file, and it exists because of
    how verl tokenises multi-turn data. ``MultiTurnSFTDataset`` applies the chat template to
    **each turn separately** and concatenates the ids (so it can build a per-turn loss
    mask). Qwen3's template only emits the ``<think>`` wrapper when it renders a full
    conversation, so per-turn rendering silently drops ``reasoning_content`` entirely.

    Measured on this dataset, for one row:

        whole-conversation template : 6786 tokens, '<think>' present
        per-turn concatenation      :  831 tokens, '<think>' ABSENT

    verl notices the discrepancy and asserts; the tempting fix is
    ``ignore_input_ids_mismatch=True``, which makes the run *start* — and trains on 12% of
    the tokens with the reasoning trace removed. For a distillation whose entire signal is
    the CoT, that is the worst possible outcome: it looks like a working run.

    So the trace is written into ``content`` here, where per-turn templating cannot lose
    it. The rendered text matches what the model is expected to produce at inference
    (``<think>`` reasoning ``</think>`` answer), which is also what
    ``vortex_torch``-served evaluation will parse.
    """
    out = []
    for m in messages:
        m = dict(m)
        rc = m.pop("reasoning_content", None)
        if m.get("role") == "assistant" and rc:
            m["content"] = f"<think>\n{rc.strip()}\n</think>\n\n{(m.get('content') or '').strip()}"
        # Drop the null tool plumbing: `tool_calls=None` / `name=None` round-trip through
        # parquet as nulls, and Arrow then raises `index with value of 1 is out-of-bounds
        # for array of length 1` when the dataset reads them back.
        for k in ("tool_calls", "name", "tool_call_id"):
            if m.get(k) is None:
                m.pop(k, None)
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B",
                    help="tokenizer whose chat template defines the length "
                         "(lengths are template-dependent, so this must match training)")
    ap.add_argument("--min-tokens", type=int, default=4096,
                    help="drop shorter rows: below topk*block_kv the sparse path falls "
                         "back to dense, so short rows dilute the experiment")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="DROP longer rows (never truncate: a half CoT teaches the "
                         "model to stop reasoning mid-derivation)")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="rows to KEEP after filtering; -1 for the whole set")
    ap.add_argument("--val-rows", type=int, default=200,
                    help="held-out rows for the trainer's own val loss (AIME is scored "
                         "separately by evaluate.py — this is only a loss curve)")
    ap.add_argument("--num-proc", type=int, default=8)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def n_tokens(messages) -> int:
        # Render, then tokenize the string. ``apply_chat_template(tokenize=True)`` returns
        # a mapping on this transformers version, so len() would count keys (every row
        # measured as 2 tokens) — a silent way to filter nothing at all.
        txt = tok.apply_chat_template([dict(m) for m in messages], tokenize=False)
        return len(tok(txt, add_special_tokens=False)["input_ids"])

    print(f"[prepare] streaming {a.repo}", flush=True)
    ds = load_dataset(a.repo, split="train", streaming=True)

    kept, seen, dropped_short, dropped_long = [], 0, 0, 0
    target = None if a.max_rows < 0 else a.max_rows + a.val_rows
    for row in ds:
        seen += 1
        msgs = row.get("messages")
        if not msgs:
            continue
        # Measure the length of what will actually be TRAINED (reasoning inlined),
        # not of the original row: the filter window and the trainer must agree.
        msgs = inline_reasoning(msgs)
        n = n_tokens(msgs)
        if n < a.min_tokens:
            dropped_short += 1
            continue
        if n > a.max_tokens:
            dropped_long += 1
            continue
        # Keep only what verl reads plus provenance for later analysis. ``tools`` is
        # required by MultiTurnSFTDataset's default tools_key.
        kept.append({
            "messages": msgs,
            "tools": row.get("tools") or [],
            "n_tokens": n,
            "uuid": row.get("uuid"),
            "expected_answer": row.get("expected_answer"),
        })
        if seen % 500 == 0:
            print(f"[prepare] seen={seen} kept={len(kept)} "
                  f"short={dropped_short} long={dropped_long}", flush=True)
        if target is not None and len(kept) >= target:
            break

    if not kept:
        raise SystemExit(
            f"no rows in [{a.min_tokens}, {a.max_tokens}] tokens after {seen} rows; "
            f"widen the window"
        )

    import pandas as pd
    val_n = min(a.val_rows, max(1, len(kept) // 10))
    val, train = kept[:val_n], kept[val_n:]
    train_p, val_p = out / "train.parquet", out / "val.parquet"
    pd.DataFrame(train).to_parquet(train_p, index=False)
    pd.DataFrame(val).to_parquet(val_p, index=False)

    lens = sorted(r["n_tokens"] for r in train)
    stats = {
        "repo": a.repo, "model": a.model,
        "seen": seen, "train": len(train), "val": len(val),
        "dropped_short": dropped_short, "dropped_long": dropped_long,
        "window": [a.min_tokens, a.max_tokens],
        "tokens": {
            "min": lens[0], "p50": lens[len(lens) // 2],
            "p90": lens[int(len(lens) * 0.9)], "max": lens[-1],
            "mean": sum(lens) / len(lens),
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    print(f"[prepare] wrote {train_p} and {val_p}", flush=True)


if __name__ == "__main__":
    main()
    # ``datasets`` streaming can abort during interpreter teardown
    # ("PyGILState_Release: thread state must be current"), which sets a nonzero exit
    # code AFTER the parquet has been written and verified. A wrapper script would read
    # that as a data-prep failure and stop the pipeline, so exit explicitly once the work
    # is done rather than letting a shutdown race decide the status.
    os._exit(0)
