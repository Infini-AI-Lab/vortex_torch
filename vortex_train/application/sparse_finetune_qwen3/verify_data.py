"""Verify the templating and label masking before spending GPU hours on a run.

Every check here corresponds to a mistake that would train the model on the wrong
thing while the loss curve still looked healthy — which is why this is a separate,
cheap, CPU-only script rather than something to eyeball once:

1. the reasoning block is **not** double-wrapped (``<think>`` appears exactly once);
2. the prompt is masked and the answer is not, at the exact template boundary;
3. the supervised span decodes back to the assistant turn verbatim;
4. ``<|im_end|>`` is supervised — the model must learn to stop;
5. padding contributes to neither loss nor attention;
6. lengths respect ``[min_length, max_length]``.

    python -m application.sparse_finetune_qwen3.verify_data --rows 20
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from application.sparse_finetune_qwen3.data import IGNORE, ReasoningTraces, collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--datasets", nargs="+",
                    default=["Jackrong/Qwen3.5-reasoning-700x",
                             "r0b0tlab/qwen3.8-max-distillation-50k"])
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=40960)
    ap.add_argument("--min-length", type=int, nargs="+", default=[0, 0],
                    help="per-dataset minimum; 0 here so the checks see every schema")
    ap.add_argument("--block-kv", type=int, default=64)
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", "/scratch/zhuominc/hf")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ds = ReasoningTraces(tok, max_length=a.max_length,
                         min_length=a.min_length if len(a.min_length) > 1
                         else a.min_length[0],
                         datasets=a.datasets, limit=a.rows, shuffle_buffer=0)

    think_open = tok.convert_tokens_to_ids("<think>")
    think_close = tok.convert_tokens_to_ids("</think>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    pad_id = tok.pad_token_id or tok.eos_token_id

    failures: list[str] = []
    n = 0
    lens = []
    for ex in ds:
        n += 1
        ids = ex.input_ids.tolist()
        lab = ex.labels.tolist()
        lens.append(ex.n_total)
        tag = f"row {n}"

        # 1. the reasoning block appears exactly once IN THE SUPERVISED SPAN -- not
        #    re-wrapped by the template. Counted per span, not per sequence: the
        #    qwen3.8-max system prompt legitimately contains a literal <think>/</think>
        #    because it specifies the output FORMAT, so a whole-sequence count of 2 is
        #    correct there and a count of 1 would mean the format spec went missing.
        ans_ids = ids[ex.n_prompt:]
        n_open, n_close = ans_ids.count(think_open), ans_ids.count(think_close)
        if n_open != 1 or n_close != 1:
            failures.append(f"{tag}: answer has <think> x{n_open}, </think> x{n_close} "
                            f"(expected 1 each -- double-wrapped?)")

        # 2. prompt masked, answer supervised, boundary exactly at n_prompt
        if any(l != IGNORE for l in lab[: ex.n_prompt]):
            failures.append(f"{tag}: prompt is not fully masked")
        if any(l == IGNORE for l in lab[ex.n_prompt:]):
            failures.append(f"{tag}: answer contains masked positions")
        if ex.n_prompt >= ex.n_total:
            failures.append(f"{tag}: empty answer span")

        # 3. the supervised span is exactly the assistant turn
        answer = tok.decode(ids[ex.n_prompt:])
        if not answer.startswith("<think>"):
            failures.append(f"{tag}: answer does not start with <think>: {answer[:60]!r}")
        if not answer.endswith("<|im_end|>"):
            failures.append(f"{tag}: answer does not end exactly with <|im_end|> "
                            f"(trailing separator not trimmed?): {answer[-24:]!r}")
        # the prompt must open a system or user turn, and must end with exactly one
        # assistant header (the generation prompt). A <think> inside the prompt is only
        # legitimate when it came from a system format spec -- checked below.
        prompt = tok.decode(ids[: ex.n_prompt])
        if not prompt.startswith(("<|im_start|>system", "<|im_start|>user")):
            failures.append(f"{tag}: prompt opens with neither system nor user: "
                            f"{prompt[:40]!r}")
        if prompt.count("<|im_start|>assistant") != 1:
            failures.append(f"{tag}: prompt should end with exactly one assistant header")
        if think_open in ids[: ex.n_prompt] and "<|im_start|>system" not in prompt:
            failures.append(f"{tag}: <think> in the prompt but no system turn to "
                            f"explain it -- the answer may have leaked into the mask")

        # 4. the stop token is supervised -- otherwise the model never learns to stop
        if ids[-1] != im_end:
            failures.append(f"{tag}: last token is not <|im_end|> (id {ids[-1]})")
        elif lab[-1] != im_end:
            failures.append(f"{tag}: final <|im_end|> is masked; the model would not "
                            f"learn to stop")

        # 6. length bound (upper only: per-dataset minimums are checked by the stream)
        if ex.n_total > a.max_length:
            failures.append(f"{tag}: length {ex.n_total} exceeds {a.max_length}")

        # 7. the SYSTEM turn, when the dataset has one, must survive into the prompt
        #    and must be masked. Dropping it would train the completion without the
        #    format contract it satisfies; supervising it would train the model to
        #    generate its own instructions.
        if "<|im_start|>system" in prompt:
            n_sys = prompt.count("<|im_start|>system")
            if n_sys != 1:
                failures.append(f"{tag}: {n_sys} system turns in the prompt")
            if prompt.index("<|im_start|>system") != 0:
                failures.append(f"{tag}: the system turn is not first")
        elif ex.source == "r0b0tlab/qwen3.8-max-distillation-50k":
            failures.append(f"{tag}: qwen3.8-max row lost its system prompt")

        # 5. padding is inert, and padding to a block multiple keeps blocks whole
        batch = collate([ex], pad_id=pad_id, pad_to_multiple_of=a.block_kv)
        T = batch["input_ids"].shape[1]
        if T % a.block_kv:
            failures.append(f"{tag}: padded length {T} is not a multiple of {a.block_kv}")
        if int(batch["attention_mask"].sum()) != ex.n_total:
            failures.append(f"{tag}: attention mask does not cover exactly the real tokens")
        pad_slice = batch["labels"][0, ex.n_total:]
        if pad_slice.numel() and not bool((pad_slice == IGNORE).all()):
            failures.append(f"{tag}: padded positions are not masked in labels")

    import json as _json
    print(f"checked {n} rows; stats {_json.dumps(ds.stats(), indent=2)}")
    if lens:
        lens.sort()
        print(f"lengths: min {lens[0]}  p50 {lens[len(lens)//2]}  max {lens[-1]}")
    # Show one rendering in full so the template is auditable by eye, not just asserted.
    print("\n--- example boundary (first row) ---")
    ds2 = ReasoningTraces(tok, max_length=a.max_length, min_length=0,
                          datasets=a.datasets, limit=len(a.datasets),
                          shuffle_buffer=0)
    for ex in ds2:
        print(f"  [{ex.source}]  prompt={ex.n_prompt} total={ex.n_total}")
        ids = ex.input_ids.tolist()
        print("prompt tail :", repr(tok.decode(ids[max(0, ex.n_prompt - 24):ex.n_prompt])))
        print("answer head :", repr(tok.decode(ids[ex.n_prompt:ex.n_prompt + 24])))
        print("answer tail :", repr(tok.decode(ids[-16:])))

    print()
    if failures:
        print(f"*** {len(failures)} FAILURE(S) ***")
        for f in failures[:25]:
            print("  ", f)
        sys.exit(1)
    print("ALL TEMPLATE/MASKING CHECKS PASSED")


if __name__ == "__main__":
    main()
