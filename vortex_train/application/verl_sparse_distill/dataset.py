"""A verl SFT dataset that preserves long chain-of-thought.

Plugged in through verl's own extension point (``data.custom_cls``), so verl needs no
modification here either.

Why verl's built-in dataset cannot be used for this task
--------------------------------------------------------
``MultiTurnSFTDataset`` applies the chat template to **each turn separately** and
concatenates the token ids. That is a sound design — it is how it builds a per-turn loss
mask — but Qwen3's template only emits the ``<think>`` wrapper when it renders a *whole
conversation*, and it *strips* a ``<think>`` block that appears inside a turn's
``content``. Both directions therefore lose the reasoning trace. Measured on
``Nemotron-SFT-Math-v4`` with Qwen3:

    whole-conversation template : 13 094 tokens, ``<think>`` present
    verl per-turn concatenation :  1 166 tokens, ``<think>`` ABSENT

verl detects the discrepancy and asserts, which is the helpful behaviour. The tempting fix
is ``ignore_input_ids_mismatch=True``: the run then starts and trains on ~9% of the tokens
with the entire CoT deleted. For a distillation whose only signal *is* the CoT, that is
worse than a crash — it produces a plausible loss curve for an experiment that is not
happening. Inlining the trace into ``content`` at prep time does not help either, because
the template strips it back out.

So this dataset templates the conversation **once**, exactly as inference will, and derives
the loss mask by locating the assistant span rather than by re-tokenising turns.

What it returns
---------------
The same keys verl's trainer consumes: ``input_ids``, ``attention_mask``, ``position_ids``,
``loss_mask``. ``loss_mask`` is 1 on the assistant response (thinking trace included, which
is the point) and 0 on the prompt, so the prompt is conditioned on but not trained.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class ReasoningSFTDataset(Dataset):
    """Whole-conversation SFT with an assistant-span loss mask.

    Signature matches what ``verl.trainer.sft_trainer`` passes to a ``custom_cls``:
    ``(parquet_files, tokenizer, config, processor=None, max_samples=-1)``.
    """

    def __init__(self, parquet_files, tokenizer, config, processor=None,
                 max_samples: int = -1, **kwargs):
        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]
        self.tokenizer = tokenizer
        self.messages_key = config.get("messages_key", "messages")
        self.max_length = int(config.get("max_length", 32768))
        self.truncation = config.get("truncation", "error")
        assert self.truncation in ("error", "left", "right")

        frames = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        if max_samples is not None and max_samples > 0:
            df = df.iloc[:max_samples]
        self.messages = [list(m) for m in df[self.messages_key].tolist()]
        print(f"[ReasoningSFTDataset] {len(self.messages)} rows from {parquet_files}",
              flush=True)

    def __len__(self) -> int:
        return len(self.messages)

    def _render(self, msgs: list[dict]) -> tuple[str, str]:
        """Return ``(full, prompt_only)`` rendered text.

        ``prompt_only`` is everything up to where the assistant's reply begins, produced by
        templating the non-assistant turns with ``add_generation_prompt=True``. Its token
        length is where the loss mask starts, which is why it must come from the *same*
        template call convention as ``full`` — measuring the boundary any other way risks
        an off-by-a-few that silently trains on part of the prompt.
        """
        full = self.tokenizer.apply_chat_template(msgs, tokenize=False)
        head: list[dict] = []
        for m in msgs:
            if m.get("role") == "assistant":
                break
            head.append(dict(m))
        prompt = self.tokenizer.apply_chat_template(
            head, tokenize=False, add_generation_prompt=True
        )
        return full, prompt

    def __getitem__(self, i: int) -> dict:
        msgs = [dict(m) for m in self.messages[i]]
        full, prompt = self._render(msgs)

        ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        n_prompt = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        if len(ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(
                    f"row {i} is {len(ids)} tokens > max_length={self.max_length}. "
                    f"Filter the data with prepare_data.py --max-tokens instead of "
                    f"truncating: a CoT cut mid-derivation teaches the model to stop "
                    f"reasoning."
                )
            ids = ids[-self.max_length:] if self.truncation == "left" \
                else ids[: self.max_length]
            n_prompt = min(n_prompt, len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        # Train the response only. The prompt is context, so masking it keeps the loss
        # comparable to any other SFT run on this data.
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[n_prompt:] = 1
        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
