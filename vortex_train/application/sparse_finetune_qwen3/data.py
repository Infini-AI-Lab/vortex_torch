"""Dataset -> token ids + labels for Qwen3 reasoning-trace finetuning.

**Templating is the whole risk in this file**, and the supported datasets do not share a
schema, so each gets an explicit adapter rather than a guessed field order. Verified
against the installed tokenizer and the live datasets:

``Jackrong/Qwen3.5-reasoning-700x``
    ``input`` / ``output`` flat strings (plus a ``conversation`` list with
    ``from``/``value`` — note the **singular** key, unlike the older dataset's
    ``conversations``). No system turn. ``output`` already contains exactly one
    ``<think>…</think>``. ``input`` already ends with "Let's think step by step and
    output the final answer within ``\\boxed{}``", so no instruction is added.
    p50 ≈ 11k tokens, 46% ≥ 12k — the long-context regime this project cares about.

``r0b0tlab/qwen3.8-max-distillation-50k``
    ``messages`` with ``role``/``content``: **system, user, assistant**. The system turn
    prescribes the exact ``<think>`` / ``\\boxed{}`` output format, and the assistant
    content already follows it. p50 ≈ 445 tokens, **zero** rows ≥ 12k.

Three mistakes this file is written to avoid, each of which would train the model on the
wrong thing while the loss curve still looked healthy:

1. **Double-wrapping the reasoning block.** Both datasets already carry
   ``<think>…</think>`` inside the assistant text, and Qwen3's chat template does not add
   those tags itself (checked with and without ``enable_thinking``: it emits the
   assistant content verbatim). Content is passed through untouched.
2. **Dropping the system prompt.** For the ``qwen3.8-max`` rows the system turn *is* the
   format contract the assistant text satisfies. Training the completion without that
   condition teaches the model to emit the format unprompted — a different objective than
   the data represents.
3. **Training on the prompt.** Loss covers only the assistant turn. The boundary comes
   from re-rendering the prompt with ``add_generation_prompt=True`` — the exact prefix
   the template emits — not from searching for a token, which breaks on any template
   change.

Layout, confirmed against the installed tokenizer::

    [<|im_start|>system\\n{sys}<|im_end|>\\n]<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n{a}<|im_end|>
    |-------------------------- masked (-100) ---------------------------|--- supervised ---|
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset

IGNORE = -100


@dataclass
class Example:
    input_ids: torch.Tensor        # [T] int64
    labels: torch.Tensor           # [T] int64, IGNORE on the prompt
    n_prompt: int
    n_total: int
    source: str = ""               # which dataset this row came from


# --------------------------------------------------------------------- adapters
def _adapt_flat(row) -> tuple[str | None, str, str] | None:
    """``input`` / ``output`` schema. No system turn."""
    user, assistant = row.get("input"), row.get("output")
    if not user or not assistant:
        return None
    return None, user, assistant


def _adapt_messages(row) -> tuple[str | None, str, str] | None:
    """``messages`` role/content schema.

    Keeps the system turn: it prescribes the ``<think>``/``\\boxed{}`` format the
    assistant content follows, so it is part of the training condition rather than
    decoration. Takes the last user/assistant pair, so a multi-turn row degrades to its
    final exchange instead of being silently mangled.
    """
    msgs = row.get("messages")
    if not msgs:
        return None
    system = next((m["content"] for m in msgs if m.get("role") == "system"), None)
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    assistants = [m["content"] for m in msgs if m.get("role") == "assistant"]
    if not users or not assistants:
        return None
    return system, users[-1], assistants[-1]


#: dataset name -> adapter. Registered explicitly: inferring the field layout from
#: whichever keys happen to exist is how a schema change becomes a silent mistraining
#: instead of an error.
ADAPTERS = {
    "Jackrong/Qwen3.5-reasoning-700x": _adapt_flat,
    "Jackrong/GLM-5.1-Reasoning-1M-Cleaned": _adapt_flat,
    "r0b0tlab/qwen3.8-max-distillation-50k": _adapt_messages,
}


def adapter_for(name: str):
    if name in ADAPTERS:
        return ADAPTERS[name]
    raise KeyError(
        f"no adapter for dataset {name!r}. Add one to ADAPTERS after inspecting its "
        f"schema — the layout must not be guessed, because a wrong guess trains on the "
        f"wrong span with a healthy-looking loss. Known: {sorted(ADAPTERS)}"
    )


# ---------------------------------------------------------------------- render
def render(tokenizer, system: str | None, user: str, assistant: str,
           max_length: int) -> Example | None:
    """Tokenise one exchange. Returns ``None`` if it does not fit.

    Truncation is deliberately not used: a reasoning trace cut mid-``<think>`` teaches
    the model to abandon reasoning without concluding, which is worse than dropping the
    example. Over-long rows are skipped and counted.
    """
    prefix = [{"role": "system", "content": system}] if system else []
    prompt_msgs = prefix + [{"role": "user", "content": user}]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": assistant}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False)

    # The full rendering must literally begin with the prompt rendering; if a template
    # change ever breaks that, the label boundary would be wrong and silently so.
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "the chat template's full rendering does not start with its own generation "
            "prompt; the prompt/answer boundary cannot be trusted"
        )

    # Qwen3's template closes the assistant turn with "<|im_end|>\n". That newline is a
    # separator for the NEXT turn, and supervising it teaches the model to emit one more
    # token after it has already stopped. Trim it so the last supervised token is the EOS
    # a sampler actually stops on.
    eos = tokenizer.eos_token or "<|im_end|>"
    if full_text.endswith(eos + "\n"):
        full_text = full_text[: -len("\n")]

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length or len(prompt_ids) >= len(full_ids):
        return None

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = IGNORE
    return Example(input_ids, labels, len(prompt_ids), len(full_ids))


class ReasoningTraces(IterableDataset):
    """Streams one or more datasets, renders each row, skips what does not fit.

    Streaming because the sources are large and a finetune of this size does not justify
    materialising them. ``IterableDataset`` also pins the worker count at 1, which avoids
    duplicated shards without a split-by-worker scheme.

    ``min_length`` exists because sparsity only engages above ``topk * block_kv`` tokens
    (1152 in this config): below that every block is selected and the sparse path *is*
    dense, so an unfiltered stream would train fine and measure nothing about sparsity.
    But the two datasets sit in different regimes — ``Qwen3.5-reasoning-700x`` is p50 11k
    while ``qwen3.8-max`` is p50 445 with **no** rows past 12k — so a single global
    ``min_length`` of 12288 would silently discard 100% of the short one. Hence
    ``min_length`` may be given **per dataset**: the long set exercises sparsity, the
    short set contributes format and answer supervision.

    Under data parallelism each rank takes a disjoint stride (``i % world == rank``),
    applied before the length filter so the split does not depend on how many rows a rank
    happens to keep.
    """

    def __init__(
        self,
        tokenizer,
        *,
        datasets: list[str] | str = "Jackrong/Qwen3.5-reasoning-700x",
        max_length: int = 40960,
        min_length: int | list[int] = 0,
        dataset_name: str | None = None,     # back-compat alias for `datasets`
        split: str = "train",
        seed: int = 0,
        shuffle_buffer: int = 1000,
        limit: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        skip: int = 0,
        interleave: bool = True,
    ) -> None:
        if dataset_name is not None:
            datasets = dataset_name
        self.tokenizer = tokenizer
        self.datasets = [datasets] if isinstance(datasets, str) else list(datasets)
        self.min_lengths = ([min_length] * len(self.datasets)
                            if isinstance(min_length, int) else list(min_length))
        if len(self.min_lengths) != len(self.datasets):
            raise ValueError(
                f"min_length has {len(self.min_lengths)} entries for "
                f"{len(self.datasets)} datasets"
            )
        for name in self.datasets:
            adapter_for(name)          # fail now, not 200 rows into a run
        self.max_length = max_length
        self.split = split
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.limit = limit
        self.rank = rank
        self.world_size = world_size
        self.skip = skip
        self.interleave = interleave
        self.counts = {n: {"yielded": 0, "too_long": 0, "too_short": 0, "malformed": 0}
                       for n in self.datasets}
        self.n_yielded = 0
        self.n_skipped = 0

    # ------------------------------------------------------------------ stream
    def _one(self, name: str, min_len: int):
        from datasets import load_dataset

        adapt = adapter_for(name)
        c = self.counts[name]
        ds = load_dataset(name, split=self.split, streaming=True)
        if self.shuffle_buffer:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        for i, row in enumerate(ds):
            if self.world_size > 1 and (i % self.world_size) != self.rank:
                continue
            got = adapt(row)
            if got is None:
                c["malformed"] += 1
                continue
            ex = render(self.tokenizer, *got, self.max_length)
            if ex is None:
                c["too_long"] += 1
                continue
            if ex.n_total < min_len:
                c["too_short"] += 1
                continue
            c["yielded"] += 1
            ex.source = name
            yield ex

    def __iter__(self):
        os.environ.setdefault("HF_HOME", "/scratch/zhuominc/hf")
        streams = [self._one(n, m) for n, m in zip(self.datasets, self.min_lengths)]

        if not self.interleave or len(streams) == 1:
            gen = (ex for s in streams for ex in s)
        else:
            # Round-robin, not concatenate: sequential streams would train entirely on
            # dataset A before ever reaching B, so the LR schedule would have decayed
            # away before B contributed anything.
            def rr():
                live = list(streams)
                while live:
                    for s in list(live):
                        try:
                            yield next(s)
                        except StopIteration:
                            live.remove(s)
            gen = rr()

        for ex in gen:
            if self.n_skipped < self.skip:
                self.n_skipped += 1
                continue
            self.n_yielded += 1
            yield ex
            if self.limit is not None and self.n_yielded >= self.limit:
                return

    def stats(self) -> dict:
        seen = sum(sum(c.values()) for c in self.counts.values())
        return {
            "yielded": self.n_yielded,
            "skipped_for_resume": self.n_skipped,
            "seen": seen,
            "keep_rate": (self.n_yielded / seen) if seen else 0.0,
            "per_dataset": self.counts,
        }


def collate(batch: list[Example], *, pad_id: int, pad_to_multiple_of: int = 64):
    """Right-pad to a multiple of ``block_kv`` and return the model's kwargs.

    Padding to a block multiple keeps the last KV block whole: the kernels mask a ragged
    tail correctly either way, but a partly-padded block still participates in scoring,
    and a centroid computed over padding is meaningless.
    """
    t_max = max(e.n_total for e in batch)
    if pad_to_multiple_of:
        t_max = -(-t_max // pad_to_multiple_of) * pad_to_multiple_of

    ids = torch.full((len(batch), t_max), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), t_max), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), t_max), dtype=torch.long)
    for i, e in enumerate(batch):
        n = e.n_total
        ids[i, :n] = e.input_ids
        lab[i, :n] = e.labels
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}
