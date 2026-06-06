"""Build a task's eval jsonl for a specific model's tokenizer (generalized).

The eval ``prompt`` field is **tokenizer/chat-template bound**: a different model
needs a different jsonl (hence ``aime26.jsonl`` vs ``aime26_glm.jsonl`` vs
``aime26_minimax.jsonl``). This script is the task-generalized successor to
``make_aime24.py`` / ``make_aime25.py`` / ``make_aime26.py``: it downloads the
task's HF dataset, applies ``--model``'s chat template, and writes a jsonl with
the ``{id, question, answer, conversations, prompt}`` schema the runners expect.

Usage
-----
::

    python examples/misc/make_task.py --task aime26 --model Qwen/Qwen3-4B \\
        --output examples/math/aime26__qwen3_4b.jsonl

    # custom HF dataset:
    python examples/misc/make_task.py --hf-repo math-ai/aime25 --split test \\
        --model Qwen/Qwen3-4B --output examples/math/my.jsonl

The companion runner consumes it via
``run_submission.py --data examples/math/<file>.jsonl --config <submission>.json``.
The model used here MUST match the ``model_path`` in that submission's JSON.
"""

import argparse
import json
import os
import re

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import load_dataset
from transformers import AutoTokenizer


MATH_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  The last line "
    "of your response should be of the following format: 'Therefore, the final "
    "answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) "
    "where ANSWER is just the final number or expression that solves the "
    "problem. Think step by step before answering.\n\n{Question}"
)

# task -> (hf_repo, split). amc23 is a best-effort guess (no make_amc23.py
# exists); override with --hf-repo/--split if it is wrong.
TASK_DATASETS = {
    "aime24": ("HuggingFaceH4/aime_2024", "train"),
    "aime25": ("math-ai/aime25", "test"),
    "aime26": ("math-ai/aime26", "test"),
    "amc23":  ("math-ai/amc23", "test"),
}


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None,
                        help="Built-in task: aime24, aime25, aime26, amc23.")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HF dataset repo (overrides --task mapping).")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (default: per-task).")
    parser.add_argument("--model", type=str, required=True,
                        help="Model id/path whose tokenizer/chat-template to use.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL (default: examples/<task>__<model>.jsonl).")
    parser.add_argument("--question-field", type=str, default="problem")
    parser.add_argument("--answer-field", type=str, default="answer")
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    args = parser.parse_args()

    if args.hf_repo:
        hf_repo, split = args.hf_repo, (args.split or "test")
        task_label = _slug(hf_repo.split("/")[-1])
    elif args.task:
        if args.task not in TASK_DATASETS:
            raise SystemExit(
                f"unknown task {args.task!r}; built-in: "
                f"{', '.join(sorted(TASK_DATASETS))}. Or pass --hf-repo.")
        hf_repo, default_split = TASK_DATASETS[args.task]
        split = args.split or default_split
        task_label = args.task
    else:
        raise SystemExit("pass --task <name> or --hf-repo <repo>")

    output = args.output or f"examples/{task_label}__{_slug(args.model)}.jsonl"

    print(f"[make_task] task={task_label} repo={hf_repo} split={split} "
          f"model={args.model}\n[make_task] -> {output}")

    dataset = load_dataset(hf_repo, split=split)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = []
    for idx, data in enumerate(dataset):
        question = data[args.question_field]
        content = MATH_QUERY_TEMPLATE.format(Question=question)
        conversations = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        results.append({
            "id": idx,
            "question": question,
            "answer": data[args.answer_field],
            "conversations": conversations,
            "prompt": prompt,
        })

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[make_task] wrote {len(results)} entries to {output}")


if __name__ == "__main__":
    main()
