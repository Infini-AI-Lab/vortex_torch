import json
import argparse
from typing import Any
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import hf_hub_download


# livecodebench/code_generation_lite ships a Python loader script that
# `datasets>=4` refuses to execute. Mirror its version_tag → file mapping
# locally and stream the jsonl files directly via huggingface_hub.
LCB_VERSION_FILES: dict[str, list[str]] = {
    "v1":      ["test.jsonl"],
    "v2":      ["test.jsonl", "test2.jsonl"],
    "v3":      ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "v4":      ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "v5":      ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "v6":      ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    "latest":  ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def load_lcb(version: str):
    if version not in LCB_VERSION_FILES:
        raise ValueError(f"unknown LCB version {version!r}; expected one of {sorted(LCB_VERSION_FILES)}")
    for filename in LCB_VERSION_FILES[version]:
        path = hf_hub_download("livecodebench/code_generation_lite", filename, repo_type="dataset")
        with open(path, "r") as f:
            for line in f:
                yield json.loads(line)


def prepare_prompt(line: dict[str, Any]) -> str:
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"

    if starter_code := line.get("starter_code", None):
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path for tokenizer")
    parser.add_argument("--output", type=str, default="examples/code/lcbv5.jsonl", help="Output JSONL path")
    parser.add_argument("--version", type=str, default="v5", help="LiveCodeBench version tag (v1..v6, latest)")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard", "all"], help="Filter problems by difficulty bucket")
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="Enable thinking mode in chat template")
    args = parser.parse_args()

    dataset = load_lcb(args.version)
    if args.difficulty != "all":
        dataset = (d for d in dataset if d.get("difficulty") == args.difficulty)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = []
    for idx, data in enumerate(dataset):
        query = prepare_prompt(data)
        conversations = [{"role": "user", "content": query}]

        prompt = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

        fn_name = json.loads(data["metadata"]).get("func_name", None)

        results.append({
            "id": idx,
            "question_id": data.get("question_id"),
            "difficulty": data.get("difficulty"),
            "fn_name": fn_name,
            "conversations": conversations,
            "prompt": prompt,
        })

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
