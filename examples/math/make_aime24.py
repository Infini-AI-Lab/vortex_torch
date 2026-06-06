import json
import sys
import argparse
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path for tokenizer")
    parser.add_argument("--output", type=str, default="examples/aime24.jsonl", help="Output JSONL path")
    parser.add_argument("--enable-thinking", action="store_true", default=False, help="Enable thinking mode in chat template")
    args = parser.parse_args()

    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = []
    for idx, data in enumerate(dataset):
        content = MATH_QUERY_TEMPLATE.format(Question=data["problem"])
        conversations = [{"role": "user", "content": content}]

        prompt = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

        results.append({
            "id": idx,
            "question": data["problem"],
            "answer": data["answer"],
            "url": data["url"],
            "conversations": conversations,
            "prompt": prompt,
        })

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
