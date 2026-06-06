"""Regenerate aime26_glm.jsonl `prompt` fields with GLM-4.7-Flash thinking ON.

The original prompts were rendered with enable_thinking=False, ending in
'<|assistant|></think>' which disables chain-of-thought. We re-render each
prompt from the `conversations` field using the tokenizer's chat template at
its default (add_generation_prompt=True, thinking enabled), so the prompt ends
in '<|assistant|><think>'. All other fields are preserved verbatim.
"""
import argparse
import json
import sys

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--in-path", default="examples/math/aime26_glm.jsonl")
    ap.add_argument("--out-path", default="examples/math/aime26_glm.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    with open(args.in_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    new_lines = []
    for i, rec in enumerate(records):
        new_prompt = tok.apply_chat_template(
            rec["conversations"],
            tokenize=False,
            add_generation_prompt=True,   # thinking ON by default
        )
        if not new_prompt.endswith("<|assistant|><think>"):
            sys.exit(f"record {i}: unexpected prompt ending: {new_prompt[-40:]!r}")

        old = rec.get("prompt", "")
        # Sanity: the only intended change is the trailing think tag. The body
        # (everything up to '<|assistant|>') must be unchanged.
        if old:
            old_body = old.rsplit("<|assistant|>", 1)[0]
            new_body = new_prompt.rsplit("<|assistant|>", 1)[0]
            if old_body != new_body:
                print(f"record {i}: WARNING body changed beyond the think tag",
                      file=sys.stderr)

        rec["prompt"] = new_prompt
        new_lines.append(json.dumps(rec, ensure_ascii=False))

    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"wrote {len(new_lines)} records -> {args.out_path}")
    print("sample new ending:", repr(records[0]["prompt"][-50:]))


if __name__ == "__main__":
    main()
