"""Inspect model outputs from an OpenAI-compatible chat completions server.

This is intentionally simple: it sends dataset prompts one by one and writes
the full generated text next to the source row. It does not measure latency,
TTFT, TPOT, throughput, or request rate.

Example
-------
::

    python examples/misc/inspect_openai_output.py \\
        --base-url http://127.0.0.1:30000/v1 \\
        --api-key None \\
        --model Qwen/Qwen3-1.7B \\
        --prompt-file examples/validation_16K.jsonl \\
        --limit 5 \\
        --max-tokens 128 \\
        --output-file examples/inspect_output_16K.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def nested_get(record: Dict[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        value = value[part]
    return value


def load_rows(path: Path, limit: Optional[int], start: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no < start or not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def normalize_expected(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def build_extra_body(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def inspect_outputs(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai package to run this script.") from exc

    rows = load_rows(args.prompt_file, args.limit, args.start)
    extra_body = build_extra_body(args.extra_body)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    output_rows: List[Dict[str, Any]] = []

    for local_i, row in enumerate(rows):
        source_index = row.get("index", args.start + local_i)
        prompt = str(nested_get(row, args.prompt_field))
        expected = normalize_expected(row.get(args.expected_field))

        kwargs: Dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        error = None
        output_text = ""
        try:
            response = client.chat.completions.create(**kwargs)
            output_text = response.choices[0].message.content or ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        contains_expected = {
            item: item in output_text
            for item in expected
        }
        all_expected_found = (
            all(contains_expected.values()) if contains_expected else None
        )

        output_row = {
            "index": source_index,
            "length": row.get("length"),
            "prompt": prompt if args.save_prompt else None,
            "prompt_preview": prompt[: args.preview_chars],
            "expected": expected,
            "output": output_text,
            "contains_expected": contains_expected,
            "all_expected_found": all_expected_found,
            "error": error,
        }
        output_rows.append(output_row)

        preview = output_text.replace("\n", "\\n")[: args.preview_chars]
        status = "ok" if error is None else "error"
        if all_expected_found is not None:
            status += f", all_expected_found={all_expected_found}"
        print(f"[{local_i + 1}/{len(rows)} index={source_index}] {status}: {preview}")

    write_jsonl(args.output_file, output_rows)
    print(f"wrote {len(output_rows)} rows to {args.output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and inspect OpenAI-compatible model outputs.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "None"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-field", default="input")
    parser.add_argument("--expected-field", default="outputs")
    parser.add_argument("--output-file", type=Path, default=Path("examples/inspect_output.jsonl"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--extra-body", type=Path, default=None)
    parser.add_argument("--preview-chars", type=int, default=240)
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Store the full prompt in the output JSONL. By default only a preview is saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    inspect_outputs(args)


if __name__ == "__main__":
    main()
