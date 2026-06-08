"""Benchmark TTFT and TPOT for an OpenAI-compatible chat completions server.

The script generates overlapping streaming requests at one or more request
rates. This lets a serving engine exercise continuous batching instead of only
seeing batch size 1 traffic.

Example
-------
::

    python examples/misc/benchmark_openai_ttft_tpot.py \\
        --base-url http://127.0.0.1:30000/v1 \\
        --api-key None \\
        --model Qwen/Qwen3-1.7B \\
        --request-rates 1,2,4,8,16 \\
        --duration-s 60 \\
        --max-tokens 128 \\
        --prompt-file examples/ruler/validation_4k.jsonl \\
        --prompt-field input \\
        --output-dir benchmark_ttft_tpot

The script also writes ``tpot_vs_request_rate.pdf`` after all request rates
finish. To plot an existing summary without rerunning the benchmark:
::

    python examples/misc/benchmark_openai_ttft_tpot.py \\
        --plot-summary benchmark_ttft_tpot/summary_all_rates.json

For more accurate output-token counts, install transformers and pass
``--tokenizer MODEL_OR_PATH``. Without a tokenizer, TPOT is computed from
streaming chunks, which can differ from model tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


DEFAULT_PROMPTS = [
    "Write a concise explanation of how attention works in a transformer.",
    "Summarize the tradeoffs between dense and sparse attention for long context.",
    "Give three practical tips for improving LLM serving throughput.",
    "Explain KV cache memory usage in a language model serving system.",
]


@dataclass
class RequestMetrics:
    request_id: int
    request_rate: float
    scheduled_at_s: float
    started_at_s: float
    first_token_at_s: Optional[float]
    last_token_at_s: Optional[float]
    finished_at_s: float
    ttft_s: Optional[float]
    tpot_s: Optional[float]
    latency_s: float
    prompt_chars: int
    prompt_tokens: Optional[int]
    output_chars: int
    output_chunks: int
    output_tokens: Optional[int]
    error: Optional[str]


def parse_request_rates(value: str) -> List[float]:
    rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rates:
        raise argparse.ArgumentTypeError("at least one request rate is required")
    if any(rate <= 0 for rate in rates):
        raise argparse.ArgumentTypeError("request rates must be positive")
    return rates


def rate_label(request_rate: float) -> str:
    return str(request_rate).replace(".", "p")


def rate_output_paths(output_dir: Path, request_rate: float) -> tuple[Path, Path]:
    label = rate_label(request_rate)
    raw_path = output_dir / f"requests_rate_{label}.jsonl"
    summary_path = output_dir / f"summary_rate_{label}.json"
    return raw_path, summary_path


def load_existing_rate_summary(
    output_dir: Path,
    request_rate: float,
) -> Optional[Dict[str, Any]]:
    raw_path, summary_path = rate_output_paths(output_dir, request_rate)
    if not raw_path.exists() or not summary_path.exists():
        return None

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except json.JSONDecodeError:
        return None

    if summary.get("request_rate") != request_rate:
        return None
    return summary


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": statistics.fmean(clean),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def load_prompts(path: Optional[Path], field: str, limit: Optional[int]) -> List[str]:
    if path is None:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                value = record
                for part in field.split("."):
                    value = value[part]
                if isinstance(value, list):
                    value = "\n".join(str(item) for item in value)
                prompts.append(str(value))
                if limit is not None and len(prompts) >= limit:
                    break
    if not prompts:
        raise ValueError("no prompts were loaded")
    return prompts


def load_token_counter(tokenizer_name: Optional[str]) -> Optional[Callable[[str], int]]:
    if tokenizer_name is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "--tokenizer requires transformers. Install it or omit --tokenizer."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


def maybe_add_extra_body(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.extra_body is None:
        return None
    with Path(args.extra_body).open("r", encoding="utf-8") as f:
        return json.load(f)


async def run_one_request(
    client: Any,
    args: argparse.Namespace,
    request_id: int,
    request_rate: float,
    scheduled_at_s: float,
    prompt: str,
    token_counter: Optional[Callable[[str], int]],
    extra_body: Optional[Dict[str, Any]],
) -> RequestMetrics:
    started = time.perf_counter()
    first_token_at: Optional[float] = None
    last_token_at: Optional[float] = None
    output_parts: List[str] = []
    output_chunks = 0
    error: Optional[str] = None
    prompt_tokens = token_counter(prompt) if token_counter is not None else None

    try:
        kwargs: Dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
        }
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if not text:
                continue
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            output_parts.append(text)
            output_chunks += 1
    except Exception as exc:  # Keep load generation alive across request failures.
        error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    output_text = "".join(output_parts)
    output_tokens = token_counter(output_text) if token_counter is not None else None
    denominator = output_tokens if output_tokens is not None else output_chunks
    ttft = first_token_at - started if first_token_at is not None else None
    tpot = None
    if first_token_at is not None and last_token_at is not None and denominator > 1:
        tpot = (last_token_at - first_token_at) / (denominator - 1)

    return RequestMetrics(
        request_id=request_id,
        request_rate=request_rate,
        scheduled_at_s=scheduled_at_s,
        started_at_s=started,
        first_token_at_s=first_token_at,
        last_token_at_s=last_token_at,
        finished_at_s=finished,
        ttft_s=ttft,
        tpot_s=tpot,
        latency_s=finished - started,
        prompt_chars=len(prompt),
        prompt_tokens=prompt_tokens,
        output_chars=len(output_text),
        output_chunks=output_chunks,
        output_tokens=output_tokens,
        error=error,
    )


async def run_rate(
    client: Any,
    args: argparse.Namespace,
    request_rate: float,
    prompts: List[str],
    token_counter: Optional[Callable[[str], int]],
    extra_body: Optional[Dict[str, Any]],
) -> List[RequestMetrics]:
    start = time.perf_counter()
    next_request_id = 0
    tasks: set[asyncio.Task[RequestMetrics]] = set()
    completed: List[RequestMetrics] = []

    async def wait_for_some() -> None:
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        tasks.clear()
        tasks.update(pending)
        completed.extend(task.result() for task in done)

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= args.duration_s:
            break
        if args.max_requests is not None and next_request_id >= args.max_requests:
            break

        while len(tasks) >= args.max_concurrency:
            await wait_for_some()

        prompt = prompts[next_request_id % len(prompts)]
        tasks.add(
            asyncio.create_task(
                run_one_request(
                    client=client,
                    args=args,
                    request_id=next_request_id,
                    request_rate=request_rate,
                    scheduled_at_s=time.perf_counter(),
                    prompt=prompt,
                    token_counter=token_counter,
                    extra_body=extra_body,
                )
            )
        )
        next_request_id += 1

        sleep_s = (
            random.expovariate(request_rate)
            if args.arrival == "poisson"
            else 1.0 / request_rate
        )
        await asyncio.sleep(sleep_s)

    if tasks:
        done = await asyncio.gather(*tasks)
        completed.extend(done)
    completed.sort(key=lambda item: item.request_id)
    return completed


def summarize_rate(
    request_rate: float,
    results: List[RequestMetrics],
    elapsed_s: float,
) -> Dict[str, Any]:
    successful = [item for item in results if item.error is None]
    total_output_tokens = sum(item.output_tokens or 0 for item in successful)
    total_output_chunks = sum(item.output_chunks for item in successful)
    output_count_name = "output_tokens" if total_output_tokens else "output_chunks"
    output_count = total_output_tokens if total_output_tokens else total_output_chunks

    return {
        "request_rate": request_rate,
        "elapsed_s": elapsed_s,
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "achieved_request_rate": len(results) / elapsed_s if elapsed_s > 0 else None,
        "output_count_name": output_count_name,
        "total_output_count": output_count,
        "output_count_per_s": output_count / elapsed_s if elapsed_s > 0 else None,
        "ttft_s": summarize(item.ttft_s for item in successful),
        "tpot_s": summarize(item.tpot_s for item in successful),
        "latency_s": summarize(item.latency_s for item in successful),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def plot_tpot_vs_request_rate(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path("/tmp/matplotlib")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to write the TPOT PDF. Install matplotlib "
            "or rerun with --no-plot."
        ) from exc

    points = sorted(summaries, key=lambda item: item["request_rate"])
    request_rates = [item["request_rate"] for item in points]
    percentiles = ["p50", "p90", "p95", "p99"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for percentile_name in percentiles:
        values_ms = [
            item["tpot_s"].get(percentile_name) * 1000.0
            if item["tpot_s"].get(percentile_name) is not None
            else None
            for item in points
        ]
        ax.plot(
            request_rates,
            values_ms,
            marker="o",
            linewidth=2,
            label=percentile_name,
        )

    ax.set_title("TPOT vs Request Rate")
    ax.set_xlabel("Request rate (requests/s)")
    ax.set_ylabel("TPOT (ms/token)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Percentile")
    if request_rates and min(request_rates) > 0:
        ax.set_xticks(request_rates)
        ax.set_xticklabels([f"{rate:g}" for rate in request_rates])
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_summary_file(summary_path: Path, output_path: Optional[Path]) -> Path:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summaries = payload["summaries"]
    if output_path is None:
        output_path = summary_path.parent / "tpot_vs_request_rate.pdf"
    plot_tpot_vs_request_rate(summaries, output_path)
    return output_path


async def async_main(args: argparse.Namespace) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai package to run this benchmark.") from exc

    random.seed(args.seed)
    prompts = load_prompts(args.prompt_file, args.prompt_field, args.prompt_limit)
    token_counter = load_token_counter(args.tokenizer)
    extra_body = maybe_add_extra_body(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    all_summaries: List[Dict[str, Any]] = []

    for request_rate in args.request_rates:
        existing_summary = None
        if not args.rerun_existing:
            existing_summary = load_existing_rate_summary(args.output_dir, request_rate)
        if existing_summary is not None:
            all_summaries.append(existing_summary)
            print(
                f"[rate={request_rate:g}/s] skipped existing outputs in "
                f"{args.output_dir}"
            )
            continue

        print(f"[rate={request_rate:g}/s] starting benchmark")
        started = time.perf_counter()
        results = await run_rate(
            client=client,
            args=args,
            request_rate=request_rate,
            prompts=prompts,
            token_counter=token_counter,
            extra_body=extra_body,
        )
        elapsed_s = time.perf_counter() - started
        summary = summarize_rate(request_rate, results, elapsed_s)
        all_summaries.append(summary)

        raw_path, summary_path = rate_output_paths(args.output_dir, request_rate)
        write_jsonl(raw_path, (asdict(item) for item in results))
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        ttft = summary["ttft_s"]
        tpot = summary["tpot_s"]
        print(
            f"[rate={request_rate:g}/s] done: "
            f"ok={summary['successful_requests']}/{summary['requests']} "
            f"TTFT p50={ttft['p50']} p95={ttft['p95']} "
            f"TPOT p50={tpot['p50']} p95={tpot['p95']}"
        )

        if args.cooldown_s > 0:
            await asyncio.sleep(args.cooldown_s)

    combined_path = args.output_dir / "summary_all_rates.json"
    combined_path.write_text(
        json.dumps({"summaries": all_summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote summaries to {combined_path}")
    if not args.no_plot:
        plot_path = args.plot_output or (args.output_dir / "tpot_vs_request_rate.pdf")
        plot_tpot_vs_request_rate(all_summaries, plot_path)
        print(f"wrote TPOT plot to {plot_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure TTFT and TPOT under different request rates.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "None"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--request-rates", type=parse_request_rates, default=None)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=1024)
    parser.add_argument("--arrival", choices=["poisson", "constant"], default="poisson")
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--prompt-field", default="input")
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--extra-body",
        default=None,
        help="Path to JSON object passed as OpenAI extra_body, for backend-specific options.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_ttft_tpot"))
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun request rates even when per-rate outputs already exist.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Path for the TPOT-vs-request-rate PDF.",
    )
    parser.add_argument(
        "--plot-summary",
        type=Path,
        default=None,
        help="Plot an existing summary_all_rates.json and exit.",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.plot_summary is not None:
        plot_path = plot_summary_file(args.plot_summary, args.plot_output)
        print(f"wrote TPOT plot to {plot_path}")
        return
    if args.model is None:
        raise SystemExit("--model is required unless --plot-summary is used")
    if args.request_rates is None:
        raise SystemExit("--request-rates is required unless --plot-summary is used")
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be positive")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
