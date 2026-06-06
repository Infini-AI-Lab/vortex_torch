"""Autonomous submission writer for the vortex_torch sparse-attention competition.

Fireworks-API variant of ``auto_agent.py``.  Drives a Fireworks model through
an agentic loop using Fireworks' OpenAI-compatible Chat Completions API
(SDK: ``openai``) that:

  1. Reads the AGENTS.md brief + the six tutorials in ``AI/tutorials/``
     (all bundled into a system message).
  2. Writes a new submission pair
     ``submissions/<name>.py`` + ``submissions/<name>.json``.
  3. Submits the AIME24 benchmark to **Slurm** via
     :file:`experiments/run_submission.slurm`, polls ``squeue`` /
     ``sacct`` until the job terminates, then parses the summary
     JSON under ``summary_submissions/``.
  4. Iterates on failures (missing op, bad dtype, low mean@16, etc.)
     until either the target metric is reached or ``--max-iterations``
     is exhausted.

The benchmark itself MUST run on Slurm — the login environment
where this script is launched does not necessarily have a GPU. The
agent uses the ``bash`` tool to ``sbatch``, poll, and read the
Slurm stdout/stderr logs.

The optimisation objective is **maximise throughput (tokens/sec)
subject to `mean@16 >= --min-mean-at-16`** — the same contract
described in `AI/AGENTS.md`.

Usage
-----
::

    export FIREWORKS_API_KEY=...
    python examples/misc/auto_agent_fwks.py \\
        --submission-name my_agent_v1 \\
        --max-iterations 6 \\
        --min-mean-at-16 0.62 \\
        --baseline-throughput 5500

Only ``FIREWORKS_API_KEY`` must be set externally.  Everything else has
a safe default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD = REPO_ROOT / "AI" / "AGENTS.md"
TUTORIAL_DIR = REPO_ROOT / "AI" / "tutorials"
TUTORIAL_FILES = [
    "overview.md",
    "program_create_cache.md",
    "program_forward_cache.md",
    "program_forward_indexer.md",
    "cache_op.md",
    "indexer_op.md",
]
ALGORITHMS_PY = REPO_ROOT / "vortex_torch" / "flow" / "algorithms.py"
EXAMPLE_JSON = REPO_ROOT / "submissions" / "example_block_sparse_attention.json"
EXAMPLE_PY = REPO_ROOT / "submissions" / "example_block_sparse_attention.py"
SUMMARY_DIR = REPO_ROOT / "summary_submissions"


# ---------------------------------------------------------------------------
# Fireworks config
# ---------------------------------------------------------------------------

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
MODEL = "accounts/fireworks/models/kimi-k2p6"
MAX_COMPLETION_TOKENS = 32768
TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Tools exposed to the model (OpenAI-compatible function-calling schema)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the vortex_torch repo. "
                "Paths may be absolute or relative to the repo root. "
                "Returns up to 400 KB of text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write (overwriting) a UTF-8 text file. Intended for "
                "submissions/<name>.py and submissions/<name>.json. "
                "Parent directories are created automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List filenames in a directory (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command from the repo root. "
                "Used to run check_engine_config / the AIME24 benchmark / "
                "quick python -c probes. Combined stdout+stderr are returned, "
                "truncated to 20 KB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command":     {"type": "string"},
                    "timeout_sec": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default 1800).",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _tool_read_file(path: str) -> str:
    p = _resolve(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    data = p.read_bytes()
    if len(data) > 400_000:
        data = data[:400_000] + b"\n... [truncated] ..."
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"ERROR: file is not utf-8 text ({p})"


def _tool_write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {p}"


def _tool_list_dir(path: str) -> str:
    p = _resolve(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    return "\n".join(sorted(x.name for x in p.iterdir()))


def _tool_bash(command: str, timeout_sec: int = 1800) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        return f"TIMEOUT after {timeout_sec}s\n--- partial stdout ---\n{e.stdout or ''}\n--- partial stderr ---\n{e.stderr or ''}"
    out = (proc.stdout or "") + (proc.stderr or "")
    if len(out) > 20_000:
        out = out[:10_000] + "\n... [truncated middle] ...\n" + out[-10_000:]
    return f"exit={proc.returncode}\n{out}"


def _dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    try:
        if name == "read_file":
            return _tool_read_file(args["path"])
        if name == "write_file":
            return _tool_write_file(args["path"], args["content"])
        if name == "list_dir":
            return _tool_list_dir(args["path"])
        if name == "bash":
            return _tool_bash(args["command"], args.get("timeout_sec", 1800))
        return f"ERROR: unknown tool {name!r}"
    except KeyError as e:
        return f"ERROR: missing required arg {e} for tool {name}"
    except Exception as e:  # pragma: no cover — surface to model
        return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

def _load_bundle() -> str:
    """Concatenate AGENTS.md + tutorials + reference example into one string."""
    parts: List[str] = []

    def _section(title: str, path: Path) -> None:
        parts.append(f"\n\n===== {title} ({path.relative_to(REPO_ROOT)}) =====\n")
        parts.append(path.read_text(encoding="utf-8"))

    _section("AGENTS.md",            AGENTS_MD)
    for name in TUTORIAL_FILES:
        _section(f"tutorial: {name}", TUTORIAL_DIR / name)
    _section("reference flows",      ALGORITHMS_PY)
    _section("example submission .py",   EXAMPLE_PY)
    _section("example submission .json", EXAMPLE_JSON)
    return "".join(parts)


def _build_system_prompt() -> str:
    preamble = (
        "You are an expert automation agent. Your job is to write a new "
        "sparse-attention `vFlow` submission for the vortex_torch "
        "framework, run it against the AIME24 benchmark **on Slurm**, "
        "and iterate based on the numerical results.\n\n"
        "Rules:\n"
        "  - Obey the contract described in AGENTS.md exactly.\n"
        "  - Never touch files outside `submissions/` (plus reading the "
        "    `summary_submissions/` JSON and the Slurm logs under "
        "    `logs/submission/` after each run).\n"
        "  - The login / host environment you are running in does NOT "
        "    necessarily have a GPU. Do NOT invoke "
        "    `python examples/math/run_submission_aime24.py` directly — it "
        "    will fail or (worse) hang. Every real benchmark run must "
        "    be submitted to Slurm via "
        "    `sbatch experiments/run_submission.slurm "
        "submissions/<your_name>.json`, then polled with `squeue -j "
        "<jobid>` / `sacct -j <jobid> --format=JobID,State,ExitCode -X "
        "-n -P` until the job terminates.\n"
        "  - You MAY run the CHEAP pre-flight locally (it's fast and "
        "    catches most shape / dispatch bugs without needing a GPU):\n"
        "      `python -c \"from vortex_torch.engine.sgl import "
        "check_engine_config; check_engine_config('submissions/"
        "<your_name>.json')\"`\n"
        "    If the pre-flight fails, fix the flow before spending a "
        "    Slurm job.\n"
        "  - After the Slurm job ends, read\n"
        "      * the freshest JSON in `summary_submissions/` (contains "
        "        `mean@16`, `pass@16`, `throughput`, etc.), and\n"
        "      * `logs/submission/vortex_submission_<jobid>.out` + "
        "        `.err` if you need to debug.\n"
        "  - When you are satisfied (or when you have run out of "
        "    reasonable ideas), produce a FINAL summary of what you "
        "    built and the numbers you achieved.\n"
        "  - Use tools efficiently: cache what you've already read."
    )
    bundle = _load_bundle()
    return preamble + "\n\n" + bundle


# ---------------------------------------------------------------------------
# Initial user prompt
# ---------------------------------------------------------------------------

def _build_initial_user_prompt(
    submission_name: str,
    min_mean: float,
    baseline_throughput: Optional[float],
    max_iterations: int,
) -> str:
    baseline_line = (
        f"   Reference: the `example_block_sparse_attention` baseline "
        f"   delivers roughly `throughput ≈ {baseline_throughput:.0f} tok/s`. "
        f"   You should aim to comfortably exceed it.\n"
        if baseline_throughput is not None else ""
    )
    return (
        f"## Objective\n"
        f"**Maximise `throughput` (tokens/sec) on the AIME24 benchmark "
        f"while keeping `mean@16>= {min_mean}`.** `mean@16` is a "
        f"quality floor, NOT something to maximise. Once it clears the "
        f"floor, every further change should buy throughput — tighten "
        f"sparsity (lower `vortex_topk_val` / `vortex_topk_ratio`), "
        f"prune cache-side ops, drop unused cache fields, narrow "
        f"`vortex_layers_skip`, try fp8 `kv_cache_dtype`. Always prefer "
        f"the faster variant when two variants both clear the floor.\n\n"
        f"## Protocol\n"
        f"1. Write two files:\n"
        f"   - `submissions/{submission_name}.py`  — the "
        f"`@register(\"{submission_name}_cls\")` vFlow subclass.\n"
        f"   - `submissions/{submission_name}.json` — the engine config "
        f"(copy and tweak the example; keep `vortex_dtype=bfloat16` "
        f"unless you have a strong reason; ensure `vortex_module_name` "
        f"matches your `@register` name and `vortex_module_path` points "
        f"at your .py).\n"
        f"2. Run the CHEAP pre-flight locally (no GPU needed, fails "
        f"   fast on shape / dispatch bugs):\n"
        f"   `python -c \"from vortex_torch.engine.sgl import "
        f"check_engine_config; check_engine_config('submissions/"
        f"{submission_name}.json')\"`\n"
        f"   Fix any error here before spending a Slurm job.\n"
        f"3. Submit the full AIME24 benchmark to **Slurm**. DO NOT run "
        f"   `python examples/math/run_submission_aime24.py` directly — the "
        f"   host environment may not have a GPU:\n"
        f"   a. `sbatch experiments/run_submission.slurm "
        f"submissions/{submission_name}.json`  — captures the returned "
        f"      `Submitted batch job <JOBID>`.\n"
        f"   b. Poll: `sacct -j <JOBID> --format=JobID,State,ExitCode "
        f"-X -n -P` (or `squeue -j <JOBID>`) every ~60 seconds. Use a "
        f"      single `bash` invocation with a `timeout_sec` large "
        f"      enough for the run (jobs typically take 10-40 min).\n"
        f"   c. When state is `COMPLETED` (or any terminal state — "
        f"      `FAILED` / `TIMEOUT` / `CANCELLED` / `OUT_OF_MEMORY`), "
        f"      proceed; on failure read "
        f"      `logs/submission/vortex_submission_<JOBID>.err` and "
        f"      `.out` to diagnose.\n"
        f"4. Read the freshest JSON in `summary_submissions/` (filename "
        f"   includes `{submission_name}` and a timestamp). Decide "
        f"   whether to iterate:\n"
        f"   - If `mean@16 < {min_mean}` → quality floor failed; "
        f"     trade throughput for accuracy (widen layer-skip, raise "
        f"     `vortex_topk_val`, back off fp8 kv, simplify scoring).\n"
        f"   - If `mean@16 >= {min_mean}` → floor cleared; now push "
        f"     throughput up while keeping `mean@16` on the right side "
        f"     of the floor.\n"
        f"{baseline_line}"
        f"5. If the pre-flight check (step 2) fails, or the Slurm job "
        f"   crashed (step 3c), edit the submission and go back to "
        f"   step 2 — do NOT re-submit to Slurm until pre-flight "
        f"   passes.\n\n"
        f"## Budget\n"
        f"You have at most **{max_iterations} outer benchmark runs**. "
        f"Spend them wisely: do not burn a run on a change whose "
        f"expected throughput / accuracy impact you cannot articulate.\n\n"
        f"Start by briefly describing (1 short paragraph) the "
        f"sparse-attention idea you plan to implement and why you "
        f"expect it to be throughput-competitive, then proceed."
    )


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _load_openai_sdk() -> Tuple[Any, Any]:
    try:
        import openai
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "The 'openai' package is required.  "
            "Install or upgrade with: pip install -U openai"
        ) from e
    return openai, OpenAI


def _pretty_assistant_message(msg: Any) -> str:
    """Short human-readable summary of the assistant's response."""
    out: List[str] = []
    content = getattr(msg, "content", None)
    if content:
        out.append(f"[text] {content.strip()[:800]}")
    for tc in getattr(msg, "tool_calls", None) or []:
        fn = tc.function
        args_preview = (fn.arguments or "")[:500]
        out.append(f"[tool_use name={fn.name} id={tc.id} args={args_preview}]")
    return "\n".join(out) if out else "[empty]"


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    """Serialize the assistant message back to dict form for the next turn."""
    d: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return d


def _call_model(
    openai_module: Any,
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
) -> Any:
    """Call the Fireworks OpenAI-compatible Chat Completions API."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except (AttributeError, TypeError) as e:
            raise SystemExit(
                "Your installed 'openai' package does not support the "
                "Chat Completions options used by this script. Upgrade with: "
                "pip install -U openai"
            ) from e
        except openai_module.RateLimitError as e:
            last_err = e
            wait = 2 ** attempt * 5
            print(f"[rate-limit] sleeping {wait}s …  ({e})", file=sys.stderr)
            time.sleep(wait)
        except openai_module.APIError as e:
            last_err = e
            wait = 2 ** attempt * 5
            print(f"[api-error attempt {attempt}] {e} — retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"Fireworks API kept failing after 4 retries: {last_err}")


def run_agent(
    submission_name: str,
    min_mean: float,
    baseline_throughput: Optional[float],
    max_iterations: int,
    max_tool_calls: int,
    log_path: Path,
    model: str = MODEL,
) -> Dict[str, Any]:
    openai_module, OpenAI = _load_openai_sdk()
    client = OpenAI(
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        base_url=FIREWORKS_BASE_URL,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {
            "role": "user",
            "content": _build_initial_user_prompt(
                submission_name,
                min_mean,
                baseline_throughput,
                max_iterations,
            ),
        },
    ]

    turns = 0
    finish_reason: Optional[str] = None
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(obj: Dict[str, Any]) -> None:
        log_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log_fp.flush()

    _log({
        "event": "start",
        "model": model,
        "submission": submission_name,
        "objective": "maximise throughput subject to mean@16 >= floor",
        "min_mean@16": min_mean,
        "baseline_throughput": baseline_throughput,
        "max_iterations": max_iterations,
    })

    while turns < max_tool_calls:
        turns += 1

        response = _call_model(openai_module, client, model, messages)
        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason
        usage = response.usage.model_dump() if response.usage else {}

        print(f"\n========== turn {turns} "
              f"(finish_reason={finish_reason}, "
              f"usage={usage}) ==========")
        print(_pretty_assistant_message(msg))

        _log({
            "event": "assistant_turn",
            "turn": turns,
            "finish_reason": finish_reason,
            "usage": usage,
            "content_summary": _pretty_assistant_message(msg),
        })

        messages.append(_assistant_message_dict(msg))
        tool_calls = getattr(msg, "tool_calls", None) or []

        # --- done? ---
        if finish_reason == "stop" and not tool_calls:
            print("\n[agent] model ended its turn — stopping.")
            break
        if finish_reason not in ("tool_calls", "stop") and not tool_calls:
            print(f"\n[agent] non-tool finish_reason={finish_reason!r} — stopping.")
            break
        if not tool_calls:
            print("[agent] no tool calls emitted — stopping.")
            break

        # --- execute any tool calls ---
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                fn_args = {}
                result = f"ERROR: could not parse tool arguments: {e}"
            else:
                result = _dispatch_tool(fn_name, fn_args)

            print(f"\n[tool {fn_name}] -> {result[:500]}"
                  f"{' …' if len(result) > 500 else ''}")
            _log({
                "event": "tool_result",
                "turn": turns,
                "tool": fn_name,
                "tool_call_id": tc.id,
                "input": fn_args,
                "output_preview": result[:2000],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    log_fp.close()

    return {
        "turns": turns,
        "finish_reason": finish_reason,
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fireworks-driven sparse-attention submission writer."
    )
    p.add_argument(
        "--submission-name",
        default=f"agent_{uuid.uuid4().hex[:8]}",
        help="Stem for submissions/<name>.py and .json.",
    )
    p.add_argument(
        "--model",
        default=MODEL,
        help=f"Fireworks model id (default: {MODEL}).",
    )
    p.add_argument(
        "--min-mean-at-16",
        type=float,
        default=0.62,
        help=(
            "Minimum acceptable mean@16 on AIME24 (quality floor). "
            "The agent maximises throughput subject to this floor."
        ),
    )
    p.add_argument(
        "--baseline-throughput",
        type=float,
        default=5500,
        help=(
            "Optional tokens/sec reference (e.g. "
            "example_block_sparse_attention's throughput) to give the "
            "agent a target to beat."
        ),
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="How many benchmark runs the agent may attempt.",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        default=80,
        help="Hard cap on total LLM turns (tool calls + final answer).",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "auto_agent",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise SystemExit("FIREWORKS_API_KEY is not set.")

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = args.logs_dir / f"{args.submission_name}_{ts}.jsonl"

    print(f"[auto_agent_fwks] objective: maximise throughput s.t. "
          f"mean@16 >= {args.min_mean_at_16}")
    print(f"[auto_agent_fwks] model              = {args.model}")
    print(f"[auto_agent_fwks] submission_name    = {args.submission_name}")
    print(f"[auto_agent_fwks] min mean@16 (floor)= {args.min_mean_at_16}")
    if args.baseline_throughput is not None:
        print(f"[auto_agent_fwks] baseline_throughput= "
              f"{args.baseline_throughput:.0f} tok/s")
    print(f"[auto_agent_fwks] max_iterations     = {args.max_iterations}")
    print(f"[auto_agent_fwks] log                -> {log_path}")

    result = run_agent(
        submission_name=args.submission_name,
        min_mean=args.min_mean_at_16,
        baseline_throughput=args.baseline_throughput,
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        log_path=log_path,
        model=args.model,
    )

    print("\n[auto_agent_fwks] done.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
