"""Autonomous submission writer for the vortex_torch sparse-attention competition.

Drives Claude through an agentic loop (SDK: ``anthropic``) that:

  1. Reads the AGENTS.md brief + the six tutorials in ``AI/tutorials/``
     (all bundled into a cached system prompt).
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

    export ANTHROPIC_API_KEY=...
    python examples/misc/auto_agent.py \\
        --submission-name my_agent_v1 \\
        --max-iterations 6 \\
        --min-mean-at-16 0.62 \\
        --baseline-throughput 5500

Only ``ANTHROPIC_API_KEY`` must be set externally.  Everything else has
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
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "The 'anthropic' package is required.  "
        "Install with: pip install 'anthropic>=0.79'"
    ) from e


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
# Claude config
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7"
MAX_TOKENS = 16384


# ---------------------------------------------------------------------------
# Tools exposed to Claude
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the vortex_torch repo. "
            "Paths may be absolute or relative to the repo root. "
            "Returns up to 400 KB of text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write (overwriting) a UTF-8 text file. Intended for "
            "submissions/<name>.py and submissions/<name>.json. "
            "Parent directories are created automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List filenames in a directory (non-recursive).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command from the repo root. "
            "Used to run check_engine_config / the AIME24 benchmark / "
            "quick python -c probes. Combined stdout+stderr are returned, "
            "truncated to 20 KB."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command":     {"type": "string"},
                "timeout_sec": {
                    "type": "integer",
                    "description": "Optional timeout (default 1800).",
                },
            },
            "required": ["command"],
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
# System prompt assembly (with prompt caching)
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


def _build_system_blocks() -> List[Dict[str, Any]]:
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
        "    be submitted to Slurm. The cluster has 8 GPUs per node, "
        "    so pick the slurm file that fits the shape of the work:\n"
        "      * **1 variant**  → "
        "        `sbatch experiments/run_submission.slurm "
        "submissions/<name>.json`\n"
        "      * **2-8 variants in parallel** (STRONGLY PREFERRED when "
        "        you have multiple candidate designs) → "
        "        `sbatch experiments/run_submission_batch.slurm "
        "submissions/v1.json submissions/v2.json ...`  — a single "
        "        node-wide job that backgrounds each child with "
        "        `CUDA_VISIBLE_DEVICES=0..7`. Per-child logs live at "
        "        `logs/submission/batch_<JOBID>/gpu<i>_<stem>.{out,err}`.\n"
        "    Poll with `sacct -j <jobid> --format=JobID,State,ExitCode "
        "-X -n -P` until the job terminates.\n"
        "  - When iterating, actively think in terms of **variant sets**: "
        "    whenever you have a few candidate knob settings (different "
        "    `vortex_topk_val`, different `kv_cache_dtype`, different "
        "    `mem_fraction_static` ∈ [0.5, 0.95] (0.8-0.9 sweet spot, "
        "    higher = more throughput but OOM risk), different "
        "    scoring functions), generate them as separate submission "
        "    pairs (`submissions/<name>_v1.json`, `_v2.json`, ...) and "
        "    submit them together via `run_submission_batch.slurm`. You "
        "    get 8 comparable data points in the wall-clock time of one.\n"
        "  - You MAY run the CHEAP pre-flight locally (it's fast and "
        "    catches most shape / dispatch bugs without needing a GPU):\n"
        "      `python -c \"from vortex_torch.engine.sgl import "
        "check_engine_config; check_engine_config('submissions/"
        "<your_name>.json')\"`\n"
        "    If the pre-flight fails, fix the flow before spending a "
        "    Slurm job.\n"
        "  - Each run is written into a per-submission subfolder so "
        "    iterations never collide:\n"
        "        summary_submissions/<name>/\n"
        "            <timestamp>__<content_hash>.json   # full summary + embedded .py/.json\n"
        "            latest.json                        # symlink → newest run\n"
        "            INDEX.jsonl                        # one-row-per-run index\n"
        "    `content_hash` is sha256(config.json || module.py) truncated "
        "    to 12 chars, so identical code → identical hash → you can "
        "    see re-runs at a glance. After the Slurm job ends, read:\n"
        "      * `summary_submissions/<name>/latest.json` for the newest "
        "        run (mean@16, pass@16, throughput, content_hash, "
        "        slurm_job_id, etc.), or grep "
        "        `summary_submissions/<name>/INDEX.jsonl` to compare runs.\n"
        "      * `logs/submission/vortex_submission_<jobid>.{out,err}` "
        "        if you need to debug.\n"
        "  - When you are satisfied (or when you have run out of "
        "    reasonable ideas), produce a FINAL summary of what you "
        "    built and the numbers you achieved.\n"
        "  - Use tools efficiently: cache what you've already read."
    )
    bundle = _load_bundle()
    return [
        {"type": "text", "text": preamble},
        {
            "type": "text",
            "text": bundle,
            "cache_control": {"type": "ephemeral"},
        },
    ]


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
        f"`vortex_layers_skip`, try fp8 `kv_cache_dtype`, raise "
        f"`mem_fraction_static` toward 0.9 (default 0.8). Always prefer "
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
        f"   host environment may not have a GPU. Nodes have 8 GPUs, so:\n"
        f"   a. If you are comparing **2-8 variants in this round** — "
        f"      write their submission pairs as "
        f"      `submissions/{submission_name}_v1.{{py,json}}`, "
        f"      `{submission_name}_v2.*`, ... and submit them in ONE "
        f"      batched job:\n"
        f"      `sbatch experiments/run_submission_batch.slurm "
        f"submissions/{submission_name}_v1.json "
        f"submissions/{submission_name}_v2.json ...`\n"
        f"      This is strongly preferred over 8 separate sbatches — "
        f"      one node, 8 sibling GPUs, one poll.\n"
        f"      If this round has only one variant:\n"
        f"      `sbatch experiments/run_submission.slurm "
        f"submissions/{submission_name}.json`\n"
        f"   b. Capture the returned `Submitted batch job <JOBID>`.\n"
        f"   c. Poll: `sacct -j <JOBID> --format=JobID,State,ExitCode "
        f"-X -n -P` (or `squeue -j <JOBID>`) every ~60 seconds. Use a "
        f"      single `bash` invocation with a `timeout_sec` large "
        f"      enough for the run (jobs typically take 10-40 min).\n"
        f"   d. When state is `COMPLETED` (or any terminal state — "
        f"      `FAILED` / `TIMEOUT` / `CANCELLED` / `OUT_OF_MEMORY`), "
        f"      proceed; on failure read "
        f"      `logs/submission/vortex_submission_<JOBID>.err`  or "
        f"      (for a batched run) "
        f"      `logs/submission/batch_<JOBID>/gpu<i>_<stem>.err` "
        f"      to diagnose.\n"
        f"4. Read `summary_submissions/{submission_name}/latest.json` "
        f"   (a symlink to the newest run; the actual filename is "
        f"   `<timestamp>__<content_hash>.json`). Decide whether to "
        f"   iterate:\n"
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

def _pretty_content_blocks(blocks) -> str:
    """Short human-readable summary of the assistant's response blocks."""
    out: List[str] = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            txt = b.text.strip()
            if txt:
                out.append(f"[text] {txt[:800]}")
        elif t == "thinking":
            out.append(f"[thinking ~{len(b.thinking or '')} chars]")
        elif t == "tool_use":
            args_preview = json.dumps(b.input)[:500]
            out.append(f"[tool_use name={b.name} id={b.id} args={args_preview}]")
        else:
            out.append(f"[{t}]")
    return "\n".join(out)


def _blocks_for_messages(blocks) -> List[Dict[str, Any]]:
    """Serialize the SDK's content blocks back into dict form for the next turn."""
    msg_blocks: List[Dict[str, Any]] = []
    for b in blocks:
        t = b.type
        if t == "text":
            msg_blocks.append({"type": "text", "text": b.text})
        elif t == "thinking":
            msg_blocks.append({
                "type": "thinking",
                "thinking": b.thinking,
                "signature": getattr(b, "signature", ""),
            })
        elif t == "tool_use":
            msg_blocks.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": b.input,
            })
        # other block types (e.g. compaction) get passed through verbatim
        else:
            msg_blocks.append(b.model_dump())
    return msg_blocks


def run_agent(
    submission_name: str,
    min_mean: float,
    baseline_throughput: Optional[float],
    max_iterations: int,
    max_tool_calls: int,
    log_path: Path,
) -> Dict[str, Any]:
    client = anthropic.Anthropic()
    system_blocks = _build_system_blocks()

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": _build_initial_user_prompt(
                submission_name,
                min_mean,
                baseline_throughput,
                max_iterations,
            ),
        }
    ]

    turns = 0
    stop_reason: Optional[str] = None
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(obj: Dict[str, Any]) -> None:
        log_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log_fp.flush()

    _log({
        "event": "start",
        "submission": submission_name,
        "objective": "maximise throughput subject to mean@16 >= floor",
        "min_mean@16": min_mean,
        "baseline_throughput": baseline_throughput,
        "max_iterations": max_iterations,
    })

    while turns < max_tool_calls:
        turns += 1

        # --- call Claude ---
        for attempt in range(4):
            try:
                with client.messages.stream(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_blocks,
                    tools=TOOLS,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "high"},
                    messages=messages,
                ) as stream:
                    response = stream.get_final_message()
                break
            except anthropic.RateLimitError as e:
                wait = 2 ** attempt * 5
                print(f"[rate-limit] sleeping {wait}s …  ({e})", file=sys.stderr)
                time.sleep(wait)
            except anthropic.APIError as e:
                wait = 2 ** attempt * 5
                print(f"[api-error attempt {attempt}] {e} — retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
        else:
            _log({"event": "api_failure_terminal"})
            raise SystemExit("Claude API kept failing after 4 retries.")

        stop_reason = response.stop_reason
        usage = response.usage.model_dump() if response.usage else {}
        print(f"\n========== turn {turns} "
              f"(stop_reason={stop_reason}, "
              f"usage={usage}) ==========")
        print(_pretty_content_blocks(response.content))

        _log({
            "event": "assistant_turn",
            "turn": turns,
            "stop_reason": stop_reason,
            "usage": usage,
            "content_summary": _pretty_content_blocks(response.content),
        })

        # --- append assistant message ---
        messages.append({
            "role": "assistant",
            "content": _blocks_for_messages(response.content),
        })

        # --- done? ---
        if stop_reason == "end_turn":
            print("\n[agent] model ended its turn — stopping.")
            break
        if stop_reason not in ("tool_use",):
            print(f"\n[agent] non-tool stop_reason={stop_reason!r} — stopping.")
            break

        # --- execute any tool_use blocks ---
        tool_results: List[Dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = _dispatch_tool(block.name, block.input or {})
            print(f"\n[tool {block.name}] -> {result[:500]}"
                  f"{' …' if len(result) > 500 else ''}")
            _log({
                "event": "tool_result",
                "turn": turns,
                "tool": block.name,
                "tool_use_id": block.id,
                "input": block.input,
                "output_preview": result[:2000],
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if not tool_results:
            print("[agent] stop_reason=tool_use but no tool blocks — stopping.")
            break

        messages.append({"role": "user", "content": tool_results})

    log_fp.close()

    return {
        "turns": turns,
        "stop_reason": stop_reason,
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Claude-driven sparse-attention submission writer."
    )
    p.add_argument(
        "--submission-name",
        default=f"agent_{uuid.uuid4().hex[:8]}",
        help="Stem for submissions/<name>.py and .json.",
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = args.logs_dir / f"{args.submission_name}_{ts}.jsonl"

    print(f"[auto_agent] objective: maximise throughput s.t. "
          f"mean@16 >= {args.min_mean_at_16}")
    print(f"[auto_agent] submission_name     = {args.submission_name}")
    print(f"[auto_agent] min mean@16 (floor) = {args.min_mean_at_16}")
    if args.baseline_throughput is not None:
        print(f"[auto_agent] baseline_throughput = "
              f"{args.baseline_throughput:.0f} tok/s")
    print(f"[auto_agent] max_iterations      = {args.max_iterations}")
    print(f"[auto_agent] log                 -> {log_path}")

    result = run_agent(
        submission_name=args.submission_name,
        min_mean=args.min_mean_at_16,
        baseline_throughput=args.baseline_throughput,
        max_iterations=args.max_iterations,
        max_tool_calls=args.max_tool_calls,
        log_path=log_path,
    )

    print("\n[auto_agent] done.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
