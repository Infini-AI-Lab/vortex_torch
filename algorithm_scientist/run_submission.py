"""Run a sparse-attention submission against a math benchmark (generalized).

This is the **task-generalized** successor to ``run_submission_aime24.py`` /
``run_submission_amc23.py``. Those two remain as thin back-compat shims; new
workflows (``/iterate``, ``/batch-benchmark``) call this script with
``--task``.

Given a submission's engine JSON it:

  1. Validates the config via :func:`check_engine_config`.
  2. Boots an sglang engine with the submission's ``vortex_*`` settings plus
     the fixed protocol constants below.
  3. Runs the selected task's ``examples/math/<task>.jsonl`` with 16 trials.
  4. Scores with lighteval's ``MultilingualExtractiveMatchMetric`` (identical
     to the per-task runners — every supported task is math, same schema:
     ``{prompt, question, answer}``).
  5. Writes a per-run summary JSON into the task's summary dir, mirroring the
     config's path under ``submissions/`` (per-agent isolation, content-hashed
     filenames) — exactly like the original runners.

Tasks
-----
``aime24``, ``aime25``, ``aime26``, ``amc23`` are built-in. Any other math
benchmark with the same ``{prompt, question, answer}`` schema can be run via
``--data examples/math/<file>.jsonl`` (summary dir defaults to ``summary_submissions``
or ``--summary-dir``). LiveCodeBench (``lcbv5``) is *not* supported here — it
needs code-execution scoring, not extractive math matching.

Usage
-----
::

    python algorithm_scientist/run_submission.py --task aime25 \\
        --config submissions/<tag>/batch_0_id0.json

    python algorithm_scientist/run_submission.py \\
        --data examples/math/my_math.jsonl --summary-dir summary_my_math \\
        --config submissions/<tag>/foo.json
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import vortex_torch  # noqa: F401  — registers the attention backend
from vortex_torch.engine.sgl import check_engine_config, get_engine

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    MultilingualExtractiveMatchMetric,
)
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.models.model_output import ModelResponse


# ---------------------------------------------------------------------------
# Fixed benchmark protocol — do not change
# ---------------------------------------------------------------------------

TRIALS                      = 16
MAX_INPUT_LENGTH            = 4096
GENERATION_MAX_NEW_TOKENS   = 32768
# Tensor-parallel degree (GPUs per run). Resolved per-run from, in priority:
#   --tp CLI arg  >  the submission JSON's "tp_size"  >  DEFAULT_TP_SIZE.
# Big models (e.g. MiniMax-M2.7 229B) need tp_size > 1; the caller must make
# exactly `tp_size` GPUs visible via CUDA_VISIBLE_DEVICES (comma-separated).
DEFAULT_TP_SIZE             = 1

# Built-in tasks: (dataset path, summary dir). All share the AIME/AMC math
# schema and the extractive-match scorer below.
TASKS: Dict[str, Tuple[str, str]] = {
    "aime24": ("examples/math/aime24.jsonl", "summary_submissions"),
    "aime25": ("examples/math/aime25.jsonl", "summary_aime25_submissions"),
    "aime26": ("examples/math/aime26.jsonl", "summary_aime26_submissions"),
    "amc23":  ("examples/math/amc23.jsonl",  "summary_amc23_submissions"),
}


def resolve_task(args: argparse.Namespace) -> Tuple[str, str, str]:
    """Return ``(task_label, data_path, summary_dir)`` from the CLI args."""
    if args.data is not None:
        data_path = str(args.data)
        label = Path(data_path).stem
        summary_dir = args.summary_dir or "summary_submissions"
        return label, data_path, summary_dir
    task = (args.task or "aime24").lower()
    if task == "lcbv5":
        raise SystemExit(
            "lcbv5 (LiveCodeBench) needs code-execution scoring, not the "
            "extractive math metric this runner uses. Use a dedicated "
            "code-eval harness for lcbv5."
        )
    if task not in TASKS:
        raise SystemExit(
            f"unknown task {task!r}. Built-in: {', '.join(sorted(TASKS))}. "
            f"For a custom math jsonl pass --data <path> instead."
        )
    data_path, summary_dir = TASKS[task]
    summary_dir = args.summary_dir or summary_dir
    return task, data_path, summary_dir


# ---------------------------------------------------------------------------
# Engine booting
# ---------------------------------------------------------------------------

def _load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    print(f"[pre-flight] validating {config_path}")
    config = check_engine_config(config_path)
    print(f"[pre-flight] OK — module={config.get('vortex_module_name')}")
    return config


def _resolve_tp_size(config: Dict[str, Any], tp_override: "int | None") -> int:
    """CLI --tp wins; else the JSON's tp_size; else DEFAULT_TP_SIZE."""
    if tp_override is not None:
        tp = int(tp_override)
    else:
        tp = int(config.get("tp_size", DEFAULT_TP_SIZE) or DEFAULT_TP_SIZE)
    if tp < 1:
        raise SystemExit(f"tp_size must be >= 1, got {tp}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        n_visible = len([x for x in visible.split(",") if x.strip() != ""])
        if n_visible and n_visible != tp:
            print(f"[engine] WARNING: tp_size={tp} but CUDA_VISIBLE_DEVICES "
                  f"exposes {n_visible} GPU(s) ({visible!r}); sglang expects "
                  f"exactly tp_size GPUs visible.")
    return tp


def _build_engine_kwargs(config: Dict[str, Any], tp_size: int) -> Dict[str, Any]:
    kwargs = dict(config)
    kwargs["tp_size"]              = tp_size
    kwargs["vortex_max_seq_lens"]  = MAX_INPUT_LENGTH + GENERATION_MAX_NEW_TOKENS
    kwargs["context_length"]       = max(
        kwargs.get("context_length", 0),
        MAX_INPUT_LENGTH + GENERATION_MAX_NEW_TOKENS,
    )
    return kwargs


def _boot_engine(kwargs: Dict[str, Any]):
    print(f"[engine] booting — model={kwargs.get('model_path', '<default>')}, "
          f"module={kwargs.get('vortex_module_name')}, "
          f"kv={kwargs.get('kv_cache_dtype', 'auto')}, "
          f"mem_fraction_static={kwargs.get('mem_fraction_static', '<default>')}")
    return get_engine(**kwargs)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _load_requests(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _make_scorer():
    return MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(ExprExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(),
                                LatexExtractionConfig(boxed_match_priority=0)),
        aggregation_function=max,
    )


def _score_results(requests, outputs, scorer):
    results = []
    for data, item in zip(requests, outputs):
        golds = [data["answer"]]
        target = Doc(query=data["question"], choices=golds, gold_index=0)
        prediction = item["text"]
        try:
            score = scorer.compute(
                model_response=ModelResponse(text=[prediction]),
                doc=target,
            )
        except Exception:
            score = 0.0
        results.append({
            "score": float(score),
            "prediction": [prediction],
            "choices": golds,
            "query": data["question"],
            "e2e_latency": item["meta_info"]["e2e_latency"],
            "num_tokens": item["meta_info"]["completion_tokens"],
        })
    return results


def _summarize(results) -> Dict[str, Any]:
    total_accuracy = 0.0
    total_tokens = 0
    e2e_time = 0.0
    count = 0
    unique_result: Dict[str, float] = {}

    for item in results:
        total_accuracy += item["score"]
        total_tokens += item["num_tokens"]
        e2e_time = max(e2e_time, item["e2e_latency"])
        count += 1
        q = item["query"]
        unique_result[q] = max(item["score"], unique_result.get(q, 0.0))

    return {
        f"mean@{TRIALS}": total_accuracy / count if count else 0.0,
        f"pass@{TRIALS}": (sum(unique_result.values()) / len(unique_result)
                           if unique_result else 0.0),
        "total_example": count,
        "e2e_time": e2e_time,
        "total_tokens": total_tokens,
        "throughput": (total_tokens / e2e_time) if e2e_time > 0 else 0.0,
    }


def run(config_path: Path, data_path: str, tp_size: "int | None" = None) -> Dict[str, Any]:
    config = _load_and_validate_config(config_path)
    tp = _resolve_tp_size(config, tp_size)
    engine_kwargs = _build_engine_kwargs(config, tp)
    print(f"[engine] tp_size={tp}")
    llm = _boot_engine(engine_kwargs)

    requests = _load_requests(Path(data_path)) * TRIALS
    prompts = [req["prompt"] for req in requests]

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
    }
    print(f"[benchmark] {len(prompts)} prompts "
          f"({TRIALS} trials × {len(prompts) // TRIALS} questions)")

    outputs = llm.generate(prompts, sampling_params)
    scorer = _make_scorer()
    results = _score_results(requests, outputs, scorer)
    summary = _summarize(results)

    summary["args"] = {
        "config_path":                str(config_path),
        "vortex_module_name":         config.get("vortex_module_name"),
        "vortex_module_path":         config.get("vortex_module_path"),
        "model_path":                 engine_kwargs.get("model_path"),
        "trials":                     TRIALS,
        "max_input_length":           MAX_INPUT_LENGTH,
        "generation_max_new_tokens":  GENERATION_MAX_NEW_TOKENS,
        "mem_fraction_static":        engine_kwargs.get("mem_fraction_static"),
        "tp_size":                    engine_kwargs.get("tp_size"),
        "data_path":                  data_path,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission and run it on a math benchmark "
                    "(aime24/25/26, amc23, or a custom --data jsonl).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("submissions/example_block_sparse_attention.json"),
        help="Path to the submission JSON (default: the bundled example).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="aime24",
        help="Built-in task: aime24 (default), aime25, aime26, amc23.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Custom math jsonl ({prompt,question,answer}); overrides --task.",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default=None,
        help="Override the summary output dir (default: per-task).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor-parallel degree (GPUs per run). Overrides the JSON's "
             "tp_size. The caller must expose exactly this many GPUs via "
             "CUDA_VISIBLE_DEVICES. Default: JSON tp_size or 1.",
    )
    return parser.parse_args()


def _read_submission_artifacts(
    config_path: Path,
) -> Tuple[str, Optional[str], str]:
    config_text = config_path.read_text(encoding="utf-8")

    module_text: Optional[str] = None
    try:
        cfg = json.loads(config_text)
        module_rel = cfg.get("vortex_module_path")
        if module_rel:
            module_path = Path(module_rel)
            if not module_path.is_absolute():
                module_path = (config_path.parent.parent / module_path).resolve()
            if module_path.is_file():
                module_text = module_path.read_text(encoding="utf-8")
    except (json.JSONDecodeError, OSError):
        pass

    hasher = hashlib.sha256()
    hasher.update(config_text.encode("utf-8"))
    if module_text is not None:
        hasher.update(b"\x00")
        hasher.update(module_text.encode("utf-8"))
    content_hash = hasher.hexdigest()[:12]
    return config_text, module_text, content_hash


def _append_index(index_path: Path, row: Dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary_subpath(config_path: Path) -> Path:
    """Mirror the config's location under ``submissions/`` into the summary tree."""
    parts = config_path.with_suffix("").parts  # drop the .json
    if "submissions" in parts:
        idx = parts.index("submissions")
        tail = parts[idx + 1:]
        if tail:
            return Path(*tail)
    return Path(config_path.stem)


def _write_summary(summary: Dict[str, Any], config_path: Path, summary_dir: str) -> str:
    rel = _summary_subpath(config_path)
    tag = str(rel)
    run_dir = Path(summary_dir) / rel
    run_dir.mkdir(parents=True, exist_ok=True)

    config_text, module_text, content_hash = _read_submission_artifacts(config_path)

    finished_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    summary = dict(summary)
    summary["content_hash"] = content_hash
    summary["finished_at"] = finished_at
    summary["cuda_visible_devices"] = cuda_visible
    summary["submission_json"] = config_text
    summary["submission_py"] = module_text

    fname = f"{finished_at}__{content_hash}.json"
    out_path = run_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    _append_index(
        run_dir / "INDEX.jsonl",
        {
            "finished_at":    finished_at,
            "content_hash":   content_hash,
            "cuda_visible":   cuda_visible,
            "submission":     tag,
            "file":           fname,
            f"mean@{TRIALS}":         summary.get(f"mean@{TRIALS}"),
            f"pass@{TRIALS}":         summary.get(f"pass@{TRIALS}"),
            "throughput":     summary.get("throughput"),
            "e2e_time":       summary.get("e2e_time"),
            "total_tokens":   summary.get("total_tokens"),
            "total_example":  summary.get("total_example"),
        },
    )

    latest = run_dir / "latest.json"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(fname)
    except OSError:
        latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    return str(out_path)


if __name__ == "__main__":
    args = parse_args()
    task_label, data_path, summary_dir = resolve_task(args)

    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    if not Path(data_path).is_file():
        raise SystemExit(f"dataset not found: {data_path}")

    print(f"[task] {task_label}  data={data_path}  summary_dir={summary_dir}")
    summary = run(args.config, data_path, tp_size=args.tp)
    out_path = _write_summary(summary, args.config, summary_dir)

    print("[summary]")
    for k, v in summary.items():
        if k in ("args", "submission_json", "submission_py"):
            continue
        print(f"  {k}: {v}")
    print(f"[summary] written to {out_path}")
    print(f"[summary] latest  -> {Path(out_path).parent / 'latest.json'}")
    print(f"[summary] index   -> {Path(out_path).parent / 'INDEX.jsonl'}")
