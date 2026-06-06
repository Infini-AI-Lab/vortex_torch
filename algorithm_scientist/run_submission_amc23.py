"""Run a sparse-attention submission against AMC23 (fixed protocol).

Given a submission's engine JSON (e.g. ``submissions/<your_name>.json``),
this script:

  1. Validates the config via :func:`check_engine_config`
     (JSON shape, pow-2 constraints, ``@register(<name>)`` exists,
     flow compiles on a small grid).
  2. Boots an sglang engine with the submission's ``vortex_*`` settings
     plus the fixed AMC23 protocol constants below.
  3. Runs :file:`examples/math/amc23.jsonl` with 16 trials.
  4. Scores with lighteval's ``MultilingualExtractiveMatchMetric``
     (same setup as ``examples/math/verify_algo.py``).
  5. Writes a per-run summary JSON to ``summary_amc23_submissions/``.

All benchmark-protocol settings are fixed — the **only** CLI argument
is the submission config path.

Usage
-----
::

    python algorithm_scientist/run_submission_amc23.py \\
        --config submissions/example_block_sparse_attention.json
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
TP_SIZE                     = 1
DATA_PATH                   = "examples/math/amc23.jsonl"
SUMMARY_DIR                 = "summary_amc23_submissions"


# ---------------------------------------------------------------------------
# Engine booting
# ---------------------------------------------------------------------------

def _load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Run the pre-flight check and return the parsed config dict."""
    print(f"[pre-flight] validating {config_path}")
    config = check_engine_config(config_path)
    print(f"[pre-flight] OK — module={config.get('vortex_module_name')}")
    return config


def _build_engine_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the submission JSON with the fixed protocol settings.

    Everything else (``model_path``, ``mem_fraction_static``, etc.) is
    either taken from the submission JSON or filled in by
    :func:`get_engine`'s defaults.
    """
    kwargs = dict(config)
    kwargs["tp_size"]              = TP_SIZE
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


def run(config_path: Path) -> Dict[str, Any]:
    config = _load_and_validate_config(config_path)
    engine_kwargs = _build_engine_kwargs(config)
    llm = _boot_engine(engine_kwargs)

    requests = _load_requests(Path(DATA_PATH)) * TRIALS
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
        "tp_size":                    TP_SIZE,
        "data_path":                  DATA_PATH,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI (only one knob: the submission config)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission and run it on AMC23 with the fixed protocol.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("submissions/example_block_sparse_attention.json"),
        help="Path to the submission JSON (default: the bundled example).",
    )
    return parser.parse_args()


def _read_submission_artifacts(
    config_path: Path,
) -> Tuple[str, Optional[str], str]:
    """Return (config_json_text, module_py_text or None, content_hash).

    ``content_hash`` is a short sha256 over the config JSON bytes + the
    module .py bytes (if resolvable). Two runs of the exact same
    (.py, .json) pair produce the same hash — so re-runs are visible
    at a glance, and a code change forces a new hash.
    """
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


def _append_index(
    index_path: Path,
    row: Dict[str, Any],
) -> None:
    """Append a one-line JSONL record so the directory has a grep-able index."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary_subpath(config_path: Path) -> Path:
    """Mirror the config's location under ``submissions/`` into the summary tree.

    ``submissions/example_x.json``                       → ``example_x``
    ``submissions/<tag>/batch_3_id5.json``               → ``<tag>/batch_3_id5``
    ``submissions/<tag>/<sub>/<stem>.json`` (nested)     → ``<tag>/<sub>/<stem>``
    Configs not under ``submissions/`` fall back to just the file stem
    (preserving the old single-level layout for ad-hoc paths).

    The agent-tagged path (``submissions/<tag>/...``) preserves per-agent
    isolation in ``summary_amc23_submissions/`` so two agents producing the
    same ``batch_x_idy`` stem never clobber each other's ``latest.json``.
    """
    parts = config_path.with_suffix("").parts  # drop the .json
    if "submissions" in parts:
        idx = parts.index("submissions")
        tail = parts[idx + 1:]
        if tail:
            return Path(*tail)
    return Path(config_path.stem)


def _write_summary(summary: Dict[str, Any], config_path: Path) -> str:
    """Write the summary into a per-submission subfolder with a content-hashed name.

    Layout (mirrors the config's location relative to ``submissions/``):
        summary_amc23_submissions/
            <tag>/<config_stem>/                          — agent-tagged batches
                <timestamp>__<hash12>.json                — full summary + embedded artifacts
                INDEX.jsonl                               — one row per run (headline metrics)
                latest.json                               — symlink to the most recent run
            <config_stem>/                                — top-level examples / ad-hoc
                ...
    """
    rel = _summary_subpath(config_path)
    tag = str(rel)                            # used downstream as the "submission" label
    run_dir = Path(SUMMARY_DIR) / rel
    run_dir.mkdir(parents=True, exist_ok=True)

    config_text, module_text, content_hash = _read_submission_artifacts(config_path)

    # Microsecond precision so 8 concurrent runs of the same submission
    # (one per GPU in a batched node job) never clobber each other.
    finished_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    summary = dict(summary)  # shallow copy so we don't mutate the caller's dict
    summary["content_hash"] = content_hash
    summary["finished_at"] = finished_at
    summary["cuda_visible_devices"] = cuda_visible
    summary["submission_json"] = config_text
    summary["submission_py"] = module_text

    fname = f"{finished_at}__{content_hash}.json"
    out_path = run_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    # Append a one-line index row so you can `grep / sort / less` at a glance.
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

    # Maintain a `latest.json` pointer that always points at the newest run.
    latest = run_dir / "latest.json"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(fname)
    except OSError:
        # Filesystem doesn't support symlinks — fall back to a plain copy.
        latest.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    return str(out_path)


if __name__ == "__main__":
    args = parse_args()

    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    if not Path(DATA_PATH).is_file():
        raise SystemExit(f"dataset not found: {DATA_PATH}")

    summary = run(args.config)
    out_path = _write_summary(summary, args.config)

    print("[summary]")
    for k, v in summary.items():
        if k in ("args", "submission_json", "submission_py"):
            continue
        print(f"  {k}: {v}")
    print(f"[summary] written to {out_path}")
    print(f"[summary] latest  -> {Path(out_path).parent / 'latest.json'}")
    print(f"[summary] index   -> {Path(out_path).parent / 'INDEX.jsonl'}")
