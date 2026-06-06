"""Run a sparse-attention submission against the RULER benchmark (fixed protocol).

Given a submission's engine JSON (e.g. ``submissions/<your_name>.json``),
this script:

  1. Validates the config via :func:`check_engine_config`
     (JSON shape, pow-2 constraints, ``@register(<name>)`` exists,
     flow compiles on a small grid).
  2. Boots an sglang engine with the submission's ``vortex_*`` settings
     plus the fixed RULER protocol constants below.
  3. Runs :file:`examples/ruler/validation.jsonl` with substring-match scoring.
  4. Writes a per-run summary JSON to ``summary_ruler_submissions/``.

All benchmark-protocol settings are fixed — the **only** CLI argument
is the submission config path.

Usage
-----
::

    python algorithm_scientist/run_ruler.py \\
        --config submissions/example_block_sparse_attention.json
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import AutoTokenizer

import vortex_torch  # noqa: F401  — registers the attention backend
from vortex_torch.engine.sgl import MODEL_PATH, check_engine_config, get_engine


# ---------------------------------------------------------------------------
# Fixed benchmark protocol — do not change
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS              = 64
DATA_PATH                   = "examples/ruler/validation.jsonl"
SUMMARY_DIR                 = "summary_ruler_submissions"


# ---------------------------------------------------------------------------
# Engine booting
# ---------------------------------------------------------------------------

def _load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    print(f"[pre-flight] validating {config_path}")
    config = check_engine_config(config_path)
    print(f"[pre-flight] OK — module={config.get('vortex_module_name')}")
    return config


def _build_engine_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pass the submission JSON through; :func:`get_engine`'s defaults
    fill in anything (``model_path``, ``mem_fraction_static``, …) the
    JSON omits."""
    return dict(config)


def _boot_engine(kwargs: Dict[str, Any]):
    print(f"[engine] booting — model={kwargs.get('model_path', '<default>')}, "
          f"module={kwargs.get('vortex_module_name')}, "
          f"kv={kwargs.get('kv_cache_dtype', 'auto')}, "
          f"mem_fraction_static={kwargs.get('mem_fraction_static', '<default>')}")
    return get_engine(**kwargs)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _load_ruler_data(data_path: Path):
    with open(data_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    inputs  = [row["input"]     for row in rows]
    answers = [row["outputs"][0] for row in rows]
    return inputs, answers


def _build_prompts(inputs, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [[{"role": "user", "content": x}] for x in inputs]
    return [
        tokenizer.apply_chat_template(
            t,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for t in texts
    ]


def _score_results(outputs_raw, gold_answers):
    results = []
    for res, answer in zip(outputs_raw, gold_answers):
        hit = 1.0 if answer in res["text"] else 0.0
        results.append({
            "score":       hit,
            "prediction":  res["text"],
            "gold":        answer,
            "e2e_latency": res["meta_info"]["e2e_latency"],
            "num_tokens":  res["meta_info"]["completion_tokens"],
        })
    return results


def _summarize(results) -> Dict[str, Any]:
    total_accuracy = sum(r["score"] for r in results)
    total_tokens   = sum(r["num_tokens"] for r in results)
    e2e_time       = max((r["e2e_latency"] for r in results), default=0.0)
    count          = len(results)
    return {
        "accuracy":      total_accuracy / count if count else 0.0,
        "total_example": count,
        "e2e_time":      e2e_time,
        "total_tokens":  total_tokens,
        "throughput":    (total_tokens / e2e_time) if e2e_time > 0 else 0.0,
    }


def run(config_path: Path) -> Dict[str, Any]:
    config        = _load_and_validate_config(config_path)
    engine_kwargs = _build_engine_kwargs(config)
    llm           = _boot_engine(engine_kwargs)

    # The tokenizer must match the model the engine actually booted with.
    tokenizer_model = engine_kwargs.get("model_path") or MODEL_PATH

    inputs, gold_answers = _load_ruler_data(Path(DATA_PATH))
    prompts = _build_prompts(inputs, tokenizer_model)

    sampling_params = {
        "temperature":    0.6,
        "top_p":          0.95,
        "top_k":          20,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    print(f"[benchmark] {len(prompts)} prompts")

    outputs_raw = llm.generate(prompts, sampling_params)
    results     = _score_results(outputs_raw, gold_answers)
    summary     = _summarize(results)

    summary["args"] = {
        "config_path":         str(config_path),
        "vortex_module_name":  config.get("vortex_module_name"),
        "vortex_module_path":  config.get("vortex_module_path"),
        "model_path":          tokenizer_model,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "mem_fraction_static": engine_kwargs.get("mem_fraction_static"),
        "data_path":           DATA_PATH,
    }
    return summary


# ---------------------------------------------------------------------------
# Summary writing (uses RULER-specific output directory)
# ---------------------------------------------------------------------------

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
    parts = config_path.with_suffix("").parts
    if "submissions" in parts:
        idx  = parts.index("submissions")
        tail = parts[idx + 1:]
        if tail:
            return Path(*tail)
    return Path(config_path.stem)


def _write_summary(summary: Dict[str, Any], config_path: Path) -> str:
    rel     = _summary_subpath(config_path)
    tag     = str(rel)
    run_dir = Path(SUMMARY_DIR) / rel
    run_dir.mkdir(parents=True, exist_ok=True)

    config_text, module_text, content_hash = _read_submission_artifacts(config_path)

    finished_at  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    summary = dict(summary)
    summary["content_hash"]         = content_hash
    summary["finished_at"]          = finished_at
    summary["cuda_visible_devices"] = cuda_visible
    summary["submission_json"]      = config_text
    summary["submission_py"]        = module_text

    fname    = f"{finished_at}__{content_hash}.json"
    out_path = run_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    _append_index(
        run_dir / "INDEX.jsonl",
        {
            "finished_at":   finished_at,
            "content_hash":  content_hash,
            "cuda_visible":  cuda_visible,
            "submission":    tag,
            "file":          fname,
            "accuracy":      summary.get("accuracy"),
            "throughput":    summary.get("throughput"),
            "e2e_time":      summary.get("e2e_time"),
            "total_tokens":  summary.get("total_tokens"),
            "total_example": summary.get("total_example"),
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


# ---------------------------------------------------------------------------
# CLI (only one knob: the submission config)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission and run it on RULER with the fixed protocol.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("submissions/example_block_sparse_attention.json"),
        help="Path to the submission JSON (default: the bundled example).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    if not Path(DATA_PATH).is_file():
        raise SystemExit(f"dataset not found: {DATA_PATH}")

    summary  = run(args.config)
    out_path = _write_summary(summary, args.config)

    print("[summary]")
    for k, v in summary.items():
        if k in ("args", "submission_json", "submission_py"):
            continue
        print(f"  {k}: {v}")
    print(f"[summary] written to {out_path}")
    print(f"[summary] latest  -> {Path(out_path).parent / 'latest.json'}")
    print(f"[summary] index   -> {Path(out_path).parent / 'INDEX.jsonl'}")
