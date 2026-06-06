"""Aggregate benchmark results into a table for the paper toolchain.

Scans the ``summary_*submissions/`` trees (one per task) for a given agent tag,
reads each submission's ``latest.json``, and emits a combined table (markdown +
JSON) of ``mean@16`` / ``pass@16`` / ``throughput`` per submission per task.
``/write-paper`` consumes the JSON to populate figures/tables.

Usage
-----
::

    python algorithm_scientist/collect_results.py --tag claude_opus_4_8
    python algorithm_scientist/collect_results.py --tag <tag> --json results.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# task label -> summary dir (mirror of run_submission.TASKS).
SUMMARY_DIRS = {
    "aime24": "summary_submissions",
    "aime25": "summary_aime25_submissions",
    "aime26": "summary_aime26_submissions",
    "amc23":  "summary_amc23_submissions",
}


def _read_latest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(tag: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, sdir in SUMMARY_DIRS.items():
        base = Path(sdir) / tag
        if not base.is_dir():
            continue
        for latest in sorted(base.rglob("latest.json")):
            s = _read_latest(latest)
            if not s:
                continue
            name = str(latest.parent.relative_to(base))
            rows.append({
                "task": task,
                "submission": name,
                "model": (s.get("args") or {}).get("model_path"),
                "content_hash": s.get("content_hash"),
                "mean@16": s.get("mean@16"),
                "pass@16": s.get("pass@16"),
                "throughput": s.get("throughput"),
                "e2e_time": s.get("e2e_time"),
                "total_tokens": s.get("total_tokens"),
                "finished_at": s.get("finished_at"),
            })
    return rows


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_(no results found)_"
    hdr = "| task | submission | model | mean@16 | pass@16 | throughput | hash |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in sorted(rows, key=lambda x: (x["task"], str(x["submission"]))):
        lines.append(
            f"| {r['task']} | {r['submission']} | {r.get('model') or '—'} "
            f"| {_fmt(r['mean@16'])} | {_fmt(r['pass@16'])} "
            f"| {_fmt(r['throughput'], 1)} | {r.get('content_hash') or '—'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Aggregate benchmark results for a tag.")
    ap.add_argument("--tag", required=True, help="agent tag (submissions/<tag>)")
    ap.add_argument("--json", type=str, default=None, help="also write JSON here")
    args = ap.parse_args()

    rows = collect(args.tag)
    print(to_markdown(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n[collect_results] wrote {len(rows)} rows -> {args.json}")


if __name__ == "__main__":
    main()
