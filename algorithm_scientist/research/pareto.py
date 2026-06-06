"""Pareto leaderboard — the running (throughput, mean@16) frontier across runs.

Scans the ``summary_*submissions/`` trees (all agent tags, or one ``--tag``) for
a task, reads each submission's ``latest.json``, computes the Pareto-non-dominated
frontier (maximize BOTH throughput and mean@16), and marks each submission as
on-frontier or dominated (and by what). The "where do we stand" view; full
attention is the reference if a baseline run is present.

Usage
-----
::

    python algorithm_scientist/research/pareto.py --task aime24
    python algorithm_scientist/research/pareto.py --task aime24 --tag claude_opus_4_8
"""

import argparse
import json
from pathlib import Path

SUMMARY_DIRS = {
    "aime24": "summary_submissions",
    "aime25": "summary_aime25_submissions",
    "aime26": "summary_aime26_submissions",
    "amc23": "summary_amc23_submissions",
}


def _collect(task, tag):
    base = Path(SUMMARY_DIRS[task])
    if not base.is_dir():
        return []
    roots = [base / tag] if tag else [base]
    pts = []
    for root in roots:
        if not root.is_dir():
            continue
        for latest in sorted(root.rglob("latest.json")):
            try:
                s = json.loads(latest.read_text())
            except Exception:
                continue
            acc, tput = s.get("mean@16"), s.get("throughput")
            if acc is None or tput is None:
                continue
            name = str(latest.parent.relative_to(base))
            pts.append({"name": name, "mean@16": acc, "throughput": tput,
                        "hash": s.get("content_hash"),
                        "model": (s.get("args") or {}).get("model_path")})
    return pts


def _frontier(pts):
    """A point is dominated if another has >= throughput AND >= mean@16 (one strictly)."""
    for p in pts:
        dominator = None
        for q in pts:
            if q is p:
                continue
            ge = q["throughput"] >= p["throughput"] and q["mean@16"] >= p["mean@16"]
            gt = q["throughput"] > p["throughput"] or q["mean@16"] > p["mean@16"]
            if ge and gt:
                dominator = q["name"]
                break
        p["on_frontier"] = dominator is None
        p["dominated_by"] = dominator
    return pts


def main():
    ap = argparse.ArgumentParser(description="Pareto leaderboard on (throughput, mean@16).")
    ap.add_argument("--task", default="aime24", choices=list(SUMMARY_DIRS))
    ap.add_argument("--tag", default=None, help="restrict to one agent tag")
    args = ap.parse_args()

    pts = _frontier(_collect(args.task, args.tag))
    if not pts:
        print(f"(no results for task={args.task}"
              f"{', tag=' + args.tag if args.tag else ''})")
        return

    front = sorted([p for p in pts if p["on_frontier"]],
                   key=lambda p: -p["throughput"])
    print(f"# Pareto frontier — {args.task}  ({len(pts)} runs, {len(front)} on frontier)\n")
    print(f"{'':1} {'submission':<34} {'mean@16':>8} {'tput':>8}  status")
    for p in sorted(pts, key=lambda p: (-p["mean@16"], -p["throughput"])):
        star = "★" if p["on_frontier"] else " "
        status = "FRONTIER" if p["on_frontier"] else f"< {p['dominated_by']}"
        print(f"{star} {p['name']:<34} {p['mean@16']:>8.4f} {p['throughput']:>8.1f}  {status}")
    print("\n★ = Pareto-non-dominated. Pick frontier points as the running best "
          "(memory.md §5); a new variant is only interesting if it joins or pushes the frontier.")


if __name__ == "__main__":
    main()
