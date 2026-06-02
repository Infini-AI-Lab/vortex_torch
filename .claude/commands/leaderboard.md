---
description: Show the running Pareto frontier on (throughput, mean@16) across all runs for a task — who's on the frontier, who's dominated, where a new result lands.
argument-hint: [--task aime24|aime25|aime26|amc23] [--tag <agent-tag>]
---

The "where do we stand" view. Computes the Pareto-non-dominated frontier from
every `summary_*submissions/` result for a task.

```bash
python algorithm_scientist/research/pareto.py --task ${TASK:-aime24} ${TAG:+--tag $TAG}
```

- `--task` (default aime24); `--tag` restricts to one agent, else all tags.
- ★ rows are on the frontier; others show what dominates them.

Use it to: keep `algorithm_scientist/memory.md §5` (running winners) in sync with
the actual frontier; decide whether a just-finished variant is interesting (it is
only if it **joins or pushes** the frontier — equal-or-worse on both axes is
noise); and pick which frontier points to feature in `/write-paper`. Pair with
`efficiency.py` to sanity-check whether a frontier point's throughput is near its
analytical ceiling (if not → `/add-ops investigate`).
