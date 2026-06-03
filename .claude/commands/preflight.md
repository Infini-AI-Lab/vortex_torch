---
description: Run check_engine_config locally (CPU-only) on a submission.
argument-hint: <submission-name>
---

Resolve `$1` to a config path. Most submissions live under your
agent tag, so try in order: `submissions/<tag>/$1.json`,
`submissions/$1.json` (top-level for examples), or treat `$1`
itself as a path if it ends in `.json`. Then run the cheap
pre-flight check:

```bash
python algorithm_scientist/detect_env.py >/dev/null 2>&1   # establish the env if not already
RUN="conda run -n vortex_v1 python"        # ← detect_env.py's recommended prefix; substitute if different
$RUN -c "from vortex_torch.engine.sgl import check_engine_config; check_engine_config('<resolved path>')"
```

Capture the output. If it raises, paste the relevant traceback line
and cross-reference it with [AI/AGENTS.md §5](AI/AGENTS.md)
"Common failures" table. Suggest the minimum edit needed to make it
pass, but do **not** edit files unless the user asks you to.

If it passes, remind the user that the next step is `/benchmark $1`
(debug, single variant) or `/batch-benchmark` (the sanctioned
batch that fills every local GPU, launched directly via python).
