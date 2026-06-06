# `algorithm_scientist/` — the agent-researcher bundle

Everything an agent (or a human) needs to turn a sparse-attention idea into a
measured accuracy/throughput result, in one folder. Scripts run from the repo
root (`vortex_torch`) and reference each other with `algorithm_scientist/…`
paths.

The loop is **driven directly by Claude Code / Codex** through the slash
commands in the repo-root [`.claude/`](../.claude/) folder — there is no
standalone driver script. The commands call the scripts here.

## What lives here

| File | Role |
|---|---|
| [`detect_env.py`](detect_env.py) | Probe the host (conda / uv / venv / docker) and recommend a run prefix where `import vortex_torch` works. Run this **first** every session. |
| [`free_gpus.sh`](free_gpus.sh) | Print the currently-free GPU indices (exit 1 if none). Every GPU launch detects free GPUs with this — never hardcode a device. |
| [`run_submission.py`](run_submission.py) | The generalized task runner — `--task`/`--data` choose the benchmark, model comes from the submission JSON's `model_path` (16 trials, single GPU, 32768 max-new-tokens). Writes per-submission summaries under `summary_submissions/<tag>/<stem>/`. |
| [`run_submission_aime24.py`](run_submission_aime24.py) / [`run_submission_amc23.py`](run_submission_amc23.py) | Back-compat shims over `run_submission.py` for the AIME24 / AMC23 protocols. |
| [`run_ruler.py`](run_ruler.py) | RULER quality gate used to pre-filter variants before a full task run. |
| [`support_model.py`](support_model.py) | Check/wire vortex support for a new model (geometry → backend → tiny-engine boot → RULER). |
| [`collect_results.py`](collect_results.py) | Aggregate `summary_*submissions/` into a leaderboard / Pareto view. |
| [`research/`](research/) | Offline research toolkit: `capture_trace`, `eval_recall`, `analyze_attention`, `efficiency` (roofline), `pareto`, a `journal.md`, and `methods/` baselines (quest, h2o, streaming, centroid, …). |
| [`memory.md`](memory.md) | Persistent state across sessions — read at start, update before stopping. |
| [`iterate_kickoff.md`](iterate_kickoff.md) | Paste-in prompt to boot a fresh session straight into the iterate loop. |

## Authoritative instructions

The text instructions the agent reads live in [`AI/`](../AI/):

- [`AI/AGENTS.md`](../AI/AGENTS.md) — submission contract, rules, benchmark protocol, objective.
- [`AI/tutorials/`](../AI/tutorials/) — user-facing tutorials.
- [`AI/workflows/`](../AI/workflows/) — multi-step workflows (add-op, paper, research toolkit, run-tasks, support-model).
- [`AI/developer_guides/`](../AI/developer_guides/) — framework-internal deep dives.

## Driving it

Open a Claude Code (or Codex) session at the repo root and use the slash
commands wired in [`.claude/`](../.claude/):

- `/setup-env` → establish a working env (then use the returned run prefix).
- `/iterate <model> <task>` → the long-horizon design → preflight → RULER → run → analyse loop.
- `/innovate <N> [model]` → draft novel compile-checked submissions.
- `/research <question>` → full researcher loop (survey → analyse → hypothesize → recall → RULER → run → journal).
- `/ablate`, `/add-ops`, `/support-model`, `/leaderboard`, `/write-paper`, `/reproduce-paper`, `/debug` → the supporting tools.

To run a single variant by hand (debug only):

```bash
RUN="$(python algorithm_scientist/detect_env.py | sed -n 's/^run-prefix: //p')"   # e.g. conda run -n vortex_v1 python
CUDA_VISIBLE_DEVICES=$(algorithm_scientist/free_gpus.sh | head -1) \
  $RUN algorithm_scientist/run_submission.py --task aime24 --config submissions/<tag>/<name>.json
```
