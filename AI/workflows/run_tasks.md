# Running submissions on a task with any model

The eval `prompt` field in `examples/math/<task>.jsonl` is **tokenizer/chat-template
bound** — a different model needs a regenerated jsonl (that is why
`aime26.jsonl`, `aime26_glm.jsonl`, `aime26_minimax.jsonl` all exist). So the
model in a submission's JSON (`model_path`) and the model used to build the eval
jsonl **must match**.

## Two executables (prepared for this workflow)

- `examples/misc/make_task.py` — build a task jsonl for a model's tokenizer.
- `algorithm_scientist/run_submission.py` — task-generalized runner
  (`--task` or `--data`); supersedes `run_submission_aime24.py` / `_amc23.py`
  (kept as back-compat shims).

## Built-in tasks

| task   | HF dataset                | split | summary dir                     |
|--------|---------------------------|-------|---------------------------------|
| aime24 | HuggingFaceH4/aime_2024   | train | summary_submissions             |
| aime25 | math-ai/aime25            | test  | summary_aime25_submissions      |
| aime26 | math-ai/aime26            | test  | summary_aime26_submissions      |
| amc23  | math-ai/amc23 (best-effort) | test | summary_amc23_submissions      |

`lcbv5` (LiveCodeBench) is **not** runnable here — it needs code-execution
scoring, not extractive math matching. Use a dedicated harness.

## Flow (model + task)

```bash
# 1. Default model (Qwen3-1.7B) + a built-in task — jsonl already exists:
python algorithm_scientist/run_submission.py --task aime25 \
    --config submissions/<tag>/batch_0_id0.json

# 2. A different model — regenerate the jsonl for ITS tokenizer first, then
#    point the runner at it with --data:
MODEL=Qwen/Qwen3-4B
python examples/misc/make_task.py --task aime26 --model "$MODEL" \
    --output examples/math/aime26__qwen3_4b.jsonl
python algorithm_scientist/run_submission.py \
    --data examples/math/aime26__qwen3_4b.jsonl \
    --config submissions/<tag>/batch_0_id0.json   # this JSON's model_path == $MODEL
```

The runner writes the same content-hashed, per-agent-isolated summaries as
before (`<summary_dir>/<tag>/<stem>/{<ts>__<hash>.json, latest.json,
INDEX.jsonl}`). Fixed protocol: 16 trials, 4096-token input cap, **32768**
max new tokens, `mem`/topk from the JSON, math extractive-match scorer →
`mean@16`, `pass@16`, `throughput`.

**Runtime & timeout.** Wall-clock ≈ (questions × 16 × output tokens) ÷
throughput — it varies a lot by model + task (and the 32768 budget raises it).
`/iterate` and `/batch-benchmark` have the **agent decide** a `TIMEOUT_MIN` per
run and wrap each child in `timeout ${TIMEOUT_MIN}m` (exit 124 / no `latest.json`
⇒ timed out → failed). There is no fixed limit.

## Wiring a model into a submission

A submission JSON sets `"model_path": "<hf-id>"`. `/iterate --model <hf-id>`
writes that field into every generated config and builds the matching task
jsonl with `make_task.py` before launching. If the model is MLA (DeepSeek/GLM)
the flow must be a `vFlowMLA` subclass — run `/support-model <hf-id>` first.
