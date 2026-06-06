---
description: Check whether a model is supported under vortex (geometry → backend → boot tiny engine → RULER), and wire support if it isn't, then re-verify.
argument-hint: <hf-id-or-local-path>
---

Determine whether `$1` runs under vortex_torch's sparse-attention path, and if
not, **wire it up** and re-verify. Reference:
[AI/workflows/support_model.md](../../AI/workflows/support_model.md).

## Step 0 — env + static check (CPU, no GPU)

```bash
python algorithm_scientist/detect_env.py    # dont assume vortex_v1; adopt the recommended prefix
RUN="conda run -n vortex_v1 python"   # the recommended prefix (substitute if different)
$RUN algorithm_scientist/support_model.py "$1"
```
This prints the geometry (MLA vs MHA/GQA), recommended backend(s), recommended
conda env (`vortex_glm` for GLM-family, else `vortex_v1`), and shapes. Activate
the recommended env now.
- **exit 0** (known geometry) → go to Step 1.
- **exit 2** (unknown geometry) → go to Step 3 (wire support).

## Step 1 — live verify (GPU)

**Detect a free GPU** (the set is dynamic — never hardcode), then boot a tiny
engine on a default flow with `model_path: $1` and run the RULER quality gate:
```bash
FREE_GPUS=($(algorithm_scientist/free_gpus.sh)) || { echo "no free GPU — wait"; exit 1; }
CUDA_VISIBLE_DEVICES=${FREE_GPUS[0]} $RUN algorithm_scientist/run_ruler.py \
  --config <a minimal submission JSON whose model_path is $1>
```
A RULER ≥ 0.85 means the attention path is structurally sound for this model.
(See [[feedback_ruler_tp1]] for the tp=1 RULER setup.)

## Step 2 — tiny end-to-end generation

Run a handful of trials via `algorithm_scientist/run_submission.py --task aime24`
(regenerate the task jsonl for this model first with `examples/misc/make_task.py
--model $1` — the eval prompt is tokenizer-bound), or launch
`examples/misc/server_launch.sh $1 1` and send one chat completion. Confirm coherent
decode. If all three pass → **report SUPPORTED** with the backend + env, and stop.

## Step 3 — wire a new (unsupported) model

When the static check returns exit 2 or the boot fails:
1. Read `config.json` (`kv_lora_rank` ⇒ MLA). Read
   [vortex_torch/engine/sgl/integration.py](../../vortex_torch/engine/sgl/integration.py)
   (backend shims) and `memory_pool*.py` (KV pools).
2. **MLA, unhandled:** extend the MLA shim / `memory_pool_mla.py`; make
   `vFlowMLA.initialize` receive `kv_lora_rank` / `qk_rope_head_dim`.
   **MHA/GQA, unhandled** (odd head_dim, sinks, SWA): adjust the
   flashinfer/trtllm shim and `memory_pool.py`.
3. Keep edits minimal and upstream-shaped (the integration module's whole point
   is keeping sglang close to upstream).
4. Re-run Step 0 → Step 1 → Step 2. Record the model + what was needed in
   `algorithm_scientist/memory.md`.

If a needed kernel/op is missing for this geometry, hand off to **`/add-ops`**
(the `vortex-op-author` subagent), then resume.

## Output

`model | geometry | backend | env | RULER | tiny-gen | verdict (SUPPORTED /
WIRED+verified / BLOCKED:<reason>)`.
