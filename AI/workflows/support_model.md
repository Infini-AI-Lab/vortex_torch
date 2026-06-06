# Supporting a model

A model is "supported" when vortex's attention backend handles its geometry and
a real engine boots and generates correctly. Two geometries:

- **MHA / GQA** (Qwen, Llama, Mistral, Olmo) → non-MLA vortex path; backends
  `flashinfer` / `trtllm`; flow is a standard `vFlow`.
- **MLA** (DeepSeek, GLM) → MLA path; backends `trtllm_mla` (decode) / `triton`
  / `cuda_mla`; flow must be a `vFlowMLA` subclass.

The backend shims live in
[`vortex_torch/engine/sgl/integration.py`](../../vortex_torch/engine/sgl/integration.py)
(`_make_flashinfer_shim`, `_make_trtllm_mla_shim`, `_make_triton_shim`,
`_create_cuda_mla_backend`). KV pools are in `engine/sgl/memory_pool*.py`.

## Env split

GLM-family (`glm4_moe*`) only loads in the **`vortex_glm`** conda env
(transformers 5.0). Everything else uses **`vortex_v1`**. The static checker
reports which.

## Verify flow (what `/support-model <hf-id>` runs)

1. **Static check (CPU, no GPU):**
   ```bash
   python algorithm_scientist/support_model.py <hf-id>
   ```
   Prints geometry, recommended backend(s), recommended env, shapes. Exit 0 =
   recognized geometry; exit 2 = unknown → needs wiring.
2. **Live verify (GPU):** boot a tiny engine on a default flow with this model
   (`model_path: <hf-id>`) and run a quick RULER sanity pass
   (`algorithm_scientist/run_ruler.py`, see [[feedback_ruler_tp1]] for the tp=1
   setup). A passing RULER (≥0.85) = the attention path is structurally sound.
3. **Tiny generation:** one math prompt through `run_submission.py --task aime24`
   on a handful of trials (or the server in `examples/misc/server_launch.sh`) to
   confirm end-to-end decode.

## Wiring a new (unsupported) model

When the static check returns exit 2 / RULER fails to boot:

1. Determine geometry from `config.json` (`kv_lora_rank` ⇒ MLA).
2. If MLA and unhandled: extend the MLA shims / pool in `integration.py` /
   `memory_pool_mla.py`; ensure `vFlowMLA.initialize` gets `kv_lora_rank` /
   `qk_rope_head_dim`.
3. If MHA/GQA and unhandled (unusual head_dim, sink tokens, SWA): adjust the
   flashinfer/trtllm shim and the non-MLA pool (`memory_pool.py`).
4. Re-run the static check, then RULER, then the tiny generation. Record the
   model + what was needed in `algorithm_scientist/memory.md`.

Keep changes minimal and upstream-shaped — the integration module exists to keep
sglang close to upstream (see its module docstring).
