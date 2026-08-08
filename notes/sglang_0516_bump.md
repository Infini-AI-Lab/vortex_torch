# sglang 0.4.9/0.5.9 → 0.5.16 bump notes

Vendored tree: `third_party/sglang/v0.5.16/sglang` (upstream tag `v0.5.16`,
commit `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`). Upstream's own `.claude/`
rules dir is stripped from the vendored copy so it can't leak sglang's internal
dev conventions into this repo's agent context.

## The vortex patch on v0.5.9 (baseline)

Diffing vendored v0.5.9 against upstream v0.5.9 gives **13 hunks across 6
files** — that is the entire surface to re-apply:

| file | hunk |
|---|---|
| `server_args.py` | `cuda_mla` in backend choices; `vortex: Optional[Any]` field; `__getattr__` legacy `vortex_*` shim + `_VORTEX_LEGACY_DEFAULTS`; Olmo3 flashinfer assert relaxed; PD-disagg forces `disable_overlap_schedule`; `--vortex-config` CLI arg |
| `model_runner.py` | `self.block_size = server_args.vortex_block_size`; `self.sparse_attention = build_sparse_flow(self)` |
| `model_runner_kv_cache_mixin.py` | `get_cell_size_per_token` vortex sub-branches (MLA + MHA); `int()` wrap on token estimate; `4096 → 1024` req cap under vortex; dense-MLA branch guarded with `and not enable_vortex_sparsity`; `make_kv_pool` branch |
| `models/utils.py` | `enable_fused_set_kv_buffer` honours `pool.supports_fused_set_kv_buffer` |
| `input_buffers.py` | `req_pool_indices` int32 → int64 |
| `disaggregation/decode.py` | decode-side `rebuild_aux(...)` after KV transfer |

## Upstream drift 0.5.9 → 0.5.16 that breaks the patch

1. **`model_runner_kv_cache_mixin.py` no longer exists.** Split into
   `model_executor/pool_configurator.py` (cell-size / pool sizing, class
   `DefaultPoolConfigurator._compute_cell_size`) and
   `mem_cache/kv_cache_configurator.py` (`KVCacheConfigurator` — builds
   req_to_token_pool + token_to_kv_pool + allocator, entry `configure()`).
   `ModelRunner.init_memory_pool` → `alloc_memory_pool()`, which is now called
   by `tp_worker`, *separately from and after* `initialize()`.
2. **`get_attention_tp_size()` is gone** from `layers/dp_attention.py`. The
   replacement is `sglang.srt.runtime_context.get_parallel().attn_tp_size`.
   vortex imports the old symbol in `integration.py`, `memory_pool.py`,
   `attention_backend/flashinfer.py`, `trtllm.py`.
3. **`srt/utils.py` became a package** (`srt/utils/`), but
   `srt/utils/__init__.py` does `from sglang.srt.utils.common import *`, so
   `is_flashinfer_available` / `kill_process_tree` imports still resolve.
4. **The `input_buffers.py` int32→int64 hunk is obsolete** — upstream now
   allocates every `req_pool_indices` as `int64` (`runner/base_runner.py`,
   `runner_utils/buffers.py`, `cpu_graph_runner.py`, …). Drop the hunk.
5. **`is_deepseek_nsa` / `NSATokenToKVPool` renamed** to `is_deepseek_dsa` /
   `DSATokenToKVPool`, `get_nsa_index_head_dim` → `get_dsa_index_head_dim`.
6. **`enable_fused_set_kv_buffer` lost the opt-out hook** (see below).
7. **The cuda-graph `AttentionBackend` ABI was replaced.** 0.5.9 drove graph
   metadata through
   `init_forward_metadata_{capture,replay}_cuda_graph`; 0.5.16 removed that
   pair from the ABC *and* from the graph runners, replacing it with a
   3-method contract: `init_forward_metadata` /
   `init_forward_metadata_out_graph(fb, in_capture=False)` /
   `init_forward_metadata_in_graph(fb)`. All **five** vortex backends
   (`flashinfer`, `trtllm`, `trtllm_mla`, `triton_mla`, `cuda_mla`) implement
   only the legacy pair, so under 0.5.16 their graph metadata would **never be
   initialized** — silently, since nothing calls the old methods any more.
   `init_forward_metadata` itself is unchanged (still the eager entry point).

   Handled by `LegacyCudaGraphABIMixin` in
   `vortex_torch/engine/sgl/compat.py`: it implements
   `init_forward_metadata_out_graph`, unpacks the ForwardBatch-like view the
   runner passes (`build_replay_fb_view` supplies `batch_size`,
   `req_pool_indices`, `seq_lens`, `seq_lens_sum`, `seq_lens_cpu`,
   `encoder_lens`, `forward_mode`, `spec_info` — everything the legacy
   signatures need) and dispatches to capture (`in_capture=True`) or replay.
   `init_forward_metadata_in_graph` stays the base no-op: vortex's planning is
   host-side, so it records no graph-recordable metadata ops. Each backend now
   declares `class VortexXBackend(*attention_backend_base())`, which interposes
   the mixin only on releases that need it — so the same source still works
   against the vendored v0.5.9 tree.

## Runtime breakages found by actually booting (all fixed)

Found by running RULER on a B200; each one was a hard crash, not a warning.

8. **`make_kv_pool` read runner state that does not exist yet.** 0.5.16 builds
   the pool inside `KVCacheConfigurator.configure()`, *before* the runner is
   given `max_total_num_tokens` (the configurator returns it) and before
   `req_to_token_pool` is assigned; the layer span moved to `layer_info`
   (`num_effective_layers` / `start_layer` / `end_layer` are gone from
   `ModelRunner`). `make_kv_pool(runner, *, size, layer_info, req_to_token_pool)`
   now takes them explicitly, defaulting to the runner attributes so the 0.5.9
   call site still works. The vortex pool needs `req_to_token_pool` because
   `Context.create` sizes `max_new_tokens_per_batch` from `req_to_token_pool.size`.
9. **`ForwardMode.is_draft_extend` was removed** (only `is_draft_extend_v2`
   remains). vortex asserts on it in the flashinfer / trtllm backends →
   `compat.is_draft_extend(forward_mode)`.
10. **`extend_prefix_lens` is int64 under the new prefill cuda graph.** 0.5.16
    captures a *prefill* graph (new), and
    `prefill_cuda_graph_runner.py` allocates its static `extend_prefix_lens`
    buffer as `torch.int64`, while vortex's `sglang_plan_prefill` kernel reads
    `cached_seq_lens` / `input_seq_lens` as `int32` → `RuntimeError: expected
    scalar type Int but found Long`. Both backends now coerce with
    `.to(torch.int32)` at the call site.
11. **`ForwardBatch.token_to_kv_pool` was removed.** The pool is now reached via
    `forward_context.get_token_to_kv_pool()`, which reads
    `get_attn_backend().token_to_kv_pool`. 19 call sites across the 6 backend
    modules went through `compat.token_to_kv_pool(forward_batch)`, and every
    backend `__init__` now calls `compat.bind_kv_pool(self, model_runner)` so
    upstream's own accessor resolves for vortex backends too (this is what
    `enable_fused_set_kv_buffer` uses to find the pool — so the opt-out in
    §"fused-KV-store hazard" depends on it).

12. **vortex called the removed graph ABI on a wrapped upstream backend.** The
    mirror image of §7: the MLA backends wrap a stock `TritonAttnBackend`
    (`self._dense`) for the dense layers and forwarded
    `init_forward_metadata_{capture,replay}_cuda_graph` to it — which 0.5.16's
    backend no longer has (`AttributeError: 'TritonAttnBackend' object has no
    attribute ...`). `compat.dense_{capture,replay}_cuda_graph(dense, ...)` now
    call whichever ABI the wrapped object implements, synthesizing the
    ForwardBatch-like view for the 0.5.16 `_out_graph` form.
13. **`TritonAttnBackend.qo_indptr` widened int32 → int64** (`kv_indptr` stayed
    int32). vortex forwards the dense backend's indptrs straight into
    flashinfer's prefill `plan()`, which requires int32, so the buffer was
    *reinterpreted* — producing nonsense offsets and
    `PrefillSplitQOKVIndptr ... qo_indptr[3]0 - qo_indptr[2]2483 should be
    non-negative`. Note the reported values don't match the real tensor, which is
    the tell-tale of a dtype reinterpretation rather than a bad batch.
    `mla_prefill.plan` now coerces both indptrs to int32.
14. **MLA extend was dispatched to the weight-absorbed MQA path.**
    `DeepseekV2AttentionMLA.dispatch_attn_forward_method` selects the extend
    implementation from `AttentionBackendRegistry`, keyed by backend name.
    `cuda_mla` isn't an upstream name, so it fell through to the `triton`
    handler, which returns `MHA` **only when the batch has no cached prefix** and
    otherwise `MLA`. Under MLA that hands the backend the *fused latent* K (576 =
    `kv_lora_rank 512 + qk_rope 64`) instead of per-head K/V →
    `RuntimeError: shape '[-1, 20, 256]' is invalid for input of size 9404928` as
    soon as the radix cache produces a prefix hit. vortex reconstructs prefix K/V
    from the latent itself (`mla_prefill._reconstruct_prefix_kv`), so MHA is
    correct for every extend batch. `integration._register_mla_forward_method()`
    registers a `cuda_mla` / `cuda_mla_profile` handler returning MHA for extend
    and MLA for decode. The registry is a plain public dict, so this needs **no**
    edit to the vendored sglang.

    This also means the MLA path was only ever exercised prefix-free before;
    the prefix branch is newly covered by this validation.

## Validation results (sglang 0.5.16, 8x B200, RULER validation_4k, block=32 topk=29)

`examples/ruler/sweep_mha.sh` — Qwen3-4B, 9 flows x 2 indexer backends, **18/18 pass**:

| flow | flashinfer | trtllm |
|---|---|---|
| block_sparse_attention | 100.0% | 100.0% |
| gqa_block_sparse_attention | 100.0% | 100.0% |
| gqa_quest_sparse_attention | 100.0% | 100.0% |
| lserve_sparse_attention | 100.0% | 100.0% |
| lserve_centroid_sparse_attention | 100.0% | 100.0% |
| masked_quest_sparse_attention | 100.0% | 100.0% |
| centered_block_sparse_attention | 100.0% | 100.0% |
| running_avg_block_sparse | 100.0% | 100.0% |
| venergy_gated_centroid | 98.0% | 99.0% |

`examples/ruler/sweep_mla.sh` — GLM-4.7-Flash on `cuda_mla`, **2/2 pass**:

| flow | cuda_mla |
|---|---|
| rope_aware_block_sparse_mla | 98.0% |
| lserve_centroid_mla | 100.0% |

All 20 runs clear the 0.85 RULER gate. Prefill + decode cuda graphs enabled
throughout (the prefill graph is new in 0.5.16).

## Build / install gotchas

- **sglang 0.5.16 needs a Rust toolchain** it did not before: it added two PyO3
  extensions (`sglang.srt.grpc._core`, `sglang.srt.multimodal._core`) and
  `setuptools-rust>=1.10` to build-requires, so `pip install -e python` fails
  with `error: can't find Rust compiler`. Neither extension is used by vortex,
  and upstream gates them: **`SGLANG_BUILD_RUST_EXTS=none`** skips both.
- **`huggingface_hub==0.36.2`** in vortex's `pyproject.toml` made the resolve
  impossible (transformers 5.12.1 needs `>=1.5,<2`). Relaxed to `>=1.5,<2`.
- The toolkit image defaults to **python 3.10**; vortex needs `>=3.12`
  (`python3.12` is present alongside it).

## The fused-KV-store (RoPE) hazard in 0.5.16

`srt/models/utils.py::enable_fused_set_kv_buffer` in 0.5.16:

```python
pool = get_token_to_kv_pool()          # global, not forward_batch
return (
    _is_cuda and pool.dtype == torch.bfloat16
    and not isinstance(pool, SWAKVPool)
    and not is_prefill_context_parallel_enabled()
    and getattr(forward_batch, "dcp_kv_mask", None) is None
) or (
    _is_hip and ...                     # NO dtype / pool guard at all
)
```

The `supports_fused_set_kv_buffer` opt-out vortex added in 0.5.9 is **gone**.
The fused kernel (invoked from inside fused RoPE, `create_fused_set_kv_buffer_arg`)
writes K/V straight into `get_{key,value}_buffer(layer_id).view(num_tokens, -1)`
assuming upstream's token-major layout. `VortexCachePool` is
**block-interleaved**, so with a bf16 KV cache on CUDA the fused path silently
writes K/V to the wrong addresses → decode reads garbage (≈0% accuracy) or a
CUDA illegal-memory-access. The HIP branch is worse: it has no guard, so it
fires for *any* dtype.

Re-applying the `supports_fused_set_kv_buffer` opt-out is therefore
**mandatory**, and it must gate **both** the CUDA and HIP branches. Affected
model files that call the fused path: `gpt_oss.py`, `gemma4_causal.py`, and
any other `create_fused_set_kv_buffer_arg` caller.

## Dependency set (install as ONE resolve, never one-by-one)

0.5.16 `python/pyproject.toml` pins a much newer stack than 0.5.9:

| pkg | 0.5.9 | 0.5.16 |
|---|---|---|
| torch | 2.9.1 (cu128) | **2.11.0** |
| transformers | 4.57.1 | **5.12.1** |
| flashinfer_python | 0.6.3 | **0.6.14 [cu13]** |
| kernel pkg | sgl-kernel | **sglang-kernel==0.4.5** |
| cuda-python | — | **>=13.0** (CUDA 13) |

Install via a single `pip install -e third_party/sglang/v0.5.16/sglang/python`
so pip resolves sglang + sglang-kernel + torch + flashinfer together.

Consequences for this repo:
- `pyproject.toml` pins `transformers==4.57.1` — **conflicts**, must move to 5.x.
- The `vortex_glm` / `install_vortex_glm.sh` split existed only because GLM
  needs transformers>=5. Under 0.5.16 the default env *is* transformers 5, so
  the split collapses into one env.
- CUDA 13 wheels: the container image must have a CUDA 13-capable driver.
