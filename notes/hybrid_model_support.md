# Extending vortex to hybrid (full-attention + linear/RNN) models

Study model: **Qwen/Qwen3.5-4B** (`Qwen3_5ForConditionalGeneration`, `qwen3_5`).

## The model

| property | value |
|---|---|
| text layers | 32 |
| `layer_types` | `linear_attention` × 3, then `full_attention`, repeating (`full_attention_interval: 4`) |
| **full-attention layer ids** | **`[3, 7, 11, 15, 19, 23, 27, 31]`** — 8 of 32 |
| linear layers | GDN (gated delta net): `linear_num_key_heads 16`, `linear_num_value_heads 32`, `linear_key_head_dim 128`, `linear_conv_kernel_dim 4` |
| full-attn geometry | `num_attention_heads 16`, `num_key_value_heads 4` (GQA G=4), `head_dim 256` |
| also | multimodal (`vision_config` present), mrope (`mrope_interleaved`, `partial_rotary_factor 0.25`) |

Vortex applies to the **8 full-attention layers only**. The 24 linear layers keep
a fixed-size recurrent state, so there is no KV to sparsify — and no benefit:
they're already O(1) per token in context length.

That is the whole value proposition and also the whole difficulty: the KV budget
vortex is optimizing is now 1/4 of the layers, so the *relative* win from
sparsifying it shrinks, while every index in the system stops being contiguous.

## What sglang 0.5.16 already gives us (do not rebuild)

Both halves of hybrid execution exist upstream and are **pure composition**,
which is what makes a graceful extension possible:

1. **`HybridLinearAttnBackend`**
   (`layers/attention/hybrid_linear_attn_backend.py`) holds a
   `full_attn_backend` + a `linear_attn_backend` + `full_attn_layers`, and
   dispatches per layer id via `_is_full_attn(layer)`. It forwards
   `init_forward_metadata_*` to both children.
2. **`HybridLinearKVPool`** (`mem_cache/memory_pool.py`) holds a `full_kv_pool` +
   a `mamba_pool`, and — critically — **remaps global layer ids to dense
   full-attention indices** before delegating:
   ```python
   self.full_attention_layer_id_mapping = {id: i for i, id in enumerate(full_attention_layer_ids)}
   def _transfer_full_attention_id(self, layer_id): return self.full_attention_layer_id_mapping[layer_id]
   ```
   It also accepts an injected `full_kv_pool=` (used by the shared-byte-buffer
   path), so a caller can supply its own full-attention pool.

The composition point for the backend is
`attention_registry.attn_backend_wrapper(runner, full_attn_backend)`: sglang
builds the full-attention backend **first** (which is where vortex already
intercepts, via the `flashinfer` / `trtllm` creator shims) and *then* wraps it.

## The actual gap: vortex assumes a contiguous layer span

Vortex's pools index their per-layer cache as `layer_id - start_layer`:

```python
# vortex_torch/engine/sgl/memory_pool.py
def get_key_buffer(self, layer_id): return self.cache[layer_id - self.start_layer]["k"]
for layer_id in range(self.start_layer, self.start_layer + self.layer_num): ...
```

That is correct for a homogeneous model (every layer attends) and wrong for a
hybrid one. With `layer_num=8` and full-attn ids `[3,7,...,31]`:

| global layer | dense index (correct) | `layer_id - start_layer` (vortex today) |
|---|---|---|
| 3 | 0 | 3 → **wrong slot** |
| 7 | 1 | 7 → **wrong slot** |
| 11 | 2 | 11 → **out of range** |
| … | … | … |
| 31 | 7 | 31 → **out of range** |

Two distinct failure modes: silent corruption for ids < `layer_num`, IndexError
above. Nothing else in vortex is layer-coupled — `Context` / the cache and
indexer flows carry **no per-layer state** (verified: no `layer_id` in
`vortex_torch/cache/context.py` or `flow/algorithms.py`), so this is a pool-level
concern, not a flow-level one.

### Second gap: the vortex pool hook preempts the hybrid branch

`kv_cache_configurator._build_token_to_kv_pool` was given the vortex branch
**first**, deliberately, so that vortex+MLA could not be captured by the
dense-MLA branch:

```python
if self.server_args.enable_vortex_sparsity:      # <- ours, placed first
    ... make_kv_pool(...)
elif is_dsv4_model: ...
elif self.use_mla_backend and not self.mambaish_config: ...
else:
    if self.is_hybrid_swa: ...
    elif self.mambaish_config:                   # <- hybrid branch, never reached
        ... _build_hybrid_linear_kv_pool(...)
```

So on a hybrid model today, vortex silently takes over the whole pool, builds a
32-slot (or wrongly-indexed 8-slot) flat cache, allocates **no mamba pool**, and
the linear layers have nowhere to put their state. This must become a
three-way decision, not a two-way one.

## Design: compose, don't special-case

The extension mirrors what upstream already does — vortex supplies the
*full-attention* pool and lets `HybridLinearKVPool` own the composition and the
id remapping:

```
HybridLinearKVPool(
    full_kv_pool = VortexCachePool(layer_num=8, ...),   # <- injected
    mamba_pool   = req_to_token_pool.mamba_pool,        # <- upstream's
    full_attention_layer_ids = [3,7,11,...,31],
)
```

Then the id remapping is upstream's problem (it already solves it), and vortex's
`layer_id - start_layer` arithmetic becomes correct *by construction* because the
pool it receives only ever sees dense indices 0..7.

Concretely, three changes:

1. **`vortex_torch/engine/sgl/compat/` — a `layer_map` helper.** Resolve the
   full-attention layer id list for a runner across releases/architectures
   (`mambaish_config(...).full_attention_layer_ids`, `layer_types`,
   `full_attention_interval`), returning `None` for homogeneous models. One
   place that knows how upstream describes hybridity.
2. **`integration.make_kv_pool` — grow a `full_attention_layer_ids=` argument.**
   When present, size the vortex pool to `len(ids)` and hand it back to be
   wrapped, rather than sizing to all layers. The homogeneous path is unchanged.
3. **No vendored-hook change was needed after all.** The hook keeps its position
   at the head of the chain; `make_kv_pool` itself became hybrid-aware, which is
   better — the branch logic lives in vortex, not in the vendored tree. Two
   facts made this work: the hybrid `req_to_token_pool` (which owns
   `mamba_pool`) is built *before* the KV pool in `_init_pools`, so the hook
   already receives it; and `kv_cell_size` needs nothing either, because
   `DefaultPoolConfigurator` already passes `num_layers = len(full_attention_
   layer_ids)` for mambaish models. The mamba state is sized separately upstream
   (per-request, not per-token), so it correctly stays out of the per-token figure.

The backend side needs **no new vortex code**: `attn_backend_wrapper` already
wraps whatever full-attention backend it is given, and vortex's creator shims
already produce that backend. `layers_skip` keeps working and stays expressed in
**global** layer ids (so `--vortex-layers-skip 3` makes the first full-attn layer
dense), which is the only meaning that stays stable if the interleave changes.

## Known blockers to resolve before an end-to-end run

1. **Vortex's backends `assert not self.is_multimodal`**
   (`flashinfer.py:97`, `trtllm.py:97`). Qwen3.5-4B is
   `Qwen3_5ForConditionalGeneration` with a `vision_config`, so it trips this
   even for pure-text prompts. Needs either a text-only path or the assertion
   narrowed to what actually breaks (mrope / mm token handling), not blanket-
   removed.
2. **mrope** (`mrope_interleaved`, `mrope_section [11,11,10]`,
   `partial_rotary_factor 0.25`). Vortex's indexer scores blocks from post-rope
   K; a partial/interleaved rope changes which channels carry position, so
   centroid- and quest-style flows may need the same treatment MLA needed
   (`rope_aware_*`). `block_sparse_attention` (raw q·k on stored K) is the safe
   first flow.
3. **`head_dim=256`** — larger than the 128 used by every flow validated so far;
   worth checking the indexer's block/tile assumptions.
4. **GDN on Blackwell** restricts the full-attn backend to
   `{triton, trtllm_mha, fa4}` (or `+flashinfer` on sm120). Vortex's non-MLA
   path *requires* the `flashinfer` sglang backend, so on B200 (sm100) this
   assertion in `attn_backend_wrapper` conflicts and must be reconciled.

Blocker 4 is the one that decides feasibility on the current cluster; check it
before writing the pool code.

---

# Implementation (done) — validated on Qwen3.5-4B

## What was built

`compat/hybrid.py` — `full_attention_layer_ids(model_config)` resolves the
full-attention layer set (sglang's `mambaish_config` first, then `layer_types`,
then `full_attention_interval`), returning `None` for homogeneous models so the
non-hybrid path carries no special-casing. `in_span(...)` restricts to a
pipeline-parallel rank's layers.

`integration.make_kv_pool` — now three-way. Homogeneous is unchanged; hybrid
builds the vortex pool over **only** the full-attention layers
(`_make_vortex_pool(..., layer_num=len(ids), start_layer=0)`) and wraps it via
`_wrap_hybrid` in upstream's `HybridLinearKVPool`, injecting it as
`full_kv_pool=` next to `req_to_token_pool.mamba_pool`. Composition and the
global→dense layer-id remapping are therefore upstream's, not reimplemented.

`memory_pool.VortexCachePool` — three ABI gaps that only surface when wrapped:
`get_kv_size_bytes()` (was `NotImplementedError`; the wrapper calls it for its
memory report), a `dcp_kv_mask` parameter (accepted, asserted `None` — vortex's
set_kv kernel has no masked write path), and **`layer_id_override`**, which is
how the wrapper passes the dense index. That last one is the fix that makes
indexing correct: `cache_slot` now comes from the override while `layer_id`
stays global, because `layers_skip` is expressed in global ids.

`compat.vortex_cache(forward_batch, layer_id)` — `get_cache` is vortex's own
accessor and the wrapper does not forward it; this reaches through
`full_kv_pool` + `_transfer_full_attention_id`. Replaced 5 call sites.

`integration._make_mha_shim` — the former `_make_flashinfer_shim`, now installed
on **both** `flashinfer` and `trtllm_mha`. Hybrid-GDN models cannot use the
`flashinfer` name on Blackwell (see blocker 4), so `trtllm_mha` is the route in.

The multimodal assertion was narrowed, not deleted: `is_multimodal` is an
architecture-name lookup, and vortex scores blocks from the K the model already
wrote, so mrope / partial-rotary are the model's business. The blanket
`assert not is_multimodal` became a per-batch check that rejects a batch actually
carrying image tokens — the thing that is genuinely unvalidated.

**No new vortex code was needed on the attention-backend side.** Upstream's
`attn_backend_wrapper` builds the full-attention backend first — which is where
vortex's shim already intercepts — then wraps it in `HybridLinearAttnBackend`,
which dispatches per layer id. Vortex gets the 8 full-attention layers; sglang's
`GDNAttnBackend` keeps the 24 linear ones.

## Bugs found on the way (all pre-existing, none hybrid-specific)

Three `import vortex_torch` + attribute-access sites failed inside the spawned
scheduler worker, where the package is mid-import so its lazy `__getattr__` /
re-exports are not yet bound. Symptom was a bare
`AttributeError: module 'vortex_torch' has no attribute 'integration'` (and
later `'flow'`) with the real cause hidden — and it fired **even with vortex
disabled**, i.e. the 0.5.16 hook was broken for every model, not just hybrid
ones. Fixed by importing the submodule directly in the three vendored hooks,
`build_sparse_flow`, and two backends (`from vortex_torch.utils import
is_hopper`, not from the package root).

`_from_sglang` now tolerates a config `mambaish_config` cannot interrogate
(a bare HF config lacks `linear_attn_registry_result`) and falls through.

## Validation — RULER, Qwen3.5-4B, 8x B200

`block_sparse_attention`, block=page=32, topk=29, `layers_skip=[]`,
`attn_backend=trtllm_mha`, `indexer_backend=trtllm`, cuda graphs on:

| context | dense | vortex sparse | vortex sees |
|---|---|---|---|
| 4K  | 100/100 = 100.0% | 100/100 = 100.0% | 928 tok = 23% of ctx |
| 16K | 100/100 = 100.0% | 100/100 = 100.0% | 928 tok = **6% of ctx** |

Sparsity is genuinely binding (topk 29 x 32 = 928 tokens, well under both
contexts), so the 16K result is the load-bearing one: retrieval survives with 6%
of the context visible to the full-attention layers.

Structure was verified separately (`STRUCTURE_OK`), because RULER passing would
*also* be consistent with vortex being silently bypassed:
`full_attention_layer_ids == [3,7,...,31]`, 8 layers allocated instead of 32
(**4x KV over-allocation avoided** — sizing to 32 OOMs a 178 GiB B200 on a 4B
model, which is how the bug was first caught), and global→dense mapping
`{3:0, 7:1, ..., 31:7}` where naive `layer_id - start_layer` is wrong for all 8.

## Not yet established

* **No throughput number.** RULER is a correctness gate. The interesting question
  for hybrid models is whether sparsifying 1/4 of the layers pays for the indexer
  — worth an AIME24-style run before any efficiency claim.
* **Only `block_sparse_attention`.** Centroid/quest-style flows may interact with
  mrope's partial rotary; untested here.
* **head_dim=256** is larger than the 128 every previously-validated flow used.
  It works, but the indexer's tile choices were not tuned for it.
* **Text-only.** Image batches are explicitly rejected, not supported.
* 4K/16K only; 32K data exists and would be the better stress of the 6% regime.
