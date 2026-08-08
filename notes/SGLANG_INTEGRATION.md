# vortex_torch ↔ sglang integration (v0.6 design)

**Branch:** `v0.6`  ·  **sglang:** vendored `third_party/sglang/v0.5.9`  ·  **status:** validated (100% RULER on GLM‑4.7‑Flash, `cuda_mla`)

> **Historical (sglang 0.5.9).** The vendored 0.5.9 tree was removed on the
> `v0.9` branch; the integration now targets `third_party/sglang/v0.5.16`.
> The hook *design* below still holds, but several hooks moved file — see
> [sglang_0516_bump.md](sglang_0516_bump.md) for the current surface.

## 1. Problem

vortex_torch plugs sparse attention into sglang's decode loop. Historically the
integration lived as **scattered edits inside the vendored sglang source** — config
knobs, attention‑backend branches, model‑runner init, KV‑pool selection, etc.
Those edits are spread across upstream files, so adopting a new sglang release
(e.g. 0.5.12) means re‑deriving and re‑applying a fuzzy patch by hand, and any
upstream refactor of those files silently breaks the integration.

**Goal:** keep sglang as close to upstream as possible. Move everything that
*can* live in vortex_torch out of sglang; make the residue that genuinely can't
a tiny, clearly‑marked, easily‑portable set of hooks.

## 2. Audit — the actual integration surface (ground truth)

Diffing the vendored tree against a pristine `sglang==0.5.9` shows the surface was
smaller than it felt: **7 edited source files, zero added files** (every vortex
backend / KV pool / kernel already lives under `vortex_torch/engine/sgl/`). The
edits fell into three kinds:

| kind | files | nature |
|---|---|---|
| **A. Externalizable logic** | `attention_registry.py` | backend selection — pure dispatch, replicable from outside via the public registry |
| **B. Mid‑`__init__` logic** | `model_runner.py`, `model_runner_kv_cache_mixin.py` | runs inside `ModelRunner.initialize`; can't be cleanly monkey‑patched (timing), but can delegate to vortex via a one‑liner |
| **C. Already‑good hooks / config / bugfix** | `server_args.py`, `models/utils.py`, `disaggregation/decode.py`, `input_buffers.py` | config that must be in‑source; two duck‑typed no‑op hooks; one upstream bug fix |

Two facts decided how far "external" can go:

1. **sglang spawns its scheduler worker** (`mp.set_start_method("spawn")`). A
   spawned process re‑imports modules fresh, so monkey‑patches applied in the
   parent **do not** reach the worker. Whatever wires the backends must run *in
   the worker*, before the backend is selected.
2. **`ServerArgs` is a dataclass** consumed as `ServerArgs(**kwargs)` (by
   `Engine`) and **pickled across the spawn boundary**. Its `vortex_*` fields
   therefore must be *real in‑source dataclass fields* — they can't be injected
   by a runtime monkey‑patch without fragile dataclass surgery, and a parent‑only
   patch wouldn't survive pickling into the worker.

## 3. Design

A **single module, `vortex_torch/engine/sgl/integration.py`,** owns the entire
runtime interface to sglang. Plain `import vortex_torch` stays lightweight; the
module is exposed lazily (PEP‑562 `__getattr__` in `vortex_torch/__init__.py`),
so it (and sglang) are only pulled in when a hook actually fires.

### 3.1 What got fully externalized → **0 sglang edits**

**Attention‑backend selection.** sglang's registry is a plain public dict
(`ATTENTION_BACKENDS[name] = creator`). `integration.integrate()` captures the
upstream creators and installs flag‑aware shims:

- `flashinfer` / `triton` / `trtllm_mla` → wrapped: if `enable_vortex_sparsity`
  (and the MLA/non‑MLA condition matches) return the vortex backend, **else call
  the original creator unchanged**.
- `cuda_mla` → newly registered (the hand‑written CUDA block‑table MLA decode).

`integrate()` is idempotent and is invoked from `build_sparse_flow` (below),
which runs **inside the spawned worker** before backend selection — solving the
spawn‑timing problem without an sglang edit. `attention_registry.py` is now
**byte‑identical to pristine 0.5.9.**

### 3.2 What became one‑line hooks (irreducible mid‑`__init__` logic)

Each scattered block collapsed to a single `# [VORTEX HOOK]` call into
`integration.py`:

| site | hook | replaces |
|---|---|---|
| `model_runner.py` (≈L590) | `self.sparse_attention = vortex_torch.integration.build_sparse_flow(self)` | ~24 lines building/initializing the vFlow |
| `kv_cache_mixin.py` (cell‑size) | `cell_size = vortex_torch.integration.kv_cell_size(self, num_layers, kv_size)` | the sparse memory‑estimate formula |
| `kv_cache_mixin.py` (pool) | `self.token_to_kv_pool = vortex_torch.integration.make_kv_pool(self)` | the two `VortexMLACachePool` / `VortexCachePool` branches |

`build_sparse_flow` returns `None` when sparsity is off, and also calls
`integrate()` — so the same line both builds the flow and registers the backends
in the worker.

### 3.3 What stayed in sglang on purpose

- **`server_args.py`** — the ~19 `vortex_*` dataclass fields, their argparse
  entries, the `"cuda_mla"` choice, and the PD‑disaggregation overlap‑off rule.
  In‑source is *required* (spawn pickling + `ServerArgs(**kwargs)`, §2) and is
  the cleanest form anyway: one concentrated, greppable config block.
- **`models/utils.py`** — `enable_fused_set_kv_buffer` honours
  `getattr(pool, "supports_fused_set_kv_buffer", True)`. Duck‑typed, no vortex
  symbols; vortex pools opt out to avoid silent KV corruption.
- **`disaggregation/decode.py`** — `if hasattr(pool, "rebuild_aux"): pool.rebuild_aux(loc)`.
  Duck‑typed, no‑op for non‑vortex pools.
- **`input_buffers.py`** — `req_pool_indices` `int32 → int64`. This is an
  **upstream sglang bug fix**, not a vortex feature; kept as a labelled 1‑line
  carry‑forward (and a candidate to upstream).

The §3.3 items are intentionally left in place: they're either structurally
required (config) or already the ideal pattern (duck‑typed, version‑robust, no
vortex import). A `model_runner.py` line — `self.block_size = server_args.vortex_block_size`
— is also kept (one trivial field read, needed before `initialize()`).

## 4. The integration module API (`vortex_torch/engine/sgl/integration.py`)

```text
integrate() -> bool
    Register vortex backends into ATTENTION_BACKENDS (idempotent, spawn-safe).
    Returns False if sglang isn't importable (CPU-only tooling).

build_sparse_flow(runner) -> flow | None
    Build + initialize runner.sparse_attention; also (re)runs integrate().

make_kv_pool(runner) -> VortexMLACachePool | VortexCachePool
    Construct the vortex KV pool; MLA vs MHA chosen by runner.use_mla_backend.

kv_cell_size(runner, num_layers, kv_size) -> int
    Sparse KV bytes-per-token for the available-memory estimate.
```

The backend shims live in the same module (`_make_flashinfer_shim`,
`_make_trtllm_mla_shim`, `_make_triton_shim`, `_create_cuda_mla_backend`).

## 5. Final footprint (vs pristine sglang 0.5.9)

| file | before (v0.5) | after (v0.6) | what remains |
|---|---|---|---|
| `attention_registry.py` | ~50 lines of vortex branches | **0 (identical to pristine)** | — externalized |
| `model_runner.py` | ~26 lines | ~6 | 1 field read + 1 hook |
| `model_runner_kv_cache_mixin.py` | ~40 lines | ~22 | 2 hooks + 1 guard + 1 fp8 literal |
| `server_args.py` | config block | config block (unchanged) | required in‑source config |
| `models/utils.py` | duck‑typed hook | unchanged | ideal pattern, kept |
| `disaggregation/decode.py` | duck‑typed hook | unchanged | ideal pattern, kept |
| `input_buffers.py` | 1‑line bugfix | unchanged | upstream bug fix |

All non‑trivial **logic** is gone from sglang; what's left is config + thin
hooks + duck‑typed shims + one bug fix. Everything new is one file:
`vortex_torch/engine/sgl/integration.py` (+ a lazy export in `__init__.py`).

## 6. Validation

Ran the full vortex MLA path (the most complex selection) through the refactored
integration:

```
MODEL=zai-org/GLM-4.7-Flash BACKEND=cuda_mla MODULE=lserve_centroid_mla
BLOCK=32 PAGE_SIZE=32 TOPK=61 GREEDY=1 IMPL_BACKEND=triton USE_TENSOR_CORE=1
  → Ruler Accuracy (MLA sparse): 100.00%   (415.9 tok/s, tp=1)
```

Identical to the pre‑refactor result. The shim correctly routed `cuda_mla` (new
registration) and the worker registered backends in time via `build_sparse_flow`.
CPU‑side check: `integrate()` returns True, `cuda_mla` present, and
flashinfer/triton/trtllm_mla resolve to the vortex shims.

## 7. Porting to a new sglang (e.g. 0.5.12) — checklist

The work is now bounded and explicit. To rebase onto sglang `X`:

1. **Drop in pristine sglang `X`.** `attention_registry.py` needs **no change** —
   `integrate()` wires it at runtime (verify the public `ATTENTION_BACKENDS`
   dict and `register_attention_backend` still exist; they're stable API).
2. **Re‑apply the `server_args.py` config block** (19 fields + argparse + the
   `"cuda_mla"` choice + PD overlap rule). One concentrated paste.
3. **Re‑apply 3 one‑line `# [VORTEX HOOK]` calls** in `model_runner.py` (1) and
   `model_runner_kv_cache_mixin.py` (2), plus the `block_size` field read and the
   dense‑MLA `and not enable_vortex_sparsity` fall‑through guard. Grep
   `[VORTEX HOOK]` to find them.
4. **Re‑apply the 2 duck‑typed hooks** (`supports_fused_set_kv_buffer`,
   `rebuild_aux`) and the **`input_buffers` int32→int64** bug fix (if still
   present upstream).
5. Verify the names `integration.py` depends on still exist: `runner.use_mla_backend`,
   `runner.server_args`, `runner.model_config.{kv_lora_rank,qk_rope_head_dim,head_dim}`,
   `get_attention_tp_size`, `runner.num_effective_layers`, `runner.{start,end}_layer`.

If sglang refactors `ModelRunner.initialize` the hook *locations* may move, but
the hook *content* never changes — it's all in `integration.py`.

> ⚠️ Adopting 0.5.12 separately surfaced two unrelated upstream issues (see
> `marks/mla/`): the `flashinfer` MLA backend still emits garbage on GLM
> geometry, and stock 0.5.12 has a `Glm4MoeLiteSparseMoeBlock._shared_expert_tp1`
> regression. Neither is a vortex‑integration concern, but both must be handled
> when actually moving the model onto 0.5.12.

## 8. Files changed on `v0.6`

- **new** `vortex_torch/engine/sgl/integration.py` — the single integration module.
- `vortex_torch/__init__.py` — lazy `integration` export (keeps `import vortex_torch` light).
- `…/attention_registry.py` — reverted to pristine (0 edits).
- `…/model_runner.py` — sparse‑flow block → 1 hook.
- `…/model_runner_kv_cache_mixin.py` — pool + cell‑size blocks → 2 hooks.
- (unchanged, kept by design) `server_args.py`, `models/utils.py`,
  `disaggregation/decode.py`, `input_buffers.py`.

---

# v0.6.1 — Independent `VortexConfig` (server_args fully collapsed)

The last big in‑source surface — the 19 scattered `vortex_*` fields on
`ServerArgs` — is now a single `VortexConfig` object owned by vortex_torch.

## Design
- **`vortex_torch/engine/sgl/config.py` :: `VortexConfig`** — a dataclass holding
  all ~18 hyper‑parameters (defaults mirror the old `ServerArgs.vortex_*`
  defaults exactly).
- **`ServerArgs` keeps one field**, `vortex: Optional[Any] = None`. This is the
  spawn‑safe channel: sglang pickles `ServerArgs` to its worker, and the
  `VortexConfig` rides along. (19 fields + ~90 argparse lines → 1 field + 1
  `--vortex-config` JSON arg.)
- **Backward‑compat `__getattr__` shim on `ServerArgs`** forwards every legacy
  `server_args.vortex_*` read and `enable_vortex_sparsity` to the config object
  (or to legacy defaults when disabled). This is what lets the *many* existing
  read sites — `indexer/context.py` & `cache/context.py` `Context.create`, the 5
  vortex backends, the 2 KV pools, the sglang hooks — keep working **unchanged**.
  The shim is a plain literal map (no `import vortex_torch` in sglang → no
  layering flip) and is in‑source so it survives the spawn/unpickle.
- **Flat‑kwargs adapter** (`install_serverargs_adapter`, eagerly installed at
  `import vortex_torch`) wraps `ServerArgs.__init__` so
  `sgl.Engine(vortex_topk_val=..., enable_vortex_sparsity=True, ...)` keeps
  working — flat kwargs fold into `vortex=VortexConfig(...)`. New/explicit code
  may pass `vortex=VortexConfig(...)`; the CLI accepts `--vortex-config '<json>'`.
- The one **write** site (`VortexTRTLLMBackend` forcing the planner to trtllm)
  now writes the config object (`server_args.vortex.attention_backend = ...`),
  keeping a single source of truth (the shim is read‑only).

Net: `import vortex_torch` stays light (engine sub‑packages made lazy / PEP‑562);
sglang gains the `vortex` field + a ~20‑line `__getattr__` shim and loses 19
fields + ~90 argparse lines.

## A pre‑existing bug fixed along the way
The CPU preflight (`check_engine_config`) was crashing for **every** flow with
`TypeError: 'NoneType' object is not iterable` — `verify.py::_make_indexer_ctx`
builds a `_blank_ctx` (all slots `None`) but never set `query_arg_names`, which
the indexer codegen (`interface.py`) lists into the entry‑point signature.
Confirmed pre‑existing (baseline fails identically with the refactor stashed).
Fixed by setting `ctx.query_arg_names = ["q"]` in `_make_indexer_ctx` (MHA
default, matching `Context.__init__`) + hardening `interface.py` to
`getattr(ctx, "query_arg_names", None) or ["q"]`.

## Validation
- **CPU**: adapter installs; flat kwargs → `VortexConfig`; shim returns correct
  values enabled/disabled; **pickle round‑trip** (spawn safety) preserved.
- **Preflight**: 8/8 base `_flow_algorithms_test` flows compile (was 0/8 due to
  the bug above).
- **End‑to‑end RULER (greedy, tp=1)** — flat `vortex_*` kwargs → adapter →
  spawn → shim → backend, all 100%:

  | model | backend | planner | module | RULER |
  |---|---|---|---|---|
  | GLM‑4.7‑Flash (MLA) | cuda_mla | trtllm | rope_aware_block_sparse_mla | 20/20 |
  | GLM‑4.7‑Flash (MLA) | triton | trtllm | rope_aware_block_sparse_mla | 20/20 |
  | Qwen3‑4B (GQA) | flashinfer | flashinfer | gqa_block_sparse_attention | 20/20 |
  | Qwen3‑4B (GQA) | flashinfer | trtllm | gqa_block_sparse_attention | 20/20 |
  | DeepSeek‑V2‑Lite (MLA) | trtllm_mla | trtllm | rope_aware_block_sparse_mla | 20/20 |

  All five vortex backend code paths validated (VortexCudaMLA, VortexTritonMLA,
  VortexFlashInfer, VortexTRTLLM, VortexTRTLLMMLA). `trtllm_mla` is
  DeepSeek‑geometry‑locked, hence run on DeepSeek‑V2‑Lite rather than GLM/Qwen.

## Files changed (v0.6.1, on top of v0.6)
- **new** `vortex_torch/engine/sgl/config.py` — `VortexConfig` + adapter + accessor.
- `…/server_args.py` — 19 `vortex_*` fields → 1 `vortex` field + `__getattr__`
  shim + legacy‑defaults map; argparse block → single `--vortex-config`.
- `vortex_torch/__init__.py` — eager (light) adapter install.
- `vortex_torch/engine/__init__.py`, `…/engine/sgl/__init__.py` — lazy (PEP‑562)
  re‑exports so the light `config` submodule is reachable without importing `api`.
- `…/attention_backend/trtllm.py` — the one write now targets the config object.
- `vortex_torch/flow/verify.py`, `…/indexer/compiler/interface.py` — preflight
  `query_arg_names` fix.
