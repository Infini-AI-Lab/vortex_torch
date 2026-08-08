# WIP — backend simplification + backend-name UX (resume here)

**Status at reboot:** two in-flight changes, both *uncommitted*. Last pushed
commit on `v0.9` is `be42afb`. Nothing below has been validated on GPU yet.

## Where things stand

| item | state |
|---|---|
| `attention_backend/base.py` (new file) | **written, untested, not wired up** |
| `config.py` sglang-backend auto-default | **written, syntax-checked, unit test not yet run** |
| backends rewired onto the base class | **NOT started** |
| GPU validation of either change | **NOT started** |

`git status` should show exactly: new `attention_backend/base.py`, modified
`engine/sgl/config.py`, new `notes/WIP_backend_simplification.md`.

---

## Task 1 — collapse the duplicated backend skeleton

### The measurement that motivates it (already done, trust it)

All five backends (`flashinfer`, `trtllm`, `trtllm_mla`, `triton_mla`,
`cuda_mla`) share one skeleton. Normalising whitespace and hashing bodies:

* `_compile` — 4 "distinct" versions that are really **2** (MHA / MLA); the
  MHA pair (`flashinfer`/`trtllm`) differs **only in comments**, and the MLA trio
  differs **only by one comment line**. `triton_mla` and `cuda_mla` are
  byte-identical.
* The single substantive difference across all five: the dummy query's inner dim
  — `head_dim` (MHA) vs `kv_cache_dim = kv_lora_rank + qk_rope_head_dim` (MLA).
* `get_cuda_graph_seq_len_fill_value` — 2 versions (MHA returns 1 w/ comment,
  MLA returns 1).
* `init_cuda_graph_state` — 2 versions (MHA no-op `pass`, MLA delegates to
  `self._dense`).
* MLA `init_forward_metadata` / capture / replay — identical across the trio
  **except** `cuda_mla` adds one line, `self._plan(seq_lens)`, after
  `plan_decode`.
* MLA `__init__` — ~25 identical lines of geometry read, differing only in
  per-backend asserts (page_size 32/64 for trtllm_mla; page==block for the other
  two; block_size in {16,32,64} for cuda_mla).
* MLA `forward_extend` — all three are `return self._dense.forward_extend(...)`.

Reproduce with:

```bash
python3 - <<'PY'
import ast, pathlib, hashlib, re
base = pathlib.Path("vortex_torch/engine/sgl/attention_backend")
for fname in ["_compile","get_cuda_graph_seq_len_fill_value","init_cuda_graph_state"]:
    print(f"=== {fname} ===")
    for f in ["flashinfer.py","trtllm.py","trtllm_mla.py","triton_mla.py","cuda_mla.py"]:
        src=(base/f).read_text(); t=ast.parse(src)
        seg=next((ast.get_source_segment(src,n) for n in ast.walk(t)
                  if isinstance(n,ast.FunctionDef) and n.name==fname), None)
        if seg: print(f"  {f:16s} {hashlib.sha1(re.sub(r'\\s+',' ',seg).strip().encode()).hexdigest()[:8]}")
PY
```

### What `base.py` already contains

`VortexBackendBase(*attention_backend_base())`
  - `indexer_query_dim_attr` class attr (`"head_dim"`), `indexer_query_dim`
    property — the one real MHA/MLA difference.
  - `_compile_indexer(model_runner)` — the shared trace-on-zero-sized-dummies +
    `compile_indexer` body (replaces all five `_compile`s).

`VortexMLABackendBase(VortexBackendBase)`
  - `indexer_query_dim_attr = "kv_cache_dim"`.
  - `_init_mla_geometry(model_runner)` — the ~25 shared lines.
  - `_plan_sparse(seq_lens, req_pool_indices)` → `plan_decode(...)` then
    `_after_plan_decode(seq_lens)`.
  - `_after_plan_decode` — **no-op hook**; `cuda_mla` overrides it with
    `self._plan(seq_lens)` (its load-balanced work queue, which needs the
    `sparse_seqlens` that `plan_decode` just wrote).
  - `init_forward_metadata` (dense delegate → `_plan_sparse` on decode/idle, else
    `_init_extend_metadata` hook), `init_cuda_graph_state`,
    `init_forward_metadata_{capture,replay}_cuda_graph`,
    `get_cuda_graph_seq_len_fill_value`, `forward_extend`.
  - `_init_extend_metadata` — no-op hook; **`cuda_mla` must override it** with its
    `self._prefill.plan(...)` block (the `has_prefix` / `MLAPrefill` setup
    currently inline in its `init_forward_metadata` else-branch).

### Remaining work

1. Rewire `triton_mla.py` first (simplest: no `_after_plan_decode`, no
   `_init_extend_metadata`) to validate the base class, then `trtllm_mla.py`,
   then `cuda_mla.py` (needs both hooks), then the MHA pair
   (`flashinfer`/`trtllm` — these use `_compile_indexer` only; they do **not**
   wrap a dense backend, keep their own graph methods).
2. Keep each subclass's asserts in its own `__init__` after
   `_init_mla_geometry(...)`.
3. Delete the now-dead per-backend `_compile` bodies and unused imports
   (`as_vtensor`, `FORMAT`, `MetaData`, `compile_indexer`, `Context`,
   `publish_pools`, `GraphMetadataArgs`, `capture_dense`, `replay_dense` —
   check each file individually).
4. `cuda_mla_profile.py` subclasses `VortexCudaMLABackend`; verify it still works.

---

## Task 2 — the confusing two-knob backend selection

**The complaint:** to use vortex's *trtllm* path the user must pass sglang's
`--attention-backend flashinfer`, which reads like a mistake.

**Why:** two orthogonal knobs.
* `--attention-backend` = which **sglang registry slot** vortex's shim hijacks
  (`_make_mha_shim` is installed on `flashinfer` **and** `trtllm_mha`).
* `--vortex-attention-backend` = which **vortex backend** to build
  (`flashinfer` → `VortexFlashInferBackend`, `trtllm` → `VortexTRTLLMBackend`).

The sglang name is nearly irrelevant for MHA (both slots build the same vortex
backend) — except hybrid-GDN models on Blackwell/sm100, where upstream allows
only `{triton, trtllm_mha, fa4}` and **asserts** on `flashinfer`, so `trtllm_mha`
is the only way in.

**Fix already written** in `vortex_torch/engine/sgl/config.py`:
`_SGLANG_BACKEND_FOR = {"flashinfer": "flashinfer", "trtllm": "flashinfer"}` plus
`_default_sglang_backend(kwargs)`, called from the `install_serverargs_adapter`
wrapper before `_orig_init`. It fills `attention_backend` **only** when the
caller left it unset; an explicit value always wins; MLA is untouched (its
sglang name genuinely selects a different decode kernel).

### Remaining work

1. Run this unit test (was interrupted at reboot; note the `sys.modules` line —
   without it the dataclass decorator fails when loading the module out-of-band):

```bash
python3 - <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("cfg", "vortex_torch/engine/sgl/config.py")
m = importlib.util.module_from_spec(spec); sys.modules["cfg"] = m; spec.loader.exec_module(m)
V = m.VortexConfig
for label, kw in [
    ("vortex trtllm, no sglang name",  dict(vortex=V(attention_backend="trtllm"))),
    ("vortex flashinfer, no name",     dict(vortex=V(attention_backend="flashinfer"))),
    ("explicit trtllm_mha wins",       dict(vortex=V(attention_backend="trtllm"), attention_backend="trtllm_mha")),
    ("vortex off -> untouched",        dict(vortex=None)),
    ("unknown vortex be -> untouched", dict(vortex=V(attention_backend="something_else"))),
]:
    before = kw.get("attention_backend"); m._default_sglang_backend(kw)
    print(f"{label:34s} {before!r:14s} -> {kw.get('attention_backend')!r}")
PY
```
   Expect: first two → `'flashinfer'`; `trtllm_mha` preserved; last two → `None`.

2. **Decide the hybrid default.** Auto-defaulting to `flashinfer` is *wrong for
   hybrid models* (upstream asserts). Options: (a) make
   `_default_sglang_backend` hybrid-aware via
   `compat.full_attention_layer_ids(...)` → pick `trtllm_mha` — but it only has
   `kwargs`, not a `ModelConfig`, so this may belong in `__post_init__` instead;
   (b) leave it and document. **(a) is preferable if reachable.**
3. Consider the same defaulting for the CLI/JSON path
   (`check_engine_config` / `--vortex-config`), and update
   `examples/ruler/run_ruler_mha.py --attn-backend` help text (it still says
   "default: flashinfer" as if the user must choose).
4. Only after Task 1+2 are wired: re-validate (below), then commit + push.

---

## Replication — environment and validation

**Cluster (local host has no GPU).** Job template
`algorithm_scientist/kraken/job.yaml`; full instructions in
`algorithm_scientist/kraken/README.md`. Edit `jobName` (must be unique), then:

```bash
kraken -p green jobs create -i algorithm_scientist/kraken/job.yaml
kraken -p green jobs update-kubeconfig -j <jobName>
kubectl get pods | grep <jobName>
# STOP IT WHEN DONE — it holds a whole 8-GPU node:
kraken -p green jobs stop --job-name <jobName>
```

`/scratch` is FSx and **persists across jobs**: `/scratch/zhuominc/venv-0516`
(the env), `/scratch/zhuominc/hf` (Qwen3-4B, Qwen3.5-4B, GLM-4.7-Flash all
cached). Ship the tree each time (the vendored sglang is not on a remote branch):

```bash
tar czf /tmp/vx.tgz --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
  --exclude='.venv*' --exclude=logs --exclude=third_party/flashinfer \
  --exclude=third_party/flash-attention .
kubectl cp /tmp/vx.tgz <pod>:/scratch/vx.tgz
kubectl exec <pod> -- bash -lc 'rm -rf /scratch/zhuominc/vortex_torch &&
  mkdir -p /scratch/zhuominc/vortex_torch && cd /scratch/zhuominc/vortex_torch &&
  tar xzf /scratch/vx.tgz'
```
`rm -rf` first — a stale `compat.py` would shadow the `compat/` package.

**Gotchas that cost time before:** don't `pkill -f` inside `kubectl exec` (kills
your own session, exit 143); pre-fetch HF models then run with
`HF_HUB_OFFLINE=1` (10s default timeout fails under load), but **unset it to
build a dataset** (`make_task.py` needs the Hub); heredocs with f-strings get
mangled by shell escaping — write the script to a file and `kubectl cp` it.

### Validation gates (all four must pass before commit)

```bash
POD=<jobName>-worker-0
# 1+2. hybrid Qwen3.5-4B, cudagraph ON (the default), 4K and 16K
for len in 4k 16k; do
  kubectl exec $POD -- bash -lc "cd /scratch/zhuominc/vortex_torch &&
    HF_HOME=/scratch/zhuominc/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    /scratch/zhuominc/venv-0516/bin/python examples/ruler/run_ruler_mha.py \
      --model Qwen/Qwen3.5-4B --data examples/ruler/validation_${len}.jsonl \
      --module block_sparse_attention --attn-backend trtllm_mha \
      --indexer-backend trtllm --block 32 --topk 29 --layers-skip ''"
done
# 3+4. homogeneous regression, Qwen3-4B, both indexer backends
for be in flashinfer trtllm; do ... --model Qwen/Qwen3-4B \
  --data examples/ruler/validation_4k.jsonl --indexer-backend $be ...; done
```
Expected (matches `be42afb`): **all four 100/100 = 100.0%**, `cuda_graph=on`,
zero tracebacks. `venergy_gated_centroid` is the only flow that isn't 100%
(98–99%) — don't use it as a canary.

For a broader regression: `examples/ruler/sweep_mha.sh` with
`MODEL=Qwen/Qwen3-4B PY=/scratch/zhuominc/venv-0516/bin/python`. Baseline in
`notes/sglang_0516_bump.md` (18/18: 16 at 100%, venergy 98%/99%).

**Always verify the local tree matches what was validated before committing:**

```bash
for f in <changed files>; do
  r=$(kubectl exec $POD -- md5sum /scratch/zhuominc/vortex_torch/$f | awk '{print $1}')
  l=$(md5sum $f | awk '{print $1}')
  [ "$l" = "$r" ] && echo "MATCH $f" || echo "DIFFER $f"
done
```

---

## Context worth keeping

* **Design rule this work follows:** resolve at init, keep per-step paths free of
  host work. See the "Hot-path audit" section of
  `notes/hybrid_model_support.md` — a `.sum()` in the cuda-graph replay path and
  a per-layer `getattr` probe were both removed for this reason. Don't
  reintroduce either while refactoring.
* **cudagraph is sglang's default and mandatory.** `_out_graph` runs *outside*
  `graph.capture()` (upstream calls it before the captured `run_once`), which is
  why the `.to(torch.int32)` coercions there are safe. The decode graph path uses
  pre-allocated `ctx.metadata` buffers with `use_cuda_graph=True`.
* Background: `notes/sglang_0516_bump.md` (the 0.5.16 bump + the compat layer),
  `notes/hybrid_model_support.md` (hybrid design, results, open items).
* Measured results so far live in `summary_ratio/aime24_qwen3_4b_topk_sweep/`,
  `summary_ratio/aime24_qwen3_5_4b_topk_sweep/`,
  `summary_ratio/aime24_qwen3_5_4b_dense/`.
* **Open (not started):** the AIME24 throughput question for hybrid models is
  now answered — sparse is 1.47–1.67x dense on Qwen3.5-4B, best mean@16 at
  topk=93 (+0.017 vs dense, within ~2.2 sigma). Still untested: centroid/quest
  flows under mrope, `head_dim=256` tile tuning, 32K RULER, image batches
  (explicitly rejected).
