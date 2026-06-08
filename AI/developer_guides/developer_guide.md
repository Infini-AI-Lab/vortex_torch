# vortex_torch — Developer guide

Audience: framework developers modifying `vortex_torch` itself (adding
ops, changing the compiler, tuning codegen, wiring up a new runtime).
If you're writing a sparse-attention flow, you want the user tutorials
instead — start at [`overview.md`](overview.md).

This guide walks the stack top-down: what each layer owns, how layers
talk to each other, what conventions they must preserve, and the
canonical places to add new functionality. Read it once to get the
mental model, then use it as a reference.

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Repository layout](#2-repository-layout)
3. [Core types](#3-core-types)
4. [Modes: profile vs execute](#4-modes-profile-vs-execute)
5. [Ops and their dispatch tables](#5-ops-and-their-dispatch-tables)
6. [Graph construction at profile time](#6-graph-construction-at-profile-time)
7. [The compilation pipeline](#7-the-compilation-pipeline)
8. [Codegen — from subgraphs to Triton kernels](#8-codegen--from-subgraphs-to-triton-kernels)
9. [The per-kernel template (indexer + cache)](#9-the-per-kernel-template-indexer--cache)
10. [Format addressing math](#10-format-addressing-math)
11. [Dtype handling, fp8, clamp-before-cast](#11-dtype-handling-fp8-clamp-before-cast)
12. [Interface emission](#12-interface-emission)
13. [Runtime integration with sglang](#13-runtime-integration-with-sglang)
14. [The planner and `winfo_*` arrays](#14-the-planner-and-winfo_-arrays)
15. [Verification harness](#15-verification-harness)
16. [Extending the framework](#16-extending-the-framework)
17. [Conventions and gotchas](#17-conventions-and-gotchas)
18. [Reading a generated kernel](#18-reading-a-generated-kernel)
19. [Debugging the compiler](#19-debugging-the-compiler)
20. [Worked example — adding a new op end-to-end](#20-worked-example--adding-a-new-op-end-to-end)

---

## 1. Architecture

`vortex_torch` is a JIT-compiled sparse-attention framework. A user
writes a `vFlow` subclass with three methods; the framework lowers
that description into two fused Triton modules (one per "side") that
plug into sglang's decode loop:

```
user writes: vFlow(
  create_cache, forward_cache, forward_indexer
)
               │
               ▼
     ┌─────────────────────────┐
     │  profile pass           │
     │  (builds op DAG on ctx) │
     └──────────┬──────────────┘
                │
                ▼
     ┌─────────────────────────┐
     │  graph construction     │
     │  (DCE, W-fusion, topo)  │
     └──────────┬──────────────┘
                │
                ▼
     ┌─────────────────────────┐
     │  codegen (triton_impl)  │
     │  writes .py to disk     │
     └──────────┬──────────────┘
                │
                ▼
     ┌─────────────────────────┐
     │  importlib loads the    │
     │  class; runtime calls   │
     │  .forward(...)          │
     └─────────────────────────┘
```

**Two parallel pipelines** — cache-side and indexer-side — each get
their own `Context`, their own compiler, and their own generated file.
They share only the physical paged buffer (`cache['k']`, `cache['v']`,
and user-declared summary fields); they never share Python state.

**Two runtime phases** on the cache side:

1. Token write → low-level K/V copy kernel (bf16 or fp8-quantised).
2. Block completion → the *compiled cache `forward(...)`* runs on the
   newly-filled block and updates summary fields.

**One runtime phase** on the indexer side, per layer per decode step:
the *compiled indexer `forward(...)`* builds a `[S, 1, 1]` score,
topK selects pages, flashinfer attends only to those pages.

---

## 2. Repository layout

```
vortex_torch/
├── abs/                        # shared types (vTensor, vOp, ContextBase)
│   ├── tensor.py               # vTensor + FORMAT enum
│   ├── op.py                   # vOp + Schedule
│   └── context_base.py         # ContextBase, vortex_dtype slot
├── utils.py                    # Mode, dtype resolution, INDENT, indent_block
├── flow/
│   ├── flow.py                 # vFlow abstract class + initialize()
│   ├── registry.py             # @register decorator + global class map
│   ├── loader.py               # build_vflow(), module-from-path importer
│   ├── algorithms.py           # six reference flows
│   └── verify.py               # verify_flow_compilable + CLI
├── indexer/                    # indexer-side ops + compiler + context
│   ├── context.py              # indexer Context (winfo_*, indptr tensors, dtype)
│   ├── matmul.py, reduce.py, elementwise*.py, scan.py, output_func.py,
│   │   save_load.py, transpose.py, mask.py   # op classes
│   └── compiler/
│       ├── graph.py            # OpDAG, subgraph partitioning
│       ├── compile.py          # compile() → writes .py, loads class
│       ├── interface.py        # entry-class emission + intermediate alloc
│       ├── impl.py             # AVAILABLE_IMPL_BACKENDS
│       └── triton_impl/
│           ├── kernel_gen.py   # the per-workload kernel template
│           ├── register.py     # (op_class, schedule) → codegen function
│           ├── dtype_cast.py   # load_cast_expr / store_cast_expr / is_fp8
│           └── <one file per op>  # generate_<op>_impl(...) functions
├── cache/                      # cache-side ops + compiler + context
│   ├── context.py              # cache Context (page_size, block_size, dtype)
│   ├── <op classes>            # same naming as indexer; NO save_load/topk/scan
│   ├── triton_kernels/         # pre-baked non-codegen kernels (K/V copy, fp8)
│   └── compiler/               # mirror of indexer/compiler
├── engine/
│   └── sgl.py                  # get_engine_from_json + check_engine_config
└── third_party/sglang/v0.5.9/sglang/...  # patched sglang with the VTX backend
```

Rule of thumb: **every op has one class and one codegen function**;
they live in mirrored paths on the two sides. Indexer has more ops
than cache (topK, Softmax, Normalize, Conv1d, Save, Load, Transpose,
`Reduce(dim=0)` are indexer-only).

---

## 3. Core types

### 3.1 `FORMAT` enum (abs/tensor.py)

```python
class FORMAT(Enum):
    BATCHED = 0   # one row per (batch, kv_head)
    RAGGED  = 1   # one row per position in the flat per-(batch, head) listing
    PAGED   = 2   # one row per absolute block_id in the paged KV pool
```

The format determines *how* the compiler addresses a tensor's leading
axis at the Triton level. Inner dims (`D_0`, `D_1`) are always literal
and shared across formats. See [`tensor.md`](tensor.md) for the full
addressing math.

| format | leading-axis semantics | where it shows up |
|---|---|---|
| BATCHED | `(batch, kv_head)` linearised | `q` on the indexer side; per-(batch, head) summaries from `Reduce(dim=0)` |
| PAGED   | absolute `block_id` in the cache pool | every cache field |
| RAGGED  | position in the per-(batch, head) block listing | indexer intermediates, the final score fed to topK |

### 3.2 `vTensor` (abs/tensor.py)

Pure metadata. No storage, no `torch.Tensor` inheritance, no op
support. Fields:

| field | type | purpose |
|---|---|---|
| `shape` | `Tuple[int, int, int]` | logical rank-3 shape |
| `dtype` | `torch.dtype` | dtype of the eventual backing buffer |
| `device` | `str` or `torch.device` | same |
| `_format` | `FORMAT` | addressing rule |
| `tensor_id` | `int` | graph-level identity; indexes into `ctx.tensor_list` |

Constructor:

```python
vTensor(
    shape: Tuple[int, int, int],
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    _format: FORMAT,
    tensor_id: int,
)
```

A helper `as_vtensor(real_torch_tensor, _format, tensor_id)` converts
an existing `torch.Tensor` into a vTensor by copying shape/dtype/device
— used when seeding caller-provided tensors (`q`, `o`, cache fields)
during `_initialize_graph`.

`vTensor.dim()` returns `len(shape)`, so `assert x.dim() == 3`
continues to work in `profile()` validators without any special-case
handling.

vTensors are constructed in `op.profile()` methods and in the runtime
backend's `_initialize_graph` (for `q`, `o`, and cache fields). The
compiler traverses them via `tensor_id`; nothing ever reads a
vTensor's shape except at profile time.

**Key property**: vTensors don't actually hold memory. The real
`torch.empty(...)` calls happen in the generated `<name>_CompiledFunc.__init__`.
What the compiler sees at profile time are pure shape/dtype descriptors
that costs nothing to allocate.

### 3.2.1 Shape convention `(D_0, D_1)` — why rank-3

Every vTensor is rank-3 `[L, D_0, D_1]`:

- `L` is the leading axis (BATCHED: `b * H_kv + h`, PAGED: `block_id`,
  RAGGED: flat per-(batch, head) position).
- `D_0` / `D_1` are the inner dims.

The compiler's load/store expressions all use the fixed `[:, None]` /
`[None, :]` broadcasting over `D_0` and `D_1`. Flows that want a
different inner rank should flatten into this rank-3 layout — e.g. a
"per-block scalar" becomes `(1, 1)` and a "per-block vector of length
`head_dim`" becomes `(1, head_dim)`.

### 3.3 `vOp` (abs/op.py)

Abstract base for every op class. Each subclass must implement
`profile(...)`; `execute(...)` is optional (codegen-only ops omit it).

Key attributes:

- `schedule`: `Schedule.W` (workload-fused, inlines into the
  per-workload or per-block kernel) or `Schedule.S` (standalone,
  launches its own kernel). Default is `Schedule.S`.
- `output_format`: resolved by `profile()` from the dispatch table,
  then read by `build_subgraph` and later by the codegen.
- `output_buffer`: the `vTensor` this op produces (set in
  `profile()`). Single-output assumption throughout the compiler.

`vOp.__call__` dispatches to `profile` or `execute` based on
`ctx.mode`:

```python
def __call__(self, *args, ctx, **kwargs):
    if ctx.mode == Mode.profile:
        return self.profile(*args, ctx=ctx, **kwargs)
    return self.execute(*args, ctx=ctx, **kwargs)
```

**Critical invariant**: an op instance can be called at most once per
compile (profile phase). Every call to `profile()` mutates `ctx.*_list`
with a new op entry and a new tensor entry. Reusing an instance across
two call sites will double-register it; the fuser + codegen will
produce nonsense. User-facing tutorials say "declare a new instance
per call site"; the same rule applies internally.

### 3.4 `Schedule` enum

```python
class Schedule(Enum):
    W = 0   # "workload-fused" — inlines into the surrounding fused kernel
    S = 1   # "standalone" — launches its own Triton kernel
```

The fuser only merges connected `Schedule.W` ops; every `Schedule.S`
op ends up in its own subgraph.

### 3.5 `ContextBase` (abs/context_base.py)

Shared base for `indexer.Context` and `cache.Context`. Its `__slots__`:

- `name`, `mode`, `_created` — lifecycle tracking.
- `vortex_dtype` — dtype for every auto-allocated intermediate tensor
  (default `torch.bfloat16`).

Methods: `profile()` / `execute()` flip `mode`; `create()` (abstract)
populates the concrete context from a runtime object.

### 3.6 `indexer.Context`

Adds all indexer runtime state:

- **Graph state** (populated during profile): `tensor_list`, `op_list`,
  `output_tensor_to_op_list`, `op_to_input_tensor_list`,
  `op_to_output_tensor_list`, `side_effect_op_ids` (Save ops).
- **Name map**: `tensor_id_to_tensor_name_map` — maps caller-provided
  tensor ids (`q`, `o`, `cache['k']`, …) to the Python expression
  used in the generated code.
- **Layout constants**: `page_size`, `block_size`, `num_blocks_per_page`,
  `max_num_pages`, `max_num_blocks`, `num_pages_per_workload`,
  `workload_chunk_size`, `max_num_blocks_per_request`, `max_bs`,
  `num_kv_heads`, `head_dim`, …
- **Runtime index tensors**: `dense_kv_indices`, `dense_kv_indptr`,
  `sparse_kv_indices`, `sparse_kv_indptr`, `kv_last_page_len`,
  `batch_size`.
- **winfo**: `winfo_q_indices`, `winfo_is_first_workload_per_batch`,
  `winfo_kv_offsets`, `winfo_kv_lens`, `winfo_num_workloads`,
  `winfo_chunk_size`.
- **Codegen state**: `sparse_attention_name` (unique per flow
  instance), `impl_backend` (`"triton"`), `compilation_header_lines`,
  `auxilary_func_def_lines`, `compilation_cache_dir`.

### 3.7 `cache.Context`

Narrower than indexer: no BATCHED-related fields (cache has no
BATCHED format), no `winfo_*`, no `topk_*`. Adds cache-specific
layout: `max_new_tokens_per_batch`, `total_num_pages`,
`total_num_blocks`, `head_num`, …

---

## 4. Modes: profile vs execute

Every `ctx` has `mode ∈ {profile, execute}`. `vOp.__call__` routes to
the matching method. The backend flips modes explicitly:

```python
# in VTXGraphAttnBackend._initialize_graph:
self.ctx.create(self, model_runner)
self.ctx.profile()                                      # enter profile mode
indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)    # builds graph
compiled_cls = vortex_torch.indexer.compiler.compile.compile(self.ctx)
self.compiled_indexer = compiled_cls()
self.ctx.execute()                                      # profile is done; lock it
```

**Profile phase contracts**:

- Inputs are `vTensor`s (metadata only).
- `profile()` validates shapes/dtypes, resolves the output format
  (compiler-generated ops infer it inline from the input formats; a
  small set of custom kernels and conversion ops still consult a
  per-class dispatch table — see §5.1), allocates a new `vTensor` for
  its output, and appends to `ctx.tensor_list` / `ctx.op_list` /
  edges.
- No real compute happens; no torch ops are called.

**Execute phase contracts**:

- Inputs are real `torch.Tensor`s (or the framework-provided cache
  dicts).
- The user never calls `execute()` on individual ops; the runtime
  calls the `*_CompiledFunc.forward(...)` on the fused module, which
  internally launches Triton kernels.
- Most codegen-only ops don't define `execute` — calling it raises
  `NotImplementedError`.

---

## 5. Ops and how their output format is resolved

Every op class lives in `{indexer,cache}/<family>.py` and has:

- an `__init__` with op-specific parameters (e.g. `dim`, `alpha`,
  `beta`);
- a `profile(...)` method that validates the inputs, picks
  `output_format`, constructs a `vTensor` for the output, and
  registers the op into `ctx`.

There are two flavors of format resolution:

- **Compiler-generated ops** — the per-workload kernel handles all
  format combinations uniformly, so the op only needs to pick the
  output format and the surrounding kernel takes care of the rest.
  These ops infer `output_format` inline from a single rule keyed on
  the input format(s); they no longer carry a per-class `_impl_map`.
- **Custom kernels & format-conversion ops** — their codegen
  branches on the tensor formats, so they keep an explicit dispatch
  table (`_impl_map` keyed on input format, or `_supported_formats`
  for the no-output variants like `topK` and `Fill`). Mismatches are
  caught at `profile()` time with an explicit assertion.

### 5.1 Output-format rules

**Indexer side** (mostly compiler-generated):

| op family | output format rule | `_impl_map`? |
|---|---|---|
| unary elementwise (`Relu`, `Silu`, `Sigmoid`, `Abs`, `Add_Mul`, `Log`, `Exp`) | `BATCHED iff x._format == BATCHED, else RAGGED` | no |
| binary elementwise (`Add`, `Multiply`, `Maximum`, …, `Where*`) | `BATCHED iff both inputs are BATCHED, else RAGGED` | no |
| `GeMM` / `GeMV` | `BATCHED iff both inputs are BATCHED, else RAGGED` | no |
| `Reduce(dim ∈ {1, 2})` | `BATCHED iff x._format == BATCHED, else RAGGED` | no |
| `Transpose`, `Reshape`, `MaskSlice`, `Kron` | `BATCHED iff input(s) are all BATCHED, else RAGGED` | no |
| `Reduce(dim == 0)` (custom Schedule.S kernel) | RAGGED → BATCHED | inline asserts `x._format == RAGGED` |
| `Softmax`, `Normalize`, `Conv1d` (Schedule.S) | preserve input format | `{RAGGED: RAGGED}` |
| `topK`, `approxTopK` | writes into a caller-provided `o` | `_supported_formats = {RAGGED}` (no output allocated) |
| `Save` (RAGGED → PAGED) | PAGED | `{RAGGED: PAGED}` |
| `Load` (PAGED → RAGGED) | RAGGED | `{PAGED: RAGGED}` |

Intuition for the BATCHED-iff-input-BATCHED rule: a `BATCHED` tensor
already has its packed-S axis collapsed to one row per
`(batch, kv_head)`. Anything an indexer op derives from BATCHED
inputs stays BATCHED; anything that touches a non-BATCHED partner
re-expands across the packed axis and becomes RAGGED. PAGED is
reserved for `Save` outputs (the only producer of PAGED on the
indexer side).

**Cache side** (compiler-generated except `Fill`):

| op family | output format rule | `_impl_map`? |
|---|---|---|
| unary elementwise | `output._format if output is provided else RAGGED` | no |
| binary elementwise | same | no |
| `GeMM` | same | no |
| `Reduce`, `ReduceInterleave` | same | no |
| `MaskSlice` | same | no |
| `Reshape` | same | no |
| `Fill` (pure producer) | overwrites its input; PAGED-only | `_supported_formats = {PAGED}` |

The rule is simpler on the cache side: the caller either provides an
`output: vTensor` (PAGED to write back into a cache field, RAGGED
for an intermediate) or omits it (auto-allocates a RAGGED
intermediate). The op asserts `output._format ∈ {PAGED, RAGGED}` —
BATCHED has no meaning on the cache side.

### 5.2 Output-format invariants enforced by the compiler

- **Indexer intermediates** are `RAGGED` or `BATCHED` only — never
  `PAGED`. The only PAGED writer on the indexer side is `Save`, and
  the destination is always a caller-provided cache field.
- **BATCHED outputs from `Schedule.W` ops are guarded** at store
  time by `winfo_is_first_workload_per_batch` (see §9.3).
- **Cache intermediates** are `RAGGED` (auto-allocated intermediate)
  or `PAGED` (write-back to a declared field). No BATCHED on the
  cache side.

### 5.3 Schedule assignment

Most ops are `Schedule.W`. `Schedule.S` is currently used by:

- `topK` / `approxTopK` (drive C++/Triton extensions, distinct from
  the fused kernel).
- `Softmax`, `Normalize`, `Conv1d` (scan-style, can't fuse per-workload).
- `Reduce(dim=0)` — the special case where the *same* op class
  dispatches to different codegen based on schedule. `dim ∈ {1, 2}`
  keeps `Schedule.W`; `dim == 0` sets `Schedule.S` in `__init__`.

### 5.4 The `Save` special case

`Save` is a "pure side-effect writer". Its profile intentionally does
**not** update `ctx.output_tensor_to_op_list[target]` — if it did,
any `Load` of the same cache field would become a graph-level
consumer of Save, creating a Load → Save → Load cycle through the
compute chain. Instead, Save appends its op id to
`ctx.side_effect_op_ids`; the DAG builder seeds DFS from that set
separately. See `graph.py:_build_op_dag`.

### 5.5 What `profile()` looks like in practice

Two representative shapes — a compiler-generated op (no `_impl_map`)
and a custom kernel that keeps one.

**Compiler-generated — `Elementwise_Binary` (indexer)**:
```python
self.output_format = (
    FORMAT.BATCHED
    if (x._format == FORMAT.BATCHED and y._format == FORMAT.BATCHED)
    else FORMAT.RAGGED
)
```

**Compiler-generated — cache `Elementwise`**:
```python
if output is None:
    self.output_format = FORMAT.RAGGED         # auto-allocated intermediate
else:
    assert output._format in (FORMAT.PAGED, FORMAT.RAGGED)
    self.output_format = output._format        # honor caller's destination
```

**Custom kernel — `Softmax` (indexer)** still uses a dispatch table
because its standalone codegen reads `t._format`:
```python
_impl_map: Dict[FORMAT, FORMAT] = {FORMAT.RAGGED: FORMAT.RAGGED}
...
assert x_fmt in self._impl_map
self.output_format = self._impl_map[x_fmt]
```

**Format-conversion — `Save` (indexer)**:
```python
_impl_map: Dict[FORMAT, FORMAT] = {FORMAT.RAGGED: FORMAT.PAGED}
...
assert x_fmt in self._impl_map
self.output_format = self._impl_map[x_fmt]
assert o._format == self.output_format
```

### 5.6 What `profile()` looks like — annotated

Canonical unary-ish example (`indexer/reduce.py:Reduce.profile`):

```python
def profile(self, x: vTensor, ctx: Context) -> vTensor:
    prefix = self._prefix()
    assert isinstance(x, vTensor), f"{prefix}x must be vTensor"
    assert x.dim() == 3, f"{prefix}x must be 3D"

    D0, D1 = x.shape[1], x.shape[2]

    if self.dim == 0:
        # Cross-row reduction: RAGGED → BATCHED, Schedule.S (custom kernel).
        assert x._format == FORMAT.RAGGED
        self.output_format = FORMAT.BATCHED
        out_D0, out_D1 = D0, D1
    else:
        # Per-row reduction (Schedule.W, compiler-generated):
        # BATCHED stays BATCHED, everything else becomes RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )
        out_D0 = 1 if self.dim == 1 else D0
        out_D1 = 1 if self.dim == 2 else D1

    # Pure-metadata vTensor — leading dim 0 as a placeholder; the compiler
    # picks the real allocation size from ctx.
    self.output_buffer = vTensor(
        shape=(0, out_D0, out_D1),
        dtype=ctx.vortex_dtype,
        device=x.device,
        _format=self.output_format,
        tensor_id=len(ctx.tensor_list),
    )

    # Register into the graph.
    ctx.tensor_list.append(self.output_buffer)
    ctx.output_tensor_to_op_list.append(len(ctx.op_list))
    ctx.op_list.append(self)
    ctx.op_to_input_tensor_list.append([x.tensor_id])
    ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])

    return self.output_buffer
```

Notes:

- **Leading dim 0 in the vTensor shape is a convention**, not a bug:
  the real leading dim depends on `ctx.max_num_blocks` /
  `ctx.max_bs * ctx.num_kv_heads` and is only known at codegen time,
  so `profile()` leaves it as 0. The allocator in `interface.py`
  substitutes the real value.
- **`dtype=ctx.vortex_dtype`**: every intermediate uses the shared
  dtype knob, never `x.dtype`. This is how a flow can be compiled at
  bf16 or fp8 without changing a single line of the flow's code.
- **`self.output_format` and `self.output_buffer` are cached on the
  op instance.** Don't reuse an op across call sites — these fields
  would get silently overwritten.

### 5.7 Save / Load / Fill: caller-provided-destination pattern

Ops that write into a caller-provided cache field use a different
registration pattern: instead of allocating a new output vTensor,
they **reuse the caller's tensor_id** and override producer-ship.

Example (`indexer/save_load.py:Save.profile` — the side-effect variant):

```python
def profile(self, x: vTensor, o: vTensor, ctx: Context) -> vTensor:
    ...
    save_op_id = len(ctx.op_list)
    ctx.op_list.append(self)
    ctx.op_to_input_tensor_list.append([x.tensor_id])
    ctx.op_to_output_tensor_list.append([o.tensor_id])   # reuse caller's tid
    ctx.side_effect_op_ids.append(save_op_id)            # but don't claim producer
    return o
```

Compare with `Fill.profile`, which *does* claim producer-ship:

```python
def profile(self, x: vTensor, loc, ctx: Context) -> vTensor:
    ...
    # Register as a pure producer of x (same pattern as Load).
    ctx.output_tensor_to_op_list[x.tensor_id] = len(ctx.op_list)
    ctx.op_list.append(self)
    ctx.op_to_input_tensor_list.append([])     # Fill has no graph-level inputs
    ctx.op_to_output_tensor_list.append([x.tensor_id])
    return x
```

Fill is a "pure producer of its input tensor" — the input's old
contents are irrelevant (Fill overwrites with a constant), so it
claims producer-ship without conflict. Save can't do this because
Load reads the same tensor later in the same graph.

---

## 6. Graph construction at profile time

`profile()` on every op class pushes the same bookkeeping into ctx:

```python
ctx.tensor_list.append(new_vtensor)                        # the output
ctx.output_tensor_to_op_list.append(len(ctx.op_list))      # this op is the producer
ctx.op_list.append(self)                                   # register the op
ctx.op_to_input_tensor_list.append([in_1.tensor_id, ...])
ctx.op_to_output_tensor_list.append([new_vtensor.tensor_id])
```

Exceptions:

- **Caller-provided outputs (e.g. cache-side `output=cache['k']`)**:
  the op uses the target's `tensor_id` as its output and **overrides**
  `output_tensor_to_op_list[tid]` to claim producer-ship.
- **Save**: skips the producer override entirely (see §5.4).
- **Fill**: marks itself as the producer of the field it fills (same
  pattern as a caller-provided output).

After `forward_indexer` / `forward_cache` runs in profile mode, `ctx`
is a complete op DAG described by four parallel lists.

### 6.1 Input tensors (not produced by any op)

`q`, `o`, and every `cache['<name>']` are seeded into `ctx.tensor_list`
*before* the user's `forward_*` runs, with `output_tensor_to_op_list[tid]`
set to `None` (no producer). The runtime backend
(`VTXGraphAttnBackend._initialize_graph` for indexer,
`VTXGraphCachePool._initialize_graph` for cache) does this seeding
and also populates `ctx.tensor_id_to_tensor_name_map` so the codegen
knows how to refer to each tensor by name in the generated Python.

---

## 7. The compilation pipeline

Driven by `compile(ctx)` in `indexer/compiler/compile.py` (the cache
side mirrors it):

```python
def compile(ctx):
    full_graph, sub_graphs = contruct_graph(ctx)
    file_path, cls_name = generate_interface(full_graph, sub_graphs, ctx)
    # importlib loads the file, returns the CompiledFunc class
```

`contruct_graph` is the meat. In `graph.py` it's broken into three
phases:

### 7.1 Phase 1 — DCE + op DAG

`_build_op_dag(op_list, output_tensor_to_op_list,
op_to_input_tensor_list, op_to_output_tensor_list,
final_output_tensor_ids, side_effect_op_ids)`:

- Starts from `final_output_tensor_ids` (always includes `tensor_id=1`,
  i.e. `o`) plus any Save op ids.
- Reverse-DFS through producers. Every op reached is "alive".
- Returns an `OpDAG` plus the set of reachable op ids.

`final_output_tensor_ids` also includes "orphan sinks" — tensors
produced by some op but never consumed by another op. These keep
side-effect ops alive even if their output is unused.

### 7.2 Phase 2 — `Schedule.W` fusion

`_fuse_w_ops(op_dag, op_list)` uses a union-find to greedily merge
`Schedule.W` op pairs whose fusion would not create a cycle in the
subgraph DAG. The `can_merge_bidir` check is the O(N) hot path.
`Schedule.S` ops never merge with anything.

### 7.3 Phase 3 — subgraph assembly + topological sort

Each surviving union-find root becomes one subgraph. Inputs are any
tensor whose producer is `None` or lives in a different subgraph;
outputs are any tensor in `final_output_tensor_ids` or consumed from
outside.

Subgraphs are topo-sorted over the subgraph DAG (not the op DAG);
the sort must be cycle-free. If it isn't, the compiler raises
`"Cycle detected in subgraph DAG"` — almost always because a
side-effect op (Save) accidentally became a consumer edge (§5.4 is
the fix for that).

`_build_local_graph(...)` materialises each subgraph: remaps global
tensor/op ids to local ones, stores `global_input_tensor_ids` and
`global_output_tensor_ids` so the caller can thread inputs/outputs
at the interface layer.

### 7.4 Data structures used during compilation

```python
class OpDAG:
    nodes: List[int]                             # op_ids in topo order
    successors: Dict[int, Set[int]]              # consumer → producer edges
    predecessors: Dict[int, Set[int]]            # producer → consumer edges

class UnionFind:
    parent: Dict[int, int]
    rank: Dict[int, int]
    find(x): ...   # with path compression
    union(x, y): ...

class Graph:
    # Materialised, local-id-space view of a subgraph.
    tensor_list: List[vTensor]
    op_list: List[vOp]
    output_tensor_to_op_list: List[Optional[int]]
    op_to_input_tensor_list: List[List[int]]
    op_to_output_tensor_list: List[int]
    input_tensor_ids: List[int]                  # local ids that are subgraph inputs
    output_tensor_ids: List[int]                 # local ids that are subgraph outputs
    global_input_tensor_ids: List[int]           # same, but in the full-graph id space
    global_output_tensor_ids: List[int]
    schedule: Schedule                           # W or S (S subgraphs have exactly one op)
```

The `input_tensor_ids` / `output_tensor_ids` are used by `kernel_gen.py`
to emit per-format load / store paths. The `global_*_tensor_ids` are
used by `interface.py:generate_subgraph_entry_point` to pass the right
caller-provided tensors.

### 7.5 `final_output_tensor_ids` — the DFS roots

Computed at the top of `contruct_graph`:

```python
final_output_tensor_ids = sorted({
    1,                                                   # attention output `o` (always tid 1)
    *(tid for tid, producer in enumerate(output_tensor_to_op_list)
      if producer is not None and not tensor_to_consumers.get(tid)),  # orphan sinks
    *save_target_tids,                                   # cache fields written by Save
})
```

Three sources of "must survive DCE":

1. `tid == 1` — the user's `o` output, always alive.
2. Orphan sinks — a tensor that's produced but has no consumer
   anywhere. This keeps ops like an unused `GeMM(BATCHED, BATCHED)`
   alive (if its only purpose is side-effect write to a BATCHED
   slot).
3. Save targets — cache fields that Save writes via the
   `side_effect_op_ids` path.

### 7.6 Why topo sort over *subgraphs*

A `Schedule.W` subgraph bundles multiple ops. The topo sort's unit is
the subgraph, not the op. If A is in subgraph `α` and B in subgraph
`β`, and A feeds B, then `α → β` in the subgraph DAG regardless of
whether other ops in `β` also feed ops in `α`.

`_fuse_w_ops` is conservative here — it only merges pairs whose fusion
*doesn't* introduce a new cycle in the subgraph DAG. The
`can_merge_bidir` check is the expensive part; merges that would
create a cycle are skipped.

The final topo order is stored in `op_to_subgraph_id` and used to
write the subgraphs out in the correct execution order in
`interface.py`.

---

## 8. Codegen — from subgraphs to Triton kernels

`generate_interface(full_graph, sub_graphs, ctx)` (in `interface.py`)
emits one `.py` file. Its structure:

```python
<header lines>                       # triton imports, etc.

<aux function definitions>           # standalone kernels (topK, Softmax, reduce_dim0, …)

<one subgraph func per subgraph>:
    @triton.jit
    def <name>_subgraph_i_kernel(...): ...     # only for Schedule.W subgraphs

    def <name>_subgraph_i_impl(...): ...        # launcher (W) or direct op call (S)

    def <name>_subgraph_i_interface(...): ...   # thin wrapper called by the entry class

class <name>_CompiledFunc:
    def __init__(self):
        self.tensor_X = torch.empty(...)        # every intermediate
    def forward(self, q, o, cache, ctx):
        <name>_subgraph_0_interface(...)
        <name>_subgraph_1_interface(...)
        ...
```

The compile module imports the file and returns the `CompiledFunc`
class. Runtime calls `forward(q, o, cache, ctx)` and the kernels
launch.

### 8.1 Per-op codegen dispatch

`triton_impl/register.py` maps `(op_class, Schedule) → codegen
function`. The codegen function takes `(graph, op_id, ctx)` and
returns a string. For `Schedule.W` ops this is just the *compute
fragment* that goes inside the surrounding fused kernel; for
`Schedule.S` ops it returns the impl-function *body*.

Example (`triton_impl/reduce.py`):

```python
# Schedule.W: inline tl.sum/tl.max/...
def generate_reduce_impl(graph, op_id, ctx) -> str:
    ...
    if op.reduce_type == ReduceType.Mean:
        return f"tensor_{out}_block = tl.sum(tensor_{in}_block, ...) * {1.0/t_i.shape[op.dim]}"

# Schedule.S: a standalone kernel that walks dense_kv_indptr
def generate_reduce_dim0_impl(graph, op_id, ctx) -> str:
    kernel_name = f"{ctx.sparse_attention_name}_reduce_dim0_{rt}_kernel"
    ctx.auxilary_func_def_lines.append(_reduce_dim0_kernel_body(...))
    return <launcher lines>
```

Registering a new op is three lines in `register.py` plus a new
codegen function file.

### 8.2 `dtype_cast.py`

Shared between per-op generators that need to bitcast fp8, cast to
fp32, or clamp-before-narrow. Two functions:

- `load_cast_expr(load_expr, t)`: decodes fp8 → fp8eX → fp32;
  otherwise `load_expr.to(tl.float32)`.
- `store_cast_expr(block_expr, dtype)`: fp32 → target dtype; for fp8
  targets emits
  `tl.minimum(tl.maximum(..., -MAX), MAX).to(tl.float8eX).to(tl.uint8, bitcast=True)`
  with `MAX = 57344` (e5m2) or `448` (e4m3fn).

### 8.3 Full register.py layout (indexer)

```python
IMPL_REGISTRY = {
    # Schedule.W — fused into the per-workload kernel
    (GeMM,               Schedule.W): generate_gemm_impl,
    (Reduce,             Schedule.W): generate_reduce_impl,
    (Elementwise_Binary, Schedule.W): generate_elementwise_binary_impl,
    (Elementwise,        Schedule.W): generate_elementwise_impl,
    (Transpose,          Schedule.W): generate_transpose_impl,
    (Save,               Schedule.W): generate_save_impl,
    (Load,               Schedule.W): generate_load_impl,
    (MaskSlice,          Schedule.W): generate_mask_slice_impl,

    # Schedule.S — standalone; each op launches its own kernel
    (topK,      Schedule.S): generate_topk_impl,
    (Softmax,   Schedule.S): generate_softmax_impl,
    (Normalize, Schedule.S): generate_normalize_impl,
    (Conv1d,    Schedule.S): generate_conv1d_impl,
    (Reduce,    Schedule.S): generate_reduce_dim0_impl,
}

def get_impl_func(op):
    schedule = op.schedule
    for (op_type, sched), impl_func in IMPL_REGISTRY.items():
        if sched == schedule and issubclass(op.__class__, op_type):
            return impl_func
    raise NotImplementedError(...)
```

Key points:

- **Same `Reduce` class, two entries.** Dispatch picks the right
  codegen based on `op.schedule`, which itself was set by
  `Reduce.__init__` from `dim` (`dim=0 → S`, otherwise `W`). This is
  how the same user-facing class `Mean(dim=1)` vs `Mean(dim=0)`
  emits fundamentally different kernels.
- **`issubclass` check.** `get_impl_func` uses `issubclass(op.__class__,
  op_type)` so `Mean`, `Max`, `Min`, `L2Norm`, `Sum` (all subclasses
  of `Reduce`) dispatch to the same codegen function.
- **No default fallback.** If no `(class, schedule)` matches, the
  compile fails with `NotImplementedError`. You always have to add a
  new entry when adding a new op.

### 8.4 What codegen functions return

The return type depends on the op's schedule:

| schedule | return value | used as |
|---|---|---|
| W | a compute fragment string like `tensor_3_block = tl.sum(tensor_2_block, axis=1)` | inserted into `generate_computation_str` inside the fused kernel |
| S | the impl-function *body* (launcher code) | inserted inside `<name>_subgraph_i_impl`, the function that launches the standalone kernel |

For S ops, the codegen must also **append the `@triton.jit` kernel
definition to `ctx.auxilary_func_def_lines`** so it lives at the top
of the generated file, before any function that calls it. This is
the canonical pattern:

```python
def generate_softmax_impl(graph, op_id, ctx) -> str:
    ...
    func_def_lines = [
        "@triton.jit",
        "def softmax_kernel(...):",
        "    ...",
    ]
    ctx.auxilary_func_def_lines.extend(func_def_lines)   # kernel def

    impl_lines = [
        "    softmax_kernel[(eff_batch_size,)](...)",    # launcher
    ]
    return "\n".join(impl_lines)
```

### 8.5 Simple W codegen examples

Most W ops are one-liners. The simplest:

```python
# elementwise.py:generate_elementwise_impl (partial)
def generate_elementwise_impl(graph, op_id, ctx) -> str:
    in_id  = graph.op_to_input_tensor_list[op_id][0]
    out_id = graph.op_to_output_tensor_list[op_id]
    op     = graph.op_list[op_id]
    if op.op_type == ElementwiseOpType.Relu:
        return (
            f"tensor_{out_id}_block = tl.where("
            f"tensor_{in_id}_block >= {op.alpha}, "
            f"tensor_{in_id}_block, {op.beta})"
        )
    if op.op_type == ElementwiseOpType.Exp:
        return f"tensor_{out_id}_block = tl.exp({op.beta} * tensor_{in_id}_block + {op.alpha})"
    ...
```

The surrounding kernel has already loaded `tensor_{in_id}_block` as
fp32 (via `_load_cast_expr`) and will store `tensor_{out_id}_block`
via `_store_cast_expr`. The op just emits the compute.

Binary ops look similar:

```python
# elementwise_binary.py (partial)
if op.op_type == ElementwiseBinaryOpType.Multiply:
    return f"tensor_{out_id}_block = tensor_{x_id}_block * tensor_{y_id}_block"
if op.op_type == ElementwiseBinaryOpType.Add:
    return f"tensor_{out_id}_block = {op.alpha} * tensor_{x_id}_block + {op.beta} * tensor_{y_id}_block"
if op.op_type == ElementwiseBinaryOpType.Maximum:
    return f"tensor_{out_id}_block = tl.maximum(tensor_{x_id}_block, tensor_{y_id}_block)"
```

Every binary op here is format-agnostic: the surrounding kernel has
loaded each input's block correctly based on its `_format`, and the
store path will write the output's block correctly based on *its*
`_format`. The op's only job is to emit the arithmetic.

### 8.6 A full Schedule.S example — `reduce_dim0`

Longer example in `indexer/compiler/triton_impl/reduce.py:
generate_reduce_dim0_impl`. Walkthrough:

1. Validate the op is `Reduce(dim=0)` with RAGGED→BATCHED formats.
2. Choose a unique kernel name: `f"{ctx.sparse_attention_name}_reduce_dim0_{rt_name}_kernel"`.
3. Template the kernel body based on `op.reduce_type` (Sum, Mean,
   Max, Min, L2Norm):
   - Init expression: `tl.zeros((x_D0, x_D1))`, `tl.full((...), -1e30)`, etc.
   - Accumulate expression: `acc += tl.sum(slab, axis=0)`,
     `acc = tl.maximum(acc, tl.max(slab, axis=0))`, etc.
   - Finalize expression: `""`, `acc = acc / num_pages.to(tl.float32)`,
     `acc = tl.sqrt(acc)`.
4. Use `_store_cast_expr` to format the final store (handles fp8
   clamp + bitcast).
5. Append the fully-formed `@triton.jit` function to
   `ctx.auxilary_func_def_lines`.
6. Return the launcher body:
   ```python
   eff_batch_size = ctx.batch_size * ctx.num_kv_heads
   <kernel_name>[(eff_batch_size,)](
       tensor_{in_id},
       tensor_{out_id},
       ctx.dense_kv_indptr,
       ctx.block_reserved_bos,
       ctx.block_reserved_eos,
       tensor_{in_id}.shape[-2],
       tensor_{in_id}.shape[-1],
       num_warps=4,
       num_stages=1,
   )
   ```

The kernel itself walks `ctx.dense_kv_indptr[pid:pid+1]` to find the
`(batch, head)` slice it's responsible for, loops over blocks in
that slice (chunked at `BLOCK_P=512` for Triton register pressure),
accumulates in fp32, writes one `[D_0, D_1]` tile to the BATCHED
output.

---

## 9. The per-kernel template (indexer + cache)

### 9.1 Indexer `@triton.jit` kernel (kernel_gen.py)

One kernel per `Schedule.W` subgraph. Signature has five winfo-ish
scalars plus `{tensor_X_ptr, tensor_X_dim0}` pairs for every input
and output. Body:

```python
@triton.jit
def <name>_subgraph_i_kernel(
    indices,                             # dense_kv_indices
    winfo_x_indices,                     # BATCHED row index per workload
    [winfo_is_first_workload_per_batch,] # only if subgraph has a BATCHED output
    winfo_y_offsets,                     # ragged_idx base per workload
    winfo_y_lens,                        # valid block count per workload
    winfo_num_workloads,
    <tensor ptrs + dim0s>,
):
    pid = tl.program_id(0)
    # partition winfo into contiguous slices for this pid
    workload_ptr = tl.arange(0, workload_chunk_size)

    # per-tensor initialisation (dim ptrs, zero blocks for BATCHED)
    <generate_initialization_str>

    for i in range(start, end):
        # if num_pages_per_workload > 1: compute page_valid mask
        <prepare_workload_str>
        # load every input tensor into tensor_X_block (fp32)
        <generate_load_tensor_str>
        # emit every op's compute expression
        <generate_computation_str>
        # store every output tensor (ragged, paged, or batched)
        <generate_store_tensor_str>
```

### 9.2 Cache `@triton.jit` kernel (cache kernel_gen.py)

One kernel per cache subgraph. Simpler grid — one program per
`(token_id, head_id)`:

```python
@triton.jit
def <name>_subgraph_i_kernel(loc, <tensor ptrs>, NUM_KV_HEAD, PAGE_SIZE,
                              BLOCK_SIZE, NUM_BLOCKS_PER_PAGE):
    token_id = tl.program_id(0)
    head_id  = tl.program_id(1)
    token_position = tl.load(loc + token_id)
    if (token_position + 1) % BLOCK_SIZE != 0:
        return   # only fire at block completion
    page_id  = (token_position // PAGE_SIZE) * NUM_KV_HEAD + head_id
    block_id = page_id * NUM_BLOCKS_PER_PAGE + (token_position % PAGE_SIZE) // BLOCK_SIZE

    # dim ptrs for each input/output
    <initialisation>
    # block-addressed load for each PAGED input
    <_block_load_lines>
    # inline compute
    <generate_computation_str>
    # block-addressed store for each PAGED output
    <_block_store_lines>
```

Note the absence of loops, winfo, or workload chunking. The cache
kernel does exactly one block per program invocation and bails out
otherwise.

### 9.3 BATCHED store gate

A `Schedule.W` indexer kernel running over multiple workloads can
revisit the same `(batch, head)` slot. For BATCHED outputs this
would double-write the same scalar. The compiler's
`generate_store_tensor_str` emits a single load of
`winfo_is_first_workload_per_batch[i]` (hoisted once outside the
per-tensor loop) and gates each BATCHED store:

```python
_is_first_workload = tl.load(winfo_is_first_workload_per_batch + i)
...
if _is_first_workload != 0:
    tl.store(tensor_3_block_ptr, ...)
```

The planner sets `winfo_is_first_workload_per_batch[j] = 1` iff `j` is
the first workload of its `(batch, head)`. See §14.

### 9.4 Dedup for read-and-written cache fields

When a tensor appears in *both* `sub_graph.input_tensor_ids` and
`output_tensor_ids` (typical Load/Save pair on the same cache field),
the kernel signature / launcher argument list / interface wrapper all
dedupe: the tensor is declared exactly once, on the output side. See
`interface.py:generate_subgraph_func` and
`kernel_gen.py:generate_triton_impl` (both carry an
`output_set = set(...)` check).

### 9.5 Per-format load paths (indexer)

`generate_load_tensor_str` emits three independent blocks: one for
BATCHED inputs, one for PAGED inputs, one for RAGGED inputs. Order
doesn't matter — each block is self-contained and references only
the `winfo_*` scalars set up at the top of the loop.

**BATCHED load** (one-row-per-workload, broadcast across pages):
```python
new_batch_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
tensor_X_ptr_row_start = new_batch_idx_i32 * {t.shape[1]}
tensor_X_block_ptr = tl.make_block_ptr(
    base=tensor_X_ptr,
    shape=(tensor_X_dim0 * {t.shape[1]}, {t.shape[2]}),
    strides=({t.shape[2]}, 1),
    offsets=(tensor_X_ptr_row_start, 0),
    block_shape=({t.shape[1]}, {t.shape[2]}),
    order=(1, 0),
)
tensor_X_block = tl.reshape(
    tl.load(tensor_X_block_ptr, boundary_check=(0, 1),
            padding_option="zero", cache_modifier=".ca"),
    (1, {t.shape[1]}, {t.shape[2]})
).to(tl.float32)
```

The leading `1` on the reshape is explicit — BATCHED tiles broadcast
against the workload axis.

**PAGED load, single-page workload** (one page per workload):
```python
ragged_idx_i32 = tl.load(winfo_y_offsets + i).to(tl.int32)
page_idx_i32   = tl.load(indices + ragged_idx_i32).to(tl.int32)
tensor_X_ptr_row_start = page_idx_i32 * {t.shape[1]}
tensor_X_block_ptr = tl.make_block_ptr(
    base=tensor_X_ptr,
    shape=(tensor_X_dim0 * {t.shape[1]}, {t.shape[2]}),
    strides=({t.shape[2]}, 1),
    offsets=(tensor_X_ptr_row_start, 0),
    block_shape=({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}),
    order=(1, 0),
)
tensor_X_block = tl.reshape(
    tl.load(tensor_X_block_ptr, ..., cache_modifier=".cv"),
    ({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]})
).to(tl.float32)
```

The `cache_modifier=".cv"` tells the L1 to treat these loads as
streaming (they won't be reused); BATCHED loads use `.ca` because q is
reused across pages.

**PAGED load, multi-page workload** (when `num_pages_per_workload > 1`):
uses `page_indices_i32 = tl.load(indices + ragged_idx_i32 +
page_idx_i32_ptr * num_blocks_per_page)` to get a vector of page ids
for the workload, then scatters the loads over `page_valid` to handle
short tails.

**RAGGED load** (one-row-per-position in the per-(batch, head) listing):
```python
ragged_idx_i32 = tl.load(winfo_y_offsets + i).to(tl.int32)
tensor_X_ptr_row_start = ragged_idx_i32 * {t.shape[1]}
tensor_X_block_ptr = tl.make_block_ptr(
    base=tensor_X_ptr,
    shape=(tensor_X_dim0 * {t.shape[1]}, {t.shape[2]}),
    strides=({t.shape[2]}, 1),
    offsets=(tensor_X_ptr_row_start, 0),
    block_shape=({t.shape[1] * ctx.workload_chunk_size}, {t.shape[2]}),
    order=(1, 0),
)
tensor_X_block = tl.reshape(
    tl.load(tensor_X_block_ptr, ...),
    ({ctx.workload_chunk_size}, {t.shape[1]}, {t.shape[2]})
).to(tl.float32)
```

Same leading-dim semantics as PAGED but without the
`indices[ragged_idx]` lookup — RAGGED rows are indexed directly by
`ragged_idx`.

### 9.6 Per-format store paths (indexer)

Mirror of the loads. `generate_store_tensor_str` emits three blocks
for PAGED / RAGGED / BATCHED outputs.

**RAGGED store** is the most common (it's what every intermediate
score lands in):
```python
_len = tl.load(winfo_y_lens + i)
valid = workload_ptr < _len
tensor_X_block_ptr = (tensor_X_ptr
    + ragged_idx_i32 * {t.shape[1] * t.shape[2]}
    + workload_ptr[:,None,None] * {t.shape[1] * t.shape[2]}
    + tensor_X_dim1_ptr[None,:,None] * {t.shape[2]}
    + tensor_X_dim2_ptr[None,None,:]
)
tl.store(tensor_X_block_ptr,
         {_store_cast_expr("tensor_X_block", t.dtype)},
         mask=valid[:, None, None])
```

The `valid` mask handles short-tail workloads where `winfo_kv_lens[i]
< workload_chunk_size`.

**BATCHED store** with the is-first gate:
```python
_is_first_workload = tl.load(winfo_is_first_workload_per_batch + i)
...
tensor_X_block_ptr = (tensor_X_ptr
    + new_batch_idx_i32 * {t.shape[1] * t.shape[2]}
    + tensor_X_dim1_ptr[None, :, None] * {t.shape[2]}
    + tensor_X_dim2_ptr[None, None, :]
)
if _is_first_workload != 0:
    tl.store(tensor_X_block_ptr,
             {_store_cast_expr("tensor_X_block", t.dtype)})
```

The load of `_is_first_workload` is hoisted once out of the per-tensor
loop so multiple BATCHED outputs share one `tl.load`.

**PAGED store** is rarer on the indexer side (only `Save` emits it) —
uses the same `make_block_ptr` pattern as PAGED load but for write.

### 9.7 Cache-side per-format paths

Much simpler because the grid is `(token_id, head_id)` and each
program owns one block. The load uses `block_id * (D_0 * D_1)` as
the base address directly — no `indices[ragged_idx]` lookup, no
workload chunking:

```python
tensor_X_off = block_id * {t.shape[1] * t.shape[2]}
tensor_X_ptr_2d = (tensor_X_ptr + tensor_X_off
    + tensor_X_dim1_ptr[:, None] * {t.shape[2]}
    + tensor_X_dim2_ptr[None, :])
tensor_X_block = {_load_cast_expr("tl.load(tensor_X_ptr_2d)", t)}
```

Stores follow the same pattern. Cache RAGGED uses
`(token_id * NUM_KV_HEAD + head_id) * (D_0 * D_1)` as the base
(per-(token, head) flat layout), which is different from indexer
RAGGED.

---

## 10. Format addressing math

The compiler emits these literal Triton pointer expressions (you don't
write them; the kernel_gen emits them based on each tensor's
`_format`):

```
BATCHED (indexer-only):
  addr = ptr + new_batch_idx_i32 * (D_0 * D_1) + ...
  # new_batch_idx_i32 = winfo_x_indices[i]   (= b * H_kv + h)

PAGED (both sides):
  addr = ptr + block_id * (D_0 * D_1) + ...
  # indexer: block_id = tl.load(indices + ragged_idx_i32)
  # cache:   block_id computed from token_position + head_id inline

RAGGED (both sides, but different semantics):
  indexer:
    addr = ptr + ragged_idx_i32 * (D_0 * D_1)
         + workload_ptr[:, None, None] * (D_0 * D_1)
         + dim1_ptr * D_1 + dim2_ptr
    # ragged_idx_i32 = winfo_kv_offsets[i]  (base row of this workload)
  cache:
    addr = ptr + (token_id * NUM_KV_HEAD + head_id) * (D_0 * D_1)
         + dim1_ptr * D_1 + dim2_ptr
    # different! per-(token, head) row, not per-workload chunk
```

Same formats, different addressing on each side because the program
grids differ:

| side | grid | RAGGED indexing |
|---|---|---|
| indexer | (SM slice, ) | `ragged_idx + workload_ptr` per workload iteration |
| cache   | (token, head) | `(token_id, head_id)` directly |

### 10.1 The dense_kv_indices / indptr duality

`dense_kv_indptr` is a CSR-style array of shape `(max_bs * num_kv_heads
+ 1,)`. `dense_kv_indices` is the flat listing of absolute `block_id`s,
ordered by `(batch, head)` entry.

- `dense_kv_indptr[b*H_kv + h]` = first row in the flat listing for
  `(batch, head)`.
- `dense_kv_indices[ragged_idx]` = the absolute `block_id` at that
  ragged position.

PAGED ↔ RAGGED translation in the kernel is one `tl.load`:

```python
ragged_idx_i32 = tl.load(winfo_y_offsets + i)
page_idx_i32   = tl.load(indices + ragged_idx_i32)
```

`sparse_kv_indptr` / `sparse_kv_indices` are the post-`topK` versions,
written by the C++ topk extension.

---

## 11. Dtype handling, fp8, clamp-before-cast

### 11.1 `vortex_dtype`

Both contexts carry `vortex_dtype`. Every intermediate vTensor is
allocated with this dtype. The engine config key is `vortex_dtype`
(bf16/fp16/fp32/fp8_e5m2/fp8_e4m3fn). Default: `torch.bfloat16`.
Resolver lives in `utils.py:resolve_dtype`.

### 11.2 KV cache dtype

`cache['k']` / `cache['v']` dtype is set by `kv_cache_dtype`
(independent of `vortex_dtype`). The engine allows `"auto"` (→ bf16),
`"fp8_e4m3"`, `"fp8_e5m2"`.

### 11.3 FP8 plumbing

At codegen time, any tensor with dtype ∈ `{float8_e4m3fn, float8_e5m2}`
gets:

- **On load**: `tl.load(ptr).to(tl.float8eX, bitcast=True).to(tl.float32)`.
  The wrapper views the tensor as `uint8` before passing it to the
  kernel (`tensor_X = tensor_X.view(torch.uint8)`); the kernel
  bitcasts back.
- **On store**: clamp to the dtype's representable range, then cast:
  ```
  tl.minimum(tl.maximum(block, -MAX), MAX).to(tl.float8eX).to(tl.uint8, bitcast=True)
  ```
  with `MAX = 57344` (e5m2) or `MAX = 448` (e4m3fn).

The `.view(torch.uint8)` rebinding is injected in the generated `_impl`
function (see `kernel_gen.py:generate_triton_impl`'s `fp8_rebind_lines`
logic). This is crucial — PyTorch can't launch a Triton kernel with
fp8 pointers directly.

---

## 12. Interface emission

`interface.py` (both sides) generates the entry-class file. The main
moving pieces:

### 12.1 Intermediate allocation

For every subgraph's `output_tensor_ids`, the compiler emits
`self.tensor_X = torch.empty(...)` in `__init__` unless the tensor is
"caller-provided" (already has a Python name in
`tensor_id_to_tensor_name_map`). The leading dim depends on the
format:

| format | leading dim (indexer) | leading dim (cache) |
|---|---|---|
| RAGGED | `ctx.max_num_blocks` | `ctx.max_new_tokens_per_batch * ctx.head_num` |
| BATCHED | `ctx.max_bs * ctx.num_kv_heads` | n/a |
| PAGED | not allowed as intermediate | not allowed as intermediate |

The allocator test is `if t.tensor_id in tensor_id_to_tensor_name_map:
continue`. Caller-provided tensors (q, o, cache fields) have names
seeded by the runtime before the profile pass; user flow compute
doesn't add names, so its intermediate outputs take the allocation
path.

### 12.2 Subgraph interface wrappers

`generate_subgraph_func` produces two functions per subgraph:
`<name>_subgraph_i_impl` (the kernel launcher or Schedule.S op body)
and `<name>_subgraph_i_interface` (a thin wrapper the entry-class
`forward` calls with the actual tensor arguments).

The wrappers use `sub_graph.global_input_tensor_ids` /
`global_output_tensor_ids` to know which arguments to thread. Dedup
(§9.4) happens here too.

### 12.3 Entry-class `forward` body

Sequence of `<name>_subgraph_i_interface(...)` calls in subgraph-topo
order, each line annotated with the global tensor id for readability.

The file is written to `ctx.compilation_cache_dir` (default
`~/.vortex_compilation_cache/`). The compile function then imports
the file via `importlib.util.spec_from_file_location`, fetches the
class, and returns it.

---

## 13. Runtime integration with sglang

Patched sglang ships in `third_party/sglang/v0.5.9/sglang/`. Two glue files:

### 13.1 `VTXGraphAttnBackend` (sglang/srt/layers/attention/vtx_graph_backend.py)

The attention backend that replaces flashinfer's default. Its
responsibilities:

- Allocate `kv_indptr_*` / `kv_indices_*` / `qo_indptr` / `batch_table`
  buffers at init time.
- Build `indexer.Context`, run the profile+compile in
  `_initialize_graph` with dummy tensors, store the compiled class.
- `init_forward_metadata`: call the C++ planner, plan flashinfer's
  two decode wrappers (`[0]` = dense, `[1]` = sparse).
- `forward_extend`: prefill — runs two flashinfer wrappers (ragged +
  paged) and merges their partial attention outputs.
- `forward_decode`: for sparse layers, call
  `self.compiled_indexer.forward(q, o=sparse_kv_indices_buf,
  cache=..., ctx=self.ctx)`, then flashinfer sparse wrapper. For
  skipped layers, skip the indexer and run the dense wrapper.

### 13.2 `VTXGraphCachePool` (sglang/srt/mem_cache/vtx_graph_memory_pool.py)

The memory pool for the paged KV. Responsibilities:

- Allocate `self.cache = [{name: torch.zeros(...)} for _ in
  range(layer_num)]` — one dict per layer, one tensor per declared
  cache field.
- Build `cache.Context`, profile + compile `forward_cache` in
  `_initialize_graph`.
- Pre-bake a K/V copy launcher (`set_kv_buffer_*`) for the incoming
  token dtype.
- `set_kv_buffer(layer, loc, cache_k, cache_v, ...)`: run the K/V
  copy (always), then — if the layer isn't skipped — run
  `self.compiled_cache.forward(self.cache[layer_id], loc, ctx)`.

The two compiled modules never share state; their only communication
channel is the `self.cache[layer_id]` dict of paged buffers.

---

## 14. The planner and `winfo_*` arrays

`indexer/planner_sglang.py` and `indexer/utils_sglang.py` wrap a C++
CUDA extension (`sglang_plan_decode_v2`) that populates the runtime
metadata arrays. At decode time, the backend calls:

```python
self.plan_decode(cached_seq_lens, req_to_token, req_indices, ctx)
```

which dispatches into the C++ kernel and fills:

| array | shape | meaning |
|---|---|---|
| `winfo_q_indices[j]` | `(max_num_workloads,)` | `b * H_kv + h` — BATCHED row for workload `j` |
| `winfo_is_first_workload_per_batch[j]` | `(max_num_workloads,) uint8` | 1 iff workload `j` is the first of its `(batch, head)` |
| `winfo_kv_offsets[j]` | `(max_num_workloads,)` | ragged_idx of the first block of workload `j` |
| `winfo_kv_lens[j]` | `(max_num_workloads,)` | number of valid blocks (`≤ workload_chunk_size`) |
| `winfo_num_workloads` | `(1,)` | total workloads scheduled |
| `winfo_chunk_size` | `(1,)` | `workload_chunk_size` (for runtime bookkeeping) |
| `dense_kv_indptr` | `(max_bs * H_kv + 1,)` | CSR-style per-(batch, head) offset into `dense_kv_indices` |
| `dense_kv_indices` | `(max_num_blocks,)` | flat list of absolute `block_id`s, ordered by (batch, head) entry |

The C++ planner enforces the `is_first_workload_per_batch` bit
(`(j == start) ? 1 : 0`) while expanding each `(batch, head)`'s
segment into multiple workloads.

No cache-side planner. The cache kernel's grid is driven directly by
`loc.shape[0]` (the number of freshly-written tokens) and
`ctx.head_num`.

---

## 15. Verification harness

Two tools worth knowing:

### 15.1 `verify_flow_compilable(flow, ...)`

CPU-only compile sweep over a grid of `(G, D, block_size,
page_block_ratio, num_pages_per_workload)`. Builds fake
indexer/cache contexts with small CPU-backed metadata tensors, runs
profile + compile for every combination, reports pass/fail. No Triton
JIT (the kernels never run).

Default grid: `G ∈ (1, 2, 4, 8, 16)`, `D ∈ (32, 64, 128)`,
`block_size ∈ (4, 8, 16, 32, 64, 128)`,
`page_block_ratios ∈ (1, 2, 4, 8, 16)`,
`pages_per_workload ∈ (1, 2)`. 450 configs per flow.

CLI: `python -m vortex_torch.flow.verify <flow_name> [--vortex-dtype DT] [...]`.

### 15.2 `check_engine_config(config_path)`

Pre-flight validation for engine JSONs (`engine/sgl.py`). Nine
checks:

1. JSON file exists and parses.
2. `vortex_block_size` is a positive power of 2.
3. `vortex_workload_chunk_size` is a positive power of 2.
4. `vortex_block_reserved_bos/_eos` are ints ≥ 1.
5. `vortex_layers_skip` is empty or a list of ints.
6. `vortex_module_path` resolves to an existing file.
7. That file declares `@register("<vortex_module_name>")`.
8. Compiling succeeds for the actual model's GQA shapes. The
   resolver reads `config.json` from the JSON's `model_path` (or
   `engine.sgl.MODEL_PATH` if absent): treats the value as a local
   directory if it exists, otherwise downloads via
   `huggingface_hub.hf_hub_download`. It then derives
   `num_kv_heads = config.num_key_value_heads`,
   `G = num_attention_heads // num_key_value_heads`, and
   `head_dim = config.head_dim` (falling back to
   `hidden_size // num_attention_heads`). The sweep runs at exactly
   those shapes with `pages_per_workload_values=(16, 32)`.
9. If the flow uses `Save(...)` in the indexer, the JSON sets
   `"disable_radix_cache": true` (otherwise sglang's prefix cache
   would share per-request persistent state across requests with
   matching prompt prefixes).

Raises `EngineConfigError` with a focused message on first failure.

### 15.3 `_flow_algorithms_test` — standard end-to-end RULER suite

The two tools above (`verify_flow_compilable`, `check_engine_config`)
only prove that the compiler accepts a flow. They do **not** prove the
emitted kernel produces correct decode output. Whenever you touch any
of the following, run the standard suite below:

  * the indexer / cache compiler (`indexer/compiler/`,
    `cache/compiler/`) — codegen, graph construction, scheduling, the
    `Schedule.W` / `Schedule.S` split,
  * the planner (`planner_sglang.py`) or any `winfo_*` field,
  * the sglang integration (`engine/sgl/attention_backend/*`),
  * any op's `output_shape` / `output_format` / `profile()` logic,
  * any backend-conditional snippet in `indexer/compiler/backend.py`
    (the `IndexerBackend` traits used by both `flashinfer` and
    `trtllm`).

The suite covers **eight reference flows** under
`submissions/_flow_algorithms_test/`, one per `@register` class in
`vortex_torch/flow/algorithms.py`:

| Algorithm | Notable surface it exercises |
|---|---|
| `block_sparse_attention`          | Baseline `topK` flow; minimum compile path. |
| `gqa_block_sparse_attention`      | GQA aggregation (head-group mean). |
| `gqa_quest_sparse_attention`      | GQA + quest-style max/min pages, multi-tensor PAGED loads. |
| `lserve_sparse_attention`         | Static + dynamic budget interaction. |
| `masked_quest_sparse_attention`   | `MaskSlice` on top of quest scoring. |
| `centered_block_sparse_attention` | Per-(batch, kv_head) cross-row `Reduce(dim=0)` — the **only** `Schedule.S` reduce in the suite, hits `custom_impl/reduce_dim0.py`. |
| `running_avg_block_sparse`        | `Save`+`Load` round-trip — forces `disable_radix_cache: true`. |
| `venergy_gated_centroid`          | Centroid-style flow with elementwise gating. |

Why each flow lives in its own file (not just one
`@register(...)` per class in `flow/algorithms.py`):
`engine/sgl._check_disable_radix_cache` does a **text scan** of
`vortex_module_path`. If all eight pointed at
`vortex_torch/flow/algorithms.py` (which contains the `Save(...)` site
inside `RunningAvgBlockSparse`), the scan would force
`disable_radix_cache: true` on every algorithm — incorrect for the
seven that don't use Save/Load, and a measurable throughput hit. Each
algorithm therefore lives in its own file under
`submissions/_flow_algorithms_test/<algo>.py` with a `_sub`-suffixed
`@register` name (sglang auto-imports `flow.algorithms` at startup, so
re-registering the same name would collide).

#### How to run

```bash
# 1. Cheap CPU pre-flight first (catches schema + compile errors).
for cfg in submissions/_flow_algorithms_test/*.json; do
  python -c "from vortex_torch.engine.sgl import check_engine_config; \
             check_engine_config('$cfg')"
done

# 2. RULER on examples/ruler/validation_4k.jsonl. Wave size ≤ 4 — 5+ sglang
#    engines booting on the same host hit startup contention that
#    inflates the per-run e2e and skews throughput. A 5 s stagger
#    between launches inside a wave further reduces the spike.
FREE_GPUS=( $(algorithm_scientist/free_gpus.sh) )
PARALLEL=${#FREE_GPUS[@]}
[ "$PARALLEL" -gt 4 ] && PARALLEL=4
CFGS=( submissions/_flow_algorithms_test/*.json )
for start in $(seq 0 $PARALLEL $((${#CFGS[@]} - 1))); do
  end=$((start + PARALLEL)); [ "$end" -gt "${#CFGS[@]}" ] && end=${#CFGS[@]}
  for i in $(seq $start $((end - 1))); do
    cfg="${CFGS[$i]}"
    gpu="${FREE_GPUS[$((i - start))]}"
    CUDA_VISIBLE_DEVICES=$gpu \
      python algorithm_scientist/run_ruler.py --config "$cfg" &
    sleep 5
  done
  wait
done
```

Each run writes `summary_ruler_submissions/_flow_algorithms_test/<algo>/latest.json`.

#### Pass criteria

The reference numbers live in
[`vortex_torch/flow/ALGORITHMS_RESULTS.md`](../../vortex_torch/flow/ALGORITHMS_RESULTS.md).
Treat them as a regression baseline:

| Signal | Pass | Investigate |
|---|---|---|
| **Pre-flight (all 8)** | every config compiles | any `EngineConfigError` |
| **RULER accuracy** | ≥ **0.97** for the 0.98-baseline rows (`block_sparse_attention`, `centered_block_sparse_attention`), ≥ **0.99** for the 1.00-baseline rows | accuracy drop ≥ 0.02 below the noted baseline |
| **Throughput** | within ±10 % of the recorded number (on this host; see contention note) | drop ≥ 15 % on a config that wasn't in a contended wave |
| **Backend coverage** | both `flashinfer` and `trtllm` produce the same accuracy bucket; trtllm typically +3-8 % faster on this workload | accuracy diverges between backends, or trtllm regresses below flashinfer |

When investigating an accuracy regression, look at the per-algorithm
generated module under `~/.vortex_compilation_cache/` (or the temp dir
printed by `check_engine_config`) and walk through
[§18 Reading a generated kernel](#18-reading-a-generated-kernel).

#### Why this suite is the right thing to run

The eight flows were chosen so that, between them, they touch every
moving part of the compiler that a refactor is likely to break:

  * Schedule.W vs Schedule.S dispatch (`indexer/compiler/impl.py`,
    `triton_impl/register.py`, `cuda_impl/register.py`,
    `custom_impl/register.py`).
  * Every `IndexerBackend` field — `indices_src`,
    `per_row_kernel_param`, `start_and_count_snippet`,
    `topk_per_row_args`, `topk_trailing_args`,
    `extra_kernel_constexpr_args`, `extra_launcher_args`.
  * PAGED single-page **and** multi-page load/store paths.
  * BATCHED outputs with the `_is_first_workload` gate.
  * RAGGED → BATCHED `Reduce(dim=0)` (the cross-row form).
  * `Save` / `Load` of per-request state and the
    `disable_radix_cache` enforcement.
  * FP8 `kv_cache_dtype` plumbing (set `kv_cache_dtype: "fp8_e4m3"`
    in one config when stress-testing the dtype-cast helpers).

A green run on all eight, on both backends, is the closest thing
this repo has to a "the compiler is healthy" signal.

---

## 16. Extending the framework

### 16.1 Adding a new op

Decide which side (indexer, cache, or both) and which schedule.

**Indexer-side, Schedule.W** (most common):

1. Create `indexer/<family>.py` with a class that inherits `vOp`,
   defines `__init__` and `profile(...)`. For a compiler-generated
   op, pick `output_format` inline using the rules in §5.1
   (typically `BATCHED iff all inputs are BATCHED, else RAGGED`).
2. Create `indexer/compiler/triton_impl/<op>.py` exporting
   `generate_<op>_impl(graph, op_id, ctx) -> str`. Return a single
   compute expression like `tensor_{out}_block = <triton expr>`.
3. Register: add `from .<op> import generate_<op>_impl` and
   `(<OpClass>, Schedule.W): generate_<op>_impl` to
   `triton_impl/register.py:IMPL_REGISTRY`.
4. Export from `indexer/__init__.py`.

**Indexer-side, Schedule.S** (custom kernel):

1. Same class skeleton, but set `self.schedule = Schedule.S`. Custom
   kernels usually want an explicit `_impl_map` (or
   `_supported_formats`) so unsupported input formats fail fast at
   `profile()` time — the standalone codegen reads `t._format` and
   would otherwise blow up downstream.
2. `profile` still allocates a vTensor output (RAGGED or BATCHED).
3. Codegen function returns the *impl body* (launcher code). Push the
   kernel definition onto `ctx.auxilary_func_def_lines`. Example in
   `triton_impl/reduce.py:generate_reduce_dim0_impl`.
4. Register under `(OpClass, Schedule.S)`.

**Cache-side**: mirror on the cache tree. Cache has no `Schedule.S`
slot today — all cache ops are `Schedule.W`. Output-format rule is
"`output._format` if the caller provided one, else `RAGGED`"; assert
`output._format ∈ {PAGED, RAGGED}`.

### 16.2 Supporting a new input-format combination

Compiler-generated ops are already format-agnostic — the
surrounding kernel emits the right load / store path for every
combination, so no per-class change is needed. Verify with
`verify_flow_compilable` on a flow that exercises the new combo.

For custom kernels (Schedule.S indexer ops, `Save` / `Load`,
`Fill`), extend the op's `_impl_map` (or `_supported_formats`) and
match it in the codegen function — these ops branch on `t._format`
inside their generated kernel.

If a brand-new combination needs a new load or store path, extend
kernel_gen's `generate_load_tensor_str` /
`generate_store_tensor_str`. BATCHED inputs and RAGGED outputs are
the most commonly missing paths; PAGED inputs are fully covered for
both indexer and cache.

### 16.3 Adding a new dtype

1. Add the `torch.dtype` ↔ `tl` name mapping in
   `triton_impl/dtype_cast.py:_TORCH_TO_TL`.
2. If it's an fp8-like narrow dtype needing clamp-before-cast, add
   both the clamp max and the `.to(tl.float8eX).to(tl.uint8,
   bitcast=True)` path in `store_cast_expr` and the bitcast-back path
   in `load_cast_expr`.
3. Add it to `vortex_torch.utils.DTYPE_STR_TO_TORCH` so
   `resolve_dtype("<name>")` works.
4. Make sure `VTXGraphCachePool` has a `set_kv_buffer` launcher for
   it if the dtype is KV-cache-capable.

### 16.4 Adding a new schedule

Touching `Schedule` is invasive. `_fuse_w_ops` only merges `Schedule.W`;
everything else runs in its own subgraph. If you need a third
scheduling class, update:

- `indexer/compiler/graph.py:_fuse_w_ops` to decide when to merge.
- `indexer/compiler/triton_impl/kernel_gen.py:generate_triton_impl`
  to emit a different per-subgraph structure.
- `register.py` to dispatch on the new schedule.

In practice, the W/S split has been sufficient so far.

---

## 17. Conventions and gotchas

### 17.1 One-output-per-op

Every op has exactly one output tensor (`op_to_output_tensor_list[op_id]`
is a single-element list). Multi-output ops are modelled as two
separate op instances chained on an intermediate. Breaking this
assumption will surface in `_build_local_graph`'s assertion.

### 17.2 Op instance per call site

A `vOp` instance is only valid for one profile-time call. The
compiler stores per-op state (resolved format, intermediate buffer)
on the instance itself (`self.output_format`, `self.output_buffer`).
Reusing an instance silently corrupts the graph.

### 17.3 Name collisions in generated code

`ctx.sparse_attention_name` is the uuid-suffixed prefix for every
kernel and class the compiler emits. Keep it unique per flow
instance. Standalone kernels (topk, softmax, reduce_dim0) must
embed it in their function names too — see
`generate_reduce_dim0_impl` for the pattern
(`f"{ctx.sparse_attention_name}_reduce_dim0_{rt}_kernel"`).

### 17.4 The producer/consumer convention

`output_tensor_to_op_list[tid]`:

- `None` → `tid` is a graph input (`q`, `o`, `cache['k']`, …).
- `op_id` → op `op_id` produces `tid`. DFS backtracks from consumers
  to producers via this map.

`op_to_input_tensor_list[op_id]`:

- List of tensor ids this op reads. Order matters (binary ops
  distinguish `x` and `y`).

Side-effect ops (Save) are **not** producers of their target cache
field; they're roots of the DFS instead. This is the one exception
to the convention.

### 17.5 `op_to_output_tensor_list` vs `output_tensor_to_op_list`

They're inverses *except* for:

- Side-effect ops: Save's `op_to_output_tensor_list` entry is the
  cache-field tensor, but `output_tensor_to_op_list` for that tensor
  stays `None`.
- Fill: the "input" is the target cache field, and Fill is registered
  as its producer (standard override).

### 17.6 The `indexer` vs `cache` mental-model divide

Cache has **no BATCHED format** and **no cross-block reduction**.
When porting a new op: indexer first if it can be either-side, cache
only if the computation genuinely runs per block. Cross-block work
belongs to `Reduce(dim=0)` on the indexer side.

### 17.7 Don't break sidebar state on `ctx`

Several fields on `Context` are read by codegen from across the
codebase:

- `ctx.tensor_id_to_tensor_name_map` — seeded by the runtime
  backend, extended by `interface.py:generate_entry_point` during
  allocation. Renaming keys breaks both sides.
- `ctx.compilation_header_lines` — triton imports and such. Use
  `dict.fromkeys(...)` to de-dupe before emitting (see
  `generate_interface`).
- `ctx.auxilary_func_def_lines` — standalone kernels append here.
  Multiple S ops of the same class in one flow each get their own
  kernel with a unique name; collisions break imports.

### 17.8 Never trust `t.dtype` to be a bare name

`t.dtype` is a `torch.dtype`. `f"{t.dtype}"` prints
`torch.bfloat16` (fully qualified). The compiler's allocation string
used to prepend a spurious `torch.` and accidentally produce
`torch.torch.bfloat16` — which happens to work (torch self-imports)
but is ugly. Always use `f"{t.dtype}"` directly.

### 17.9 Generated files live in `~/.vortex_compilation_cache/`

Flow compilations don't auto-clean. The engine looks for the compiled
class by name, not by content hash; stale files can cause subtle
"same flow name, different behaviour" bugs. When in doubt,
`rm -rf ~/.vortex_compilation_cache/` before a rerun.

### 17.10 `verify_flow_compilable` only runs the compiler, never the kernels

It catches profile-time errors, graph-construction errors, and
codegen-emission errors — but it won't surface a runtime bug in the
Triton code. That only shows up under sglang with real inputs. For
new ops, dump a small `sparse_attention_name` and inspect the emitted
`.py` by hand.

---

---

## 18. Reading a generated kernel

Every compile run drops a pair of `.py` files into
`~/.vortex_compilation_cache/` (or a custom dir via
`vortex_compilation_cache_dir`). You'll read these constantly.

### 18.1 File structure

```
<name>_compiled_func.py              # indexer side
<name>_cache_compiled_func.py        # cache side
```

Each file contains:

1. **Header** — triton imports, maybe a `from vortex_torch_C import topk_output`.
2. **Auxiliary kernels** — every `@triton.jit` kernel emitted by
   `Schedule.S` codegens (`topk_output` is a C extension so only the
   import; softmax, normalize, reduce_dim0 live here as full
   `@triton.jit` functions).
3. **Per-subgraph functions** — for each subgraph `i`:
   - `<name>_subgraph_i_kernel` (only if W): `@triton.jit` that
     contains the entire fused body.
   - `<name>_subgraph_i_impl`: Python launcher (fp8 rebinding +
     `kernel[(grid,)](...)`) or for S ops, just the launcher body.
   - `<name>_subgraph_i_interface`: thin wrapper the entry class
     calls.
4. **The entry class** — `<name>_CompiledFunc` with
   `__init__` (intermediate allocations) and `forward(...)` (sequence
   of interface calls).

### 18.2 What to search for

Common questions and the grep that answers them:

| question | grep |
|---|---|
| "Which tensors are allocated as intermediates?" | `grep "torch.empty" <file>` |
| "Does this kernel use BATCHED store?" | `grep "_is_first_workload" <file>` |
| "Does this flow use fp8?" | `grep "tl.float8\|view(torch.uint8)" <file>` |
| "What's the per-page score expression?" | find the `tl.store(..., mask=valid[:, None, None])` line; the stored value is right before it |
| "In what order do the subgraphs execute?" | search for `_interface(` inside `class .*CompiledFunc` |
| "Which global tensor id maps to which user name?" | the comments `# global input tensor 2` next to each arg in the entry class's forward body |

### 18.3 Mapping back to your source

Each `tensor_<N>_block` in the kernel corresponds to one vTensor in
`ctx.tensor_list`, with the same ordering as they were registered
during `profile()`. If you kept the order consistent with your
`forward_indexer` / `forward_cache` source, you can trace the number
back by counting:

- `tensor_0` = first tensor registered (usually `q`).
- `tensor_1` = second (usually `o` — though it's an input, not an
  output).
- `tensor_2`, `tensor_3`, ... = cache fields in declaration order.
- higher numbers = intermediates produced by your ops, in the order
  you called them.

The entry-class `forward` body has the ground-truth mapping as
comments (`# global input tensor 2`).

### 18.4 Worked example — reading the `gqa_quest_sparse_attention` kernel

From the compiled indexer:

```python
# Inputs: tensor_0 = q (BATCHED), tensor_1 = cache['max'] (PAGED),
#         tensor_2 = cache['min'] (PAGED)
# Outputs: tensor_7 = score intermediate (RAGGED [S, 1, 1])

for i in range(start, end):
    # --- LOAD q (BATCHED) ---
    new_batch_idx_i32 = tl.load(winfo_x_indices + i).to(tl.int32)
    tensor_0_block = ... .to(tl.float32)    # shape (1, H_q, D)

    # --- LOAD cache['max'], cache['min'] (PAGED) ---
    ragged_idx_i32 = tl.load(winfo_y_offsets + i).to(tl.int32)
    page_idx_i32   = tl.load(indices + ragged_idx_i32).to(tl.int32)
    tensor_1_block = ... .to(tl.float32)    # shape (wl_chunk_size, 1, D)
    tensor_2_block = ... .to(tl.float32)    # shape (wl_chunk_size, 1, D)

    # --- COMPUTE ---
    tensor_3_block = tensor_0_block * tensor_1_block    # (wl_chunk_size, H_q, D)
    tensor_4_block = tensor_0_block * tensor_2_block    # (wl_chunk_size, H_q, D)
    tensor_5_block = tl.maximum(tensor_3_block, tensor_4_block)
    tensor_6_block = tl.sum(tensor_5_block, keep_dims=True, axis=2)     # (..., H_q, 1)
    tensor_7_block = tl.max(tensor_6_block, keep_dims=True, axis=1)     # (..., 1, 1)

    # --- STORE tensor_7 (RAGGED) ---
    _len = tl.load(winfo_y_lens + i)
    valid = workload_ptr < _len
    tl.store(tensor_7_block_ptr, tensor_7_block.to(tl.bfloat16),
             mask=valid[:, None, None])
```

Every line corresponds exactly to one op in the user's
`forward_indexer`:

- `tensor_0_block * tensor_1_block` → `self.mul_max(q, cache["max"])`
- `tensor_0_block * tensor_2_block` → `self.mul_min(q, cache["min"])`
- `tl.maximum(...)` → `self.maximum_op(s_max, s_min)`
- `tl.sum(..., axis=2)` → `self.sum(s)` (Sum(dim=2))
- `tl.max(..., axis=1)` → `self.max_op(score)` (Max(dim=1))

---

## 19. Debugging the compiler

A catalogue of common error modes and how to diagnose each.

### 19.1 `AssertionError: no implementation for x_fmt=...` (custom kernels)

**Cause**: One of the ops that still uses `_impl_map` /
`_supported_formats` — `Save`, `Load`, `topK`, `approxTopK`,
`Softmax`, `Normalize`, `Conv1d`, `Fill`, `Reduce(dim=0)` — got an
input format it doesn't have a codegen path for. Either (a) the
flow is using a combination the table doesn't support, or (b) an
earlier op returned an unexpected format.

**Diagnose**: Add a print in `profile()` before the assertion:
```python
print(f"{prefix}got x_fmt={x._format}; map keys: {list(self._impl_map.keys())}")
```
Trace back through `forward_indexer` / `forward_cache` to find the
op whose output format is different from what you expected — most
common culprit is a `Reduce(dim=0)` collapsing a RAGGED tensor to
BATCHED upstream of an op that only handles RAGGED.

Compiler-generated ops no longer raise this error — they derive
their output format inline from input formats (see §5.1), so a
"wrong format" surfaces later as a shape or kernel-launch mismatch
rather than a clean profile-time assertion.

### 19.2 `RuntimeError: Cycle detected in subgraph DAG`

**Cause**: Two subgraphs end up with a mutual dependency. Almost
always caused by a tensor that's both Load-read and Save-written in
the same flow, where Save accidentally became a consumer-edge
source.

**Diagnose**: Confirm that Save's `profile()` calls
`ctx.side_effect_op_ids.append(save_op_id)` **instead of**
`ctx.output_tensor_to_op_list[o.tensor_id] = ...`. If a new op you're
adding writes to a cache field *and* the field is also read
elsewhere, use the Save pattern.

### 19.3 `KeyError: <N>` from `generate_subgraph_entry_point`

**Cause**: A tensor appears in `sub_graph.global_input_tensor_ids` or
`global_output_tensor_ids` but isn't in `tensor_id_to_tensor_name_map`.
Usually an intermediate whose allocation was skipped because the
compiler treated it as a final output.

**Diagnose**: Check that `interface.py:generate_entry_point` allocates
the tensor — the new check is:
```python
if t.tensor_id in tensor_id_to_tensor_name_map:
    continue   # caller-provided
```
If your tensor is neither caller-provided nor allocated, the map
lookup fails.

### 19.4 `SyntaxError: duplicate argument 'tensor_X'`

**Cause**: A cache tensor is both an input (Load) and an output
(Save) of the same subgraph, and the dedup-by-output_set check is
missing or broken.

**Diagnose**: Grep the generated `.py` for the duplicated argument.
Then verify the dedup in:
- `interface.py:generate_subgraph_func` (interface wrapper args)
- `kernel_gen.py:generate_triton_kernel` (kernel signature)
- `kernel_gen.py:generate_triton_impl` (launcher args)

All three must skip the input-arg for tensors that are also in the
output set.

### 19.5 `triton.runtime.errors.OutOfResources`

**Cause**: The kernel's register pressure is too high. Usually from:
- Too many simultaneous large `(workload_chunk_size, D_0, D_1)` tiles
  alive at once.
- A Schedule.S kernel with `BLOCK_P = 512` holding an
  `(512, D_0, D_1)` slab in registers when `D_0 * D_1` is large.

**Diagnose**: Reduce `BLOCK_P` in the S kernel, or reduce
`workload_chunk_size`, or lower `num_stages` in the launch.

### 19.6 Generated kernel compiles but attention output is wrong

**Cause**: Subtle — usually one of:
- An op emits the wrong shape. Read the kernel; confirm the reshape
  at the end of a load gives the expected `(workload_chunk_size,
  D_0, D_1)` / `(1, D_0, D_1)` shape.
- A `valid` mask is missing on a RAGGED store, so tail positions
  write garbage.
- BATCHED store fired from a non-first workload (the
  `winfo_is_first_workload_per_batch` gate was dropped).
- Dtype mismatch: something stored as bf16 but read back as fp32
  directly (or vice versa).

**Diagnose**: Force-narrow the bug by disabling layers in
`vortex_layers_skip` and re-running; find the first sparse layer
that diverges from the dense baseline.

### 19.7 Stale kernel from `~/.vortex_compilation_cache/`

**Cause**: The engine loads compiled modules by name. Two flows with
the same `vortex_module_name` can clobber each other.

**Diagnose**: `ls -la ~/.vortex_compilation_cache/` and compare
timestamps. If in doubt, `rm -rf ~/.vortex_compilation_cache/` before
a rerun.

### 19.8 `verify_flow_compilable` passes but runtime crashes

**Cause**: Verify only exercises the profile + codegen path; it
doesn't launch kernels. Runtime crashes come from actual Triton
issues (register pressure, illegal block_ptr args, etc.) that
verify can't see.

**Diagnose**: Compile a minimal config, look at the generated file by
hand, and if nothing obvious is wrong, add a print statement inside
the Triton kernel (Triton supports `tl.device_print`) to narrow the
failing tensor.

---

## 20. Worked example — adding a new op end-to-end

A full walkthrough: let's add a `Clamp(lo, hi)` indexer op that does
elementwise `max(lo, min(x, hi))`. Schedule.W, shape-preserving.

### 20.1 The op class

Create `vortex_torch/indexer/clamp.py`:

```python
import torch
from typing import Optional
from .context import Context
from ..abs import vTensor, FORMAT, vOp
from ..utils import Schedule


class Clamp(vOp):
    r"""Elementwise clamp: y = max(lo, min(x, hi)). Shape-preserving."""

    def __init__(self, lo: float, hi: float):
        super().__init__()
        assert lo < hi, f"Clamp: lo ({lo}) must be < hi ({hi})"
        self.lo = float(lo)
        self.hi = float(hi)
        self.output_format: Optional[FORMAT] = None
        self.output_buffer: Optional[vTensor] = None
        self.schedule = Schedule.W

    def profile(self, x: vTensor, ctx: Context) -> vTensor:
        prefix = self._prefix()
        assert isinstance(x, vTensor), f"{prefix}x must be vTensor"
        assert x.dim() == 3, f"{prefix}x must be 3D"

        # Compiler-generated op: BATCHED iff input is BATCHED, else RAGGED.
        self.output_format = (
            FORMAT.BATCHED if x._format == FORMAT.BATCHED else FORMAT.RAGGED
        )

        self.output_buffer = vTensor(
            shape=(0, x.shape[1], x.shape[2]),
            dtype=ctx.vortex_dtype,
            device=x.device,
            _format=self.output_format,
            tensor_id=len(ctx.tensor_list),
        )
        ctx.tensor_list.append(self.output_buffer)
        ctx.output_tensor_to_op_list.append(len(ctx.op_list))
        ctx.op_list.append(self)
        ctx.op_to_input_tensor_list.append([x.tensor_id])
        ctx.op_to_output_tensor_list.append([self.output_buffer.tensor_id])
        return self.output_buffer
```

### 20.2 The codegen function

Create `vortex_torch/indexer/compiler/triton_impl/clamp.py`:

```python
from ..graph import Graph
from ...context import Context
from ...clamp import Clamp


def generate_clamp_impl(graph: Graph, op_id: int, ctx: Context) -> str:
    in_id  = graph.op_to_input_tensor_list[op_id][0]
    out_id = graph.op_to_output_tensor_list[op_id]
    op     = graph.op_list[op_id]
    assert issubclass(op.__class__, Clamp)
    return (
        f"tensor_{out_id}_block = tl.minimum("
        f"tl.maximum(tensor_{in_id}_block, {op.lo}), {op.hi})"
    )
```

Format-agnostic: the surrounding kernel handles BATCHED/RAGGED
loads and stores; the op just emits the arithmetic.

### 20.3 Register

Edit `vortex_torch/indexer/compiler/triton_impl/register.py`:

```python
from ...clamp import Clamp
from .clamp import generate_clamp_impl

IMPL_REGISTRY = {
    ...,
    (Clamp, Schedule.W): generate_clamp_impl,
    ...,
}
```

### 20.4 Export

Edit `vortex_torch/indexer/__init__.py`:

```python
from .clamp import Clamp

__all__ = [
    ...,
    "Clamp",
]
```

### 20.5 Verify

Write a small test flow that uses `Clamp`, then:

```bash
python -m vortex_torch.flow.verify <your_test_flow> \
    --B 2 --num-kv-heads 2 --max-page-size 64 --max-num-pages-per-request 64
```

Should pass 450/450 configs. Open a generated `.py` and grep for
`tl.minimum(tl.maximum(...)` — you'll see your op inlined into a
fused kernel.

### 20.6 Total diff

- Two new files (`clamp.py` × 2), ~60 lines.
- Two lines added to `register.py`.
- One line added to `__init__.py`.

That's the full footprint for a new W-scheduled indexer op. Cache-side
equivalent would mirror this in the `cache/` tree; `Schedule.S` ops
would add ~50 lines for the standalone kernel template.

---

## Open questions / future work

- **Multi-output ops**: the "one output per op" constraint forces
  split ops for kernels that naturally produce multiple tensors. A
  cleaner fix would be lists throughout.
- **Cache-side `Schedule.S`**: no op needs it today, but if we add
  cross-block cache maintenance (e.g. periodic compaction), the fuser
  + kernel_gen will need a standalone path.
- **Dynamic shapes**: all shape constants are baked at profile time
  via `ctx.max_*`. Supporting truly dynamic decode lengths would
  require rethinking the interface layer.
- **Multiple backends**: `impl_backend` is a string that today only
  accepts `"triton"`. The `impl.py` files have a single-entry
  `AVAILABLE_IMPL_BACKENDS` dict. A CUDA-graph / CUTLASS / TMA
  backend would slot in here.
- **Register pressure modelling**: `Schedule.W` fusion is greedy —
  it'll merge until `can_merge_bidir` fails. A cost model that
  accounts for register footprint per fused op would let the fuser
  stop before oversubscribing.
- **Dispatch on more than `(class, schedule)`**: some ops want to
  pick a codegen based on `(class, schedule, dtype)` (e.g. fp8
  matmul with tensor-core paths). Today that dispatch is hand-rolled
  inside the codegen function.
