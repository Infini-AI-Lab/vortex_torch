<p align="center">
  <img
    alt="Vortex"
    src="assets/vortex_logo_flat.png"
    width="55%"
  />
</p>

<h3 align="center">
Vortex: Programmable Sparse Attention for Agents as Algorithm Designers
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2606.06453"><img src="https://img.shields.io/badge/Paper-arXiv%3A2606.06453-b31b1b?logo=arxiv&logoColor=white" alt="Paper" /></a>
  <a href="https://infini-ai-lab.github.io/vortex_torch/docs/"><img src="https://img.shields.io/badge/Docs-Documentation-1f6feb?logo=readthedocs&logoColor=white" alt="Documentation" /></a>
  <a href="https://infini-ai-lab.github.io/vortex_torch/"><img src="https://img.shields.io/badge/Website-vortex__torch-2ea44f?logo=githubpages&logoColor=white" alt="Website" /></a>
</p>


**Vortex turns sparse-attention algorithm design into something AI agents can do.** Sparse attention is increasingly essential for serving LLMs as generation lengths grow — but deploying and evaluating *new* sparse-attention algorithms at scale has been highly engineering-intensive, slowing both **human researchers** and **AI agents** as they explore the design space.

Vortex couples a **Python-embedded frontend** over a **page-centric tensor abstraction** — concise enough to express a broad range of sparse-attention algorithms — with an **efficient backend tightly integrated into modern LLM serving stacks** (SGLang). A new algorithm goes from idea to deployed-and-benchmarked in minutes, turning its theoretical efficiency into real-world throughput without touching core model code.

This makes Vortex a platform for **autonomous algorithm discovery**: AI agents generate and refine diverse sparse-attention algorithms with Vortex — the best reaching up to **3.46× higher throughput** than full attention while preserving accuracy. Vortex also extends sparse attention to emerging architectures and very large models that are otherwise hard to experiment with (up to **4.7×** on the MLA-based GLM-4.7-Flash and **1.37×** on the 229B-parameter MiniMax-M2.7), and doubles as a research instrument for understanding **where the routing signal lives** in sparse attention.

<p align="center">
  <img src="assets/fig1_workflow.png" alt="A workflow to study sparse attention algorithms with Vortex" width="40%" />
  &nbsp;&nbsp;
  <img src="assets/fig1_results.png" alt="Agent-generated sparse attention on Qwen3-1.7B / AIME" width="52%" />
</p>
<p align="center">
  <em><b>(a)</b> A workflow to study sparse attention algorithms using Vortex. &nbsp;
  <b>(b)</b> Agent-generated sparse attention (Qwen3-1.7B, AIME, NVIDIA H200): each point is one
  algorithm generated or optimized by AI agents with Vortex — the best reaches up to
  3.46× the throughput of full attention while preserving accuracy.</em>
</p>


---

## ✨ Key Features

- **Easy Programming**  
  Program sparse attention with a PyTorch-like frontend. No worrying about batching, caching & paged attention.

- **High Performance**  
  Built to work with FlashInfer & CUDA Graph & Radix Attention for efficient LLM inference.

- **Agent Native**  
  Designed for autonomous algorithm discovery — AI agents generate, benchmark, and refine sparse attention end-to-end, with a Claude Code workspace and OpenHands demo built in.

---

## 🚀 Installation

```bash
git clone --recursive https://github.com/Infini-AI-Lab/vortex_torch.git

# Install SGLang dependency
cd third_party/sglang/v0.5.9/sglang
pip install -e "python"
cd ../../../../

# Install Vortex
cd vortex_torch
pip install -e .
```

---

## 🤖 Innovate Sparse Attention with OpenHands

Vortex is designed not only for hand-crafted sparsity patterns but also for AI-generated sparse attention.

Our demo shows how to use SOTA agents OpenHands (https://openhands.dev/) to generate sparse attention algorithms.

<figure>
  <img src="assets/demov2.0.gif" alt="Demo — OpenHands generating a sparse attention algorithm" />
  <figcaption align="center"><em>OpenHands generates a sparse attention algorithm (up to 2.7X speedup in this example).</em></figcaption>
</figure>

```bash
export LLM_API_KEY=YOUR_API_KEY
python AI/openhands_gen.py

```

The usage and installation guide of OpenHands can be found in https://docs.openhands.dev/sdk. 

Note: Some operators are not yet fused or fully optimized, which may lead to increased memory usage. Tune down the `mem_fraction_static` if CUDA OOM. This can also impact generation speed during inference. 

---

## 🧠 Iterate & Innovate with Claude Code

<p align="center">
  <img src="assets/fig11_iter_a.png" alt="mean@16 per iteration" width="32%" />
  <img src="assets/fig11_iter_b.png" alt="throughput per iteration" width="32%" />
  <img src="assets/fig11_iter_c.png" alt="accuracy–throughput frontier colored by iteration" width="32%" />
</p>
<p align="center">
  <em>Long-horizon autonomous optimization on AIME24 (Claude Opus 4.7, Qwen3-1.7B, 23 iterations,
  92 submissions): <b>(a)</b> mean@16 per iteration, <b>(b)</b> throughput per iteration,
  <b>(c)</b> the accuracy–throughput frontier of all submissions, colored by iteration order.</em>
</p>

Vortex ships a [Claude Code](https://claude.com/claude-code) workspace
(`.claude/`) that turns the framework into an autonomous *algorithm
scientist*: Claude writes sparse-attention submissions, compiles them,
benchmarks them on AIME24, and pushes the accuracy/throughput Pareto
frontier outward — one batch at a time. Start a session from the repo
root and drive it with slash commands:

| Command | What it does |
| --- | --- |
| `/new-submission <name>` | Scaffold a new submission pair (`.py` + `.json`). |
| `/preflight <name>` | Cheap CPU-only config check before spending GPU time. |
| `/innovate <N> [theme]` | **Innovate** — draft `N` *genuinely novel* algorithms in one shot. All must compile; no benchmark loop. Great for brainstorming ideas the literature doesn't cover. |
| `/iterate [--max-iterations <N>]` | **Iterate** — the long-horizon loop: design 4 orthogonal variants → pre-flight → RULER quality gate → benchmark on AIME24 → analyse → repeat. Autonomously maps the Pareto frontier. |
| `/batch-benchmark <n1> <n2> <n3> <n4>` | Launch a 4-variant batch on the currently-free GPUs. |
| `/review <name>` | Audit a submission against the contract without editing it. |

**Innovate** (explore) and **iterate** (exploit) are complementary:
`/innovate` generates fresh, compile-checked ideas with no GPU cost,
while `/iterate` benchmarks four variants per batch and folds the
results back into `algorithm_scientist/memory.md` so later sessions
resume from the running best. The full contract, knobs, and benchmark
protocol live under [`AI/`](AI/) (start with
[`AI/AGENTS.md`](AI/AGENTS.md)) and
[`papers/guide.md`](papers/guide.md).

```bash
# from a Claude Code session opened at the repo root
/innovate 4 channel-sparsity      # draft 4 novel ideas to explore
/iterate --max-iterations 3       # run 3 autonomous benchmark batches
```

---

## 🧩 Quick Example: Custom Sparse Attention

<p align="center">
  <img src="assets/fig16_mm_a.png" alt="mean@16 vs throughput" width="32%" />
  <img src="assets/fig16_mm_b.png" alt="pass@4 vs throughput" width="32%" />
  <img src="assets/fig16_mm_c.png" alt="pass@8 vs throughput" width="32%" />
</p>
<p align="center">
  <em>Scaling to a 229B model with tensor parallelism — MiniMax-M2.7 (229B) on AIME26 with
  32K-token generation on four NVIDIA B200 GPUs (TP=4): <b>(a)</b> mean@16, <b>(b)</b> pass@4,
  <b>(c)</b> pass@8 versus end-to-end throughput. Block top-k and Quest sweep the number of
  attended blocks; the star marks the full-attention operating point.</em>
</p>

A working setup is **two files**:

1. **The flow module** (this section) — a `.py` file that *defines* your
   sparse-attention algorithm as a `vFlow` subclass and `@register`s it
   under a name. It contains only vortex ops; it never imports sglang.
2. **The launch script** ([next section](#-launch-it-with-sglang)) —
   imports `sglang` + `vortex_torch` and starts the engine pointing at
   the flow by its registered name.

### 1. Define the flow — `custom_sparse_attention.py`

A `vFlow` declares its cache layout in `create_cache`, refreshes
per-page state in `forward_cache`, and scores/selects pages every decode
step in `forward_indexer`. Save the snippet below anywhere on disk (e.g.
`custom_sparse_attention.py`) — you'll point the engine at it by path +
registered name.

```python
from typing import Dict
import torch

from vortex_torch.flow import vFlow, register
from vortex_torch.indexer import GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("custom_sparse_attention")
class CustomSparseAttention(vFlow):

    def __init__(self):
        super().__init__()
        # Indexer-side ops (run every decode step)
        self.mean = Mean(dim=1)        # average over the query heads
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ
        self.output_func = topK()      # must end in topK / approxTopK

        # Cache-side ops (run once per finished page)
        self.reduction = CMean(dim=1)  # one centroid (mean key) per page

    def forward_indexer(
        self,
        q: torch.Tensor,                 # viewed as [B, H_q, D]
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],  # viewed as [S, r, c] per create_cache()
        ctx: ContextBase,
    ):
        # No native torch ops here — every tensor flows through vortex ops.
        q_mean = self.mean(q, ctx=ctx)                          # [B, 1, D]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)  # [S, 1, 1]
        self.output_func(score, o, ctx=ctx)                     # selected pages -> o

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],  # viewed as [B, r, c] per create_cache()
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        # triggered only when a page is finished
        self.reduction(cache["k"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, head_dim: int):
        # "k" and "v" are provided automatically — do not declare them
        return {
            "centroids": (1, head_dim),
        }
```

---

## 🏃 Launch it with SGLang

The launch script is a **separate file** from the flow. It imports
sglang and vortex_torch, then starts the engine. Importing `vortex_torch`
is what wires vortex into sglang's decode loop (it installs the
`ServerArgs` ↔ `VortexConfig` adapter), so the import is required even
though you don't call it directly.

```python
import sglang as sgl
import vortex_torch  # noqa: F401 — import for side effect: installs the VortexConfig adapter
from vortex_torch.engine.sgl.config import VortexConfig

llm = sgl.Engine(
    # --- standard sglang engine args ---
    model_path="Qwen/Qwen3-0.6B",
    page_size=16,                     # KV page size (pages are vortex's unit of sparsity)
    attention_backend="flashinfer",   # Mandatory
    disable_overlap_schedule=True,    # Mandatory
    disable_cuda_graph=False,
    mem_fraction_static=0.85,         # turn down if you hit CUDA OOM

    # --- all vortex knobs live in one object ---
    # Passing `vortex=` turns sparsity ON (no separate enable flag needed);
    # omit it and you get plain dense attention.
    vortex=VortexConfig(
        module_path="path/to/custom_sparse_attention.py",
        module_name="custom_sparse_attention",  # the @register name of your vFlow
        topk_val=30,                  # keep the 30 highest-scoring pages per query
        layers_skip=[0],              # layer 0 runs full/dense attention
        block_reserved_bos=1,         # always keep the first page (attention sink)
        block_reserved_eos=1,         # always keep the last (most recent) page
        max_seq_lens=8192,
    ),
)
```

### What is `VortexConfig`?

`VortexConfig` is a single dataclass
([`vortex_torch/engine/sgl/config.py`](vortex_torch/engine/sgl/config.py))
that holds **every** vortex sparse-attention hyper-parameter in one place,
instead of ~18 loose `vortex_*` arguments scattered across sglang's
`ServerArgs`. Its presence on the engine is also the on/off switch: pass a
`VortexConfig` and sparsity is enabled; leave it out and the model runs
ordinary dense attention.

**Every field, with what it controls and an example value:**

| Field | Explanation | Example |
| --- | --- | --- |
| `module_path` | Path to the `.py` file holding your flow. `None` → vortex searches `vortex_torch.flow.algorithms`. | `"submissions/custom.py"` |
| `module_name` | The `@register(...)` name of the `vFlow` to load. Must match exactly. | `"custom_sparse_attention"` |
| `topk_val` | **Static page budget** — the fixed minimum number of pages each sequence keeps, regardless of length. The core accuracy↔throughput knob. | `30` |
| `topk_ratio` | **Dynamic page budget** — a fraction of the sequence's pages; the engine keeps `max(static floor, topk_ratio × num_pages)`. `0.0` disables it (use `topk_val` only). | `0.0625` |
| `max_topk_val` | Upper bound on the selected-page count, used to size/pick the top-k kernel variant. `None` → derived from `max_seq_lens`. | `256` |
| `layers_skip` | Layer indices that **bypass sparse attention and run dense** (e.g. early layers that need global context). `None` → all layers sparse. | `[0, 4, 8, 12]` |
| `block_reserved_bos` | Pages at the **start** of the sequence that are always selected (attention sink). Int ≥ 1. | `1` |
| `block_reserved_eos` | Pages at the **end** (most-recent tokens) that are always selected. Int ≥ 1. | `1` |
| `max_seq_lens` | Maximum sequence length to plan buffers for. `-1` → use the model default. | `8192` |
| `block_size` | Vortex **page size** (the unit of sparsity). Positive power of 2; smaller = finer granularity, larger = less cache-summary overhead. Defaults to sglang's `page_size`. | `16` |
| `workload_chunk_size` | Planner granularity — how many blocks are grouped into one indexer workload. Positive power of 2; a throughput-tuning knob. | `32` |
| `dtype` | dtype for **intermediate** indexer tensors. `"bfloat16"` is the tested default; `"float16"`/`"float32"`/`"fp8_e4m3"`/`"fp8_e5m2"` are accepted. | `"bfloat16"` |
| `compilation_cache_dir` | Directory for the JIT-compiled kernel cache. `None` → next to the compiler module. | `"/tmp/vortex_cache"` |
| `schedule_policy` | **A CUDA C++ snippet that computes each sequence's page budget** (see below). `None` → the default budget formula. | `None` |
| `attention_backend` | Sparse-attention kernel family: `"flashinfer"` (default) or `"trtllm"`. | `"flashinfer"` |
| `impl_backend` | Indexer op implementation backend: `"triton"` (default) or `"cuda"`. | `"triton"` |
| `use_tensor_core` | Enable tensor-core (bf16 `tl.dot`) codegen in the triton kernel. Only valid with `impl_backend="triton"`. | `False` |

### Programmable budget — the `schedule_policy`

`schedule_policy` is the most interesting knob: instead of a fixed
formula, the per-sequence **page budget is computed by a CUDA C++ snippet
you provide**. Vortex injects it as the body of a `__device__` function,
JIT-compiles it into the decode planner (cached by content hash), and runs
it for every sequence on every backend (flashinfer *and* trtllm/MLA) —
not just trtllm. When you leave it `None`, vortex uses this default body,
which is exactly the standard budget formula:

```cpp
// default schedule_policy — returns the number of pages to attend to.
const int static_kv_budget  = topk_val + block_reserved_bos + block_reserved_eos;
const int dynamic_kv_budget = int(cached_block_len * topk_ratio);
return max(static_kv_budget, dynamic_kv_budget);   // topk_val dominates short seqs, ratio long ones
```

The snippet must `return` an `int` (the page budget). The variables in
scope are the live per-sequence values:

| Variable | Meaning |
| --- | --- |
| `cached_block_len` | Number of cached KV pages currently in this sequence (its length, in pages). |
| `topk_val` | The configured static budget. |
| `topk_ratio` | The configured dynamic ratio. |
| `block_reserved_bos` / `block_reserved_eos` | Reserved sink / recent pages. |

Because it's real device code, you can express **budget policies the two
scalar knobs can't** — e.g. a length-adaptive budget that grows slowly
with context and then caps:

```python
vortex=VortexConfig(
    module_name="custom_sparse_attention",
    topk_val=32,
    schedule_policy=r"""
        // base budget + 1 extra page per 64 cached pages, capped at 256
        const int base  = topk_val + block_reserved_bos + block_reserved_eos;
        const int extra = cached_block_len / 64;
        return min(base + extra, 256);
    """,
)
```

Other natural uses: a hard ceiling on long contexts, a step function that
jumps the budget past a length threshold, or a `sqrt(cached_block_len)`-style
sublinear schedule. The planner is JIT-compiled once per distinct snippet,
so there's no per-step overhead.

Prefer the explicit `VortexConfig(...)` object above. The legacy flat form
— `sgl.Engine(enable_vortex_sparsity=True, vortex_topk_val=30, vortex_module_name=..., ...)`
— still works (the adapter folds those `vortex_*` kwargs into a
`VortexConfig` for you), but the object is clearer and self-documenting.

---

## 🧬 MLA Models (DeepSeek-V3 / GLM-4.7 / Kimi-style)

<p align="center">
  <img src="assets/fig12_glm_a.png" alt="mean@16 vs throughput" width="32%" />
  <img src="assets/fig12_glm_b.png" alt="pass@4 vs throughput" width="32%" />
  <img src="assets/fig12_glm_c.png" alt="pass@8 vs throughput" width="32%" />
</p>
<p align="center">
  <em>Sparse attention on an MLA model — GLM-4.7-Flash on AIME26 with 32K-token generation on a
  single NVIDIA B200: <b>(a)</b> mean@16, <b>(b)</b> pass@4, <b>(c)</b> pass@8 versus end-to-end
  throughput. Three MLA flows are expressed in vFlow (rope-aware block-sparse, rope-unaware
  block-sparse, and Quest), sweeping attended blocks with block sizes 16/32/64.</em>
</p>

Models with **Multi-head Latent Attention** (DeepSeek-V2/V3, GLM-4.7-Flash,
Kimi, …) compress the KV cache into a single shared low-rank *latent*
instead of per-head K/V. Vortex supports them with a parallel base class,
**`vFlowMLA`**. The shape of a flow is the same — `create_cache` /
`forward_cache` / `forward_indexer` — but with MLA conventions:

- The cache exposes **one auto-provided field, `cache["latent"]`**
  (the fused `[ kv_c | k_pe ]`, inner dim `kv_lora_rank + qk_rope_head_dim`)
  — there is no `"k"` / `"v"`.
- `create_cache(block_size, kv_lora_rank, qk_rope_head_dim)` declares only
  your aux tensors (e.g. centroids).
- `forward_indexer` receives the **fused absorbed query**
  `q = [ q_nope_out | q_pe ]`. Because both query and latent are the
  concatenations, a *single* dot `⟨q, centroid⟩` already equals the full
  decode logit `⟨q_nope, kv_c⟩ + ⟨q_pe, k_pe⟩` — RoPE included.

```python
from typing import Dict
import torch

from vortex_torch.flow import vFlowMLA, register
from vortex_torch.indexer import GeMM, Mean, topK
from vortex_torch.cache import Mean as CMean
from vortex_torch.abs import ContextBase


@register("rope_aware_block_sparse_mla")
class RopeAwareBlockSparseMLA(vFlowMLA):

    def __init__(self):
        super().__init__()
        self.mean = Mean(dim=1)        # average the fused query over its H heads
        self.gemm = GeMM()             # GeMM(x, y) = y @ xᵀ → per-page score
        self.output_func = topK()      # terminal: write selected page ids to o
        self.reduction = CMean(dim=1)  # centroid = mean of the fused latent per page

    def forward_indexer(
        self,
        q: torch.Tensor,                 # [B, H, latent_dim]  ([q_nope_out | q_pe])
        o: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        ctx: ContextBase,
    ):
        q_mean = self.mean(q, ctx=ctx)                          # [B, 1, latent_dim]
        score = self.gemm(q_mean, cache["centroids"], ctx=ctx)  # [S, 1, 1] — FULL logit
        self.output_func(score, o, ctx=ctx)

    def forward_cache(
        self,
        cache: Dict[str, torch.Tensor],  # cache["latent"] auto-provided: [B, 1, latent_dim]
        loc: torch.Tensor,
        ctx: ContextBase,
    ):
        self.reduction(cache["latent"], cache["centroids"], loc=loc, ctx=ctx)

    def create_cache(self, block_size: int, kv_lora_rank: int, qk_rope_head_dim: int):
        # "latent" is auto-provided — declare only the aux centroid (full width).
        return {
            "centroids": (1, kv_lora_rank + qk_rope_head_dim),
        }
```

Launching is the same `VortexConfig` flow, with the **MLA decode backend**
selected on the engine (`attention_backend="trtllm_mla"`) and the
tensor-core indexer enabled:

```python
import sglang as sgl
import vortex_torch  # noqa: F401
from vortex_torch.engine.sgl.config import VortexConfig

llm = sgl.Engine(
    model_path="zai-org/GLM-4.7-Flash",   # any MLA model (DeepSeek-V3, Kimi, …)
    trust_remote_code=True,
    page_size=32,
    attention_backend="trtllm_mla",       # vortex CUDA MLA decode kernel
    kv_cache_dtype="auto",
    mem_fraction_static=0.9,
    vortex=VortexConfig(
        module_name="rope_aware_block_sparse_mla",
        attention_backend="trtllm",       # 2D block-table indexer
        impl_backend="triton",
        use_tensor_core=True,             # tensor-core indexer GeMM
        block_size=32,
        topk_val=61,
        block_reserved_bos=1,
        block_reserved_eos=2,
        max_seq_lens=8192,
    ),
)
```

A runnable, single-GPU MLA demo (RULER scoring, dense-vs-sparse) lives in
[`examples/ruler/run_ruler_mla.py`](examples/ruler/run_ruler_mla.py).

---

## 🌐 Server Mode (OpenAI-compatible endpoint)

To serve vortex sparse attention over HTTP instead of driving the engine
in-process, use [`examples/misc/server_launch.sh`](examples/misc/server_launch.sh).
It boots an sglang server with an **OpenAI-compatible API** on
`127.0.0.1:30000`:

```bash
# ./server_launch.sh <MODEL_NAME> <TP_SIZE>
examples/misc/server_launch.sh Qwen/Qwen3-4B 1
```

Two details make server mode work:

1. **`import vortex_torch` must run first.** The script doesn't call
   `python -m sglang.launch_server` directly — that builds `ServerArgs` in
   the parent before vortex is imported, so the adapter that folds the
   config wouldn't be installed yet. Instead it imports `vortex_torch`,
   then calls sglang's `run_server`, so the `ServerArgs` ↔ `VortexConfig`
   adapter is in place before the args are pickled to the scheduler worker.
2. **Knobs are passed as JSON via `--vortex-config`.** The per-knob
   `--vortex-*` CLI flags no longer exist; the script writes the
   `VortexConfig` fields (prefix stripped) to a temp JSON file and feeds it
   through `--vortex-config '<json>'`. Providing a non-null config
   implicitly enables sparsity. Edit the JSON block in the script to retune.

Once up, query it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---


## 📚 Citation

If you find Vortex useful in your research, please cite:

```bibtex
@misc{chen2026vortexefficientprogrammablesparse,
      title={Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents}, 
      author={Zhuoming Chen and Xinrui Zhong and Qilong Feng and Ranajoy Sadhukhan and Yang Zhou and Michael Qizhe Shieh and Zhihao Jia and Beidi Chen},
      year={2026},
      eprint={2606.06453},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.06453}, 
}
```


