"""Compile-time verification for vFlow sparse attention modules.

The :func:`verify_flow_compilable` helper sweeps through combinations of
``(G, D, block_size, page_size, num_pages_per_workload)`` and, for each,
constructs a pair of **pure-metadata** indexer / cache contexts, runs
the flow's ``profile`` methods, and invokes the two compilers. Nothing
is ever allocated on the GPU and ``ctx.create()`` is never called —
only the fields that the generators actually read at codegen time need
to be populated with real values. Tensor-valued fields that are
referenced as ``ctx.X`` in emitted source are backed by small real
CPU tensors so that user ``profile()`` code that happens to read them
won't break.

This is useful for:

  * Lint-checking a new :class:`vFlow` implementation.
  * CI on hosts without a GPU.
  * Catching shape / dtype / format mismatches across all shapes a
    deployment is expected to support.
"""

from __future__ import annotations

import os
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..abs import vTensor, FORMAT
from ..utils import Mode, resolve_dtype
from ..indexer.context import Context as IndexerContext
from ..indexer.metadata import MetaData as IndexerMetaData
from ..indexer.compiler.compile import compile as compile_indexer
from ..cache.context import Context as CacheContext
from ..cache.compiler.compile import compile as compile_cache
from .flow import vFlow


# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------

#: Default query-head multiplier (``G`` in ``q.shape = [B, G, D]``).
DEFAULT_G_VALUES: Tuple[int, ...] = (1, 2, 4, 8, 16)
#: Default head-dim values to sweep.
DEFAULT_D_VALUES: Tuple[int, ...] = (32, 64, 128)
#: Default block sizes to sweep.
DEFAULT_BLOCK_SIZES: Tuple[int, ...] = (4, 8, 16, 32, 64, 128)
#: Default ``page_size / block_size`` ratios (``page_size = 2^k * block_size``).
DEFAULT_PAGE_BLOCK_RATIOS: Tuple[int, ...] = (1, 2, 4, 8, 16)
#: Default multi-page-per-workload values to sweep.
DEFAULT_PAGES_PER_WORKLOAD: Tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class Config:
    """A single point in the verification grid."""
    B: int
    G: int
    D: int
    num_kv_heads: int
    block_size: int
    page_size: int
    num_blocks_per_page: int
    num_pages_per_workload: int
    workload_chunk_size: int  # == num_pages_per_workload * num_blocks_per_page

    @property
    def num_qo_heads(self) -> int:
        # GQA convention: each KV head is shared by ``G`` query heads.
        return self.G * self.num_kv_heads

    def label(self) -> str:
        return (
            f"B={self.B} G={self.G} D={self.D} "
            f"kvh={self.num_kv_heads} qoh={self.num_qo_heads} "
            f"block={self.block_size} page={self.page_size} "
            f"nppw={self.num_pages_per_workload} "
            f"wcs={self.workload_chunk_size}"
        )


@dataclass
class Failure:
    """A single failing config, annotated with which phase broke."""
    cfg: Config
    phase: str  # "indexer" | "cache"
    traceback: str

    def short(self) -> str:
        last = self.traceback.splitlines()[-1] if self.traceback else ""
        return f"[{self.phase}] {self.cfg.label()}  ::  {last}"


@dataclass
class VerifyReport:
    """Result of a verification sweep.

    A ``Config`` lands in :attr:`passed` only if **both** the indexer and
    cache phases (when enabled) compiled cleanly. If either phase raises,
    the ``Config`` is recorded in :attr:`failed` together with the phase
    name so you can tell at a glance which half of the pipeline is
    broken.
    """
    passed: List[Config] = field(default_factory=list)
    failed: List[Failure] = field(default_factory=list)
    skipped: List[Tuple[Config, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed) + len(self.skipped)

    @property
    def ok(self) -> bool:
        return not self.failed

    def failures_by_phase(self) -> Dict[str, List[Failure]]:
        buckets: Dict[str, List[Failure]] = {}
        for f in self.failed:
            buckets.setdefault(f.phase, []).append(f)
        return buckets

    def summary(self, *, max_failures: int = 10) -> str:
        lines = [
            f"vFlow compile verification: "
            f"{len(self.passed)}/{self.total} configs passed "
            f"({len(self.failed)} failed, {len(self.skipped)} skipped)."
        ]
        if self.failed:
            buckets = self.failures_by_phase()
            for phase, items in buckets.items():
                lines.append(f"  {phase}: {len(items)} failure(s)")
                for f in items[:max_failures]:
                    lines.append(f"    - {f.short()}")
                if len(items) > max_failures:
                    lines.append(f"    ... ({len(items) - max_failures} more)")
        if self.skipped:
            lines.append(f"  Skipped: {len(self.skipped)}")
            for cfg, reason in self.skipped[:max_failures]:
                lines.append(f"    - {cfg.label()}  ::  {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fake-context builders
# ---------------------------------------------------------------------------

def _blank_ctx(ctx):
    """Zero every slot so attribute access always returns something."""
    for slot in ctx.__slots__:
        object.__setattr__(ctx, slot, None)
    return ctx


def _make_indexer_ctx(
    *,
    cfg: Config,
    max_num_pages_per_request: int,
    max_new_tokens_per_batch: int,
    cache_dir: str,
    sparse_attention_name: str,
    tensor_device: str = "cpu",
    vortex_attention_backend: str = "flashinfer",
    vortex_impl_backend: str = "triton",
    vortex_use_tensor_core: bool = False,
) -> IndexerContext:
    """Build an indexer ``Context`` honoring the invariants from ``create()``.

    Invariants enforced (matching :meth:`Context.create`):

    - ``num_blocks_per_page = page_size // block_size``
    - ``workload_chunk_size % num_blocks_per_page == 0``
    - ``num_pages_per_workload = workload_chunk_size // num_blocks_per_page``
    - ``max_num_pages = max_num_pages_per_request * B * num_kv_heads``
    - ``max_num_blocks = max_num_pages * num_blocks_per_page``
    - ``max_num_blocks_per_request = max_num_pages_per_request * num_blocks_per_page``
    - ``max_num_workloads = max_num_blocks // workload_chunk_size + B * num_kv_heads``

    Tensor-valued fields (``*_kv_indices``, ``*_kv_indptr``,
    ``kv_last_page_len``, ``winfo_*``) are backed by small **CPU**
    ``torch.Tensor`` objects so that any user-side ``profile()`` that
    reads them (e.g. ``.shape``, ``.dtype``) works, while avoiding any
    CUDA allocation.
    """
    assert cfg.page_size % cfg.block_size == 0
    assert cfg.workload_chunk_size % cfg.num_blocks_per_page == 0

    ctx = _blank_ctx(IndexerContext())

    # --- head / shape ---
    ctx.group_size = cfg.G
    ctx.num_kv_heads = cfg.num_kv_heads
    ctx.num_qo_heads = cfg.num_qo_heads
    ctx.head_dim = cfg.D

    # --- hardware / paging ---
    ctx.num_sms = 132
    ctx.page_size = cfg.page_size
    ctx.block_size = cfg.block_size
    ctx.num_blocks_per_page = cfg.num_blocks_per_page
    ctx.num_pages_per_workload = cfg.num_pages_per_workload
    ctx.workload_chunk_size = cfg.workload_chunk_size

    # --- capacity model (mirrors Context.create) ---
    ctx.max_num_pages_per_request = max_num_pages_per_request
    ctx.max_num_pages = max_num_pages_per_request * cfg.B * cfg.num_kv_heads
    ctx.max_num_blocks = ctx.max_num_pages * cfg.num_blocks_per_page
    ctx.max_num_blocks_per_request = max_num_pages_per_request * cfg.num_blocks_per_page
    # Mirror Context.create's trtllm 128-bit alignment for parity with
    # what the runtime will actually allocate (see the comment in
    # ``indexer/context.py``). The verify path is CPU-only smoke test
    # so the only visible effect is a slightly wider ``dense_block_tables``
    # row, which is harmless.
    if vortex_attention_backend == "trtllm" and ctx.max_num_blocks_per_request % 4 != 0:
        ctx.max_num_blocks_per_request = (
            (ctx.max_num_blocks_per_request + 3) // 4 * 4
        )
    ctx.max_num_workloads = (
        ctx.max_num_blocks // max(1, cfg.workload_chunk_size)
        + cfg.B * cfg.num_kv_heads
    )

    # --- misc indexer knobs ---
    ctx.max_bs = cfg.B
    ctx.side_effect_op_ids = []
    ctx.topk_val = 256
    ctx.topk_ratio = 0.0
    ctx.block_reserved_bos = 2
    ctx.block_reserved_eos = 2
    ctx.vortex_dtype = torch.bfloat16
    # ``ctx.vortex_attention_backend`` is overridden later by the caller
    # (see ``_make_indexer_ctx`` callers) — default to flashinfer so any
    # caller that doesn't pin a backend still gets the historical
    # CSR/flashinfer codegen path.
    ctx.vortex_attention_backend = vortex_attention_backend

    # --- per-forward-batch state — own MetaData with small CPU tensors so
    # that any user-side ``profile()`` that reads them (.shape / .dtype)
    # works without a CUDA allocation. Honours ctx.vortex_attention_backend.
    ctx.metadata = IndexerMetaData.preallocate(ctx, device=tensor_device)
    ctx.metadata.set_batch_size(cfg.B)

    # --- graph + codegen scratch ---
    ctx.tensor_list = []
    ctx.op_list = []
    ctx.output_tensor_to_op_list = []
    ctx.op_to_input_tensor_list = []
    ctx.op_to_output_tensor_list = []
    ctx.tensor_id_to_tensor_name_map = {}
    # Entry-point query arg name(s); MHA default (matches Context.__init__).
    # _blank_ctx() nulled this slot, so set it or codegen's interface.py trips
    # on `list(None)`. MLA verify flows would override with the absorbed pair.
    ctx.query_arg_names = ["q"]
    ctx.compilation_header_lines = []
    ctx.auxilary_func_def_lines = []
    ctx.sparse_attention_name = sparse_attention_name
    ctx.impl_backend = vortex_impl_backend
    ctx.use_tensor_core = vortex_use_tensor_core
    ctx.compilation_cache_dir = cache_dir

    # --- lifecycle ---
    ctx._aux_total_bytes = 0
    ctx._aux_total_flops = 0
    ctx.mode = Mode.profile
    ctx._created = True
    return ctx


def _make_cache_ctx(
    *,
    cfg: Config,
    max_new_tokens_per_batch: int,
    cache_dir: str,
    sparse_attention_name: str,
) -> CacheContext:
    """Build a cache ``Context`` mirroring :meth:`cache.Context.create`."""
    ctx = _blank_ctx(CacheContext())

    # Graph + codegen scratch
    ctx.tensor_list = []
    ctx.op_list = []
    ctx.output_tensor_to_op_list = []
    ctx.op_to_input_tensor_list = []
    ctx.op_to_output_tensor_list = []
    ctx.tensor_id_to_tensor_name_map = {}
    ctx.compilation_header_lines = []
    ctx.auxilary_func_def_lines = []
    ctx.sparse_attention_name = sparse_attention_name
    ctx.impl_backend = "triton"
    ctx.compilation_cache_dir = cache_dir

    # Page / block
    ctx.page_size = cfg.page_size
    ctx.block_size = cfg.block_size
    ctx.num_blocks_per_page = cfg.num_blocks_per_page
    # Capacity: one page per request per head is enough for codegen
    ctx.total_num_pages = cfg.B * cfg.num_kv_heads
    ctx.total_num_blocks = ctx.total_num_pages * cfg.num_blocks_per_page

    # Model
    ctx.head_num = cfg.num_kv_heads
    ctx.head_dim = cfg.D
    ctx.max_new_tokens_per_batch = max_new_tokens_per_batch
    ctx.num_sms = 132

    # Intermediate-tensor dtype
    ctx.vortex_dtype = torch.bfloat16

    # Lifecycle
    ctx._aux_total_bytes = 0
    ctx._aux_total_flops = 0
    ctx.mode = Mode.profile
    ctx._created = True
    return ctx


def _seed_indexer_tensors(
    ctx: IndexerContext,
    *,
    cfg: Config,
    q_dtype: torch.dtype,
    cache_meta_info: Dict[str, Tuple[Tuple[int, int], torch.dtype]],
    device: str,
) -> Tuple[vTensor, vTensor, Dict[str, vTensor]]:
    """Build ``q`` (tid 0), ``o`` (tid 1), and cache vTensors in the indexer view."""
    tid = 0
    q = vTensor(
        shape=(cfg.B, cfg.G, cfg.D),
        dtype=q_dtype, device=device,
        _format=FORMAT.BATCHED, tensor_id=tid,
    )
    ctx.tensor_list.append(q)
    ctx.output_tensor_to_op_list.append(None)
    ctx.tensor_id_to_tensor_name_map[tid] = "q"
    tid += 1

    o = vTensor(
        shape=(cfg.B, 1, 1),
        dtype=torch.int32, device=device,
        _format=FORMAT.RAGGED, tensor_id=tid,
    )
    ctx.tensor_list.append(o)
    ctx.output_tensor_to_op_list.append(None)
    ctx.tensor_id_to_tensor_name_map[tid] = "o"
    tid += 1

    # Cache entries — indexer view is ``[S_pack, r, c]`` (PAGED).
    cache: Dict[str, vTensor] = {}
    for key, ((r, c), dt) in cache_meta_info.items():
        ct = vTensor(
            shape=(ctx.max_num_blocks, r, c),
            dtype=dt, device=device,
            _format=FORMAT.PAGED, tensor_id=tid,
        )
        ctx.tensor_list.append(ct)
        ctx.output_tensor_to_op_list.append(None)
        ctx.tensor_id_to_tensor_name_map[tid] = f"cache[{key!r}]"
        cache[key] = ct
        tid += 1

    return q, o, cache


def _seed_cache_tensors(
    ctx: CacheContext,
    *,
    cfg: Config,
    cache_meta_info: Dict[str, Tuple[Tuple[int, int], torch.dtype]],
    device: str,
) -> Dict[str, vTensor]:
    """Build cache vTensors in the cache-update (batch-major PAGED) view."""
    cache: Dict[str, vTensor] = {}
    for tid, (key, ((r, c), dt)) in enumerate(cache_meta_info.items()):
        ct = vTensor(
            shape=(cfg.B * cfg.num_kv_heads, r, c),
            dtype=dt, device=device,
            _format=FORMAT.PAGED, tensor_id=tid,
        )
        ctx.tensor_list.append(ct)
        ctx.output_tensor_to_op_list.append(None)
        ctx.tensor_id_to_tensor_name_map[tid] = f"cache[{key!r}]"
        cache[key] = ct
    return cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_flow_compilable(
    flow: vFlow,
    *,
    B: int = 2,
    num_kv_heads: int = 4,
    G_values: Sequence[int] = DEFAULT_G_VALUES,
    D_values: Sequence[int] = DEFAULT_D_VALUES,
    block_sizes: Sequence[int] = DEFAULT_BLOCK_SIZES,
    page_block_ratios: Sequence[int] = DEFAULT_PAGE_BLOCK_RATIOS,
    pages_per_workload_values: Sequence[int] = DEFAULT_PAGES_PER_WORKLOAD,
    max_page_size: int = 256,
    max_num_pages_per_request: int = 512,
    max_new_tokens_per_batch: int = 4096,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
    q_data_type: torch.dtype = torch.bfloat16,
    intermediate_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0",
    cache_dir: Optional[str] = None,
    verify_indexer: bool = True,
    verify_cache: bool = True,
    verbose: bool = False,
    vortex_attention_backend: str = "flashinfer",
    vortex_impl_backend: str = "triton",
    vortex_use_tensor_core: bool = False,
    mla_dims: Optional[Tuple[int, int]] = None,
) -> VerifyReport:
    """Run compile-only verification over a configuration sweep.

    The sweep honors the invariants enforced inside
    :meth:`Context.create`:

      * ``num_blocks_per_page = page_size // block_size``
        (configs where ``page_size`` isn't a multiple of ``block_size`` are skipped).
      * ``workload_chunk_size = num_pages_per_workload * num_blocks_per_page``
        (so ``workload_chunk_size`` is always a multiple of ``num_blocks_per_page``).
      * ``num_qo_heads = G * num_kv_heads`` (GQA).
      * ``max_num_pages = max_num_pages_per_request * B * num_kv_heads``.

    Parameters
    ----------
    flow : vFlow
        The sparse-attention module to verify.
    B : int
        Batch size used for synthesizing ``q`` (``q.shape == (B, G, D)``).
    num_kv_heads : int
        Number of KV heads (per group).
    G_values, D_values, block_sizes, page_block_ratios :
        Grid axes for ``G``, head dim ``D``, block size, and
        ``page_size / block_size`` ratio.
    pages_per_workload_values :
        Sweep over ``num_pages_per_workload``. Defaults to ``(1, 2)`` so
        both the single-page fast path and the multi-page path get
        exercised.
    max_page_size :
        Upper bound on ``page_size`` (computed as ``block * ratio``).
    max_num_pages_per_request :
        Per-request page capacity used to size ``max_num_pages`` /
        ``max_num_blocks`` and all the winfo/index tensors. Keep small
        to avoid CPU-side alloc pressure in the sweep.
    cache_dir :
        Where the compiler writes the generated ``.py`` files. Defaults
        to a fresh temp dir.
    """
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="vortex_verify_")
    os.makedirs(cache_dir, exist_ok=True)

    report = VerifyReport()

    for G in G_values:
        for D in D_values:
            for block_size in block_sizes:
                for ratio in page_block_ratios:
                    page_size = block_size * ratio
                    if page_size > max_page_size:
                        continue
                    num_blocks_per_page = page_size // block_size

                    for nppw in pages_per_workload_values:
                        workload_chunk_size = nppw * num_blocks_per_page
                        cfg = Config(
                            B=B, G=G, D=D,
                            num_kv_heads=num_kv_heads,
                            block_size=block_size,
                            page_size=page_size,
                            num_blocks_per_page=num_blocks_per_page,
                            num_pages_per_workload=nppw,
                            workload_chunk_size=workload_chunk_size,
                        )

                        # --- initialize flow (may itself fail for a config) ---
                        try:
                            if mla_dims is not None:
                                # MLA flows take (kv_lora_rank, qk_rope_head_dim)
                                # instead of a scalar head_dim; D is the fused
                                # latent dim used to synthesize q / cache tensors.
                                flow.initialize(
                                    block_size=block_size,
                                    kv_lora_rank=mla_dims[0],
                                    qk_rope_head_dim=mla_dims[1],
                                    kv_cache_dtype=kv_cache_dtype,
                                    q_data_type=q_data_type,
                                    intermediate_dtype=intermediate_dtype,
                                )
                            else:
                                flow.initialize(
                                    block_size=block_size,
                                    head_dim=D,
                                    kv_cache_dtype=kv_cache_dtype,
                                    q_data_type=q_data_type,
                                    intermediate_dtype=intermediate_dtype,
                                )
                            cache_meta = flow.get_cache_meta_info()
                        except Exception:
                            tb = traceback.format_exc()
                            report.failed.append(Failure(cfg, "initialize", tb))
                            if verbose:
                                print(f"  [FAIL:init] {cfg.label()}")
                                print(tb)
                            continue

                        # --- indexer + cache phases, tracked independently ---
                        phase_error: Optional[Failure] = None

                        if verify_indexer:
                            try:
                                _compile_indexer_once(
                                    flow, cfg,
                                    q_data_type=q_data_type,
                                    cache_meta=cache_meta,
                                    device=device,
                                    cache_dir=cache_dir,
                                    max_num_pages_per_request=max_num_pages_per_request,
                                    max_new_tokens_per_batch=max_new_tokens_per_batch,
                                    vortex_dtype=intermediate_dtype,
                                    vortex_attention_backend=vortex_attention_backend,
                                    vortex_impl_backend=vortex_impl_backend,
                                    vortex_use_tensor_core=vortex_use_tensor_core,
                                )
                            except Exception:
                                phase_error = Failure(
                                    cfg, "indexer", traceback.format_exc()
                                )

                        if phase_error is None and verify_cache:
                            try:
                                _compile_cache_once(
                                    flow, cfg,
                                    cache_meta=cache_meta,
                                    device=device,
                                    cache_dir=cache_dir,
                                    max_new_tokens_per_batch=max_new_tokens_per_batch,
                                    vortex_dtype=intermediate_dtype,
                                )
                            except Exception:
                                phase_error = Failure(
                                    cfg, "cache", traceback.format_exc()
                                )

                        if phase_error is None:
                            report.passed.append(cfg)
                            if verbose:
                                print(f"  [OK]   {cfg.label()}")
                        else:
                            report.failed.append(phase_error)
                            if verbose:
                                print(f"  [FAIL:{phase_error.phase}] {cfg.label()}")
                                print(phase_error.traceback)

    return report


def _compile_indexer_once(
    flow: vFlow,
    cfg: Config,
    *,
    q_data_type: torch.dtype,
    cache_meta: Dict[str, Tuple[Tuple[int, int], torch.dtype]],
    device: str,
    cache_dir: str,
    max_num_pages_per_request: int,
    max_new_tokens_per_batch: int,
    vortex_dtype: torch.dtype = torch.bfloat16,
    vortex_attention_backend: str = "flashinfer",
    vortex_impl_backend: str = "triton",
    vortex_use_tensor_core: bool = False,
) -> None:
    name = f"verify_idx_{uuid.uuid4().hex[:8]}"
    ctx = _make_indexer_ctx(
        cfg=cfg,
        max_num_pages_per_request=max_num_pages_per_request,
        max_new_tokens_per_batch=max_new_tokens_per_batch,
        cache_dir=cache_dir,
        sparse_attention_name=name,
        vortex_attention_backend=vortex_attention_backend,
        vortex_impl_backend=vortex_impl_backend,
        vortex_use_tensor_core=vortex_use_tensor_core,
    )
    ctx.vortex_dtype = vortex_dtype
    q, o, cache = _seed_indexer_tensors(
        ctx,
        cfg=cfg,
        q_dtype=q_data_type,
        cache_meta_info=cache_meta,
        device=device,
    )
    flow.forward_indexer(q, o, cache, ctx)
    compile_indexer(ctx)


def _compile_cache_once(
    flow: vFlow,
    cfg: Config,
    *,
    cache_meta: Dict[str, Tuple[Tuple[int, int], torch.dtype]],
    device: str,
    cache_dir: str,
    max_new_tokens_per_batch: int,
    vortex_dtype: torch.dtype = torch.bfloat16,
) -> None:
    name = f"verify_cache_{uuid.uuid4().hex[:8]}"
    ctx = _make_cache_ctx(
        cfg=cfg,
        max_new_tokens_per_batch=max_new_tokens_per_batch,
        cache_dir=cache_dir,
        sparse_attention_name=name,
    )
    ctx.vortex_dtype = vortex_dtype
    cache = _seed_cache_tensors(
        ctx, cfg=cfg, cache_meta_info=cache_meta, device=device,
    )
    # ``loc`` is a real small int32 CPU tensor — same role as at runtime.
    loc = torch.zeros((1,), dtype=torch.int32, device="cpu")
    flow.forward_cache(cache, loc, ctx)
    compile_cache(ctx)


__all__ = [
    "Config",
    "Failure",
    "VerifyReport",
    "verify_flow_compilable",
    "DEFAULT_G_VALUES",
    "DEFAULT_D_VALUES",
    "DEFAULT_BLOCK_SIZES",
    "DEFAULT_PAGE_BLOCK_RATIOS",
    "DEFAULT_PAGES_PER_WORKLOAD",
]


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def _main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: ``python -m vortex_torch.flow.verify <flow_name> [options]``.

    Resolves ``flow_name`` through :func:`vortex_torch.flow.build_vflow`,
    runs the default-grid sweep, prints the summary, and returns 0 on
    success / 1 on any failure (suitable for CI).
    """
    import argparse
    from .loader import build_vflow

    parser = argparse.ArgumentParser(
        prog="python -m vortex_torch.flow.verify",
        description="Compile-only sweep for a registered vFlow implementation.",
    )
    parser.add_argument("flow", help="Registry name (e.g. 'gqa_quest_sparse_attention')")
    parser.add_argument("--B", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--num-kv-heads", type=int, default=4,
                        help="Number of KV heads (default: 4)")
    parser.add_argument("--max-page-size", type=int, default=256,
                        help="Skip configs whose page_size exceeds this (default: 256)")
    parser.add_argument("--max-num-pages-per-request", type=int, default=512,
                        help="Per-request page capacity (default: 512)")
    parser.add_argument("--no-indexer", action="store_true",
                        help="Skip the indexer compile half")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip the cache compile half")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Per-config pass/fail output")
    parser.add_argument("--kv-cache-dtype", default="bfloat16",
                        help="Cache dtype: bfloat16 | fp8_e5m2 | fp8_e4m3fn")
    parser.add_argument("--vortex-dtype", default="bfloat16",
                        help="Intermediate-tensor dtype: bfloat16 | float16 | float32 | fp8_e5m2 | fp8_e4m3fn")
    parser.add_argument("--cache-dir", default=None,
                        help="Where to write generated .py files (default: tempdir)")
    args = parser.parse_args(argv)

    try:
        kv_cache_dtype = resolve_dtype(args.kv_cache_dtype)
    except ValueError as e:
        raise SystemExit(f"--kv-cache-dtype: {e}")
    try:
        intermediate_dtype = resolve_dtype(args.vortex_dtype)
    except ValueError as e:
        raise SystemExit(f"--vortex-dtype: {e}")

    flow = build_vflow(args.flow)
    report = verify_flow_compilable(
        flow,
        B=args.B,
        num_kv_heads=args.num_kv_heads,
        max_page_size=args.max_page_size,
        max_num_pages_per_request=args.max_num_pages_per_request,
        kv_cache_dtype=kv_cache_dtype,
        intermediate_dtype=intermediate_dtype,
        verify_indexer=not args.no_indexer,
        verify_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
    )
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
