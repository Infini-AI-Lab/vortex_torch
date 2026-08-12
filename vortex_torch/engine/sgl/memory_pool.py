"""
Copyright 2025 Zhuoming Chen
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
from contextlib import nullcontext
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.utils import (
    debug_timing,
    is_cuda
)

from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.cache import (
    Context,
    set_kv_buffer_fp8_e4m3_launcher,
    set_kv_buffer_fp8_e5m2_launcher,
    set_kv_buffer_launcher,
)
from vortex_torch.cache.compiler.compile import compile as compile_cache
from vortex_torch.flow import vFlow
from .config import cfg as _vortex_cfg
from .host_kv import HostKVCache
logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024

#: Share of currently-free HBM the host-KV staging pools may take, summed over
#: layers. Deliberately well under 1.0: the aux cache, cuda-graph pools, attention
#: workspaces and activations are all still to be allocated when the pool is
#: built, and a staging pool that consumed the remainder would OOM the engine
#: later at a much more confusing point. Raise it via ``host_kv_pool_blocks`` if a
#: workload is dominated by staging misses.
_HOST_KV_STAGING_FRACTION = 0.25
_is_cuda = is_cuda()

_SET_KV_LAUNCHERS = {
    torch.bfloat16: set_kv_buffer_launcher,
    torch.float8_e4m3fn: set_kv_buffer_fp8_e4m3_launcher,
    torch.float8_e5m2: set_kv_buffer_fp8_e5m2_launcher,
}

"""
Vortex Sparse Attention Memory pool.

In addition to Memory Pool in the original SGLang
We 
1) maintain auxilary cache tensor objects for every page.
2) internally treat each KV head as a request (as they may have different sparse patterns), 
then we interpret external auguments to the physical address
"""

class VortexCachePool(KVCache):

    # Vortex stores K/V in a block-interleaved layout (see
    # vortex_torch/cache/triton_kernels/set_kv.py — position is mapped to
    # ``(token//page) * (page * num_kv_head) + head * page + token%page``).
    # The fused-set-kv-buffer kernel that ships with sglang assumes the
    # standard token-major layout and would silently corrupt this pool,
    # producing 0% accuracy or illegal-memory-access in models that route
    # KV writes through fused RoPE (e.g. Qwen3-MoE). Opt out so
    # ``models/utils.py::enable_fused_set_kv_buffer`` returns False here.
    supports_fused_set_kv_buffer = False

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        sparse_attention: vFlow,
        model_runner,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim

        # for disagg with nvlink
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        self.num_pages = ((self.size + self.page_size) * self.head_num + self.page_size - 1) // self.page_size + 1
        
        self.sparse_attention = sparse_attention
        self.ctx = Context()
        self.block_size = model_runner.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size for block-sparse attention"
        self.num_blocks_per_page = self.page_size // self.block_size

        # Host-resident KV: K/V in pinned host memory, aux cache on the GPU,
        # selected blocks fetched per step (see engine/sgl/host_kv.py).
        self.vortex_cfg = _vortex_cfg(model_runner)
        self.host_kv_gb = float(getattr(self.vortex_cfg, "host_kv_gb", 0.0) or 0.0)
        self.host_kv = self.host_kv_gb > 0.0
        self.host_kv_policy = getattr(self.vortex_cfg, "host_kv_policy", "lru") or "lru"
        self.host_kv_caches: List["HostKVCache"] = []
        # Per-step snapshot of the prefill block ids (see snapshot_prefix).
        self._prefix_src: Optional[torch.Tensor] = None
        self._prefix_n = 0

        if self.host_kv:
            # ``size`` is decided upstream, before this pool exists: sglang turns a
            # KV byte budget into a token count via ``integration.kv_cell_size``,
            # then sizes the allocator and ``req_to_token_pool`` from the same
            # number. Under host KV that hook already excludes K/V from the HBM
            # cell size (only the aux fields are device-resident), so ``size`` is
            # correct on arrival and must NOT be rewritten here — doing so would
            # leave the allocator and req pool sized for a different token count,
            # and the runner is deliberately read-only during pool construction
            # (``compat/runner_view``) to prevent exactly that.
            #
            # What is enforced here is the user's ``host_kv_gb`` as a *cap*: the
            # host buffer is bounded by what they asked for, whatever HBM-derived
            # budget upstream computed.
            budget = self._host_kv_token_budget()
            if budget < self.size:
                logger.info(
                    "vortex host KV: clamping %d tokens to %d to fit host_kv_gb=%.2f",
                    self.size, budget, self.host_kv_gb,
                )
                self.size = budget
                self.num_pages = (
                    ((self.size + self.page_size) * self.head_num + self.page_size - 1)
                    // self.page_size + 1
                )

        self._create_buffers(model_runner)
        self._compile(model_runner)
        self.layer_transfer_counter = None
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = self.device_module.Stream() if _is_cuda else None
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        cache_size = self.get_cache_size_bytes()
        
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, Cache size: {cache_size / GB:.2f} GB"
        )
        
        self.mem_usage = cache_size / GB
        assert self.store_dtype in [torch.bfloat16, torch.uint8], f"Unsupported store dtype {self.store_dtype} for KV cache"
        if self.dtype not in _SET_KV_LAUNCHERS:
            raise ValueError(f"Unsupported dtype {self.dtype} for KV cache")
        self.set_kv_buffer_func = _SET_KV_LAUNCHERS[self.dtype]
        
    def _compile(self, model_runner) -> None:
        """Trace the sparse-attention cache flow on zero-sized dummies and compile it."""
        self.ctx.create(self, model_runner)
        self.ctx.profile()

        def register(vt, name: str) -> None:
            self.ctx.tensor_list.append(vt)
            self.ctx.output_tensor_to_op_list.append(None)
            self.ctx.tensor_id_to_tensor_name_map[vt.tensor_id] = name

        with torch.no_grad():
            loc_dummy = torch.empty((0,), dtype=torch.int64, device=self.device)
            cache_dummy = {}
            for i, (name, (shape, cache_dtype)) in enumerate(self.cache_meta_info.items()):
                vt = as_vtensor(
                    torch.zeros((0, shape[0], shape[1]), dtype=cache_dtype, device=self.device),
                    FORMAT.PAGED,
                    tensor_id=i,
                )
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")
            self.sparse_attention.forward_cache(cache=cache_dummy, loc=loc_dummy, ctx=self.ctx)

        self.compiled_cache = compile_cache(self.ctx)()
        self.ctx.summary()
        self.ctx.execute()



    def _create_buffers(self, model_runner):

        self.cache_meta_info = self.sparse_attention.get_cache_meta_info()
        num_blocks = self.num_pages * self.num_blocks_per_page

        if self.host_kv:
            self._create_host_kv_buffers(num_blocks, model_runner)
            return

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.cache = [
                    {
                        cache_name:  torch.zeros(
                                (num_blocks, cache_shape[0], cache_shape[1]),
                                dtype=cache_dtype,
                                device=self.device,
                            )

                        for (cache_name, (cache_shape, cache_dtype)) in self.cache_meta_info.items()
                    }

                    for _ in range(self.layer_num)
                ]

    def _create_host_kv_buffers(self, num_blocks: int, model_runner):
        """Split the cache: ``k``/``v`` into pinned host memory, the rest on GPU.

        The split is per-*key* inside each layer's dict, not per layer, because
        the two halves are read by different consumers with opposite access
        patterns:

        * the **auxiliary** fields (centroids / envelopes / Save state) are read
          by the indexer for *every* cached block on every step — that is the
          whole point of block scoring — so streaming them would move more data
          than the KV it is trying to avoid. They stay in HBM;
        * the **KV** blocks are read only for the blocks the indexer *selects*,
          a small fraction of the context, so they can live on the host and be
          fetched on demand.

        ``self.cache[layer]["k"]`` therefore holds a *pinned host* tensor after
        this, and every device consumer must go through
        :meth:`fetch_kv` instead of reading it directly.
        """
        self.cache = []
        self.host_kv_caches = []
        pool_blocks = self._host_kv_pool_blocks(model_runner)

        for _ in range(self.layer_num):
            layer = {}
            for name, (shape, dtype) in self.cache_meta_info.items():
                if name in ("k", "v"):
                    # pin_memory=True is mandatory, not an optimisation: a kernel
                    # cannot read pageable host memory, and the addresses here are
                    # dereferenced on the device.
                    layer[name] = torch.zeros(
                        (num_blocks, shape[0], shape[1]),
                        dtype=dtype, device="cpu", pin_memory=True,
                    )
                else:
                    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                        layer[name] = torch.zeros(
                            (num_blocks, shape[0], shape[1]),
                            dtype=dtype, device=self.device,
                        )
            self.cache.append(layer)
            self.host_kv_caches.append(
                HostKVCache(layer["k"], layer["v"], pool_blocks, self.device,
                            policy=self.host_kv_policy)
            )

        host_gb = sum(
            t.element_size() * t.numel()
            for layer in self.cache for n, t in layer.items() if n in ("k", "v")
        ) / GB
        dev_gb = (
            sum(t.element_size() * t.numel()
                for layer in self.cache for n, t in layer.items() if n not in ("k", "v"))
            + sum(c.nbytes_device() for c in self.host_kv_caches)
        ) / GB
        logger.info(
            f"vortex host KV cache: {host_gb:.2f} GB pinned host (K/V for "
            f"{num_blocks} blocks x {self.layer_num} layers), {dev_gb:.2f} GB on "
            f"GPU (aux cache + {pool_blocks}-block staging pool per layer)"
        )

    def _host_kv_pool_blocks(self, model_runner) -> int:
        """Size the per-layer GPU staging pool.

        Sized from the **per-step selection budget**, *not* the context length.
        This distinction is the whole feature: a row of the block table is
        ``max_num_blocks_per_request`` wide (the full context in blocks), but only
        the first ``sparse_seqlens[row]`` entries are live, and that count is the
        sparse budget — ``topk_val + bos + eos``, a few dozen blocks. Sizing the
        pool on the row *width* would put the entire KV back in HBM and buy
        nothing.

        ``topk_ratio > 0`` makes the budget grow with context
        (``max(static, cached_blocks * ratio)``, see ``DEFAULT_SCHEDULE_POLICY``),
        so that case is accounted for and then clamped to the row width — beyond
        which a row cannot ask for more.
        """
        override = int(getattr(self.vortex_cfg, "host_kv_pool_blocks", 0) or 0)
        if override > 0:
            return override

        sa = model_runner.server_args
        rows = int(model_runner.req_to_token_pool.size) * self.head_num
        static = (
            int(sa.vortex_topk_val)
            + int(sa.vortex_block_reserved_bos)
            + int(sa.vortex_block_reserved_eos)
        )
        max_topk = getattr(sa, "vortex_max_topk_val", None)
        if max_topk:
            static = max(static, int(max_topk) + int(sa.vortex_block_reserved_bos)
                         + int(sa.vortex_block_reserved_eos))

        max_blocks_per_req = self._max_blocks_per_request(model_runner)
        ratio = float(getattr(sa, "vortex_topk_ratio", 0.0) or 0.0)
        per_row = min(max(static, int(max_blocks_per_req * ratio)), max_blocks_per_req)

        want = HostKVCache.required_capacity(rows, per_row)
        # ``want`` covers every schedulable request at full budget simultaneously,
        # which is unaffordable: sglang sets max_running_requests from the token
        # budget, so ``rows`` is in the thousands and ``want`` reaches 144 GB of
        # staging on a 36-layer model (measured — it OOMs). Clamp it to a share of
        # the HBM freed by hosting K/V. The pool is a cache, so the clamp trades hit
        # rate, not correctness; overflow is counted and reported.
        block_numel = self.block_size * self.head_dim
        elt = torch._utils._element_size(self.dtype)
        budget = int(_HOST_KV_STAGING_FRACTION * self._free_device_bytes())
        afford = HostKVCache.affordable_capacity(
            block_numel, elt, self.layer_num, budget,
        )
        # Never go below one full-budget request per KV head, or a single-sequence
        # decode — the common case — would overflow on every step.
        floor = HostKVCache.required_capacity(self.head_num, per_row)
        cap = max(floor, min(want, afford))
        if cap < want:
            # Report the concurrency the pool actually keeps fast, not just the
            # byte figures. ``required_capacity``'s 2x factor is a *performance*
            # threshold as well as a safety one: at 1x demand every slot is pinned
            # and each miss probes O(capacity) slots, measured 32x slower (3.24 vs
            # 0.10 ms/step). So the useful statement is how many concurrent
            # requests stay on the fast side of that, which is what an operator
            # needs to size `host_kv_pool_blocks` against.
            fast_rows = max(1, cap // (2 * max(1, per_row)))
            logger.info(
                "vortex host KV staging pool: %d blocks/layer = %.1f GB across %d "
                "layers (worst case %d blocks / %.1f GB for all %d rows). Keeps "
                "~%d concurrent rows (~%d requests x %d KV heads) miss-bound; "
                "beyond that misses cost extra probing. Raise "
                "vortex_host_kv_pool_blocks if decode is staging-bound.",
                cap, cap * block_numel * elt * 2 * self.layer_num / GB,
                self.layer_num,
                want, want * block_numel * elt * 2 * self.layer_num / GB, rows,
                fast_rows, max(1, fast_rows // max(1, self.head_num)), self.head_num,
            )
        return cap

    def _free_device_bytes(self) -> int:
        """Free HBM right now, as the budget the staging pool is carved from."""
        free, _total = torch.cuda.mem_get_info()
        return int(free)

    def _host_kv_token_budget(self) -> int:
        """Tokens that fit in ``host_kv_gb`` of pinned host memory.

        Only K and V are hosted, so the budget is set by their bytes per token:
        ``2 (K,V) * layers * head_num * head_dim * elt``. The auxiliary fields do
        not appear — they stay in HBM, and folding them in here would silently
        shrink the host budget to pay for device memory.

        Rounded **down** to a whole page: a partial page cannot be allocated, and
        rounding up would exceed the size the user asked for.
        """
        elt = torch._utils._element_size(self.dtype)
        bytes_per_token = 2 * self.layer_num * self.head_num * self.head_dim * elt
        tokens = int(self.host_kv_gb * GB) // bytes_per_token
        tokens = (tokens // self.page_size) * self.page_size
        if tokens < self.page_size:
            raise ValueError(
                f"host_kv_gb={self.host_kv_gb} is too small for this model: one "
                f"page of {self.page_size} tokens needs "
                f"{bytes_per_token * self.page_size / GB:.3f} GB of host KV "
                f"({self.layer_num} layers x {self.head_num} KV heads x "
                f"{self.head_dim} dim x 2 for K/V)."
            )
        return tokens

    @staticmethod
    def _max_blocks_per_request(model_runner) -> int:
        """Blocks in the longest request — mirrors ``Context.create``'s derivation."""
        sa = model_runner.server_args
        if sa.vortex_max_seq_lens < 0:
            seq = model_runner.model_config.context_len
        else:
            seq = sa.vortex_max_seq_lens
        return (seq + sa.vortex_block_size - 1) // sa.vortex_block_size

    def fetch_kv(
        self,
        layer_id: int,
        table: torch.Tensor,
        *,
        row_lens: Optional[torch.Tensor] = None,
        indptr: Optional[torch.Tensor] = None,
        num_rows: int,
        max_per_row: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage the blocks named by ``table``, translated to staging slots.

        Returns ``(k, v, table_to_use)``. ``table`` is not modified: the caller's
        table is shared across layers, so the translated ids go to a pool-owned
        table which the caller must pass to the attention call instead. With host
        KV off this returns the GPU buffers and the table unchanged, so the
        backends can call it unconditionally.
        """
        if not self.host_kv:
            k, v = self.get_kv_buffer(layer_id)
            return k, v, table
        return self.host_kv_caches[layer_id - self.start_layer].fetch(
            table, row_lens=row_lens, indptr=indptr,
            num_rows=num_rows, max_per_row=max_per_row,
        )

    def snapshot_prefix(self, indices: torch.Tensor, indptr: torch.Tensor,
                        num_rows: int) -> int:
        """Snapshot the prefill block ids once per step; returns the block count.

        Called from the prefill planner, before any layer runs. The snapshot is
        needed because :meth:`fetch_prefix` rewrites ``indices`` in place (the
        attention wrapper holds that exact tensor), so after the first layer the
        original ids are gone — every later layer would stage the wrong blocks.
        Taking it once per step rather than per layer also means one copy, not
        ``layer_num`` copies.

        The single host sync here is on the prefill path only, which is not
        cuda-graph captured (capture asserts ``is_decode_or_idle``), so it costs
        nothing per decode step.
        """
        if not self.host_kv:
            return 0
        n = int(indptr[num_rows].item())
        if n == 0:
            self._prefix_n = 0
            return 0
        if self._prefix_src is None or self._prefix_src.numel() < n:
            self._prefix_src = torch.empty(
                max(n, 2 * (0 if self._prefix_src is None else self._prefix_src.numel())),
                dtype=indices.dtype, device=indices.device,
            )
        self._prefix_src[:n].copy_(indices[:n])
        self._prefix_n = n
        return n

    def fetch_prefix(self, layer_id: int, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Promote the snapshotted dense prefix to device (prefill-with-prefix)."""
        if not self.host_kv:
            return self.get_kv_buffer(layer_id)
        return self.host_kv_caches[layer_id - self.start_layer].fetch_prefix(
            indices, self._prefix_src, self._prefix_n,
        )

    def host_kv_tick(self, block_tables=None) -> None:
        """Advance the staging generation once per forward step (all layers).

        Also reserves the remap buffer when given the full block table: doing it
        here (per-step planner, outside capture) keeps ``fetch`` allocation-free,
        since a lazy allocation inside a captured region becomes part of the graph.
        """
        if not self.host_kv:
            return
        for c in self.host_kv_caches:
            c.tick()
            if block_tables is not None:
                c.reserve_remap(block_tables)

    def _clear_buffers(self):
        del self.cache
       

    def get_cache_size_bytes(self) -> int:
        """Bytes of **device** memory occupied by the cache.

        Device, not total: the value feeds ``self.mem_usage``, which sglang treats
        as HBM occupancy when reporting and fitting memory. Under host KV the
        pinned K/V tensors are host-resident, so counting them would overstate GPU
        use by the entire KV cache — the opposite of what the feature does. The
        GPU-side staging pools are counted instead, since those are real HBM.
        """
        total_bytes = 0

        for layer_cache in self.cache:
            if not isinstance(layer_cache, dict):
                # Be tolerant to unexpected structures
                continue

            for name, t in layer_cache.items():
                if not torch.is_tensor(t):
                    continue
                if self.host_kv and name in ("k", "v"):
                    continue                       # host-resident; not HBM

                # Prefer accurate allocated size if available (includes padding/strides)
                try:
                    total_bytes += int(t.untyped_storage().nbytes())
                except AttributeError:
                    # Fallback: logical size in bytes
                    total_bytes += int(t.element_size() * t.numel())

        # The staging pools and their metadata are device allocations that exist
        # only in host-KV mode, and are what the hosted K/V is traded for.
        total_bytes += sum(c.nbytes_device() for c in self.host_kv_caches)
        return total_bytes

    def get_host_cache_size_bytes(self) -> int:
        """Bytes of pinned **host** memory holding K/V (0 when host KV is off)."""
        if not self.host_kv:
            return 0
        return sum(
            int(t.element_size() * t.numel())
            for layer in self.cache
            for name, t in layer.items()
            if name in ("k", "v") and torch.is_tensor(t)
        )


    def get_kv_size_bytes(self):
        """``(k_bytes, v_bytes)`` summed over layers — sglang's KV accounting.

        Upstream's MHA pools return this pair and callers add them
        (``HybridLinearKVPool.__init__`` reports ``(k+v)/GB`` as ``mem_usage``).

        Vortex's cache holds more than K/V — the flow's auxiliary per-page fields
        (centroids / envelopes / Save state). Those are real occupancy, so they
        are folded into the K side rather than dropped, keeping the reported
        total equal to :meth:`get_cache_size_bytes`. Splitting them out would
        need a K-vs-V attribution that no caller uses.
        """
        def _nbytes(t):
            try:
                return int(t.untyped_storage().nbytes())
            except AttributeError:
                return int(t.element_size() * t.numel())

        k_bytes = v_bytes = 0
        for layer in self.cache:
            for name, t in layer.items():
                if not torch.is_tensor(t):
                    continue
                if name == "v":
                    v_bytes += _nbytes(t)
                else:  # "k" plus every auxiliary field
                    k_bytes += _nbytes(t)
        return k_bytes, v_bytes
    
    # for disagg (PD disaggregation, Option B)
    def get_contiguous_buf_infos(self):
        """Per-layer (data_ptr, total_bytes, page_item_bytes) for the K then
        V buffers, consumed by the disaggregation transfer engine
        (``disaggregation/{prefill,decode}.py``) for RDMA registration +
        page-granular copy (``src = base + page_idx * item_len``).

        Vortex stores K/V **page-major** (see
        ``cache/triton_kernels/set_kv.py``):

            position = (token//page)*(page*num_kv_head) + head*page + token%page

        so one *logical* page — all ``head_num`` KV heads × ``page_size``
        tokens — is a single contiguous block. The head interleaving lives
        *inside* the page, so the transfer is page-granular exactly like the
        stock ``MHATokenToKVPool``: ``item_len`` = bytes of one full logical
        page (all heads) = ``page_size*head_num*head_dim*elt``.

        Only K/V are exported. The auxiliary per-page tensors (centroids /
        envelopes / Save fields) are **not** transferred — the decode side
        rebuilds them from the received K/V via :meth:`rebuild_aux`
        (disagg Option B; correct because no ``forward_cache`` reads an
        indexer-``Save``-accumulated field).
        """
        layers = range(self.start_layer, self.start_layer + self.layer_num)
        k_bufs = [self.cache[l - self.start_layer]["k"] for l in layers]
        v_bufs = [self.cache[l - self.start_layer]["v"] for l in layers]
        page_item_numel = self.page_size * self.head_num * self.head_dim

        ptrs, data_lens, item_lens = [], [], []
        for t in list(k_bufs) + list(v_bufs):
            ptrs.append(t.data_ptr())
            data_lens.append(t.element_size() * t.numel())
            item_lens.append(t.element_size() * page_item_numel)
        return ptrs, data_lens, item_lens

    def rebuild_aux(self, loc: torch.Tensor):
        """Decode-side (PD disagg): rebuild the per-page auxiliary cache
        (centroids / min-max envelopes, etc.) and zero the persistent
        ``Save``/``Load`` fields for the pages whose K/V was just received
        from the prefill node, by running ``forward_cache`` over ``loc`` in
        one batched pass.

        This is the same ``compiled_cache.forward`` that
        :meth:`set_kv_buffer` runs incrementally during normal decode, here
        applied once over the transferred prompt's KV locations. It is
        stateless w.r.t. decode accumulation (verified: no flow's
        ``forward_cache`` reads a ``Save``-accumulated field), so it
        reproduces the monolithic decode-start state bit-identically. Pages
        in ``layers_skip`` run dense and keep no aux, mirroring
        :meth:`set_kv_buffer`.
        """
        if loc is None or loc.numel() == 0:
            return
        loc = loc.to(torch.int64)
        for layer_id in range(self.start_layer, self.start_layer + self.layer_num):
            if layer_id in self.layers_skip:
                continue
            self.compiled_cache.forward(
                self.cache[layer_id - self.start_layer], loc, ctx=self.ctx
            )

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices):
        
        raise NotImplementedError

    def load_cpu_copy(self, kv_cache_cpu, indices):
        
        raise NotImplementedError

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        raise NotImplementedError


    @debug_timing
    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
       raise NotImplementedError

    def transfer_per_layer(self, indices, flat_data, layer_id):
        
        raise NotImplementedError


    def get_key_buffer(self, layer_id: int):
        
        return self.cache[layer_id - self.start_layer]["k"]

    def get_value_buffer(self, layer_id: int):
        
        return self.cache[layer_id - self.start_layer]["v"]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return self.cache[layer_id - self.start_layer]["k"], self.cache[layer_id - self.start_layer]["v"]

        
    def get_cache(self, layer_id: int)->Dict[str, torch.Tensor]:
        
        return self.cache[layer_id - self.start_layer]

        
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):

        assert loc.dtype == torch.int64
        # Decode context parallel writes a masked subset of the KV rows. vortex's
        # set_kv launcher writes every row in `loc`, so honouring the mask would
        # need a masked variant of the kernel. Accept the argument (upstream
        # passes it unconditionally) but refuse a real mask rather than silently
        # writing rows this rank does not own.
        assert dcp_kv_mask is None, (
            "vortex sparsity does not support decode context parallel "
            "(dcp_size > 1): its set_kv kernel has no masked write path."
        )

        # Two different layer numberings meet here, and conflating them is the
        # whole difficulty of hybrid models:
        #
        #   cache_slot — which of OUR per-layer buffers to write. On a hybrid
        #     model (some layers linear-attention / RNN) this pool covers only
        #     the full-attention layers, and the wrapping HybridLinearKVPool
        #     hands us the dense full-attention index via `layer_id_override`
        #     (global 3,7,11,... -> 0,1,2,...). Without it we would index by
        #     global id and read the wrong slot or run off the end.
        #   layer_id — the model's GLOBAL layer id, which is what
        #     `layers_skip` is expressed in (so `--vortex-layers-skip 3` names
        #     the same layer regardless of the interleave).
        layer_id = layer.layer_id
        cache_slot = (
            layer_id if layer_id_override is None else layer_id_override
        ) - self.start_layer

        # KV scales (mirror sglang's MHATokenToKVPool.set_kv_buffer): the
        # per-tensor k_scale/v_scale only matter when we down-cast the model's
        # bf16 k/v into a narrower cache dtype (fp8) — divide by the scale
        # before the fp8 launcher casts, so the stored fp8 values are the
        # quantized representation. For a bf16 cache (cache_k.dtype == self.dtype)
        # the scales are a no-op and must be IGNORED rather than asserted away:
        # fp8-*weight* models such as MiniMax-M2 still attach layer.k_scale /
        # layer.v_scale (typically the 1.0 default from
        # quantization/kv_cache.py::process_weights_after_loading) even though
        # their KV cache stays bf16. The matching dequant on the read side is
        # applied by the attention backends via layer.k_scale_float /
        # layer.v_scale_float.
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k = cache_k.div(k_scale)
            if v_scale is not None:
                cache_v = cache_v.div(v_scale)

        # Under host KV, ``cache[slot]["k"]`` is a *pinned host* tensor. The
        # launcher writes through the pointer it is given, and a kernel can write
        # pinned host memory as well as read it (verified bit-exact in both
        # directions), so the write lands where the fetch kernel will look for it.
        # No staging copy is needed and none must be added: writing to the GPU
        # pool instead would leave the host copy stale, and the pool is evictable.
        self.set_kv_buffer_func(
            self.cache[cache_slot]["k"],
            self.cache[cache_slot]["v"],
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.page_size
        )
        if layer_id in self.layers_skip:
            return
        # ``forward_cache`` reads the K/V just written to build this block's
        # auxiliary fields. It reads over the host tensor for the same reason —
        # correct, and only for the handful of tokens in ``loc``, not the whole
        # context. It also means the block whose K/V just changed may be resident
        # in the staging pool holding pre-write contents, so invalidate it.
        if self.host_kv:
            self.host_kv_caches[cache_slot].invalidate_locs(loc, self.page_size,
                                                            self.head_num)
        self.compiled_cache.forward(self.cache[cache_slot], loc, ctx=self.ctx)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate KV between slots. **Not supported, on either KV placement.**

        Upstream only calls this from the sliding-window (SWA) and
        speculative-decoding paths, and vortex already refuses both
        (``assert model_runner.sliding_window_size is None`` and the
        ``is_draft_extend`` / ``is_target_verify`` asserts in every backend). It is
        therefore unreachable, and hosting KV does not change that: the plain
        prefix-radix cache reached in normal serving never moves a page, it only
        re-references one — which is why host KV works with the radix cache
        without this.

        A host-KV implementation was written and removed rather than shipped: it
        would be dead code whose only effect is to suggest the SWA/spec paths are
        supported when the asserts above reject them earlier, and an in-place block
        copy also needs overlap handling that nothing exercises. Implement it
        alongside whichever feature first needs it, with a test that reaches it.
        """
        raise NotImplementedError(
            "vortex does not support move_kv_cache (SWA / speculative decoding "
            "paths only; both are rejected earlier by the attention backends)"
        )