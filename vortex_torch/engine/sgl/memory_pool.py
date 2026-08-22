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
import os
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

from vortex_torch.abs import as_vtensor, FORMAT, Int4Packed
from vortex_torch.cache import (
    Context,
    set_kv_buffer_fp8_e4m3_launcher,
    set_kv_buffer_fp8_e5m2_launcher,
    set_kv_buffer_launcher,
)
from vortex_torch.cache.compiler.compile import compile as compile_cache
from vortex_torch.cache.triton_kernels.int4_kv import INT4_BIAS
from vortex_torch.flow import vFlow
from vortex_torch.utils import is_int4_kv
from .config import cfg as _vortex_cfg
from .cache_policy import WAYS
from .host_kv import HostKVCache, tick_all
from .int4_arena import Int4Arena, arena_slots
from .int4_store import (
    K_SCALE, STAGE_K, STAGE_V, V_SCALE,
    compression_ratio as int4_compression_ratio, gather_unpack,
)
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
        # Read from the flow's DECLARED META, not from the config flag. The flow already applied the
        # flag when it built the meta, and the meta is what every consumer strides through -- taking
        # the flag here instead leaves a window where the pool and the flow disagree, which shows up
        # as a 2x mis-stride rather than as an error. See ``utils.is_int4_kv``.
        self.kv_int4 = is_int4_kv(sparse_attention.get_cache_meta_info())
        # Read here, not from ``self.layers_skip``: that is assigned AFTER _create_buffers, which is
        # where the INT4 buffers are sized, so reading it there gets UNSET and the pool-shaped buffer
        # is skipped for a config that needs it.
        self.layers_skip_for_int4 = list(
            getattr(model_runner.server_args, "vortex_layers_skip", None) or [])
        self.disable_radix_for_int4 = bool(
            getattr(model_runner.server_args, "disable_radix_cache", False))
        #: Which forward mode the current step is, for the fused-write choice. Prefill-safe default:
        #: the fused path is only sound when a launch writes at most one token per block, so guessing
        #: decode would corrupt the first prefill. See :meth:`int4_set_decode`.
        self._int4_is_decode = False
        #: Shared dequant scratch; reserved by :meth:`int4_reserve_scratch` before capture.
        self._int4_scratch_k: Optional[torch.Tensor] = None
        self._int4_scratch_v: Optional[torch.Tensor] = None
        self._int4_scratch_tbl: Optional[torch.Tensor] = None
        #: Pool-shaped buffers for full-context reads; see :meth:`int4_unpack_pool`.
        self._int4_pool_k: Optional[torch.Tensor] = None
        self._int4_pool_v: Optional[torch.Tensor] = None
        self._int4_pool_ids: Optional[torch.Tensor] = None
        self._int4_pool_tbl: Optional[torch.Tensor] = None
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
        # Under INT4 the write path is the staging arena, not a set_kv launcher: K/V arrive in bf16
        # and are held that way until the block completes, so there is no dtype to cast to on write.
        # ``self.dtype`` still names the MODEL's KV dtype (bf16), which is correct for every other
        # consumer, and looking up a launcher for it here would install one that writes bf16 straight
        # into the packed buffer at twice the stride.
        if not self.kv_int4:
            if self.dtype not in _SET_KV_LAUNCHERS:
                raise ValueError(f"Unsupported dtype {self.dtype} for KV cache")
            self.set_kv_buffer_func = _SET_KV_LAUNCHERS[self.dtype]
        else:
            self.set_kv_buffer_func = None
        
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
            names = list(self.cache_meta_info)
            # INT4 tags K/V with the LOGICAL width and an Int4Packed marker, so every op in the flow
            # continues to see head_dim channels and the unpack happens at the codegen load site.
            # Without this, tracing hits `Max: expected output.shape[2] == x.shape[2], got 128 vs 64`
            # -- the flow's ops are written against the logical width, and rightly so.
            int4_of = {}
            if self.kv_int4:
                for kv, per_channel, scale_name in (("k", True, K_SCALE), ("v", False, V_SCALE)):
                    int4_of[kv] = Int4Packed(names.index(scale_name), per_channel, INT4_BIAS)
            for i, (name, (shape, cache_dtype)) in enumerate(self.cache_meta_info.items()):
                logical = (shape[0], shape[1] * 2) if name in int4_of else shape
                vt = as_vtensor(
                    torch.zeros((0, logical[0], logical[1]),
                                dtype=cache_dtype, device=self.device),
                    FORMAT.PAGED,
                    tensor_id=i,
                )
                vt.int4 = int4_of.get(name)
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")
            for name, shape in self.request_cache_meta_info.items():
                # Request-domain fields join the same dict, distinguished only by their FORMAT --
                # which is exactly the property that lets one fused kernel read both domains.
                vt = as_vtensor(
                    torch.zeros((0, shape[0], shape[1]), dtype=self.dtype, device=self.device),
                    FORMAT.SLOTTED,
                    tensor_id=len(self.ctx.tensor_list),
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
        self._create_request_domain(num_blocks)

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

    def _create_request_domain(self, num_blocks: int) -> None:
        """Allocate the request-bound domain: one arena + one payload dict per layer.

        Per LAYER, not per pool, for the same reason the page-domain cache is: each layer writes its
        own K/V, so a shared arena would have layer 1's staging overwrite layer 0's while layer 0's
        block is still incomplete. That sharing is invisible in a single-layer test.

        Allocated here, during pool construction, and never grown: a first-use allocation inside a
        captured decode step becomes part of the graph. The staging payload is ``n_slots x r x c``
        and so constant in context length -- the whole reason it is a separate domain.
        """
        self.request_cache_meta_info = self.sparse_attention.get_request_cache_meta_info()
        self.int4_arenas: List["Int4Arena"] = []
        self.request_cache: List[Dict[str, torch.Tensor]] = []
        if not self.kv_int4:
            # A flow may declare request-domain fields without INT4; allocating them is the same
            # operation, it just needs no arena. Nothing does yet, so refuse rather than allocate
            # buffers that no update op would ever write.
            if self.request_cache_meta_info:
                raise NotImplementedError(
                    "this flow declares request-domain fields "
                    f"({sorted(self.request_cache_meta_info)}) but nothing drives them: the only "
                    "update op today is INT4's staging arena. Add a forward_request_cache path "
                    "alongside the flow that needs it."
                )
            return

        for _ in range(self.layer_num):
            arena = Int4Arena(num_blocks, arena_slots(self.head_num),
                              self.block_size, self.head_dim, device=self.device)
            self.int4_arenas.append(arena)
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                self.request_cache.append(
                    arena.domain.allocate_payload(
                        self.request_cache_meta_info, dtype=self.dtype, device=self.device,
                    )
                )
        # Pool-shaped dequant buffers for FULL-CONTEXT reads (a dense layer in ``layers_skip``, or a
        # prefill prefix hit) -- see :meth:`int4_unpack_pool`. One row per BLOCK, so the size tracks
        # the pool, and under INT4 the pool is ~3x larger by construction: this is the single largest
        # cost of the feature and it partly offsets the compression. It is allocated HERE, at pool
        # construction, rather than from the planner, for two reasons: reserving it during cuda-graph
        # setup OOM'd after mem_fraction_static had already claimed HBM, and it must exist before any
        # captured step touches it.
        #
        # Only allocated when something can actually perform a full-context read. Sparse-only INT4
        # (no skipped layers, radix cache off) never needs it, and paying 4x the packed pool bytes for
        # a path that is never taken is the same mistake as the per-block bf16 mirror.
        self._int4_pool_k = None
        n_blocks_pool = num_blocks
        needs_full = bool(self.layers_skip_for_int4) or not self.disable_radix_for_int4
        if needs_full:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                self._int4_pool_k = torch.empty(
                    (n_blocks_pool, self.block_size, self.head_dim),
                    dtype=self.dtype, device=self.device)
                self._int4_pool_v = torch.empty_like(self._int4_pool_k)
            self._int4_pool_ids = torch.arange(
                n_blocks_pool, dtype=torch.int32, device=self.device)
            self._int4_pool_tbl = torch.empty(
                (n_blocks_pool,), dtype=torch.int32, device=self.device)
            logger.info(
                "vortex INT4 KV: %.2f GB pool-shaped dequant buffer for full-context reads "
                "(%d blocks). This is the feature's largest overhead; it is only needed because a "
                "dense layer or a prefill prefix hit reads every block. Lower mem_fraction_static "
                "if the engine OOMs during cuda-graph setup.",
                2 * n_blocks_pool * self.block_size * self.head_dim
                * torch._utils._element_size(self.dtype) / GB, n_blocks_pool,
            )

        stage_gb = sum(t.element_size() * t.numel()
                       for layer in self.request_cache for t in layer.values()) / GB
        map_gb = sum(a.nbytes() for a in self.int4_arenas) / GB
        logger.info(
            "vortex INT4 KV: %.2f GB staging (%d slots/layer x %d layers, CONSTANT in context) "
            "+ %.2f GB maps. Payload compresses %.2fx; end-to-end token_ratio %.3f.",
            stage_gb, self.int4_arenas[0].n_slots, self.layer_num, map_gb,
            int4_compression_ratio(self.block_size, self.head_dim),
            self.sparse_attention.get_token_ratio(),
        )

    def int4_set_decode(self, is_decode: bool) -> None:
        """Tell the pool which forward mode this step is, for the fused-write choice.

        Called from the attention backend's metadata hooks, because ``set_kv_buffer`` is passed no
        forward mode and the choice is not derivable from what it does see. Inferring it from shapes
        was tried -- ``fused = n_tok <= num_kv_heads`` -- and silently disabled the fusion for every
        realistic batch: a performance bug no test could catch, because both paths are correct.

        Host-side, and that is safe: it is a Python bool read at launch time to pick a constexpr, so
        it is baked into the captured graph rather than read during replay. Decode is the only
        captured mode, so the captured value is always the right one. It must still be set in BOTH
        the capture and the replay hook -- replay calls a *different* metadata hook, and a value set
        only in capture leaves the eager path stale.
        """
        self._int4_is_decode = bool(is_decode)

    def _stage_int4(self, cache_slot: int, loc, cache_k, cache_v) -> None:
        """The INT4 write: stage bf16 into the request domain, quantize completed blocks.

        Decode fuses the migration into the stage kernel (measured 2.04 -> 1.29 ms/step, almost all
        launch overhead); prefill cannot, because it writes many tokens of one block in a single
        launch and migration must see every token of the block it quantizes. See ``int4_arena``.
        """
        arena = self.int4_arenas[cache_slot]
        stage = self.request_cache[cache_slot]
        fused = self._int4_is_decode
        arena.stage(
            stage[STAGE_K], stage[STAGE_V],
            cache_k.contiguous(), cache_v.contiguous(), loc,
            self.cache[cache_slot], self.page_size,
            fused=fused, k_scale_name=K_SCALE, v_scale_name=V_SCALE,
        )
        if not fused:
            arena.migrate_complete(
                stage[STAGE_K], stage[STAGE_V], self.cache[cache_slot],
                k_scale_name=K_SCALE, v_scale_name=V_SCALE,
            )

    def int4_reserve_scratch(self, max_entries: int) -> None:
        """Pre-reserve the dequant scratch and the remapped table. Never during capture.

        Allocated from the planner, before capture, because a first-use allocation inside a captured
        step is *captured into the graph* -- the allocation itself replays, which either fails or
        hands out a different address than the one baked into the kernel arguments.

        One buffer for the whole pool rather than one per layer: layers run strictly in sequence and
        each consumes its unpacked selection within its own attention call, so reuse is safe and
        per-layer copies would cost ``layer_num`` times the HBM for no benefit.
        """
        if not self.kv_int4:
            return
        n = int(max_entries)
        if self._int4_scratch_k is not None and self._int4_scratch_k.shape[0] >= n:
            return
        self._int4_scratch_k = torch.empty(
            (n, self.block_size, self.head_dim), dtype=self.dtype, device=self.device)
        self._int4_scratch_v = torch.empty_like(self._int4_scratch_k)
        self._int4_scratch_tbl = torch.empty((n,), dtype=torch.int32, device=self.device)
        logger.info(
            "vortex INT4 KV: %.3f GB dequant scratch for %d block-table entries (shared across "
            "layers, which run in sequence)",
            2 * n * self.block_size * self.head_dim
            * torch._utils._element_size(self.dtype) / GB, n,
        )

    def int4_unpack_selection(self, layer_id: int, table: torch.Tensor, n_entries: int,
                              indptr=None, n_rows: int = 0):
        """Dequantize the SELECTED blocks into the shared scratch; return ``(k, v, table)``.

        The returned table is the identity, so the attention wrapper reads scratch row ``i`` for
        entry ``i`` and is unaware of the compaction.

        Cost tracks the SELECTION, not the pool: unpacking the whole pool instead measured 2.6
        ms/layer at 12288 blocks, i.e. ~94 ms per decode step over 36 layers. On pre-Blackwell parts
        this pass is unavoidable -- flashinfer's native 4-bit paged decode is Blackwell-only -- and it
        is affordable only because sparse attention selects topk x block rather than the context. The
        read itself is at the hardware limit (1432 GB/s against a 1779 GB/s memset roofline); it
        writes 4x more than it reads, so the only remaining lever is native INT4 attention.
        """
        cache = self.cache[layer_id - self.start_layer]
        arena = self.int4_arenas[layer_id - self.start_layer]
        stage = self.request_cache[layer_id - self.start_layer]
        n = int(n_entries)
        cap = 0 if self._int4_scratch_k is None else self._int4_scratch_k.shape[0]
        if n > cap:
            # Loudly, and naming the knob. The scratch is clamped to a share of free HBM, so this is
            # a capacity limit rather than a bug -- but reading past it is an illegal memory access
            # inside the attention kernel, i.e. a crash that names the wrong subsystem.
            raise RuntimeError(
                f"INT4 dequant scratch holds {cap} block-table entries but this step needs {n} "
                f"(layer {layer_id}). Full-context reads (a layer in --vortex-layers-skip, or a "
                f"prefill prefix hit) are bounded by the step's cached blocks, and the scratch is "
                f"clamped to a share of free HBM. Lower --mem-fraction-static to leave more room, or "
                f"shorten the context."
            )
        gather_unpack(
            cache["k"], cache["v"], cache[K_SCALE], cache[V_SCALE],
            table[:n], self._int4_scratch_tbl[:n],
            stage[STAGE_K], stage[STAGE_V], arena.slot_of,
            self._int4_scratch_k[:n], self._int4_scratch_v[:n],
            indptr=indptr, n_rows=n_rows,
        )
        return self._int4_scratch_k[:n], self._int4_scratch_v[:n], self._int4_scratch_tbl[:n]

    def int4_unpack_pool(self, layer_id: int):
        """Dequantize the WHOLE pool, block ``b`` into row ``b``. For full-context reads.

        The compacting :meth:`int4_unpack_selection` is the wrong shape here. A dense layer (one in
        ``layers_skip``) or a prefill prefix hit reads every cached block, so compaction would need a
        scratch row per TABLE ENTRY -- ``rows x max_num_blocks_per_request`` = 2.6M entries, which is
        20 GiB and which the guard in ``int4_unpack_selection`` correctly refuses. Keeping the pool's
        own addressing needs one row per BLOCK instead: 12288 blocks is 0.19 GB, and the table needs
        no rewrite at all, because block ids already index the scratch.

        The cost is proportional to the POOL rather than the selection (measured 2.6 ms/layer at
        12288 blocks), which is why the sparse path does not use this. For a full-context read there
        is nothing cheaper -- the layer genuinely needs all of it.
        """
        cache = self.cache[layer_id - self.start_layer]
        arena = self.int4_arenas[layer_id - self.start_layer]
        stage = self.request_cache[layer_id - self.start_layer]
        n = cache["k"].shape[0]
        if self._int4_pool_k is None:
            raise RuntimeError("int4_reserve_scratch was not called before a full-context read")
        gather_unpack(
            cache["k"], cache["v"], cache[K_SCALE], cache[V_SCALE],
            self._int4_pool_ids, self._int4_pool_tbl,
            stage[STAGE_K], stage[STAGE_V], arena.slot_of,
            self._int4_pool_k, self._int4_pool_v,
        )
        return self._int4_pool_k, self._int4_pool_v

    def int4_counters(self) -> Dict[str, int]:
        """Summed arena counters across layers. Diagnostics only -- syncs.

        ``declined`` and ``spin_exhausted`` must be zero: both mean a token was DROPPED, which is
        otherwise silent (see ``int4_arena``, where two wrong arena sizings gave 0% accuracy with
        every other counter reading zero).
        """
        total: Dict[str, int] = {}
        for arena in self.int4_arenas:
            for k, v in arena.counters().items():
                total[k] = total.get(k, 0) + v
        return total

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
        # Unconditional: three "policy comparison" runs produced bit-identical
        # requests/fetches, and without this there was no way to tell whether the
        # host_kv_pool_blocks override was reaching the sizer or the workload simply never
        # pressured the pool. Print the realised geometry once per engine build.
        print(f"[vortex host-kv] pool_blocks={pool_blocks} policy={self.host_kv_policy} "
              f"n_sets={max(1, pool_blocks // WAYS)} ways={WAYS} "
              f"host_gb={self.host_kv_gb}", flush=True)

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
        # One launch for all layers instead of one per layer: 64 launches x 0.0117 ms was
        # ~0.75 ms/step of pure overhead on a 64-layer model, >10x the entire all-hit fetch
        # path. reserve_remap stays per-cache (it is a no-op once the buffer exists).
        tick_all(self.host_kv_caches)
        if block_tables is not None:
            for c in self.host_kv_caches:
                c.reserve_remap(block_tables)
        self._maybe_log_hit_rate()

    #: Steps between host-KV hit-rate log lines. 0 disables. Read from the environment
    #: because the pools live in sglang's *scheduler subprocess* -- an in-process
    #: ``engine.stats()`` call from the launcher cannot reach them, so reporting has to
    #: originate here and travel out through the log.
    _HIT_LOG_EVERY = int(os.environ.get("VORTEX_HOST_KV_LOG_EVERY", "0") or 0)

    def _maybe_log_hit_rate(self) -> None:
        """Periodically print the aggregate cache hit rate across layers.

        Aggregated over layers rather than per-layer: every layer sees the same block
        selection for a given step, so per-layer rates are near-identical and the sum is
        the number that matters for PCIe traffic. ``stats()`` synchronises, which is why
        this is gated off by default and sampled every N steps rather than every step.
        """
        n = self._HIT_LOG_EVERY
        if n <= 0:
            return
        self._hit_log_step = getattr(self, "_hit_log_step", 0) + 1
        if self._hit_log_step % n:
            return
        req = fet = ovf = 0
        for c in self.host_kv_caches:
            s = c.stats()
            req += s["requests"]
            fet += s["fetches"]
            ovf += s["overflow"]
        if req <= 0:
            return
        # See HostKVCache.hit_rate: overflows are REFUSALS (zero block served), not hits.
        # Printed explicitly because a non-zero value means those entries read zeros instead
        # of KV -- a correctness alarm that a bare hit rate would hide.
        rate = max(0.0, (req - fet - ovf) / req)
        print(f"[vortex host-kv] step={self._hit_log_step} policy={self.host_kv_policy} "
              f"requests={req} fetches={fet} overflow={ovf} hit_rate={rate:.4f}", flush=True)

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

        if self.kv_int4:
            self._stage_int4(cache_slot, loc, cache_k, cache_v)
        else:
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