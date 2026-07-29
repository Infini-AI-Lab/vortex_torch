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
logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024
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

    is_vortex_pool = True

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
        self._create_buffers()
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



    def _create_buffers(self):
        
        self.cache_meta_info = self.sparse_attention.get_cache_meta_info()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):  
                self.cache = [
                    {
                        cache_name:  torch.zeros(
                                (self.num_pages * self.num_blocks_per_page, cache_shape[0], cache_shape[1]),
                                dtype=cache_dtype,
                                device=self.device,
                            )
                        
                        for (cache_name, (cache_shape, cache_dtype)) in self.cache_meta_info.items()
                    }
                    
                    for _ in range(self.layer_num)
                ]
        
    def _clear_buffers(self):
        del self.cache
       

    def get_cache_size_bytes(self) -> int:
        """
        Return total bytes occupied by all tensors in `self.cache`.
        Works even if some entries are not tensors.
        """
        total_bytes = 0

        for layer_cache in self.cache:
            if not isinstance(layer_cache, dict):
                # Be tolerant to unexpected structures
                continue

            for t in layer_cache.values():
                if not torch.is_tensor(t):
                    continue

                # Prefer accurate allocated size if available (includes padding/strides)
                try:
                    total_bytes += int(t.untyped_storage().nbytes())
                except AttributeError:
                    # Fallback: logical size in bytes
                    total_bytes += int(t.element_size() * t.numel())

        return total_bytes
    
    def get_kv_size_bytes(self):
        return self.get_cache_size_bytes()
    
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
                self.cache[layer_id - self.start_layer], loc, ctx=self.ctx,
                cur_layer=layer_id,
            )

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool

    def get_cpu_copy(self, indices, mamba_indices=None):
        
        raise NotImplementedError

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        
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
    ):
        
        assert layer_id_override is None
        assert loc.dtype == torch.int64

        layer_id = layer.layer_id

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

        self.set_kv_buffer_func(
            self.cache[layer_id - self.start_layer]["k"],
            self.cache[layer_id - self.start_layer]["v"],
            cache_k.contiguous(),
            cache_v.contiguous(),
            loc,
            self.page_size
        )
        if layer_id in self.layers_skip:
            return
        self.compiled_cache.forward(
            self.cache[layer_id - self.start_layer], loc, ctx=self.ctx,
            cur_layer=layer_id,
        )
        
    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        
        raise NotImplementedError
