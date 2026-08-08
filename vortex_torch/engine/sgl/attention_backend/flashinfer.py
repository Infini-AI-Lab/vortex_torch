from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Union, Dict, Tuple
from functools import partial
import torch
# Import from the defining module, not the package root: these backend
# modules are imported from inside vortex_torch's own import chain (the
# sglang hook), where the root package has not yet bound its re-exports.
from vortex_torch.utils import is_hopper
from vortex_torch.indexer import Context
from vortex_torch.indexer.utils_sglang import (
    get_chunkwise_hn2nh_transpose,
    get_chunkwise_nh2hn_transpose,
    get_decode_planner,
    get_prefill_planner,
)
if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from vortex_torch.engine.sgl.compat import (
    get_attention_tp_size,
    is_draft_extend,
    publish_pools,
    token_to_kv_pool,
)
from .base import VortexBackendBase
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.layers.attention.flashinfer_backend import should_use_tensor_core
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens

@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VortexFlashInferBackend(VortexBackendBase):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # sglang >= 0.5.16 reads the KV / req pools off the *backend*
        # (forward_context.get_token_to_kv_pool); publish them here.
        publish_pools(self, model_runner)

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        assert model_runner.sliding_window_size is None
        assert not model_runner.model_config.is_encoder_decoder 
        assert not self.skip_prefill
        # `is_multimodal` is deliberately NOT asserted on: it is an
        # architecture-name lookup (the model *can* take images), not a property
        # of this request, and vortex scores blocks from the K the model already
        # wrote — so image support and rope variants are the model's business.
        # What is unvalidated is a batch actually carrying image tokens, which
        # `init_forward_metadata` rejects per-batch.
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None
        self.num_wrappers = 2
        self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        # if (
        #     "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
        #     or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
        #     or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
        # ):
        #     global_config.flashinfer_workspace_size = 512 * 1024 * 1024

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            global_workspace_buffer = torch.empty(
                512 * 1024 * 1024,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_workspace_buffer
        max_bs = model_runner.req_to_token_pool.size
        
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.group_size = self.num_qo_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.decode_use_tensor_cores = should_use_tensor_core(self.data_type, self.num_qo_heads, self.num_kv_heads)
        assert self.q_data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        assert self.data_type in [torch.bfloat16, torch.float8_e5m2, torch.float8_e4m3fn]
        self.is_fp8 = (self.data_type in [torch.float8_e5m2, torch.float8_e4m3fn])
        
        # Assign key configuration and parameters
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.page_size = model_runner.server_args.page_size
        self.block_size = model_runner.server_args.vortex_block_size
        self.layers_skip = model_runner.server_args.vortex_layers_skip
        self.num_blocks_per_page = self.page_size // self.block_size
        assert self.page_size % self.block_size == 0, "Page size must be a multiple of block size."
        # ===========================
        # Prefill KV-indptr buffers
        # ===========================

        self.kv_indptr_prefill = torch.zeros(
            (max_bs * self.num_kv_heads + 1,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Decode-path buffers live on ``self.ctx.metadata`` (pre-allocated
        # in ``_compile_indexer``). The flashinfer wrappers consume them directly
        # via ``self.ctx.metadata.dense_kv_indptr`` etc.
        # ===========================

        # ===========================
        # KV indices (prefill) — still owned by this object (the flashinfer
        # prefill wrapper has its own indptr/indices arrays).
        # ===========================

        self.kv_indices_prefill = torch.zeros(
            (
                (max_bs * self.num_kv_heads * model_runner.model_config.context_len + self.page_size - 1)
                // self.page_size,
            ),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # KV last-page-len (prefill — decode lives on MetaData)
        # ===========================

        self.kv_last_page_len_prefill = torch.ones(
            (max_bs * self.num_kv_heads,),
            dtype=torch.int32,
            device=model_runner.device
        )

        # ===========================
        # Query/Output indptr buffers
        # ===========================

        self.qo_indptr = [
            torch.zeros(
                (max_bs + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
            torch.zeros(
                (max_bs * self.num_kv_heads + 1,),
                dtype=torch.int32,
                device=model_runner.device
            ),
        ]

        # ===========================
        # Batch table (token-level mapping)
        # ===========================

        self.batch_table = torch.zeros(
            (model_runner.server_args.max_prefill_tokens,),
            dtype=torch.uint16,
            device=model_runner.device
        )

        
        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend= "auto" if not is_hopper() else "fa3"
        )

        self.prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend="fa2" if ((not is_hopper()) or self.is_fp8) else "fa3",
                    )
        
        self.decode_wrappers = [
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
            BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                ),
        ]
        
        self.plan_decode = get_decode_planner(model_runner.server_args.vortex_schedule_policy)
        self.plan_prefill = get_prefill_planner()
        self.chunkwise_nh2hn_transpose = get_chunkwise_nh2hn_transpose()
        self.chunkwise_hn2nh_transpose = get_chunkwise_hn2nh_transpose()

        self.sparse_attention = model_runner.sparse_attention
        self.ctx = Context()
        self._compile_indexer(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata: Dict[int, List[BatchDecodeWithPagedKVCacheWrapper]] = {}
        self.plan_graph: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]]
    


    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        
        assert not is_draft_extend(forward_batch.forward_mode)
        assert not forward_batch.forward_mode.is_target_verify()
        # Multimodal architectures are fine (see __init__); a batch that really
        # carries image tokens is not validated, so reject it explicitly instead
        # of silently sparsifying attention over image embeddings. Gated on the
        # init-time flag so text-only models pay nothing per forward, and using
        # upstream's own predicate (which also looks past audio-only inputs)
        # rather than re-deriving it.
        if self.is_multimodal and forward_batch.contains_image_inputs():
            raise AssertionError(
                "vortex sparsity has not been validated on batches containing "
                "image tokens; send text-only requests or disable vortex sparsity."
            )
        
        if forward_batch.forward_mode.is_decode_or_idle():
            
            bs = len(forward_batch.req_pool_indices)
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx
            )
            
            self.decode_wrappers[0].plan(
                indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.dense_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_wrappers[1].plan(
                indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.sparse_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            self.forward_metadata = DecodeMetadata([self.decode_wrappers[0], self.decode_wrappers[1]])

        elif forward_batch.forward_mode.is_extend():
            
            # sglang_plan_prefill reads cached_seq_lens / input_seq_lens as
            # int32. extend_prefix_lens is int32 on the eager path, but the
            # 0.5.16 prefill cuda-graph runner allocates its static
            # extend_prefix_lens buffer as int64, so coerce rather than trust
            # the caller's dtype.
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
            extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            bs = len(forward_batch.req_pool_indices)
            
            self.plan_prefill(
                cached_seq_lens=prefix_lens,
                dense_kv_indptr=self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                dense_kv_indices=self.kv_indices_prefill,
                input_seq_lens=(forward_batch.seq_lens.to(torch.int32) - prefix_lens),
                qo_indptr_ragged=self.qo_indptr[0][:bs+1],
                qo_indptr_paged=self.qo_indptr[1][:bs*self.num_kv_heads+1],
                kv_last_page_len=self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                batch_table=self.batch_table,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads
            )
            
   
            self.prefill_wrapper_ragged.plan(
                self.qo_indptr[0][:bs+1],
                self.qo_indptr[0][:bs+1],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
            
            self.prefill_wrapper_paged.plan(
                self.qo_indptr[1][:bs*self.num_kv_heads+1],
                self.kv_indptr_prefill[:bs*self.num_kv_heads+1],
                self.kv_indices_prefill,
                self.kv_last_page_len_prefill[:bs*self.num_kv_heads],
                self.group_size,
                1,
                self.head_dim,
                self.page_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
                custom_mask=None,
                non_blocking=True,
            )
            

            self.forward_metadata = PrefillMetadata(extend_no_prefix)

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        pass
    
    
    def capture_plan_graph(
        self, 
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        bs: int):
        
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):  
        assert bs == num_tokens
        
        if forward_mode.is_decode_or_idle():
            decode_wrappers = [
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.ctx.metadata.dense_kv_indices,
                        paged_kv_last_page_len_buffer=self.ctx.metadata.kv_last_page_len[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
                BatchDecodeWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        use_cuda_graph=True,
                        use_tensor_cores=self.decode_use_tensor_cores,
                        paged_kv_indptr_buffer=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads + 1],
                        paged_kv_indices_buffer=self.ctx.metadata.sparse_kv_indices,
                        paged_kv_last_page_len_buffer=self.ctx.metadata.kv_last_page_len[
                            :bs*self.num_kv_heads
                        ],
                    ),
                
            ]

            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
            
            decode_wrappers[0].plan(
                indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.dense_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            decode_wrappers[1].plan(
                indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
                indices=self.ctx.metadata.sparse_kv_indices,
                last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.head_dim,
                page_size=self.block_size,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )
            
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)             
        else:
            raise NotImplementedError
            

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert forward_mode.is_decode_or_idle()
        
        self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )
        
        self.decode_cuda_graph_metadata[bs][0].plan(
            indptr=self.ctx.metadata.dense_kv_indptr[:bs*self.num_kv_heads+1],
            indices=self.ctx.metadata.dense_kv_indices,
            last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )
        
        self.decode_cuda_graph_metadata[bs][1].plan(
            indptr=self.ctx.metadata.sparse_kv_indptr[:bs*self.num_kv_heads+1],
            indices=self.ctx.metadata.sparse_kv_indices,
            last_page_len=self.ctx.metadata.kv_last_page_len[:bs*self.num_kv_heads],
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=self.head_dim,
            page_size=self.block_size,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc
        
        logits_soft_cap = layer.logit_cap

        q = q.contiguous()

        if self.forward_metadata.extend_no_prefix:
            o = self.prefill_wrapper_ragged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

        else:
            o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            
            q_t = self.chunkwise_nh2hn_transpose(
                q.view(-1, self.num_qo_heads, self.head_dim),
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            
            k_cache, v_cache = token_to_kv_pool(forward_batch).get_kv_buffer(layer.layer_id)
            k_cache = k_cache.view(-1, self.page_size, 1, self.head_dim)
            v_cache = v_cache.view(-1, self.page_size, 1, self.head_dim)
            o2, s2 = self.prefill_wrapper_paged.forward_return_lse(
                q_t,
                (k_cache, v_cache),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
                )
            o2_t, s2_t = self.chunkwise_hn2nh_transpose(
                o2,  s2,
                self.qo_indptr[0],
                self.batch_table,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim
            )
            
            o, _ = merge_state(o1, s1, o2_t, s2_t)

        if save_kv_cache:
                token_to_kv_pool(forward_batch).set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """
        Decode-time forward pass with optional sparse attention.
        Expects KV to be sourced from token_to_kv_pool; can also save new KV.
        """

        # Sanity checks and setup
        assert not layer.is_cross_attention
        cache_loc = forward_batch.out_cache_loc

        # Optionally write incoming K/V to decode cache
        if k is not None:
            assert v is not None
            if save_kv_cache:
                token_to_kv_pool(forward_batch).set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Read Cache from memory pool
        cache = self.vortex_cache(layer.layer_id)
        
        cache_k = cache["k"].view(-1, self.block_size, 1, self.head_dim)
        cache_v = cache["v"].view(-1, self.block_size, 1, self.head_dim)

        # Use the *_float scalars (not layer.k_scale / layer.v_scale, which are
        # GPU tensors): the flashinfer decode wrapper expects python-float
        # scales, and reading the tensor would force a device->host sync that
        # breaks cuda-graph capture. For a bf16 KV cache these are the 1.0
        # default (no-op); for an fp8 cache they dequantize the stored K/V,
        # matching the div() applied on the write side in set_kv_buffer.
        k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0

        # Decide whether to use sparsity on this layer
        use_sparsity = (layer.layer_id not in self.layers_skip)

        if use_sparsity:
            # Prepare Q in grouped shape expected by sparse path
            q = q.contiguous().view(-1, self.group_size, layer.head_dim)

            # Build sparse indices into paged KV buffers
            self.compiled_indexer.forward(
                q=q,
                o=self.forward_metadata.decode_wrappers[1]._paged_kv_indices_buf,
                cache=cache,
                ctx=self.ctx
            )

            # Sparse attention compute
            o = self.forward_metadata.decode_wrappers[1].forward(
                q,
                (cache_k, cache_v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        else:
            # Dense attention path
            o = self.forward_metadata.decode_wrappers[0].forward(
                q.contiguous().view(-1, self.group_size, layer.head_dim),
                (cache_k, cache_v),
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        # Restore to merged head dimension
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)