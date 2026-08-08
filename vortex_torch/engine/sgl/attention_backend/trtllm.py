from __future__ import annotations

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union, Dict
import torch
# Import from the defining module, not the package root: these backend
# modules are imported from inside vortex_torch's own import chain (the
# sglang hook), where the root package has not yet bound its re-exports.
from vortex_torch.utils import is_hopper
from vortex_torch.abs import as_vtensor, FORMAT
from vortex_torch.indexer import Context, MetaData
from vortex_torch.indexer.compiler.compile import compile as compile_indexer
from vortex_torch.indexer.utils_sglang import (
    get_chunkwise_hn2nh_transpose,
    get_chunkwise_nh2hn_transpose,
    get_decode_planner_trtllm,
    get_prefill_planner,
)
if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
    import logging

    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from vortex_torch.engine.sgl.compat import (
    attention_backend_base,
    get_attention_tp_size,
    is_draft_extend,
    publish_pools,
    token_to_kv_pool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

if is_flashinfer_available():
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache


@dataclass
class DecodeMetadata:
    # Index 0 = dense path; index 1 = sparse path (refreshed per layer).
    block_tables: List[torch.Tensor]
    seq_lens: List[torch.Tensor]
    bs: int  # effective batch = real_bs * num_kv_heads

@dataclass
class PrefillMetadata:
    extend_no_prefix: bool


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None


class VortexTRTLLMBackend(*attention_backend_base()):
    """Flashinfer trtllm attention kernels."""

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
        # NOTE: `is_multimodal` is an architecture-name lookup (the model *can*
        # take images), not a property of this request. It is not by itself a
        # reason vortex cannot run: the indexer scores blocks from the K the
        # model already wrote to the cache, so image tokens and any rope variant
        # (mrope / partial rotary) are the model's business, not vortex's. What
        # vortex has NOT been validated against is a batch that actually carries
        # image tokens, so gate on the inputs rather than on the architecture.
        # Enforced per-batch in `init_forward_metadata`.
        assert kv_indptr_buf is None
        assert kv_last_page_len_buf is None

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
        # Q/O may be bf16 or fp8 (e4m3/e5m2); KV in bf16 or fp8.
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
        # in ``_compile``), NOT on this object. See ``MetaData.preallocate``.
        # ===========================

        # ===========================
        # KV indices (prefill) — still used by the flashinfer prefill wrapper.
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
        # KV last page length tracking (prefill — decode lives on MetaData).
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

        # ===========================
        # trtllm decode buffers are pre-allocated on ``self.ctx.metadata``
        # in ``_compile`` — block_tables / seq_lens / winfo all live there.
        # ===========================
        # Effective batch is `bs * num_kv_heads` because each kv-head is folded
        # into the batch dim (MQA-style) with num_kv_heads=1 for the kernel.
        self.max_blocks_per_seq = (
            (model_runner.model_config.context_len + self.block_size - 1)
            // self.block_size
        )
        # trtllm requires a workspace zero-initialised on first use; keep it
        # independent of the flashinfer prefill workspace.
        self.trtllm_workspace_buffer = torch.zeros(
            512 * 1024 * 1024,
            dtype=torch.uint8,
            device=model_runner.device,
        )

        self.plan_decode = get_decode_planner_trtllm(model_runner.server_args.vortex_schedule_policy)
        self.plan_prefill = get_prefill_planner()
        self.chunkwise_nh2hn_transpose = get_chunkwise_nh2hn_transpose()
        self.chunkwise_hn2nh_transpose = get_chunkwise_hn2nh_transpose()

        # Tell the indexer / topk codegen which decode kernel layout to use.
        # Must be set BEFORE _compile (ctx.create reads it). Write the config
        # object (single source of truth); the server_args.vortex_* shim is read-only.
        if getattr(model_runner.server_args, "vortex", None) is not None:
            model_runner.server_args.vortex.attention_backend = "trtllm"

        self.sparse_attention = model_runner.sparse_attention
        self.ctx = Context()
        self._compile(model_runner)
        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None
        self.decode_cuda_graph_metadata: Dict[int, DecodeMetadata] = {}



    def _compile(self, model_runner: "ModelRunner") -> None:
        """Trace the sparse-attention indexer on zero-sized dummies and compile it."""
        device = model_runner.device
        dtype = self.q_data_type
        indexer = self.sparse_attention.forward_indexer

        self.ctx.create(self, model_runner)
        # Allocate every per-forward-batch buffer (winfo_*, dense/sparse
        # block_tables + seqlens, kv_last_page_len) on a single MetaData
        # owned by the context. The decode planner writes into this
        # MetaData; the indexer kernels read from it.
        self.ctx.metadata = MetaData.preallocate(self.ctx, device=device)
        self.ctx.assert_created()
        self.ctx.profile()

        def register(vt, name: str) -> None:
            self.ctx.tensor_list.append(vt)
            self.ctx.output_tensor_to_op_list.append(None)
            self.ctx.tensor_id_to_tensor_name_map[vt.tensor_id] = name

        def make_dummy(shape, fmt, tensor_id, *, tdtype=dtype, zeros=False):
            factory = torch.zeros if zeros else torch.empty
            return as_vtensor(factory(shape, device=device, dtype=tdtype), fmt, tensor_id=tensor_id)

        with torch.no_grad():
            q_dummy = make_dummy((0, self.group_size, self.head_dim), FORMAT.BATCHED, tensor_id=0)
            register(q_dummy, "q")

            o_dummy = make_dummy((0, 1, 1), FORMAT.RAGGED, tensor_id=1)
            register(o_dummy, "o")

            cache_dummy = {}
            for i, (name, (shape, cache_dtype)) in enumerate(
                self.sparse_attention.get_cache_meta_info().items()
            ):
                vt = make_dummy(
                    (0, shape[0], shape[1]),
                    FORMAT.PAGED,
                    tensor_id=2 + i,
                    tdtype=cache_dtype,
                    zeros=True,
                )
                cache_dummy[name] = vt
                register(vt, f"cache['{name}']")

            indexer(q_dummy, o_dummy, cache_dummy, ctx=self.ctx)

        self.compiled_indexer = compile_indexer(self.ctx)()
        self.ctx.summary()
        self.ctx.execute()

    
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
            # plan_decode is the trtllm planner: it fills block_tables[0],
            # seq_lens[0], and the BOS+EOS slots of block_tables[1] /
            # seq_lens[1]. The topk kernel fills the middle of
            # block_tables[1] later in forward_decode (sparse path).
            self.plan_decode(
                cached_seq_lens=forward_batch.seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=forward_batch.req_pool_indices,
                ctx=self.ctx
            )

            eff_bs = bs * self.num_kv_heads

            md = self.ctx.metadata
            self.forward_metadata = DecodeMetadata(
                block_tables=[
                    md.dense_block_tables[:eff_bs],
                    md.sparse_block_tables[:eff_bs],
                ],
                seq_lens=[
                    md.dense_seqlens[:eff_bs],
                    md.sparse_seqlens[:eff_bs],
                ],
                bs=eff_bs,
            )

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
            # trtllm planner fills block_tables[0] / seq_lens[0] and the
            # BOS+EOS slots of block_tables[1] / seq_lens[1]; topk fills
            # the middle of path 1 inside forward_decode.
            self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )

            eff_bs = bs * self.num_kv_heads

            md = self.ctx.metadata
            metadata = DecodeMetadata(
                block_tables=[
                    md.dense_block_tables[:eff_bs],
                    md.sparse_block_tables[:eff_bs],
                ],
                seq_lens=[
                    md.dense_seqlens[:eff_bs],
                    md.sparse_seqlens[:eff_bs],
                ],
                bs=eff_bs,
            )
            self.decode_cuda_graph_metadata[bs] = metadata
            self.forward_metadata = metadata
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

        # trtllm planner fills block_tables[0] / seq_lens[0] and the BOS+EOS
        # slots of block_tables[1] / seq_lens[1]; topk fills the middle of
        # path 1 inside forward_decode.
        self.plan_decode(
                cached_seq_lens=seq_lens.to(torch.int32),
                req_to_token=self.req_to_token,
                req_indices=req_pool_indices,
                ctx=self.ctx
            )

        self.forward_metadata = self.decode_cuda_graph_metadata[bs]

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

        # NHD per-tensor cache layout: [num_pages, block_size, 1, head_dim]
        cache_k = cache["k"].view(-1, self.block_size, 1, self.head_dim)
        cache_v = cache["v"].view(-1, self.block_size, 1, self.head_dim)

        # trtllm doesn't support logits soft-cap on decode — assert it's off.
        assert not layer.logit_cap, "trtllm decode does not support logits soft-cap"

        # Fold sm_scale, k_scale, v_scale into bmm1/bmm2 scales for trtllm.
        # Use the *_float scalars (not layer.k_scale / layer.v_scale, which are
        # GPU tensors): trtllm_batch_decode_with_kv_cache expects python-float
        # bmm scales, and reading the tensor here would force a device->host
        # sync that breaks cuda-graph capture. For a bf16 KV cache these are
        # the 1.0 default (no-op); for an fp8 cache they dequantize the stored
        # K/V, matching the div() applied on the write side in set_kv_buffer.
        k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
        v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
        bmm1_scale = layer.scaling * k_scale
        bmm2_scale = v_scale

        # Decide whether to use sparsity on this layer
        use_sparsity = (layer.layer_id not in self.layers_skip)

        if use_sparsity:
            # Prepare Q in grouped shape expected by sparse path
            q = q.contiguous().view(-1, self.group_size, layer.head_dim)

            # In trtllm mode the topk kernel writes the selected block ids
            # directly into the 2D ``sparse_block_tables``; BOS+EOS slots
            # and ``sparse_seqlens`` were prefilled by plan_decode.
            self.compiled_indexer.forward(
                q=q,
                o=self.ctx.metadata.sparse_block_tables,
                cache=cache,
                ctx=self.ctx,
            )
            o = trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(cache_k, cache_v),
                workspace_buffer=self.trtllm_workspace_buffer,
                block_tables=self.forward_metadata.block_tables[1],
                seq_lens=self.forward_metadata.seq_lens[1],
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                kv_layout="NHD",
            )
        else:
            # Dense attention path
            o = trtllm_batch_decode_with_kv_cache(
                query=q.contiguous().view(-1, self.group_size, layer.head_dim),
                kv_cache=(cache_k, cache_v),
                workspace_buffer=self.trtllm_workspace_buffer,
                block_tables=self.forward_metadata.block_tables[0],
                seq_lens=self.forward_metadata.seq_lens[0],
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                kv_layout="NHD",
            )

        # Restore to merged head dimension
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)