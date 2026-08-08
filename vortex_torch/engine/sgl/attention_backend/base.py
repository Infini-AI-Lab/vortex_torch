"""Shared skeleton for vortex's sglang attention backends.

Every vortex backend does the same five things, and before this module each one
spelled all five out again:

1. read the model geometry off the ``ModelRunner`` and publish the pools;
2. trace ``sparse_attention.forward_indexer`` on zero-sized dummies and compile
   it (:meth:`_compile_indexer`);
3. plan the sparse block tables once per batch, in the eager and both
   cuda-graph phases;
4. delegate the dense/full-attention work to a wrapped upstream backend;
5. run the compiled indexer + a decode kernel per layer.

Only (1)'s asserts, (3)'s optional extra step and (5) actually differ between
backends. This module owns the rest.

Two layers:

* :class:`VortexBackendBase` — the indexer trace/compile, which is identical for
  every backend up to the query's inner dim (``head_dim`` for MHA/GQA,
  ``kv_lora_rank + qk_rope_head_dim`` for MLA).
* :class:`VortexMLABackendBase` — adds the MLA geometry read and the
  wrap-a-dense-backend plumbing shared by the three MLA backends
  (``cuda_mla`` / ``triton_mla`` / ``trtllm_mla``), whose per-batch planning was
  byte-identical apart from one optional hook.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vortex_torch.abs import FORMAT, as_vtensor
from vortex_torch.indexer import Context, MetaData
from vortex_torch.indexer.compiler.compile import compile as compile_indexer
from vortex_torch.engine.sgl.compat import (
    GraphMetadataArgs,
    attention_backend_base,
    capture_dense,
    publish_pools,
    replay_dense,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class VortexBackendBase(*attention_backend_base()):
    """Indexer trace + compile, shared by every vortex backend.

    Subclasses set the geometry attributes this reads (``group_size``,
    ``q_data_type``, ``sparse_attention``, ``ctx``) and define
    :attr:`indexer_query_dim`.
    """

    #: Inner dim of the dummy query the indexer is traced against. MHA/GQA use
    #: ``head_dim``; MLA uses the fused latent width (``kv_cache_dim``), since
    #: its "query" is the absorbed ``[q_nope_out | q_pe]``.
    indexer_query_dim_attr = "head_dim"

    @property
    def indexer_query_dim(self) -> int:
        return getattr(self, self.indexer_query_dim_attr)

    def _compile_indexer(self, model_runner: "ModelRunner") -> None:
        """Trace ``forward_indexer`` on zero-sized dummies, then compile it.

        Zero-sized so the trace records shapes/ops without allocating or running
        anything; the compiled graph is what executes per batch. Every tensor the
        flow reads must be registered on the context, in the order the flow's
        signature expects: ``q``, ``o``, then one entry per cache field.
        """
        device = model_runner.device
        indexer = self.sparse_attention.forward_indexer

        self.ctx.create(self, model_runner)
        self.ctx.metadata = MetaData.preallocate(self.ctx, device=device)
        self.ctx.assert_created()
        self.ctx.profile()

        def register(vtensor, name: str) -> None:
            self.ctx.tensor_list.append(vtensor)
            self.ctx.output_tensor_to_op_list.append(None)
            self.ctx.tensor_id_to_tensor_name_map[vtensor.tensor_id] = name

        def dummy(shape, fmt, tensor_id, *, dtype=None, zeros=False):
            factory = torch.zeros if zeros else torch.empty
            tensor = factory(shape, device=device, dtype=dtype or self.q_data_type)
            return as_vtensor(tensor, fmt, tensor_id=tensor_id)

        with torch.no_grad():
            q = dummy((0, self.group_size, self.indexer_query_dim), FORMAT.BATCHED, 0)
            register(q, "q")
            o = dummy((0, 1, 1), FORMAT.RAGGED, 1)
            register(o, "o")

            cache = {}
            meta = self.sparse_attention.get_cache_meta_info().items()
            for i, (name, (shape, cache_dtype)) in enumerate(meta):
                vtensor = dummy(
                    (0, shape[0], shape[1]), FORMAT.PAGED, 2 + i,
                    dtype=cache_dtype, zeros=True,
                )
                cache[name] = vtensor
                register(vtensor, f"cache['{name}']")

            indexer(q, o, cache, ctx=self.ctx)

        self.compiled_indexer = compile_indexer(self.ctx)()
        self.ctx.summary()
        self.ctx.execute()


class VortexMLABackendBase(VortexBackendBase):
    """Base for the MLA backends, which wrap an upstream dense backend.

    Owns the geometry read, the per-batch sparse planning in all three phases
    (eager / graph capture / graph replay) and the dense delegation. A subclass
    supplies its own ``_dense``, its decode kernel, and — if it needs one — the
    :meth:`_after_plan_decode` hook.
    """

    indexer_query_dim_attr = "kv_cache_dim"

    def _init_mla_geometry(self, model_runner: "ModelRunner") -> None:
        """Read the MLA geometry + vortex knobs off the runner.

        Call first from a subclass ``__init__`` (after ``super().__init__()``).
        MLA presents one latent "KV head" of width ``kv_lora_rank +
        qk_rope_head_dim``, so ``head_dim`` is that fused width and every query
        head is in one group.
        """
        publish_pools(self, model_runner)
        server_args = model_runner.server_args
        config = model_runner.model_config

        self.max_context_len = config.context_len
        self.device = model_runner.device

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.head_dim = self.kv_cache_dim
        # tp is handled by sglang, so the config head count is already local.
        self.num_qo_heads = config.num_attention_heads
        self.num_kv_heads = 1
        self.group_size = self.num_qo_heads

        self.q_data_type = model_runner.dtype
        self.data_type = model_runner.kv_cache_dtype

        self.page_size = server_args.page_size
        self.block_size = server_args.vortex_block_size
        assert self.page_size % self.block_size == 0, (
            f"page_size ({self.page_size}) must be a multiple of "
            f"vortex_block_size ({self.block_size})."
        )
        self.num_blocks_per_page = self.page_size // self.block_size

        self.layers_skip = server_args.vortex_layers_skip
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.sparse_attention = model_runner.sparse_attention  # a vFlowMLA
        self.ctx = Context()

    # ------------------------------------------------------------------ #
    # per-batch planning — eager, graph capture, graph replay
    # ------------------------------------------------------------------ #
    def _plan_sparse(self, seq_lens: torch.Tensor, req_pool_indices: torch.Tensor) -> None:
        """Fill the sparse block tables for this decode batch, then run the hook.

        ``cached_seq_lens`` must be int32 for the planner kernel; the cast is a
        no-op when it already is.
        """
        self.plan_decode(
            cached_seq_lens=seq_lens.to(torch.int32),
            req_to_token=self.req_to_token,
            req_indices=req_pool_indices,
            ctx=self.ctx,
        )
        self._after_plan_decode(seq_lens)

    def _after_plan_decode(self, seq_lens: torch.Tensor) -> None:
        """Extra per-batch step, run right after the block tables are filled.

        Default no-op. ``cuda_mla`` overrides it to build its load-balanced work
        queue, which needs the ``sparse_seqlens`` ``plan_decode`` just wrote.
        """

    def init_forward_metadata(self, forward_batch) -> None:
        self._dense.init_forward_metadata(forward_batch)
        if forward_batch.forward_mode.is_decode_or_idle():
            self._plan_sparse(forward_batch.seq_lens, forward_batch.req_pool_indices)
        else:
            self._init_extend_metadata(forward_batch)

    def _init_extend_metadata(self, forward_batch) -> None:
        """Prefill/extend-side setup. Default: the dense backend handles it."""

    def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf=None):
        self._dense.init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)

    def init_forward_metadata_capture_cuda_graph(
        self, bs, num_tokens, req_pool_indices, seq_lens, encoder_lens,
        forward_mode, spec_info,
    ):
        capture_dense(
            self._dense,
            GraphMetadataArgs(
                bs=bs, req_pool_indices=req_pool_indices, seq_lens=seq_lens,
                forward_mode=forward_mode, encoder_lens=encoder_lens,
                spec_info=spec_info,
            ),
        )
        if forward_mode.is_decode_or_idle():
            self._plan_sparse(seq_lens, req_pool_indices)

    def init_forward_metadata_replay_cuda_graph(
        self, bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens,
        forward_mode, spec_info, seq_lens_cpu,
    ):
        replay_dense(
            self._dense,
            GraphMetadataArgs(
                bs=bs, req_pool_indices=req_pool_indices, seq_lens=seq_lens,
                forward_mode=forward_mode, encoder_lens=encoder_lens,
                spec_info=spec_info, seq_lens_cpu=seq_lens_cpu,
                seq_lens_sum=seq_lens_sum,
            ),
        )
        if forward_mode.is_decode_or_idle():
            self._plan_sparse(seq_lens, req_pool_indices)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Defer to the wrapped backend.

        Not hardcoded: this value pads `seq_lens` in captured graphs, and the
        dense backend is what reads those padded entries on skipped layers. If
        the two disagree the padding is wrong for one of them.
        """
        return self._dense.get_cuda_graph_seq_len_fill_value()

    def forward_extend(self, *args, **kwargs):
        """MLA prefill is dense; the wrapped backend owns it."""
        return self._dense.forward_extend(*args, **kwargs)
