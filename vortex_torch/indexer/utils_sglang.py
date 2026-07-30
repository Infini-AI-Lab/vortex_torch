import os
import torch
from typing import Tuple
from .context import Context
from .planner_sglang import get_sglang_plan_decode_v2_module
from .prefill_sglang import get_sglang_prefill_module


def get_decode_planner(policy: str = None):

    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )
    def plan_decode(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context
    ):
        md = ctx.metadata
        module.sglang_plan_decode_v2(
            cached_seq_lens,
            md.dense_kv_indptr,
            md.dense_kv_indices,
            md.sparse_kv_indptr,
            md.sparse_kv_indices,
            md.kv_last_page_len,
            req_to_token,
            req_indices,
            md.winfo_q_indices,
            md.winfo_is_first_workload_per_batch,
            md.winfo_kv_offsets,
            md.winfo_kv_lens,
            md.winfo_num_workloads,
            md.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size
        )

        md.set_batch_size(cached_seq_lens.shape[0])

    return plan_decode


def get_decode_planner_trtllm(policy: str = None):
    """Decode planner variant that emits trtllm-ready outputs directly.

    The trtllm planner is **indptr-free**: it never reads or writes
    ``dense_kv_indptr`` / ``sparse_kv_indptr`` / ``dense_kv_indices`` /
    ``sparse_kv_indices``. Outputs filled by the underlying CUDA kernel:

      * ``ctx.dense_block_tables``  — every selected page for the dense path
      * ``ctx.sparse_block_tables`` — only the BOS+EOS slots; the middle is
        filled by the topk kernel later
      * ``ctx.dense_seqlens`` / ``ctx.sparse_seqlens`` — int32 token counts
        consumed by ``trtllm_batch_decode_with_kv_cache``, the trtllm topk
        kernels, and the Schedule.S Triton kernels (which derive per-row
        block counts via ``ceil(tokens / block_size)``)
      * ``ctx.kv_last_page_len`` — same semantics as before
      * ``ctx.winfo_*`` — workload-scheduler outputs; ``winfo_kv_offsets[j]``
        carries ``row * max_blocks_per_seq + col`` so the Schedule.W kernel
        preamble (with ``indices = dense_block_tables.view(-1)``) resolves
        page ids correctly.
    """
    module = get_sglang_plan_decode_v2_module(
        policy_body=policy,
        verbose=True,
        fallback_to_default=True,
    )

    def plan_decode_trtllm(
        cached_seq_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        ctx: Context,
    ):
        md = ctx.metadata
        module.sglang_plan_decode_v2_trtllm(
            cached_seq_lens,
            md.kv_last_page_len,
            md.dense_block_tables,
            md.sparse_block_tables,
            md.dense_seqlens,
            md.sparse_seqlens,
            req_to_token,
            req_indices,
            md.winfo_q_indices,
            md.winfo_is_first_workload_per_batch,
            md.winfo_kv_offsets,
            md.winfo_kv_lens,
            md.winfo_num_workloads,
            md.winfo_chunk_size,
            ctx.page_size,
            ctx.block_size,
            ctx.num_kv_heads,
            ctx.topk_val,
            ctx.topk_ratio,
            ctx.block_reserved_bos,
            ctx.block_reserved_eos,
            ctx.workload_chunk_size,
        )
        if os.environ.get("VORTEX_NAIVE_WORKLOAD_PLANNER", "0") == "1":
            # Controlled planner ablation. Keep the normal block-table and
            # seqlen preparation above, but replace its compact prefix-sum
            # worklist with a padded (row x chunk) rectangular worklist.
            # This deliberately includes zero-length entries for shorter rows.
            rows = cached_seq_lens.shape[0] * ctx.num_kv_heads
            dense_blocks = torch.div(
                md.dense_seqlens[:rows] + ctx.block_size - 1,
                ctx.block_size,
                rounding_mode="floor",
            )
            block_count = torch.clamp(
                dense_blocks - ctx.block_reserved_eos, min=0
            )
            max_chunks = (
                int(block_count.max().item()) + ctx.workload_chunk_size - 1
            ) // ctx.workload_chunk_size
            n = rows * max_chunks
            flat = torch.arange(n, dtype=torch.int32, device=cached_seq_lens.device)
            row = torch.div(flat, max_chunks, rounding_mode="floor")
            chunk = flat - row * max_chunks
            offset_in_row = chunk * ctx.workload_chunk_size
            lens = torch.clamp(
                block_count[row.long()] - offset_in_row,
                min=0,
                max=ctx.workload_chunk_size,
            )
            md.winfo_q_indices[:n].copy_(row)
            md.winfo_kv_offsets[:n].copy_(
                row * ctx.max_num_blocks_per_request + offset_in_row
            )
            md.winfo_kv_lens[:n].copy_(lens)
            md.winfo_is_first_workload_per_batch[:n].copy_(
                (chunk == 0).to(torch.uint8)
            )
            md.winfo_num_workloads.fill_(n)
            md.winfo_chunk_size.fill_(ctx.workload_chunk_size)
        md.set_batch_size(cached_seq_lens.shape[0])

    return plan_decode_trtllm


def get_prefill_planner():
    """Mirror of :func:`get_decode_planner` for the prefill path.

    Triggers a one-time JIT compile of the prefill module on first call,
    then returns a closure that calls ``sglang_plan_prefill`` with the
    module reference baked in (no per-call lookup).
    """
    module = get_sglang_prefill_module()

    def plan_prefill(
        cached_seq_lens: torch.Tensor,
        dense_kv_indptr: torch.Tensor,
        dense_kv_indices: torch.Tensor,
        input_seq_lens: torch.Tensor,
        qo_indptr_ragged: torch.Tensor,
        qo_indptr_paged: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        req_to_token: torch.Tensor,
        req_indices: torch.Tensor,
        batch_table: torch.Tensor,
        page_size: int,
        num_kv_heads: int,
    ):
        module.sglang_plan_prefill(
            cached_seq_lens,
            dense_kv_indptr,
            dense_kv_indices,
            input_seq_lens,
            qo_indptr_ragged,
            qo_indptr_paged,
            kv_last_page_len,
            req_to_token,
            req_indices,
            batch_table,
            page_size,
            num_kv_heads,
        )

    return plan_prefill


def get_chunkwise_nh2hn_transpose():
    """Factory for the ``Chunkwise_NH2HN_Transpose`` kernel; mirrors
    :func:`get_decode_planner`'s closure pattern so the module reference
    is bound once at backend init."""
    module = get_sglang_prefill_module()

    def chunkwise_nh2hn_transpose(
        x: torch.Tensor,
        indptr: torch.Tensor,
        batch_table: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        return module.Chunkwise_NH2HN_Transpose(
            x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim,
        )

    return chunkwise_nh2hn_transpose


def get_chunkwise_hn2nh_transpose():
    """Factory for the ``Chunkwise_HN2NH_Transpose`` kernel; mirrors
    :func:`get_decode_planner`'s closure pattern."""
    module = get_sglang_prefill_module()

    def chunkwise_hn2nh_transpose(
        x: torch.Tensor,
        y: torch.Tensor,
        indptr: torch.Tensor,
        batch_table: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return module.Chunkwise_HN2NH_Transpose(
            x, y, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim,
        )

    return chunkwise_hn2nh_transpose
