import torch
from vortex_torch_C import (
sglang_plan_decode, 
sglang_plan_prefill, 
sglang_plan_decode_fa3,
sglang_plan_prefill_fa3, 
Chunkwise_NH2HN_Transpose, 
Chunkwise_HN2NH_Transpose,
Chunkwise_HN2NH_Transpose_FA3
)
from typing import Tuple
from .context import Context


def plan_decode(
cached_seq_lens: torch.Tensor,
req_to_token: torch.Tensor,
req_indices: torch.Tensor,
ctx: Context
):
    sglang_plan_decode(
        cached_seq_lens,
        ctx.dense_kv_indptr,
        ctx.dense_kv_indices,
        ctx.sparse_kv_indptr,
        ctx.sparse_kv_indices,
        ctx.kv_last_page_len,
        req_to_token,
        req_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.winfo_chunk_size,
        ctx.page_size,
        ctx.num_kv_heads,
        ctx.topk_val,
        ctx.page_reserved_bos,
        ctx.page_reserved_eos,
        ctx.max_chunk_size,
        ctx.min_chunk_size
    )

    ctx.set_batch_size(cached_seq_lens.shape[0])


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
num_kv_heads: int
):
    sglang_plan_prefill(
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
        num_kv_heads
    )


def chunkwise_nh2hn_transpose(
x: torch.Tensor,
indptr: torch.Tensor,
batch_table: torch.Tensor,
num_qo_heads: int,
num_kv_heads: int,
head_dim: int,
) -> torch.Tensor:
    
    
    x_t = Chunkwise_NH2HN_Transpose(
        x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim
    )
    
    return x_t




def chunkwise_hn2nh_transpose(
x: torch.Tensor,
y: torch.Tensor,
indptr: torch.Tensor,
batch_table: torch.Tensor,
num_qo_heads: int,
num_kv_heads: int,
head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    x_t, y_t = Chunkwise_HN2NH_Transpose(
            x, y, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim
        )
    
    return x_t, y_t


def plan_prefill_fa3(
cached_seq_lens: torch.Tensor,
cu_seqlens_q: torch.Tensor,
req_to_token: torch.Tensor,
req_indices: torch.Tensor,
page_table: torch.Tensor,
batch_table: torch.Tensor,
page_size: int,
num_kv_heads: int
):
    sglang_plan_prefill_fa3(
        cached_seq_lens,
        cu_seqlens_q,
        req_to_token,
        req_indices,
        page_table,
        batch_table,
        page_size,
        num_kv_heads
    )
    

def plan_decode_fa3(
cached_seq_lens: torch.Tensor,
req_to_token: torch.Tensor,
req_indices: torch.Tensor,
dense_page_table: torch.Tensor,
dense_cache_seqlens: torch.Tensor,
sparse_page_table: torch.Tensor,
sparse_cache_seqlens: torch.Tensor,
ctx: Context
):
    sglang_plan_decode_fa3(
        cached_seq_lens,
        ctx.dense_kv_indptr,
        ctx.dense_kv_indices,
        ctx.sparse_kv_indptr,
        ctx.sparse_kv_indices,
        dense_page_table,
        dense_cache_seqlens,
        sparse_page_table,
        sparse_cache_seqlens,
        req_to_token,
        req_indices,
        ctx.winfo_q_indices,
        ctx.winfo_kv_offsets,
        ctx.winfo_kv_lens,
        ctx.winfo_num_workloads,
        ctx.winfo_chunk_size,
        ctx.page_size,
        ctx.num_kv_heads,
        ctx.topk_val,
        ctx.page_reserved_bos,
        ctx.page_reserved_eos,
        ctx.max_chunk_size,
        ctx.min_chunk_size
    )
    
    ctx.set_batch_size(cached_seq_lens.shape[0])
   

def chunkwise_hn2nh_transpose_fa3(
x: torch.Tensor,
indptr: torch.Tensor,
batch_table: torch.Tensor,
num_qo_heads: int,
num_kv_heads: int,
head_dim: int,
) -> torch.Tensor:
    
    x_t = Chunkwise_HN2NH_Transpose_FA3(
            x, indptr, batch_table, num_qo_heads, num_kv_heads, head_dim
        )
    
    return x_t


def indices_to_page_table(
page_table: torch.Tensor,
ctx: Context
):
    pass