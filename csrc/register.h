#pragma once

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <cassert>
#include <torch/torch.h>
#include <optional>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

void sglang_plan_decode(
const at::Tensor&   cached_seq_lens,
at::Tensor&         dense_kv_indptr,
at::Tensor&         dense_kv_indices,
at::Tensor&         sparse_kv_indptr,
at::Tensor&         sparse_kv_indices,
at::Tensor&         kv_last_page_len,
const at::Tensor&   req_to_token,
const at::Tensor&   req_indices,
at::Tensor&         winfo_q_indices,
at::Tensor&         winfo_kv_offsets,
at::Tensor&         winfo_kv_lens,
at::Tensor&         winfo_num_workload,
at::Tensor&         winfo_chunk_size,
const int64_t       page_size,
const int64_t       num_kv_heads,
const int64_t       topk_val,
const int64_t       page_reserved_bos,
const int64_t       page_reserved_eos,
const int64_t       max_chunk_size,
const int64_t       min_chunk_size
);

void sglang_plan_prefill(
const at::Tensor&  cached_seq_lens,
at::Tensor&        dense_kv_indptr,
at::Tensor&        dense_kv_indices,
const at::Tensor&  input_seq_lens,
at::Tensor&        qo_indptr_ragged,
at::Tensor&        qo_indptr_paged,
at::Tensor&        kv_last_page_len,
const at::Tensor&  req_to_token,
const at::Tensor&  req_indices,
at::Tensor&        batch_table,
const int64_t      page_size,
const int64_t      num_kv_heads
);

at::Tensor Chunkwise_NH2HN_Transpose(
const at::Tensor&   x,
const at::Tensor&   indptr,
const at::Tensor&   batch_table,
const int64_t       num_qo_heads,
const int64_t       num_kv_heads,
const int64_t       head_dim
);


std::tuple<at::Tensor, at::Tensor> Chunkwise_HN2NH_Transpose(
const at::Tensor&   x,
const at::Tensor&   y,
const at::Tensor&   indptr,
const at::Tensor&   batch_table,
const int64_t       num_qo_heads,
const int64_t       num_kv_heads,
const int64_t       head_dim
);



void topk_output(
const at::Tensor&   x,
const at::Tensor&   dense_kv_indptr,
const at::Tensor&   dense_kv_indices,
const at::Tensor&   sparse_kv_indptr,
at::Tensor&         sparse_kv_indices,
const int64_t       eff_batch_size,
const int64_t       topk_val,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       max_seq_lengths
);


void sglang_plan_decode_fa3(
const at::Tensor&   cached_seq_lens,
at::Tensor&         dense_kv_indptr,
at::Tensor&         dense_kv_indices,
at::Tensor&         sparse_kv_indptr,
at::Tensor&         sparse_kv_indices,
at::Tensor&         dense_page_table,
at::Tensor&         dense_cache_seqlens,
at::Tensor&         sparse_page_table,
at::Tensor&         sparse_cache_seqlens,
const at::Tensor&   req_to_token,
const at::Tensor&   req_indices,
at::Tensor&         winfo_q_indices,
at::Tensor&         winfo_kv_offsets,
at::Tensor&         winfo_kv_lens,
at::Tensor&         winfo_num_workload,
at::Tensor&         winfo_chunk_size,
const int64_t       page_size,
const int64_t       num_kv_heads,
const int64_t       topk_val,
const int64_t       page_reserved_bos,
const int64_t       page_reserved_eos,
const int64_t       max_chunk_size,
const int64_t       min_chunk_size
);

void sglang_plan_prefill_fa3(
const at::Tensor&  cached_seq_lens,
const at::Tensor&  cu_seqlens_q,
const at::Tensor&  req_to_token,
const at::Tensor&  req_indices,
at::Tensor&        page_table,
at::Tensor&        batch_table,
const int64_t      page_size,
const int64_t      num_kv_heads
);

at::Tensor Chunkwise_HN2NH_Transpose_FA3(
const at::Tensor&   x,
const at::Tensor&   indptr,
const at::Tensor&   batch_table,
const int64_t       num_qo_heads,
const int64_t       num_kv_heads,
const int64_t       head_dim
);

// Unified CPU/GPU reduction kernel
void unified_reduce(
    at::Tensor&       output,
    const at::Tensor& loc,
    const at::Tensor& cpu_to_gpu_slot_map,
    const int64_t     cpu_buffer_base,
    const int64_t     gpu_buffer_base,
    const int64_t     x_D0,
    const int64_t     x_D1,
    const int64_t     num_kv_heads,
    const int64_t     page_size,
    const int64_t     reduce_type,
    const int64_t     dim,
    const int64_t     num_cpu_slots
);

// Unified CPU/GPU KV storage kernel
void store_kv_unified(
    const at::Tensor& cpu_k_buffer,
    const at::Tensor& cpu_v_buffer,
    const at::Tensor& gpu_k_buffer,
    const at::Tensor& gpu_v_buffer,
    const at::Tensor& cache_k_input,
    const at::Tensor& cache_v_input,
    const at::Tensor& loc,
    const at::Tensor& cpu_to_gpu_slot_map,
    const int64_t     page_size
);

// LRU allocation: block-local shared-memory, 32-way set-associative
void allocate_pages_lru_block(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_ages,
    at::Tensor set_slot_used_bitmap,
    at::Tensor needs_eviction_bitmap,
    at::Tensor dst_staging_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    at::Tensor overflow_flag,
    int32_t max_num_pages,
    int32_t MAX_HASH_ATTEMPTS
);

// LRU allocation: global device semaphores, TryLock + blocking fallback
void allocate_pages_lru_global(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_ages,
    at::Tensor set_slot_used_bitmap,
    at::Tensor needs_eviction_bitmap,
    at::Tensor dst_staging_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    at::Tensor overflow_flag,
    int32_t max_num_pages,
    const int32_t MAX_HASH_ATTEMPTS
);

// LRU allocation: block-local smem + relative ages + device semaphore global fallback
void allocate_pages_lru_block_global(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_ages,
    at::Tensor set_used_mask,
    at::Tensor dst_staging_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    at::Tensor overflow_flag,
    int32_t max_num_pages,
    const int32_t MAX_HASH_ATTEMPTS
);

// Copy kernel (grid-stride: auto-detect SM count, 256 threads per block)
void copy_kv(
    at::Tensor cpu_k_buffer,
    at::Tensor cpu_v_buffer,
    at::Tensor gpu_k_buffer,
    at::Tensor gpu_v_buffer,
    at::Tensor sparse_kv_indices,
    at::Tensor sparse_kv_indptr,
    at::Tensor dst_gpu_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    int32_t page_size,
    int32_t batch_size,
    int32_t num_kv_heads,
    int32_t max_num_pages
);
