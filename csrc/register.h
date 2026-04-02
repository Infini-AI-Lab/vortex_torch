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

void topk_output_sglang(
const at::Tensor&   x,
const at::Tensor&   dense_kv_indptr,
const at::Tensor&   sparse_kv_indptr,
const at::Tensor&   dense_kv_indices,
at::Tensor&         sparse_kv_indices,
const int64_t       eff_batch_size,
const int64_t       topk_val,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       max_seq_lengths,
const int64_t       mapping_mode = 0,
const double        mapping_power = 0.5,
std::optional<at::Tensor> mapping_lut = std::nullopt,
std::optional<at::Tensor> mapping_quantiles = std::nullopt,
const bool          mapping_noscale = false
);

void topk_profile_histogram(
const at::Tensor&   x,
const at::Tensor&   dense_kv_indptr,
at::Tensor&         histograms,
const int64_t       eff_batch_size,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       mapping_mode = 0,
const double        mapping_power = 0.5,
std::optional<at::Tensor> mapping_lut = std::nullopt,
std::optional<at::Tensor> mapping_quantiles = std::nullopt,
const bool          mapping_noscale = false,
const int64_t       topk_val = 0
);

void topk_profile_stage1(
const at::Tensor&   x,
const at::Tensor&   dense_kv_indptr,
const at::Tensor&   sparse_kv_indptr,
const at::Tensor&   dense_kv_indices,
at::Tensor&         sparse_kv_indices,
const int64_t       eff_batch_size,
const int64_t       topk_val,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       max_num_pages,
const int64_t       mapping_mode = 0,
const double        mapping_power = 0.5,
std::optional<at::Tensor> mapping_lut = std::nullopt,
std::optional<at::Tensor> mapping_quantiles = std::nullopt,
const bool          mapping_noscale = false
);

void topk_profile_counters(
const at::Tensor&   x,
const at::Tensor&   dense_kv_indptr,
const at::Tensor&   sparse_kv_indptr,
const at::Tensor&   dense_kv_indices,
at::Tensor&         sparse_kv_indices,
at::Tensor&         counters,
const int64_t       eff_batch_size,
const int64_t       topk_val,
const int64_t       reserved_bos,
const int64_t       reserved_eos,
const int64_t       max_num_pages,
const int64_t       mapping_mode = 0,
const double        mapping_power = 0.5,
std::optional<at::Tensor> mapping_lut = std::nullopt,
std::optional<at::Tensor> mapping_quantiles = std::nullopt,
const bool          mapping_noscale = false
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
