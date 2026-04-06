#include "register.h"
#include <cuda_fp8.h>

/**
 * Gather scattered per-head pages into a contiguous multi-head ragged buffer.
 *
 * Input: paged KV cache where pages are per-head (each page stores page_size
 * tokens for one KV head). Pages may be in CPU pinned memory (UVA) or GPU.
 *
 * Output: contiguous ragged buffer with layout [total_tokens, num_kv_heads, head_dim]
 * where each request's tokens are packed contiguously.
 *
 * The kernel handles bf16, int8 (with per-token fp16 scales → bf16 output),
 * and fp8 (with per-tensor scale → bf16 output).
 *
 * Grid: (total_pages, page_size) — one thread-block per (page, token_in_page).
 * Each thread handles head_dim elements.
 */

// QUANT_TYPE: 0=bf16, 1=int8+scale, 2=fp8_e4m3, 3=fp8_e5m2
template <int QUANT_TYPE>
__global__ void __launch_bounds__(256)
gather_pages_to_ragged_kernel(
    const void* __restrict__ src_kv,           // paged KV (CPU pinned or GPU)
    const void* __restrict__ src_scale,         // per-token scales (int8 only, GPU)
    __nv_bfloat16* __restrict__ dst_buf,        // output ragged buffer [max_tokens, num_kv_heads, head_dim]
    const int32_t* __restrict__ page_indices,   // [total_pages] selected page IDs
    const int32_t* __restrict__ kv_indptr,      // [bs * num_kv_heads + 1] per-head page ranges
    const int32_t* __restrict__ dst_offsets,     // [bs] token offset in dst for each request
    int32_t total_pages,
    int32_t num_kv_heads,
    int32_t page_size,
    int32_t head_dim,
    int32_t bs,
    float kv_scale                              // per-tensor scale (fp8 only)
) {
    const int32_t page_idx = blockIdx.x;     // index into page_indices
    const int32_t token_in_page = blockIdx.y; // token within page [0, page_size)

    if (page_idx >= total_pages) return;

    // Find which (request, head) this page belongs to by binary search on kv_indptr
    // kv_indptr has bs * num_kv_heads + 1 entries
    // kv_indptr[i*num_kv_heads + h] .. kv_indptr[i*num_kv_heads + h + 1] = page range for (request i, head h)
    const int32_t total_entries = bs * num_kv_heads;
    int32_t lo = 0, hi = total_entries;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (kv_indptr[mid + 1] <= page_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // lo = flat index into (request, head) space
    const int32_t request_id = lo / num_kv_heads;
    const int32_t head_id = lo % num_kv_heads;
    const int32_t page_start_for_this_head = kv_indptr[lo];
    const int32_t page_offset_in_head = page_idx - page_start_for_this_head;

    // Source: read from paged buffer at page_indices[page_idx]
    const int32_t src_page = page_indices[page_idx];
    const int64_t src_token_offset = (int64_t)src_page * page_size + token_in_page;

    // Destination: dst_buf[dst_token, head_id, :]
    // dst_token = dst_offsets[request_id] + page_offset_in_head * page_size + token_in_page
    const int32_t dst_token = dst_offsets[request_id] + page_offset_in_head * page_size + token_in_page;
    const int64_t dst_offset = ((int64_t)dst_token * num_kv_heads + head_id) * head_dim;

    // Copy head_dim elements with optional dequantization
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val;

        if constexpr (QUANT_TYPE == 0) {
            // bf16: direct load
            const __nv_bfloat16* src = (const __nv_bfloat16*)src_kv;
            val = __bfloat162float(src[src_token_offset * head_dim + d]);
        } else if constexpr (QUANT_TYPE == 1) {
            // int8: load int8, multiply by per-token fp16 scale
            const int8_t* src = (const int8_t*)src_kv;
            const __half* scales = (const __half*)src_scale;
            int8_t raw = src[src_token_offset * head_dim + d];
            float scale = __half2float(scales[src_token_offset]);
            val = (float)raw * scale;
        } else if constexpr (QUANT_TYPE == 2) {
            // fp8 e4m3: load uint8, bitcast to fp8, multiply by per-tensor scale
            const uint8_t* src = (const uint8_t*)src_kv;
            uint8_t raw = src[src_token_offset * head_dim + d];
            __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
            val = float(fp8_val) * kv_scale;
        } else {
            // fp8 e5m2: load uint8, bitcast to fp8, multiply by per-tensor scale
            const uint8_t* src = (const uint8_t*)src_kv;
            uint8_t raw = src[src_token_offset * head_dim + d];
            __nv_fp8_e5m2 fp8_val = *reinterpret_cast<const __nv_fp8_e5m2*>(&raw);
            val = float(fp8_val) * kv_scale;
        }

        dst_buf[dst_offset + d] = __float2bfloat16(val);
    }
}


void gather_pages_to_ragged(
    at::Tensor src_kv,           // paged KV buffer (CPU pinned or GPU)
    at::Tensor dst_buf,          // output ragged buffer [max_tokens, num_kv_heads, head_dim] bf16 GPU
    at::Tensor page_indices,     // [total_pages] int32 GPU
    at::Tensor kv_indptr,        // [bs * num_kv_heads + 1] int32 GPU
    at::Tensor dst_offsets,      // [bs] int32 GPU: token offset per request
    int32_t total_pages,
    int32_t num_kv_heads,
    int32_t page_size,
    int32_t head_dim,
    int32_t bs,
    int32_t quant_type,          // 0=bf16, 1=int8, 2=fp8_e4m3, 3=fp8_e5m2
    double kv_scale,             // per-tensor scale (fp8 only)
    at::Tensor src_scale         // per-token scales (int8 only), can be empty
) {
    if (total_pages == 0) return;

    // Grid: (total_pages, page_size), each block handles one (page, token)
    // Threads handle head_dim elements
    const int threads = std::min(256, head_dim);
    dim3 grid(total_pages, page_size);
    dim3 block(threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const void* scale_ptr = (quant_type == 1 && src_scale.defined() && src_scale.numel() > 0)
        ? src_scale.data_ptr() : nullptr;

    if (quant_type == 0) {
        gather_pages_to_ragged_kernel<0><<<grid, block, 0, stream>>>(
            src_kv.data_ptr(), scale_ptr,
            reinterpret_cast<__nv_bfloat16*>(dst_buf.data_ptr<at::BFloat16>()),
            page_indices.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(),
            dst_offsets.data_ptr<int32_t>(),
            total_pages, num_kv_heads, page_size, head_dim, bs,
            (float)kv_scale);
    } else if (quant_type == 1) {
        gather_pages_to_ragged_kernel<1><<<grid, block, 0, stream>>>(
            src_kv.data_ptr(), scale_ptr,
            reinterpret_cast<__nv_bfloat16*>(dst_buf.data_ptr<at::BFloat16>()),
            page_indices.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(),
            dst_offsets.data_ptr<int32_t>(),
            total_pages, num_kv_heads, page_size, head_dim, bs,
            (float)kv_scale);
    } else if (quant_type == 2) {
        gather_pages_to_ragged_kernel<2><<<grid, block, 0, stream>>>(
            src_kv.data_ptr(), scale_ptr,
            reinterpret_cast<__nv_bfloat16*>(dst_buf.data_ptr<at::BFloat16>()),
            page_indices.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(),
            dst_offsets.data_ptr<int32_t>(),
            total_pages, num_kv_heads, page_size, head_dim, bs,
            (float)kv_scale);
    } else if (quant_type == 3) {
        gather_pages_to_ragged_kernel<3><<<grid, block, 0, stream>>>(
            src_kv.data_ptr(), scale_ptr,
            reinterpret_cast<__nv_bfloat16*>(dst_buf.data_ptr<at::BFloat16>()),
            page_indices.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(),
            dst_offsets.data_ptr<int32_t>(),
            total_pages, num_kv_heads, page_size, head_dim, bs,
            (float)kv_scale);
    } else {
        TORCH_CHECK(false, "Invalid quant_type for gather_pages_to_ragged");
    }
}
