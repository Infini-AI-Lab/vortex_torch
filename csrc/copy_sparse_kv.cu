#include "register.h"

// Single unified copy kernel with float4 vectorization
// Handles both eviction (GPU -> CPU) and fetching (CPU -> GPU)
template<typename T>
__global__ void copy_with_eviction_indptr_kernel(
    T* __restrict__ cpu_k_buffer,
    T* __restrict__ cpu_v_buffer,
    T* __restrict__ gpu_k_buffer,
    T* __restrict__ gpu_v_buffer,
    const int32_t* __restrict__ src_page_ids,
    const int32_t* __restrict__ dst_gpu_slots,
    const bool* __restrict__ owners_bitmap,
    const int32_t* __restrict__ evicted_cpu_pages,
    const int32_t* __restrict__ sparse_indptr,
    int32_t indptr_last_idx,
    int32_t page_size,
    int32_t head_dim
) {
    const int32_t num_pages = sparse_indptr[indptr_last_idx];
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;
    if (!owners_bitmap[page_idx]) return;

    const int32_t gpu_slot = dst_gpu_slots[page_idx];
    if (gpu_slot < 0) return;

    const int32_t src_cpu_page = src_page_ids[page_idx];
    const int64_t total_elements = (int64_t)page_size * head_dim;

    // Evict GPU -> CPU (if needed)
    const int32_t evicted_cpu_page = evicted_cpu_pages[page_idx];
    if (evicted_cpu_page >= 0) {
        const int64_t gpu_offset = (int64_t)gpu_slot * page_size * head_dim;
        const int64_t cpu_offset = (int64_t)evicted_cpu_page * page_size * head_dim;

        // Use float4 (16 bytes) for maximum PCIe bandwidth
        const int64_t vec_elements = (total_elements * sizeof(T)) / 16;
        const float4* src_k = reinterpret_cast<const float4*>(gpu_k_buffer + gpu_offset);
        const float4* src_v = reinterpret_cast<const float4*>(gpu_v_buffer + gpu_offset);
        float4* dst_k = reinterpret_cast<float4*>(cpu_k_buffer + cpu_offset);
        float4* dst_v = reinterpret_cast<float4*>(cpu_v_buffer + cpu_offset);

        for (int64_t i = threadIdx.x; i < vec_elements; i += blockDim.x) {
            dst_k[i] = src_k[i];
            dst_v[i] = src_v[i];
        }
    }

    // Fetch CPU -> GPU
    const int64_t src_cpu_offset = (int64_t)src_cpu_page * page_size * head_dim;
    const int64_t dst_gpu_offset = (int64_t)gpu_slot * page_size * head_dim;

    // Use float4 (16 bytes) for maximum PCIe bandwidth
    const int64_t vec_elements = (total_elements * sizeof(T)) / 16;
    const float4* fetch_src_k = reinterpret_cast<const float4*>(cpu_k_buffer + src_cpu_offset);
    const float4* fetch_src_v = reinterpret_cast<const float4*>(cpu_v_buffer + src_cpu_offset);
    float4* fetch_dst_k = reinterpret_cast<float4*>(gpu_k_buffer + dst_gpu_offset);
    float4* fetch_dst_v = reinterpret_cast<float4*>(gpu_v_buffer + dst_gpu_offset);

    for (int64_t i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        fetch_dst_k[i] = fetch_src_k[i];
        fetch_dst_v[i] = fetch_src_v[i];
    }
}

// Copy kernel - single kernel on current stream
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
) {
    const int32_t head_dim = cpu_k_buffer.size(-1);
    const int32_t indptr_last_idx = batch_size * num_kv_heads;
    const int threads = 512;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (cpu_k_buffer.dtype() == torch::kBFloat16) {
        copy_with_eviction_indptr_kernel<at::BFloat16><<<max_num_pages, threads, 0, stream>>>(
            cpu_k_buffer.data_ptr<at::BFloat16>(),
            cpu_v_buffer.data_ptr<at::BFloat16>(),
            gpu_k_buffer.data_ptr<at::BFloat16>(),
            gpu_v_buffer.data_ptr<at::BFloat16>(),
            sparse_kv_indices.data_ptr<int32_t>(),
            dst_gpu_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            sparse_kv_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            page_size, head_dim
        );
    } else if (cpu_k_buffer.dtype() == torch::kFloat16) {
        copy_with_eviction_indptr_kernel<at::Half><<<max_num_pages, threads, 0, stream>>>(
            cpu_k_buffer.data_ptr<at::Half>(),
            cpu_v_buffer.data_ptr<at::Half>(),
            gpu_k_buffer.data_ptr<at::Half>(),
            gpu_v_buffer.data_ptr<at::Half>(),
            sparse_kv_indices.data_ptr<int32_t>(),
            dst_gpu_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            sparse_kv_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            page_size, head_dim
        );
    } else if (cpu_k_buffer.dtype() == torch::kFloat32) {
        copy_with_eviction_indptr_kernel<float><<<max_num_pages, threads, 0, stream>>>(
            cpu_k_buffer.data_ptr<float>(),
            cpu_v_buffer.data_ptr<float>(),
            gpu_k_buffer.data_ptr<float>(),
            gpu_v_buffer.data_ptr<float>(),
            sparse_kv_indices.data_ptr<int32_t>(),
            dst_gpu_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            sparse_kv_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            page_size, head_dim
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for copy_kv");
    }
}
