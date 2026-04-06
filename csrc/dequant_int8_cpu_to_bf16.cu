#include "register.h"

/**
 * Dequantize int8 pages from CPU pinned memory to GPU bf16 destination.
 *
 * Reads int8 data from CPU pinned memory via UVA, reads fp16 scales from GPU,
 * computes bf16 = int8 * scale, and writes to GPU destination buffer.
 *
 * Supports separate source (CPU page IDs) and destination (GPU page IDs) indexing,
 * enabling both in-place (forward_cache) and compact (extend gather) layouts.
 *
 * Grid: one block per page, threads handle tokens within the page.
 */
__global__ void __launch_bounds__(256)
dequant_int8_cpu_to_bf16_kernel(
    const int8_t* __restrict__ cpu_int8_buffer,   // CPU pinned int8 [num_cpu_pages, page_size, head_dim]
    const __half* __restrict__ gpu_scale_buffer,   // GPU fp16 [num_cpu_pages, page_size] (1 scale per token)
    __nv_bfloat16* __restrict__ gpu_dst_buffer,    // GPU bf16 destination
    const int32_t* __restrict__ src_page_ids,      // [num_pages] which CPU pages to read
    const int32_t* __restrict__ dst_page_ids,      // [num_pages] where to write in destination
    int32_t num_pages,
    int32_t page_size,
    int32_t head_dim
) {
    const int64_t tokens_per_page = page_size;
    const int64_t elems_per_page = tokens_per_page * head_dim;

    for (int32_t page_idx = blockIdx.x; page_idx < num_pages; page_idx += gridDim.x) {
        const int32_t src_page = src_page_ids[page_idx];
        const int32_t dst_page = dst_page_ids[page_idx];

        const int64_t src_base = (int64_t)src_page * elems_per_page;
        const int64_t dst_base = (int64_t)dst_page * elems_per_page;
        const int64_t scale_base = (int64_t)src_page * tokens_per_page;

        // Each thread handles multiple elements across tokens in this page
        for (int64_t i = threadIdx.x; i < elems_per_page; i += blockDim.x) {
            // Determine which token within the page this element belongs to
            const int32_t token_in_page = (int32_t)(i / head_dim);

            // Load int8 from CPU pinned memory (UVA)
            float val = (float)cpu_int8_buffer[src_base + i];

            // Load scale from GPU (one scale per token per head)
            float scale = __half2float(gpu_scale_buffer[scale_base + token_in_page]);

            // Dequantize and store as bf16
            gpu_dst_buffer[dst_base + i] = __float2bfloat16(val * scale);
        }
    }
}


static int get_dequant_num_blocks() {
    static int num_blocks = 0;
    if (num_blocks == 0) {
        int device;
        cudaGetDevice(&device);
        int sm_count;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        num_blocks = (sm_count < 256) ? 256 : ((sm_count < 512) ? 512 : 1024);
    }
    return num_blocks;
}


void dequant_int8_cpu_to_bf16(
    at::Tensor cpu_int8_buffer,    // CPU pinned int8
    at::Tensor gpu_scale_buffer,   // GPU fp16
    at::Tensor gpu_dst_buffer,     // GPU bf16 destination
    at::Tensor src_page_ids,       // int32 GPU tensor: which CPU pages to read
    at::Tensor dst_page_ids,       // int32 GPU tensor: where to write in dst
    int32_t page_size,
    int32_t head_dim
) {
    TORCH_CHECK(cpu_int8_buffer.dtype() == torch::kInt8, "cpu_int8_buffer must be int8");
    TORCH_CHECK(gpu_scale_buffer.dtype() == torch::kFloat16, "gpu_scale_buffer must be fp16");
    TORCH_CHECK(gpu_dst_buffer.dtype() == torch::kBFloat16, "gpu_dst_buffer must be bf16");
    TORCH_CHECK(src_page_ids.dtype() == torch::kInt32, "src_page_ids must be int32");
    TORCH_CHECK(dst_page_ids.dtype() == torch::kInt32, "dst_page_ids must be int32");

    const int32_t num_pages = src_page_ids.size(0);
    if (num_pages == 0) return;

    const int threads = 256;
    const int num_blocks = std::min(get_dequant_num_blocks(), num_pages);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dequant_int8_cpu_to_bf16_kernel<<<num_blocks, threads, 0, stream>>>(
        cpu_int8_buffer.data_ptr<int8_t>(),
        reinterpret_cast<const __half*>(gpu_scale_buffer.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(gpu_dst_buffer.data_ptr<at::BFloat16>()),
        src_page_ids.data_ptr<int32_t>(),
        dst_page_ids.data_ptr<int32_t>(),
        num_pages,
        page_size,
        head_dim
    );
}
