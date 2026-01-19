#include "register.h"

// Unified KV storage kernel that writes to CPU or GPU based on slot mapping
// Uses CUDA Unified Virtual Addressing (UVA) for transparent CPU memory access
template <int BLOCK_SIZE>
__global__ void store_kv_unified_kernel(
    const __nv_bfloat16* __restrict__ cache_k_input,
    const __nv_bfloat16* __restrict__ cache_v_input,
    const int64_t* __restrict__ loc,
    const int* __restrict__ cpu_to_gpu_slot_map,
    void* cpu_k_base,
    void* cpu_v_base,
    void* gpu_k_base,
    void* gpu_v_base,
    const int page_size,
    const int head_dim,
    const int num_kv_heads,
    const int num_cpu_slots
) {
    const int token_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int tx = threadIdx.x;

    // Safety check: dimensions must be positive to prevent division by zero
    if (page_size <= 0 || head_dim <= 0) {
        return;
    }

    // Load token position
    const int64_t token_position = loc[token_id];
    const int page_idx = token_position / page_size;
    const int offset_in_page = token_position % page_size;

    // Compute page_id including head dimension
    const int page_id = page_idx * num_kv_heads + head_id;

    // CPU/GPU routing logic via slot map
    // Convention: mapped_slot >= 0 means page is in GPU at that slot
    //            mapped_slot == -1 means page is in CPU
    bool is_cpu_page = false;
    int actual_slot = page_id;

    if (cpu_to_gpu_slot_map != nullptr) {
        int mapped_slot = cpu_to_gpu_slot_map[page_id];
        if (mapped_slot >= 0) {
            // Page is in GPU at the mapped slot
            is_cpu_page = false;
            actual_slot = mapped_slot;
        } else {
            // Page is in CPU, use page_id directly as CPU slot
            is_cpu_page = true;
            actual_slot = page_id;
        }
    }

    // Compute base pointers for this page
    __nv_bfloat16* k_page_base;
    __nv_bfloat16* v_page_base;

    if (is_cpu_page) {
        k_page_base = ((__nv_bfloat16*)cpu_k_base) + ((int64_t)actual_slot * page_size * head_dim);
        v_page_base = ((__nv_bfloat16*)cpu_v_base) + ((int64_t)actual_slot * page_size * head_dim);
    } else {
        k_page_base = ((__nv_bfloat16*)gpu_k_base) + ((int64_t)actual_slot * page_size * head_dim);
        v_page_base = ((__nv_bfloat16*)gpu_v_base) + ((int64_t)actual_slot * page_size * head_dim);
    }

    // Compute source offset in input tensors
    // Input layout: [num_tokens, num_heads, head_dim]
    const int64_t input_offset = ((int64_t)token_id * num_kv_heads + head_id) * head_dim;

    // Compute destination offset in page
    // Page layout: [page_size, head_dim]
    const int page_offset = offset_in_page * head_dim;

    // Each thread copies multiple elements using grid-stride loop
    for (int i = tx; i < head_dim; i += BLOCK_SIZE) {
        __nv_bfloat16 k_val = cache_k_input[input_offset + i];
        __nv_bfloat16 v_val = cache_v_input[input_offset + i];

        // Write to destination (CPU or GPU)
        k_page_base[page_offset + i] = k_val;
        v_page_base[page_offset + i] = v_val;
    }
}

// C++ wrapper function for PyBind11
void store_kv_unified(
    const at::Tensor& cpu_k_buffer,
    const at::Tensor& cpu_v_buffer,
    const at::Tensor& gpu_k_buffer,
    const at::Tensor& gpu_v_buffer,
    const at::Tensor& cache_k_input,
    const at::Tensor& cache_v_input,
    const at::Tensor& loc,
    const at::Tensor& cpu_to_gpu_slot_map,
    const int64_t page_size
) {
    // Validate inputs
    TORCH_CHECK(cache_k_input.dtype() == torch::kBFloat16, "cache_k must be bfloat16");
    TORCH_CHECK(cache_v_input.dtype() == torch::kBFloat16, "cache_v must be bfloat16");
    TORCH_CHECK(loc.dtype() == torch::kInt64, "loc must be int64");
    TORCH_CHECK(cpu_to_gpu_slot_map.dtype() == torch::kInt32, "slot_map must be int32");

    TORCH_CHECK(cache_k_input.dim() == 3, "cache_k must be 3D [tokens, heads, head_dim]");
    TORCH_CHECK(cache_v_input.dim() == 3, "cache_v must be 3D [tokens, heads, head_dim]");

    const int num_tokens = cache_k_input.size(0);
    const int num_kv_heads = cache_k_input.size(1);
    const int head_dim = cache_k_input.size(2);

    TORCH_CHECK(cache_v_input.size(0) == num_tokens, "cache_v tokens mismatch");
    TORCH_CHECK(cache_v_input.size(1) == num_kv_heads, "cache_v heads mismatch");
    TORCH_CHECK(cache_v_input.size(2) == head_dim, "cache_v head_dim mismatch");
    TORCH_CHECK(loc.size(0) == num_tokens, "loc size mismatch");

    // Validate buffer shapes
    if (cpu_k_buffer.defined() && cpu_k_buffer.numel() > 0) {
        TORCH_CHECK(cpu_k_buffer.dim() == 3, "cpu_k_buffer must be 3D");
        TORCH_CHECK(cpu_k_buffer.size(1) == page_size, "cpu_k_buffer page_size mismatch");
        TORCH_CHECK(cpu_k_buffer.size(2) == head_dim, "cpu_k_buffer head_dim mismatch");
        TORCH_CHECK(cpu_k_buffer.is_pinned(), "cpu_k_buffer must be pinned memory");
    }

    if (cpu_v_buffer.defined() && cpu_v_buffer.numel() > 0) {
        TORCH_CHECK(cpu_v_buffer.dim() == 3, "cpu_v_buffer must be 3D");
        TORCH_CHECK(cpu_v_buffer.size(1) == page_size, "cpu_v_buffer page_size mismatch");
        TORCH_CHECK(cpu_v_buffer.size(2) == head_dim, "cpu_v_buffer head_dim mismatch");
        TORCH_CHECK(cpu_v_buffer.is_pinned(), "cpu_v_buffer must be pinned memory");
    }

    TORCH_CHECK(gpu_k_buffer.dim() == 3, "gpu_k_buffer must be 3D");
    TORCH_CHECK(gpu_k_buffer.size(1) == page_size, "gpu_k_buffer page_size mismatch");
    TORCH_CHECK(gpu_k_buffer.size(2) == head_dim, "gpu_k_buffer head_dim mismatch");
    TORCH_CHECK(gpu_k_buffer.is_cuda(), "gpu_k_buffer must be on CUDA");

    TORCH_CHECK(gpu_v_buffer.dim() == 3, "gpu_v_buffer must be 3D");
    TORCH_CHECK(gpu_v_buffer.size(1) == page_size, "gpu_v_buffer page_size mismatch");
    TORCH_CHECK(gpu_v_buffer.size(2) == head_dim, "gpu_v_buffer head_dim mismatch");
    TORCH_CHECK(gpu_v_buffer.is_cuda(), "gpu_v_buffer must be on CUDA");

    // Get buffer base pointers
    void* cpu_k_base = (cpu_k_buffer.defined() && cpu_k_buffer.numel() > 0)
        ? (void*)cpu_k_buffer.data_ptr<at::BFloat16>()
        : nullptr;
    void* cpu_v_base = (cpu_v_buffer.defined() && cpu_v_buffer.numel() > 0)
        ? (void*)cpu_v_buffer.data_ptr<at::BFloat16>()
        : nullptr;
    void* gpu_k_base = (void*)gpu_k_buffer.data_ptr<at::BFloat16>();
    void* gpu_v_base = (void*)gpu_v_buffer.data_ptr<at::BFloat16>();

    const int num_cpu_slots = (cpu_k_buffer.defined() && cpu_k_buffer.numel() > 0)
        ? cpu_k_buffer.size(0)
        : 0;

    // Launch configuration
    dim3 grid(num_tokens, num_kv_heads);
    const int block_size = 256;
    dim3 block(block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    store_kv_unified_kernel<256><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(cache_k_input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cache_v_input.data_ptr<at::BFloat16>()),
        loc.data_ptr<int64_t>(),
        (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
        cpu_k_base,
        cpu_v_base,
        gpu_k_base,
        gpu_v_base,
        page_size,
        head_dim,
        num_kv_heads,
        num_cpu_slots
    );

    // No explicit synchronization needed - PyTorch handles stream ordering
}
