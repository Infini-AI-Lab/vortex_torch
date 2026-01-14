#include "register.h"

// Unified reduction kernel that reads from CPU or GPU based on slot mapping
// Uses CUDA Unified Virtual Addressing (UVA) for transparent CPU memory access
template <int BLOCK_SIZE, int REDUCE_TYPE, int DIM>
__global__ void unified_reduce_kernel(
    __nv_bfloat16* __restrict__ output,
    const int64_t* __restrict__ loc,
    const int* __restrict__ cpu_to_gpu_slot_map,
    const void* cpu_buffer_base,
    const void* gpu_buffer_base,
    const int x_D0,              // rows per page (page_size)
    const int x_D1,              // cols per page (head_dim)
    const int NUM_KV_HEAD,
    const int PAGE_SIZE,
    const int num_cpu_slots
) {
    const int token_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int tx = threadIdx.x;

    // Safety check: dimensions must be positive to prevent division by zero
    if (PAGE_SIZE <= 0 || x_D0 <= 0 || x_D1 <= 0) {
        return;
    }

    // Load token position and check page boundary
    const int64_t token_position = loc[token_id];

    // Only process at page end (page-boundary trigger)
    if ((token_position + 1) % PAGE_SIZE != 0) {
        return;
    }

    const int page_id = (token_position / PAGE_SIZE) * NUM_KV_HEAD + head_id;

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

    // Compute base pointer for this page (UVA enables CPU access)
    const __nv_bfloat16* page_base;
    if (is_cpu_page) {
        page_base = (const __nv_bfloat16*)cpu_buffer_base + (actual_slot * x_D0 * x_D1);
    } else {
        page_base = (const __nv_bfloat16*)gpu_buffer_base + (actual_slot * x_D0 * x_D1);
    }

    // Initialize accumulator based on reduction type
    float accum = (REDUCE_TYPE == 0) ? 0.0f :      // Mean
                  (REDUCE_TYPE == 1) ? -INFINITY :  // Max
                  (REDUCE_TYPE == 2) ? INFINITY :   // Min
                  0.0f;                              // L2Norm

    if (DIM == 1) {
        // Reduce over rows (axis=0) -> output length x_D1
        const int col = tx;
        if (col < x_D1) {
            // Accumulate across all rows for this column
            #pragma unroll 4
            for (int row = 0; row < x_D0; row++) {
                // Use __ldg for read-only loads (texture cache optimization)
                __nv_bfloat16 val = (is_cpu_page) ?
                    __ldg(&page_base[row * x_D1 + col]) :
                    page_base[row * x_D1 + col];

                float fval = __bfloat162float(val);

                if (REDUCE_TYPE == 0) {          // Mean
                    accum += fval;
                } else if (REDUCE_TYPE == 1) {   // Max
                    accum = fmaxf(accum, fval);
                } else if (REDUCE_TYPE == 2) {   // Min
                    accum = fminf(accum, fval);
                } else {                          // L2Norm
                    accum += fval * fval;
                }
            }

            // Finalize reduction
            if (REDUCE_TYPE == 0) {
                accum /= x_D0;  // Mean: divide by count
            } else if (REDUCE_TYPE == 3) {
                accum = sqrtf(accum);  // L2Norm: sqrt
            }

            // Write output (always to GPU memory)
            output[page_id * x_D1 + col] = __float2bfloat16(accum);
        }

    } else {  // DIM == 2
        // Reduce over cols (axis=1) -> output length x_D0
        const int row = tx;
        if (row < x_D0) {
            // Accumulate across all columns for this row
            #pragma unroll 4
            for (int col = 0; col < x_D1; col++) {
                __nv_bfloat16 val = (is_cpu_page) ?
                    __ldg(&page_base[row * x_D1 + col]) :
                    page_base[row * x_D1 + col];

                float fval = __bfloat162float(val);

                if (REDUCE_TYPE == 0) {
                    accum += fval;
                } else if (REDUCE_TYPE == 1) {
                    accum = fmaxf(accum, fval);
                } else if (REDUCE_TYPE == 2) {
                    accum = fminf(accum, fval);
                } else {
                    accum += fval * fval;
                }
            }

            if (REDUCE_TYPE == 0) {
                accum /= x_D1;
            } else if (REDUCE_TYPE == 3) {
                accum = sqrtf(accum);
            }

            output[page_id * x_D0 + row] = __float2bfloat16(accum);
        }
    }
}

// C++ wrapper function for PyBind11
void unified_reduce(
    at::Tensor& output,
    const at::Tensor& loc,
    const at::Tensor& cpu_to_gpu_slot_map,
    const int64_t cpu_buffer_base,
    const int64_t gpu_buffer_base,
    const int64_t x_D0,
    const int64_t x_D1,
    const int64_t num_kv_heads,
    const int64_t page_size,
    const int64_t reduce_type,
    const int64_t dim,
    const int64_t num_cpu_slots
) {
    const int NNZ = loc.size(0);
    const int NUM_KV_HEAD = num_kv_heads;

    // Grid matches Triton kernels: (num_tokens, num_heads)
    dim3 grid(NNZ, NUM_KV_HEAD);

    // Block size: 256 threads optimal for A100/H100
    const int block_size = 256;
    dim3 block(block_size);

    // Get current CUDA stream for proper synchronization
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Dispatch to pre-instantiated kernel templates
    // 8 variants: 4 reduce types × 2 dimensions

    if (reduce_type == 0 && dim == 1) {  // Mean, reduce rows
        unified_reduce_kernel<256, 0, 1><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 0 && dim == 2) {  // Mean, reduce cols
        unified_reduce_kernel<256, 0, 2><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 1 && dim == 1) {  // Max, reduce rows
        unified_reduce_kernel<256, 1, 1><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 1 && dim == 2) {  // Max, reduce cols
        unified_reduce_kernel<256, 1, 2><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 2 && dim == 1) {  // Min, reduce rows
        unified_reduce_kernel<256, 2, 1><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 2 && dim == 2) {  // Min, reduce cols
        unified_reduce_kernel<256, 2, 2><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 3 && dim == 1) {  // L2Norm, reduce rows
        unified_reduce_kernel<256, 3, 1><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else if (reduce_type == 3 && dim == 2) {  // L2Norm, reduce cols
        unified_reduce_kernel<256, 3, 2><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            loc.data_ptr<int64_t>(),
            (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr,
            (const void*)cpu_buffer_base,
            (const void*)gpu_buffer_base,
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots
        );
    } else {
        TORCH_CHECK(false, "Invalid reduce_type or dim combination. "
                          "reduce_type must be 0-3 (Mean/Max/Min/L2Norm), dim must be 1-2");
    }

    // No explicit synchronization needed - PyTorch handles stream ordering
}
