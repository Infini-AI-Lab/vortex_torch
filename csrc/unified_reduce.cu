#include "register.h"
#include <cuda_fp8.h>

// Load a value from memory and convert to float, handling bf16 and fp8 dtypes.
// QUANT_TYPE: 0=bf16, 1=fp8_e4m3, 2=fp8_e5m2
template <int QUANT_TYPE>
__device__ __forceinline__ float load_and_dequant(
    const void* base, int64_t offset, float kv_scale, bool is_cpu_page
) {
    if constexpr (QUANT_TYPE == 0) {
        // bf16
        const __nv_bfloat16* ptr = (const __nv_bfloat16*)base + offset;
        __nv_bfloat16 val = is_cpu_page ? __ldg(ptr) : *ptr;
        return __bfloat162float(val);
    } else if constexpr (QUANT_TYPE == 1) {
        // fp8 e4m3 stored as uint8
        const uint8_t* ptr = (const uint8_t*)base + offset;
        uint8_t raw = is_cpu_page ? __ldg(ptr) : *ptr;
        __nv_fp8_e4m3 fp8_val = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        return float(fp8_val) * kv_scale;
    } else {
        // fp8 e5m2 stored as uint8
        const uint8_t* ptr = (const uint8_t*)base + offset;
        uint8_t raw = is_cpu_page ? __ldg(ptr) : *ptr;
        __nv_fp8_e5m2 fp8_val = *reinterpret_cast<const __nv_fp8_e5m2*>(&raw);
        return float(fp8_val) * kv_scale;
    }
}

// Unified reduction kernel that reads from CPU or GPU based on slot mapping.
// Uses CUDA Unified Virtual Addressing (UVA) for transparent CPU memory access.
// Supports bf16 (QUANT_TYPE=0), fp8_e4m3 (1), and fp8_e5m2 (2).
template <int BLOCK_SIZE, int REDUCE_TYPE, int DIM, int QUANT_TYPE>
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
    const int num_cpu_slots,
    const float kv_scale
) {
    const int token_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int tx = threadIdx.x;

    if (PAGE_SIZE <= 0 || x_D0 <= 0 || x_D1 <= 0) {
        return;
    }

    const int64_t token_position = loc[token_id];

    // Only process at page end (page-boundary trigger)
    if ((token_position + 1) % PAGE_SIZE != 0) {
        return;
    }

    const int page_id = (token_position / PAGE_SIZE) * NUM_KV_HEAD + head_id;

    // CPU/GPU routing logic via slot map
    bool is_cpu_page = false;
    int actual_slot = page_id;

    if (cpu_to_gpu_slot_map != nullptr) {
        int mapped_slot = cpu_to_gpu_slot_map[page_id];
        if (mapped_slot >= 0) {
            is_cpu_page = false;
            actual_slot = mapped_slot;
        } else {
            is_cpu_page = true;
            actual_slot = page_id;
        }
    }

    // Element size depends on quant type: bf16=2, fp8=1
    constexpr int elem_size = (QUANT_TYPE == 0) ? sizeof(__nv_bfloat16) : sizeof(uint8_t);

    // Compute base byte offset for this page
    const void* page_base;
    if (is_cpu_page) {
        page_base = (const char*)cpu_buffer_base + ((int64_t)actual_slot * x_D0 * x_D1 * elem_size);
    } else {
        page_base = (const char*)gpu_buffer_base + ((int64_t)actual_slot * x_D0 * x_D1 * elem_size);
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
            #pragma unroll 4
            for (int row = 0; row < x_D0; row++) {
                float fval = load_and_dequant<QUANT_TYPE>(
                    page_base, (int64_t)row * x_D1 + col, kv_scale, is_cpu_page);

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
                accum /= x_D0;
            } else if (REDUCE_TYPE == 3) {
                accum = sqrtf(accum);
            }

            output[(int64_t)page_id * x_D1 + col] = __float2bfloat16(accum);
        }

    } else {  // DIM == 2
        // Reduce over cols (axis=1) -> output length x_D0
        const int row = tx;
        if (row < x_D0) {
            #pragma unroll 4
            for (int col = 0; col < x_D1; col++) {
                float fval = load_and_dequant<QUANT_TYPE>(
                    page_base, (int64_t)row * x_D1 + col, kv_scale, is_cpu_page);

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

            output[(int64_t)page_id * x_D0 + row] = __float2bfloat16(accum);
        }
    }
}


// Helper macro: dispatch one (reduce_type, dim) combo across all quant types
#define DISPATCH_QUANT(RT, DM) \
    if (quant_type == 0) { \
        unified_reduce_kernel<256, RT, DM, 0><<<grid, block, 0, stream>>>( \
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), \
            loc.data_ptr<int64_t>(), slot_map_ptr, \
            (const void*)cpu_buffer_base, (const void*)gpu_buffer_base, \
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots, kv_scale); \
    } else if (quant_type == 1) { \
        unified_reduce_kernel<256, RT, DM, 1><<<grid, block, 0, stream>>>( \
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), \
            loc.data_ptr<int64_t>(), slot_map_ptr, \
            (const void*)cpu_buffer_base, (const void*)gpu_buffer_base, \
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots, kv_scale); \
    } else { \
        unified_reduce_kernel<256, RT, DM, 2><<<grid, block, 0, stream>>>( \
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), \
            loc.data_ptr<int64_t>(), slot_map_ptr, \
            (const void*)cpu_buffer_base, (const void*)gpu_buffer_base, \
            x_D0, x_D1, NUM_KV_HEAD, page_size, num_cpu_slots, kv_scale); \
    }


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
    const int64_t num_cpu_slots,
    const int64_t quant_type,
    const double kv_scale
) {
    const int NNZ = loc.size(0);
    const int NUM_KV_HEAD = num_kv_heads;

    dim3 grid(NNZ, NUM_KV_HEAD);
    const int block_size = 256;
    dim3 block(block_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int* slot_map_ptr = (num_cpu_slots > 0) ? cpu_to_gpu_slot_map.data_ptr<int>() : nullptr;

    // Dispatch: 4 reduce types × 2 dims × 3 quant types = 24 template instantiations
    if (reduce_type == 0 && dim == 1) { DISPATCH_QUANT(0, 1); }
    else if (reduce_type == 0 && dim == 2) { DISPATCH_QUANT(0, 2); }
    else if (reduce_type == 1 && dim == 1) { DISPATCH_QUANT(1, 1); }
    else if (reduce_type == 1 && dim == 2) { DISPATCH_QUANT(1, 2); }
    else if (reduce_type == 2 && dim == 1) { DISPATCH_QUANT(2, 1); }
    else if (reduce_type == 2 && dim == 2) { DISPATCH_QUANT(2, 2); }
    else if (reduce_type == 3 && dim == 1) { DISPATCH_QUANT(3, 1); }
    else if (reduce_type == 3 && dim == 2) { DISPATCH_QUANT(3, 2); }
    else {
        TORCH_CHECK(false, "Invalid reduce_type or dim combination. "
                          "reduce_type must be 0-3 (Mean/Max/Min/L2Norm), dim must be 1-2");
    }
}

#undef DISPATCH_QUANT
