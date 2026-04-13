// Archived: not compiled. See csrc/archived/README.md
//
// Flexible-radix (RADIX_BITS 4..10) "ori fast path" for TopK. It was the
// zero-mapping-overhead fast path used when mapping_mode == MAPPING_NONE.
// No longer tested — mode 0 now routes through the fused TopKOutput_Kernel
// with mapping.mode == MAPPING_NONE, which pays no extra cost because
// mapped_convert_to_uint8 collapses to convert_to_uint8 in that branch.
//
// The code below was extracted verbatim from csrc/topk_sglang.cu as of the
// fused-kernel refactor. It references helpers (kSmem, convert_to_uint32,
// vortex_to_float, VORTEX_MAX_TOPK, kThreadsPerBlock, setup_kernel_smem_once,
// CHECK_CUDA, topk_mapping.cuh types) from the surrounding translation unit.
// Dropping this file into a build as-is will not compile; it is reference
// only.

template <int BITS>
__device__ __forceinline__ uint16_t convert_to_uintN(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return key >> (16 - BITS);
}

// ======================================================================
// Ori fast path: zero-overhead topk with no mapping infrastructure.
// Template on RADIX_BITS: 4-10 (16 to 1024 bins).
// ======================================================================
template <typename ScoreT, int RADIX_BITS = 8>
__device__ void fast_topk_ori(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k)
{
    int topk = target_k;
    constexpr auto BLOCK_SIZE = 1024;
    constexpr auto RADIX = 1 << RADIX_BITS;
    constexpr auto RADIX_PAD = RADIX / 2;
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));
    static_assert(RADIX_BITS >= 4 && RADIX_BITS <= 10, "RADIX_BITS must be 4-10");
    static_assert(RADIX <= BLOCK_SIZE, "RADIX must not exceed BLOCK_SIZE");

    alignas(128) __shared__ int s_histogram_buf[2][RADIX + RADIX_PAD];
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_threshold_bin_id;
    alignas(128) __shared__ int s_num_input[2];

    auto& s_histogram = s_histogram_buf[0];
    extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;

    // Stage 1: coarse histogram with RADIX bins
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = convert_to_uintN<RADIX_BITS>(vortex_to_float(input[idx + row_start]));
        ::atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    const auto run_cumsum = [&] {
        for (int i = 0; i < RADIX_BITS; ++i) {
            if (C10_LIKELY(tx < RADIX)) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = s_histogram_buf[k][tx];
                if (tx < RADIX - j) {
                    value += s_histogram_buf[k][tx + j];
                }
                s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };
    // Stage 2 cumsum: always 256 sub-bins (8-bit radix on raw float bits)
    const auto run_cumsum_s2 = [&] {
        for (int i = 0; i < 8; ++i) {
            if (C10_LIKELY(tx < 256)) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = s_histogram_buf[k][tx];
                if (tx < 256 - j) {
                    value += s_histogram_buf[k][tx + j];
                }
                s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto bin = static_cast<int>(convert_to_uintN<RADIX_BITS>(vortex_to_float(input[idx + row_start])));
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        return;
    } else {
        __syncthreads();
        if (tx < 257) s_histogram[tx] = 0;
        __syncthreads();

        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto raw_input = vortex_to_float(input[idx + row_start]);
            const auto bin = static_cast<int>(convert_to_uintN<RADIX_BITS>(raw_input));
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const auto pos = ::atomicAdd(&s_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    s_input_idx[0][pos] = idx;
                    const auto b32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (b32 >> 24) & 0xFF;
                    ::atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }

    // Stage 2: refine with 8-bit radix passes
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        __shared__ int s_last_remain;
        const auto r_idx = round % 2;

        const auto _raw_num_input = s_num_input[r_idx];
        const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

        run_cumsum_s2();
        if (tx < 256 && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
            s_threshold_bin_id = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = topk - s_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        topk -= s_histogram[threshold_bin + 1];

        if (topk == 0) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = s_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    index[pos] = idx;
                }
            }
            __syncthreads();
            break;
        } else {
            __syncthreads();
            if (tx < 257) s_histogram[tx] = 0;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = s_input_idx[r_idx][i];
                const auto raw_input = vortex_to_float(input[idx + row_start]);
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&s_counter, 1);
                    index[pos] = idx;
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&s_last_remain, -1);
                        if (pos > 0) {
                            index[target_k - pos] = idx;
                        }
                    } else {
                        const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
                        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                            s_input_idx[r_idx ^ 1][pos] = idx;
                            const auto b32 = convert_to_uint32(raw_input);
                            const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&s_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Ori fast-path wrapper: zero mapping overhead, flexible radix
template <typename ScoreT, int RADIX_BITS = 8>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutput_Ori_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     topk_val,
    const int     page_reserved_bos,
    const int     page_reserved_eos)
{
    const int bx = blockIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const ScoreT* __restrict__ score_blk = score + start;
    const int*    __restrict__ idx_blk   = dense_kv_indices + start;
    int*          __restrict__ out_blk   = sparse_kv_indices
                                         + sparse_kv_indptr[bx]
                                         + page_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];
    fast_topk_ori<ScoreT, RADIX_BITS>(score_blk, s_indices, 0, nblk, topk_val);
    __syncthreads();

    const int tx = threadIdx.x;
    for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

// Helper: launch TopKOutput_Ori_Kernel with radix_bits dispatch
template <typename ScoreT>
void launch_ori_kernel(
    const ScoreT* score, const int* dense_kv_indptr, const int* sparse_kv_indptr,
    const int* dense_kv_indices, int* sparse_kv_indices,
    int topk_val, int reserved_bos, int reserved_eos,
    int radix_bits, dim3 nblks, dim3 nthreads, cudaStream_t stream)
{
    #define LAUNCH_ORI(BITS) \
        setup_kernel_smem_once<TopKOutput_Ori_Kernel<ScoreT, BITS>, kSmem>(); \
        TopKOutput_Ori_Kernel<ScoreT, BITS><<<nblks, nthreads, kSmem, stream>>>( \
            score, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices, sparse_kv_indices, \
            topk_val, reserved_bos, reserved_eos)
    switch (radix_bits) {
        case 4:  LAUNCH_ORI(4);  break;
        case 5:  LAUNCH_ORI(5);  break;
        case 6:  LAUNCH_ORI(6);  break;
        case 7:  LAUNCH_ORI(7);  break;
        case 9:  LAUNCH_ORI(9);  break;
        case 10: LAUNCH_ORI(10); break;
        default: LAUNCH_ORI(8);  break;
    }
    #undef LAUNCH_ORI
}

// ======================================================================
// Explicit ori baseline entry point — always uses the ori fast path
// ======================================================================
void topk_output_sglang_ori(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    const int64_t     eff_batch_size,
    const int64_t     topk_val,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const int64_t     radix_bits)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_output_sglang_ori: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");
    TORCH_CHECK(radix_bits >= 4 && radix_bits <= 10,
                "topk_output_sglang_ori: radix_bits must be 4-10, got ", radix_bits);

    CHECK_CUDA(x);
    CHECK_CUDA(dense_kv_indptr);
    CHECK_CUDA(sparse_kv_indptr);
    CHECK_CUDA(dense_kv_indices);
    CHECK_CUDA(sparse_kv_indices);

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        launch_ori_kernel<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(), sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(), sparse_kv_indices.data_ptr<int>(),
            topk_val, reserved_bos, reserved_eos,
            radix_bits, nblks, nthreads, stream);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        launch_ori_kernel<float>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(), sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(), sparse_kv_indices.data_ptr<int>(),
            topk_val, reserved_bos, reserved_eos,
            radix_bits, nblks, nthreads, stream);
    } else {
        TORCH_CHECK(false, "topk_output_sglang_ori: unsupported dtype ", x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_output_sglang_ori kernel failed: ", ::cudaGetErrorString(result));
}
