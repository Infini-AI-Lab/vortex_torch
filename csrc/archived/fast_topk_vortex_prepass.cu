// Archived: not compiled. See csrc/archived/README.md
//
// fast_topk_vortex — the heavy fused remap+topk kernel with auto-range,
// pivot, tail-window, topk-window pre-passes and LUT/quantile support.
// Extracted from csrc/topk_sglang.cu as part of the remap-benchmark refactor.
// Replaced by a lean fast_topk_clean_fused that applies a simple element-wise
// transform (from topk_mapping.cuh apply_transform) in Stage-1 bucketing —
// no pre-pass, no LUT, no auto-range.
//
// References types/constants from its former translation unit (TopKMappingParams,
// needs_*, mapped_convert_to_uint8, kSmem, kThreadsPerBlock, COUNTER_*). This
// file will not compile standalone; kept for history only.

// ======================================================================
// Templated version of fast_topk_cuda_tl with mapping support:
//   - ScoreT: float or __nv_bfloat16
//   - StopAfterStage1: return after Stage 1 route/filter (for profiling)
//   - WriteCounters: write diagnostic counters to global memory

//   - mapping: configurable value-remapping for Stage 1 bin assignment
template <typename ScoreT, bool StopAfterStage1 = false, bool WriteCounters = false>
__device__ void fast_topk_vortex(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k,
    const TopKMappingParams& mapping,
    int*          counters = nullptr)
{
    int topk = target_k;
    constexpr auto BLOCK_SIZE = 1024;
    constexpr auto RADIX = 256;
    constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

    alignas(128) __shared__ int vh_histogram_buf[2][RADIX + 128];
    alignas(128) __shared__ int vh_counter;
    alignas(128) __shared__ int vh_threshold_bin_id;
    alignas(128) __shared__ int vh_num_input[2];

    // Shared memory for mapping LUT / quantiles (loaded once per block)
    __shared__ uint8_t s_mapping_lut[256];
    __shared__ float s_mapping_quantiles[256];

    // Auto-range for transform modes (3/4/6/7)
    __shared__ float s_range_min, s_range_inv_range;

    auto& vh_histogram = vh_histogram_buf[0];
    extern __shared__ int vh_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;

    // Load mapping tables into shared memory if needed
    if (mapping.mode == MAPPING_LUT_CDF && mapping.lut != nullptr) {
        if (tx < 256) s_mapping_lut[tx] = mapping.lut[tx];
        __syncthreads();
    }
    if (mapping.mode == MAPPING_QUANTILE && mapping.quantiles != nullptr) {
        if (tx < 256) s_mapping_quantiles[tx] = mapping.quantiles[tx];
        __syncthreads();
    }

    // Pre-pass: compute per-block min/max of transformed values for linear bucketing.
    // sample_stride > 1 reduces pre-pass cost by scanning every Nth element;
    // the approximated range may miss extreme outliers but Stage 2 uses raw
    // float bits for exact ordering, so correctness is preserved.
    if (needs_auto_range(mapping.mode) && !mapping.noscale) {
        const int stride = (mapping.sample_stride > 1) ? mapping.sample_stride : 1;
        float local_min = __FLT_MAX__, local_max = -__FLT_MAX__;
        for (int idx = tx * stride; idx < length; idx += BLOCK_SIZE * stride) {
            float val = apply_transform(vortex_to_float(input[idx + row_start]), mapping);
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_xor_sync(0xFFFFFFFF, local_min, offset));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
        // Cross-warp reduction via shared memory
        __shared__ float s_warp_mins[32], s_warp_maxs[32];
        int warp_id = tx >> 5, lane_id = tx & 31;
        if (lane_id == 0) { s_warp_mins[warp_id] = local_min; s_warp_maxs[warp_id] = local_max; }
        __syncthreads();
        if (tx < (BLOCK_SIZE >> 5)) {
            local_min = s_warp_mins[tx]; local_max = s_warp_maxs[tx];
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_min = fminf(local_min, __shfl_xor_sync(0xFFFFFFFF, local_min, offset));
                local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
            }
            if (tx == 0) {
                s_range_min = local_min;
                float range = local_max - local_min;
                s_range_inv_range = (range > 0.0f) ? 255.0f / range : 0.0f;
            }
        }
        __syncthreads();
    } else if (needs_pivot(mapping.mode)) {
        // Pivot pre-pass: compute mean of all elements, store in s_range_min.
        // MAPPING_SUBTRACT uses convert_to_uint8(x - range_min), so centering
        // around the mean helps distribute values more evenly across bins.
        float local_sum = 0.0f;
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            local_sum += vortex_to_float(input[idx + row_start]);
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
        __shared__ float s_warp_sums[32];
        int warp_id = tx >> 5, lane_id = tx & 31;
        if (lane_id == 0) s_warp_sums[warp_id] = local_sum;
        __syncthreads();
        if (tx < (BLOCK_SIZE >> 5)) {
            local_sum = s_warp_sums[tx];
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
            }
            if (tx == 0) {
                s_range_min = local_sum / float(length);  // mean as pivot
                s_range_inv_range = 0.0f;
            }
        }
        __syncthreads();
    } else if (needs_tail_window(mapping.mode)) {
        // Adaptive tail-window pre-pass: estimate tau_low = Q(1 - rho*k/n)
        // and local_max via a sampled quantile estimator.  All 256 coarse bins
        // are then allocated to [tau_low, local_max]; scores below tau_low
        // collapse into bin 0 via linear_map_to_uint8 clamping.
        constexpr int MAX_SAMPLES = 1024;
        __shared__ float s_samples[MAX_SAMPLES];
        __shared__ int   s_sample_count;

        if (tx == 0) s_sample_count = 0;
        __syncthreads();

        // Compute sampling stride so we collect ~MAX_SAMPLES from the segment
        const int desired_stride = (length + MAX_SAMPLES - 1) / MAX_SAMPLES;
        const int sample_stride = max(desired_stride, 1);

        // Each thread samples elements and finds local_max simultaneously
        float local_max = -__FLT_MAX__;
        for (int idx = tx * sample_stride; idx < length; idx += BLOCK_SIZE * sample_stride) {
            float val = vortex_to_float(input[idx + row_start]);
            local_max = fmaxf(local_max, val);
            int slot = ::atomicAdd(&s_sample_count, 1);
            if (slot < MAX_SAMPLES) {
                s_samples[slot] = val;
            }
        }

        // Reduce local_max across block
        for (int offset = 16; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        __shared__ float s_warp_maxs_tw[32];
        {
            int warp_id = tx >> 5, lane_id = tx & 31;
            if (lane_id == 0) s_warp_maxs_tw[warp_id] = local_max;
        }
        __syncthreads();
        if (tx < (BLOCK_SIZE >> 5)) {
            local_max = s_warp_maxs_tw[tx];
            for (int offset = 16; offset > 0; offset >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
            if (tx == 0) s_warp_maxs_tw[0] = local_max;
        }
        __syncthreads();
        local_max = s_warp_maxs_tw[0];

        int nsamp = min(s_sample_count, MAX_SAMPLES);

        // Simple odd-even transposition sort on the sample buffer.
        // nsamp <= 1024, and we have 1024 threads, so each thread
        // handles one element.  O(nsamp) parallel rounds suffice.
        __syncthreads();
        if (nsamp >= 2) {
            for (int pass = 0; pass < nsamp; ++pass) {
                // Even phase: compare (0,1), (2,3), ...
                if (tx * 2 + 1 < nsamp) {
                    int i = tx * 2;
                    if (s_samples[i] > s_samples[i + 1]) {
                        float tmp = s_samples[i];
                        s_samples[i] = s_samples[i + 1];
                        s_samples[i + 1] = tmp;
                    }
                }
                __syncthreads();
                // Odd phase: compare (1,2), (3,4), ...
                if (tx * 2 + 2 < nsamp) {
                    int i = tx * 2 + 1;
                    if (s_samples[i] > s_samples[i + 1]) {
                        float tmp = s_samples[i];
                        s_samples[i] = s_samples[i + 1];
                        s_samples[i + 1] = tmp;
                    }
                }
                __syncthreads();
            }
        }

        // Estimate tau_low = Q(1 - rho * k / n)
        if (tx == 0) {
            float rho = mapping.power_exp;  // reused as tail expansion factor
            if (rho <= 0.0f) rho = 4.0f;
            int k = (mapping.target_k > 0) ? mapping.target_k : target_k;
            float frac = 1.0f - rho * float(k) / float(length);
            frac = fmaxf(frac, 0.0f);  // clamp: never go below rank 0

            float tau_low;
            if (nsamp < 4 || frac <= 0.0f) {
                // Too few samples or the tail covers everything: full range
                tau_low = -__FLT_MAX__;
            } else {
                float fidx = frac * float(nsamp - 1);
                int lo = __float2int_rd(fidx);
                lo = min(max(lo, 0), nsamp - 2);
                float t = fidx - float(lo);
                tau_low = s_samples[lo] * (1.0f - t) + s_samples[lo + 1] * t;
            }

            // Fallback: if tau_low >= local_max, use full-range linear mapping
            if (tau_low >= local_max) {
                // Find the actual minimum from sorted samples
                tau_low = (nsamp > 0) ? s_samples[0] : local_max;
            }

            float range = local_max - tau_low;
            s_range_min = tau_low;
            s_range_inv_range = (range > 1e-10f) ? 255.0f / range : 0.0f;
        }
        __syncthreads();
    } else if (needs_topk_window(mapping.mode)) {
        // Topk-window pre-pass with streaming variance heuristic.
        // tau_low = max - rho * sigma * sqrt(2 * log(n/k))
        float local_max = -__FLT_MAX__;
        float local_sum = 0.0f, local_sum_sq = 0.0f;
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            float val = vortex_to_float(input[idx + row_start]);
            local_max = fmaxf(local_max, val);
            local_sum += val;
            local_sum_sq += val * val;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
            local_sum_sq += __shfl_xor_sync(0xFFFFFFFF, local_sum_sq, offset);
        }
        __shared__ float s_warp_maxs_tw2[32], s_warp_sums_tw2[32], s_warp_sq_tw2[32];
        {
            int warp_id = tx >> 5, lane_id = tx & 31;
            if (lane_id == 0) {
                s_warp_maxs_tw2[warp_id] = local_max;
                s_warp_sums_tw2[warp_id] = local_sum;
                s_warp_sq_tw2[warp_id] = local_sum_sq;
            }
        }
        __syncthreads();
        if (tx < (BLOCK_SIZE >> 5)) {
            local_max = s_warp_maxs_tw2[tx];
            local_sum = s_warp_sums_tw2[tx];
            local_sum_sq = s_warp_sq_tw2[tx];
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
                local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
                local_sum_sq += __shfl_xor_sync(0xFFFFFFFF, local_sum_sq, offset);
            }
            if (tx == 0) {
                float rho = mapping.power_exp;
                if (rho <= 0.0f) rho = 4.0f;
                int k = (mapping.target_k > 0) ? mapping.target_k : target_k;
                float n = float(length);
                float mean = local_sum / n;
                float var = local_sum_sq / n - mean * mean;
                float sigma = (var > 0.0f) ? sqrtf(var) : 0.0f;
                float ratio = n / fmaxf(float(k), 1.0f);
                float z = sqrtf(2.0f * __logf(fmaxf(ratio, 1.0f)));
                float tau_low = local_max - rho * sigma * z;
                if (tau_low >= local_max) tau_low = local_max - 1.0f;
                float range = local_max - tau_low;
                s_range_min = tau_low;
                s_range_inv_range = (range > 1e-10f) ? 255.0f / range : 0.0f;
            }
        }
        __syncthreads();
    } else {
        if (tx == 0) { s_range_min = 0.0f; s_range_inv_range = 0.0f; }
        __syncthreads();
    }

    // Stage 1: 8-bit coarse histogram (with optional mapping)
    // Bin cache: store computed bins in vh_input_idx[1] (reinterpreted as uint8_t*)
    // to avoid recomputing mapped_convert_to_uint8 in the route/filter pass.
    // vh_input_idx[1] is unused until Stage 2 double-buffering starts after route.
    constexpr int BIN_CACHE_CAPACITY = SMEM_INPUT_SIZE * static_cast<int>(sizeof(int));  // uint8 entries
    uint8_t* bin_cache = reinterpret_cast<uint8_t*>(vh_input_idx[1]);
    const bool use_bin_cache = (length <= BIN_CACHE_CAPACITY);

    if (tx < RADIX + 1) vh_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = mapped_convert_to_uint8(
            vortex_to_float(input[idx + row_start]),
            mapping, s_mapping_lut, s_mapping_quantiles,
            s_range_min, s_range_inv_range);
        ::atomicAdd(&vh_histogram[bin], 1);
        if (use_bin_cache) {
            bin_cache[idx] = bin;
        }
    }
    __syncthreads();

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX);
            if (C10_LIKELY(tx < RADIX)) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = vh_histogram_buf[k][tx];
                if (tx < RADIX - j) {
                    value += vh_histogram_buf[k][tx + j];
                }
                vh_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum();
    if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
        vh_threshold_bin_id = tx;
        vh_num_input[0] = 0;
        vh_counter = 0;
    }
    __syncthreads();

    const auto threshold_bin = vh_threshold_bin_id;
    topk -= vh_histogram[threshold_bin + 1];

    if (WriteCounters && tx == 0 && counters) {
        counters[COUNTER_THRESHOLD_BIN] = threshold_bin;
        counters[COUNTER_REMAINING_K] = topk;
    }

    if (topk == 0) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            int bin;
            if (use_bin_cache) {
                bin = static_cast<int>(bin_cache[idx]);
            } else {
                bin = static_cast<int>(
                    mapped_convert_to_uint8(
                        vortex_to_float(input[idx + row_start]),
                        mapping, s_mapping_lut, s_mapping_quantiles,
                        s_range_min, s_range_inv_range));
            }
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            }
        }
        __syncthreads();
        if (WriteCounters && tx == 0 && counters) {
            counters[COUNTER_NUM_ABOVE] = vh_counter;
            counters[COUNTER_NUM_EQUAL] = 0;
            counters[COUNTER_REFINE_ROUNDS] = 0;
            counters[COUNTER_STAGE2_INPUT] = 0;
        }
        return;
    } else {
        __syncthreads();
        if (tx < RADIX + 1) vh_histogram[tx] = 0;
        __syncthreads();

        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto raw_input = vortex_to_float(input[idx + row_start]);
            int bin;
            if (use_bin_cache) {
                bin = static_cast<int>(bin_cache[idx]);
            } else {
                bin = static_cast<int>(
                    mapped_convert_to_uint8(raw_input, mapping,
                                            s_mapping_lut, s_mapping_quantiles,
                                            s_range_min, s_range_inv_range));
            }
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&vh_counter, 1);
                index[pos] = idx;
            } else if (bin == threshold_bin) {
                const auto pos = ::atomicAdd(&vh_num_input[0], 1);
                if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                    vh_input_idx[0][pos] = idx;
                    const auto b32 = convert_to_uint32(raw_input);
                    const auto sub_bin = (b32 >> 24) & 0xFF;
                    ::atomicAdd(&vh_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
        if (WriteCounters && tx == 0 && counters) {
            counters[COUNTER_NUM_ABOVE] = vh_counter;
            counters[COUNTER_NUM_EQUAL] = vh_num_input[0];
            counters[COUNTER_STAGE2_INPUT] = vh_num_input[0];
        }
        if (StopAfterStage1) return;
    }

    // Stage 2: refine with 8-bit radix passes (unchanged — uses raw float bits)
    if constexpr (WriteCounters) {
        // Default: all 4 rounds used; overwritten at break if resolved early
        if (tx == 0 && counters) counters[COUNTER_REFINE_ROUNDS] = 4;
    }
#pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        __shared__ int vh_last_remain;
        const auto r_idx = round % 2;

        const auto _raw_num_input = vh_num_input[r_idx];
        const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE))
                                   ? _raw_num_input
                                   : int(SMEM_INPUT_SIZE);

        run_cumsum();
        if (tx < RADIX && vh_histogram[tx] > topk && vh_histogram[tx + 1] <= topk) {
            vh_threshold_bin_id = tx;
            vh_num_input[r_idx ^ 1] = 0;
            vh_last_remain = topk - vh_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = vh_threshold_bin_id;
        topk -= vh_histogram[threshold_bin + 1];

        if (topk == 0) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(
                    vortex_to_float(input[idx + row_start])) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                }
            }
            __syncthreads();
            if constexpr (WriteCounters) {
                if (tx == 0 && counters) {
                    counters[COUNTER_REFINE_ROUNDS] = round + 1;
                }
            }
            break;
        } else {
            __syncthreads();
            if (tx < RADIX + 1) vh_histogram[tx] = 0;
            __syncthreads();
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = vh_input_idx[r_idx][i];
                const auto raw_input = vortex_to_float(input[idx + row_start]);
                const auto offset = 24 - round * 8;
                const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = ::atomicAdd(&vh_counter, 1);
                    index[pos] = idx;
                } else if (bin == threshold_bin) {
                    if (round == 3) {
                        const auto pos = ::atomicAdd(&vh_last_remain, -1);
                        if (pos > 0) {
                            index[target_k - pos] = idx;
                        }
                    } else {
                        const auto pos = ::atomicAdd(&vh_num_input[r_idx ^ 1], 1);
                        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                            vh_input_idx[r_idx ^ 1][pos] = idx;
                            const auto b32 = convert_to_uint32(raw_input);
                            const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
                            ::atomicAdd(&vh_histogram[sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Wrapper kernel: one CUDA block per batch*head segment
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutput_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    const int     topk_val,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const TopKMappingParams mapping)
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
    fast_topk_vortex<ScoreT>(score_blk, s_indices, 0, nblk, topk_val, mapping);
    __syncthreads();

    // Remap position indices -> page indices via dense_kv_indices
    const int tx = threadIdx.x;
    for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}


