/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
// On ROCm, the per-workgroup LDS budget depends on the target arch, so we inject a
// per-arch value from `setup_rocm.py` via `-DSGL_TOPK_DYNAMIC_SMEM_BYTES=...`.
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;  // bytes
#endif
#else
// Reduced from 128KB to 32KB to improve occupancy.
// Each radix pass needs at most ~TopK candidates in the threshold bin,
// so 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);  // 32KB (bytes)
#endif

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Include mapping strategies (must come after convert_to_uint8 definition)
#include "topk_mapping.cuh"

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
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
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
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
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, row_start, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, row_start, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

// ======================================================================
// Vortex integration: BOS/EOS-aware segmented TopK with index remapping
// ======================================================================

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);

template <>
__device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }

template <>
__device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

constexpr int VORTEX_MAX_TOPK = 2048;

// Per-segment diagnostic counters written by WriteCounters mode
constexpr int COUNTER_THRESHOLD_BIN = 0;   // Stage 1 coarse threshold bin id
constexpr int COUNTER_NUM_ABOVE     = 1;   // elements routed above threshold in Stage 1
constexpr int COUNTER_NUM_EQUAL     = 2;   // elements in threshold bin (Stage 2 input)
constexpr int COUNTER_REMAINING_K   = 3;   // topk slots remaining after Stage 1 routing
constexpr int COUNTER_REFINE_ROUNDS = 4;   // Stage 2 rounds used (0 = resolved in Stage 1)
constexpr int COUNTER_STAGE2_INPUT  = 5;   // candidates entering first Stage 2 refine round
constexpr int NUM_TOPK_COUNTERS     = 6;

// Templated version of fast_topk_cuda_tl:
//   - ScoreT: float or __nv_bfloat16
//   - StopAfterStage1: return after Stage 1 route/filter (for profiling)
//   - WriteCounters: write diagnostic counters to global memory
//   - target_k: runtime parameter (replaces compile-time TopK)
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

// ======================================================================
// Profiling Stage1 kernel: runs pre-pass + hist + cumsum + route/filter,
// stops before Stage 2 refinement (for sub-phase timing)
// ======================================================================
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKStage1_Kernel(
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
    fast_topk_vortex<ScoreT, /*StopAfterStage1=*/true>(
        score_blk, s_indices, 0, nblk, topk_val, mapping);
    __syncthreads();

    // Remap position indices -> page indices via dense_kv_indices
    const int tx = threadIdx.x;
    for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

// ======================================================================
// Profiling counters kernel: runs full pipeline + writes diagnostic
// counters to a separate global-memory tensor
// ======================================================================
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKCounters_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    int*          __restrict__ counters,     // [eff_batch_size, NUM_TOPK_COUNTERS]
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
    fast_topk_vortex<ScoreT, /*StopAfterStage1=*/false, /*WriteCounters=*/true>(
        score_blk, s_indices, 0, nblk, topk_val, mapping,
        counters + bx * NUM_TOPK_COUNTERS);
    __syncthreads();

    // Remap position indices -> page indices via dense_kv_indices
    const int tx = threadIdx.x;
    for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

// ======================================================================
// Profiling histogram kernel: runs only Stage 1 and returns per-segment
// 256-bin histograms for distribution analysis
// ======================================================================
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKHistogram_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    int*          __restrict__ histograms,  // [eff_batch_size, 256]
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const TopKMappingParams mapping)
{
    constexpr auto RADIX = 256;
    constexpr auto BLOCK_SIZE = kThreadsPerBlock;
    __shared__ int s_histogram[RADIX];
    __shared__ uint8_t s_mapping_lut[256];
    __shared__ float s_mapping_quantiles[256];
    __shared__ float s_range_min, s_range_inv_range;

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int nblk  = end - start;

    const ScoreT* __restrict__ score_blk = score + start;

    // Load mapping tables into shared memory if needed
    if (mapping.mode == MAPPING_LUT_CDF && mapping.lut != nullptr) {
        if (tx < 256) s_mapping_lut[tx] = mapping.lut[tx];
        __syncthreads();
    }
    if (mapping.mode == MAPPING_QUANTILE && mapping.quantiles != nullptr) {
        if (tx < 256) s_mapping_quantiles[tx] = mapping.quantiles[tx];
        __syncthreads();
    }

    // Pre-pass: compute per-block min/max for transform modes (supports sampled stride)
    if (needs_auto_range(mapping.mode) && !mapping.noscale) {
        const int stride = (mapping.sample_stride > 1) ? mapping.sample_stride : 1;
        float local_min = __FLT_MAX__, local_max = -__FLT_MAX__;
        for (int idx = tx * stride; idx < nblk; idx += BLOCK_SIZE * stride) {
            float val = apply_transform(vortex_to_float(score_blk[idx]), mapping);
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_xor_sync(0xFFFFFFFF, local_min, offset));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
        }
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
        // Pivot pre-pass: compute mean for MAPPING_SUBTRACT
        float local_sum = 0.0f;
        for (int idx = tx; idx < nblk; idx += BLOCK_SIZE) {
            local_sum += vortex_to_float(score_blk[idx]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
        }
        __shared__ float s_warp_sums_h[32];
        int warp_id = tx >> 5, lane_id = tx & 31;
        if (lane_id == 0) s_warp_sums_h[warp_id] = local_sum;
        __syncthreads();
        if (tx < (BLOCK_SIZE >> 5)) {
            local_sum = s_warp_sums_h[tx];
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
            }
            if (tx == 0) {
                s_range_min = local_sum / float(nblk);
                s_range_inv_range = 0.0f;
            }
        }
        __syncthreads();
    } else {
        if (tx == 0) { s_range_min = 0.0f; s_range_inv_range = 0.0f; }
        __syncthreads();
    }

    // Initialize shared histogram
    if (tx < RADIX) s_histogram[tx] = 0;
    __syncthreads();

    // Build histogram over the segment with mapping
    for (int idx = tx; idx < nblk; idx += BLOCK_SIZE) {
        const auto bin = mapped_convert_to_uint8(
            vortex_to_float(score_blk[idx]),
            mapping, s_mapping_lut, s_mapping_quantiles,
            s_range_min, s_range_inv_range);
        ::atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    // Write to global memory
    int* __restrict__ out = histograms + bx * RADIX;
    if (tx < RADIX) {
        out[tx] = s_histogram[tx];
    }
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, row_starts_opt, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmem>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

// ======================================================================
// Vortex host entry point — same interface as topk_output in topk.cu
// ======================================================================
void topk_output_sglang(
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
    const int64_t     mapping_mode,
    const double      mapping_power,
    std::optional<at::Tensor> mapping_lut,
    std::optional<at::Tensor> mapping_quantiles,
    const bool        mapping_noscale)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_output: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");

    // Build mapping params from optional tensors
    TopKMappingParams mapping{};
    mapping.mode = static_cast<int>(mapping_mode);
    mapping.power_exp = static_cast<float>(mapping_power);
    mapping.lut = nullptr;
    mapping.quantiles = nullptr;
    mapping.noscale = mapping_noscale;
    mapping.sample_stride = 1;

    if (mapping_lut.has_value()) {
        const auto& lut = mapping_lut.value();
        CHECK_CUDA(lut);
        TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 256 && lut.scalar_type() == at::ScalarType::Byte,
                     "mapping_lut must be a 1D uint8 tensor of size 256");
        mapping.lut = lut.data_ptr<uint8_t>();
    }
    if (mapping_quantiles.has_value()) {
        const auto& q = mapping_quantiles.value();
        CHECK_CUDA(q);
        TORCH_CHECK(q.dim() == 1 && q.size(0) == 256 && q.scalar_type() == at::ScalarType::Float,
                     "mapping_quantiles must be a 1D float32 tensor of size 256");
        mapping.quantiles = q.data_ptr<float>();
    }

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        setup_kernel_smem_once<TopKOutput_Kernel<__nv_bfloat16>, kSmem>();
        TopKOutput_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos,
            mapping);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        setup_kernel_smem_once<TopKOutput_Kernel<float>, kSmem>();
        TopKOutput_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos,
            mapping);
    } else {
        TORCH_CHECK(false,
                    "topk_output: unsupported dtype ",
                    x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_output kernel failed: ", ::cudaGetErrorString(result));
}

// ======================================================================
// Profiling: collect per-segment 256-bin histograms of Stage 1 bins
// ======================================================================
void topk_profile_histogram(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    at::Tensor&       histograms,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     mapping_mode,
    const double      mapping_power,
    std::optional<at::Tensor> mapping_lut,
    std::optional<at::Tensor> mapping_quantiles,
    const bool        mapping_noscale)
{
    CHECK_CUDA(x);
    CHECK_CUDA(dense_kv_indptr);
    CHECK_CUDA(histograms);
    TORCH_CHECK(histograms.dim() == 2 && histograms.size(0) == eff_batch_size
                && histograms.size(1) == 256,
                "histograms must be [eff_batch_size, 256]");
    TORCH_CHECK(histograms.scalar_type() == at::ScalarType::Int,
                "histograms must be int32");

    // Build mapping params
    TopKMappingParams mapping{};
    mapping.mode = static_cast<int>(mapping_mode);
    mapping.power_exp = static_cast<float>(mapping_power);
    mapping.lut = nullptr;
    mapping.quantiles = nullptr;
    mapping.noscale = mapping_noscale;
    mapping.sample_stride = 1;

    if (mapping_lut.has_value()) {
        const auto& lut = mapping_lut.value();
        CHECK_CUDA(lut);
        TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 256 && lut.scalar_type() == at::ScalarType::Byte,
                     "mapping_lut must be a 1D uint8 tensor of size 256");
        mapping.lut = lut.data_ptr<uint8_t>();
    }
    if (mapping_quantiles.has_value()) {
        const auto& q = mapping_quantiles.value();
        CHECK_CUDA(q);
        TORCH_CHECK(q.dim() == 1 && q.size(0) == 256 && q.scalar_type() == at::ScalarType::Float,
                     "mapping_quantiles must be a 1D float32 tensor of size 256");
        mapping.quantiles = q.data_ptr<float>();
    }

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        TopKHistogram_Kernel<__nv_bfloat16><<<nblks, nthreads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            histograms.data_ptr<int>(),
            reserved_bos,
            reserved_eos,
            mapping);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        TopKHistogram_Kernel<float><<<nblks, nthreads, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            histograms.data_ptr<int>(),
            reserved_bos,
            reserved_eos,
            mapping);
    } else {
        TORCH_CHECK(false,
                    "topk_profile_histogram: unsupported dtype ",
                    x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_profile_histogram kernel failed: ", ::cudaGetErrorString(result));
}

// Helper: build TopKMappingParams from host arguments
static TopKMappingParams build_mapping_params(
    int64_t mapping_mode, double mapping_power,
    std::optional<at::Tensor>& mapping_lut,
    std::optional<at::Tensor>& mapping_quantiles,
    bool mapping_noscale = false,
    int sample_stride = 1)
{
    TopKMappingParams mapping{};
    mapping.mode = static_cast<int>(mapping_mode);
    mapping.power_exp = static_cast<float>(mapping_power);
    mapping.lut = nullptr;
    mapping.quantiles = nullptr;
    mapping.noscale = mapping_noscale;
    mapping.sample_stride = sample_stride;

    if (mapping_lut.has_value()) {
        const auto& lut = mapping_lut.value();
        CHECK_CUDA(lut);
        TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 256 && lut.scalar_type() == at::ScalarType::Byte,
                     "mapping_lut must be a 1D uint8 tensor of size 256");
        mapping.lut = lut.data_ptr<uint8_t>();
    }
    if (mapping_quantiles.has_value()) {
        const auto& q = mapping_quantiles.value();
        CHECK_CUDA(q);
        TORCH_CHECK(q.dim() == 1 && q.size(0) == 256 && q.scalar_type() == at::ScalarType::Float,
                     "mapping_quantiles must be a 1D float32 tensor of size 256");
        mapping.quantiles = q.data_ptr<float>();
    }
    return mapping;
}

// ======================================================================
// Profiling: Stage 1 only (pre-pass + hist + cumsum + route/filter)
// ======================================================================
void topk_profile_stage1(
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
    const int64_t     mapping_mode,
    const double      mapping_power,
    std::optional<at::Tensor> mapping_lut,
    std::optional<at::Tensor> mapping_quantiles,
    const bool        mapping_noscale)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_profile_stage1: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");

    auto mapping = build_mapping_params(mapping_mode, mapping_power, mapping_lut, mapping_quantiles, mapping_noscale);

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        setup_kernel_smem_once<TopKStage1_Kernel<__nv_bfloat16>, kSmem>();
        TopKStage1_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos,
            mapping);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        setup_kernel_smem_once<TopKStage1_Kernel<float>, kSmem>();
        TopKStage1_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos,
            mapping);
    } else {
        TORCH_CHECK(false,
                    "topk_profile_stage1: unsupported dtype ",
                    x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_profile_stage1 kernel failed: ", ::cudaGetErrorString(result));
}

// ======================================================================
// Profiling: full pipeline + diagnostic counters
// ======================================================================
void topk_profile_counters(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    at::Tensor&       counters,
    const int64_t     eff_batch_size,
    const int64_t     topk_val,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const int64_t     mapping_mode,
    const double      mapping_power,
    std::optional<at::Tensor> mapping_lut,
    std::optional<at::Tensor> mapping_quantiles,
    const bool        mapping_noscale)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_profile_counters: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");
    CHECK_CUDA(counters);
    TORCH_CHECK(counters.dim() == 2 && counters.size(0) == eff_batch_size
                && counters.size(1) == NUM_TOPK_COUNTERS,
                "counters must be [eff_batch_size, ", NUM_TOPK_COUNTERS, "]");
    TORCH_CHECK(counters.scalar_type() == at::ScalarType::Int,
                "counters must be int32");

    auto mapping = build_mapping_params(mapping_mode, mapping_power, mapping_lut, mapping_quantiles, mapping_noscale);

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        setup_kernel_smem_once<TopKCounters_Kernel<__nv_bfloat16>, kSmem>();
        TopKCounters_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            counters.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos,
            mapping);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        setup_kernel_smem_once<TopKCounters_Kernel<float>, kSmem>();
        TopKCounters_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            counters.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos,
            mapping);
    } else {
        TORCH_CHECK(false,
                    "topk_profile_counters: unsupported dtype ",
                    x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_profile_counters kernel failed: ", ::cudaGetErrorString(result));
}

