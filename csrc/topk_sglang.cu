/**
 * Vortex TopK kernels.
 *
 * Three production kernels:
 *   - fast_topk_clean<ScoreT>         : unmapped baseline (two-stage radix).
 *   - fast_topk_clean_fused<ScoreT>   : remap + topk fused (apply_transform
 *                                      applied inline in Stage-1 bucketing).
 *   - TopKRemapOnly_Kernel<ScoreT>    : standalone element-wise remap pass
 *                                      used by the split-phase benchmark.
 *
 * Profiling kernels (counter collection, histogram collection) live in
 * topk_sglang_profile.cu and MUST NOT be used for latency measurements —
 * they intentionally write extra diagnostic state to global memory.
 *
 * Archived / historical kernels: csrc/archived/ (fast_topk_vortex with
 * pre-pass modes, TopKOutput_Ori_Kernel with flexible radix_bits, the
 * original SGLang reference kernel).
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

// ---- Vortex additions ----

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);
template <>
__device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

constexpr int VORTEX_MAX_TOPK = 2048;

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
// Templated clean baseline: identical algorithm to fast_topk_cuda_tl but
// parameterised on ScoreT (float or __nv_bfloat16) for the GQA / paged
// call paths that operate on bf16 attention scores. No mapping, no
// pre-pass — pure two-stage radix topk on fp16 bit-pattern bins.
// ======================================================================
template <typename ScoreT>
__device__ void fast_topk_clean(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k)
{
  int topk = target_k;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8-bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(vortex_to_float(input[idx + row_start]));
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
      const auto bin = static_cast<int>(convert_to_uint8(vortex_to_float(input[idx + row_start])));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = vortex_to_float(input[idx + row_start]);
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
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

  // stage 2: refine with 8-bit radix passes on raw fp32 bits
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

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
      if (tx < RADIX + 1) s_histogram[tx] = 0;
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

// ======================================================================
// Templated fused kernel: apply_transform(score) -> convert_to_uint8
// is fused into Stage 1. Stage 2 still uses raw bits for tie-breaking
// (on the *remapped* value, not the original score) — this is a
// benchmarking kernel, the remapped Stage-2 ordering is acceptable.
// No pre-pass, no LUT, no shared-memory mapping state.
// ======================================================================
template <typename ScoreT>
__device__ void fast_topk_clean_fused(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k,
    const TopKMappingParams mapping)
{
  int topk = target_k;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int f_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int f_counter;
  alignas(128) __shared__ int f_threshold_bin_id;
  alignas(128) __shared__ int f_num_input[2];

  // Shared-memory tables for MAPPING_LUT_CDF / MAPPING_QUANTILE. Loaded
  // once at kernel entry and read per element in Stage 1. Other modes
  // leave them untouched.
  __shared__ uint8_t s_mapping_lut[256];
  __shared__ float   s_mapping_quantiles[256];

  auto& f_histogram = f_histogram_buf[0];
  extern __shared__ int f_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  if (mapping.mode == MAPPING_LUT_CDF && mapping.lut != nullptr) {
    if (tx < 256) s_mapping_lut[tx] = mapping.lut[tx];
    __syncthreads();
  }
  if (mapping.mode == MAPPING_QUANTILE && mapping.quantiles != nullptr) {
    if (tx < 256) s_mapping_quantiles[tx] = mapping.quantiles[tx];
    __syncthreads();
  }

  if (tx < RADIX + 1) f_histogram[tx] = 0;
  __syncthreads();

  // Stage 1: LUT/QUANTILE do a shared-memory lookup, everything else
  // applies the element-wise transform then buckets via convert_to_uint8.
  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const float raw = vortex_to_float(input[idx + row_start]);
    const auto bin = compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles);
    ::atomicAdd(&f_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = f_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += f_histogram_buf[k][tx + j];
        }
        f_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && f_histogram[tx] > topk && f_histogram[tx + 1] <= topk) {
    f_threshold_bin_id = tx;
    f_num_input[0] = 0;
    f_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = f_threshold_bin_id;
  topk -= f_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw = vortex_to_float(input[idx + row_start]);
      const auto bin = static_cast<int>(
          compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&f_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) f_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw = vortex_to_float(input[idx + row_start]);
      const float remapped = apply_transform(raw, mapping);
      const auto bin = static_cast<int>(
          compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&f_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&f_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          f_input_idx[0][pos] = idx;
          const auto b32 = convert_to_uint32(remapped);
          const auto sub_bin = (b32 >> 24) & 0xFF;
          ::atomicAdd(&f_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine on raw bits of the remapped value
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int f_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = f_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && f_histogram[tx] > topk && f_histogram[tx + 1] <= topk) {
      f_threshold_bin_id = tx;
      f_num_input[r_idx ^ 1] = 0;
      f_last_remain = topk - f_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = f_threshold_bin_id;
    topk -= f_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = f_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform(raw, mapping);
        const auto bin = (convert_to_uint32(remapped) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&f_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) f_histogram[tx] = 0;
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = f_input_idx[r_idx][i];
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform(raw, mapping);
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(remapped) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&f_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&f_last_remain, -1);
            if (pos > 0) {
              index[target_k - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&f_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              f_input_idx[r_idx ^ 1][pos] = idx;
              const auto b32 = convert_to_uint32(remapped);
              const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
              ::atomicAdd(&f_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

// Wrapper kernels: one CUDA block per (batch*head) segment.

template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutput_Clean_Kernel(
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
  fast_topk_clean<ScoreT>(score_blk, s_indices, 0, nblk, topk_val);
  __syncthreads();

  const int tx = threadIdx.x;
  for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
    out_blk[i] = idx_blk[s_indices[i]];
  }
}

template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutput_Fused_Kernel(
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
  fast_topk_clean_fused<ScoreT>(score_blk, s_indices, 0, nblk, topk_val, mapping);
  __syncthreads();

  const int tx = threadIdx.x;
  for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
    out_blk[i] = idx_blk[s_indices[i]];
  }
}

// Remap-only kernel: applies the element-wise transform to each score
// in the [dense_kv_indptr[b] + reserved_bos, dense_kv_indptr[b+1] - reserved_eos)
// range and writes the result into a float32 output tensor. Used by
// the split-phase benchmark (remap → unmapped topk).
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKRemapOnly_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    float*        __restrict__ remapped,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const TopKMappingParams mapping)
{
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;

  const int start = dense_kv_indptr[bx] + page_reserved_bos;
  const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
  const int nblk  = end - start;
  if (nblk <= 0) return;

  const ScoreT* __restrict__ score_blk = score + start;
  float*        __restrict__ remap_blk = remapped + start;

  for (int i = tx; i < nblk; i += kThreadsPerBlock) {
    remap_blk[i] = apply_transform(vortex_to_float(score_blk[i]), mapping);
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
// Vortex host entry point — unmapped baseline topk (no remap).
// This is the "original topk kernel" used as the benchmarking baseline.
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
    const int64_t     max_num_pages)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_output: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");

    CHECK_CUDA(x);
    CHECK_CUDA(dense_kv_indptr);
    CHECK_CUDA(sparse_kv_indptr);
    CHECK_CUDA(dense_kv_indices);
    CHECK_CUDA(sparse_kv_indices);

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        setup_kernel_smem_once<TopKOutput_Clean_Kernel<__nv_bfloat16>, kSmem>();
        TopKOutput_Clean_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val, reserved_bos, reserved_eos);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        setup_kernel_smem_once<TopKOutput_Clean_Kernel<float>, kSmem>();
        TopKOutput_Clean_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val, reserved_bos, reserved_eos);
    } else {
        TORCH_CHECK(false, "topk_output: unsupported dtype ", x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_output kernel failed: ", ::cudaGetErrorString(result));
}

// ======================================================================
// Fused remap + topk host entry. Applies apply_transform(score, mapping)
// inline inside the Stage-1 histogram build — single kernel launch,
// single pass over the score tensor.
// ======================================================================
void topk_output_sglang_fused(
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
    std::optional<at::Tensor> mapping_quantiles)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_output_sglang_fused: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");

    CHECK_CUDA(x);
    CHECK_CUDA(dense_kv_indptr);
    CHECK_CUDA(sparse_kv_indptr);
    CHECK_CUDA(dense_kv_indices);
    CHECK_CUDA(sparse_kv_indices);

    TopKMappingParams mapping{};
    mapping.mode = static_cast<int>(mapping_mode);
    mapping.power_exp = static_cast<float>(mapping_power);
    mapping.lut = nullptr;
    mapping.quantiles = nullptr;
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
        setup_kernel_smem_once<TopKOutput_Fused_Kernel<__nv_bfloat16>, kSmem>();
        TopKOutput_Fused_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val, reserved_bos, reserved_eos, mapping);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        setup_kernel_smem_once<TopKOutput_Fused_Kernel<float>, kSmem>();
        TopKOutput_Fused_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val, reserved_bos, reserved_eos, mapping);
    } else {
        TORCH_CHECK(false, "topk_output_sglang_fused: unsupported dtype ", x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_output_sglang_fused kernel failed: ", ::cudaGetErrorString(result));
}

// ======================================================================
// Standalone remap kernel. Writes apply_transform(score) into a
// float32 output buffer without running topk. Used by the split-phase
// benchmark (remap → unmapped topk) to measure each phase independently.
// ======================================================================
void topk_remap_only(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    at::Tensor&       remapped,          // float32, same numel as x
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     mapping_mode,
    const double      mapping_power)
{
    CHECK_CUDA(x);
    CHECK_CUDA(dense_kv_indptr);
    CHECK_CUDA(remapped);
    TORCH_CHECK(remapped.scalar_type() == at::ScalarType::Float,
                "remapped output must be float32");

    TopKMappingParams mapping{};
    mapping.mode = static_cast<int>(mapping_mode);
    mapping.power_exp = static_cast<float>(mapping_power);
    mapping.lut = nullptr;
    mapping.quantiles = nullptr;

    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        TopKRemapOnly_Kernel<__nv_bfloat16><<<nblks, nthreads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            remapped.data_ptr<float>(),
            reserved_bos, reserved_eos, mapping);
    } else if (x.scalar_type() == at::ScalarType::Float) {
        TopKRemapOnly_Kernel<float><<<nblks, nthreads, 0, stream>>>(
            x.data_ptr<float>(),
            dense_kv_indptr.data_ptr<int>(),
            remapped.data_ptr<float>(),
            reserved_bos, reserved_eos, mapping);
    } else {
        TORCH_CHECK(false, "topk_remap_only: unsupported dtype ", x.scalar_type());
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_remap_only kernel failed: ", ::cudaGetErrorString(result));
}
