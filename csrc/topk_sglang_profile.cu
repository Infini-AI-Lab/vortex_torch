/**
 * TopK profiling kernels: histogram collection, stage-1-only timing,
 * and diagnostic counter collection.
 *
 * Separated from topk_sglang.cu to reduce template instantiation
 * pressure on CUDA shared memory resources.
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

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);
template <>
__device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}


constexpr int VORTEX_MAX_TOPK = 2048;

// Diagnostic counters written by the profiling kernel. These kernels are
// NOT used for latency measurements — they intentionally add global-memory
// writes that distort timings. Latency is measured against the clean
// production kernels in topk_sglang.cu.
constexpr int COUNTER_THRESHOLD_BIN = 0;
constexpr int COUNTER_NUM_ABOVE     = 1;
constexpr int COUNTER_NUM_EQUAL     = 2;
constexpr int COUNTER_REMAINING_K   = 3;
constexpr int COUNTER_REFINE_ROUNDS = 4;
constexpr int COUNTER_STAGE2_INPUT  = 5;
constexpr int NUM_TOPK_COUNTERS     = 6;

#include "topk_mapping.cuh"

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

// ======================================================================
// Profiling variant of fast_topk_clean_fused that writes diagnostic
// counters at the end of Stage 1 and at each Stage 2 early-exit.
// Shape / semantics identical to the production kernel, with one extra
// global-memory write pass at the end of each stage. Do not use for
// latency measurements.
// ======================================================================
template <typename ScoreT>
__device__ void fast_topk_profile(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int           row_start,
    int           length,
    int           target_k,
    const TopKMappingParams mapping,
    int*          __restrict__ counters)  // [NUM_TOPK_COUNTERS]
{
  int topk = target_k;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int p_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int p_counter;
  alignas(128) __shared__ int p_threshold_bin_id;
  alignas(128) __shared__ int p_num_input[2];

  __shared__ uint8_t s_mapping_lut[256];
  __shared__ float   s_mapping_quantiles[256];

  auto& p_histogram = p_histogram_buf[0];
  extern __shared__ int p_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  if (mapping.mode == MAPPING_LUT_CDF && mapping.lut != nullptr) {
    if (tx < 256) s_mapping_lut[tx] = mapping.lut[tx];
    __syncthreads();
  }
  if (mapping.mode == MAPPING_QUANTILE && mapping.quantiles != nullptr) {
    if (tx < 256) s_mapping_quantiles[tx] = mapping.quantiles[tx];
    __syncthreads();
  }

  if (tx < RADIX + 1) p_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const float raw = vortex_to_float(input[idx + row_start]);
    const auto bin = compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles);
    ::atomicAdd(&p_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = p_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += p_histogram_buf[k][tx + j];
        }
        p_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && p_histogram[tx] > topk && p_histogram[tx + 1] <= topk) {
    p_threshold_bin_id = tx;
    p_num_input[0] = 0;
    p_counter = 0;
  }
  __syncthreads();

  const int threshold_bin_0 = p_threshold_bin_id;
  const int threshold_bin_size = p_histogram[threshold_bin_0];  // pre-reset count
  topk -= p_histogram[threshold_bin_0 + 1];

  if (tx == 0 && counters) {
    counters[COUNTER_THRESHOLD_BIN] = threshold_bin_0;
    counters[COUNTER_NUM_EQUAL]     = threshold_bin_size;
    counters[COUNTER_REMAINING_K]   = topk;
  }

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw = vortex_to_float(input[idx + row_start]);
      const auto bin = static_cast<int>(
          compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles));
      if (bin > threshold_bin_0) {
        const auto pos = ::atomicAdd(&p_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    if (tx == 0 && counters) {
      counters[COUNTER_NUM_ABOVE]     = p_counter;
      counters[COUNTER_REFINE_ROUNDS] = 0;
      counters[COUNTER_STAGE2_INPUT]  = 0;
    }
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) p_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const float raw = vortex_to_float(input[idx + row_start]);
      const float remapped = apply_transform(raw, mapping);
      const auto bin = static_cast<int>(
          compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles));
      if (bin > threshold_bin_0) {
        const auto pos = ::atomicAdd(&p_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin_0) {
        const auto pos = ::atomicAdd(&p_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          p_input_idx[0][pos] = idx;
          const auto b32 = convert_to_uint32(remapped);
          const auto sub_bin = (b32 >> 24) & 0xFF;
          ::atomicAdd(&p_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
    if (tx == 0 && counters) {
      counters[COUNTER_NUM_ABOVE]    = p_counter;
      counters[COUNTER_STAGE2_INPUT] = p_num_input[0];
    }
  }

  // Stage 2 refinement (4 rounds max). Default rounds=4, overwritten on exit.
  if (tx == 0 && counters) counters[COUNTER_REFINE_ROUNDS] = 4;
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int p_last_remain;
    const auto r_idx = round % 2;
    const auto _raw_num_input = p_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && p_histogram[tx] > topk && p_histogram[tx + 1] <= topk) {
      p_threshold_bin_id = tx;
      p_num_input[r_idx ^ 1] = 0;
      p_last_remain = topk - p_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = p_threshold_bin_id;
    topk -= p_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = p_input_idx[r_idx][i];
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform(raw, mapping);
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(remapped) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&p_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      if (tx == 0 && counters) counters[COUNTER_REFINE_ROUNDS] = round + 1;
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) p_histogram[tx] = 0;
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = p_input_idx[r_idx][i];
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform(raw, mapping);
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(remapped) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&p_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&p_last_remain, -1);
            if (pos > 0) {
              index[target_k - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&p_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              p_input_idx[r_idx ^ 1][pos] = idx;
              const auto b32 = convert_to_uint32(remapped);
              const auto sub_bin = (b32 >> (offset - 8)) & 0xFF;
              ::atomicAdd(&p_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

// Wrapper: one block per (batch*head) segment. Writes counters per
// segment into a [eff_batch_size, NUM_TOPK_COUNTERS] int32 tensor.
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKProfileCounters_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    int*          __restrict__ counters,
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
  fast_topk_profile<ScoreT>(
      score_blk, s_indices, 0, nblk, topk_val, mapping,
      counters + bx * NUM_TOPK_COUNTERS);
  __syncthreads();

  const int tx = threadIdx.x;
  for (int i = tx; i < topk_val; i += kThreadsPerBlock) {
    out_blk[i] = idx_blk[s_indices[i]];
  }
}

// Histogram-only profiling kernel: builds a 256-bin histogram of the
// remapped bins for each segment. Purely diagnostic — never timed.
template <typename ScoreT>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKProfileHistogram_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    int*          __restrict__ histograms,   // [eff_batch_size, 256]
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const TopKMappingParams mapping)
{
  constexpr auto RADIX = 256;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  __shared__ int s_histogram[RADIX];
  __shared__ uint8_t s_mapping_lut[256];
  __shared__ float   s_mapping_quantiles[256];

  const int bx = blockIdx.x;
  const int tx = threadIdx.x;

  const int start = dense_kv_indptr[bx] + page_reserved_bos;
  const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
  const int nblk  = end - start;

  if (mapping.mode == MAPPING_LUT_CDF && mapping.lut != nullptr) {
    if (tx < 256) s_mapping_lut[tx] = mapping.lut[tx];
    __syncthreads();
  }
  if (mapping.mode == MAPPING_QUANTILE && mapping.quantiles != nullptr) {
    if (tx < 256) s_mapping_quantiles[tx] = mapping.quantiles[tx];
    __syncthreads();
  }

  if (tx < RADIX) s_histogram[tx] = 0;
  __syncthreads();

  if (nblk > 0) {
    const ScoreT* __restrict__ score_blk = score + start;
    for (int i = tx; i < nblk; i += BLOCK_SIZE) {
      const float raw = vortex_to_float(score_blk[i]);
      const auto bin = compute_stage1_bin(raw, mapping, s_mapping_lut, s_mapping_quantiles);
      ::atomicAdd(&s_histogram[bin], 1);
    }
  }
  __syncthreads();

  int* __restrict__ out = histograms + bx * RADIX;
  if (tx < RADIX) out[tx] = s_histogram[tx];
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

static TopKMappingParams build_mapping_params(
    int64_t mapping_mode, double mapping_power,
    std::optional<at::Tensor>& mapping_lut,
    std::optional<at::Tensor>& mapping_quantiles)
{
  TopKMappingParams m{};
  m.mode = static_cast<int>(mapping_mode);
  m.power_exp = static_cast<float>(mapping_power);
  m.lut = nullptr;
  m.quantiles = nullptr;
  if (mapping_lut.has_value()) {
    const auto& lut = mapping_lut.value();
    TORCH_CHECK(lut.is_cuda(), "mapping_lut must be a CUDA tensor");
    TORCH_CHECK(lut.dim() == 1 && lut.size(0) == 256 && lut.scalar_type() == at::ScalarType::Byte,
                "mapping_lut must be a 1D uint8 tensor of size 256");
    m.lut = lut.data_ptr<uint8_t>();
  }
  if (mapping_quantiles.has_value()) {
    const auto& q = mapping_quantiles.value();
    TORCH_CHECK(q.is_cuda(), "mapping_quantiles must be a CUDA tensor");
    TORCH_CHECK(q.dim() == 1 && q.size(0) == 256 && q.scalar_type() == at::ScalarType::Float,
                "mapping_quantiles must be a 1D float32 tensor of size 256");
    m.quantiles = q.data_ptr<float>();
  }
  return m;
}

// ======================================================================
// Profiling: per-segment 256-bin histograms of Stage 1 remapped bins.
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
    std::optional<at::Tensor> mapping_quantiles)
{
  CHECK_CUDA(x);
  CHECK_CUDA(dense_kv_indptr);
  CHECK_CUDA(histograms);
  TORCH_CHECK(histograms.dim() == 2 && histograms.size(0) == eff_batch_size
              && histograms.size(1) == 256,
              "histograms must be [eff_batch_size, 256]");
  TORCH_CHECK(histograms.scalar_type() == at::ScalarType::Int,
              "histograms must be int32");

  auto mapping = build_mapping_params(mapping_mode, mapping_power, mapping_lut, mapping_quantiles);

  dim3 nblks(eff_batch_size);
  dim3 nthreads(kThreadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (x.scalar_type() == at::ScalarType::BFloat16) {
    TopKProfileHistogram_Kernel<__nv_bfloat16><<<nblks, nthreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        dense_kv_indptr.data_ptr<int>(),
        histograms.data_ptr<int>(),
        reserved_bos, reserved_eos, mapping);
  } else if (x.scalar_type() == at::ScalarType::Float) {
    TopKProfileHistogram_Kernel<float><<<nblks, nthreads, 0, stream>>>(
        x.data_ptr<float>(),
        dense_kv_indptr.data_ptr<int>(),
        histograms.data_ptr<int>(),
        reserved_bos, reserved_eos, mapping);
  } else {
    TORCH_CHECK(false, "topk_profile_histogram: unsupported dtype ", x.scalar_type());
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "topk_profile_histogram kernel failed: ", ::cudaGetErrorString(result));
}

// ======================================================================
// Profiling: full pipeline + per-segment diagnostic counters.
// Adds extra global-memory writes — never use for latency measurement.
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
    std::optional<at::Tensor> mapping_quantiles)
{
  TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
              "topk_profile_counters: topk_val (", topk_val,
              ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");
  CHECK_CUDA(x);
  CHECK_CUDA(dense_kv_indptr);
  CHECK_CUDA(sparse_kv_indptr);
  CHECK_CUDA(dense_kv_indices);
  CHECK_CUDA(sparse_kv_indices);
  CHECK_CUDA(counters);
  TORCH_CHECK(counters.dim() == 2 && counters.size(0) == eff_batch_size
              && counters.size(1) == NUM_TOPK_COUNTERS,
              "counters must be [eff_batch_size, ", NUM_TOPK_COUNTERS, "]");
  TORCH_CHECK(counters.scalar_type() == at::ScalarType::Int, "counters must be int32");

  auto mapping = build_mapping_params(mapping_mode, mapping_power, mapping_lut, mapping_quantiles);

  dim3 nblks(eff_batch_size);
  dim3 nthreads(kThreadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (x.scalar_type() == at::ScalarType::BFloat16) {
    setup_kernel_smem_once<TopKProfileCounters_Kernel<__nv_bfloat16>, kSmem>();
    TopKProfileCounters_Kernel<__nv_bfloat16><<<nblks, nthreads, kSmem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        counters.data_ptr<int>(),
        topk_val, reserved_bos, reserved_eos, mapping);
  } else if (x.scalar_type() == at::ScalarType::Float) {
    setup_kernel_smem_once<TopKProfileCounters_Kernel<float>, kSmem>();
    TopKProfileCounters_Kernel<float><<<nblks, nthreads, kSmem, stream>>>(
        x.data_ptr<float>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        counters.data_ptr<int>(),
        topk_val, reserved_bos, reserved_eos, mapping);
  } else {
    TORCH_CHECK(false, "topk_profile_counters: unsupported dtype ", x.scalar_type());
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "topk_profile_counters kernel failed: ", ::cudaGetErrorString(result));
}
