/**
 * Vortex TopK parallel kernel (single-kernel, last-CTA-wins merge).
 *
 * Motivation: the single-CTA fused kernel in topk_sglang.cu pins each
 * batch segment to one CTA, which underutilises the GPU for small
 * effective batch sizes (e.g. bs=4 on H100 leaves ~97% of SMs idle).
 *
 * This kernel launches `num_splits * eff_batch_size` CTAs in a single
 * launch. CTAs sharing the same `bx` (batch index) partition that
 * batch's score range `num_splits` ways and each compute a per-partition
 * top-K via the same two-stage radix the fused kernel uses. Partial
 * results are written into a per-batch workspace.
 *
 * Merge is done WITHOUT a second kernel launch. Each CTA, after
 * finishing its partition's top-K, does `atomicAdd(&done_counter[bx],
 * 1)`. The CTA whose atomicAdd returns `num_splits - 1` is the last
 * one to arrive for batch bx, and it alone carries out the merge:
 * reads the `num_splits * topk_val` candidates from the workspace,
 * runs a small two-stage radix on the already-remapped keys, writes
 * final top-K page IDs to sparse_kv_indices.
 *
 * Correctness: per-partition top-K is a conservative upper bound on
 * the global top-K (worst case: all top-K items land in one
 * partition). Every global top-K item is therefore guaranteed to be
 * in some partition's top-K, and the merge picks the final top-K
 * from the union — sorted-scores match the fused kernel exactly.
 * Tie-breaking can differ because radix tie-breaks depend on atomic
 * race order.
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
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <optional>

#include "register.h"

namespace {

// ---- Launch constants (match topk_sglang.cu) --------------------------------

constexpr int kThreadsPerBlock = 1024;

#ifdef USE_ROCM
#ifdef SGL_TOPK_DYNAMIC_SMEM_BYTES
constexpr size_t kSmem = static_cast<size_t>(SGL_TOPK_DYNAMIC_SMEM_BYTES);
#else
constexpr size_t kSmem = 48 * 1024;
#endif
#else
constexpr size_t kSmem = 8 * 1024 * sizeof(uint32_t);   // 32 KB
#endif

constexpr size_t kFusedSmemMax = 96 * 1024;             // combined kernel dynamic smem ceiling
constexpr int    VORTEX_MAX_TOPK = 2048;

// ---- Program-lifetime done-counter array ----------------------------------
// Used by the last-CTA-wins barrier. __device__ linkage → zero-initialised at
// program startup. atomicInc(ptr, num_splits-1) cycles each entry back to 0
// after every launch, so we never pay a cudaMemset on entry to the host fn.
// Sized for the largest realistic effective batch we'd ever run through the
// parallel kernel (decode bs×heads). Host validates the cap.
constexpr int kMaxParallelEffBs = 8192;
__device__ int g_parallel_done_counter[kMaxParallelEffBs];

// ---- Device helpers (duplicated from topk_sglang.cu) -----------------------

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8_dense(float x) -> uint8_t {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t key  = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return static_cast<uint8_t>((key >> 16) & 0xFFu);
}

template <typename T>
__device__ __forceinline__ float vortex_to_float(T x);
template <>
__device__ __forceinline__ float vortex_to_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ float vortex_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

#include "topk_mapping.cuh"

// ============================================================================
// fast_topk_partition<ScoreT, MODE>
//
// Per-partition two-stage radix. Same algorithm as the fused kernel's
// fast_topk_clean_fused in topk_sglang.cu, with identical mapping-mode
// dispatch and bucket selection. Returns slice-local indices of the
// top `target_k` elements in `index`.
//
// Reuses the caller-provided extern shared memory region `f_input_idx`
// (2 × SMEM_INPUT_SIZE ints) and the `s_bins` byte cache immediately
// after it. The caller also supplies the static histogram / counter
// storage through the template's body — each device-function-private
// __shared__ declaration gets its own offset, but total static smem
// stays small enough to fit comfortably alongside the dynamic region.
// ============================================================================
template <typename ScoreT, int MODE>
__device__ void fast_topk_partition(
    const ScoreT* __restrict__ input,
    int*          __restrict__ index,
    int*          __restrict__ f_input_idx_raw,   // 2 × SMEM_INPUT_SIZE ints
    uint8_t*      __restrict__ s_bins,            // `length` bytes
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

  auto& f_histogram = f_histogram_buf[0];

  // Treat the caller's extern-smem region as two banks of SMEM_INPUT_SIZE ints.
  auto f_input_idx = [&](int bank, int pos) -> int& {
    return f_input_idx_raw[bank * SMEM_INPUT_SIZE + pos];
  };

  const int tx = threadIdx.x;

  constexpr bool use_dense_bucket = (MODE == MAPPING_DENSE_MANT);

  if (tx < RADIX + 1) f_histogram[tx] = 0;
  __syncthreads();

  // Stage 1 pass 1: bin every element and cache the bin in s_bins so
  // pass 2 doesn't re-load scores or re-apply the mapping.
  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const float raw = vortex_to_float(input[idx + row_start]);
    const float remapped = apply_transform_tmpl<MODE>(raw, mapping.power_exp);
    int bin;
    if constexpr (use_dense_bucket) bin = static_cast<int>(convert_to_uint8_dense(remapped));
    else                            bin = static_cast<int>(convert_to_uint8(remapped));
    s_bins[idx] = static_cast<uint8_t>(bin);
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
        if (tx < RADIX - j) value += f_histogram_buf[k][tx + j];
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
      const int bin = static_cast<int>(s_bins[idx]);
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

    constexpr int sub_bin_offset_start = use_dense_bucket ? 8 : 24;
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const int bin = static_cast<int>(s_bins[idx]);
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&f_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform_tmpl<MODE>(raw, mapping.power_exp);
        const auto pos = ::atomicAdd(&f_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          f_input_idx(0, pos) = idx;
          const auto b32 = convert_to_uint32(remapped);
          const auto sub_bin = (b32 >> sub_bin_offset_start) & 0xFF;
          ::atomicAdd(&f_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  constexpr int stage2_offset_start = use_dense_bucket ? 8 : 24;
  constexpr int stage2_max_rounds   = use_dense_bucket ? 2 : 4;
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    if (round >= stage2_max_rounds) break;
    __shared__ int f_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = f_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input
                                                                  : int(SMEM_INPUT_SIZE);
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
        const auto idx = f_input_idx(r_idx, i);
        const auto offset = stage2_offset_start - round * 8;
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform_tmpl<MODE>(raw, mapping.power_exp);
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
        const auto idx = f_input_idx(r_idx, i);
        const float raw = vortex_to_float(input[idx + row_start]);
        const float remapped = apply_transform_tmpl<MODE>(raw, mapping.power_exp);
        const auto offset = stage2_offset_start - round * 8;
        const auto bin = (convert_to_uint32(remapped) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&f_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == stage2_max_rounds - 1) {
            const auto pos = ::atomicAdd(&f_last_remain, -1);
            if (pos > 0) index[target_k - pos] = idx;
          } else {
            const auto pos = ::atomicAdd(&f_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              f_input_idx(r_idx ^ 1, pos) = idx;
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

// ============================================================================
// fast_topk_merge<MODE>
//
// Run by the last-arriving CTA of each batch. Input is the combined
// candidate list (`num_splits * topk_val` float keys + int indices,
// with idx==-1 marking sentinel slots). Reuses the same extern-smem
// region `s_input_idx_raw` that Phase 1 used — its earlier contents
// are dead at this point. Output: top-`target_k` positions into
// `index`, indexing the combined candidate list.
//
// Bucketing matches the fused kernel's bucketing for the given MODE
// so the merged top-K is lossless modulo atomic tie-break order.
// ============================================================================
template <int MODE>
__device__ void fast_topk_merge(
    const float* __restrict__ input,
    const int*   __restrict__ valid_mask,
    int*         __restrict__ index,
    int*         __restrict__ s_input_idx_raw,   // 2 × SMEM_INPUT_SIZE ints
    int          row_start,
    int          length,
    int          target_k)
{
  int topk = target_k;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));
  constexpr bool use_dense_bucket = (MODE == MAPPING_DENSE_MANT);
  constexpr int  stage2_offset_start = use_dense_bucket ? 8 : 24;
  constexpr int  stage2_max_rounds   = use_dense_bucket ? 2 : 4;

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  auto s_input_idx = [&](int bank, int pos) -> int& {
    return s_input_idx_raw[bank * SMEM_INPUT_SIZE + pos];
  };

  const int tx = threadIdx.x;

  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    if (valid_mask[idx + row_start] < 0) continue;   // sentinel; skip
    const float v = input[idx + row_start];
    int bin;
    if constexpr (use_dense_bucket) bin = static_cast<int>(convert_to_uint8_dense(v));
    else                            bin = static_cast<int>(convert_to_uint8(v));
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
        if (tx < RADIX - j) value += s_histogram_buf[k][tx + j];
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
      if (valid_mask[idx + row_start] < 0) continue;
      const float v = input[idx + row_start];
      int bin;
      if constexpr (use_dense_bucket) bin = static_cast<int>(convert_to_uint8_dense(v));
      else                            bin = static_cast<int>(convert_to_uint8(v));
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
      if (valid_mask[idx + row_start] < 0) continue;
      const auto raw_input = input[idx + row_start];
      int bin;
      if constexpr (use_dense_bucket) bin = static_cast<int>(convert_to_uint8_dense(raw_input));
      else                            bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx(0, pos) = idx;
          const auto b32 = convert_to_uint32(raw_input);
          const auto sub_bin = (b32 >> stage2_offset_start) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    if (round >= stage2_max_rounds) break;
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input
                                                                  : int(SMEM_INPUT_SIZE);
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
        const auto idx = s_input_idx(r_idx, i);
        const auto offset = stage2_offset_start - round * 8;
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
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx(r_idx, i);
        const auto raw_input = input[idx + row_start];
        const auto offset = stage2_offset_start - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == stage2_max_rounds - 1) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) index[target_k - pos] = idx;
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx(r_idx ^ 1, pos) = idx;
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

// ============================================================================
// Combined kernel.
//
// Grid: (num_splits, eff_batch_size). Every CTA:
//   1. Computes its partition's top-K (fast_topk_partition).
//   2. Writes (remapped key, batch-local idx) pairs + sentinels to the
//      per-batch workspace slot.
//   3. __threadfence() to publish the writes, then atomicAdd on the
//      per-batch done-counter. The CTA whose atomicAdd returns
//      num_splits - 1 is the last one for this batch.
//   4. If last: run the merge (fast_topk_merge) on the combined
//      num_splits*topk_val candidates and write final page IDs to
//      sparse_kv_indices. Other CTAs exit.
// ============================================================================
template <typename ScoreT, int MODE>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopKOutput_Parallel_Kernel(
    const ScoreT* __restrict__ score,
    const int*    __restrict__ dense_kv_indptr,
    const int*    __restrict__ sparse_kv_indptr,
    const int*    __restrict__ dense_kv_indices,
    int*          __restrict__ sparse_kv_indices,
    float*        __restrict__ partial_keys,   // [eff_bs * num_splits * topk_val]
    int*          __restrict__ partial_idx,    // [eff_bs * num_splits * topk_val]
    const int     topk_val,
    const int     num_splits,
    const int     page_reserved_bos,
    const int     page_reserved_eos,
    const int     chunk_bytes,                 // smem bytes reserved for s_bins
    const TopKMappingParams mapping)
{
  // ---- Dynamic smem layout -------------------------------------------------
  // [ f_input_idx (2 × SMEM_INPUT_SIZE ints = kSmem bytes)
  //   s_bins      (chunk_bytes, only valid during Phase 1) ]
  // The merge doesn't touch s_bins, so its extern region overlaps
  // f_input_idx harmlessly.
  extern __shared__ int smem_scratch[];
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));
  int*      f_input_idx_raw = smem_scratch;
  uint8_t*  s_bins          = reinterpret_cast<uint8_t*>(&smem_scratch[2 * SMEM_INPUT_SIZE]);
  (void)chunk_bytes;  // sizing is the host's responsibility; kernel just uses it

  // s_indices doubles as the partition's radix output AND the merge's radix
  // output — they run sequentially on the same CTA, so the same ~2K slots
  // are reused. Stores up to VORTEX_MAX_TOPK = 2048 entries.
  __shared__ int s_indices[VORTEX_MAX_TOPK];
  // Broadcasts whether this CTA is the last-arriving one for its batch.
  __shared__ int s_is_last;

  const int p  = blockIdx.x;
  const int bx = blockIdx.y;
  const int tx = threadIdx.x;

  const int start = dense_kv_indptr[bx] + page_reserved_bos;
  const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
  const int total_len = end - start;

  // Short batch: fused kernel returns without writing; match that.
  if (total_len <= topk_val) return;

  const size_t slot_base = (static_cast<size_t>(bx) * num_splits + p) * topk_val;
  float* keys_out = partial_keys + slot_base;
  int*   idx_out  = partial_idx  + slot_base;

  const int chunk       = (total_len + num_splits - 1) / num_splits;
  const int part_start  = p * chunk;
  const int raw_part_end = part_start + chunk;
  const int part_end    = raw_part_end < total_len ? raw_part_end : total_len;
  const int part_len    = (part_end > part_start) ? (part_end - part_start) : 0;

  // Sentinel tail: merge filters these by idx == -1. Only fill the range
  // that won't be overwritten with real data.
  const int real_fill = (part_len < topk_val) ? part_len : topk_val;
  const int tail_count = topk_val - real_fill;
  if (tail_count > 0) {
    for (int i = tx; i < tail_count; i += blockDim.x) {
      keys_out[real_fill + i] = -CUDART_INF_F;
      idx_out [real_fill + i] = -1;
    }
    __syncthreads();
  }

  const ScoreT* __restrict__ slice_ptr = score + start + part_start;

  // ---- Phase 1: per-partition top-K ---------------------------------------
  if (part_len > 0) {
    if (part_len <= topk_val) {
      // Whole slice fits under topk_val — emit it directly.
      for (int i = tx; i < part_len; i += blockDim.x) {
        const float raw = vortex_to_float(slice_ptr[i]);
        const float remapped = apply_transform_tmpl<MODE>(raw, mapping.power_exp);
        keys_out[i] = remapped;
        idx_out [i] = part_start + i;
      }
    } else {
      fast_topk_partition<ScoreT, MODE>(
          slice_ptr, s_indices, f_input_idx_raw, s_bins,
          0, part_len, topk_val, mapping);
      __syncthreads();
      for (int i = tx; i < topk_val; i += blockDim.x) {
        const int sl = s_indices[i];
        const float raw = vortex_to_float(slice_ptr[sl]);
        const float remapped = apply_transform_tmpl<MODE>(raw, mapping.power_exp);
        keys_out[i] = remapped;
        idx_out [i] = part_start + sl;
      }
    }
  }

  // Publish workspace writes so the last-CTA can observe them.
  __threadfence();
  __syncthreads();

  // ---- Arrive at the barrier via atomicInc --------------------------------
  // atomicInc(ptr, N-1) stores `((old >= N-1) ? 0 : old+1)` and returns old.
  // So with N == num_splits the counter cycles 0→1→…→N-1→0 per call, which
  // means we never need to memset done_counter between calls — after the
  // last-CTA's increment it's back at 0, ready for the next launch.
  // (Relies on the caller allocating done_counter zero-initialised once.)
  if (tx == 0) {
    const unsigned int old = ::atomicInc(
        reinterpret_cast<unsigned int*>(&g_parallel_done_counter[bx]),
        static_cast<unsigned int>(num_splits - 1));
    s_is_last = (old == static_cast<unsigned int>(num_splits - 1)) ? 1 : 0;
  }
  __syncthreads();

  if (s_is_last == 0) return;

  // ---- Merge: last CTA selects final top-K --------------------------------
  const int candidate_len = num_splits * topk_val;
  const size_t batch_base  = static_cast<size_t>(bx) * candidate_len;
  const float* keys_blk    = partial_keys + batch_base;
  const int*   idx_blk     = partial_idx  + batch_base;
  int*         out_blk     = sparse_kv_indices
                            + sparse_kv_indptr[bx]
                            + page_reserved_bos;
  const int*   dense_blk   = dense_kv_indices + start;

  fast_topk_merge<MODE>(
      keys_blk, idx_blk, s_indices, f_input_idx_raw,
      0, candidate_len, topk_val);
  __syncthreads();

  for (int i = tx; i < topk_val; i += blockDim.x) {
    const int pos = s_indices[i];
    const int batch_local = idx_blk[pos];
    out_blk[i] = (batch_local >= 0) ? dense_blk[batch_local] : -1;
  }
}

// ---- setup_kernel_smem_once (duplicated) -----------------------------------

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f),
        ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    return ::cudaFuncSetAttribute(
        f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  TORCH_CHECK(result == cudaSuccess,
              "set_up_kernel_once (parallel) failed:", ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// ============================================================================
// Host entry point.
//
// Signature matches topk_output_sglang_fused plus `num_splits`.
// `num_splits <= 1` delegates to the single-CTA fused kernel so callers
// can unconditionally use this path.
// ============================================================================
void topk_output_sglang_parallel(
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
    const int64_t     num_splits,
    const int64_t     mapping_mode,
    const double      mapping_power,
    std::optional<at::Tensor> mapping_lut,
    std::optional<at::Tensor> mapping_quantiles)
{
    TORCH_CHECK(topk_val <= VORTEX_MAX_TOPK,
                "topk_output_sglang_parallel: topk_val (", topk_val,
                ") exceeds VORTEX_MAX_TOPK (", VORTEX_MAX_TOPK, ")");
    TORCH_CHECK(num_splits >= 1,
                "topk_output_sglang_parallel: num_splits must be >= 1");

    if (num_splits <= 1) {
        topk_output_sglang_fused(
            x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
            sparse_kv_indices, eff_batch_size, topk_val,
            reserved_bos, reserved_eos, max_num_pages,
            mapping_mode, mapping_power, mapping_lut, mapping_quantiles);
        return;
    }

    CHECK_CUDA(x);
    CHECK_CUDA(dense_kv_indptr);
    CHECK_CUDA(sparse_kv_indptr);
    CHECK_CUDA(dense_kv_indices);
    CHECK_CUDA(sparse_kv_indices);

    (void)mapping_lut;
    (void)mapping_quantiles;

    TopKMappingParams mapping{};
    mapping.mode      = static_cast<int>(mapping_mode);
    mapping.power_exp = static_cast<float>(mapping_power);
    mapping.lut       = nullptr;
    mapping.quantiles = nullptr;

    // Dynamic smem = kSmem (f_input_idx) + chunk_bytes (s_bins for the
    // partition radix; the merge doesn't touch s_bins).
    const int64_t chunk_pages = (max_num_pages + num_splits - 1) / num_splits;
    const size_t chunk_bytes = (static_cast<size_t>(chunk_pages) + size_t(15)) & ~size_t(15);
    const size_t smem_bytes = kSmem + chunk_bytes;
    TORCH_CHECK(smem_bytes <= kFusedSmemMax,
                "topk_output_sglang_parallel: smem ", smem_bytes,
                " exceeds ceiling ", kFusedSmemMax);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(eff_batch_size <= kMaxParallelEffBs,
                "topk_output_sglang_parallel: eff_batch_size (", eff_batch_size,
                ") exceeds kMaxParallelEffBs (", kMaxParallelEffBs,
                "). Raise the __device__ counter array size.");

    // Per-call workspace. at::empty, no zero-init — kernel fills every used
    // slot (valid prefix + sentinel tail). done_counter is a __device__
    // global (above) so no workspace allocation needed for it.
    const int64_t ws_elems = eff_batch_size * num_splits * topk_val;
    auto opts_f32 = at::TensorOptions().device(x.device()).dtype(at::kFloat);
    auto opts_i32 = at::TensorOptions().device(x.device()).dtype(at::kInt);
    at::Tensor partial_keys = at::empty({ws_elems}, opts_f32);
    at::Tensor partial_idx  = at::empty({ws_elems}, opts_i32);

    dim3 grid(static_cast<unsigned>(num_splits),
              static_cast<unsigned>(eff_batch_size));
    dim3 nthreads(kThreadsPerBlock);

    #define VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MODE_VAL)                     \
        do {                                                                         \
            setup_kernel_smem_once<                                                  \
                TopKOutput_Parallel_Kernel<DTYPE, MODE_VAL>,                         \
                kFusedSmemMax>();                                                    \
            TopKOutput_Parallel_Kernel<DTYPE, MODE_VAL>                              \
                <<<grid, nthreads, smem_bytes, stream>>>(                            \
                    PTR_EXPR,                                                        \
                    dense_kv_indptr.data_ptr<int>(),                                 \
                    sparse_kv_indptr.data_ptr<int>(),                                \
                    dense_kv_indices.data_ptr<int>(),                                \
                    sparse_kv_indices.data_ptr<int>(),                               \
                    partial_keys.data_ptr<float>(),                                  \
                    partial_idx.data_ptr<int>(),                                     \
                    static_cast<int>(topk_val),                                      \
                    static_cast<int>(num_splits),                                    \
                    static_cast<int>(reserved_bos),                                  \
                    static_cast<int>(reserved_eos),                                  \
                    static_cast<int>(chunk_bytes),                                   \
                    mapping);                                                        \
        } while (0)

    #define VORTEX_PARALLEL_DISPATCH_MODE(DTYPE, PTR_EXPR)                          \
        do {                                                                         \
            switch (mapping.mode) {                                                  \
                case MAPPING_NONE:        VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_NONE); break; \
                case MAPPING_POWER:       VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_POWER); break; \
                case MAPPING_LOG:         VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_LOG); break; \
                case MAPPING_ASINH:       VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_ASINH); break; \
                case MAPPING_LOG1P:       VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_LOG1P); break; \
                case MAPPING_TRUNC8:      VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_TRUNC8); break; \
                case MAPPING_ERF:         VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_ERF); break; \
                case MAPPING_TANH:        VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_TANH); break; \
                case MAPPING_SUBTRACT:    VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_SUBTRACT); break; \
                case MAPPING_EXP_STRETCH: VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_EXP_STRETCH); break; \
                case MAPPING_SHIFT_POW2:  VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW2); break; \
                case MAPPING_SHIFT_POW3:  VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW3); break; \
                case MAPPING_LINEAR_STEEP:VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_LINEAR_STEEP); break; \
                case MAPPING_HALF_SQUARE: VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_HALF_SQUARE); break; \
                case MAPPING_HALF_CUBE:   VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_HALF_CUBE); break; \
                case MAPPING_DENSE_MANT:  VORTEX_PARALLEL_DISPATCH(DTYPE, PTR_EXPR, MAPPING_DENSE_MANT); break; \
                default:                                                             \
                    TORCH_CHECK(false,                                               \
                        "topk_output_sglang_parallel: unsupported mapping_mode ",    \
                        mapping.mode);                                               \
            }                                                                        \
        } while (0)

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        VORTEX_PARALLEL_DISPATCH_MODE(
            __nv_bfloat16,
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
    } else if (x.scalar_type() == at::ScalarType::Float) {
        VORTEX_PARALLEL_DISPATCH_MODE(float, x.data_ptr<float>());
    } else {
        TORCH_CHECK(false, "topk_output_sglang_parallel: unsupported dtype ",
                    x.scalar_type());
    }

    #undef VORTEX_PARALLEL_DISPATCH_MODE
    #undef VORTEX_PARALLEL_DISPATCH

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk_output_sglang_parallel kernel failed: ",
                ::cudaGetErrorString(result));
}
