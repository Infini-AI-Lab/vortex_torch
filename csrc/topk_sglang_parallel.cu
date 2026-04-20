/**
 * Vortex TopK — single-kernel parallel+merge pipeline.
 *
 * ONE kernel launch. Per-chunk selection and cross-chunk merge both run
 * inside the same grid-(N, Batch) launch. The last-arriving CTA for
 * each batch (detected by a program-lifetime __device__ done-counter +
 * atomicInc wrap-around) carries out the merge — no second launch, no
 * per-call cudaMemset for barrier state.
 *
 * Correctness:
 *   Stage 1 per-chunk uses ONE 8-bit radix histogram + ONE 8-bit
 *   refinement round on the threshold bin (16 bits of selection
 *   precision). For bf16 input (8 mantissa bits effective), this is
 *   lossless — two items with the same 16-bit key are bit-identical as
 *   bf16 values.
 *
 *   Stage 2 merge operates on N*K pre-remapped keys in shared memory
 *   and uses the same 8-bit-hist + 8-bit-refine pattern, which is
 *   strictly sufficient to pick the correct top-K from the union.
 *
 * Low-overhead primitives:
 *   - Warp-level ballot+popc compaction on the "bin > threshold" path
 *     so each warp issues ONE atomicAdd on the block counter instead
 *     of one per thread.
 *   - Program-lifetime __device__ done-counter sized for realistic
 *     batch×head counts; atomicInc wraps back to 0 at num_chunks so
 *     there's no memset on the hot path.
 *   - Vectorised float4/int4 loads from global → smem in the merge.
 *
 * Supported mapping modes (IDs from csrc/topk_mapping.cuh):
 *   3=POWER, 6=ASINH, 7=LOG1P, 9=ERF, 10=TANH, 11=SUBTRACT,
 *   13=EXP_STRETCH, 15=SHIFT_POW2, 16=SHIFT_POW3, 17=LINEAR_STEEP.
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

// ---- Launch constants ------------------------------------------------------

constexpr int    kThreadsPerBlock = 1024;
constexpr int    kWarpSize        = 32;
constexpr int    RADIX            = 256;
constexpr size_t kMaxDynSmem      = 96 * 1024;
constexpr int    VORTEX_MAX_TOPK  = 2048;

// Stage-2 holds N*K (key, idx) pairs in smem = 8 B/item.
constexpr int    kMergeCap        = 8192;

// Max batch the single kernel can sequence. Sized for realistic
// bs×heads (decode). __device__ globals are zero-initialised at
// program start; atomicInc wrap-around keeps each entry at 0 between
// launches, so no host-side memset on the hot path.
constexpr int    kMaxBatch        = 8192;
__device__ unsigned int g_done_counter[kMaxBatch];

// ---- Device helpers --------------------------------------------------------

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Required symbol for topk_mapping.cuh's compute_stage1_bin. Not used
// directly by the kernel body here, but the header includes a forward
// declaration that resolves against this definition at link time.
__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
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
// 8-step suffix cumsum over 256 bins. After the call s_hist[0][i] is
// the count of items with bin >= i (monotone non-increasing).
// ============================================================================
__device__ __forceinline__ void run_cumsum_256(int s_hist[2][RADIX + 128]) {
  const int tx = threadIdx.x;
#pragma unroll 8
  for (int i = 0; i < 8; ++i) {
    static_assert(1 << 8 == RADIX);
    if (C10_LIKELY(tx < RADIX)) {
      const int j = 1 << i;
      const int k = i & 1;
      int value = s_hist[k][tx];
      if (tx < RADIX - j) value += s_hist[k][tx + j];
      s_hist[k ^ 1][tx] = value;
    }
    __syncthreads();
  }
}

// ============================================================================
// Warp-level ballot+popc compaction.
//
// Every participating thread offers a boolean `selected`. Exactly ONE
// atomicAdd per warp — issued by the first active lane — reserves
// `warp_count` slots; other selected lanes derive their slot via a
// popc prefix sum. Safe when called from inside a divergent region
// (uses __activemask(), not a fixed all-ones mask).
// ============================================================================
__device__ __forceinline__ int warp_compact_slot(bool selected, int* s_counter) {
  const uint32_t mask         = __activemask();
  const uint32_t ballot       = __ballot_sync(mask, selected);
  const int      lane         = threadIdx.x & (kWarpSize - 1);
  const int      warp_count   = __popc(ballot);
  const int      rank_in_warp = __popc(ballot & ((1u << lane) - 1u));

  const int first_lane = __ffs(mask) - 1;
  int base = 0;
  if (lane == first_lane) {
    base = (warp_count > 0) ? ::atomicAdd(s_counter, warp_count) : 0;
  }
  base = __shfl_sync(mask, base, first_lane);
  return selected ? (base + rank_in_warp) : -1;
}

// ============================================================================
// Combined kernel — Stage 1 (per-chunk) + barrier + Stage 2 (merge).
//
// Grid   = (Batch, N).  One CTA per (batch, chunk).
// Block  = kThreadsPerBlock = 1024.
//
// Shared-memory layout (reused across phases):
//   Phase 1 needs:
//     s_remapped[chunk_size]  (float)  — cached apply_transform output.
//     s_bins[chunk_size]      (uint8)  — cached coarse bin.
//   Merge needs:
//     s_scores[N*K]           (float)  — pair buffer, loaded vectorised.
//     s_indices[N*K]          (int32)  — pair buffer.
//   kSmemBytes is sized to host max of both.
//
// Sync between phases:
//   After Phase 1's workspace writes, __threadfence() publishes them,
//   then thread 0 does `atomicInc(&g_done_counter[bx], N-1)` which
//   cycles 0→1→…→N-1→0 so no reset is needed between calls. The CTA
//   whose returned `old == N-1` is the last one — it falls through
//   into the merge; other CTAs return.
// ============================================================================
template <typename ScoreT, int MODE>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopK_Parallel_Kernel(
    const ScoreT* __restrict__ score,         // [Batch, N, chunk_size]
    int32_t*      __restrict__ global_idx,    // [Batch, K]
    float*        __restrict__ partial_keys,  // [Batch, N, K] workspace
    int32_t*      __restrict__ partial_idx,   // [Batch, N, K] workspace
    int                         N,
    int                         chunk_size,
    int                         K,
    float                       mapping_power)
{
  const int b  = blockIdx.x;
  const int n  = blockIdx.y;
  const int tx = threadIdx.x;

  // Addresses for this CTA's chunk slice and its slot in the workspace.
  const ScoreT* chunk_in      = score        + (static_cast<int64_t>(b) * N + n) * chunk_size;
  float*        chunk_keys_out = partial_keys + (static_cast<int64_t>(b) * N + n) * K;
  int32_t*      chunk_idx_out  = partial_idx  + (static_cast<int64_t>(b) * N + n) * K;
  const int32_t idx_base        = n * chunk_size;  // batch-local offset

  // ---------------------------------------------------------------- smem
  extern __shared__ char smem_raw[];

  // Shared-memory counters / histogram live in static smem so the
  // Phase-1 and merge phases can share the same dynamic pool.
  alignas(128) __shared__ int s_hist_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin;
  alignas(128) __shared__ int s_sub_threshold_bin;
  alignas(128) __shared__ int s_last_remain;
  alignas(128) __shared__ int s_is_last;
  auto& s_hist = s_hist_buf[0];

  // =========================================================================
  // Phase 1: per-chunk TopK via 8-bit radix + 8-bit refinement.
  // =========================================================================
  //
  // Dynamic smem region used as:
  //   s_remapped : chunk_size * 4 B  (cached apply_transform output)
  //   s_bins     : chunk_size * 1 B  (cached Stage-1 bin)
  //
  // Refinement is a second 8-bit bucket on bits [23:16] of the
  // sign-flipped u32 key, used to refine the threshold bin. 8 + 8 =
  // 16 bits of selection precision → lossless for bf16.
  float*   s_remapped = reinterpret_cast<float*>(smem_raw);
  uint8_t* s_bins     = reinterpret_cast<uint8_t*>(s_remapped + chunk_size);

  // ---- Degenerate chunk_size <= K : emit everything as-is. -------------
  if (chunk_size <= K) {
    for (int i = tx; i < K; i += blockDim.x) {
      if (i < chunk_size) {
        const float raw = vortex_to_float(chunk_in[i]);
        chunk_keys_out[i] = apply_transform_tmpl<MODE>(raw, mapping_power);
        chunk_idx_out [i] = i + idx_base;
      } else {
        chunk_keys_out[i] = -CUDART_INF_F;
        chunk_idx_out [i] = -1;
      }
    }
  } else {
    // ---- Histogram pass 1: transform + bucket; cache both to smem. ----
    if (tx < RADIX + 1) s_hist[tx] = 0;
    if (tx == 0) { s_counter = 0; s_threshold_bin = -1; s_last_remain = 0; }
    __syncthreads();

    for (int idx = tx; idx < chunk_size; idx += blockDim.x) {
      const float raw      = vortex_to_float(chunk_in[idx]);
      const float remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
      const uint32_t b32   = convert_to_uint32(remapped);
      const int bin        = (b32 >> 24) & 0xFF;
      s_remapped[idx] = remapped;
      s_bins    [idx] = static_cast<uint8_t>(bin);
      ::atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    run_cumsum_256(s_hist_buf);

    if (tx < RADIX && s_hist[tx] > K && s_hist[tx + 1] <= K) {
      s_threshold_bin = tx;
      s_last_remain   = K - s_hist[tx + 1];
    }
    __syncthreads();
    const int threshold_bin = s_threshold_bin;

    // ---- Emit bin > threshold (warp-popc) and build refinement hist. ----
    if (tx < RADIX + 1) s_hist[tx] = 0;
    __syncthreads();

    const int num_iters = (chunk_size + blockDim.x - 1) / blockDim.x;
    for (int it = 0; it < num_iters; ++it) {
      const int idx = it * blockDim.x + tx;
      const bool in_range = (idx < chunk_size);
      int bin = -1;
      if (in_range) bin = static_cast<int>(s_bins[idx]);
      const bool take_above = in_range && (bin > threshold_bin);

      const int slot = warp_compact_slot(take_above, &s_counter);
      if (take_above) {
        chunk_keys_out[slot] = s_remapped[idx];
        chunk_idx_out [slot] = idx + idx_base;
      } else if (in_range && bin == threshold_bin) {
        const uint32_t b32 = convert_to_uint32(s_remapped[idx]);
        const int sub_bin  = (b32 >> 16) & 0xFF;
        ::atomicAdd(&s_hist[sub_bin], 1);
      }
    }
    __syncthreads();

    // ---- Refinement cumsum → sub-threshold bin. ------------------------
    run_cumsum_256(s_hist_buf);
    if (tx < RADIX && s_hist[tx] > s_last_remain
                   && s_hist[tx + 1] <= s_last_remain) {
      s_sub_threshold_bin = tx;
      // budget for items at the sub-threshold bin
      s_last_remain = s_last_remain - s_hist[tx + 1];
    }
    if (tx == 0 && s_sub_threshold_bin == -1) {
      // Only possible if last_remain == 0 (bin > threshold already emitted
      // exactly K items). Nothing more to do; make the sub bin a sentinel.
      s_sub_threshold_bin = RADIX;   // no sub-threshold bin
    }
    __syncthreads();
    const int sub_threshold_bin = s_sub_threshold_bin;

    // ---- Emit threshold-bin items using sub-threshold logic. ----------
    for (int it = 0; it < num_iters; ++it) {
      const int idx = it * blockDim.x + tx;
      const bool in_range = (idx < chunk_size);
      int bin = -1;
      if (in_range) bin = static_cast<int>(s_bins[idx]);
      int sub_bin = -1;
      if (in_range && bin == threshold_bin) {
        const uint32_t b32 = convert_to_uint32(s_remapped[idx]);
        sub_bin = (b32 >> 16) & 0xFF;
      }

      const bool take_sub_above = (sub_bin > sub_threshold_bin);
      const int slot = warp_compact_slot(take_sub_above, &s_counter);
      if (take_sub_above) {
        chunk_keys_out[slot] = s_remapped[idx];
        chunk_idx_out [slot] = idx + idx_base;
      } else if (sub_bin == sub_threshold_bin) {
        const int pos = ::atomicAdd(&s_last_remain, -1);
        if (pos > 0) {
          chunk_keys_out[K - pos] = s_remapped[idx];
          chunk_idx_out [K - pos] = idx + idx_base;
        }
      }
    }
    __syncthreads();
  }

  // =========================================================================
  // Barrier: publish this CTA's workspace writes and atomicInc the
  // per-batch done-counter. The CTA that sees old == N-1 is the last
  // arriving one; every other CTA returns here.
  // =========================================================================
  __threadfence();
  __syncthreads();
  if (tx == 0) {
    const unsigned int old = ::atomicInc(
        &g_done_counter[b], static_cast<unsigned int>(N - 1));
    s_is_last = (old == static_cast<unsigned int>(N - 1)) ? 1 : 0;
  }
  __syncthreads();
  if (s_is_last == 0) return;

  // =========================================================================
  // Phase 2 (merge, only in last-arriving CTA):
  //   load N*K candidates into smem (vectorised) →
  //   8-bit histogram in smem →
  //   threshold → warp-popc emit above + tie-bin refinement.
  // =========================================================================
  const int total  = N * K;
  const float*   keys_in = partial_keys + static_cast<int64_t>(b) * total;
  const int32_t* idx_in  = partial_idx  + static_cast<int64_t>(b) * total;
  int32_t*       out_idx = global_idx   + static_cast<int64_t>(b) * K;

  // Reuse the same dynamic smem region as Phase 1 — Phase 1's caches
  // are dead now. Layout: [ s_scores : total floats | s_indices : total int32 ].
  float*   s_scores  = reinterpret_cast<float*>(smem_raw);
  int32_t* s_indices = reinterpret_cast<int32_t*>(s_scores + total);

  // Vectorised 128-bit loads when `total` is a multiple of 4.
  if ((total & 3) == 0) {
    const float4* keys_v = reinterpret_cast<const float4*>(keys_in);
    const int4*   idx_v  = reinterpret_cast<const int4*>  (idx_in);
    float4*       ss_v   = reinterpret_cast<float4*>      (s_scores);
    int4*         si_v   = reinterpret_cast<int4*>        (s_indices);
    const int total4 = total >> 2;
    for (int i = tx; i < total4; i += blockDim.x) {
      ss_v[i] = keys_v[i];
      si_v[i] = idx_v [i];
    }
  } else {
    for (int i = tx; i < total; i += blockDim.x) {
      s_scores [i] = keys_in[i];
      s_indices[i] = idx_in [i];
    }
  }

  if (tx < RADIX + 1) s_hist[tx] = 0;
  if (tx == 0) {
    s_counter           = 0;
    s_threshold_bin     = -1;
    s_sub_threshold_bin = -1;
    s_last_remain       = 0;
  }
  __syncthreads();

  // (2) 8-bit histogram in smem.
  const int num_iters_m = (total + blockDim.x - 1) / blockDim.x;
  for (int it = 0; it < num_iters_m; ++it) {
    const int i = it * blockDim.x + tx;
    if (i < total && s_indices[i] >= 0) {
      const uint32_t b32 = convert_to_uint32(s_scores[i]);
      const int bin = (b32 >> 24) & 0xFF;
      ::atomicAdd(&s_hist[bin], 1);
    }
  }
  __syncthreads();

  run_cumsum_256(s_hist_buf);

  // Fast path: no threshold search needed when valid_count ≤ K.
  const int valid_count = s_hist[0];
  if (valid_count <= K) {
    for (int it = 0; it < num_iters_m; ++it) {
      const int i = it * blockDim.x + tx;
      const bool take = (i < total) && (s_indices[i] >= 0);
      const int slot = warp_compact_slot(take, &s_counter);
      if (take) out_idx[slot] = s_indices[i];
    }
    return;
  }

  if (tx < RADIX && s_hist[tx] > K && s_hist[tx + 1] <= K) {
    s_threshold_bin = tx;
    s_last_remain   = K - s_hist[tx + 1];
  }
  __syncthreads();
  const int threshold_bin_m = s_threshold_bin;

  // (3) Emit above threshold via warp-popc; build sub-bin histogram on
  //     bits [23:16] for the tie-bin refinement.
  if (tx < RADIX + 1) s_hist[tx] = 0;
  __syncthreads();

  for (int it = 0; it < num_iters_m; ++it) {
    const int i = it * blockDim.x + tx;
    bool in_valid = false;
    int bin = -1;
    uint32_t b32 = 0;
    if (i < total) {
      const int32_t idx = s_indices[i];
      if (idx >= 0) {
        in_valid = true;
        b32 = convert_to_uint32(s_scores[i]);
        bin = (b32 >> 24) & 0xFF;
      }
    }
    const bool take_above = in_valid && (bin > threshold_bin_m);
    const int  slot       = warp_compact_slot(take_above, &s_counter);
    if (take_above) {
      out_idx[slot] = s_indices[i];
    } else if (in_valid && bin == threshold_bin_m) {
      const int sub_bin = (b32 >> 16) & 0xFF;
      ::atomicAdd(&s_hist[sub_bin], 1);
    }
  }
  __syncthreads();

  // (4) Refinement cumsum → sub-threshold bin.
  run_cumsum_256(s_hist_buf);
  if (tx < RADIX && s_hist[tx] > s_last_remain
                 && s_hist[tx + 1] <= s_last_remain) {
    s_sub_threshold_bin = tx;
    s_last_remain = s_last_remain - s_hist[tx + 1];
  }
  if (tx == 0 && s_sub_threshold_bin == -1) {
    s_sub_threshold_bin = RADIX;   // no tie-bin refinement needed
  }
  __syncthreads();
  const int sub_threshold_bin_m = s_sub_threshold_bin;

  // (5) Emit tie-bin items via warp-popc + sub-threshold budget.
  for (int it = 0; it < num_iters_m; ++it) {
    const int i = it * blockDim.x + tx;
    bool in_threshold = false;
    int sub_bin = -1;
    if (i < total) {
      const int32_t idx = s_indices[i];
      if (idx >= 0) {
        const uint32_t b32 = convert_to_uint32(s_scores[i]);
        const int bin = (b32 >> 24) & 0xFF;
        if (bin == threshold_bin_m) {
          in_threshold = true;
          sub_bin = (b32 >> 16) & 0xFF;
        }
      }
    }
    const bool take_sub_above = in_threshold && (sub_bin > sub_threshold_bin_m);
    const int  slot           = warp_compact_slot(take_sub_above, &s_counter);
    if (take_sub_above) {
      out_idx[slot] = s_indices[i];
    } else if (in_threshold && sub_bin == sub_threshold_bin_m) {
      const int pos = ::atomicAdd(&s_last_remain, -1);
      if (pos > 0) out_idx[K - pos] = s_indices[i];
    }
  }
}

// ---- setup_kernel_smem_once ------------------------------------------------

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
    return ::cudaFuncSetAttribute(
        f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
  }();
  TORCH_CHECK(result == cudaSuccess,
              "fast_fused_topk_merge setup failed: ",
              ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// ============================================================================
// Host entry point.
//
//   score                [batch_size, num_chunks, chunk_size]  bf16 or f32
//   global_topk_indices  [batch_size, topk_val]                int32  (output)
//
// ONE kernel launch. The per-chunk selection (Phase 1) and the
// cross-chunk merge (Phase 2) are fused in TopK_Parallel_Kernel via a
// last-CTA-wins atomicInc barrier. A per-call workspace holds the
// [batch, N, K] partial top-K that the last CTA reads from; the
// done-counter is a program-lifetime __device__ global so nothing
// needs memsetting on the hot path.
// ============================================================================
void fast_fused_topk_merge(
    const at::Tensor& score,
    at::Tensor&       global_topk_indices,
    const int64_t     batch_size,
    const int64_t     num_chunks,
    const int64_t     chunk_size,
    const int64_t     topk_val,
    const int64_t     mapping_mode,
    const double      mapping_power)
{
  CHECK_CUDA(score);
  CHECK_CUDA(global_topk_indices);

  TORCH_CHECK(topk_val > 0 && topk_val <= VORTEX_MAX_TOPK,
              "fast_fused_topk_merge: topk_val=", topk_val,
              " must be in (0, ", VORTEX_MAX_TOPK, "]");
  TORCH_CHECK(num_chunks >= 1,  "num_chunks must be >= 1");
  TORCH_CHECK(batch_size >= 1,  "batch_size must be >= 1");
  TORCH_CHECK(batch_size <= kMaxBatch,
              "fast_fused_topk_merge: batch_size ", batch_size,
              " exceeds the __device__ done-counter cap (", kMaxBatch, ")");
  TORCH_CHECK(chunk_size >= 1,  "chunk_size must be >= 1");
  TORCH_CHECK(num_chunks * topk_val <= kMergeCap,
              "fast_fused_topk_merge: num_chunks*topk_val (",
              num_chunks * topk_val, ") exceeds merge cap (", kMergeCap,
              "). Reduce num_chunks or topk_val.");
  TORCH_CHECK(global_topk_indices.scalar_type() == at::kInt,
              "global_topk_indices must be int32");
  TORCH_CHECK(global_topk_indices.numel() >= batch_size * topk_val,
              "global_topk_indices is too small for batch_size * topk_val");

  TORCH_CHECK(
      mapping_mode == MAPPING_POWER        ||
      mapping_mode == MAPPING_ASINH        ||
      mapping_mode == MAPPING_LOG1P        ||
      mapping_mode == MAPPING_ERF          ||
      mapping_mode == MAPPING_TANH         ||
      mapping_mode == MAPPING_SUBTRACT     ||
      mapping_mode == MAPPING_EXP_STRETCH  ||
      mapping_mode == MAPPING_SHIFT_POW2   ||
      mapping_mode == MAPPING_SHIFT_POW3   ||
      mapping_mode == MAPPING_LINEAR_STEEP,
      "fast_fused_topk_merge: mapping_mode=", mapping_mode,
      " not supported. Valid: POWER(3), ASINH(6), LOG1P(7), ERF(9), "
      "TANH(10), SUBTRACT(11), EXP_STRETCH(13), SHIFT_POW2(15), "
      "SHIFT_POW3(16), LINEAR_STEEP(17).");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const float mp = static_cast<float>(mapping_power);

  // Dynamic smem must fit whichever phase is larger:
  //   Phase 1:  chunk_size floats + chunk_size bytes.
  //   Phase 2:  num_chunks*topk_val * (float + int32).
  const size_t p1_bytes = static_cast<size_t>(chunk_size) * sizeof(float)
                        + ((static_cast<size_t>(chunk_size) + 15) & ~size_t(15));
  const size_t p2_bytes = static_cast<size_t>(num_chunks) *
                          static_cast<size_t>(topk_val) *
                          (sizeof(float) + sizeof(int32_t));
  const size_t smem_bytes = p1_bytes > p2_bytes ? p1_bytes : p2_bytes;
  TORCH_CHECK(smem_bytes <= kMaxDynSmem,
              "fast_fused_topk_merge: smem ", smem_bytes,
              " > ceiling ", kMaxDynSmem);

  // Per-call workspace for the [batch, N, K] partial top-K. at::empty
  // hits the caching allocator (no cudaMalloc in the hot path after
  // warmup). The done-counter lives in __device__ memory — no memset.
  auto opts_f32 = at::TensorOptions().device(score.device()).dtype(at::kFloat);
  auto opts_i32 = at::TensorOptions().device(score.device()).dtype(at::kInt);
  const int64_t ws_elems = batch_size * num_chunks * topk_val;
  at::Tensor partial_keys = at::empty({ws_elems}, opts_f32);
  at::Tensor partial_idx  = at::empty({ws_elems}, opts_i32);

  dim3 grid(static_cast<unsigned>(batch_size),
            static_cast<unsigned>(num_chunks));
  dim3 block(kThreadsPerBlock);

  #define LAUNCH(DTYPE, PTR_EXPR, MODE_VAL)                                    \
    do {                                                                       \
      setup_kernel_smem_once<TopK_Parallel_Kernel<DTYPE, MODE_VAL>,            \
                             kMaxDynSmem>();                                   \
      TopK_Parallel_Kernel<DTYPE, MODE_VAL>                                    \
          <<<grid, block, smem_bytes, stream>>>(                               \
              PTR_EXPR,                                                        \
              global_topk_indices.data_ptr<int32_t>(),                         \
              partial_keys.data_ptr<float>(),                                  \
              partial_idx.data_ptr<int32_t>(),                                 \
              static_cast<int>(num_chunks),                                    \
              static_cast<int>(chunk_size),                                    \
              static_cast<int>(topk_val),                                      \
              mp);                                                             \
    } while (0)

  #define DISPATCH_MODE(DTYPE, PTR_EXPR)                                       \
    do {                                                                       \
      switch (mapping_mode) {                                                  \
        case MAPPING_POWER:        LAUNCH(DTYPE, PTR_EXPR, MAPPING_POWER);        break; \
        case MAPPING_ASINH:        LAUNCH(DTYPE, PTR_EXPR, MAPPING_ASINH);        break; \
        case MAPPING_LOG1P:        LAUNCH(DTYPE, PTR_EXPR, MAPPING_LOG1P);        break; \
        case MAPPING_ERF:          LAUNCH(DTYPE, PTR_EXPR, MAPPING_ERF);          break; \
        case MAPPING_TANH:         LAUNCH(DTYPE, PTR_EXPR, MAPPING_TANH);         break; \
        case MAPPING_SUBTRACT:     LAUNCH(DTYPE, PTR_EXPR, MAPPING_SUBTRACT);     break; \
        case MAPPING_EXP_STRETCH:  LAUNCH(DTYPE, PTR_EXPR, MAPPING_EXP_STRETCH);  break; \
        case MAPPING_SHIFT_POW2:   LAUNCH(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW2);   break; \
        case MAPPING_SHIFT_POW3:   LAUNCH(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW3);   break; \
        case MAPPING_LINEAR_STEEP: LAUNCH(DTYPE, PTR_EXPR, MAPPING_LINEAR_STEEP); break; \
        default: TORCH_CHECK(false, "unreachable mode");                       \
      }                                                                        \
    } while (0)

  if (score.scalar_type() == at::ScalarType::BFloat16) {
    DISPATCH_MODE(__nv_bfloat16,
                  reinterpret_cast<__nv_bfloat16*>(score.data_ptr<at::BFloat16>()));
  } else if (score.scalar_type() == at::ScalarType::Float) {
    DISPATCH_MODE(float, score.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "fast_fused_topk_merge: unsupported dtype ",
                score.scalar_type());
  }

  #undef DISPATCH_MODE
  #undef LAUNCH

  const auto rc = cudaGetLastError();
  TORCH_CHECK(rc == cudaSuccess,
              "fast_fused_topk_merge kernel failed: ", ::cudaGetErrorString(rc));
}
