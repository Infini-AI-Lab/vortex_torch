/**
 * Vortex TopK — Hopper Thread Block Cluster + Distributed Shared Memory
 *                single-kernel fused top-K merge.
 *
 * Grid          = Batch * N CTAs.
 * Cluster dim   = N (runtime, set via cudaLaunchAttributeClusterDimension).
 * Each cluster  = one batch. cluster.block_rank() identifies the chunk.
 *
 * Stage 1 (every CTA): 8-bit radix + 8-bit refinement over its chunk,
 * writing the local top-K (fp32 remapped score + int32 index) into THIS
 * CTA's shared memory — never through global memory.
 *
 * Stage 2 (CTA 0 only): after cluster.sync(), read every CTA's
 * s_export_scores / s_export_indices directly via
 * cg::cluster_group::map_shared_rank() — the reads compile to
 * `ld.shared::cluster`. Build a merged 8-bit histogram, find the
 * coarse threshold, run the standard 8-bit refinement, and emit K
 * indices to global memory using warp-popc compaction.
 *
 * A second cluster.sync() at the end guarantees no CTA exits while
 * CTA 0 is still issuing DSMEM reads into its exported SMEM.
 *
 * sm_90+ only (Hopper, Blackwell). The kernel body is guarded by
 * __CUDA_ARCH__ >= 900 so the file compiles cleanly against the
 * sm_86/sm_89 gencode targets in setup.py — the host entrypoint
 * TORCH_CHECKs the runtime device compute capability.
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

#include <cooperative_groups.h>

#include "register.h"

namespace {

constexpr int    kThreadsPerBlock = 1024;
constexpr int    kWarpSize        = 32;
constexpr int    RADIX            = 256;
constexpr size_t kMaxDynSmem      = 96 * 1024;
constexpr int    VORTEX_MAX_TOPK  = 2048;
constexpr int    kMaxClusterDim   = 8;   // portable TBC cap

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Required by topk_mapping.cuh's forward decl (even though the cluster
// kernel never calls compute_stage1_bin directly).
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

// 8-step suffix cumsum: after the call s_hist[0][i] = count of items
// with bin >= i (monotone non-increasing). Same routine as
// topk_sglang_parallel.cu.
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

// Warp-level ballot+popc compaction. Exactly one atomicAdd per warp,
// issued by the first active lane. Safe from a divergent region.
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

namespace cg = cooperative_groups;

template <typename ScoreT, int MODE>
__global__ __launch_bounds__(kThreadsPerBlock)
void TopK_Cluster_Kernel(
    const ScoreT* __restrict__ score,       // [Batch, N, chunk_size]
    int32_t*      __restrict__ global_idx,  // [Batch, K]
    int                         N,
    int                         chunk_size,
    int                         K,
    float                       mapping_power)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  // Grid layout: dim3(Batch * N). blockIdx.x = b * N + rank.
  const int b  = (blockIdx.x - rank) / N;
  const int tx = threadIdx.x;

  const ScoreT* chunk_in = score + (static_cast<int64_t>(b) * N + rank) * chunk_size;
  const int32_t idx_base = rank * chunk_size;

  // Static SMEM ------------------------------------------------------------
  alignas(128) __shared__ int     s_hist_buf[2][RADIX + 128];
  alignas(128) __shared__ int     s_counter;
  alignas(128) __shared__ int     s_threshold_bin;
  alignas(128) __shared__ int     s_sub_threshold_bin;
  alignas(128) __shared__ int     s_last_remain;
  alignas(128) __shared__ int     s_export_count;
  // Rank 0 only: contiguous staging buffer for the final K indices before
  // the coalesced int4 write to global memory. Sized to VORTEX_MAX_TOPK so
  // we don't need to carve it out of the dynamic smem layout (which must
  // keep the exports at offset 0 for DSMEM visibility).
  alignas(16)  __shared__ int32_t s_final_indices[VORTEX_MAX_TOPK];
  auto& s_hist = s_hist_buf[0];

  // Dynamic SMEM ------------------------------------------------------------
  //   [0, K*4)                       s_export_scores  (fp32)  <- DSMEM-visible
  //   [K*4, K*8)                     s_export_indices (int32) <- DSMEM-visible
  //   [K*8, K*8 + overlay)           Stage-1 cache on ALL ranks; reused by
  //                                  rank 0 in Stage 2 as the N*K merge pool.
  //       Stage 1  : s_remapped[chunk] (fp32) + s_bins[chunk] (uint8 padded)
  //       Stage 2  : s_merge_scores[N*K] (fp32) + s_merge_indices[N*K] (int32)
  //
  // Exports sit at the start of the SMEM pool so the base offset is the
  // same on every cluster CTA — cg::map_shared_rank uses that offset
  // modulo the cluster stride to read a remote CTA.
  extern __shared__ char smem_raw[];
  float*   s_export_scores  = reinterpret_cast<float*>  (smem_raw);
  int32_t* s_export_indices = reinterpret_cast<int32_t*>(smem_raw + K * sizeof(float));
  float*   s_remapped       = reinterpret_cast<float*>  (smem_raw + K * (sizeof(float) + sizeof(int32_t)));
  uint8_t* s_bins           = reinterpret_cast<uint8_t*>(s_remapped + chunk_size);

  // Initialize counters + pad export indices to -1 (so the degenerate
  // chunk_size < K case leaves recognisable empty slots).
  for (int i = tx; i < K; i += blockDim.x) {
    s_export_indices[i] = -1;
    s_export_scores [i] = -CUDART_INF_F;
  }
  if (tx == 0) {
    s_counter           = 0;
    s_threshold_bin     = -1;
    s_sub_threshold_bin = -1;
    s_last_remain       = 0;
    s_export_count      = 0;
  }
  if (tx < RADIX + 1) s_hist[tx] = 0;
  __syncthreads();

  // =========================================================================
  // Stage 1 — local top-K for this chunk.
  // =========================================================================
  if (chunk_size <= K) {
    // Degenerate: emit every valid element.
    for (int idx = tx; idx < chunk_size; idx += blockDim.x) {
      const float raw      = vortex_to_float(chunk_in[idx]);
      const float remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
      const int slot = warp_compact_slot(true, &s_counter);
      if (slot >= 0 && slot < K) {
        s_export_scores [slot] = remapped;
        s_export_indices[slot] = idx + idx_base;
      }
    }
    __syncthreads();
    if (tx == 0) s_export_count = min(s_counter, K);
  } else {
    // Histogram pass 1 ------------------------------------------------------
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

    // Emit bin > threshold; build sub-bin histogram on the tie bin -----
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
        s_export_scores [slot] = s_remapped[idx];
        s_export_indices[slot] = idx + idx_base;
      } else if (in_range && bin == threshold_bin) {
        const uint32_t b32 = convert_to_uint32(s_remapped[idx]);
        const int sub_bin  = (b32 >> 16) & 0xFF;
        ::atomicAdd(&s_hist[sub_bin], 1);
      }
    }
    __syncthreads();

    // Refinement cumsum → sub-threshold bin --------------------------------
    run_cumsum_256(s_hist_buf);
    if (tx < RADIX && s_hist[tx] > s_last_remain
                   && s_hist[tx + 1] <= s_last_remain) {
      s_sub_threshold_bin = tx;
      s_last_remain = s_last_remain - s_hist[tx + 1];
    }
    if (tx == 0 && s_sub_threshold_bin == -1) {
      s_sub_threshold_bin = RADIX;  // no tie refinement needed
    }
    __syncthreads();
    const int sub_threshold_bin = s_sub_threshold_bin;

    // Emit tie-bin items ---------------------------------------------------
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
        s_export_scores [slot] = s_remapped[idx];
        s_export_indices[slot] = idx + idx_base;
      } else if (sub_bin == sub_threshold_bin) {
        const int pos = ::atomicAdd(&s_last_remain, -1);
        if (pos > 0) {
          s_export_scores [K - pos] = s_remapped[idx];
          s_export_indices[K - pos] = idx + idx_base;
        }
      }
    }
    __syncthreads();
    if (tx == 0) s_export_count = K;
  }

  // =========================================================================
  // Stage 2 — CENTRALIZED RANK-0 PULL.
  //
  // Control flow:
  //   barrier #1 (all CTAs)  : release all Stage-1 exports cluster-wide.
  //   rank != 0              : wait at barrier #2 so their exported SMEM
  //                            stays alive while rank 0 pulls, then exit.
  //   rank 0  Step A         : vectorised DSMEM pull of every rank's
  //                            s_export_* into a local N*K merge pool.
  //   rank 0  Step B+C       : single-block 8-bit radix select over the
  //                            merge pool, staging winners into
  //                            s_final_indices via warp-popc + LOCAL
  //                            atomicAdd on &s_counter / &s_last_remain.
  //                            (No DSMEM atomics anywhere.)
  //   rank 0  Step D         : int4-coalesced global store of
  //                            s_final_indices[K] → global_idx[b, :K].
  //   barrier #2 (all CTAs)  : release idle ranks.
  // =========================================================================

  // cluster.sync() is both a cross-CTA barrier AND a cluster-wide release
  // fence on shared memory, so rank 0's upcoming DSMEM reads of remote
  // s_export_* observe the Stage-1 writes above.
  cluster.sync();

  if (rank != 0) {
    cluster.sync();  // final barrier — keeps SMEM alive during rank 0's pull
    return;
  }

  // ---- rank 0 only from here on ------------------------------------------

  // Pre-fill the staging buffer with -1 so that if fewer than K valid
  // candidates exist, the unused tail emits as -1 sentinels rather than
  // stale static-SMEM data.
  for (int i = tx; i < K; i += blockDim.x) s_final_indices[i] = -1;

  // Reset histogram + counters for Stage 2's radix select.
  if (tx < RADIX + 1) s_hist[tx] = 0;
  if (tx == 0) {
    s_counter           = 0;
    s_threshold_bin     = -1;
    s_sub_threshold_bin = -1;
    s_last_remain       = 0;
  }

  // Merge pool: overlays the now-dead Stage-1 cache region. Layout:
  //   [K*8,                   K*8 + N*K*4)      s_merge_scores  (fp32)
  //   [K*8 + N*K*4,           K*8 + N*K*8)      s_merge_indices (int32)
  const int total = N * K;
  float*   s_merge_scores  = reinterpret_cast<float*>  (smem_raw + K * sizeof(float)
                                                               + K * sizeof(int32_t));
  int32_t* s_merge_indices = reinterpret_cast<int32_t*>(s_merge_scores + total);
  __syncthreads();

  // =========================================================================
  // Step A — vectorised DSMEM pull.
  //
  // map_shared_rank(ptr, 0) degenerates to a local load, so we can sweep
  // r=0..N-1 uniformly without special-casing the self-copy.
  // =========================================================================
  #pragma unroll
  for (int r = 0; r < kMaxClusterDim; ++r) {
    if (r >= N) break;
    const float*   rem_scores  = cluster.map_shared_rank(s_export_scores,  r);
    const int32_t* rem_indices = cluster.map_shared_rank(s_export_indices, r);
    float*         dst_scores  = s_merge_scores  + r * K;
    int32_t*       dst_indices = s_merge_indices + r * K;

    if ((K & 3) == 0) {
      const float4* src_s4 = reinterpret_cast<const float4*>(rem_scores);
      const int4*   src_i4 = reinterpret_cast<const int4*  >(rem_indices);
      float4*       dst_s4 = reinterpret_cast<float4*>      (dst_scores);
      int4*         dst_i4 = reinterpret_cast<int4*>        (dst_indices);
      const int K4 = K >> 2;
      for (int i = tx; i < K4; i += blockDim.x) {
        dst_s4[i] = src_s4[i];
        dst_i4[i] = src_i4[i];
      }
    } else {
      for (int i = tx; i < K; i += blockDim.x) {
        dst_scores [i] = rem_scores [i];
        dst_indices[i] = rem_indices[i];
      }
    }
  }
  __syncthreads();

  // =========================================================================
  // Step B+C — local 8-bit radix select over the N*K merge pool, with
  // warp-popc compaction into s_final_indices. Ported from
  // topk_sglang_parallel.cu Phase 2.
  // =========================================================================

  // (1) Coarse 8-bit histogram on bits [31:24] of the sign-flipped score.
  const int num_iters_m = (total + blockDim.x - 1) / blockDim.x;
  for (int it = 0; it < num_iters_m; ++it) {
    const int i = it * blockDim.x + tx;
    if (i < total && s_merge_indices[i] >= 0) {
      const uint32_t b32 = convert_to_uint32(s_merge_scores[i]);
      const int bin = (b32 >> 24) & 0xFF;
      ::atomicAdd(&s_hist[bin], 1);
    }
  }
  __syncthreads();

  run_cumsum_256(s_hist_buf);

  // Fast path: fewer valid candidates than K — emit them all, skip refinement.
  const int valid_count = s_hist[0];
  if (valid_count <= K) {
    for (int it = 0; it < num_iters_m; ++it) {
      const int i = it * blockDim.x + tx;
      const bool take = (i < total) && (s_merge_indices[i] >= 0);
      const int slot = warp_compact_slot(take, &s_counter);
      if (take && slot < K) s_final_indices[slot] = s_merge_indices[i];
    }
  } else {
    if (tx < RADIX && s_hist[tx] > K && s_hist[tx + 1] <= K) {
      s_threshold_bin = tx;
      s_last_remain   = K - s_hist[tx + 1];
    }
    __syncthreads();
    const int threshold_bin_m = s_threshold_bin;

    // (2) Emit above-threshold winners; build sub-bin histogram on tie-bin.
    if (tx < RADIX + 1) s_hist[tx] = 0;
    __syncthreads();

    for (int it = 0; it < num_iters_m; ++it) {
      const int i = it * blockDim.x + tx;
      bool in_valid = false;
      int bin = -1;
      uint32_t b32 = 0;
      if (i < total) {
        const int32_t idx = s_merge_indices[i];
        if (idx >= 0) {
          in_valid = true;
          b32 = convert_to_uint32(s_merge_scores[i]);
          bin = (b32 >> 24) & 0xFF;
        }
      }
      const bool take_above = in_valid && (bin > threshold_bin_m);
      const int  slot       = warp_compact_slot(take_above, &s_counter);
      if (take_above) {
        s_final_indices[slot] = s_merge_indices[i];
      } else if (in_valid && bin == threshold_bin_m) {
        const int sub_bin = (b32 >> 16) & 0xFF;
        ::atomicAdd(&s_hist[sub_bin], 1);
      }
    }
    __syncthreads();

    // (3) Refinement cumsum → sub-threshold bin.
    run_cumsum_256(s_hist_buf);
    if (tx < RADIX && s_hist[tx] > s_last_remain
                   && s_hist[tx + 1] <= s_last_remain) {
      s_sub_threshold_bin = tx;
      s_last_remain = s_last_remain - s_hist[tx + 1];
    }
    if (tx == 0 && s_sub_threshold_bin == -1) {
      s_sub_threshold_bin = RADIX;  // no tie-bin refinement needed
    }
    __syncthreads();
    const int sub_threshold_bin_m = s_sub_threshold_bin;

    // (4) Emit tie-bin items: hard wins via warp-popc, remainder via local
    //     atomic budget. Both atomics hit rank-0's native SMEM only.
    for (int it = 0; it < num_iters_m; ++it) {
      const int i = it * blockDim.x + tx;
      bool in_threshold = false;
      int  sub_bin = -1;
      if (i < total) {
        const int32_t idx = s_merge_indices[i];
        if (idx >= 0) {
          const uint32_t b32 = convert_to_uint32(s_merge_scores[i]);
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
        s_final_indices[slot] = s_merge_indices[i];
      } else if (in_threshold && sub_bin == sub_threshold_bin_m) {
        const int pos = ::atomicAdd(&s_last_remain, -1);
        if (pos > 0) s_final_indices[K - pos] = s_merge_indices[i];
      }
    }
  }

  __syncthreads();

  // =========================================================================
  // Step D — coalesced int4 store of s_final_indices[K] → global_idx[b, :K].
  // =========================================================================
  int32_t* out_idx = global_idx + static_cast<int64_t>(b) * K;
  if ((K & 3) == 0) {
    const int4* src = reinterpret_cast<const int4*>(s_final_indices);
    int4*       dst = reinterpret_cast<int4*>      (out_idx);
    const int K4 = K >> 2;
    for (int i = tx; i < K4; i += blockDim.x) dst[i] = src[i];
  } else {
    for (int i = tx; i < K; i += blockDim.x) out_idx[i] = s_final_indices[i];
  }

  // Final barrier: releases ranks 1..N-1 that were holding their SMEM
  // alive while rank 0 was pulling in Step A.
  cluster.sync();
#else
  // sm_86/sm_89 fallback: host dispatcher TORCH_CHECKs compute
  // capability, so this stub is never actually invoked. The empty
  // body still needs to reference the params so nvcc doesn't warn.
  (void)score; (void)global_idx;
  (void)N;     (void)chunk_size; (void)K; (void)mapping_power;
#endif
}

// One-shot cudaFuncSetAttribute for dynamic smem ceiling.
template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
    return ::cudaFuncSetAttribute(
        f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
  }();
  TORCH_CHECK(result == cudaSuccess,
              "fast_cluster_topk_merge setup failed: ",
              ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// ============================================================================
// Host entry point — fast_cluster_topk_merge.
//
//   score                [batch_size, num_chunks, chunk_size]  bf16 or f32
//   global_topk_indices  [batch_size, topk_val]                int32 (out)
//
// No workspace tensors — Stage-1 partial top-K lives in shared memory,
// consumed by CTA 0 of the cluster via DSMEM.
// ============================================================================
void fast_cluster_topk_merge(
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
              "fast_cluster_topk_merge: topk_val=", topk_val,
              " must be in (0, ", VORTEX_MAX_TOPK, "]");
  TORCH_CHECK(num_chunks >= 1 && num_chunks <= kMaxClusterDim,
              "fast_cluster_topk_merge: num_chunks=", num_chunks,
              " must be in [1, ", kMaxClusterDim, "] (portable TBC cap)");
  TORCH_CHECK(batch_size >= 1,  "batch_size must be >= 1");
  TORCH_CHECK(chunk_size >= 1,  "chunk_size must be >= 1");
  TORCH_CHECK(global_topk_indices.scalar_type() == at::kInt,
              "global_topk_indices must be int32");
  TORCH_CHECK(global_topk_indices.numel() >= batch_size * topk_val,
              "global_topk_indices is too small for batch_size * topk_val");

  TORCH_CHECK(
      mapping_mode == MAPPING_NONE         ||
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
      "fast_cluster_topk_merge: mapping_mode=", mapping_mode,
      " not supported. Valid: NONE(0), POWER(3), ASINH(6), LOG1P(7), "
      "ERF(9), TANH(10), SUBTRACT(11), EXP_STRETCH(13), SHIFT_POW2(15), "
      "SHIFT_POW3(16), LINEAR_STEEP(17).");

  // Hardware capability gate — Thread Block Clusters require sm_90+.
  int dev;
  TORCH_CHECK(::cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice failed");
  cudaDeviceProp prop{};
  TORCH_CHECK(::cudaGetDeviceProperties(&prop, dev) == cudaSuccess,
              "cudaGetDeviceProperties failed");
  TORCH_CHECK(prop.major >= 9,
              "fast_cluster_topk_merge requires sm_90+ (Hopper/Blackwell). "
              "Detected compute capability ", prop.major, ".", prop.minor, ".");

  // Dynamic smem layout (per CTA):
  //   exports : topk_val * (float + int32) = topk_val * 8 B  (DSMEM-visible)
  //   overlay : used by Stage 1 as the remap/bin cache (all ranks), reused
  //             by Stage 2 on rank 0 as the N*K merge pool. Sized to the
  //             larger of the two so either fits.
  //     cache_bytes = chunk_size * (float + uint8), uint8 region padded.
  //     merge_bytes = num_chunks * topk_val * (float + int32).
  const size_t export_bytes = static_cast<size_t>(topk_val) *
                              (sizeof(float) + sizeof(int32_t));
  const size_t cache_bytes  = static_cast<size_t>(chunk_size) * sizeof(float) +
                              ((static_cast<size_t>(chunk_size) + 15) & ~size_t(15));
  const size_t merge_bytes  = static_cast<size_t>(num_chunks) *
                              static_cast<size_t>(topk_val) *
                              (sizeof(float) + sizeof(int32_t));
  const size_t overlay_bytes = (cache_bytes > merge_bytes) ? cache_bytes : merge_bytes;
  const size_t smem_bytes    = export_bytes + overlay_bytes;
  TORCH_CHECK(smem_bytes <= kMaxDynSmem,
              "fast_cluster_topk_merge: smem ", smem_bytes,
              " > ceiling ", kMaxDynSmem,
              " (topk_val=", topk_val, ", num_chunks=", num_chunks,
              ", chunk_size=", chunk_size, ")");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const float mp = static_cast<float>(mapping_power);

  const dim3 grid(static_cast<unsigned>(batch_size * num_chunks), 1, 1);
  const dim3 block(kThreadsPerBlock, 1, 1);

  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = static_cast<unsigned>(num_chunks);
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;

  cudaLaunchConfig_t cfg{};
  cfg.gridDim          = grid;
  cfg.blockDim         = block;
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream           = stream;
  cfg.attrs            = attrs;
  cfg.numAttrs         = 1;

  #define LAUNCH(DTYPE, PTR_EXPR, MODE_VAL)                                  \
    do {                                                                     \
      setup_kernel_smem_once<TopK_Cluster_Kernel<DTYPE, MODE_VAL>,           \
                             kMaxDynSmem>();                                 \
      const auto rc_launch = ::cudaLaunchKernelEx(                           \
          &cfg, TopK_Cluster_Kernel<DTYPE, MODE_VAL>,                        \
          PTR_EXPR,                                                          \
          global_topk_indices.data_ptr<int32_t>(),                           \
          static_cast<int>(num_chunks),                                      \
          static_cast<int>(chunk_size),                                      \
          static_cast<int>(topk_val),                                        \
          mp);                                                               \
      TORCH_CHECK(rc_launch == cudaSuccess,                                  \
                  "fast_cluster_topk_merge launch failed: ",                 \
                  ::cudaGetErrorString(rc_launch));                          \
    } while (0)

  #define DISPATCH_MODE(DTYPE, PTR_EXPR)                                     \
    do {                                                                     \
      switch (mapping_mode) {                                                \
        case MAPPING_NONE:         LAUNCH(DTYPE, PTR_EXPR, MAPPING_NONE);         break; \
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
        default: TORCH_CHECK(false, "unreachable mode");                     \
      }                                                                      \
    } while (0)

  if (score.scalar_type() == at::ScalarType::BFloat16) {
    DISPATCH_MODE(__nv_bfloat16,
                  reinterpret_cast<__nv_bfloat16*>(score.data_ptr<at::BFloat16>()));
  } else if (score.scalar_type() == at::ScalarType::Float) {
    DISPATCH_MODE(float, score.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "fast_cluster_topk_merge: unsupported dtype ",
                score.scalar_type());
  }

  #undef DISPATCH_MODE
  #undef LAUNCH

  const auto rc = cudaGetLastError();
  TORCH_CHECK(rc == cudaSuccess,
              "fast_cluster_topk_merge kernel failed: ", ::cudaGetErrorString(rc));
}
