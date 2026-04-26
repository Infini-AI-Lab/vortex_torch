/**
 * Profile-only fixtures for the adaptive split TopK.  Two distinct fixtures:
 *
 * [1] LEGACY split-2 histogram fixture (TopK_Phase1_Only_Kernel,
 *     TopK_Phase2_Only_Kernel, topk_adaptive_phase1_only,
 *     topk_adaptive_phase2_only).
 *     Implements an 8-bit coarse histogram + 8-bit refinement on bits [23:16]
 *     with a fixed split count of 2 (kNumSplits=2, kThreads=1024).
 *     Kept for historical comparison only; this is NOT the K=30 production
 *     path and is NOT representative of the current split kernel in
 *     topk_sglang_merge.cu.  barrier = full_adaptive - (phase1 + phase2).
 *
 * [2] K=30 ablation fixture (Ablation_*_Kernel,
 *     topk_output_adaptive_workspace_ablation).
 *     Measures isolated costs: local sort, workspace write, atomic/fence,
 *     merge-only (multiple variants), memset-only, and the full adaptive path.
 *     Split configs exactly match production (kAblCfg1..kAblCfg32).
 *     ScoreT is bf16 only; mode is hardcoded to MAPPING_NONE.
 *     Production kernel lives in topk_sglang_merge.cu.
 *     Current production merge = MERGE_CUB_WARP (kAblMode_MergeCubWarp = 6).
 *     MERGE_PROD_DEFAULT (mode 5) is the legacy per-SPLITS dispatch kept for
 *     ablation comparison (not the current production merge).
 *
 * ablation_mode constants and merge variant constants are defined below in
 * the K=30 ablation namespace section.
 */

 #include <ATen/core/TensorBase.h>
 #include <ATen/core/TensorBody.h>
 #include <ATen/cuda/CUDAContext.h>
 #include <c10/cuda/CUDAStream.h>
 #include <c10/util/Exception.h>
 #include <cuda.h>
 #include <cuda_bf16.h>
 #include <cuda_fp16.h>
 #include <math_constants.h>
 
 #include <cub/block/block_radix_sort.cuh>
 #include <cub/warp/warp_merge_sort.cuh>
 #include <cub/block/block_merge_sort.cuh>
 
 #include <cstdint>
 #include <type_traits>
 
 #include "register.h"
 
 namespace {
 
 constexpr int    kRadix         = 256;
 constexpr int    kThreads       = 1024;
 constexpr int    kWarpSize      = 32;
 constexpr int    kNumSplits     = 2;
 constexpr size_t kMaxDynSmem    = 96 * 1024;
 
 __device__ __forceinline__ uint32_t convert_to_uint32(float x) {
   uint32_t bits = __float_as_uint(x);
   return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
 }
 __device__ __forceinline__ uint8_t convert_to_uint8(float x) {
   __half h = __float2half_rn(x);
   uint16_t bits = __half_as_ushort(h);
   uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                  : static_cast<uint16_t>(bits | 0x8000);
   return static_cast<uint8_t>(key >> 8);
 }
 
 __device__ __forceinline__ void run_cumsum_256(int s_hist[2][kRadix + 128]) {
   const int tx = threadIdx.x;
 #pragma unroll 8
   for (int i = 0; i < 8; ++i) {
     if (tx < kRadix) {
       const int j = 1 << i;
       const int k = i & 1;
       int v = s_hist[k][tx];
       if (tx < kRadix - j) v += s_hist[k][tx + j];
       s_hist[k ^ 1][tx] = v;
     }
     __syncthreads();
   }
 }
 
 __device__ __forceinline__ int warp_compact_slot(bool selected, int* s_counter) {
   const uint32_t mask = __activemask();
   const uint32_t ballot = __ballot_sync(mask, selected);
   const int lane = threadIdx.x & (kWarpSize - 1);
   const int warp_count = __popc(ballot);
   const int rank = __popc(ballot & ((1u << lane) - 1u));
   const int first = __ffs(mask) - 1;
   int base = 0;
   if (lane == first) {
     base = (warp_count > 0) ? ::atomicAdd(s_counter, warp_count) : 0;
   }
   base = __shfl_sync(mask, base, first);
   return selected ? (base + rank) : -1;
 }
 
 // ============================================================================
 // Phase 1 ONLY: per-chunk radix select, writes unordered (score, idx) pairs
 // into partial_scores/partial_indices. No barrier, no merge.
 // ============================================================================
 __global__ __launch_bounds__(kThreads)
 void TopK_Phase1_Only_Kernel(
     const __nv_bfloat16* __restrict__ score,
     const int*           __restrict__ dense_kv_indptr,
     const int*           __restrict__ dense_kv_indices,
     float*               __restrict__ partial_scores,
     int32_t*             __restrict__ partial_indices,
     const int            topk_val,
     const int            reserved_bos,
     const int            reserved_eos)
 {
   const int b  = blockIdx.x;
   const int n  = blockIdx.y;
   const int tx = threadIdx.x;
 
   const int row_start = dense_kv_indptr[b] + reserved_bos;
   const int row_end   = dense_kv_indptr[b + 1] - reserved_eos;
   const int row_len   = row_end - row_start;
   const int half      = (row_len + 1) / 2;
   const int ck_begin  = (n == 0) ? 0    : half;
   const int ck_end    = (n == 0) ? half : row_len;
   const int ck_len    = ck_end - ck_begin;
 
   const __nv_bfloat16* chunk_in = score            + row_start + ck_begin;
   const int*           idx_map  = dense_kv_indices + row_start + ck_begin;
   float*   part_keys = partial_scores  + (static_cast<int64_t>(b) * kNumSplits + n) * topk_val;
   int32_t* part_idx  = partial_indices + (static_cast<int64_t>(b) * kNumSplits + n) * topk_val;
 
   extern __shared__ char smem_raw[];
   alignas(128) __shared__ int s_hist_buf[2][kRadix + 128];
   alignas(128) __shared__ int s_counter;
   alignas(128) __shared__ int s_threshold_bin;
   alignas(128) __shared__ int s_sub_threshold_bin;
   alignas(128) __shared__ int s_last_remain;
   auto& s_hist = s_hist_buf[0];
 
   float*   s_remapped = reinterpret_cast<float*>(smem_raw);
   uint8_t* s_bins     = reinterpret_cast<uint8_t*>(s_remapped + ck_len);
 
   if (ck_len <= topk_val) {
     for (int i = tx; i < topk_val; i += blockDim.x) {
       if (i < ck_len) {
         part_keys[i] = __bfloat162float(chunk_in[i]);
         part_idx [i] = idx_map[i];
       } else {
         part_keys[i] = -CUDART_INF_F;
         part_idx [i] = -1;
       }
     }
     return;
   }
 
   if (tx < kRadix + 1) s_hist[tx] = 0;
   if (tx == 0) {
     s_counter = 0;
     s_threshold_bin = -1;
     s_sub_threshold_bin = -1;
     s_last_remain = 0;
   }
   __syncthreads();
 
   for (int i = tx; i < ck_len; i += blockDim.x) {
     const float v = __bfloat162float(chunk_in[i]);
     const uint8_t bin = convert_to_uint8(v);
     s_remapped[i] = v;
     s_bins    [i] = bin;
     ::atomicAdd(&s_hist[bin], 1);
   }
   __syncthreads();
   run_cumsum_256(s_hist_buf);
 
   if (tx < kRadix && s_hist[tx] > topk_val && s_hist[tx + 1] <= topk_val) {
     s_threshold_bin = tx;
     s_last_remain   = topk_val - s_hist[tx + 1];
   }
   __syncthreads();
   const int threshold_bin = s_threshold_bin;
 
   if (tx < kRadix + 1) s_hist[tx] = 0;
   __syncthreads();
 
   const int num_iters = (ck_len + blockDim.x - 1) / blockDim.x;
   for (int it = 0; it < num_iters; ++it) {
     const int i = it * blockDim.x + tx;
     const bool in_range = (i < ck_len);
     int bin = -1;
     if (in_range) bin = s_bins[i];
     const bool take_above = in_range && (bin > threshold_bin);
     const int slot = warp_compact_slot(take_above, &s_counter);
     if (take_above) {
       part_keys[slot] = s_remapped[i];
       part_idx [slot] = idx_map[i];
     } else if (in_range && bin == threshold_bin) {
       const uint32_t b32 = convert_to_uint32(s_remapped[i]);
       const int sub_bin = (b32 >> 16) & 0xFF;
       ::atomicAdd(&s_hist[sub_bin], 1);
     }
   }
   __syncthreads();
   run_cumsum_256(s_hist_buf);
   if (tx < kRadix && s_hist[tx] > s_last_remain
                   && s_hist[tx + 1] <= s_last_remain) {
     s_sub_threshold_bin = tx;
     s_last_remain = s_last_remain - s_hist[tx + 1];
   }
   if (tx == 0 && s_sub_threshold_bin == -1) s_sub_threshold_bin = kRadix;
   __syncthreads();
   const int sub_threshold_bin = s_sub_threshold_bin;
 
   for (int it = 0; it < num_iters; ++it) {
     const int i = it * blockDim.x + tx;
     const bool in_range = (i < ck_len);
     int bin = -1;
     if (in_range) bin = s_bins[i];
     int sub_bin = -1;
     if (in_range && bin == threshold_bin) {
       const uint32_t b32 = convert_to_uint32(s_remapped[i]);
       sub_bin = (b32 >> 16) & 0xFF;
     }
     const bool take_sub = (sub_bin > sub_threshold_bin);
     const int slot = warp_compact_slot(take_sub, &s_counter);
     if (take_sub) {
       part_keys[slot] = s_remapped[i];
       part_idx [slot] = idx_map[i];
     } else if (sub_bin == sub_threshold_bin) {
       const int pos = ::atomicAdd(&s_last_remain, -1);
       if (pos > 0) {
         part_keys[topk_val - pos] = s_remapped[i];
         part_idx [topk_val - pos] = idx_map[i];
       }
     }
   }
 }
 
 // ============================================================================
 // Phase 2 ONLY: read pre-populated (kNumSplits * K) candidates, radix-select
 // top-K, write final indices. Grid = (batch,).
 // ============================================================================
 __global__ __launch_bounds__(kThreads)
 void TopK_Phase2_Only_Kernel(
     const float*   __restrict__ partial_scores,
     const int32_t* __restrict__ partial_indices,
     const int*     __restrict__ sparse_kv_indptr,
     int32_t*       __restrict__ sparse_kv_indices,
     const int      topk_val,
     const int      reserved_bos)
 {
   const int b  = blockIdx.x;
   const int tx = threadIdx.x;
   const int total = kNumSplits * topk_val;
 
   const float*   keys_in = partial_scores  + static_cast<int64_t>(b) * total;
   const int32_t* idx_in  = partial_indices + static_cast<int64_t>(b) * total;
   int32_t*       out_idx = sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos;
 
   extern __shared__ char smem_raw[];
   alignas(128) __shared__ int s_hist_buf[2][kRadix + 128];
   alignas(128) __shared__ int s_counter;
   alignas(128) __shared__ int s_threshold_bin;
   alignas(128) __shared__ int s_sub_threshold_bin;
   alignas(128) __shared__ int s_last_remain;
   auto& s_hist = s_hist_buf[0];
 
   float*   s_scores  = reinterpret_cast<float*>(smem_raw);
   int32_t* s_indices = reinterpret_cast<int32_t*>(s_scores + total);
 
   if ((total & 3) == 0) {
     const float4* kv = reinterpret_cast<const float4*>(keys_in);
     const int4*   iv = reinterpret_cast<const int4*>  (idx_in);
     float4* sv = reinterpret_cast<float4*>(s_scores);
     int4*   iiv = reinterpret_cast<int4*>  (s_indices);
     const int total4 = total >> 2;
     for (int i = tx; i < total4; i += blockDim.x) {
       sv[i]  = kv[i];
       iiv[i] = iv[i];
     }
   } else {
     for (int i = tx; i < total; i += blockDim.x) {
       s_scores[i]  = keys_in[i];
       s_indices[i] = idx_in[i];
     }
   }
 
   if (tx < kRadix + 1) s_hist[tx] = 0;
   if (tx == 0) {
     s_counter = 0;
     s_threshold_bin = -1;
     s_sub_threshold_bin = -1;
     s_last_remain = 0;
   }
   __syncthreads();
 
   const int num_iters = (total + blockDim.x - 1) / blockDim.x;
   for (int it = 0; it < num_iters; ++it) {
     const int i = it * blockDim.x + tx;
     if (i < total && s_indices[i] >= 0) {
       const uint32_t b32 = convert_to_uint32(s_scores[i]);
       const int bin = (b32 >> 24) & 0xFF;
       ::atomicAdd(&s_hist[bin], 1);
     }
   }
   __syncthreads();
   run_cumsum_256(s_hist_buf);
 
   const int valid_count = s_hist[0];
   if (valid_count <= topk_val) {
     for (int it = 0; it < num_iters; ++it) {
       const int i = it * blockDim.x + tx;
       const bool take = (i < total) && (s_indices[i] >= 0);
       const int slot = warp_compact_slot(take, &s_counter);
       if (take && slot < topk_val) out_idx[slot] = s_indices[i];
     }
     return;
   }
 
   if (tx < kRadix && s_hist[tx] > topk_val && s_hist[tx + 1] <= topk_val) {
     s_threshold_bin = tx;
     s_last_remain   = topk_val - s_hist[tx + 1];
   }
   __syncthreads();
   const int threshold_bin_m = s_threshold_bin;
 
   if (tx < kRadix + 1) s_hist[tx] = 0;
   __syncthreads();
 
   for (int it = 0; it < num_iters; ++it) {
     const int i = it * blockDim.x + tx;
     bool in_valid = false;
     int bin = -1;
     uint32_t b32 = 0;
     if (i < total) {
       const int32_t ii = s_indices[i];
       if (ii >= 0) {
         in_valid = true;
         b32 = convert_to_uint32(s_scores[i]);
         bin = (b32 >> 24) & 0xFF;
       }
     }
     const bool take_above = in_valid && (bin > threshold_bin_m);
     const int slot = warp_compact_slot(take_above, &s_counter);
     if (take_above) {
       out_idx[slot] = s_indices[i];
     } else if (in_valid && bin == threshold_bin_m) {
       const int sub_bin = (b32 >> 16) & 0xFF;
       ::atomicAdd(&s_hist[sub_bin], 1);
     }
   }
   __syncthreads();
 
   run_cumsum_256(s_hist_buf);
   if (tx < kRadix && s_hist[tx] > s_last_remain
                   && s_hist[tx + 1] <= s_last_remain) {
     s_sub_threshold_bin = tx;
     s_last_remain = s_last_remain - s_hist[tx + 1];
   }
   if (tx == 0 && s_sub_threshold_bin == -1) s_sub_threshold_bin = kRadix;
   __syncthreads();
   const int sub_threshold_bin_m = s_sub_threshold_bin;
 
   for (int it = 0; it < num_iters; ++it) {
     const int i = it * blockDim.x + tx;
     bool in_thr = false;
     int sub_bin = -1;
     if (i < total) {
       const int32_t ii = s_indices[i];
       if (ii >= 0) {
         const uint32_t b32 = convert_to_uint32(s_scores[i]);
         const int bin = (b32 >> 24) & 0xFF;
         if (bin == threshold_bin_m) {
           in_thr = true;
           sub_bin = (b32 >> 16) & 0xFF;
         }
       }
     }
     const bool take = in_thr && (sub_bin > sub_threshold_bin_m);
     const int slot = warp_compact_slot(take, &s_counter);
     if (take) {
       out_idx[slot] = s_indices[i];
     } else if (in_thr && sub_bin == sub_threshold_bin_m) {
       const int pos = ::atomicAdd(&s_last_remain, -1);
       if (pos > 0) out_idx[topk_val - pos] = s_indices[i];
     }
   }
 }
 
 template <auto* f, size_t max_dyn>
 void setup_smem_once() {
   [[maybe_unused]] static const auto r = [] {
     return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
   }();
   TORCH_CHECK(r == cudaSuccess, "profile kernel setup failed: ",
               ::cudaGetErrorString(r));
 }
 
 }  // namespace
 
 #define CHECK_CUDA_T(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
 
 // ============================================================================
 // Host entry — Phase 1 only.
 // ============================================================================
 void topk_adaptive_phase1_only(
     const at::Tensor& x,
     const at::Tensor& dense_kv_indptr,
     const at::Tensor& dense_kv_indices,
     at::Tensor&       partial_scores,
     at::Tensor&       partial_indices,
     const int64_t     eff_batch_size,
     const int64_t     topk_val,
     const int64_t     reserved_bos,
     const int64_t     reserved_eos,
     const int64_t     max_num_pages)
 {
   CHECK_CUDA_T(x); CHECK_CUDA_T(dense_kv_indptr);
   CHECK_CUDA_T(dense_kv_indices); CHECK_CUDA_T(partial_scores); CHECK_CUDA_T(partial_indices);
   TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
               "profile kernels require bfloat16 input");
 
   const int chunk_max = (static_cast<int>(max_num_pages) + 1) / 2;
   const size_t smem = static_cast<size_t>(chunk_max) * sizeof(float)
                     + ((static_cast<size_t>(chunk_max) + 15) & ~size_t(15));
   TORCH_CHECK(smem <= kMaxDynSmem, "phase1 smem too large");
 
   setup_smem_once<TopK_Phase1_Only_Kernel, kMaxDynSmem>();
   cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
   dim3 grid(static_cast<unsigned>(eff_batch_size),
             static_cast<unsigned>(kNumSplits));
   TopK_Phase1_Only_Kernel<<<grid, dim3(kThreads), smem, stream>>>(
       reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
       dense_kv_indptr.data_ptr<int32_t>(),
       dense_kv_indices.data_ptr<int32_t>(),
       partial_scores.data_ptr<float>(),
       partial_indices.data_ptr<int32_t>(),
       static_cast<int>(topk_val),
       static_cast<int>(reserved_bos),
       static_cast<int>(reserved_eos));
   TORCH_CHECK(cudaGetLastError() == cudaSuccess, "phase1 launch failed");
 }
 
 // ============================================================================
 // Host entry — Phase 2 only (expects partial_* pre-populated).
 // ============================================================================
 void topk_adaptive_phase2_only(
     const at::Tensor& partial_scores,
     const at::Tensor& partial_indices,
     const at::Tensor& sparse_kv_indptr,
     at::Tensor&       sparse_kv_indices,
     const int64_t     eff_batch_size,
     const int64_t     topk_val,
     const int64_t     reserved_bos)
 {
   CHECK_CUDA_T(partial_scores); CHECK_CUDA_T(partial_indices);
   CHECK_CUDA_T(sparse_kv_indptr); CHECK_CUDA_T(sparse_kv_indices);
 
   const size_t smem = static_cast<size_t>(kNumSplits) *
                       static_cast<size_t>(topk_val) *
                       (sizeof(float) + sizeof(int32_t));
   TORCH_CHECK(smem <= kMaxDynSmem, "phase2 smem too large");
 
   setup_smem_once<TopK_Phase2_Only_Kernel, kMaxDynSmem>();
   cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
   dim3 grid(static_cast<unsigned>(eff_batch_size));
   TopK_Phase2_Only_Kernel<<<grid, dim3(kThreads), smem, stream>>>(
       partial_scores.data_ptr<float>(),
       partial_indices.data_ptr<int32_t>(),
       sparse_kv_indptr.data_ptr<int32_t>(),
       sparse_kv_indices.data_ptr<int32_t>(),
       static_cast<int>(topk_val),
       static_cast<int>(reserved_bos));
   TORCH_CHECK(cudaGetLastError() == cudaSuccess, "phase2 launch failed");
 }
 
 // =============================================================================
 // K=30 phase ablation kernels and host entry. Bench-only fixture for
 // `bench_ablation.py`. NOT a production code path. The production K=30
 // random-split parallel kernel and dispatcher live in topk_sglang_merge.cu.
 //
 // All kernels here are hardcoded to ScoreT=bf16, MAPPING_NONE, partition
 // = CONTIGUOUS to keep the template instantiation count small. They share
 // no code with the production path beyond the function declarations in
 // register.h; helpers are duplicated below in the anonymous namespace.
 // =============================================================================
 namespace {
 
 constexpr int kLocalK_Top30    = 32;
 constexpr int kMaxFinalK_Top30 = 32;
 constexpr int kPartContiguous  = 1;  // PART_CONTIGUOUS in topk_sglang_merge.cu
 
 // Per-split (NUM_THREADS, ITEMS_PER_THREAD) — must match the production
 // configurations in topk_sglang_merge.cu (kCfg1..kCfg32) so the ablation
 // numbers reflect the production launch parameters.
 struct AblSplitCfg { int num_threads, items_per_thread; };
 constexpr AblSplitCfg kAblCfg1  = { 1024,  8 };
 constexpr AblSplitCfg kAblCfg2  = { 1024,  8 };
 constexpr AblSplitCfg kAblCfg4  = {  512,  8 };
 constexpr AblSplitCfg kAblCfg8  = {  256, 16 };
 constexpr AblSplitCfg kAblCfg16 = {  128, 16 };
 constexpr AblSplitCfg kAblCfg32 = {   64, 16 };

 // ---------------------------------------------------------------------------
 // Ablation mode constants (ablation_mode argument to
 // topk_output_adaptive_workspace_ablation).
 // ---------------------------------------------------------------------------
 constexpr int kAblMode_FullAdaptive       = 0;  // full production path (reference)
 constexpr int kAblMode_LocalWithWorkspace = 1;  // local sort + workspace write, no merge
 constexpr int kAblMode_LocalNoWorkspace   = 2;  // local sort only, no write, no merge
 constexpr int kAblMode_WorkspaceWriteOnly = 3;  // synthetic write to workspace
 constexpr int kAblMode_AtomicOnly         = 4;  // atomic counter cost only
 constexpr int kAblMode_MergeProdDefault   = 5;  // merge: legacy per-SPLITS dispatch
                                                 //   (2-way for SPLITS=2, pairwise for SPLITS=4,
                                                 //    k-way for SPLITS>=8). NOT current production.
 constexpr int kAblMode_MergeCubWarp       = 6;  // merge: cub::WarpMergeSort — current production
 constexpr int kAblMode_MergeCubBlock      = 7;  // merge: cub::BlockMergeSort benchmark
 constexpr int kAblMode_MemsetOnly         = 8;  // counter memset cost only
 constexpr int kAblMode_MergeManual2Way    = 9;  // merge: manual 2-way (requires split=2)
 constexpr int kAblMode_MergePairwise4     = 10; // merge: pairwise tree  (requires split=4)
 constexpr int kAblMode_MergeKwayAll       = 11; // merge: force k-way for all split counts

 // ---------------------------------------------------------------------------
 // Merge variant indices for Ablation_MergeOnly_Kernel<SPLITS, MERGE_VARIANT>.
 // ---------------------------------------------------------------------------
 constexpr int MERGE_PROD_DEFAULT = 0; // legacy: 2-way(SPLITS=2)/pairwise(SPLITS=4)/k-way(>=8)
 constexpr int MERGE_CUB_WARP     = 1; // cub::WarpMergeSort  — matches current production merge
                                       //   kMergeIPT=SPLITS; register pressure grows with SPLITS
 constexpr int MERGE_CUB_BLOCK    = 2; // cub::BlockMergeSort (benchmark; 64 threads)
 constexpr int MERGE_MANUAL_2WAY  = 3; // manual 2-way merge  (requires SPLITS=2)
 constexpr int MERGE_PAIRWISE_4   = 4; // pairwise tree       (requires SPLITS=4)
 constexpr int MERGE_KWAY         = 5; // force k-way for all SPLITS (explicit baseline)

 template <typename T>
 __device__ __forceinline__ float vortex_to_float_p(T x);
 template <>
 __device__ __forceinline__ float vortex_to_float_p<float>(float x) { return x; }
 template <>
 __device__ __forceinline__ float vortex_to_float_p<__nv_bfloat16>(__nv_bfloat16 x) {
   return __bfloat162float(x);
 }
 
 struct AblGreaterUint32 {
   __device__ __forceinline__ bool operator()(uint32_t a, uint32_t b) const {
     return a > b;
   }
 };
 
 // k-way merge — same algorithm as topk_sglang_merge.cu's merge_sorted_kway.
 template <int SPLITS, int LOCAL_K, int MAX_FINAL_K>
 __device__ __forceinline__ void abl_merge_sorted_kway(
     const uint32_t* __restrict__ keys_in,
     const int32_t*  __restrict__ idx_in,
     int32_t*        __restrict__ out_idx,
     int             final_k)
 {
   const int  lane       = threadIdx.x & 31;
   const bool is_my_list = (lane < SPLITS);
   const uint32_t full   = 0xFFFFFFFFu;
 
   int      ptr     = 0;
   uint32_t cur_key = is_my_list ? keys_in[lane * LOCAL_K] : 0u;
   int32_t  cur_idx = is_my_list ? idx_in [lane * LOCAL_K] : -1;
 
   #pragma unroll
   for (int t = 0; t < MAX_FINAL_K; ++t) {
     uint32_t best_key  = cur_key;
     int      best_lane = lane;
     #pragma unroll
     for (int offset = 16; offset > 0; offset >>= 1) {
       uint32_t okey  = __shfl_xor_sync(full, best_key,  offset);
       int      olane = __shfl_xor_sync(full, best_lane, offset);
       bool take = (okey > best_key) || (okey == best_key && olane < best_lane);
       best_key  = take ? okey  : best_key;
       best_lane = take ? olane : best_lane;
     }
     int32_t win_idx = __shfl_sync(full, cur_idx, best_lane);
     if (lane == 0 && t < final_k && win_idx >= 0) out_idx[t] = win_idx;
     if (lane == best_lane) {
       ++ptr;
       if (is_my_list && ptr < LOCAL_K) {
         cur_key = keys_in[lane * LOCAL_K + ptr];
         cur_idx = idx_in [lane * LOCAL_K + ptr];
       } else {
         cur_key = 0u;
         cur_idx = -1;
       }
     }
   }
 }
 
 __device__ __forceinline__ void abl_merge_2way_manual_lane0(
     const uint32_t* __restrict__ l0_keys, const int32_t* __restrict__ l0_idx, int n0,
     const uint32_t* __restrict__ l1_keys, const int32_t* __restrict__ l1_idx, int n1,
     int32_t*        __restrict__ out_idx,
     int             final_k)
 {
   if (threadIdx.x != 0) return;
   int p0 = 0, p1 = 0;
   for (int t = 0; t < final_k; ++t) {
     const uint32_t k0 = (p0 < n0) ? l0_keys[p0] : 0u;
     const uint32_t k1 = (p1 < n1) ? l1_keys[p1] : 0u;
     if (k0 >= k1 && p0 < n0) {
       out_idx[t] = l0_idx[p0]; ++p0;
     } else if (p1 < n1) {
       out_idx[t] = l1_idx[p1]; ++p1;
     } else {
       out_idx[t] = -1;
     }
   }
 }
 
 __device__ __forceinline__ void abl_merge_pairwise_4(
     const uint32_t* __restrict__ keys_in,
     const int32_t*  __restrict__ idx_in,
     int32_t*        __restrict__ out_idx, int final_k,
     uint32_t* __restrict__ tmp01_keys, int32_t* __restrict__ tmp01_idx,
     uint32_t* __restrict__ tmp23_keys, int32_t* __restrict__ tmp23_idx)
 {
   constexpr int LK = kLocalK_Top30;
   const int lane = threadIdx.x & 31;
   if (lane == 0) {
     int p0 = 0, p1 = 0;
     #pragma unroll
     for (int t = 0; t < LK; ++t) {
       const uint32_t k0 = (p0 < LK) ? keys_in[0 * LK + p0] : 0u;
       const uint32_t k1 = (p1 < LK) ? keys_in[1 * LK + p1] : 0u;
       if (k0 >= k1 && p0 < LK) { tmp01_keys[t] = k0; tmp01_idx[t] = idx_in[0*LK+p0]; ++p0; }
       else if (p1 < LK)        { tmp01_keys[t] = k1; tmp01_idx[t] = idx_in[1*LK+p1]; ++p1; }
       else                     { tmp01_keys[t] = 0u; tmp01_idx[t] = -1; }
     }
   } else if (lane == 1) {
     int p2 = 0, p3 = 0;
     #pragma unroll
     for (int t = 0; t < LK; ++t) {
       const uint32_t k2 = (p2 < LK) ? keys_in[2 * LK + p2] : 0u;
       const uint32_t k3 = (p3 < LK) ? keys_in[3 * LK + p3] : 0u;
       if (k2 >= k3 && p2 < LK) { tmp23_keys[t] = k2; tmp23_idx[t] = idx_in[2*LK+p2]; ++p2; }
       else if (p3 < LK)        { tmp23_keys[t] = k3; tmp23_idx[t] = idx_in[3*LK+p3]; ++p3; }
       else                     { tmp23_keys[t] = 0u; tmp23_idx[t] = -1; }
     }
   }
   __syncwarp();
   if (lane == 0) {
     int p0 = 0, p1 = 0;
     for (int t = 0; t < final_k; ++t) {
       const uint32_t k0 = (p0 < LK) ? tmp01_keys[p0] : 0u;
       const uint32_t k1 = (p1 < LK) ? tmp23_keys[p1] : 0u;
       if (k0 >= k1 && p0 < LK) { out_idx[t] = tmp01_idx[p0]; ++p0; }
       else if (p1 < LK)        { out_idx[t] = tmp23_idx[p1]; ++p1; }
       else                     { out_idx[t] = -1; }
     }
   }
 }
 
 // uint32 sort key for an fp32 value.
 __device__ __forceinline__ uint32_t abl_to_uint32(float x) {
   uint32_t bits = __float_as_uint(x);
   return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
 }
 
 // ---- Ablation kernels --------------------------------------------------------
 
 template <int SPLITS, int NUM_THREADS, int ITEMS_PER_THREAD>
 __global__ __launch_bounds__(NUM_THREADS)
 void Ablation_LocalOnly_Kernel(
     const __nv_bfloat16* __restrict__ score,
     const int*    __restrict__ dense_kv_indptr,
     const int*    __restrict__ dense_kv_indices,
     uint32_t*     __restrict__ partial_keys,
     int32_t*      __restrict__ partial_indices,
     const int     reserved_bos,
     const int     reserved_eos)
 {
   // Stage 1 + workspace write. No atomic. No merge.
   using KeyT = uint32_t; using ValueT = int32_t;
   using BlockSortT = cub::BlockRadixSort<KeyT, NUM_THREADS, ITEMS_PER_THREAD, ValueT>;
   __shared__ typename BlockSortT::TempStorage sort_smem;
   const int b  = blockIdx.x;
   const int n  = blockIdx.y;
   const int tx = threadIdx.x;
   const int row_start = dense_kv_indptr[b] + reserved_bos;
   const int row_end   = dense_kv_indptr[b + 1] - reserved_eos;
   const int row_len   = max(0, row_end - row_start);
   if (row_len <= 0) return;
   const int group_begin = (row_len * n) / SPLITS;
   const int group_end   = (row_len * (n + 1)) / SPLITS;
   const int group_len   = group_end - group_begin;
   const __nv_bfloat16* row_scores = score            + row_start;
   const int*           row_idxmap = dense_kv_indices + row_start;
   KeyT keys[ITEMS_PER_THREAD]; ValueT values[ITEMS_PER_THREAD];
   #pragma unroll
   for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
     const int local_rank = tx + k * NUM_THREADS;
     if (local_rank < group_len) {
       const int pos = group_begin + local_rank;
       const float raw = vortex_to_float_p(row_scores[pos]);
       keys  [k] = abl_to_uint32(raw);
       values[k] = row_idxmap[pos];
     } else { keys[k] = 0u; values[k] = -1; }
   }
   BlockSortT(sort_smem).SortDescending(keys, values);
   __syncthreads();
   constexpr int LK = kLocalK_Top30;
   const int64_t part_off = (static_cast<int64_t>(b) * SPLITS + n) * LK;
   uint32_t* part_keys = partial_keys + part_off;
   int32_t*  part_idx  = partial_indices + part_off;
   #pragma unroll
   for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
     const int rank = tx * ITEMS_PER_THREAD + k;
     if (rank < LK) { part_keys[rank] = keys[k]; part_idx[rank] = values[k]; }
   }
 }
 
 template <int SPLITS, int NUM_THREADS, int ITEMS_PER_THREAD>
 __global__ __launch_bounds__(NUM_THREADS)
 void Ablation_LocalNoWorkspace_Kernel(
     const __nv_bfloat16* __restrict__ score,
     const int*    __restrict__ dense_kv_indptr,
     const int*    __restrict__ dense_kv_indices,
     int32_t*      __restrict__ scratch,
     const int     reserved_bos,
     const int     reserved_eos)
 {
   using KeyT = uint32_t; using ValueT = int32_t;
   using BlockSortT = cub::BlockRadixSort<KeyT, NUM_THREADS, ITEMS_PER_THREAD, ValueT>;
   __shared__ typename BlockSortT::TempStorage sort_smem;
   const int b  = blockIdx.x;
   const int n  = blockIdx.y;
   const int tx = threadIdx.x;
   const int row_start = dense_kv_indptr[b] + reserved_bos;
   const int row_end   = dense_kv_indptr[b + 1] - reserved_eos;
   const int row_len   = max(0, row_end - row_start);
   if (row_len <= 0) return;
   const int group_begin = (row_len * n) / SPLITS;
   const int group_end   = (row_len * (n + 1)) / SPLITS;
   const int group_len   = group_end - group_begin;
   const __nv_bfloat16* row_scores = score            + row_start;
   const int*           row_idxmap = dense_kv_indices + row_start;
   KeyT keys[ITEMS_PER_THREAD]; ValueT values[ITEMS_PER_THREAD];
   #pragma unroll
   for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
     const int local_rank = tx + k * NUM_THREADS;
     if (local_rank < group_len) {
       const int pos = group_begin + local_rank;
       const float raw = vortex_to_float_p(row_scores[pos]);
       keys  [k] = abl_to_uint32(raw);
       values[k] = row_idxmap[pos];
     } else { keys[k] = 0u; values[k] = -1; }
   }
   BlockSortT(sort_smem).SortDescending(keys, values);
   if (tx == 0) scratch[blockIdx.x * gridDim.y + blockIdx.y] = values[0];
 }
 
 template <int SPLITS>
 __global__ __launch_bounds__(32)
 void Ablation_WorkspaceWriteOnly_Kernel(
     uint32_t* __restrict__ partial_keys,
     int32_t*  __restrict__ partial_indices)
 {
   constexpr int LK = kLocalK_Top30;
   const int b   = blockIdx.x;
   const int n   = blockIdx.y;
   const int lane = threadIdx.x;
   const int64_t part_off = (static_cast<int64_t>(b) * SPLITS + n) * LK;
   if (lane < LK) {
     partial_keys   [part_off + lane] = static_cast<uint32_t>(b * 31 + n * 7 + lane);
     partial_indices[part_off + lane] = b * 1009 + n * 17 + lane;
   }
 }
 
 template <int SPLITS>
 __global__ __launch_bounds__(32)
 void Ablation_AtomicOnly_Kernel(
     int32_t* __restrict__ done_counter,
     int32_t* __restrict__ scratch)
 {
   const int b  = blockIdx.x;
   const int tx = threadIdx.x;
   __shared__ int s_is_last;
   __threadfence();
   __syncthreads();
   if (tx == 0) {
     const int old = ::atomicAdd(&done_counter[b], 1);
     s_is_last = (old == SPLITS - 1) ? 1 : 0;
   }
   __syncthreads();
   if (s_is_last && tx == 0) scratch[b] = 1;
 }
 
 // Correctness notes for Ablation_MergeOnly_Kernel:
 //   - MERGE_PROD_DEFAULT exactly mirrors topk_sglang_merge.cu Stage 2 tie
 //     preference: lower list index wins on equal key (k-way and pairwise);
 //     lane 0 favors list 0 on equal key (2-way manual).
 //   - CUB variants sort by uint32 key only.  Tie-breaking for duplicate keys
 //     is implementation-defined and will NOT match production index order.
 //     Use unique keys for exact index comparison in correctness tests.
 //   - For throughput benchmarking, duplicate keys are acceptable since only
 //     latency is measured.
 template <int SPLITS, int MERGE_VARIANT>
 __global__ __launch_bounds__(64)
 void Ablation_MergeOnly_Kernel(
     const uint32_t* __restrict__ partial_keys,
     const int32_t*  __restrict__ partial_indices,
     const int*      __restrict__ sparse_kv_indptr,
     int*            __restrict__ sparse_kv_indices,
     const int       topk_val,
     const int       reserved_bos)
 {
   constexpr int LK = kLocalK_Top30;
   constexpr int kCandidates = SPLITS * LK;
   const int b  = blockIdx.x;
   const int tx = threadIdx.x;
   const int64_t row_off = static_cast<int64_t>(b) * kCandidates;
   const uint32_t* keys_in = partial_keys    + row_off;
   const int32_t*  idx_in  = partial_indices + row_off;
   int32_t* out_idx = sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos;
 
   if constexpr (MERGE_VARIANT == MERGE_PROD_DEFAULT) {
     // Mirrors topk_sglang_merge.cu Stage 2 exactly: different strategy per SPLITS.
     if constexpr (SPLITS == 2) {
       if (tx < 32) abl_merge_2way_manual_lane0(
           keys_in,       idx_in,       LK,
           keys_in + LK,  idx_in + LK,  LK,
           out_idx, topk_val);
     } else if constexpr (SPLITS == 4) {
       __shared__ uint32_t s_pd01k[LK]; __shared__ int32_t s_pd01i[LK];
       __shared__ uint32_t s_pd23k[LK]; __shared__ int32_t s_pd23i[LK];
       if (tx < 32) abl_merge_pairwise_4(keys_in, idx_in, out_idx, topk_val,
                                         s_pd01k, s_pd01i, s_pd23k, s_pd23i);
     } else {
       if (tx < 32) abl_merge_sorted_kway<SPLITS, LK, kMaxFinalK_Top30>(
           keys_in, idx_in, out_idx, topk_val);
     }
   } else if constexpr (MERGE_VARIANT == MERGE_CUB_WARP) {
     // kMergeIPT grows with SPLITS: SPLITS=16 → kMergeIPT=16, SPLITS=32 → 32.
     // Large IPT increases register pressure and may cause spilling on sm_90+.
     constexpr int kMergeIPT = (kCandidates + 31) / 32;
     using WarpMergeT = cub::WarpMergeSort<uint32_t, kMergeIPT, 32, int32_t>;
     __shared__ typename WarpMergeT::TempStorage warp_merge;
     if (tx < 32) {
       uint32_t wkeys[kMergeIPT]; int32_t wvalues[kMergeIPT];
       #pragma unroll
       for (int k = 0; k < kMergeIPT; ++k) {
         const int rank = tx * kMergeIPT + k;
         wkeys  [k] = (rank < kCandidates) ? keys_in[rank] : 0u;
         wvalues[k] = (rank < kCandidates) ? idx_in [rank] : -1;
       }
       WarpMergeT(warp_merge).Sort(wkeys, wvalues, AblGreaterUint32{});
       #pragma unroll
       for (int k = 0; k < kMergeIPT; ++k) {
         const int rank = tx * kMergeIPT + k;
         if (rank < topk_val) out_idx[rank] = wvalues[k];
       }
     }
   } else if constexpr (MERGE_VARIANT == MERGE_CUB_BLOCK) {
     constexpr int kBlockThreads = 64;
     constexpr int kMergeIPT = (kCandidates + kBlockThreads - 1) / kBlockThreads;
     using BlockMergeT = cub::BlockMergeSort<uint32_t, kBlockThreads, kMergeIPT, int32_t>;
     __shared__ typename BlockMergeT::TempStorage block_merge;
     if (tx < kBlockThreads) {
       uint32_t wkeys[kMergeIPT]; int32_t wvalues[kMergeIPT];
       #pragma unroll
       for (int k = 0; k < kMergeIPT; ++k) {
         const int rank = tx * kMergeIPT + k;
         wkeys  [k] = (rank < kCandidates) ? keys_in[rank] : 0u;
         wvalues[k] = (rank < kCandidates) ? idx_in [rank] : -1;
       }
       BlockMergeT(block_merge).Sort(wkeys, wvalues, AblGreaterUint32{});
       #pragma unroll
       for (int k = 0; k < kMergeIPT; ++k) {
         const int rank = tx * kMergeIPT + k;
         if (rank < topk_val) out_idx[rank] = wvalues[k];
       }
     }
   } else if constexpr (MERGE_VARIANT == MERGE_MANUAL_2WAY) {
     static_assert(SPLITS == 2, "manual_2way merge requires SPLITS=2");
     if (tx < 32) abl_merge_2way_manual_lane0(
         keys_in,           idx_in,           LK,
         keys_in + LK,      idx_in + LK,      LK,
         out_idx, topk_val);
   } else if constexpr (MERGE_VARIANT == MERGE_PAIRWISE_4) {
     static_assert(SPLITS == 4, "pairwise_tree merge requires SPLITS=4");
     __shared__ uint32_t s_t01k[LK];
     __shared__ int32_t  s_t01i[LK];
     __shared__ uint32_t s_t23k[LK];
     __shared__ int32_t  s_t23i[LK];
     if (tx < 32) abl_merge_pairwise_4(
         keys_in, idx_in, out_idx, topk_val,
         s_t01k, s_t01i, s_t23k, s_t23i);
   } else if constexpr (MERGE_VARIANT == MERGE_KWAY) {
     // Force k-way for all SPLITS — explicit baseline to isolate k-way cost.
     if (tx < 32) abl_merge_sorted_kway<SPLITS, LK, kMaxFinalK_Top30>(
         keys_in, idx_in, out_idx, topk_val);
   }
 }
 
 }  // namespace
 
 void topk_output_adaptive_workspace_ablation(
     const at::Tensor& x,
     const at::Tensor& dense_kv_indptr,
     const at::Tensor& sparse_kv_indptr,
     const at::Tensor& dense_kv_indices,
     at::Tensor&       sparse_kv_indices,
     at::Tensor&       partial_keys,
     at::Tensor&       partial_indices,
     at::Tensor&       done_counter,
     at::Tensor&       scratch,
     const int64_t     eff_batch_size,
     const int64_t     topk_val,
     const int64_t     reserved_bos,
     const int64_t     reserved_eos,
     const int64_t     max_num_pages,
     const int64_t     ablation_mode,
     const int64_t     forced_splits)
 {
   TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
               "ablation kernels are bf16-only");
   TORCH_CHECK(topk_val > 0 && topk_val <= kMaxFinalK_Top30,
               "ablation kernels are K<=32 only");
 
   int split = forced_splits > 0 ? static_cast<int>(forced_splits) : 8;
   TORCH_CHECK(split == 1 || split == 2 || split == 4 || split == 8 ||
               split == 16 || split == 32,
               "forced_splits must be {1,2,4,8,16,32}, got ", split);
 
   cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
 
   // memset_only: just clear the counter.
   if (ablation_mode == 8) {
     if (split > 1) {
       ::cudaMemsetAsync(done_counter.data_ptr<int32_t>(), 0,
                         sizeof(int32_t) * static_cast<size_t>(eff_batch_size),
                         stream);
     }
     return;
   }
   // atomic_only needs the counter pre-cleared so each call sees a fresh state.
   if (ablation_mode == 4 && split > 1) {
     ::cudaMemsetAsync(done_counter.data_ptr<int32_t>(), 0,
                       sizeof(int32_t) * static_cast<size_t>(eff_batch_size),
                       stream);
   }
 
   uint32_t* part_keys_ptr = reinterpret_cast<uint32_t*>(partial_keys.data_ptr<int32_t>());
   int32_t*  part_idx_ptr  = partial_indices.data_ptr<int32_t>();
   int32_t*  done_ptr      = done_counter.data_ptr<int32_t>();
   int32_t*  scratch_ptr   = scratch.data_ptr<int32_t>();
 
   dim3 grid_full (static_cast<unsigned>(eff_batch_size),
                   static_cast<unsigned>(split));
   dim3 grid_merge(static_cast<unsigned>(eff_batch_size), 1u);
 
   const __nv_bfloat16* x_ptr =
       reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>());
 
   #define LAUNCH_ABL(KERNEL, GRID, NT, ...)                                       \
     do { KERNEL<<<GRID, dim3(NT), 0, stream>>>(__VA_ARGS__); } while (0)
 
   switch (ablation_mode) {
     case 1: {
       switch (split) {
         case 1:  LAUNCH_ABL((Ablation_LocalOnly_Kernel<1,  kAblCfg1.num_threads,  kAblCfg1.items_per_thread>),  grid_full, kAblCfg1.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(),
                             part_keys_ptr, part_idx_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 2:  LAUNCH_ABL((Ablation_LocalOnly_Kernel<2,  kAblCfg2.num_threads,  kAblCfg2.items_per_thread>),  grid_full, kAblCfg2.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(),
                             part_keys_ptr, part_idx_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 4:  LAUNCH_ABL((Ablation_LocalOnly_Kernel<4,  kAblCfg4.num_threads,  kAblCfg4.items_per_thread>),  grid_full, kAblCfg4.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(),
                             part_keys_ptr, part_idx_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 8:  LAUNCH_ABL((Ablation_LocalOnly_Kernel<8,  kAblCfg8.num_threads,  kAblCfg8.items_per_thread>),  grid_full, kAblCfg8.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(),
                             part_keys_ptr, part_idx_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 16: LAUNCH_ABL((Ablation_LocalOnly_Kernel<16, kAblCfg16.num_threads, kAblCfg16.items_per_thread>), grid_full, kAblCfg16.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(),
                             part_keys_ptr, part_idx_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 32: LAUNCH_ABL((Ablation_LocalOnly_Kernel<32, kAblCfg32.num_threads, kAblCfg32.items_per_thread>), grid_full, kAblCfg32.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(),
                             part_keys_ptr, part_idx_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
       }
       break;
     }
     case 2: {
       switch (split) {
         case 1:  LAUNCH_ABL((Ablation_LocalNoWorkspace_Kernel<1,  kAblCfg1.num_threads,  kAblCfg1.items_per_thread>),  grid_full, kAblCfg1.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(), scratch_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 2:  LAUNCH_ABL((Ablation_LocalNoWorkspace_Kernel<2,  kAblCfg2.num_threads,  kAblCfg2.items_per_thread>),  grid_full, kAblCfg2.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(), scratch_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 4:  LAUNCH_ABL((Ablation_LocalNoWorkspace_Kernel<4,  kAblCfg4.num_threads,  kAblCfg4.items_per_thread>),  grid_full, kAblCfg4.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(), scratch_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 8:  LAUNCH_ABL((Ablation_LocalNoWorkspace_Kernel<8,  kAblCfg8.num_threads,  kAblCfg8.items_per_thread>),  grid_full, kAblCfg8.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(), scratch_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 16: LAUNCH_ABL((Ablation_LocalNoWorkspace_Kernel<16, kAblCfg16.num_threads, kAblCfg16.items_per_thread>), grid_full, kAblCfg16.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(), scratch_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
         case 32: LAUNCH_ABL((Ablation_LocalNoWorkspace_Kernel<32, kAblCfg32.num_threads, kAblCfg32.items_per_thread>), grid_full, kAblCfg32.num_threads,
                             x_ptr, dense_kv_indptr.data_ptr<int32_t>(),
                             dense_kv_indices.data_ptr<int32_t>(), scratch_ptr,
                             static_cast<int>(reserved_bos), static_cast<int>(reserved_eos)); break;
       }
       break;
     }
     case 3: {
       switch (split) {
         case 1:  LAUNCH_ABL((Ablation_WorkspaceWriteOnly_Kernel<1>),  grid_full, 32, part_keys_ptr, part_idx_ptr); break;
         case 2:  LAUNCH_ABL((Ablation_WorkspaceWriteOnly_Kernel<2>),  grid_full, 32, part_keys_ptr, part_idx_ptr); break;
         case 4:  LAUNCH_ABL((Ablation_WorkspaceWriteOnly_Kernel<4>),  grid_full, 32, part_keys_ptr, part_idx_ptr); break;
         case 8:  LAUNCH_ABL((Ablation_WorkspaceWriteOnly_Kernel<8>),  grid_full, 32, part_keys_ptr, part_idx_ptr); break;
         case 16: LAUNCH_ABL((Ablation_WorkspaceWriteOnly_Kernel<16>), grid_full, 32, part_keys_ptr, part_idx_ptr); break;
         case 32: LAUNCH_ABL((Ablation_WorkspaceWriteOnly_Kernel<32>), grid_full, 32, part_keys_ptr, part_idx_ptr); break;
       }
       break;
     }
     case 4: {
       switch (split) {
         case 1:  LAUNCH_ABL((Ablation_AtomicOnly_Kernel<1>),  grid_full, 32, done_ptr, scratch_ptr); break;
         case 2:  LAUNCH_ABL((Ablation_AtomicOnly_Kernel<2>),  grid_full, 32, done_ptr, scratch_ptr); break;
         case 4:  LAUNCH_ABL((Ablation_AtomicOnly_Kernel<4>),  grid_full, 32, done_ptr, scratch_ptr); break;
         case 8:  LAUNCH_ABL((Ablation_AtomicOnly_Kernel<8>),  grid_full, 32, done_ptr, scratch_ptr); break;
         case 16: LAUNCH_ABL((Ablation_AtomicOnly_Kernel<16>), grid_full, 32, done_ptr, scratch_ptr); break;
         case 32: LAUNCH_ABL((Ablation_AtomicOnly_Kernel<32>), grid_full, 32, done_ptr, scratch_ptr); break;
       }
       break;
     }
     // kAblMode_MergeProdDefault=5, kAblMode_MergeCubWarp=6, kAblMode_MergeCubBlock=7
     // map to MERGE_PROD_DEFAULT=0, MERGE_CUB_WARP=1, MERGE_CUB_BLOCK=2 respectively.
     case 5: case 6: case 7: {
       const int variant = static_cast<int>(ablation_mode - 5);
       auto launch_merge = [&](auto split_const_var, int v) {
         constexpr int S = decltype(split_const_var)::value;
         switch (v) {
           case MERGE_PROD_DEFAULT:
             LAUNCH_ABL((Ablation_MergeOnly_Kernel<S, MERGE_PROD_DEFAULT>), grid_merge, 32,
                        part_keys_ptr, part_idx_ptr,
                        sparse_kv_indptr.data_ptr<int32_t>(),
                        sparse_kv_indices.data_ptr<int32_t>(),
                        static_cast<int>(topk_val),
                        static_cast<int>(reserved_bos)); break;
           case MERGE_CUB_WARP:
             LAUNCH_ABL((Ablation_MergeOnly_Kernel<S, MERGE_CUB_WARP>), grid_merge, 32,
                        part_keys_ptr, part_idx_ptr,
                        sparse_kv_indptr.data_ptr<int32_t>(),
                        sparse_kv_indices.data_ptr<int32_t>(),
                        static_cast<int>(topk_val),
                        static_cast<int>(reserved_bos)); break;
           case MERGE_CUB_BLOCK:
             LAUNCH_ABL((Ablation_MergeOnly_Kernel<S, MERGE_CUB_BLOCK>), grid_merge, 64,
                        part_keys_ptr, part_idx_ptr,
                        sparse_kv_indptr.data_ptr<int32_t>(),
                        sparse_kv_indices.data_ptr<int32_t>(),
                        static_cast<int>(topk_val),
                        static_cast<int>(reserved_bos)); break;
         }
       };
       switch (split) {
         case 1:  launch_merge(std::integral_constant<int, 1>{},  variant); break;
         case 2:  launch_merge(std::integral_constant<int, 2>{},  variant); break;
         case 4:  launch_merge(std::integral_constant<int, 4>{},  variant); break;
         case 8:  launch_merge(std::integral_constant<int, 8>{},  variant); break;
         case 16: launch_merge(std::integral_constant<int, 16>{}, variant); break;
         case 32: launch_merge(std::integral_constant<int, 32>{}, variant); break;
       }
       break;
     }
     case 9: {  // kAblMode_MergeManual2Way
       TORCH_CHECK(split == 2, "ablation 9 (merge_manual_2way) requires forced_splits=2");
       LAUNCH_ABL((Ablation_MergeOnly_Kernel<2, MERGE_MANUAL_2WAY>), grid_merge, 32,
                  part_keys_ptr, part_idx_ptr,
                  sparse_kv_indptr.data_ptr<int32_t>(),
                  sparse_kv_indices.data_ptr<int32_t>(),
                  static_cast<int>(topk_val),
                  static_cast<int>(reserved_bos));
       break;
     }
     case 10: {  // kAblMode_MergePairwise4
       TORCH_CHECK(split == 4, "ablation 10 (merge_pairwise_4) requires forced_splits=4");
       LAUNCH_ABL((Ablation_MergeOnly_Kernel<4, MERGE_PAIRWISE_4>), grid_merge, 32,
                  part_keys_ptr, part_idx_ptr,
                  sparse_kv_indptr.data_ptr<int32_t>(),
                  sparse_kv_indices.data_ptr<int32_t>(),
                  static_cast<int>(topk_val),
                  static_cast<int>(reserved_bos));
       break;
     }
     case 11: {  // kAblMode_MergeKwayAll: force k-way regardless of SPLITS
       auto launch_kway = [&](auto split_tag) {
         constexpr int S = decltype(split_tag)::value;
         LAUNCH_ABL((Ablation_MergeOnly_Kernel<S, MERGE_KWAY>), grid_merge, 32,
                    part_keys_ptr, part_idx_ptr,
                    sparse_kv_indptr.data_ptr<int32_t>(),
                    sparse_kv_indices.data_ptr<int32_t>(),
                    static_cast<int>(topk_val),
                    static_cast<int>(reserved_bos));
       };
       switch (split) {
         case 1:  launch_kway(std::integral_constant<int, 1>{}); break;
         case 2:  launch_kway(std::integral_constant<int, 2>{}); break;
         case 4:  launch_kway(std::integral_constant<int, 4>{}); break;
         case 8:  launch_kway(std::integral_constant<int, 8>{}); break;
         case 16: launch_kway(std::integral_constant<int, 16>{}); break;
         case 32: launch_kway(std::integral_constant<int, 32>{}); break;
       }
       break;
     }
     case 0: {
       // full_parallel — re-enter the production workspace API with forced
       // split + CONTIGUOUS partition. This makes the "0" mode useful as the
       // 100% reference for the other ablations.
       topk_output_adaptive_workspace(
           x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
           sparse_kv_indices, partial_keys, partial_indices, done_counter,
           eff_batch_size, topk_val, reserved_bos, reserved_eos,
           max_num_pages, /*mapping_mode=*/0, /*mapping_power=*/0.0,
           forced_splits, /*forced_partition=*/kPartContiguous, /*local_mode=*/0);
       break;
     }
     default:
       TORCH_CHECK(false, "unknown ablation_mode=", ablation_mode,
                   " (valid range: 0–11)");
   }
   #undef LAUNCH_ABL
 
   const auto rc = cudaGetLastError();
   TORCH_CHECK(rc == cudaSuccess,
               "ablation launch failed: ", ::cudaGetErrorString(rc));
 }
