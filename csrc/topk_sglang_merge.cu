/**
 * Vortex adaptive split TopK — random-split parallel K=30 path + fused
 * fallback. Lives in topk_sglang_merge.cu (NOT a new file).
 *
 * Dispatch summary (host-side topk_output_adaptive_workspace):
 *
 *   topk_val >= 1024  → immediate call to topk_output_sglang_fused.
 *                       No workspace touched, no done_counter memset, no
 *                       split kernel launched. Required for 32k → 2048.
 *
 *   topk_val >  32 (and < 1024) → also forwards to fused (no specialised path).
 *
 *   topk_val <= 32:
 *       forced_splits  > 0  → use that split count (1, 2, 4, 8, 16, 32).
 *       forced_splits <= 0  → use heuristic pick_split_top30().
 *       split == 1 (heuristic only) → fall back to fused.
 *       split == 1 (forced)         → run the SPLITS=1 single-CTA path
 *                                     for benchmarking (one CUDA block sorts
 *                                     the whole row with cub::BlockRadixSort).
 *
 * Random split semantics: each split processes ONLY its slice of the row,
 * not the whole row filtered by predicate, so total work = O(n) not O(n*S).
 *
 *   group_begin = (n * split_id)     / SPLITS
 *   group_end   = (n * (split_id+1)) / SPLITS
 *   For each logical rank r in [group_begin, group_end), the physical
 *   page-table position is `permute(r, b_offset, n)`. For pow2 n we use
 *   the affine bijection
 *       pos = (r * a + b_offset) & (n - 1)
 *   with a = golden-ratio constant 2654435769 (odd → bijective mod 2^k).
 *   Per-row b_offset = b * 1013904223 + r0, where r0 is a fixed seed
 *   for reproducibility. For non-pow2 n we fall back to the contiguous
 *   mapping pos = r (the chunks then become consecutive slices).
 *
 * Local stage: cub::BlockRadixSort<uint32_t, NUM_THREADS, ITEMS_PER_THREAD,
 * int32_t>::SortDescending. Writes the top kLocalK=32 (key, idx) pairs to
 * partial workspace per (row, split).
 *
 * Merge stage (last CTA, SPLITS > 1): cub::WarpMergeSort over SPLITS*32
 * candidates. kMergeIPT = SPLITS items per thread; the 32 warp lanes
 * together hold all SPLITS*32 candidates. Each thread's SPLITS items are
 * a contiguous descending-sorted slice of the workspace (one split per
 * kMergeIPT items), so the WarpMergeSort precondition is satisfied.
 * After the sort, threads write their items to out_idx at the correct
 * global rank; only ranks < topk_val are written.
 * Final top-topk_val global page IDs land in
 * sparse_kv_indices[sparse_kv_indptr[b] + reserved_bos + rank].
 *
 * done_counter is the external workspace; the host clears it with
 * cudaMemsetAsync before each parallel-path launch. The fused-fallback
 * branches do not touch it.
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
 
 #include <cub/block/block_radix_sort.cuh>
 #include <cub/block/block_merge_sort.cuh>
 #include <cub/warp/warp_merge_sort.cuh>
 
 #include <algorithm>
 #include <cstddef>
 #include <cstdint>
 #include <optional>
 
 #include "register.h"
 
 namespace {

 constexpr int kLocalK_Top30    = 32;     // local top-K per chunk
 constexpr int kMaxFinalK_Top30 = 32;     // accept topk_val up to this
 constexpr int64_t kFusedFallbackTopK = 1024;  // K >= this routes to fused

 // =============================================================================
 // Local-stage policy for the K=30 split kernel.
 //
 //   BLOCK_FULL_SORT  : per-CTA cub::BlockRadixSort over the whole split group,
 //                      capped by the NT*IPT capacity ladder in kCfg* below.
 //                      Original baseline kernel.
 //
 //   SELECT32_SORT32  : per-CTA sglang-style 8-bit radix-select that emits
 //                      exactly LOCAL_K=32 candidates without sorting the
 //                      whole group, followed by a 32-element warp bitonic
 //                      sort (cub::WarpMergeSort with IPT=1). Inner loops
 //                      are strided over the group, so there is no NT*IPT
 //                      ceiling and arbitrary chunk_len is supported.
 //
 // Both modes share the merge stage: each CTA writes a sorted local top-32
 // to partial workspace, the last CTA per row runs merge_cub_warp_topk.
 // =============================================================================
 enum TopK30LocalMode : int {
   LOCAL_BLOCK_FULL_SORT = 0,
   LOCAL_SELECT32_SORT32 = 1,
 };
 
 // Affine permutation constants (LCG-style). a is odd → bijective mod 2^k.
 constexpr uint32_t kPermuteA      = 2654435769u;  // golden ratio fractional bits
 constexpr uint32_t kPermuteSeedB  = 1013904223u;
 constexpr uint32_t kPermuteOffset = 0x9E3779B9u;  // additional offset
 
 // ---- bit-level helpers ------------------------------------------------------
 
 // Sortable uint32 key for an fp32 value: ascending uint32 == ascending fp32.
 __device__ __forceinline__ uint32_t convert_to_uint32(float x) {
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
 
 // Stage-1 8-bit bin used by topk_mapping.cuh's compute_stage1_bin. Defined
 // here so the header pulls in cleanly, even though we don't otherwise use it.
 __device__ __forceinline__ uint8_t convert_to_uint8(float x) {
   __half h = __float2half_rn(x);
   uint16_t bits = __half_as_ushort(h);
   uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                  : static_cast<uint16_t>(bits | 0x8000);
   return static_cast<uint8_t>(key >> 8);
 }
 
 #include "topk_mapping.cuh"
 
 // Affine permutation modulo 2^k, bijective when n is a power of two.
 __device__ __forceinline__ int permute_pow2(uint32_t r, uint32_t b_off, uint32_t n_mask) {
   return static_cast<int>((r * kPermuteA + b_off) & n_mask);
 }
 
 // True iff n is a strictly positive power of two.
 __device__ __host__ __forceinline__ bool is_pow2(int n) {
   return n > 0 && ((n & (n - 1)) == 0);
 }
 
 // =============================================================================
 // Partition modes — control how a row's logical-rank space [0, n) is mapped
 // to physical positions per split CTA. Goal: keep total work O(n) (no
 // per-split full scan) while controlling memory access locality.
 //
 //   AFFINE_RANDOM   : pos = (a*r + b_off) & (n-1)             [random gather]
 //   CONTIGUOUS      : pos = group_begin + local_rank          [coalesced]
 //   STRIDED         : pos = split_id + local_rank * SPLITS    [interleaved]
 //   TILE_RANDOM_128 : tile-permute then read TILE=128 contiguous positions
 //                     within each tile.
 //   TILE_RANDOM_256 : same as 128 but with TILE=256.
 // =============================================================================
 enum PartitionMode : int {
   PART_AFFINE_RANDOM   = 0,
   PART_CONTIGUOUS      = 1,
   PART_STRIDED         = 2,
   PART_TILE_RANDOM_128 = 3,
   PART_TILE_RANDOM_256 = 4,
 };
 constexpr int kTileSize128 = 128;
 constexpr int kTileSize256 = 256;
 
 // Tile-random: divide row into TILE-sized contiguous tiles, permute the tile
 // IDs across the row using the affine bijection, and assign tiles_per_split
 // = chunk_len / TILE tiles to each split. Within a tile, reads are
 // contiguous → coalesced 128B / 256B sectors.
 template <int SPLITS, int TILE>
 __device__ __forceinline__ int tile_random_pos(
     int local_rank, int row_len, int split_id,
     uint32_t b_off, uint32_t n_mask)
 {
   const int chunk_len = row_len / SPLITS;
   if (chunk_len < TILE) {
     // Fallback to affine when tiles don't fit.
     const int group_begin = (row_len * split_id) / SPLITS;
     const int r = group_begin + local_rank;
     return permute_pow2(static_cast<uint32_t>(r), b_off, n_mask);
   }
   const int tiles_per_split  = chunk_len / TILE;
   const int tile_in_split    = local_rank / TILE;
   const int offset_in_tile   = local_rank & (TILE - 1);
   const int global_tile_rank = split_id * tiles_per_split + tile_in_split;
   const int tile_count       = row_len / TILE;
   const uint32_t tile_mask   = static_cast<uint32_t>(tile_count - 1);
   const uint32_t tile_id     =
       (static_cast<uint32_t>(global_tile_rank) * kPermuteA + b_off) & tile_mask;
   return static_cast<int>(tile_id) * TILE + offset_in_tile;
 }
 
 template <int PARTITION, int SPLITS>
 __device__ __forceinline__ int compute_pos(
     int local_rank, int row_len, int split_id,
     uint32_t b_off, uint32_t n_mask)
 {
   if constexpr (PARTITION == PART_CONTIGUOUS) {
     const int group_begin = (row_len * split_id) / SPLITS;
     return group_begin + local_rank;
   } else if constexpr (PARTITION == PART_STRIDED) {
     // Each split owns lanes [split_id, split_id+SPLITS, split_id+2*SPLITS, ...].
     // Across all splits, the union covers every position in [0, row_len)
     // exactly once when row_len is divisible by SPLITS.
     return split_id + local_rank * SPLITS;
   } else if constexpr (PARTITION == PART_TILE_RANDOM_128) {
     return tile_random_pos<SPLITS, kTileSize128>(
         local_rank, row_len, split_id, b_off, n_mask);
   } else if constexpr (PARTITION == PART_TILE_RANDOM_256) {
     return tile_random_pos<SPLITS, kTileSize256>(
         local_rank, row_len, split_id, b_off, n_mask);
   } else {
     // AFFINE_RANDOM (default).
     const int group_begin = (row_len * split_id) / SPLITS;
     const int r = group_begin + local_rank;
     return permute_pow2(static_cast<uint32_t>(r), b_off, n_mask);
   }
 }
 
 // =============================================================================
 // Descending comparator for cub::WarpMergeSort (sorts largest key first).
 // =============================================================================
 struct DescendingUint32 {
   __device__ __forceinline__ bool operator()(uint32_t a, uint32_t b) const {
     return a > b;
   }
 };

 // =============================================================================
 // Single-warp CUB merge of SPLITS sorted top-LOCAL_K lists.
 //
 // Workspace layout (keys_in / idx_in): SPLITS * LOCAL_K elements, split-major
 // with each split's LOCAL_K entries sorted descending. With LOCAL_K=32 and
 // kMergeIPT = SPLITS (= SPLITS*32 / 32), thread tx holds exactly SPLITS
 // consecutive items starting at tx*SPLITS — always a contiguous sorted slice
 // within a single split's list. cub::WarpMergeSort precondition is satisfied.
 //
 // After the sort the global top-final_k indices are written to out_idx[0..final_k-1]
 // by the threads that own those ranks; no lane conflicts.
 //
 // Register pressure: kMergeIPT = SPLITS. For SPLITS=32 each lane holds 32
 // key+value pairs (~128 B registers). Acceptable for sm_90+.
 // =============================================================================
 template <int SPLITS, int LOCAL_K, int MAX_FINAL_K>
 __device__ __forceinline__ void merge_cub_warp_topk(
     const uint32_t* __restrict__ keys_in,
     const int32_t*  __restrict__ idx_in,
     int32_t*        __restrict__ out_idx,
     int             final_k)
 {
   constexpr int kCandidates = SPLITS * LOCAL_K;
   constexpr int kMergeIPT   = (kCandidates + 31) / 32;
   using WarpMergeT = cub::WarpMergeSort<uint32_t, kMergeIPT, 32, int32_t>;
   __shared__ typename WarpMergeT::TempStorage warp_merge_smem;
   const int tx = threadIdx.x;
   if (tx < 32) {
     uint32_t wkeys[kMergeIPT];
     int32_t  wvals[kMergeIPT];
     #pragma unroll
     for (int k = 0; k < kMergeIPT; ++k) {
       const int rank = tx * kMergeIPT + k;
       wkeys[k] = (rank < kCandidates) ? keys_in[rank] : 0u;
       wvals[k] = (rank < kCandidates) ? idx_in [rank] : -1;
     }
     WarpMergeT(warp_merge_smem).Sort(wkeys, wvals, DescendingUint32{});
     #pragma unroll
     for (int k = 0; k < kMergeIPT; ++k) {
       const int rank = tx * kMergeIPT + k;
       if (rank < final_k && wvals[k] >= 0) out_idx[rank] = wvals[k];
     }
   }
 }
 
 template <auto* f, size_t max_dynamic_smem>
 inline void setup_kernel_smem_once() {
   [[maybe_unused]]
   static const auto result = [] {
     return ::cudaFuncSetAttribute(
         f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
   }();
   TORCH_CHECK(result == cudaSuccess,
               "topk_output_adaptive setup failed: ",
               ::cudaGetErrorString(result));
 }
 
 // =============================================================================
 // K=30 random-split parallel kernel.
 //
 //   Grid: (eff_batch_size, SPLITS).
 //   blockIdx.x = effective row id (0..eff_batch_size-1)
 //   blockIdx.y = split id        (0..SPLITS-1)
 //
 // Stage 1 (every CTA):
 //   - Compute group_begin/group_end for this split.
 //   - For each local rank in [0, group_len), compute physical pos via
 //     permute_pow2 (or contiguous fallback for non-pow2 n).
 //   - Apply apply_transform_tmpl<MODE>, build (uint32_key, int32_global_idx).
 //   - cub::BlockRadixSort.SortDescending. Top items at start of array.
 //
 //   For SPLITS == 1 the kernel writes the top topk_val directly to
 //   sparse_kv_indices and returns — no merge.
 //
 //   For SPLITS  > 1, write the top kLocalK=32 (key, idx) pairs to the
 //   partial workspace at offset (b*SPLITS + n)*kLocalK.
 //
 // Last-CTA-wins barrier (SPLITS > 1):
 //   __threadfence (release) → atomicAdd → if old == SPLITS-1, last CTA →
 //   __threadfence (acquire) → __syncthreads.
 //
 // Stage 2 (last CTA, SPLITS > 1):
 //   - Load SPLITS*32 candidates into one warp / one block.
 //   - Sort descending by uint32 key.
 //   - Lanes 0..topk_val-1 (or threads) write their item to sparse_kv_indices.
 // =============================================================================
 template <typename ScoreT, int MODE, int SPLITS, int NUM_THREADS,
           int ITEMS_PER_THREAD, int PARTITION>
 __global__ __launch_bounds__(NUM_THREADS)
 void TopK30_RandomSplit_Parallel_Kernel(
     const ScoreT* __restrict__ score,
     const int*    __restrict__ dense_kv_indptr,
     const int*    __restrict__ sparse_kv_indptr,
     const int*    __restrict__ dense_kv_indices,
     int*          __restrict__ sparse_kv_indices,
     uint32_t*     __restrict__ partial_keys,
     int32_t*      __restrict__ partial_indices,
     int32_t*      __restrict__ done_counter,
     const int     topk_val,
     const int     reserved_bos,
     const int     reserved_eos,
     const float   mapping_power)
 {
   using KeyT       = uint32_t;
   using ValueT     = int32_t;
   using BlockSortT = cub::BlockRadixSort<KeyT, NUM_THREADS, ITEMS_PER_THREAD, ValueT>;
 
   constexpr int kLocalK = kLocalK_Top30;
 
   __shared__ typename BlockSortT::TempStorage sort_smem;
   __shared__ int s_is_last;
 
   const int b  = blockIdx.x;
   const int n  = blockIdx.y;
   const int tx = threadIdx.x;
 
   const int row_start = dense_kv_indptr[b] + reserved_bos;
   const int row_end   = dense_kv_indptr[b + 1] - reserved_eos;
   const int row_len   = max(0, row_end - row_start);
 
   if (row_len <= 0) return;
 
   // --- Group boundaries (no overlap, no gaps across splits). ---
   const int group_begin = (static_cast<int64_t>(row_len) * n)         / SPLITS;
   const int group_end   = (static_cast<int64_t>(row_len) * (n + 1))   / SPLITS;
   const int group_len   = group_end - group_begin;
 
   // --- Permutation parameters. ---
   // For pow2 row_len, use affine bijection mod row_len. For non-pow2, fall
   // back to identity (chunks become consecutive slices).
   const bool     row_is_pow2 = is_pow2(row_len);
   const uint32_t n_mask      = row_is_pow2 ? static_cast<uint32_t>(row_len - 1) : 0u;
   const uint32_t b_off       =
       static_cast<uint32_t>(b) * kPermuteSeedB + kPermuteOffset;
 
   const ScoreT* row_scores = score            + row_start;
   const int*    row_idxmap = dense_kv_indices + row_start;
 
   // ------------------------------------------------------------------ Stage 1
   KeyT   keys[ITEMS_PER_THREAD];
   ValueT values[ITEMS_PER_THREAD];
 
   #pragma unroll
   for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
     const int local_rank = tx + k * NUM_THREADS;
     if (local_rank < group_len) {
       int pos;
       if (row_is_pow2) {
         pos = compute_pos<PARTITION, SPLITS>(
             local_rank, row_len, n, b_off, n_mask);
       } else {
         // Non-pow2 fallback: contiguous slice (also semantically valid for
         // partition=CONTIGUOUS).
         pos = group_begin + local_rank;
       }
       const float raw      = vortex_to_float(row_scores[pos]);
       const float remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       keys  [k] = convert_to_uint32(remapped);
       values[k] = row_idxmap[pos];
     } else {
       keys  [k] = 0u;
       values[k] = -1;
     }
   }
 
   BlockSortT(sort_smem).SortDescending(keys, values);
   __syncthreads();
 
   // SPLITS == 1 special case: write final output directly. No atomic, no merge.
   if constexpr (SPLITS == 1) {
     int32_t* out_idx = sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos;
     #pragma unroll
     for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
       const int rank = tx * ITEMS_PER_THREAD + k;
       if (rank < topk_val) out_idx[rank] = values[k];
     }
     return;
   }
 
   // SPLITS > 1: write local top kLocalK to partial workspace.
   const int64_t part_off = (static_cast<int64_t>(b) * SPLITS + n) * kLocalK;
   uint32_t* part_keys = partial_keys    + part_off;
   int32_t*  part_idx  = partial_indices + part_off;
 
   #pragma unroll
   for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
     const int rank = tx * ITEMS_PER_THREAD + k;
     if (rank < kLocalK) {
       part_keys[rank] = keys[k];
       part_idx [rank] = values[k];
     }
   }
 
   // -------------------------------------------------------- Last-CTA barrier
   __threadfence();
   __syncthreads();
   if (tx == 0) {
     const int old = ::atomicAdd(&done_counter[b], 1);
     s_is_last = (old == SPLITS - 1) ? 1 : 0;
     // Self-reset: the last CTA clears its slot for the next launch.
     // Eliminates the need for cudaMemsetAsync(done_counter) on the host —
     // saves ~1-2 µs of CPU launch overhead per call. Same-stream kernels are
     // sequenced, so the next launch sees done_counter[b] == 0.
     if (s_is_last) done_counter[b] = 0;
   }
   __syncthreads();
   if (s_is_last == 0) return;
   // Acquire fence: ensure the merging CTA observes other CTAs' partial writes.
   __threadfence();
   __syncthreads();

   // ------------------------------------------------------------------ Stage 2
   // cub::WarpMergeSort over all SPLITS*kLocalK candidates (warp 0 only).
   // kMergeIPT = SPLITS items per lane; each lane's items are a contiguous
   // sorted slice within a single split's list, satisfying WarpMergeSort's
   // pre-sorted-per-thread precondition.
   const int64_t row_off = static_cast<int64_t>(b) * SPLITS * kLocalK;
   const uint32_t* keys_in = partial_keys    + row_off;
   const int32_t*  idx_in  = partial_indices + row_off;
   int32_t* out_idx = sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos;

   merge_cub_warp_topk<SPLITS, kLocalK, kMaxFinalK_Top30>(
       keys_in, idx_in, out_idx, topk_val);
 }

 // =============================================================================
 // K=30 SELECT32_SORT32 local-stage kernel (Plan C).
 //
 //   Grid: (eff_batch_size, SPLITS).
 //   blockIdx.x = effective row id, blockIdx.y = split id.
 //
 // Per-CTA pipeline:
 //   Pass 1  - top-byte (bits [31:24]) histogram + suffix-sum-descending,
 //             find the threshold bin where cumulative count crosses
 //             LOCAL_K=32 (unique by monotonicity).
 //   Pass 2  - re-scan the split group: items strictly above the threshold
 //             bin go straight into the candidate buffer (count is
 //             guaranteed < LOCAL_K). Items at the threshold bin contribute
 //             to a sub-bin (bits [23:16]) histogram.
 //   Pass 3  - find the sub-threshold bin in the sub-hist, then re-scan the
 //             threshold bin and gather (sub > sub_threshold) and
 //             (sub == sub_threshold) candidates into the remaining slots.
 //   Stage D - 32-lane warp bitonic sort over the LOCAL_K candidates,
 //             descending by uint32 key. Implemented via cub::WarpMergeSort
 //             with IPT=1 (sort precondition is trivially satisfied).
 //
 // SPLITS == 1: write top topk_val directly to sparse_kv_indices, no
 //              workspace, no atomic, no merge.
 // SPLITS  > 1: write sorted local top-LOCAL_K to partial workspace, the
 //              last CTA per row runs merge_cub_warp_topk.
 //
 // No cub::BlockRadixSort smem and no NT*IPT capacity ceiling.  Each pass
 // is a strided loop over [0, group_len) so the kernel handles any
 // chunk length the splits produce, including the SPLITS=1 / 32k case.
 // =============================================================================
 template <typename ScoreT, int MODE, int SPLITS, int NUM_THREADS, int PARTITION>
 __global__ __launch_bounds__(NUM_THREADS)
 void TopK30_RandomSplit_Select32_Kernel(
     const ScoreT* __restrict__ score,
     const int*    __restrict__ dense_kv_indptr,
     const int*    __restrict__ sparse_kv_indptr,
     const int*    __restrict__ dense_kv_indices,
     int*          __restrict__ sparse_kv_indices,
     uint32_t*     __restrict__ partial_keys,
     int32_t*      __restrict__ partial_indices,
     int32_t*      __restrict__ done_counter,
     const int     topk_val,
     const int     reserved_bos,
     const int     reserved_eos,
     const float   mapping_power)
 {
   constexpr int LOCAL_K = kLocalK_Top30;
   constexpr int kRadix  = 256;

   alignas(128) __shared__ int s_hist_buf[2][kRadix + 128];
   __shared__ int      s_above_count;          // count strictly above threshold_bin (pass 2)
   __shared__ int      s_thresh_above_count;   // count (bin==t && sub>sub_t) (pass 3)
   __shared__ int      s_thresh_at_count;      // count (bin==t && sub==sub_t) (pass 3, capped)
   __shared__ int      s_threshold_bin;
   __shared__ int      s_last_remain;
   __shared__ int      s_sub_threshold_bin;
   __shared__ int      s_sub_last_remain;
   __shared__ int      s_strictly_above_sub;
   __shared__ uint32_t s_top_keys[LOCAL_K];
   __shared__ int32_t  s_top_idx [LOCAL_K];
   __shared__ int      s_is_last;

   using LocalSortT = cub::WarpMergeSort<uint32_t, 1, 32, int32_t>;
   __shared__ typename LocalSortT::TempStorage local_sort_smem;

   const int b  = blockIdx.x;
   const int n  = blockIdx.y;
   const int tx = threadIdx.x;

   const int row_start = dense_kv_indptr[b] + reserved_bos;
   const int row_end   = dense_kv_indptr[b + 1] - reserved_eos;
   const int row_len   = max(0, row_end - row_start);

   const int group_begin = (static_cast<int64_t>(row_len) * n)         / SPLITS;
   const int group_end   = (static_cast<int64_t>(row_len) * (n + 1))   / SPLITS;
   const int group_len   = group_end - group_begin;

   const bool     row_is_pow2 = is_pow2(row_len);
   const uint32_t n_mask      = row_is_pow2 ? static_cast<uint32_t>(row_len - 1) : 0u;
   const uint32_t b_off       =
       static_cast<uint32_t>(b) * kPermuteSeedB + kPermuteOffset;
   const ScoreT*  row_scores  = score            + row_start;
   const int*     row_idxmap  = dense_kv_indices + row_start;

   // ---- Init shared state. Strided over the +128 padding so any NT works. ----
   for (int i = tx; i < kRadix + 128; i += NUM_THREADS) {
     s_hist_buf[0][i] = 0;
     s_hist_buf[1][i] = 0;
   }
   if (tx == 0) {
     s_above_count        = 0;
     s_thresh_above_count = 0;
     s_thresh_at_count    = 0;
     s_threshold_bin      = -1;
     s_last_remain        = 0;
     s_sub_threshold_bin  = -1;
     s_sub_last_remain    = 0;
     s_strictly_above_sub = 0;
     s_is_last            = 0;
   }
   if (tx < LOCAL_K) {
     s_top_keys[tx] = 0u;
     s_top_idx [tx] = -1;
   }
   __syncthreads();

   // Empty-row early exit. SPLITS>1 must still participate in the merge
   // barrier so the last-CTA flag fires; padding is already (0u, -1).
   if (row_len <= 0 || group_len <= 0) {
     if constexpr (SPLITS > 1) {
       const int64_t part_off =
           (static_cast<int64_t>(b) * SPLITS + n) * LOCAL_K;
       if (tx < LOCAL_K) {
         partial_keys   [part_off + tx] = 0u;
         partial_indices[part_off + tx] = -1;
       }
       __threadfence();
       __syncthreads();
       if (tx == 0) {
         const int old = ::atomicAdd(&done_counter[b], 1);
         s_is_last = (old == SPLITS - 1) ? 1 : 0;
         if (s_is_last) done_counter[b] = 0;  // self-reset for next launch
       }
       __syncthreads();
       if (s_is_last == 0) return;
       __threadfence();
       __syncthreads();
       const int64_t row_off = static_cast<int64_t>(b) * SPLITS * LOCAL_K;
       merge_cub_warp_topk<SPLITS, LOCAL_K, kMaxFinalK_Top30>(
           partial_keys + row_off, partial_indices + row_off,
           sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos,
           topk_val);
     }
     return;
   }

   // Strided suffix-sum-descending over s_hist_buf, ping-pong; result in [0].
   // Works for any NUM_THREADS (uses a strided inner loop over kRadix).
   auto run_cumsum_strided = [&]() {
     #pragma unroll
     for (int i = 0; i < 8; ++i) {
       const int j = 1 << i;
       const int k = i & 1;
       for (int idx = tx; idx < kRadix; idx += NUM_THREADS) {
         int v = s_hist_buf[k][idx];
         if (idx + j < kRadix) v += s_hist_buf[k][idx + j];
         s_hist_buf[k ^ 1][idx] = v;
       }
       __syncthreads();
     }
   };

   // ============================================================
   // Pass 1: top-byte histogram.
   // ============================================================
   for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
     int pos;
     if (row_is_pow2) {
       pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
     } else {
       pos = group_begin + local_rank;
     }
     const float    raw      = vortex_to_float(row_scores[pos]);
     const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
     const uint32_t key      = convert_to_uint32(remapped);
     const int      bin      = static_cast<int>(key >> 24);
     ::atomicAdd(&s_hist_buf[0][bin], 1);
   }
   __syncthreads();

   run_cumsum_strided();
   // s_hist_buf[0][bin] = count of items with key>>24 >= bin.

   const int total_items = s_hist_buf[0][0];

   // Find threshold bin: the unique bin t where total_at_or_above[t] >= K
   // and strictly_above[t] < K. Strided so any NT works.
   for (int bin = tx; bin < kRadix; bin += NUM_THREADS) {
     const int total_at_or_above = s_hist_buf[0][bin];
     const int strictly_above    = (bin + 1 < kRadix) ? s_hist_buf[0][bin + 1] : 0;
     if (total_at_or_above >= LOCAL_K && strictly_above < LOCAL_K) {
       s_threshold_bin = bin;
       s_last_remain   = LOCAL_K - strictly_above;
     }
   }
   __syncthreads();

   if (total_items <= LOCAL_K) {
     // Few-elements path: collect everything in arbitrary order, pad rest.
     for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
       int pos;
       if (row_is_pow2) pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
       else             pos = group_begin + local_rank;
       const float    raw      = vortex_to_float(row_scores[pos]);
       const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       const uint32_t key      = convert_to_uint32(remapped);
       const int slot = ::atomicAdd(&s_above_count, 1);
       if (slot < LOCAL_K) {
         s_top_keys[slot] = key;
         s_top_idx [slot] = row_idxmap[pos];
       }
     }
     __syncthreads();
     // s_top_keys/idx already pre-padded to (0u, -1) at init.
   } else {
     const int threshold_bin = s_threshold_bin;

     // Reset both hist buffers for the sub-bin pass.
     for (int i = tx; i < kRadix + 128; i += NUM_THREADS) {
       s_hist_buf[0][i] = 0;
       s_hist_buf[1][i] = 0;
     }
     __syncthreads();

     // ============================================================
     // Pass 2: gather strictly-above-threshold items + build sub-hist.
     // ============================================================
     for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
       int pos;
       if (row_is_pow2) pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
       else             pos = group_begin + local_rank;
       const float    raw      = vortex_to_float(row_scores[pos]);
       const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       const uint32_t key      = convert_to_uint32(remapped);
       const int      bin      = static_cast<int>(key >> 24);
       if (bin > threshold_bin) {
         const int slot = ::atomicAdd(&s_above_count, 1);
         if (slot < LOCAL_K) {
           s_top_keys[slot] = key;
           s_top_idx [slot] = row_idxmap[pos];
         }
       } else if (bin == threshold_bin) {
         const int sub_bin = static_cast<int>((key >> 16) & 0xFF);
         ::atomicAdd(&s_hist_buf[0][sub_bin], 1);
       }
     }
     __syncthreads();

     run_cumsum_strided();

     const int last_remain = s_last_remain;
     for (int bin = tx; bin < kRadix; bin += NUM_THREADS) {
       const int total_at_or_above = s_hist_buf[0][bin];
       const int strictly_above    = (bin + 1 < kRadix) ? s_hist_buf[0][bin + 1] : 0;
       if (total_at_or_above >= last_remain && strictly_above < last_remain) {
         s_sub_threshold_bin  = bin;
         s_sub_last_remain    = last_remain - strictly_above;
         s_strictly_above_sub = strictly_above;
       }
     }
     __syncthreads();

     const int sub_threshold_bin     = s_sub_threshold_bin;
     const int sub_last_remain       = s_sub_last_remain;
     const int strictly_above_sub_bn = s_strictly_above_sub;
     const int above_base            = s_above_count;  // = strictly_above_threshold

     // ============================================================
     // Pass 3: gather threshold-bin sub-above + sub-at items.
     // ============================================================
     for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
       int pos;
       if (row_is_pow2) pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
       else             pos = group_begin + local_rank;
       const float    raw      = vortex_to_float(row_scores[pos]);
       const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       const uint32_t key      = convert_to_uint32(remapped);
       const int      bin      = static_cast<int>(key >> 24);
       if (bin == threshold_bin) {
         const int sub_bin = static_cast<int>((key >> 16) & 0xFF);
         if (sub_bin > sub_threshold_bin) {
           const int rel  = ::atomicAdd(&s_thresh_above_count, 1);
           const int slot = above_base + rel;
           if (slot < LOCAL_K) {
             s_top_keys[slot] = key;
             s_top_idx [slot] = row_idxmap[pos];
           }
         } else if (sub_bin == sub_threshold_bin) {
           const int rel = ::atomicAdd(&s_thresh_at_count, 1);
           if (rel < sub_last_remain) {
             const int slot = above_base + strictly_above_sub_bn + rel;
             if (slot < LOCAL_K) {
               s_top_keys[slot] = key;
               s_top_idx [slot] = row_idxmap[pos];
             }
           }
         }
       }
     }
     __syncthreads();
   }

   // ============================================================
   // Stage D: 32-lane warp bitonic sort over the LOCAL_K candidates.
   // cub::WarpMergeSort with IPT=1 has trivial pre-sorted-per-thread
   // precondition (each lane owns exactly 1 item).
   // ============================================================
   if (tx < 32) {
     uint32_t kk[1] = { s_top_keys[tx] };
     int32_t  vv[1] = { s_top_idx [tx] };
     LocalSortT(local_sort_smem).Sort(kk, vv, DescendingUint32{});
     s_top_keys[tx] = kk[0];
     s_top_idx [tx] = vv[0];
   }
   __syncthreads();

   // ============================================================
   // SPLITS == 1: direct write to sparse_kv_indices.
   // ============================================================
   if constexpr (SPLITS == 1) {
     int32_t* out_idx = sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos;
     if (tx < topk_val) out_idx[tx] = s_top_idx[tx];
     return;
   }

   // ============================================================
   // SPLITS > 1: write workspace, last-CTA-wins barrier, merge.
   // ============================================================
   const int64_t part_off = (static_cast<int64_t>(b) * SPLITS + n) * LOCAL_K;
   if (tx < LOCAL_K) {
     partial_keys   [part_off + tx] = s_top_keys[tx];
     partial_indices[part_off + tx] = s_top_idx [tx];
   }

   __threadfence();
   __syncthreads();
   if (tx == 0) {
     const int old = ::atomicAdd(&done_counter[b], 1);
     s_is_last = (old == SPLITS - 1) ? 1 : 0;
     if (s_is_last) done_counter[b] = 0;  // self-reset for next launch
   }
   __syncthreads();
   if (s_is_last == 0) return;
   __threadfence();
   __syncthreads();

   const int64_t row_off = static_cast<int64_t>(b) * SPLITS * LOCAL_K;
   merge_cub_warp_topk<SPLITS, LOCAL_K, kMaxFinalK_Top30>(
       partial_keys + row_off, partial_indices + row_off,
       sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos,
       topk_val);
 }

 // =============================================================================
 // Per-split (NUM_THREADS, ITEMS_PER_THREAD) configuration.
 //
 // NUM_THREADS * ITEMS_PER_THREAD must cover the per-split chunk length
 // (= ceil(max_num_pages / SPLITS)). Picked once per SPLITS rather than per
 // (SPLITS, max_num_pages) to keep the template instantiation count
 // manageable.
 //
 // cub::BlockRadixSort uses ~NT*IPT*sizeof(KeyT) bytes of static shared
 // memory; ptxas rejects kernels exceeding 48 KB static smem on sm_100a
 // without opt-in (which static smem can't easily use). Using uint32 keys
 // + 8-byte (key,value) effective footprint, we keep NT*IPT*4 <= ~32 KB →
 // NT*IPT <= 8192. Coverage:
 //
 //      | chunk_max | covers max_num_pages
 //      ------------------------------------
 //   1  |   8192    |   8192
 //   2  |   8192    |  16384
 //   4  |   4096    |  16384
 //   8  |   4096    |  32768
 //  16  |   2048    |  32768
 //  32  |   1024    |  32768
 //
 // Configs above the coverage row fall back to the fused single-CTA kernel
 // in the dispatcher (capacity check below).
 // =============================================================================
 struct SplitCfg { int splits, num_threads, items_per_thread; };
 
 constexpr SplitCfg kCfg1  = { 1, 1024,  8 };  // cap 8192
 constexpr SplitCfg kCfg2  = { 2, 1024,  8 };  // cap 8192
 constexpr SplitCfg kCfg4  = { 4,  512,  8 };  // cap 4096
 constexpr SplitCfg kCfg8  = { 8,  256, 16 };  // cap 4096
 constexpr SplitCfg kCfg16 = {16,  128, 16 };  // cap 2048
 constexpr SplitCfg kCfg32 = {32,   64, 16 };  // cap 1024
 
 // Returns the per-split capacity (NT*IPT) for a given split count, or 0 if
 // the split is not supported.
 inline int split_capacity(int split) {
   switch (split) {
     case 1:  return kCfg1.num_threads  * kCfg1.items_per_thread;
     case 2:  return kCfg2.num_threads  * kCfg2.items_per_thread;
     case 4:  return kCfg4.num_threads  * kCfg4.items_per_thread;
     case 8:  return kCfg8.num_threads  * kCfg8.items_per_thread;
     case 16: return kCfg16.num_threads * kCfg16.items_per_thread;
     case 32: return kCfg32.num_threads * kCfg32.items_per_thread;
     default: return 0;
   }
 }
 
 inline int next_supported_split(int required) {
   if (required <= 1)  return 1;
   if (required <= 2)  return 2;
   if (required <= 4)  return 4;
   if (required <= 8)  return 8;
   if (required <= 16) return 16;
   return 32;
 }

 // =============================================================================
 // Per-split NUM_THREADS for the SELECT32_SORT32 kernel.
 //
 // No NT*IPT capacity ladder: the kernel scans the split group with strided
 // loops, so any group_len works at any NT. Picked here only to balance
 // memory throughput vs occupancy. NT=128 is fine for high splits because
 // chunk_len shrinks proportionally (max_pages=32k / SPLITS=32 -> 1024).
 // =============================================================================
 struct SelectCfg { int splits, num_threads; };
 constexpr SelectCfg kSelCfg1  = { 1, 1024 };
 constexpr SelectCfg kSelCfg2  = { 2, 1024 };
 constexpr SelectCfg kSelCfg4  = { 4,  512 };
 constexpr SelectCfg kSelCfg8  = { 8,  256 };
 constexpr SelectCfg kSelCfg16 = {16,  128 };
 constexpr SelectCfg kSelCfg32 = {32,  128 };
 
 // SM-cover policy: pick the smallest supported split such that
 // total_ctas = eff_batch_size * split >= sm_count. This prioritises
 // filling the device. Capacity / merge cost are NOT considered here —
 // the dispatcher's capacity check below catches infeasible configs.
 inline int choose_split_k30_b200(int64_t eff_bs, int64_t /*max_pages*/,
                                  int forced, int sm_count)
 {
   if (forced > 0) return forced;
   constexpr int kSMCoverDefault = 180;  // B200 multiprocessorCount
   const int target_blocks = sm_count > 0 ? sm_count : kSMCoverDefault;
   const int required = static_cast<int>(
       (target_blocks + eff_bs - 1) / eff_bs);
   return next_supported_split(required);
 }
 
 // Default partition mode picker. The B200 sweep at K=30 shows CONTIGUOUS
 // dominates affine and tile-random by 10-15% at high splits (8/16/32) and
 // is within noise at low splits — coalesced loads are the bottleneck once
 // each split has more than a handful of threads. Random vs contiguous is
 // correctness-equivalent here (each split's local top-32 is merged via
 // CUB WarpMergeSort into the global top-30, regardless of partition layout).
 // Override via forced_partition for ablation.
 inline int default_partition(int /*split*/, int64_t /*max_num_pages*/) {
   return PART_CONTIGUOUS;
 }

 // Heuristic split picker for K<=32.
 //
 // ALWAYS returns an adaptive split count in {1,2,4,8,16,32}. Never falls
 // back to fused — for K=30 the dispatcher in topk_output_adaptive_workspace
 // is required to stay on the adaptive path. split=1 means "single-CTA
 // adaptive kernel", NOT "use fused sglang baseline".
 //
 // Table from B200 sweep (benchmarks/bench_topk_setting_sweep.py,
 // SELECT32_SORT32 local mode, CONTIGUOUS partition, CUB WarpMergeSort merge):
 //
 //   max_pages <= 32768  : split=1 wins or ties at every B in {1..16};
 //                         e.g. 4k/B=4 -> 17.2us @s=1 vs 23.4us @s=2.
 //   max_pages == 65536  : split=4 beats split=1 by ~18-19% within adaptive
 //                         (s=1 41.8us vs s=4 33.7us); 4 CTAs * 16k chunk
 //                         keeps the per-CTA radix select small enough that
 //                         the merge cost is amortised by the parallel scan.
 //
 // forced_splits overrides this for benchmarking.
 inline int pick_split_top30(int64_t /*eff_bs*/, int64_t max_pages) {
   if (max_pages > 32768) return 4;
   return 1;
 }

 // =============================================================================
 // Mid-K (K in {64, 128, 256, 512}) generalized SELECTK_SORTK kernel.
 //
 //   Same structure as TopK30_RandomSplit_Select32_Kernel, with LOCAL_K
 //   templated up to 512 and the local sort + final merge replaced with
 //   cub::BlockMergeSort variants sized by LOCAL_K and SPLITS*LOCAL_K
 //   respectively.
 //
 //   Per-CTA pipeline (mirrors K=30 path; only sizes change):
 //     Pass 1 — top-byte (bits [31:24]) histogram + suffix-sum-descending,
 //              find threshold bin where cumulative count crosses LOCAL_K.
 //     Pass 2 — strictly-above-threshold goes straight to candidate buffer;
 //              equal-to-threshold contributes to sub-bin (bits [23:16]) histogram.
 //     Pass 3 — sub-threshold then sub-equal candidates.
 //     Sort   — cub::BlockMergeSort over LOCAL_K candidates with NT_SORT=128
 //              and IPT_SORT = ceil(LOCAL_K, 128) / 128 (LOCAL_K=64 padded to 128).
 //
 //   Final merge (last CTA, SPLITS > 1):
 //     cub::BlockMergeSort over SPLITS * LOCAL_K candidates. Capped at 4096
 //     candidates total (NT=256, IPT=16) for register pressure.
 //
 //   Capacity policy (max SPLITS per LOCAL_K, candidates capped at 4096):
 //     LOCAL_K=64  -> SPLITS in {1, 2, 4, 8, 16, 32}  (max C=2048)
 //     LOCAL_K=128 -> SPLITS in {1, 2, 4, 8, 16, 32}  (max C=4096)
 //     LOCAL_K=256 -> SPLITS in {1, 2, 4, 8, 16}      (max C=4096)
 //     LOCAL_K=512 -> SPLITS in {1, 2, 4, 8}          (max C=4096)
 //
 //   For NT_SORT=128 we need LOCAL_K to be a multiple of 128 (the slot
 //   buffer is padded with (key=0, idx=-1) sentinels otherwise; a few
 //   threads sort dummy items, but the descending sort drops them past
 //   LOCAL_K and they are never read).
 // =============================================================================
 // SortNTConfig sizes the slot buffer + IPT for the local-stage sort.
 // We sort SLOTS_PADDED items with NT threads, IPT_SORT items per thread.
 // SLOTS_PADDED = max(LOCAL_K, NT) so all NT threads have at least one slot.
 template <int LOCAL_K, int NT>
 struct SortNTConfig {
   static constexpr int SLOTS_PADDED =
       (LOCAL_K >= NT) ? ((LOCAL_K + NT - 1) / NT) * NT : NT;
   static constexpr int NT_SORT      = NT;
   static constexpr int IPT_SORT     = SLOTS_PADDED / NT;
 };

 template <int CANDIDATES, int NT>
 struct MergeNTConfig {
   // Final merge runs in the same kernel block (last CTA), so NT_MERGE must
   // equal the kernel's NUM_THREADS or BlockMergeSort would deadlock.
   static constexpr int PADDED    = (CANDIDATES + NT - 1) / NT * NT;
   static constexpr int NT_MERGE  = NT;
   static constexpr int IPT_MERGE = PADDED / NT;
 };

 template <int SPLITS, int LOCAL_K, int NT_KERNEL>
 __device__ __forceinline__ void merge_block_sort_topk_midk(
     const uint32_t* __restrict__ keys_in,
     const int32_t*  __restrict__ idx_in,
     int32_t*        __restrict__ out_idx,
     int             final_k)
 {
   constexpr int kCandidates = SPLITS * LOCAL_K;
   constexpr int NT          = MergeNTConfig<kCandidates, NT_KERNEL>::NT_MERGE;
   constexpr int IPT         = MergeNTConfig<kCandidates, NT_KERNEL>::IPT_MERGE;
   using BlockSortT = cub::BlockMergeSort<uint32_t, NT, IPT, int32_t>;
   __shared__ typename BlockSortT::TempStorage block_merge_smem;

   const int tx = threadIdx.x;
   uint32_t bkeys[IPT];
   int32_t  bvals[IPT];
   #pragma unroll
   for (int k = 0; k < IPT; ++k) {
     const int rank = tx * IPT + k;
     bkeys[k] = (rank < kCandidates) ? keys_in[rank] : 0u;
     bvals[k] = (rank < kCandidates) ? idx_in [rank] : -1;
   }
   BlockSortT(block_merge_smem).Sort(bkeys, bvals, DescendingUint32{});
   #pragma unroll
   for (int k = 0; k < IPT; ++k) {
     const int rank = tx * IPT + k;
     if (rank < final_k && bvals[k] >= 0) out_idx[rank] = bvals[k];
   }
 }

 template <typename ScoreT, int MODE, int LOCAL_K, int SPLITS, int NUM_THREADS, int PARTITION>
 __global__ __launch_bounds__(NUM_THREADS)
 void TopKMidK_RandomSplit_SelectK_Kernel(
     const ScoreT* __restrict__ score,
     const int*    __restrict__ dense_kv_indptr,
     const int*    __restrict__ sparse_kv_indptr,
     const int*    __restrict__ dense_kv_indices,
     int*          __restrict__ sparse_kv_indices,
     uint32_t*     __restrict__ partial_keys,
     int32_t*      __restrict__ partial_indices,
     int32_t*      __restrict__ done_counter,
     const int     topk_val,
     const int     reserved_bos,
     const int     reserved_eos,
     const float   mapping_power)
 {
   constexpr int kRadix      = 256;
   constexpr int SLOTS_PADDED = SortNTConfig<LOCAL_K, NUM_THREADS>::SLOTS_PADDED;
   constexpr int NT_SORT      = SortNTConfig<LOCAL_K, NUM_THREADS>::NT_SORT;
   constexpr int IPT_SORT     = SortNTConfig<LOCAL_K, NUM_THREADS>::IPT_SORT;
   // cub::BlockMergeSort calls __syncthreads() internally — every thread in
   // the block must enter the sort branch, so NT_SORT must equal NUM_THREADS.
   static_assert(NT_SORT == NUM_THREADS,
                 "NT_SORT must equal NUM_THREADS or BlockMergeSort deadlocks");

   alignas(128) __shared__ int s_hist_buf[2][kRadix + 128];
   __shared__ int      s_above_count;
   __shared__ int      s_thresh_above_count;
   __shared__ int      s_thresh_at_count;
   __shared__ int      s_threshold_bin;
   __shared__ int      s_last_remain;
   __shared__ int      s_sub_threshold_bin;
   __shared__ int      s_sub_last_remain;
   __shared__ int      s_strictly_above_sub;
   __shared__ uint32_t s_top_keys[SLOTS_PADDED];
   __shared__ int32_t  s_top_idx [SLOTS_PADDED];
   __shared__ int      s_is_last;

   using LocalSortT = cub::BlockMergeSort<uint32_t, NT_SORT, IPT_SORT, int32_t>;
   __shared__ typename LocalSortT::TempStorage local_sort_smem;

   const int b  = blockIdx.x;
   const int n  = blockIdx.y;
   const int tx = threadIdx.x;

   const int row_start = dense_kv_indptr[b] + reserved_bos;
   const int row_end   = dense_kv_indptr[b + 1] - reserved_eos;
   const int row_len   = max(0, row_end - row_start);

   const int group_begin = (static_cast<int64_t>(row_len) * n)         / SPLITS;
   const int group_end   = (static_cast<int64_t>(row_len) * (n + 1))   / SPLITS;
   const int group_len   = group_end - group_begin;

   const bool     row_is_pow2 = is_pow2(row_len);
   const uint32_t n_mask      = row_is_pow2 ? static_cast<uint32_t>(row_len - 1) : 0u;
   const uint32_t b_off       =
       static_cast<uint32_t>(b) * kPermuteSeedB + kPermuteOffset;
   const ScoreT*  row_scores  = score            + row_start;
   const int*     row_idxmap  = dense_kv_indices + row_start;

   // ---- Init shared state. Strided over padding so any NT works. ----
   for (int i = tx; i < kRadix + 128; i += NUM_THREADS) {
     s_hist_buf[0][i] = 0;
     s_hist_buf[1][i] = 0;
   }
   if (tx == 0) {
     s_above_count        = 0;
     s_thresh_above_count = 0;
     s_thresh_at_count    = 0;
     s_threshold_bin      = -1;
     s_last_remain        = 0;
     s_sub_threshold_bin  = -1;
     s_sub_last_remain    = 0;
     s_strictly_above_sub = 0;
     s_is_last            = 0;
   }
   for (int i = tx; i < SLOTS_PADDED; i += NUM_THREADS) {
     s_top_keys[i] = 0u;
     s_top_idx [i] = -1;
   }
   __syncthreads();

   // Empty-row early exit (preserves merge barrier for SPLITS>1).
   if (row_len <= 0 || group_len <= 0) {
     if constexpr (SPLITS > 1) {
       const int64_t part_off =
           (static_cast<int64_t>(b) * SPLITS + n) * LOCAL_K;
       for (int i = tx; i < LOCAL_K; i += NUM_THREADS) {
         partial_keys   [part_off + i] = 0u;
         partial_indices[part_off + i] = -1;
       }
       __threadfence();
       __syncthreads();
       if (tx == 0) {
         const int old = ::atomicAdd(&done_counter[b], 1);
         s_is_last = (old == SPLITS - 1) ? 1 : 0;
         if (s_is_last) done_counter[b] = 0;  // self-reset for next launch
       }
       __syncthreads();
       if (s_is_last == 0) return;
       __threadfence();
       __syncthreads();
       const int64_t row_off = static_cast<int64_t>(b) * SPLITS * LOCAL_K;
       merge_block_sort_topk_midk<SPLITS, LOCAL_K, NUM_THREADS>(
           partial_keys + row_off, partial_indices + row_off,
           sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos,
           topk_val);
     }
     return;
   }

   auto run_cumsum_strided = [&]() {
     #pragma unroll
     for (int i = 0; i < 8; ++i) {
       const int j = 1 << i;
       const int k = i & 1;
       for (int idx = tx; idx < kRadix; idx += NUM_THREADS) {
         int v = s_hist_buf[k][idx];
         if (idx + j < kRadix) v += s_hist_buf[k][idx + j];
         s_hist_buf[k ^ 1][idx] = v;
       }
       __syncthreads();
     }
   };

   // ============== Pass 1: top-byte histogram. ==============
   for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
     int pos;
     if (row_is_pow2) {
       pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
     } else {
       pos = group_begin + local_rank;
     }
     const float    raw      = vortex_to_float(row_scores[pos]);
     const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
     const uint32_t key      = convert_to_uint32(remapped);
     const int      bin      = static_cast<int>(key >> 24);
     ::atomicAdd(&s_hist_buf[0][bin], 1);
   }
   __syncthreads();

   run_cumsum_strided();
   const int total_items = s_hist_buf[0][0];

   for (int bin = tx; bin < kRadix; bin += NUM_THREADS) {
     const int total_at_or_above = s_hist_buf[0][bin];
     const int strictly_above    = (bin + 1 < kRadix) ? s_hist_buf[0][bin + 1] : 0;
     if (total_at_or_above >= LOCAL_K && strictly_above < LOCAL_K) {
       s_threshold_bin = bin;
       s_last_remain   = LOCAL_K - strictly_above;
     }
   }
   __syncthreads();

   if (total_items <= LOCAL_K) {
     // Few-elements path: collect everything, pad rest.
     for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
       int pos;
       if (row_is_pow2) pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
       else             pos = group_begin + local_rank;
       const float    raw      = vortex_to_float(row_scores[pos]);
       const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       const uint32_t key      = convert_to_uint32(remapped);
       const int slot = ::atomicAdd(&s_above_count, 1);
       if (slot < LOCAL_K) {
         s_top_keys[slot] = key;
         s_top_idx [slot] = row_idxmap[pos];
       }
     }
     __syncthreads();
   } else {
     const int threshold_bin = s_threshold_bin;

     for (int i = tx; i < kRadix + 128; i += NUM_THREADS) {
       s_hist_buf[0][i] = 0;
       s_hist_buf[1][i] = 0;
     }
     __syncthreads();

     // ============== Pass 2: gather above + sub-hist. ==============
     for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
       int pos;
       if (row_is_pow2) pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
       else             pos = group_begin + local_rank;
       const float    raw      = vortex_to_float(row_scores[pos]);
       const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       const uint32_t key      = convert_to_uint32(remapped);
       const int      bin      = static_cast<int>(key >> 24);
       if (bin > threshold_bin) {
         const int slot = ::atomicAdd(&s_above_count, 1);
         if (slot < LOCAL_K) {
           s_top_keys[slot] = key;
           s_top_idx [slot] = row_idxmap[pos];
         }
       } else if (bin == threshold_bin) {
         const int sub_bin = static_cast<int>((key >> 16) & 0xFF);
         ::atomicAdd(&s_hist_buf[0][sub_bin], 1);
       }
     }
     __syncthreads();

     run_cumsum_strided();

     const int last_remain = s_last_remain;
     for (int bin = tx; bin < kRadix; bin += NUM_THREADS) {
       const int total_at_or_above = s_hist_buf[0][bin];
       const int strictly_above    = (bin + 1 < kRadix) ? s_hist_buf[0][bin + 1] : 0;
       if (total_at_or_above >= last_remain && strictly_above < last_remain) {
         s_sub_threshold_bin  = bin;
         s_sub_last_remain    = last_remain - strictly_above;
         s_strictly_above_sub = strictly_above;
       }
     }
     __syncthreads();

     const int sub_threshold_bin     = s_sub_threshold_bin;
     const int sub_last_remain       = s_sub_last_remain;
     const int strictly_above_sub_bn = s_strictly_above_sub;
     const int above_base            = s_above_count;

     // ============== Pass 3: sub-above + sub-at. ==============
     for (int local_rank = tx; local_rank < group_len; local_rank += NUM_THREADS) {
       int pos;
       if (row_is_pow2) pos = compute_pos<PARTITION, SPLITS>(local_rank, row_len, n, b_off, n_mask);
       else             pos = group_begin + local_rank;
       const float    raw      = vortex_to_float(row_scores[pos]);
       const float    remapped = apply_transform_tmpl<MODE>(raw, mapping_power);
       const uint32_t key      = convert_to_uint32(remapped);
       const int      bin      = static_cast<int>(key >> 24);
       if (bin == threshold_bin) {
         const int sub_bin = static_cast<int>((key >> 16) & 0xFF);
         if (sub_bin > sub_threshold_bin) {
           const int rel  = ::atomicAdd(&s_thresh_above_count, 1);
           const int slot = above_base + rel;
           if (slot < LOCAL_K) {
             s_top_keys[slot] = key;
             s_top_idx [slot] = row_idxmap[pos];
           }
         } else if (sub_bin == sub_threshold_bin) {
           const int rel = ::atomicAdd(&s_thresh_at_count, 1);
           if (rel < sub_last_remain) {
             const int slot = above_base + strictly_above_sub_bn + rel;
             if (slot < LOCAL_K) {
               s_top_keys[slot] = key;
               s_top_idx [slot] = row_idxmap[pos];
             }
           }
         }
       }
     }
     __syncthreads();
   }

   // ============== Sort SLOTS_PADDED candidates with cub::BlockMergeSort. ==============
   // The first LOCAL_K slots may have real data; padded slots have (0u, -1).
   // Sort uses NT_SORT threads; only those threads load/store sort items.
   if (tx < NT_SORT) {
     uint32_t kk[IPT_SORT];
     int32_t  vv[IPT_SORT];
     #pragma unroll
     for (int k = 0; k < IPT_SORT; ++k) {
       const int slot = tx * IPT_SORT + k;
       kk[k] = s_top_keys[slot];
       vv[k] = s_top_idx [slot];
     }
     LocalSortT(local_sort_smem).Sort(kk, vv, DescendingUint32{});
     #pragma unroll
     for (int k = 0; k < IPT_SORT; ++k) {
       const int slot = tx * IPT_SORT + k;
       s_top_keys[slot] = kk[k];
       s_top_idx [slot] = vv[k];
     }
   }
   __syncthreads();

   // ============== SPLITS == 1: direct write to sparse_kv_indices. ==============
   if constexpr (SPLITS == 1) {
     int32_t* out_idx = sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos;
     for (int rank = tx; rank < topk_val; rank += NUM_THREADS) {
       out_idx[rank] = s_top_idx[rank];
     }
     return;
   }

   // ============== SPLITS > 1: workspace, last-CTA barrier, merge. ==============
   const int64_t part_off = (static_cast<int64_t>(b) * SPLITS + n) * LOCAL_K;
   for (int i = tx; i < LOCAL_K; i += NUM_THREADS) {
     partial_keys   [part_off + i] = s_top_keys[i];
     partial_indices[part_off + i] = s_top_idx [i];
   }

   __threadfence();
   __syncthreads();
   if (tx == 0) {
     const int old = ::atomicAdd(&done_counter[b], 1);
     s_is_last = (old == SPLITS - 1) ? 1 : 0;
     if (s_is_last) done_counter[b] = 0;  // self-reset for next launch
   }
   __syncthreads();
   if (s_is_last == 0) return;
   __threadfence();
   __syncthreads();

   const int64_t row_off = static_cast<int64_t>(b) * SPLITS * LOCAL_K;
   merge_block_sort_topk_midk<SPLITS, LOCAL_K, NUM_THREADS>(
       partial_keys + row_off, partial_indices + row_off,
       sparse_kv_indices + sparse_kv_indptr[b] + reserved_bos,
       topk_val);
 }

 // Mid-K capacity policy. Returns true iff (LOCAL_K, SPLITS) is supported.
 inline bool midk_split_supported(int local_k, int splits) {
   const int candidates = local_k * splits;
   if (candidates > 4096) return false;
   if (splits != 1 && splits != 2 && splits != 4 && splits != 8 &&
       splits != 16 && splits != 32) return false;
   return true;
 }

 // Pick LOCAL_K from K. We use the smallest power-of-two LOCAL_K >= K.
 inline int midk_local_k_from_topk(int topk_val) {
   if (topk_val <= 64)  return 64;
   if (topk_val <= 128) return 128;
   if (topk_val <= 256) return 256;
   if (topk_val <= 512) return 512;
   return -1;  // unsupported
 }

 }  // namespace
 
 #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
 
 // =============================================================================
 // Workspace API: zero hot-path at::empty allocations.
 //
 //   topk_val >= 1024  → forwards to topk_output_sglang_fused without
 //                       touching workspace tensors or done_counter.
 //   topk_val   <= 32  → uses the K=30 random-split parallel path with
 //                       forced_splits (if > 0) or pick_split_top30().
 //   else              → also forwards to fused (no specialised path here).
 //
 // partial_keys / partial_indices must each have at least
 //   eff_batch_size * SPLITS * kLocalK_Top30 = eff_batch_size * SPLITS * 32
 // int32 elements. done_counter must have at least eff_batch_size int32
 // elements; it is cleared with cudaMemsetAsync inside this call before the
 // parallel kernel launches (and is NOT touched on the fused-fallback path).
 //
 // forced_splits encoding:
 //   <=  0 : use heuristic pick_split_top30().
 //      1 : single-CTA local sort path (for benchmarking).
 //   2/4/8/16/32 : forced parallel split.
 //   anything else : TORCH_CHECK failure.
 // =============================================================================
 void topk_output_adaptive_workspace(
     const at::Tensor& x,
     const at::Tensor& dense_kv_indptr,
     const at::Tensor& sparse_kv_indptr,
     const at::Tensor& dense_kv_indices,
     at::Tensor&       sparse_kv_indices,
     at::Tensor&       partial_keys,
     at::Tensor&       partial_indices,
     at::Tensor&       done_counter,
     const int64_t     eff_batch_size,
     const int64_t     topk_val,
     const int64_t     reserved_bos,
     const int64_t     reserved_eos,
     const int64_t     max_num_pages,
     const int64_t     mapping_mode,
     const double      mapping_power,
     const int64_t     forced_splits,
     const int64_t     forced_partition,
     const int64_t     local_mode)
 {
   // ============== Fused fallback (no workspace touch) ==============
   // K >= 1024: 32k -> 2048 lives here. Direct delegate. NO workspace check,
   // NO memset, NO split kernel launch — this is the near-zero-overhead
   // hot fast-path required for the K=2048 workload.
   //
   // K in (32, 1024) also routes here: those Ks have no specialised
   // adaptive kernel and the fused baseline is the right path. NOTE: for
   // K <= 32 (the K=30 path) we never come back to fused below — every
   // adaptive sub-path stays on the split kernel.
   if (topk_val >= kFusedFallbackTopK || topk_val > kMaxFinalK_Top30) {
     topk_output_sglang_fused(
         x, dense_kv_indptr, sparse_kv_indptr,
         dense_kv_indices, sparse_kv_indices,
         eff_batch_size, topk_val,
         reserved_bos, reserved_eos, max_num_pages,
         mapping_mode, mapping_power, std::nullopt, std::nullopt);
     return;
   }

   // ============== K <= 32 adaptive path (no fused fallback below) ==============

   CHECK_CUDA(x);
   CHECK_CUDA(dense_kv_indptr);
   CHECK_CUDA(sparse_kv_indptr);
   CHECK_CUDA(dense_kv_indices);
   CHECK_CUDA(sparse_kv_indices);

   TORCH_CHECK(topk_val > 0, "topk_val must be > 0");
   TORCH_CHECK(eff_batch_size >= 1, "eff_batch_size must be >= 1");
   TORCH_CHECK(max_num_pages >= 1, "max_num_pages must be >= 1");

   // local_mode validation. -1 (or any negative) defaults to SELECT32_SORT32,
   // which is the production mode (no NT*IPT capacity ceiling, supports the
   // full pages={4096,8192,16384,32768} x splits={1..32} matrix).
   int local_mode_int = static_cast<int>(local_mode);
   if (local_mode_int < 0) local_mode_int = LOCAL_SELECT32_SORT32;
   TORCH_CHECK(local_mode_int == LOCAL_BLOCK_FULL_SORT ||
               local_mode_int == LOCAL_SELECT32_SORT32,
               "local_mode must be 0 (BLOCK_FULL_SORT) or 1 (SELECT32_SORT32), got ",
               local_mode_int);
 
   TORCH_CHECK(
       mapping_mode == MAPPING_NONE         ||
       mapping_mode == MAPPING_POWER        ||
       mapping_mode == MAPPING_LOG          ||
       mapping_mode == MAPPING_ASINH        ||
       mapping_mode == MAPPING_LOG1P        ||
       mapping_mode == MAPPING_TRUNC8       ||
       mapping_mode == MAPPING_ERF          ||
       mapping_mode == MAPPING_TANH         ||
       mapping_mode == MAPPING_SUBTRACT     ||
       mapping_mode == MAPPING_EXP_STRETCH  ||
       mapping_mode == MAPPING_SHIFT_POW2   ||
       mapping_mode == MAPPING_SHIFT_POW3   ||
       mapping_mode == MAPPING_LINEAR_STEEP ||
       mapping_mode == MAPPING_HALF_SQUARE  ||
       mapping_mode == MAPPING_HALF_CUBE,
       "topk_output_adaptive_workspace: mapping_mode=", mapping_mode,
       " not supported.");
 
   // Resolve split count. K=30 NEVER falls back to fused: split=1 means
   // single-CTA adaptive kernel, not the fused baseline.
   int split;
   if (forced_splits > 0) {
     split = static_cast<int>(forced_splits);
     TORCH_CHECK(split == 1 || split == 2 || split == 4 || split == 8 ||
                 split == 16 || split == 32,
                 "forced_splits must be one of {1,2,4,8,16,32}, got ", split);
   } else {
     split = pick_split_top30(eff_batch_size, max_num_pages);
   }
 
   // Resolve partition mode.
   int partition;
   if (forced_partition >= 0) {
     partition = static_cast<int>(forced_partition);
     TORCH_CHECK(partition == PART_AFFINE_RANDOM   ||
                 partition == PART_CONTIGUOUS      ||
                 partition == PART_STRIDED         ||
                 partition == PART_TILE_RANDOM_128 ||
                 partition == PART_TILE_RANDOM_256,
                 "forced_partition must be 0=affine,1=contiguous,2=strided,"
                 "3=tile_random_128,4=tile_random_256");
   } else {
     partition = default_partition(split, max_num_pages);
   }
 
   // Capacity check applies ONLY to BLOCK_FULL_SORT, which uses
   // cub::BlockRadixSort and is bounded by NT*IPT static-smem footprint.
   // SELECT32_SORT32 has no such ceiling (its inner loops are strided).
   //
   // K=30 must NEVER silently fall back to fused — if BLOCK_FULL_SORT can't
   // fit the chunk, we fail loudly so the caller picks a finer split or
   // switches to SELECT32_SORT32.
   if (local_mode_int == LOCAL_BLOCK_FULL_SORT) {
     const int chunk_max = static_cast<int>((max_num_pages + split - 1) / split);
     const int cap       = split_capacity(split);
     TORCH_CHECK(cap >= chunk_max,
                 "topk_output_adaptive_workspace: BLOCK_FULL_SORT split=", split,
                 " has NT*IPT=", cap,
                 " < required chunk_max=", chunk_max,
                 " (max_num_pages=", max_num_pages,
                 "). Use SELECT32_SORT32 (local_mode=1) or a finer split.");
   }
 
   // From here we enter the parallel path. The split=1 forced case still
   // reads partial_keys/partial_indices/done_counter args but does NOT
   // touch them — we accept any tensor of the right dtype.
   CHECK_CUDA(partial_keys);
   CHECK_CUDA(partial_indices);
   CHECK_CUDA(done_counter);
   TORCH_CHECK(partial_keys.dtype()    == at::kInt,
               "partial_keys must be int32 (uint32 reinterpreted)");
   TORCH_CHECK(partial_indices.dtype() == at::kInt, "partial_indices must be int32");
   TORCH_CHECK(done_counter.dtype()    == at::kInt, "done_counter must be int32");
 
   if (split > 1) {
     TORCH_CHECK(done_counter.numel() >= eff_batch_size,
                 "done_counter[", done_counter.numel(),
                 "] too small for eff_batch_size=", eff_batch_size);
     const int64_t need = eff_batch_size * static_cast<int64_t>(split) * kLocalK_Top30;
     TORCH_CHECK(partial_keys.numel()    >= need,
                 "partial_keys too small: ", partial_keys.numel(), " < ", need);
     TORCH_CHECK(partial_indices.numel() >= need,
                 "partial_indices too small: ", partial_indices.numel(), " < ", need);
   }
 
   cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
   const float  mp     = static_cast<float>(mapping_power);

   // No cudaMemsetAsync(done_counter) — the kernel self-resets done_counter[b]
   // = 0 from the last CTA's tx==0 thread, so subsequent launches see it
   // already zero. Saves ~1-2 µs of CPU launch overhead per call. Caller
   // contract: done_counter must be zero-initialized once at workspace
   // allocation (at::zeros) and not touched by anyone else on this stream.

   uint32_t* part_keys_ptr =
       reinterpret_cast<uint32_t*>(partial_keys.data_ptr<int32_t>());
   int32_t*  part_idx_ptr  = partial_indices.data_ptr<int32_t>();
   int32_t*  done_ptr      = done_counter.data_ptr<int32_t>();
 
   dim3 grid(static_cast<unsigned>(eff_batch_size),
             static_cast<unsigned>(split));
 
   // ---- BLOCK_FULL_SORT macro chain (TopK30_RandomSplit_Parallel_Kernel) ----
   #define LAUNCH_TOP30_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT, PART) \
     do {                                                                           \
       auto* fn = &TopK30_RandomSplit_Parallel_Kernel<                              \
           DTYPE, MODE_VAL, SPLITS_VAL, NT, IPT, PART>;                             \
       fn<<<grid, dim3(NT), 0, stream>>>(                                           \
           PTR_EXPR,                                                                \
           dense_kv_indptr.data_ptr<int32_t>(),                                     \
           sparse_kv_indptr.data_ptr<int32_t>(),                                    \
           dense_kv_indices.data_ptr<int32_t>(),                                    \
           sparse_kv_indices.data_ptr<int32_t>(),                                   \
           part_keys_ptr, part_idx_ptr, done_ptr,                                   \
           static_cast<int>(topk_val),                                              \
           static_cast<int>(reserved_bos),                                          \
           static_cast<int>(reserved_eos),                                          \
           mp);                                                                     \
     } while (0)

   #define DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT)      \
     do {                                                                           \
       switch (partition) {                                                         \
         case PART_AFFINE_RANDOM:                                                   \
           LAUNCH_TOP30_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT, PART_AFFINE_RANDOM); break; \
         case PART_CONTIGUOUS:                                                      \
           LAUNCH_TOP30_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT, PART_CONTIGUOUS); break; \
         case PART_STRIDED:                                                         \
           LAUNCH_TOP30_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT, PART_STRIDED); break; \
         case PART_TILE_RANDOM_128:                                                 \
           LAUNCH_TOP30_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT, PART_TILE_RANDOM_128); break; \
         case PART_TILE_RANDOM_256:                                                 \
           LAUNCH_TOP30_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, IPT, PART_TILE_RANDOM_256); break; \
         default: TORCH_CHECK(false, "unreachable partition mode");                 \
       }                                                                            \
     } while (0)

   #define DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MODE_VAL)                          \
     do {                                                                           \
       switch (split) {                                                             \
         case 1:  DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL,  1, kCfg1.num_threads,  kCfg1.items_per_thread);  break; \
         case 2:  DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL,  2, kCfg2.num_threads,  kCfg2.items_per_thread);  break; \
         case 4:  DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL,  4, kCfg4.num_threads,  kCfg4.items_per_thread);  break; \
         case 8:  DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL,  8, kCfg8.num_threads,  kCfg8.items_per_thread);  break; \
         case 16: DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, 16, kCfg16.num_threads, kCfg16.items_per_thread); break; \
         case 32: DISPATCH_PART_BLOCK(DTYPE, PTR_EXPR, MODE_VAL, 32, kCfg32.num_threads, kCfg32.items_per_thread); break; \
         default: TORCH_CHECK(false, "unsupported split=", split);                  \
       }                                                                            \
     } while (0)

   // ---- SELECT32_SORT32 macro chain (TopK30_RandomSplit_Select32_Kernel) ----
   #define LAUNCH_TOP30_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, PART)     \
     do {                                                                           \
       auto* fn = &TopK30_RandomSplit_Select32_Kernel<                              \
           DTYPE, MODE_VAL, SPLITS_VAL, NT, PART>;                                  \
       fn<<<grid, dim3(NT), 0, stream>>>(                                           \
           PTR_EXPR,                                                                \
           dense_kv_indptr.data_ptr<int32_t>(),                                     \
           sparse_kv_indptr.data_ptr<int32_t>(),                                    \
           dense_kv_indices.data_ptr<int32_t>(),                                    \
           sparse_kv_indices.data_ptr<int32_t>(),                                   \
           part_keys_ptr, part_idx_ptr, done_ptr,                                   \
           static_cast<int>(topk_val),                                              \
           static_cast<int>(reserved_bos),                                          \
           static_cast<int>(reserved_eos),                                          \
           mp);                                                                     \
     } while (0)

   #define DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT)          \
     do {                                                                           \
       switch (partition) {                                                         \
         case PART_AFFINE_RANDOM:                                                   \
           LAUNCH_TOP30_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, PART_AFFINE_RANDOM); break; \
         case PART_CONTIGUOUS:                                                      \
           LAUNCH_TOP30_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, PART_CONTIGUOUS); break; \
         case PART_STRIDED:                                                         \
           LAUNCH_TOP30_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, PART_STRIDED); break; \
         case PART_TILE_RANDOM_128:                                                 \
           LAUNCH_TOP30_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, PART_TILE_RANDOM_128); break; \
         case PART_TILE_RANDOM_256:                                                 \
           LAUNCH_TOP30_SELECT(DTYPE, PTR_EXPR, MODE_VAL, SPLITS_VAL, NT, PART_TILE_RANDOM_256); break; \
         default: TORCH_CHECK(false, "unreachable partition mode");                 \
       }                                                                            \
     } while (0)

   #define DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MODE_VAL)                         \
     do {                                                                           \
       switch (split) {                                                             \
         case 1:  DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL,  1, kSelCfg1.num_threads);  break; \
         case 2:  DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL,  2, kSelCfg2.num_threads);  break; \
         case 4:  DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL,  4, kSelCfg4.num_threads);  break; \
         case 8:  DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL,  8, kSelCfg8.num_threads);  break; \
         case 16: DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL, 16, kSelCfg16.num_threads); break; \
         case 32: DISPATCH_PART_SELECT(DTYPE, PTR_EXPR, MODE_VAL, 32, kSelCfg32.num_threads); break; \
         default: TORCH_CHECK(false, "unsupported split=", split);                  \
       }                                                                            \
     } while (0)

   // Top-level: choose the local-mode chain, then mapping_mode → split → partition.
   // MAPPING_TRUNC8 shares its semantics with MAPPING_NONE (identity transform).
   // Routing both to MAPPING_NONE saves one template instantiation per chain.
   #define DISPATCH_MODE(DTYPE, PTR_EXPR)                                           \
     do {                                                                           \
       if (local_mode_int == LOCAL_SELECT32_SORT32) {                               \
         switch (mapping_mode) {                                                    \
           case MAPPING_NONE:                                                       \
           case MAPPING_TRUNC8:        DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_NONE);         break; \
           case MAPPING_POWER:         DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_POWER);        break; \
           case MAPPING_LOG:           DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_LOG);          break; \
           case MAPPING_ASINH:         DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_ASINH);        break; \
           case MAPPING_LOG1P:         DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_LOG1P);        break; \
           case MAPPING_ERF:           DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_ERF);          break; \
           case MAPPING_TANH:          DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_TANH);         break; \
           case MAPPING_SUBTRACT:      DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_SUBTRACT);     break; \
           case MAPPING_EXP_STRETCH:   DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_EXP_STRETCH);  break; \
           case MAPPING_SHIFT_POW2:    DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW2);   break; \
           case MAPPING_SHIFT_POW3:    DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW3);   break; \
           case MAPPING_LINEAR_STEEP:  DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_LINEAR_STEEP); break; \
           case MAPPING_HALF_SQUARE:   DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_HALF_SQUARE);  break; \
           case MAPPING_HALF_CUBE:     DISPATCH_SPLIT_SELECT(DTYPE, PTR_EXPR, MAPPING_HALF_CUBE);    break; \
           default: TORCH_CHECK(false, "unreachable mapping_mode");                 \
         }                                                                          \
       } else {                                                                     \
         switch (mapping_mode) {                                                    \
           case MAPPING_NONE:                                                       \
           case MAPPING_TRUNC8:        DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_NONE);         break; \
           case MAPPING_POWER:         DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_POWER);        break; \
           case MAPPING_LOG:           DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_LOG);          break; \
           case MAPPING_ASINH:         DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_ASINH);        break; \
           case MAPPING_LOG1P:         DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_LOG1P);        break; \
           case MAPPING_ERF:           DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_ERF);          break; \
           case MAPPING_TANH:          DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_TANH);         break; \
           case MAPPING_SUBTRACT:      DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_SUBTRACT);     break; \
           case MAPPING_EXP_STRETCH:   DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_EXP_STRETCH);  break; \
           case MAPPING_SHIFT_POW2:    DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW2);   break; \
           case MAPPING_SHIFT_POW3:    DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_SHIFT_POW3);   break; \
           case MAPPING_LINEAR_STEEP:  DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_LINEAR_STEEP); break; \
           case MAPPING_HALF_SQUARE:   DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_HALF_SQUARE);  break; \
           case MAPPING_HALF_CUBE:     DISPATCH_SPLIT_BLOCK(DTYPE, PTR_EXPR, MAPPING_HALF_CUBE);    break; \
           default: TORCH_CHECK(false, "unreachable mapping_mode");                 \
         }                                                                          \
       }                                                                            \
     } while (0)

   if (x.scalar_type() == at::ScalarType::BFloat16) {
     DISPATCH_MODE(__nv_bfloat16,
         reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
   } else if (x.scalar_type() == at::ScalarType::Float) {
     DISPATCH_MODE(float, x.data_ptr<float>());
   } else {
     TORCH_CHECK(false, "topk_output_adaptive_workspace: unsupported dtype ",
                 x.scalar_type());
   }

   #undef DISPATCH_MODE
   #undef DISPATCH_SPLIT_SELECT
   #undef DISPATCH_PART_SELECT
   #undef LAUNCH_TOP30_SELECT
   #undef DISPATCH_SPLIT_BLOCK
   #undef DISPATCH_PART_BLOCK
   #undef LAUNCH_TOP30_BLOCK
 
   const auto rc = cudaGetLastError();
   TORCH_CHECK(rc == cudaSuccess,
               "topk_output_adaptive_workspace launch failed: ",
               ::cudaGetErrorString(rc));
 }
 
 // =============================================================================
 // Legacy entry point — allocates workspace internally and forwards.
 //
 // NOTE: this path performs at::empty allocations and is therefore NOT a
 // reference for latency benchmarks. New callers should use
 // topk_output_adaptive_workspace with preallocated workspace.
 // =============================================================================
 void topk_output_adaptive(
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
     const double      mapping_power)
 {
   // Workspace big enough for the largest split this kernel may pick (32).
   constexpr int64_t kMaxSplit = 32;
   const int64_t ws_elems = eff_batch_size * kMaxSplit * kLocalK_Top30;
 
   auto opts_i32 = at::TensorOptions().device(x.device()).dtype(at::kInt);
   at::Tensor partial_keys    = at::empty({ws_elems},        opts_i32);
   at::Tensor partial_indices = at::empty({ws_elems},        opts_i32);
   at::Tensor done_counter    = at::empty({eff_batch_size},  opts_i32);
 
   topk_output_adaptive_workspace(
       x, dense_kv_indptr, sparse_kv_indptr, dense_kv_indices,
       sparse_kv_indices, partial_keys, partial_indices, done_counter,
       eff_batch_size, topk_val, reserved_bos, reserved_eos,
       max_num_pages, mapping_mode, mapping_power,
       /*forced_splits=*/-1,
       /*forced_partition=*/-1,
       /*local_mode=*/LOCAL_SELECT32_SORT32);
 }
 

// =============================================================================
// Mid-K (K in {64, 128, 256, 512}) adaptive split entry point.
//
// Separate from topk_output_adaptive_workspace so the K=30 production path
// stays untouched. workspace tensors must be sized for
//   eff_batch_size * SPLITS * LOCAL_K
// where LOCAL_K is the smallest power of two >= topk_val (max 512), and
// SPLITS is forced_splits if > 0, else 1.
//
// Dispatch contract:
//   topk_val < 64 or > 512 → TORCH_CHECK failure (use the K=30 path or fused).
//   forced_splits encoding:
//     <=  0 : default policy (currently split=1; sweep will inform a heuristic).
//     1/2/4/8/16/32 : forced split, must satisfy midk_split_supported().
// =============================================================================
void topk_output_adaptive_workspace_midk(
    const at::Tensor& x,
    const at::Tensor& dense_kv_indptr,
    const at::Tensor& sparse_kv_indptr,
    const at::Tensor& dense_kv_indices,
    at::Tensor&       sparse_kv_indices,
    at::Tensor&       partial_keys,
    at::Tensor&       partial_indices,
    at::Tensor&       done_counter,
    const int64_t     eff_batch_size,
    const int64_t     topk_val,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_num_pages,
    const int64_t     mapping_mode,
    const double      mapping_power,
    const int64_t     forced_splits)
{
  CHECK_CUDA(x);
  CHECK_CUDA(dense_kv_indptr);
  CHECK_CUDA(sparse_kv_indptr);
  CHECK_CUDA(dense_kv_indices);
  CHECK_CUDA(sparse_kv_indices);

  TORCH_CHECK(topk_val >= 64 && topk_val <= 512,
              "topk_output_adaptive_workspace_midk: topk_val=", topk_val,
              " out of range [64, 512]. Use topk_output_adaptive_workspace "
              "for K<=32 or topk_output_sglang_fused for K>512.");
  TORCH_CHECK(eff_batch_size >= 1, "eff_batch_size must be >= 1");
  TORCH_CHECK(max_num_pages >= 1, "max_num_pages must be >= 1");

  const int local_k = midk_local_k_from_topk(static_cast<int>(topk_val));
  TORCH_CHECK(local_k > 0, "unreachable: midk_local_k_from_topk failed for K=", topk_val);

  // Mid-K mappings: NONE / TRUNC8 only for now (kept template count low).
  // POWER/LOG/etc. are easy to add later once we measure their value.
  TORCH_CHECK(mapping_mode == MAPPING_NONE || mapping_mode == MAPPING_TRUNC8,
              "topk_output_adaptive_workspace_midk: mapping_mode=",
              mapping_mode, " not yet supported (use NONE or TRUNC8).");

  int split;
  if (forced_splits > 0) {
    split = static_cast<int>(forced_splits);
    TORCH_CHECK(midk_split_supported(local_k, split),
                "topk_output_adaptive_workspace_midk: split=", split,
                " not supported for LOCAL_K=", local_k,
                " (would need ", split * local_k, " merge candidates, max 4096).");
  } else {
    // Sweep-driven default. From bench_results/midk_best_adaptive_p50.csv:
    //
    //   pages <= 65536 : adaptive loses every cell vs fused on p50 — but a
    //                    user calling this entry point explicitly is asking
    //                    for adaptive anyway, so use split=1 (smallest gap).
    //   pages  > 65536 : fused unsupported (smem ceiling). Best splits:
    //                      K=64  → 16
    //                      K=128 → 16
    //                      K=256 → 2
    //                      K=512 → 4
    //
    // forced_splits > 0 still overrides this, e.g. for benchmarking.
    if (max_num_pages > 65536) {
      switch (local_k) {
        case 64:  split = 16; break;
        case 128: split = 16; break;
        case 256: split = 2;  break;
        case 512: split = 4;  break;
        default:  split = 1;
      }
    } else {
      split = 1;
    }
  }

  CHECK_CUDA(partial_keys);
  CHECK_CUDA(partial_indices);
  CHECK_CUDA(done_counter);
  TORCH_CHECK(partial_keys.dtype()    == at::kInt, "partial_keys must be int32");
  TORCH_CHECK(partial_indices.dtype() == at::kInt, "partial_indices must be int32");
  TORCH_CHECK(done_counter.dtype()    == at::kInt, "done_counter must be int32");

  if (split > 1) {
    TORCH_CHECK(done_counter.numel() >= eff_batch_size,
                "done_counter[", done_counter.numel(),
                "] too small for eff_batch_size=", eff_batch_size);
    const int64_t need = eff_batch_size * static_cast<int64_t>(split) * local_k;
    TORCH_CHECK(partial_keys.numel()    >= need,
                "partial_keys too small: ", partial_keys.numel(), " < ", need);
    TORCH_CHECK(partial_indices.numel() >= need,
                "partial_indices too small: ", partial_indices.numel(), " < ", need);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const float  mp     = static_cast<float>(mapping_power);

  // No cudaMemsetAsync — kernel self-resets done_counter (see midk kernel
  // and dispatcher comment for topk_output_adaptive_workspace).

  uint32_t* part_keys_ptr =
      reinterpret_cast<uint32_t*>(partial_keys.data_ptr<int32_t>());
  int32_t*  part_idx_ptr  = partial_indices.data_ptr<int32_t>();
  int32_t*  done_ptr      = done_counter.data_ptr<int32_t>();

  dim3 grid(static_cast<unsigned>(eff_batch_size),
            static_cast<unsigned>(split));

  // NT scales inversely with SPLITS so the per-CTA scan loop has roughly
  // the same iteration count regardless of split count. Mirror the K=30
  // kSelCfg ladder. With NT=128 at SPLITS=1, a 65k-page row would force
  // 512 iters/thread/pass — way slower than the ~64 iters fused achieves
  // with NT=1024 single-CTA. Match fused throughput at split=1.
  //
  //   SPLITS=1  : NT=1024  (chunk = full row)
  //   SPLITS=2  : NT=512
  //   SPLITS=4  : NT=256
  //   SPLITS=8  : NT=128
  //   SPLITS=16 : NT=128
  //   SPLITS=32 : NT=128

  #define LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL, SPLITS_VAL, NT_VAL)    \
    do {                                                                             \
      auto* fn = &TopKMidK_RandomSplit_SelectK_Kernel<                               \
          DTYPE, MODE_VAL, LOCAL_K_VAL, SPLITS_VAL, NT_VAL, PART_CONTIGUOUS>;        \
      fn<<<grid, dim3(NT_VAL), 0, stream>>>(                                         \
          PTR_EXPR,                                                                  \
          dense_kv_indptr.data_ptr<int32_t>(),                                       \
          sparse_kv_indptr.data_ptr<int32_t>(),                                      \
          dense_kv_indices.data_ptr<int32_t>(),                                      \
          sparse_kv_indices.data_ptr<int32_t>(),                                     \
          part_keys_ptr, part_idx_ptr, done_ptr,                                     \
          static_cast<int>(topk_val),                                                \
          static_cast<int>(reserved_bos),                                            \
          static_cast<int>(reserved_eos),                                            \
          mp);                                                                       \
    } while (0)

  #define DISPATCH_SPLIT_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL)                \
    do {                                                                             \
      switch (split) {                                                               \
        case  1: LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL,  1, 1024); break; \
        case  2: LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL,  2,  512); break; \
        case  4: LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL,  4,  256); break; \
        case  8: LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL,  8,  128); break; \
        case 16:                                                                     \
          if constexpr ((LOCAL_K_VAL) * 16 <= 4096)                                  \
            LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL, 16, 128);            \
          else                                                                       \
            TORCH_CHECK(false, "midk: split=16 unsupported for LOCAL_K=", LOCAL_K_VAL);\
          break;                                                                     \
        case 32:                                                                     \
          if constexpr ((LOCAL_K_VAL) * 32 <= 4096)                                  \
            LAUNCH_MIDK(DTYPE, PTR_EXPR, MODE_VAL, LOCAL_K_VAL, 32, 128);            \
          else                                                                       \
            TORCH_CHECK(false, "midk: split=32 unsupported for LOCAL_K=", LOCAL_K_VAL);\
          break;                                                                     \
        default: TORCH_CHECK(false, "midk: unsupported split=", split);              \
      }                                                                              \
    } while (0)

  #define DISPATCH_LK_MIDK(DTYPE, PTR_EXPR, MODE_VAL)                                \
    do {                                                                             \
      switch (local_k) {                                                             \
        case 64:  DISPATCH_SPLIT_MIDK(DTYPE, PTR_EXPR, MODE_VAL,  64); break;        \
        case 128: DISPATCH_SPLIT_MIDK(DTYPE, PTR_EXPR, MODE_VAL, 128); break;        \
        case 256: DISPATCH_SPLIT_MIDK(DTYPE, PTR_EXPR, MODE_VAL, 256); break;        \
        case 512: DISPATCH_SPLIT_MIDK(DTYPE, PTR_EXPR, MODE_VAL, 512); break;        \
        default: TORCH_CHECK(false, "midk: unreachable LOCAL_K=", local_k);          \
      }                                                                              \
    } while (0)

  #define DISPATCH_MIDK(DTYPE, PTR_EXPR)                                             \
    do {                                                                             \
      /* MAPPING_TRUNC8 aliases MAPPING_NONE; same template instantiation. */        \
      DISPATCH_LK_MIDK(DTYPE, PTR_EXPR, MAPPING_NONE);                               \
    } while (0)

  if (x.scalar_type() == at::ScalarType::BFloat16) {
    DISPATCH_MIDK(__nv_bfloat16,
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  } else if (x.scalar_type() == at::ScalarType::Float) {
    DISPATCH_MIDK(float, x.data_ptr<float>());
  } else {
    TORCH_CHECK(false, "topk_output_adaptive_workspace_midk: unsupported dtype ",
                x.scalar_type());
  }

  #undef DISPATCH_MIDK
  #undef DISPATCH_LK_MIDK
  #undef DISPATCH_SPLIT_MIDK
  #undef LAUNCH_MIDK

  const auto rc = cudaGetLastError();
  TORCH_CHECK(rc == cudaSuccess,
              "topk_output_adaptive_workspace_midk launch failed: ",
              ::cudaGetErrorString(rc));
}
