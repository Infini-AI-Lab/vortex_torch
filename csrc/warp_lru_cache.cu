#include "register.h"
#include "hash_functions.cuh"
#include "warp_mutex.cuh"
#include <cstdlib>

// Constants for warp-level operations
constexpr int kWarpSize = 32;
constexpr uint32_t kFullMask = 0xFFFFFFFFU;

// SetContext for warp-level operations
struct SetContextWarp {
  __device__ SetContextWarp(
      int32_t* cpu_to_gpu_map,
      int32_t* gpu_to_cpu_map,
      uint8_t* ages,
      WarpMutexSemaphoreImpl* mtx,
      bool* slot_bitmap,
      uint32_t set_id,
      uint32_t ways)
      : cpu_to_gpu_slot_map(cpu_to_gpu_map),
        gpu_to_cpu_page_map(gpu_to_cpu_map + set_id * ways),
        slot_ages(ages + set_id * ways),
        mutex(mtx + set_id),
        used_this_round(slot_bitmap ? slot_bitmap + set_id * ways : nullptr),
        set_base(set_id * ways),
        num_ways(ways)
  {}

  // Insert without eviction - only use empty slots (age==0)
  // Returns: gpu_slot (global slot index) on success, -1 if no empty slot available
  // is_owner: set to true if this was a new allocation (not cache hit)
  __device__ int InsertNoEvict(uint32_t lane_id, int32_t cpu_slot, int32_t MAX_PAGE_ID, bool& is_owner) {
    is_owner = false;

    // Step 1: Check for cache hit
    int32_t existing_gpu_slot = cpu_to_gpu_slot_map[cpu_slot];

    // Check if the existing_gpu_slot is in this set
    if (existing_gpu_slot >= set_base && existing_gpu_slot < set_base + num_ways) {
      // Cache hit in this set!
      int32_t lane_gpu_slot = set_base + lane_id;
      uint8_t lane_age = slot_ages[lane_id];

      // Check if this is the hit lane
      const unsigned hit_mask = __ballot_sync(kFullMask,
          lane_gpu_slot == existing_gpu_slot && lane_age != 0);

      if (hit_mask != 0) {
        int hit_lane = __ffs(static_cast<int>(hit_mask)) - 1;
        uint8_t hit_lane_age = __shfl_sync(kFullMask, lane_age, hit_lane);

        // Update ages
        if (lane_age > hit_lane_age && lane_age > 0) {
          lane_age -= 1;
        } else if (lane_id == hit_lane) {
          lane_age = num_ways;
          // Mark this slot as used this round
          used_this_round[hit_lane] = true;
        }
        slot_ages[lane_id] = lane_age;
        __syncwarp();
        return existing_gpu_slot;
      }
    }

    // Step 2: Try to find an empty slot (age==0)
    uint8_t lane_age = slot_ages[lane_id];
    bool lane_is_empty = (lane_age == 0);

    // Find empty slot
    const unsigned empty_mask = __ballot_sync(kFullMask, lane_is_empty);

    if (empty_mask == 0) {
      // No empty slots available
      return -1;
    }

    int insert_way = __ffs(static_cast<int>(empty_mask)) - 1;

    // Install new page in empty slot
    if (lane_id == insert_way) {
      gpu_to_cpu_page_map[insert_way] = cpu_slot;
      cpu_to_gpu_slot_map[cpu_slot] = set_base + insert_way;
      lane_age = num_ways;
      // Mark this slot as used this round
      used_this_round[insert_way] = true;
    } else if (lane_age > 1) {
      lane_age -= 1;
    }

    slot_ages[lane_id] = lane_age;
    __syncwarp();

    is_owner = true;
    return set_base + insert_way;
  }

  // Insert page with eviction (no cache hit check - only for pages that need eviction)
  // Returns: gpu_slot (global slot index) on success, -1 if allocation fails
  // is_owner: always set to true since this is a new allocation with eviction
  // evicted_cpu_page: the CPU page that was evicted from this slot (-1 if none)
  __device__ int InsertWithEvict(uint32_t lane_id, int32_t cpu_slot, int32_t MAX_PAGE_ID, bool& is_owner, int32_t& evicted_cpu_page) {
    // Find the slot with minimum age for eviction that hasn't been used this round
    uint8_t lane_age = slot_ages[lane_id];
    bool lane_not_used = !used_this_round[lane_id];

    // Find the slot with minimum age that hasn't been used this round
    int insert_way = -1;

    for (uint8_t target_age = 0; target_age <= num_ways && insert_way == -1; ++target_age) {
      const unsigned age_mask = __ballot_sync(kFullMask, lane_age == target_age && lane_not_used);
      if (age_mask != 0) {
        insert_way = __ffs(static_cast<int>(age_mask)) - 1;
        break;
      }
    }

    if (insert_way == -1) {
      // All slots have been used this round
      evicted_cpu_page = -1;
      return -1;
    }

    // Get the old CPU slot that will be evicted (if any)
    int32_t old_cpu_slot = gpu_to_cpu_page_map[insert_way];
    evicted_cpu_page = old_cpu_slot;  // Return the evicted page

    // Update ages and install new page
    if (lane_id == insert_way) {
      // Clear the old mapping if this slot had a previous page
      if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
        cpu_to_gpu_slot_map[old_cpu_slot] = -1;
      }
      // Install new page
      gpu_to_cpu_page_map[insert_way] = cpu_slot;
      cpu_to_gpu_slot_map[cpu_slot] = set_base + insert_way;
      lane_age = num_ways;
      // Mark this slot as used this round
      used_this_round[insert_way] = true;
    } else if (lane_age > 1) {
      lane_age -= 1;
    }

    slot_ages[lane_id] = lane_age;
    __syncwarp();

    is_owner = true;  // New allocation
    return set_base + insert_way;
  }

  __device__ void Lock(uint32_t lane_id) { mutex->Lock(lane_id); }
  __device__ void Unlock(uint32_t lane_id) { mutex->Unlock(lane_id); }

  // Members
  int32_t* cpu_to_gpu_slot_map;  // Global mapping [MAX_PAGE_ID]
  int32_t* gpu_to_cpu_page_map;  // Per-set mapping [num_ways]
  uint8_t* slot_ages;            // Per-set ages [num_ways]
  WarpMutexSemaphoreImpl* mutex; // Per-set mutex
  bool* used_this_round;         // Per-set bitmap [num_ways]
  uint32_t set_base;             // Base slot index for this set
  uint32_t num_ways;
};

// Kernel 1: Allocate without eviction (only use empty slots age==0)
__global__ void allocate_pages_lru_warp_no_evict_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    bool* __restrict__ needs_eviction_bitmap,
    int32_t N,
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t WAYS
) {
  const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_warp_id = global_thread_id / kWarpSize;
  const uint32_t lane_id = global_thread_id % kWarpSize;
  const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

  // Each warp processes batches of kWarpSize requests
  for (uint32_t batch_offset = global_warp_id * kWarpSize; batch_offset < N;
       batch_offset += num_warps * kWarpSize) {
    const uint32_t n_batch_keys = min(kWarpSize, N - batch_offset);

    // All lanes iterate through each request in the batch
    for (uint32_t i = 0; i < n_batch_keys; ++i) {
      const uint32_t key_idx = batch_offset + i;
      const int32_t cpu_slot = src_page_ids[key_idx];

      // Validate cpu_slot
      if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
        if (lane_id == 0) {
          owners_bitmap[key_idx] = false;
          dst_staging_slots[key_idx] = -1;
          needs_eviction_bitmap[key_idx] = false;
        }
        continue;
      }

      // Try primary hash only (hash_func_0)
      uint64_t h = hash_func_0((uint64_t)cpu_slot);
      uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

      SetContextWarp set_ctx(
          cpu_to_gpu_slot_map,
          gpu_to_cpu_page_map,
          slot_ages,
          set_mutexes,
          set_slot_used_bitmap,
          candidate_set,
          WAYS);

      set_ctx.Lock(lane_id);

      bool is_owner = false;
      int insert_gpu_slot = set_ctx.InsertNoEvict(lane_id, cpu_slot, MAX_PAGE_ID, is_owner);

      set_ctx.Unlock(lane_id);

      if (lane_id == 0) {
        if (insert_gpu_slot >= 0) {
          // Successfully allocated without eviction
          owners_bitmap[key_idx] = is_owner;
          dst_staging_slots[key_idx] = insert_gpu_slot;
          needs_eviction_bitmap[key_idx] = false;
        } else {
          // Failed - needs eviction in second kernel
          owners_bitmap[key_idx] = false;
          dst_staging_slots[key_idx] = -1;
          needs_eviction_bitmap[key_idx] = true;
        }
      }
      __syncwarp();
    }
  }
}

// Kernel 2: Allocate with eviction (only for pages that failed in kernel 1)
__global__ void allocate_pages_lru_warp_with_evict_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    const bool* __restrict__ needs_eviction_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    int32_t N,
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t WAYS,
    int32_t MAX_HASH_ATTEMPTS
) {
  const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t global_warp_id = global_thread_id / kWarpSize;
  const uint32_t lane_id = global_thread_id % kWarpSize;
  const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

  // Each warp processes batches of kWarpSize requests
  for (uint32_t batch_offset = global_warp_id * kWarpSize; batch_offset < N;
       batch_offset += num_warps * kWarpSize) {
    const uint32_t n_batch_keys = min(kWarpSize, N - batch_offset);

    // All lanes iterate through each request in the batch
    for (uint32_t i = 0; i < n_batch_keys; ++i) {
      const uint32_t key_idx = batch_offset + i;

      // Skip if this page doesn't need eviction
      if (!needs_eviction_bitmap[key_idx]) {
        __syncwarp();
        continue;
      }

      const int32_t cpu_slot = src_page_ids[key_idx];

      // Validate cpu_slot (should already be valid, but check anyway)
      if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
        if (lane_id == 0) {
          owners_bitmap[key_idx] = false;
          dst_staging_slots[key_idx] = -1;
        }
        __syncwarp();
        continue;
      }

      bool allocated = false;
      bool is_owner = false;
      int32_t final_gpu_slot = -1;
      int32_t final_evicted_page = -1;

      // Try different hash functions to find a set with space
      for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !allocated; ++hash_attempt) {
        uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
        uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

        SetContextWarp set_ctx(
            cpu_to_gpu_slot_map,
            gpu_to_cpu_page_map,
            slot_ages,
            set_mutexes,
            set_slot_used_bitmap,
            candidate_set,
            WAYS);

        set_ctx.Lock(lane_id);

        bool lane_is_owner = false;
        int32_t lane_evicted_page = -1;
        int insert_gpu_slot = set_ctx.InsertWithEvict(lane_id, cpu_slot, MAX_PAGE_ID, lane_is_owner, lane_evicted_page);

        if (insert_gpu_slot >= 0) {
          if (lane_id == 0) {
            is_owner = lane_is_owner;
            final_gpu_slot = insert_gpu_slot;
            final_evicted_page = lane_evicted_page;
          }
          allocated = true;
        }

        set_ctx.Unlock(lane_id);

        if (allocated) break;
      }

      if (lane_id == 0) {
        if (!allocated) {
          // All hash attempts failed - overflow
          atomicMax(overflow_flag, 1);
          owners_bitmap[key_idx] = false;
          dst_staging_slots[key_idx] = -1;
          evicted_cpu_pages[key_idx] = -1;
        } else {
          owners_bitmap[key_idx] = is_owner;
          dst_staging_slots[key_idx] = final_gpu_slot;
          evicted_cpu_pages[key_idx] = final_evicted_page;
        }
      }
      __syncwarp();
    }
  }
}

// Static variables to cache mutexes across calls
static WarpMutexSemaphoreImpl* cached_mutexes = nullptr;
static int32_t cached_num_sets = 0;

// Cleanup function to free cached resources on program exit
static void cleanup_warp_allocator_cache() {
    if (cached_mutexes != nullptr) {
        cudaFree(cached_mutexes);
        cached_mutexes = nullptr;
    }
    cached_num_sets = 0;
}

// CUDA graph compatible allocation - reads num_pages from sparse_indptr on GPU
// Kernel 1 (indptr version): Allocate without eviction
__global__ void allocate_pages_lru_warp_no_evict_indptr_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    bool* __restrict__ needs_eviction_bitmap,
    const int32_t* __restrict__ sparse_indptr,  // Read N from here
    int32_t indptr_last_idx,                     // Index to read N from
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t WAYS
) {
    // Read actual N from GPU memory (no CPU sync!)
    const int32_t N = sparse_indptr[indptr_last_idx];

    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSize;
    const uint32_t lane_id = global_thread_id % kWarpSize;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

    // Each warp processes batches of kWarpSize requests
    for (uint32_t batch_offset = global_warp_id * kWarpSize; batch_offset < N;
         batch_offset += num_warps * kWarpSize) {
        const uint32_t n_batch_keys = min(kWarpSize, N - (int)batch_offset);

        // All lanes iterate through each request in the batch
        for (uint32_t i = 0; i < n_batch_keys; ++i) {
            const uint32_t key_idx = batch_offset + i;
            const int32_t cpu_slot = src_page_ids[key_idx];

            // Validate cpu_slot
            if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                    needs_eviction_bitmap[key_idx] = false;
                }
                continue;
            }

            // Try primary hash only (hash_func_0)
            uint64_t h = hash_func_0((uint64_t)cpu_slot);
            uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

            SetContextWarp set_ctx(
                cpu_to_gpu_slot_map,
                gpu_to_cpu_page_map,
                slot_ages,
                set_mutexes,
                set_slot_used_bitmap,
                candidate_set,
                WAYS);

            set_ctx.Lock(lane_id);

            bool is_owner = false;
            int insert_gpu_slot = set_ctx.InsertNoEvict(lane_id, cpu_slot, MAX_PAGE_ID, is_owner);

            set_ctx.Unlock(lane_id);

            if (lane_id == 0) {
                if (insert_gpu_slot >= 0) {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = insert_gpu_slot;
                    needs_eviction_bitmap[key_idx] = false;
                } else {
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                    needs_eviction_bitmap[key_idx] = true;
                }
            }
            __syncwarp();
        }
    }
}

// Kernel 2 (indptr version): Allocate with eviction
__global__ void allocate_pages_lru_warp_with_evict_indptr_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    const bool* __restrict__ needs_eviction_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    const int32_t* __restrict__ sparse_indptr,
    int32_t indptr_last_idx,
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t WAYS,
    int32_t MAX_HASH_ATTEMPTS
) {
    // Read actual N from GPU memory (no CPU sync!)
    const int32_t N = sparse_indptr[indptr_last_idx];

    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSize;
    const uint32_t lane_id = global_thread_id % kWarpSize;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

    // Each warp processes batches of kWarpSize requests
    for (uint32_t batch_offset = global_warp_id * kWarpSize; batch_offset < N;
         batch_offset += num_warps * kWarpSize) {
        const uint32_t n_batch_keys = min(kWarpSize, N - (int)batch_offset);

        // All lanes iterate through each request in the batch
        for (uint32_t i = 0; i < n_batch_keys; ++i) {
            const uint32_t key_idx = batch_offset + i;

            // Skip if this page doesn't need eviction
            if (!needs_eviction_bitmap[key_idx]) {
                __syncwarp();
                continue;
            }

            const int32_t cpu_slot = src_page_ids[key_idx];

            // Validate cpu_slot
            if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                }
                __syncwarp();
                continue;
            }

            bool allocated = false;
            bool is_owner = false;
            int32_t final_gpu_slot = -1;
            int32_t final_evicted_page = -1;

            // Try different hash functions to find a set with space
            for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !allocated; ++hash_attempt) {
                uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
                uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

                SetContextWarp set_ctx(
                    cpu_to_gpu_slot_map,
                    gpu_to_cpu_page_map,
                    slot_ages,
                    set_mutexes,
                    set_slot_used_bitmap,
                    candidate_set,
                    WAYS);

                set_ctx.Lock(lane_id);

                bool lane_is_owner = false;
                int32_t lane_evicted_page = -1;
                int insert_gpu_slot = set_ctx.InsertWithEvict(lane_id, cpu_slot, MAX_PAGE_ID, lane_is_owner, lane_evicted_page);

                if (insert_gpu_slot >= 0) {
                    if (lane_id == 0) {
                        is_owner = lane_is_owner;
                        final_gpu_slot = insert_gpu_slot;
                        final_evicted_page = lane_evicted_page;
                    }
                    allocated = true;
                }

                set_ctx.Unlock(lane_id);

                if (allocated) break;
            }

            if (lane_id == 0) {
                if (!allocated) {
                    atomicMax(overflow_flag, 1);
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                    evicted_cpu_pages[key_idx] = -1;
                } else {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = final_gpu_slot;
                    evicted_cpu_pages[key_idx] = final_evicted_page;
                }
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// Combined allocation kernel - merges no-evict and with-evict stages
// This eliminates one kernel launch overhead (~10-15us)
// =============================================================================
__global__ void allocate_pages_lru_combined_indptr_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    const int32_t* __restrict__ sparse_indptr,
    int32_t indptr_last_idx,
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t WAYS,
    int32_t MAX_HASH_ATTEMPTS
) {
    // Read actual N from GPU memory (no CPU sync!)
    const int32_t N = sparse_indptr[indptr_last_idx];

    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSize;
    const uint32_t lane_id = global_thread_id % kWarpSize;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

    // Each warp processes batches of kWarpSize requests
    for (uint32_t batch_offset = global_warp_id * kWarpSize; batch_offset < N;
         batch_offset += num_warps * kWarpSize) {
        const uint32_t n_batch_keys = min(kWarpSize, N - (int)batch_offset);

        // All lanes iterate through each request in the batch
        for (uint32_t i = 0; i < n_batch_keys; ++i) {
            const uint32_t key_idx = batch_offset + i;
            const int32_t cpu_slot = src_page_ids[key_idx];

            // Validate cpu_slot
            if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                }
                __syncwarp();
                continue;
            }

            // =====================================================================
            // Phase 1: Try allocation without eviction (fast path)
            // =====================================================================
            uint64_t h = hash_func_0((uint64_t)cpu_slot);
            uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

            SetContextWarp set_ctx(
                cpu_to_gpu_slot_map,
                gpu_to_cpu_page_map,
                slot_ages,
                set_mutexes,
                set_slot_used_bitmap,
                candidate_set,
                WAYS);

            set_ctx.Lock(lane_id);

            bool is_owner = false;
            int insert_gpu_slot = set_ctx.InsertNoEvict(lane_id, cpu_slot, MAX_PAGE_ID, is_owner);

            set_ctx.Unlock(lane_id);

            // Fast path succeeded - allocation done without eviction
            if (insert_gpu_slot >= 0) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = insert_gpu_slot;
                    // evicted_cpu_pages[key_idx] already initialized to -1
                }
                __syncwarp();
                continue;
            }

            // =====================================================================
            // Phase 2: Slow path - need eviction
            // CRITICAL: First check if another warp already allocated this page
            // This handles the case of duplicate page IDs in the request list
            // =====================================================================
            int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];
            if (existing_slot >= 0) {
                // Another warp already allocated this page - just use their slot
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = false;  // Not the owner
                    dst_staging_slots[key_idx] = existing_slot;
                    // evicted_cpu_pages[key_idx] already initialized to -1
                }
                __syncwarp();
                continue;
            }

            // Try multiple hash functions to find a set with evictable slots
            bool allocated = false;
            int32_t final_gpu_slot = -1;
            int32_t final_evicted_page = -1;

            for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !allocated; ++hash_attempt) {
                uint64_t h_evict = apply_hash((uint64_t)cpu_slot, hash_attempt);
                uint32_t evict_set = (uint32_t)(h_evict % (uint64_t)num_sets);

                SetContextWarp evict_ctx(
                    cpu_to_gpu_slot_map,
                    gpu_to_cpu_page_map,
                    slot_ages,
                    set_mutexes,
                    set_slot_used_bitmap,
                    evict_set,
                    WAYS);

                evict_ctx.Lock(lane_id);

                // Re-check if another warp allocated this page while we were waiting
                // This prevents the race where two warps try to evict for the same page
                int32_t recheck_slot = cpu_to_gpu_slot_map[cpu_slot];
                if (recheck_slot >= 0) {
                    // Another warp allocated this page - use their slot
                    evict_ctx.Unlock(lane_id);
                    if (lane_id == 0) {
                        is_owner = false;
                        final_gpu_slot = recheck_slot;
                        final_evicted_page = -1;
                    }
                    allocated = true;
                    break;
                }

                bool lane_is_owner = false;
                int32_t lane_evicted_page = -1;
                int evict_slot = evict_ctx.InsertWithEvict(lane_id, cpu_slot, MAX_PAGE_ID, lane_is_owner, lane_evicted_page);

                if (evict_slot >= 0) {
                    if (lane_id == 0) {
                        is_owner = lane_is_owner;
                        final_gpu_slot = evict_slot;
                        final_evicted_page = lane_evicted_page;
                    }
                    allocated = true;
                }

                evict_ctx.Unlock(lane_id);

                if (allocated) break;
            }

            if (lane_id == 0) {
                if (!allocated) {
                    // All hash attempts failed - overflow
                    atomicMax(overflow_flag, 1);
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                } else {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = final_gpu_slot;
                    evicted_cpu_pages[key_idx] = final_evicted_page;
                }
            }
            __syncwarp();
        }
    }
}

// Fused kernel to zero bitmaps and initialize outputs in one launch
__global__ void zero_allocation_bitmaps_kernel(
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    int32_t num_slots,
    int32_t max_num_pages
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    // Zero slot bitmap
    for (int32_t i = tid; i < num_slots; i += stride) {
        set_slot_used_bitmap[i] = false;
    }

    // Initialize evicted_cpu_pages to -1 (invalid)
    for (int32_t i = tid; i < max_num_pages; i += stride) {
        evicted_cpu_pages[i] = -1;
    }

    // Zero overflow_flag
    if (tid == 0) {
        *overflow_flag = 0;
    }
}

// CUDA graph compatible launcher using sparse_indptr to get N
void allocate_pages_lru_warp_with_indptr(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_ages,
    at::Tensor set_slot_used_bitmap,
    at::Tensor needs_eviction_bitmap,  // Kept for API compatibility, not used in combined kernel
    at::Tensor dst_staging_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    at::Tensor overflow_flag,
    int32_t max_num_pages,
    const int32_t MAX_HASH_ATTEMPTS
) {
    const int32_t MAX_PAGE_ID = cpu_to_gpu_slot_map.size(0);
    const int32_t WAYS = 32;
    const int32_t num_sets = gpu_to_cpu_page_map.size(0) / WAYS;
    const int32_t num_slots = num_sets * WAYS;

    WarpMutexSemaphoreImpl* set_mutexes = cached_mutexes;

    // Only allocate/initialize if first call or num_sets changed
    if (cached_mutexes == nullptr || cached_num_sets != num_sets) {
        if (cached_mutexes != nullptr) {
            cudaFree(cached_mutexes);
        }

        cudaMalloc(&set_mutexes, num_sets * sizeof(WarpMutexSemaphoreImpl));

        const int init_threads = 256;
        const int init_blocks = (num_sets + init_threads - 1) / init_threads;
        InitCacheSetMutexWarp<<<init_blocks, init_threads>>>(num_sets, set_mutexes);

        cached_mutexes = set_mutexes;
        cached_num_sets = num_sets;

        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_warp_allocator_cache);
            cleanup_registered = true;
        }
    }

    // IMPORTANT: Get current CUDA stream for CUDA graph compatibility
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch configuration with FIXED grid size for CUDA graph compatibility
    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * kWarpSize;
    const int num_warps = (max_num_pages + kWarpSize - 1) / kWarpSize;
    const int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    // Zero bitmaps and initialize outputs in one kernel launch
    zero_allocation_bitmaps_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        set_slot_used_bitmap.data_ptr<bool>(),
        evicted_cpu_pages.data_ptr<int32_t>(),
        overflow_flag.data_ptr<int32_t>(),
        num_slots,
        max_num_pages
    );

    // Combined allocation kernel
    allocate_pages_lru_combined_indptr_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        src_page_ids.data_ptr<int32_t>(),
        cpu_to_gpu_slot_map.data_ptr<int32_t>(),
        gpu_to_cpu_page_map.data_ptr<int32_t>(),
        slot_ages.data_ptr<uint8_t>(),
        set_mutexes,
        set_slot_used_bitmap.data_ptr<bool>(),
        dst_staging_slots.data_ptr<int32_t>(),
        owners_bitmap.data_ptr<bool>(),
        evicted_cpu_pages.data_ptr<int32_t>(),
        overflow_flag.data_ptr<int32_t>(),
        sparse_indptr.data_ptr<int32_t>(),
        indptr_last_idx,
        MAX_PAGE_ID,
        num_sets,
        WAYS,
        MAX_HASH_ATTEMPTS
    );
}
