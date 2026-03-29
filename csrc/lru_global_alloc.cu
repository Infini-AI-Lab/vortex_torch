#include "register.h"
#include "hash_functions.cuh"
#include "warp_mutex.cuh"
#include <cstdlib>

// Constants for warp-level operations
constexpr int kWarpSizeLruGlobal = 32;
constexpr uint32_t kFullMaskLruGlobal = 0xFFFFFFFFU;

// SetContextWarpLruGlobal - Non-blocking variant with TryLock support
struct SetContextWarpLruGlobal {
  __device__ SetContextWarpLruGlobal(
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

  // Update LRU ages for a confirmed cache hit.
  // Properly demotes sibling ages (matching v2/v4 behavior).
  __device__ void UpdateLRU(uint32_t lane_id, int32_t hit_gpu_slot) {
    int32_t hit_way = hit_gpu_slot - set_base;
    uint8_t lane_age = slot_ages[lane_id];
    uint8_t hit_old_age = __shfl_sync(kFullMaskLruGlobal, lane_age, hit_way);

    if ((int32_t)lane_id == hit_way) {
      lane_age = num_ways;
      used_this_round[hit_way] = true;
    } else if (lane_age > hit_old_age && lane_age > 0) {
      lane_age -= 1;
    }
    slot_ages[lane_id] = lane_age;
    __syncwarp();
  }

  // Insert into an empty slot only (no hit check, no eviction).
  // Uses atomicCAS on forward map to prevent races between warps locking different sets.
  __device__ int InsertEmptyOnly(uint32_t lane_id, int32_t cpu_slot, int32_t MAX_PAGE_ID, bool& is_owner) {
    is_owner = false;
    uint8_t lane_age = slot_ages[lane_id];
    bool lane_is_empty = (lane_age == 0);

    const unsigned empty_mask = __ballot_sync(kFullMaskLruGlobal, lane_is_empty);

    if (empty_mask == 0) {
      return -1;
    }

    int insert_way = __ffs(static_cast<int>(empty_mask)) - 1;
    int32_t new_slot = set_base + insert_way;

    bool cas_success = false;
    int32_t old_fwd = -1;
    if ((int32_t)lane_id == insert_way) {
      old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_slot);
      cas_success = (old_fwd == -1);
    }
    cas_success = __shfl_sync(kFullMaskLruGlobal, cas_success ? 1 : 0, insert_way);
    old_fwd = __shfl_sync(kFullMaskLruGlobal, old_fwd, insert_way);

    if (cas_success) {
      if ((int32_t)lane_id == insert_way) {
        gpu_to_cpu_page_map[insert_way] = cpu_slot;
        lane_age = num_ways;
        used_this_round[insert_way] = true;
      } else if (lane_age > 1) {
        lane_age -= 1;
      }
      slot_ages[lane_id] = lane_age;
      __syncwarp();

      is_owner = true;
      return new_slot;
    } else {
      __syncwarp();
      is_owner = false;
      return old_fwd;
    }
  }

  // Insert with eviction - finds the LRU slot not used this round.
  __device__ int InsertWithEvict(uint32_t lane_id, int32_t cpu_slot, int32_t MAX_PAGE_ID, bool& is_owner, int32_t& evicted_cpu_page) {
    uint8_t lane_age = slot_ages[lane_id];
    bool lane_not_used = !used_this_round[lane_id];

    int insert_way = -1;

    for (uint8_t target_age = 0; target_age <= num_ways && insert_way == -1; ++target_age) {
      const unsigned age_mask = __ballot_sync(kFullMaskLruGlobal, lane_age == target_age && lane_not_used);
      if (age_mask != 0) {
        insert_way = __ffs(static_cast<int>(age_mask)) - 1;
        break;
      }
    }

    if (insert_way == -1) {
      evicted_cpu_page = -1;
      return -1;
    }

    int32_t old_cpu_slot = gpu_to_cpu_page_map[insert_way];
    int32_t new_slot = set_base + insert_way;

    bool cas_success = false;
    int32_t old_fwd = -1;
    if ((int32_t)lane_id == insert_way) {
      if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
        atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], new_slot, -1);
      }
      old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_slot);
      cas_success = (old_fwd == -1);
    }
    cas_success = __shfl_sync(kFullMaskLruGlobal, cas_success ? 1 : 0, insert_way);
    old_fwd = __shfl_sync(kFullMaskLruGlobal, old_fwd, insert_way);

    if (cas_success) {
      evicted_cpu_page = old_cpu_slot;
      if ((int32_t)lane_id == insert_way) {
        gpu_to_cpu_page_map[insert_way] = cpu_slot;
        lane_age = num_ways;
        used_this_round[insert_way] = true;
      } else if (lane_age > 1) {
        lane_age -= 1;
      }
      slot_ages[lane_id] = lane_age;
      __syncwarp();

      is_owner = true;
      return new_slot;
    } else {
      evicted_cpu_page = -1;
      if ((int32_t)lane_id == insert_way) {
        if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
          atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], -1, new_slot);
        }
        gpu_to_cpu_page_map[insert_way] = old_cpu_slot;
      }
      __syncwarp();
      is_owner = false;
      return old_fwd;
    }
  }

  __device__ void Lock(uint32_t lane_id) { mutex->Lock(lane_id); }
  __device__ bool TryLock(uint32_t lane_id) { return mutex->TryLock(lane_id); }
  __device__ void Unlock(uint32_t lane_id) { mutex->Unlock(lane_id); }

  // Members
  int32_t* cpu_to_gpu_slot_map;
  int32_t* gpu_to_cpu_page_map;
  uint8_t* slot_ages;
  WarpMutexSemaphoreImpl* mutex;
  bool* used_this_round;
  uint32_t set_base;
  uint32_t num_ways;
};

// =============================================================================
// Kernel 1: Hit check + empty slot allocation (NO eviction)
// Uses TryLock for non-blocking operation; last hash attempt uses blocking Lock.
// Pages that cannot be resolved get needs_eviction_bitmap = true.
// =============================================================================
__global__ void allocate_pages_lru_global_no_evict_indptr_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    bool* __restrict__ needs_eviction_bitmap,
    const int32_t* __restrict__ sparse_indptr,
    int32_t indptr_last_idx,
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t WAYS,
    int32_t MAX_HASH_ATTEMPTS
) {
    const int32_t N = sparse_indptr[indptr_last_idx];

    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSizeLruGlobal;
    const uint32_t lane_id = global_thread_id % kWarpSizeLruGlobal;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSizeLruGlobal;

    for (uint32_t batch_offset = global_warp_id * kWarpSizeLruGlobal; batch_offset < N;
         batch_offset += num_warps * kWarpSizeLruGlobal) {
        const uint32_t n_batch_keys = min(kWarpSizeLruGlobal, N - (int)batch_offset);

        for (uint32_t i = 0; i < n_batch_keys; ++i) {
            const uint32_t key_idx = batch_offset + i;
            const int32_t cpu_slot = src_page_ids[key_idx];

            if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                    needs_eviction_bitmap[key_idx] = false;
                }
                __syncwarp();
                continue;
            }

            bool done = false;
            bool is_owner = false;
            int32_t result_slot = -1;

            // =================================================================
            // PHASE 1: NON-BLOCKING HIT CHECK (TryLock)
            // =================================================================
            int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];

            if (existing_slot >= 0) {
                uint32_t hit_set = (uint32_t)(existing_slot / WAYS);

                SetContextWarpLruGlobal hit_ctx(
                    cpu_to_gpu_slot_map,
                    gpu_to_cpu_page_map,
                    slot_ages,
                    set_mutexes,
                    set_slot_used_bitmap,
                    hit_set,
                    WAYS);

                bool locked = hit_ctx.TryLock(lane_id);
                if (locked) {
                    int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
                    if (recheck == existing_slot) {
                        // Confirmed hit — proper LRU update with age demotion
                        hit_ctx.UpdateLRU(lane_id, existing_slot);

                        result_slot = existing_slot;
                        is_owner = false;
                        done = true;
                    }
                    hit_ctx.Unlock(lane_id);
                }
                // If !locked, fall through to Phase 2
            }

            if (done) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = result_slot;
                    needs_eviction_bitmap[key_idx] = false;
                }
                __syncwarp();
                continue;
            }

            // =================================================================
            // PHASE 2: MULTI-HASH EMPTY SLOT SEARCH (TryLock + last fallback)
            // =================================================================
            for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
                uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
                uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

                SetContextWarpLruGlobal set_ctx(
                    cpu_to_gpu_slot_map,
                    gpu_to_cpu_page_map,
                    slot_ages,
                    set_mutexes,
                    set_slot_used_bitmap,
                    candidate_set,
                    WAYS);

                bool locked;
                if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                    locked = set_ctx.TryLock(lane_id);
                    if (!locked) continue;
                } else {
                    set_ctx.Lock(lane_id);
                    locked = true;
                }

                // Re-check: another warp might have allocated this page
                int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
                if (recheck >= 0) {
                    set_ctx.Unlock(lane_id);
                    result_slot = recheck;
                    is_owner = false;
                    done = true;
                    break;
                }

                // Try empty slot only (no eviction)
                bool insert_owner = false;
                int insert_slot = set_ctx.InsertEmptyOnly(lane_id, cpu_slot, MAX_PAGE_ID, insert_owner);

                set_ctx.Unlock(lane_id);

                if (insert_slot >= 0) {
                    result_slot = insert_slot;
                    is_owner = insert_owner;
                    done = true;
                    break;
                }
            }

            // Write results
            if (lane_id == 0) {
                if (done) {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = result_slot;
                    needs_eviction_bitmap[key_idx] = false;
                } else {
                    // Needs eviction in kernel 2
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                    needs_eviction_bitmap[key_idx] = true;
                }
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// Kernel 2: Eviction (only for pages not resolved in kernel 1)
// All hits are already resolved and used_this_round is fully populated.
// Uses TryLock + last-attempt blocking Lock.
// =============================================================================
__global__ void allocate_pages_lru_global_with_evict_indptr_kernel(
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
    const int32_t N = sparse_indptr[indptr_last_idx];

    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSizeLruGlobal;
    const uint32_t lane_id = global_thread_id % kWarpSizeLruGlobal;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSizeLruGlobal;

    for (uint32_t batch_offset = global_warp_id * kWarpSizeLruGlobal; batch_offset < N;
         batch_offset += num_warps * kWarpSizeLruGlobal) {
        const uint32_t n_batch_keys = min(kWarpSizeLruGlobal, N - (int)batch_offset);

        for (uint32_t i = 0; i < n_batch_keys; ++i) {
            const uint32_t key_idx = batch_offset + i;

            // Skip pages already resolved in kernel 1
            if (!needs_eviction_bitmap[key_idx]) {
                __syncwarp();
                continue;
            }

            const int32_t cpu_slot = src_page_ids[key_idx];

            if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
                if (lane_id == 0) {
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                }
                __syncwarp();
                continue;
            }

            bool done = false;
            bool is_owner = false;
            int32_t result_slot = -1;
            int32_t result_evicted = -1;

            for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
                uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
                uint32_t candidate_set = (uint32_t)(h % (uint64_t)num_sets);

                SetContextWarpLruGlobal set_ctx(
                    cpu_to_gpu_slot_map,
                    gpu_to_cpu_page_map,
                    slot_ages,
                    set_mutexes,
                    set_slot_used_bitmap,
                    candidate_set,
                    WAYS);

                bool locked;
                if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                    locked = set_ctx.TryLock(lane_id);
                    if (!locked) continue;
                } else {
                    set_ctx.Lock(lane_id);
                    locked = true;
                }

                // Re-check: another warp might have allocated this page
                int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
                if (recheck >= 0) {
                    set_ctx.Unlock(lane_id);
                    result_slot = recheck;
                    is_owner = false;
                    done = true;
                    break;
                }

                // Evict LRU (respects used_this_round)
                bool evict_owner = false;
                int32_t evicted_page = -1;
                int evict_slot = set_ctx.InsertWithEvict(lane_id, cpu_slot, MAX_PAGE_ID, evict_owner, evicted_page);

                set_ctx.Unlock(lane_id);

                if (evict_slot >= 0) {
                    result_slot = evict_slot;
                    is_owner = evict_owner;
                    result_evicted = evicted_page;
                    done = true;
                }
            }

            // Write results
            if (lane_id == 0) {
                if (!done) {
                    atomicMax(overflow_flag, 1);
                    owners_bitmap[key_idx] = false;
                    dst_staging_slots[key_idx] = -1;
                } else {
                    owners_bitmap[key_idx] = is_owner;
                    dst_staging_slots[key_idx] = result_slot;
                    evicted_cpu_pages[key_idx] = result_evicted;
                }
            }
            __syncwarp();
        }
    }
}

// Fused kernel to zero bitmaps and initialize outputs
__global__ void zero_allocation_bitmaps_lru_global_kernel(
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    int32_t num_slots,
    int32_t max_num_pages
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < num_slots; i += stride) {
        set_slot_used_bitmap[i] = false;
    }

    for (int32_t i = tid; i < max_num_pages; i += stride) {
        evicted_cpu_pages[i] = -1;
    }

    if (tid == 0) {
        *overflow_flag = 0;
    }
}

// Static mutex cache for v3
static WarpMutexSemaphoreImpl* cached_mutexes_global = nullptr;
static int32_t cached_num_sets_global = 0;

static void cleanup_warp_allocator_cache_lru_global() {
    if (cached_mutexes_global != nullptr) {
        cudaFree(cached_mutexes_global);
        cached_mutexes_global = nullptr;
    }
    cached_num_sets_global = 0;
}

// =============================================================================
// Launcher function
// =============================================================================
void allocate_pages_lru_global(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_ages,
    at::Tensor set_slot_used_bitmap,
    at::Tensor needs_eviction_bitmap,
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

    WarpMutexSemaphoreImpl* set_mutexes = cached_mutexes_global;

    if (cached_mutexes_global == nullptr || cached_num_sets_global != num_sets) {
        if (cached_mutexes_global != nullptr) {
            cudaFree(cached_mutexes_global);
        }

        cudaMalloc(&set_mutexes, num_sets * sizeof(WarpMutexSemaphoreImpl));

        const int init_threads = 256;
        const int init_blocks = (num_sets + init_threads - 1) / init_threads;
        InitCacheSetMutexWarp<<<init_blocks, init_threads>>>(num_sets, set_mutexes);

        cached_mutexes_global = set_mutexes;
        cached_num_sets_global = num_sets;

        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_warp_allocator_cache_lru_global);
            cleanup_registered = true;
        }
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * kWarpSizeLruGlobal;
    const int num_warps = (max_num_pages + kWarpSizeLruGlobal - 1) / kWarpSizeLruGlobal;
    const int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    // Step 1: Zero bitmaps
    zero_allocation_bitmaps_lru_global_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        set_slot_used_bitmap.data_ptr<bool>(),
        evicted_cpu_pages.data_ptr<int32_t>(),
        overflow_flag.data_ptr<int32_t>(),
        num_slots,
        max_num_pages
    );

    // Step 2: Hit check + empty slot allocation (no eviction)
    allocate_pages_lru_global_no_evict_indptr_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        src_page_ids.data_ptr<int32_t>(),
        cpu_to_gpu_slot_map.data_ptr<int32_t>(),
        gpu_to_cpu_page_map.data_ptr<int32_t>(),
        slot_ages.data_ptr<uint8_t>(),
        set_mutexes,
        set_slot_used_bitmap.data_ptr<bool>(),
        dst_staging_slots.data_ptr<int32_t>(),
        owners_bitmap.data_ptr<bool>(),
        needs_eviction_bitmap.data_ptr<bool>(),
        sparse_indptr.data_ptr<int32_t>(),
        indptr_last_idx,
        MAX_PAGE_ID,
        num_sets,
        WAYS,
        MAX_HASH_ATTEMPTS
    );

    // Step 3: Eviction (only for unresolved pages)
    allocate_pages_lru_global_with_evict_indptr_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        src_page_ids.data_ptr<int32_t>(),
        cpu_to_gpu_slot_map.data_ptr<int32_t>(),
        gpu_to_cpu_page_map.data_ptr<int32_t>(),
        slot_ages.data_ptr<uint8_t>(),
        set_mutexes,
        set_slot_used_bitmap.data_ptr<bool>(),
        dst_staging_slots.data_ptr<int32_t>(),
        owners_bitmap.data_ptr<bool>(),
        needs_eviction_bitmap.data_ptr<bool>(),
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
