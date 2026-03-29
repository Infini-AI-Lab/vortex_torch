#include "register.h"
#include "hash_functions.cuh"
#include "warp_mutex.cuh"
#include <cstdlib>

// Constants
constexpr int kWarpSizeLruBG = 32;
constexpr uint32_t kFullMaskLruBG = 0xFFFFFFFFU;
constexpr int WAYS_LruBG = 32;
constexpr int SETS_PER_BLOCK_LruBG = 32;
constexpr int SLOTS_PER_BLOCK_LruBG = SETS_PER_BLOCK_LruBG * WAYS_LruBG;  // 1024

// Shared memory layout per block:
//   int32_t  smem_reverse_map[1024]  — 4096 bytes
//   uint8_t  smem_ages[1024]         — 1024 bytes
//   uint32_t smem_used_mask[32]      — 128 bytes
//   int32_t  smem_lock[32]           — 128 bytes
//   Total: ~5376 bytes per block

// =============================================================================
// Shared memory spin-lock (block-scoped) — same as v5
// =============================================================================
__device__ inline void smem_lock_acquire_lru_bg(volatile int32_t* lock) {
    while (atomicCAS((int32_t*)lock, 0, 1) != 0) {}
    __threadfence_block();
}

__device__ inline bool smem_lock_try_acquire_lru_bg(volatile int32_t* lock) {
    bool acquired = (atomicCAS((int32_t*)lock, 0, 1) == 0);
    if (acquired) __threadfence_block();
    return acquired;
}

__device__ inline void smem_lock_release_lru_bg(volatile int32_t* lock) {
    __threadfence_block();
    atomicExch((int32_t*)lock, 0);
}

__device__ inline void warp_smem_lock_lru_bg(volatile int32_t* lock, uint32_t lane_id) {
    if (lane_id == 0) smem_lock_acquire_lru_bg(lock);
    __syncwarp();
}

__device__ inline bool warp_smem_trylock_lru_bg(volatile int32_t* lock, uint32_t lane_id) {
    bool acquired = false;
    if (lane_id == 0) acquired = smem_lock_try_acquire_lru_bg(lock);
    acquired = __shfl_sync(kFullMaskLruBG, acquired ? 1 : 0, 0);
    __syncwarp();
    return acquired;
}

__device__ inline void warp_smem_unlock_lru_bg(volatile int32_t* lock, uint32_t lane_id) {
    __syncwarp();
    if (lane_id == 0) smem_lock_release_lru_bg(lock);
    __syncwarp();
}

// =============================================================================
// Warp-cooperative age update in shared memory.
// All 32 lanes participate: each reads/writes its own age slot in parallel.
// O(1) per lane instead of O(32) serial loop.
// =============================================================================
__device__ inline void warp_update_ages_smem_lru_bg(
    uint8_t* smem_ages_base, uint32_t target_way, uint32_t lane_id
) {
    uint8_t my_age = smem_ages_base[lane_id];
    uint8_t old_age = __shfl_sync(kFullMaskLruBG, my_age, target_way);
    if (lane_id == target_way) {
        my_age = WAYS_LruBG;
    } else if (my_age > old_age && my_age > 0) {
        my_age -= 1;
    }
    smem_ages_base[lane_id] = my_age;
}

// =============================================================================
// Pre-partition kernel: scatter pages into per-block buckets (same as v5)
// =============================================================================
__global__ void prepartition_pages_lru_bg_kernel(
    const int32_t* __restrict__ src_page_ids,
    const int32_t* __restrict__ sparse_indptr,
    int32_t indptr_last_idx,
    int32_t* __restrict__ block_page_ids,
    int32_t* __restrict__ block_page_indices,
    int32_t* __restrict__ block_counts,
    int32_t num_blocks,
    int32_t max_pages_per_block
) {
    const int32_t N = sparse_indptr[indptr_last_idx];
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < N; i += stride) {
        int32_t page_id = src_page_ids[i];
        uint64_t h = apply_hash((uint64_t)page_id, 0);
        int32_t target_block = (int32_t)(h % (uint64_t)num_blocks);

        int32_t pos = atomicAdd(&block_counts[target_block], 1);
        if (pos < max_pages_per_block) {
            block_page_ids[target_block * max_pages_per_block + pos] = page_id;
            block_page_indices[target_block * max_pages_per_block + pos] = i;
        }
    }
}

// =============================================================================
// Main v7 block kernel: block-local smem with uint8_t relative ages
// Hits require smem lock (unlike v5's lock-free hits)
// =============================================================================
__global__ void allocate_pages_lru_bg_kernel(
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    uint32_t* __restrict__ set_used_mask_global,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    const int32_t* __restrict__ block_page_ids,
    const int32_t* __restrict__ block_page_indices,
    const int32_t* __restrict__ block_counts,
    int32_t MAX_PAGE_ID,
    int32_t num_sets,
    int32_t num_blocks,
    int32_t max_pages_per_block,
    int32_t MAX_HASH_ATTEMPTS
) {
    // Shared memory
    __shared__ int32_t  smem_reverse_map[SLOTS_PER_BLOCK_LruBG];
    __shared__ uint8_t  smem_ages[SLOTS_PER_BLOCK_LruBG];
    __shared__ uint32_t smem_used_mask[SETS_PER_BLOCK_LruBG];
    __shared__ int32_t  smem_lock[SETS_PER_BLOCK_LruBG];

    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_in_block = threadIdx.x / kWarpSizeLruBG;
    const uint32_t lane_id = threadIdx.x % kWarpSizeLruBG;
    const uint32_t num_warps_in_block = blockDim.x / kWarpSizeLruBG;

    const uint32_t set_base_global = block_id * SETS_PER_BLOCK_LruBG;
    const uint32_t slot_base_global = set_base_global * WAYS_LruBG;

    // === BLOCK INIT: Load global state into shared memory ===
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_LruBG; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        smem_reverse_map[i] = gpu_to_cpu_page_map[global_slot];
        smem_ages[i] = slot_ages[global_slot];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_LruBG; i += blockDim.x) {
        smem_used_mask[i] = 0;
        smem_lock[i] = 0;
    }
    __syncthreads();

    const int32_t my_count = block_counts[block_id];
    const int32_t* my_page_ids = block_page_ids + block_id * max_pages_per_block;
    const int32_t* my_page_indices = block_page_indices + block_id * max_pages_per_block;

    // =====================================================================
    // PASS 1: HIT CHECK + EMPTY SLOT ALLOCATION (no eviction)
    // Hits require smem lock because age update touches all 32 ways
    // =====================================================================
    for (int32_t page_idx = warp_in_block; page_idx < my_count; page_idx += num_warps_in_block) {
        if (page_idx >= max_pages_per_block) {
            if (lane_id == 0) atomicMax(overflow_flag, 1);
            __syncwarp();
            continue;
        }

        const int32_t cpu_slot = my_page_ids[page_idx];
        const int32_t key_idx = my_page_indices[page_idx];

        if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) {
            if (lane_id == 0) {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
                evicted_cpu_pages[key_idx] = -1;
            }
            __syncwarp();
            continue;
        }

        bool done = false;
        bool is_owner = false;
        int32_t result_slot = -1;

        // --- HIT CHECK (read forward map, no lock needed for the read) ---
        int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];

        if (existing_slot >= 0) {
            uint32_t hit_set_global = (uint32_t)(existing_slot / WAYS_LruBG);
            uint32_t hit_way = existing_slot - hit_set_global * WAYS_LruBG;

            if (hit_set_global >= set_base_global &&
                hit_set_global < set_base_global + SETS_PER_BLOCK_LruBG) {
                // In-block hit: warp-cooperative age update under smem lock
                uint32_t local_set = hit_set_global - set_base_global;

                warp_smem_lock_lru_bg(&smem_lock[local_set], lane_id);
                warp_update_ages_smem_lru_bg(&smem_ages[local_set * WAYS_LruBG], hit_way, lane_id);
                if (lane_id == 0) {
                    atomicOr(&smem_used_mask[local_set], 1u << hit_way);
                }
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
            } else {
                // Out-of-block hit: just protect from eviction via global used_mask
                if (lane_id == 0) {
                    atomicOr(&set_used_mask_global[hit_set_global], 1u << hit_way);
                }
            }

            if (lane_id == 0) {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = existing_slot;
                evicted_cpu_pages[key_idx] = -1;
            }
            __syncwarp();
            continue;
        }

        // --- TRY EMPTY SLOT (no eviction, locked) ---
        for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_LruBG);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_LruBG;
            uint32_t global_slot_base = global_set * WAYS_LruBG;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock_lru_bg(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock_lru_bg(&smem_lock[local_set], lane_id);
                locked = true;
            }

            // Re-check forward map (another warp may have allocated this page)
            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            // Find empty slot
            int32_t lane_page = smem_reverse_map[local_slot_base + lane_id];
            bool lane_is_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruBG, lane_is_empty);

            if (empty_mask != 0) {
                int insert_way = __ffs((int)empty_mask) - 1;
                int32_t new_global_slot = global_slot_base + insert_way;

                bool cas_success = false;
                if (lane_id == 0) {
                    int32_t old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_global_slot);

                    if (old_fwd == -1) {
                        smem_reverse_map[local_slot_base + insert_way] = cpu_slot;
                        atomicOr(&smem_used_mask[local_set], 1u << insert_way);
                        result_slot = new_global_slot;
                        is_owner = true;
                        cas_success = true;
                    } else {
                        result_slot = old_fwd;
                        is_owner = false;
                    }
                }
                cas_success = __shfl_sync(kFullMaskLruBG, cas_success ? 1 : 0, 0);
                if (cas_success) {
                    warp_update_ages_smem_lru_bg(&smem_ages[local_slot_base], insert_way, lane_id);
                }
                done = true;
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
                break;
            }

            warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
        }

        // Write Pass 1 results
        if (lane_id == 0) {
            if (done) {
                owners_bitmap[key_idx] = is_owner;
                dst_staging_slots[key_idx] = result_slot;
                evicted_cpu_pages[key_idx] = -1;
            } else {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
                evicted_cpu_pages[key_idx] = -1;
            }
        }
        __syncwarp();
    }

    // =====================================================================
    // BARRIER: All hits resolved, smem_used_mask fully populated.
    // =====================================================================
    __syncthreads();

    // =====================================================================
    // PASS 2: EVICTION (only for pages not resolved in Pass 1)
    // =====================================================================
    for (int32_t page_idx = warp_in_block; page_idx < my_count; page_idx += num_warps_in_block) {
        if (page_idx >= max_pages_per_block) {
            __syncwarp();
            continue;
        }

        const int32_t cpu_slot = my_page_ids[page_idx];
        const int32_t key_idx = my_page_indices[page_idx];

        // Skip pages already resolved in Pass 1
        if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID || dst_staging_slots[key_idx] >= 0) {
            __syncwarp();
            continue;
        }

        bool done = false;
        bool is_owner = false;
        int32_t result_slot = -1;
        int32_t result_evicted = -1;

        for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_LruBG);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_LruBG;
            uint32_t global_slot_base = global_set * WAYS_LruBG;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock_lru_bg(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock_lru_bg(&smem_lock[local_set], lane_id);
                locked = true;
            }

            // Re-check forward map
            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            // --- Try empty slot first ---
            int32_t lane_page = smem_reverse_map[local_slot_base + lane_id];
            uint8_t lane_age = smem_ages[local_slot_base + lane_id];
            bool lane_is_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruBG, lane_is_empty);

            if (empty_mask != 0) {
                int insert_way = __ffs((int)empty_mask) - 1;
                int32_t new_global_slot = global_slot_base + insert_way;

                bool cas_success = false;
                if (lane_id == 0) {
                    int32_t old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_global_slot);

                    if (old_fwd == -1) {
                        smem_reverse_map[local_slot_base + insert_way] = cpu_slot;
                        atomicOr(&smem_used_mask[local_set], 1u << insert_way);
                        result_slot = new_global_slot;
                        is_owner = true;
                        cas_success = true;
                    } else {
                        result_slot = old_fwd;
                        is_owner = false;
                    }
                }
                cas_success = __shfl_sync(kFullMaskLruBG, cas_success ? 1 : 0, 0);
                if (cas_success) {
                    warp_update_ages_smem_lru_bg(&smem_ages[local_slot_base], insert_way, lane_id);
                }
                done = true;
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
                break;
            }

            // --- No empty slot: evict LRU via butterfly reduction on ages ---
            uint32_t used_mask = smem_used_mask[local_set];
            bool lane_evictable = !((used_mask >> lane_id) & 1);
            unsigned evictable_mask = __ballot_sync(kFullMaskLruBG, lane_evictable);

            if (evictable_mask == 0) {
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
                continue;
            }

            // Butterfly reduction to find minimum age among evictable slots
            uint32_t my_age = lane_evictable ? (uint32_t)lane_age : UINT32_MAX;

            for (int offset = 16; offset > 0; offset /= 2) {
                uint32_t other_age = __shfl_xor_sync(kFullMaskLruBG, my_age, offset);
                if (other_age < my_age) {
                    my_age = other_age;
                }
            }
            uint32_t min_age = __shfl_sync(kFullMaskLruBG, my_age, 0);

            bool has_min = lane_evictable && ((uint32_t)lane_age == min_age);
            unsigned min_mask = __ballot_sync(kFullMaskLruBG, has_min);

            if (min_mask == 0) {
                warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
                continue;
            }

            int insert_way = __ffs((int)min_mask) - 1;
            int32_t old_cpu_slot = smem_reverse_map[local_slot_base + insert_way];
            int32_t new_global_slot = global_slot_base + insert_way;

            bool cas_success = false;
            if (lane_id == 0) {
                // Clear old page's forward mapping
                if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
                    atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], new_global_slot, -1);
                }

                int32_t old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_global_slot);

                if (old_fwd == -1) {
                    smem_reverse_map[local_slot_base + insert_way] = cpu_slot;
                    atomicOr(&smem_used_mask[local_set], 1u << insert_way);
                    result_slot = new_global_slot;
                    result_evicted = old_cpu_slot;
                    is_owner = true;
                    cas_success = true;
                } else {
                    // CAS failed — revert eviction
                    if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
                        atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], -1, new_global_slot);
                    }
                    result_slot = old_fwd;
                    result_evicted = -1;
                    is_owner = false;
                }
            }
            cas_success = __shfl_sync(kFullMaskLruBG, cas_success ? 1 : 0, 0);
            if (cas_success) {
                warp_update_ages_smem_lru_bg(&smem_ages[local_slot_base], insert_way, lane_id);
            }
            done = true;
            warp_smem_unlock_lru_bg(&smem_lock[local_set], lane_id);
            break;
        }

        // Write Pass 2 results
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

    // === BLOCK END: Write shared memory back to global ===
    __syncthreads();
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_LruBG; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        gpu_to_cpu_page_map[global_slot] = smem_reverse_map[i];
        slot_ages[global_slot] = smem_ages[i];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_LruBG; i += blockDim.x) {
        set_used_mask_global[set_base_global + i] = smem_used_mask[i];
    }
}

// =============================================================================
// Global fallback kernel: device semaphore locks (v3-style)
// Only processes pages left unresolved by the block kernel.
// Early exit if overflow_flag == 0.
// =============================================================================
__global__ void allocate_pages_lru_bg_global_evict_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    uint32_t* __restrict__ set_used_mask,
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
    // Early exit: if block kernel resolved all pages, skip entirely
    __shared__ int32_t needs_work;
    if (threadIdx.x == 0) {
        needs_work = *overflow_flag;
    }
    __syncthreads();
    if (needs_work == 0) return;

    const int32_t N = sparse_indptr[indptr_last_idx];
    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSizeLruBG;
    const uint32_t lane_id = global_thread_id % kWarpSizeLruBG;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSizeLruBG;

    for (uint32_t key_idx = global_warp_id; key_idx < (uint32_t)N; key_idx += num_warps) {
        if (dst_staging_slots[key_idx] >= 0) { __syncwarp(); continue; }

        const int32_t cpu_slot = src_page_ids[key_idx];
        if (cpu_slot < 0 || cpu_slot >= MAX_PAGE_ID) { __syncwarp(); continue; }

        int32_t result_slot = -1;
        bool result_is_owner = false;
        int32_t result_evicted = -1;
        bool done = false;

        for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
            uint32_t target_set = (uint32_t)(h % (uint64_t)num_sets);
            uint32_t set_base = target_set * WAYS;

            // Acquire device semaphore (v3-style)
            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = set_mutexes[target_set].TryLock(lane_id);
                if (!locked) continue;
            } else {
                set_mutexes[target_set].Lock(lane_id);
                locked = true;
            }

            // Re-check forward map (page may have been allocated by another warp)
            int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];
            if (existing_slot >= 0) {
                // Hit — update ages using v3 warp-cooperative pattern
                uint32_t hit_set = (uint32_t)(existing_slot / WAYS);
                uint32_t hit_way = existing_slot - hit_set * WAYS;

                // Only update ages if this is the set we locked
                if (hit_set == target_set) {
                    uint8_t lane_age_val = slot_ages[set_base + lane_id];
                    uint8_t hit_old_age = __shfl_sync(kFullMaskLruBG, lane_age_val, hit_way);

                    if ((int32_t)lane_id == (int32_t)hit_way) {
                        lane_age_val = WAYS;
                        atomicOr(&set_used_mask[hit_set], 1u << hit_way);
                    } else if (lane_age_val > hit_old_age && lane_age_val > 0) {
                        lane_age_val -= 1;
                    }
                    slot_ages[set_base + lane_id] = lane_age_val;
                } else {
                    // Hit in different set — just mark used_mask
                    if (lane_id == 0) {
                        atomicOr(&set_used_mask[hit_set], 1u << hit_way);
                    }
                }

                set_mutexes[target_set].Unlock(lane_id);
                result_slot = existing_slot;
                result_is_owner = false;
                done = true;
                break;
            }

            // Read slot data via warp lanes
            int32_t lane_page = gpu_to_cpu_page_map[set_base + lane_id];
            uint8_t lane_age_val = slot_ages[set_base + lane_id];
            uint32_t used_mask_val = set_used_mask[target_set];
            bool lane_used = (used_mask_val >> lane_id) & 1;

            bool lane_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruBG, lane_empty);

            int32_t candidate_slot = -1;
            int32_t evicted_page = -1;
            int candidate_way = -1;

            if (empty_mask != 0) {
                candidate_way = __ffs(empty_mask) - 1;
                candidate_slot = set_base + candidate_way;
            } else {
                // Find LRU eviction victim
                bool lane_evictable = !lane_used;
                unsigned evictable_mask = __ballot_sync(kFullMaskLruBG, lane_evictable);

                if (evictable_mask != 0) {
                    uint32_t my_age = lane_evictable ? (uint32_t)lane_age_val : UINT32_MAX;
                    for (int offset = 16; offset > 0; offset /= 2) {
                        uint32_t other_age = __shfl_xor_sync(kFullMaskLruBG, my_age, offset);
                        if (other_age < my_age) my_age = other_age;
                    }
                    uint32_t min_age = __shfl_sync(kFullMaskLruBG, my_age, 0);

                    bool has_min = lane_evictable && ((uint32_t)lane_age_val == min_age);
                    unsigned min_mask = __ballot_sync(kFullMaskLruBG, has_min);
                    if (min_mask != 0) {
                        candidate_way = __ffs(min_mask) - 1;
                        candidate_slot = set_base + candidate_way;
                        evicted_page = __shfl_sync(kFullMaskLruBG, lane_page, candidate_way);
                    }
                }
            }

            if (candidate_slot < 0) {
                set_mutexes[target_set].Unlock(lane_id);
                continue;
            }

            // Perform insertion
            if (lane_id == 0) {
                if (evicted_page >= 0 && evicted_page < MAX_PAGE_ID) {
                    atomicCAS(&cpu_to_gpu_slot_map[evicted_page], candidate_slot, -1);
                }

                gpu_to_cpu_page_map[candidate_slot] = cpu_slot;
                int32_t old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, candidate_slot);

                if (old_fwd == -1) {
                    result_slot = candidate_slot;
                    result_is_owner = true;
                    result_evicted = evicted_page;
                } else {
                    // CAS failed — revert
                    if (evicted_page >= 0 && evicted_page < MAX_PAGE_ID) {
                        int32_t rev = atomicCAS(&cpu_to_gpu_slot_map[evicted_page], -1, candidate_slot);
                        if (rev == -1) {
                            gpu_to_cpu_page_map[candidate_slot] = evicted_page;
                        } else {
                            gpu_to_cpu_page_map[candidate_slot] = -1;
                        }
                    } else {
                        gpu_to_cpu_page_map[candidate_slot] = -1;
                    }
                    result_slot = old_fwd;
                    result_is_owner = false;
                    result_evicted = -1;
                }
            }

            // Update ages using v3 warp-cooperative pattern
            // Broadcast candidate_way and do warp-cooperative age update
            candidate_way = __shfl_sync(kFullMaskLruBG, candidate_way, 0);
            int32_t cas_result_slot = __shfl_sync(kFullMaskLruBG, result_slot, 0);
            bool cas_success = __shfl_sync(kFullMaskLruBG, result_is_owner ? 1 : 0, 0);

            if (cas_success) {
                // Warp-cooperative age update (each lane updates its own slot)
                uint8_t my_age_val = slot_ages[set_base + lane_id];
                uint8_t old_age = __shfl_sync(kFullMaskLruBG, my_age_val, candidate_way);

                if ((int32_t)lane_id == candidate_way) {
                    my_age_val = WAYS;
                    atomicOr(&set_used_mask[target_set], 1u << candidate_way);
                } else if (my_age_val > old_age && my_age_val > 0) {
                    my_age_val -= 1;
                }
                slot_ages[set_base + lane_id] = my_age_val;
            }

            set_mutexes[target_set].Unlock(lane_id);
            done = true;
        }

        if (lane_id == 0) {
            if (!done) {
                atomicMax(overflow_flag, 1);
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
                evicted_cpu_pages[key_idx] = -1;
            } else {
                owners_bitmap[key_idx] = result_is_owner;
                dst_staging_slots[key_idx] = result_slot;
                evicted_cpu_pages[key_idx] = result_evicted;
            }
        }
        __syncwarp();
    }
}

// =============================================================================
// Zero kernel
// =============================================================================
__global__ void zero_lru_bg_bitmaps_kernel(
    uint32_t* __restrict__ set_used_mask,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    int32_t* __restrict__ block_counts,
    int32_t num_sets,
    int32_t max_num_pages,
    int32_t num_blocks
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < num_sets; i += stride) {
        set_used_mask[i] = 0;
    }
    for (int32_t i = tid; i < max_num_pages; i += stride) {
        evicted_cpu_pages[i] = -1;
    }
    for (int32_t i = tid; i < num_blocks; i += stride) {
        block_counts[i] = 0;
    }
    if (tid == 0) {
        *overflow_flag = 0;
    }
}

// =============================================================================
// Static temp buffer cache
// =============================================================================
static int32_t* cached_block_page_ids_lru_bg = nullptr;
static int32_t* cached_block_page_indices_lru_bg = nullptr;
static int32_t* cached_block_counts_lru_bg = nullptr;
static int32_t cached_num_blocks_lru_bg = 0;
static int32_t cached_max_pages_per_block_lru_bg = 0;

static WarpMutexSemaphoreImpl* cached_mutexes_lru_bg = nullptr;
static int32_t cached_num_sets_lru_bg = 0;

static void cleanup_lru_bg_cache() {
    if (cached_block_page_ids_lru_bg) cudaFree(cached_block_page_ids_lru_bg);
    if (cached_block_page_indices_lru_bg) cudaFree(cached_block_page_indices_lru_bg);
    if (cached_block_counts_lru_bg) cudaFree(cached_block_counts_lru_bg);
    if (cached_mutexes_lru_bg) cudaFree(cached_mutexes_lru_bg);
    cached_block_page_ids_lru_bg = nullptr;
    cached_block_page_indices_lru_bg = nullptr;
    cached_block_counts_lru_bg = nullptr;
    cached_mutexes_lru_bg = nullptr;
    cached_num_blocks_lru_bg = 0;
    cached_max_pages_per_block_lru_bg = 0;
    cached_num_sets_lru_bg = 0;
}

// =============================================================================
// Launcher
// =============================================================================
void allocate_pages_lru_block_global(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_ages,
    at::Tensor set_used_mask,
    at::Tensor dst_staging_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    at::Tensor overflow_flag,
    int32_t max_num_pages,
    const int32_t MAX_HASH_ATTEMPTS
) {
    const int32_t MAX_PAGE_ID = cpu_to_gpu_slot_map.size(0);
    const int32_t num_sets = gpu_to_cpu_page_map.size(0) / WAYS_LruBG;
    const int32_t num_slots = num_sets * WAYS_LruBG;
    const int32_t num_blocks = num_sets / SETS_PER_BLOCK_LruBG;

    TORCH_CHECK(num_sets % SETS_PER_BLOCK_LruBG == 0,
        "num_sets (", num_sets, ") must be divisible by SETS_PER_BLOCK (", SETS_PER_BLOCK_LruBG, ")");

    const int32_t max_pages_per_block = max(max_num_pages * 4 / num_blocks, 64);

    uint32_t* set_used_mask_ptr = reinterpret_cast<uint32_t*>(set_used_mask.data_ptr<int32_t>());

    // Allocate/reuse temp buffers
    if (cached_num_blocks_lru_bg != num_blocks || cached_max_pages_per_block_lru_bg != max_pages_per_block) {
        if (cached_block_page_ids_lru_bg) cudaFree(cached_block_page_ids_lru_bg);
        if (cached_block_page_indices_lru_bg) cudaFree(cached_block_page_indices_lru_bg);
        if (cached_block_counts_lru_bg) cudaFree(cached_block_counts_lru_bg);

        size_t buf_size = (size_t)num_blocks * max_pages_per_block * sizeof(int32_t);
        cudaMalloc(&cached_block_page_ids_lru_bg, buf_size);
        cudaMalloc(&cached_block_page_indices_lru_bg, buf_size);
        cudaMalloc(&cached_block_counts_lru_bg, num_blocks * sizeof(int32_t));

        cached_num_blocks_lru_bg = num_blocks;
        cached_max_pages_per_block_lru_bg = max_pages_per_block;
    }

    // Allocate/reuse mutexes
    if (cached_mutexes_lru_bg == nullptr || cached_num_sets_lru_bg != num_sets) {
        if (cached_mutexes_lru_bg) cudaFree(cached_mutexes_lru_bg);

        cudaMalloc(&cached_mutexes_lru_bg, num_sets * sizeof(WarpMutexSemaphoreImpl));

        const int init_threads = 256;
        const int init_blocks = (num_sets + init_threads - 1) / init_threads;
        InitCacheSetMutexWarp<<<init_blocks, init_threads>>>(num_sets, cached_mutexes_lru_bg);

        cached_num_sets_lru_bg = num_sets;
    }

    // Register cleanup
    {
        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_lru_bg_cache);
            cleanup_registered = true;
        }
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Zero bitmaps + block_counts
    {
        const int threads = 256;
        int max_items = max(max(num_sets, max_num_pages), num_blocks);
        const int blocks = (max_items + threads - 1) / threads;
        zero_lru_bg_bitmaps_kernel<<<blocks, threads, 0, stream>>>(
            set_used_mask_ptr,
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_counts_lru_bg,
            num_sets,
            max_num_pages,
            num_blocks
        );
    }

    // Step 2: Pre-partition
    {
        const int threads = 256;
        const int blocks = (max_num_pages + threads - 1) / threads;
        prepartition_pages_lru_bg_kernel<<<blocks, threads, 0, stream>>>(
            src_page_ids.data_ptr<int32_t>(),
            sparse_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            cached_block_page_ids_lru_bg,
            cached_block_page_indices_lru_bg,
            cached_block_counts_lru_bg,
            num_blocks,
            max_pages_per_block
        );
    }

    // Step 3: Block kernel (pass 1 + __syncthreads + pass 2)
    {
        const int threads_per_block = SETS_PER_BLOCK_LruBG * kWarpSizeLruBG;  // 1024
        allocate_pages_lru_bg_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            cpu_to_gpu_slot_map.data_ptr<int32_t>(),
            gpu_to_cpu_page_map.data_ptr<int32_t>(),
            slot_ages.data_ptr<uint8_t>(),
            set_used_mask_ptr,
            dst_staging_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_page_ids_lru_bg,
            cached_block_page_indices_lru_bg,
            cached_block_counts_lru_bg,
            MAX_PAGE_ID,
            num_sets,
            num_blocks,
            max_pages_per_block,
            MAX_HASH_ATTEMPTS
        );
    }

    // Step 4: Global fallback (device semaphore, early exit if no work)
    {
        const int warps_per_block = 4;
        const int threads_per_block = warps_per_block * kWarpSizeLruBG;  // 128
        const int num_warps = (max_num_pages + kWarpSizeLruBG - 1) / kWarpSizeLruBG;
        const int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
        allocate_pages_lru_bg_global_evict_kernel<<<blocks, threads_per_block, 0, stream>>>(
            src_page_ids.data_ptr<int32_t>(),
            cpu_to_gpu_slot_map.data_ptr<int32_t>(),
            gpu_to_cpu_page_map.data_ptr<int32_t>(),
            slot_ages.data_ptr<uint8_t>(),
            cached_mutexes_lru_bg,
            set_used_mask_ptr,
            dst_staging_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            sparse_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            MAX_PAGE_ID,
            num_sets,
            WAYS_LruBG,
            MAX_HASH_ATTEMPTS
        );
    }
}
