#include "register.h"
#include "hash_functions.cuh"
#include "warp_mutex.cuh"
#include "cache_policies.cuh"
#include <cstdlib>

// Constants (same as lru_block_global_alloc.cu)
constexpr int kWarpSizeABG = 32;
constexpr uint32_t kFullMaskABG = 0xFFFFFFFFU;
constexpr int WAYS_ABG = 32;
constexpr int SETS_PER_BLOCK_ABG = 32;
constexpr int SLOTS_PER_BLOCK_ABG = SETS_PER_BLOCK_ABG * WAYS_ABG;  // 1024

// =============================================================================
// Shared memory spin-lock (block-scoped)
// =============================================================================
__device__ inline void smem_lock_acquire_abg(volatile int32_t* lock) {
    while (atomicCAS((int32_t*)lock, 0, 1) != 0) {}
    __threadfence_block();
}

__device__ inline bool smem_lock_try_acquire_abg(volatile int32_t* lock) {
    bool acquired = (atomicCAS((int32_t*)lock, 0, 1) == 0);
    if (acquired) __threadfence_block();
    return acquired;
}

__device__ inline void smem_lock_release_abg(volatile int32_t* lock) {
    __threadfence_block();
    atomicExch((int32_t*)lock, 0);
}

__device__ inline void warp_smem_lock_abg(volatile int32_t* lock, uint32_t lane_id) {
    if (lane_id == 0) smem_lock_acquire_abg(lock);
    __syncwarp();
}

__device__ inline bool warp_smem_trylock_abg(volatile int32_t* lock, uint32_t lane_id) {
    bool acquired = false;
    if (lane_id == 0) acquired = smem_lock_try_acquire_abg(lock);
    acquired = __shfl_sync(kFullMaskABG, acquired ? 1 : 0, 0);
    __syncwarp();
    return acquired;
}

__device__ inline void warp_smem_unlock_abg(volatile int32_t* lock, uint32_t lane_id) {
    __syncwarp();
    if (lane_id == 0) smem_lock_release_abg(lock);
    __syncwarp();
}

// =============================================================================
// Pre-partition kernel: scatter pages into per-block buckets
// =============================================================================
__global__ void prepartition_pages_abg_kernel(
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
// Main block kernel: policy-agnostic allocation with smem locks
// =============================================================================
__global__ void allocate_pages_abg_kernel(
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_state,
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
    int32_t MAX_HASH_ATTEMPTS,
    int32_t cache_policy
) {
    __shared__ int32_t  smem_reverse_map[SLOTS_PER_BLOCK_ABG];
    __shared__ uint8_t  smem_state[SLOTS_PER_BLOCK_ABG];
    __shared__ uint32_t smem_used_mask[SETS_PER_BLOCK_ABG];
    __shared__ int32_t  smem_lock[SETS_PER_BLOCK_ABG];

    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_in_block = threadIdx.x / kWarpSizeABG;
    const uint32_t lane_id = threadIdx.x % kWarpSizeABG;
    const uint32_t num_warps_in_block = blockDim.x / kWarpSizeABG;

    const uint32_t set_base_global = block_id * SETS_PER_BLOCK_ABG;
    const uint32_t slot_base_global = set_base_global * WAYS_ABG;

    // === BLOCK INIT: Load global state into shared memory ===
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_ABG; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        smem_reverse_map[i] = gpu_to_cpu_page_map[global_slot];
        smem_state[i] = slot_state[global_slot];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_ABG; i += blockDim.x) {
        smem_used_mask[i] = 0;
        smem_lock[i] = 0;
    }
    __syncthreads();

    const int32_t my_count = block_counts[block_id];
    const int32_t* my_page_ids = block_page_ids + block_id * max_pages_per_block;
    const int32_t* my_page_indices = block_page_indices + block_id * max_pages_per_block;

    // =====================================================================
    // PASS 1: HIT CHECK + EMPTY SLOT ALLOCATION (no eviction)
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

        // --- HIT CHECK ---
        int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];

        if (existing_slot >= 0) {
            uint32_t hit_set_global = (uint32_t)(existing_slot / WAYS_ABG);
            uint32_t hit_way = existing_slot - hit_set_global * WAYS_ABG;

            if (hit_set_global >= set_base_global &&
                hit_set_global < set_base_global + SETS_PER_BLOCK_ABG) {
                uint32_t local_set = hit_set_global - set_base_global;
                warp_smem_lock_abg(&smem_lock[local_set], lane_id);
                policy_update_state(cache_policy, &smem_state[local_set * WAYS_ABG], hit_way, lane_id);
                if (lane_id == 0) {
                    atomicOr(&smem_used_mask[local_set], 1u << hit_way);
                }
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
            } else {
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

        // --- TRY EMPTY SLOT (no eviction) ---
        for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_ABG);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_ABG;
            uint32_t global_slot_base = global_set * WAYS_ABG;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock_abg(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock_abg(&smem_lock[local_set], lane_id);
                locked = true;
            }

            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            int32_t lane_page = smem_reverse_map[local_slot_base + lane_id];
            bool lane_is_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskABG, lane_is_empty);

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
                cas_success = __shfl_sync(kFullMaskABG, cas_success ? 1 : 0, 0);
                if (cas_success) {
                    policy_update_state(cache_policy, &smem_state[local_slot_base], insert_way, lane_id);
                }
                done = true;
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
                break;
            }

            warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
        }

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
    // BARRIER
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
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_ABG);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_ABG;
            uint32_t global_slot_base = global_set * WAYS_ABG;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock_abg(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock_abg(&smem_lock[local_set], lane_id);
                locked = true;
            }

            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            // Try empty slot first
            int32_t lane_page = smem_reverse_map[local_slot_base + lane_id];
            uint8_t lane_state_val = smem_state[local_slot_base + lane_id];
            bool lane_is_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskABG, lane_is_empty);

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
                cas_success = __shfl_sync(kFullMaskABG, cas_success ? 1 : 0, 0);
                if (cas_success) {
                    policy_update_state(cache_policy, &smem_state[local_slot_base], insert_way, lane_id);
                }
                done = true;
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
                break;
            }

            // --- Eviction via policy-specific victim selection ---
            uint32_t used_mask = smem_used_mask[local_set];
            bool lane_evictable = !((used_mask >> lane_id) & 1);
            unsigned evictable_mask = __ballot_sync(kFullMaskABG, lane_evictable);

            if (evictable_mask == 0) {
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
                continue;
            }

            uint64_t rand_seed = (uint64_t)clock64() + (uint64_t)blockIdx.x * 1337ull + (uint64_t)page_idx;
            int insert_way = policy_select_victim(
                cache_policy, smem_state, local_slot_base, evictable_mask, lane_id, rand_seed);

            if (insert_way < 0) {
                warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
                continue;
            }

            int32_t old_cpu_slot = smem_reverse_map[local_slot_base + insert_way];
            int32_t new_global_slot = global_slot_base + insert_way;

            bool cas_success = false;
            if (lane_id == 0) {
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
                    if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
                        atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], -1, new_global_slot);
                    }
                    result_slot = old_fwd;
                    result_evicted = -1;
                    is_owner = false;
                }
            }
            cas_success = __shfl_sync(kFullMaskABG, cas_success ? 1 : 0, 0);
            if (cas_success) {
                policy_update_state(cache_policy, &smem_state[local_slot_base], insert_way, lane_id);
            }
            done = true;
            warp_smem_unlock_abg(&smem_lock[local_set], lane_id);
            break;
        }

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
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_ABG; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        gpu_to_cpu_page_map[global_slot] = smem_reverse_map[i];
        slot_state[global_slot] = smem_state[i];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_ABG; i += blockDim.x) {
        set_used_mask_global[set_base_global + i] = smem_used_mask[i];
    }
}

// =============================================================================
// Global fallback kernel: device semaphore locks, policy-agnostic
// =============================================================================
__global__ void allocate_pages_abg_global_evict_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_state,
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
    int32_t MAX_HASH_ATTEMPTS,
    int32_t cache_policy
) {
    __shared__ int32_t needs_work;
    if (threadIdx.x == 0) {
        needs_work = *overflow_flag;
    }
    __syncthreads();
    if (needs_work == 0) return;

    const int32_t N = sparse_indptr[indptr_last_idx];
    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSizeABG;
    const uint32_t lane_id = global_thread_id % kWarpSizeABG;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSizeABG;

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

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = set_mutexes[target_set].TryLock(lane_id);
                if (!locked) continue;
            } else {
                set_mutexes[target_set].Lock(lane_id);
                locked = true;
            }

            int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];
            if (existing_slot >= 0) {
                uint32_t hit_set = (uint32_t)(existing_slot / WAYS);
                uint32_t hit_way = existing_slot - hit_set * WAYS;

                if (hit_set == target_set) {
                    policy_update_state_global(cache_policy, slot_state, set_base, hit_way, lane_id, WAYS);
                    if ((int32_t)lane_id == (int32_t)hit_way) {
                        atomicOr(&set_used_mask[hit_set], 1u << hit_way);
                    }
                } else {
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

            int32_t lane_page = gpu_to_cpu_page_map[set_base + lane_id];
            uint32_t used_mask_val = set_used_mask[target_set];
            bool lane_used = (used_mask_val >> lane_id) & 1;

            bool lane_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskABG, lane_empty);

            int32_t candidate_slot = -1;
            int32_t evicted_page = -1;
            int candidate_way = -1;

            if (empty_mask != 0) {
                candidate_way = __ffs(empty_mask) - 1;
                candidate_slot = set_base + candidate_way;
            } else {
                bool lane_evictable = !lane_used;
                unsigned evictable_mask = __ballot_sync(kFullMaskABG, lane_evictable);

                if (evictable_mask != 0) {
                    uint64_t rand_seed = (uint64_t)clock64() + (uint64_t)blockIdx.x * 1337ull + (uint64_t)key_idx;
                    candidate_way = policy_select_victim_global(
                        cache_policy, slot_state, set_base, evictable_mask, lane_id, rand_seed);
                    if (candidate_way >= 0) {
                        candidate_slot = set_base + candidate_way;
                        evicted_page = __shfl_sync(kFullMaskABG, lane_page, candidate_way);
                    }
                }
            }

            if (candidate_slot < 0) {
                set_mutexes[target_set].Unlock(lane_id);
                continue;
            }

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

            candidate_way = __shfl_sync(kFullMaskABG, candidate_way, 0);
            bool cas_success = __shfl_sync(kFullMaskABG, result_is_owner ? 1 : 0, 0);

            if (cas_success) {
                policy_update_state_global(cache_policy, slot_state, set_base, candidate_way, lane_id, WAYS);
                if ((int32_t)lane_id == candidate_way) {
                    atomicOr(&set_used_mask[target_set], 1u << candidate_way);
                }
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
__global__ void zero_abg_bitmaps_kernel(
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
static int32_t* cached_block_page_ids_abg = nullptr;
static int32_t* cached_block_page_indices_abg = nullptr;
static int32_t* cached_block_counts_abg = nullptr;
static int32_t cached_num_blocks_abg = 0;
static int32_t cached_max_pages_per_block_abg = 0;

static WarpMutexSemaphoreImpl* cached_mutexes_abg = nullptr;
static int32_t cached_num_sets_abg = 0;

static void cleanup_abg_cache() {
    if (cached_block_page_ids_abg) cudaFree(cached_block_page_ids_abg);
    if (cached_block_page_indices_abg) cudaFree(cached_block_page_indices_abg);
    if (cached_block_counts_abg) cudaFree(cached_block_counts_abg);
    if (cached_mutexes_abg) cudaFree(cached_mutexes_abg);
    cached_block_page_ids_abg = nullptr;
    cached_block_page_indices_abg = nullptr;
    cached_block_counts_abg = nullptr;
    cached_mutexes_abg = nullptr;
    cached_num_blocks_abg = 0;
    cached_max_pages_per_block_abg = 0;
    cached_num_sets_abg = 0;
}

// =============================================================================
// Launcher — single entry point with cache_policy parameter
// =============================================================================
void allocate_pages_block_global(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_state,
    at::Tensor set_used_mask,
    at::Tensor dst_staging_slots,
    at::Tensor owners_bitmap,
    at::Tensor evicted_cpu_pages,
    at::Tensor overflow_flag,
    int32_t max_num_pages,
    const int32_t MAX_HASH_ATTEMPTS,
    int32_t cache_policy
) {
    const int32_t MAX_PAGE_ID = cpu_to_gpu_slot_map.size(0);
    const int32_t num_sets = gpu_to_cpu_page_map.size(0) / WAYS_ABG;
    const int32_t num_slots = num_sets * WAYS_ABG;
    const int32_t num_blocks = num_sets / SETS_PER_BLOCK_ABG;

    TORCH_CHECK(num_sets % SETS_PER_BLOCK_ABG == 0,
        "num_sets (", num_sets, ") must be divisible by SETS_PER_BLOCK (", SETS_PER_BLOCK_ABG, ")");
    TORCH_CHECK(cache_policy >= 0 && cache_policy <= 2,
        "cache_policy must be 0 (LRU), 1 (LFU), or 2 (RANDOM)");

    const int32_t max_pages_per_block = max(max_num_pages * 4 / num_blocks, 64);

    uint32_t* set_used_mask_ptr = reinterpret_cast<uint32_t*>(set_used_mask.data_ptr<int32_t>());

    // Allocate/reuse temp buffers
    if (cached_num_blocks_abg != num_blocks || cached_max_pages_per_block_abg != max_pages_per_block) {
        if (cached_block_page_ids_abg) cudaFree(cached_block_page_ids_abg);
        if (cached_block_page_indices_abg) cudaFree(cached_block_page_indices_abg);
        if (cached_block_counts_abg) cudaFree(cached_block_counts_abg);

        size_t buf_size = (size_t)num_blocks * max_pages_per_block * sizeof(int32_t);
        cudaMalloc(&cached_block_page_ids_abg, buf_size);
        cudaMalloc(&cached_block_page_indices_abg, buf_size);
        cudaMalloc(&cached_block_counts_abg, num_blocks * sizeof(int32_t));

        cached_num_blocks_abg = num_blocks;
        cached_max_pages_per_block_abg = max_pages_per_block;
    }

    // Allocate/reuse mutexes
    if (cached_mutexes_abg == nullptr || cached_num_sets_abg != num_sets) {
        if (cached_mutexes_abg) cudaFree(cached_mutexes_abg);
        cudaMalloc(&cached_mutexes_abg, num_sets * sizeof(WarpMutexSemaphoreImpl));

        const int init_threads = 256;
        const int init_blocks = (num_sets + init_threads - 1) / init_threads;
        InitCacheSetMutexWarp<<<init_blocks, init_threads>>>(num_sets, cached_mutexes_abg);

        cached_num_sets_abg = num_sets;
    }

    // Register cleanup
    {
        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_abg_cache);
            cleanup_registered = true;
        }
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Zero bitmaps + block_counts
    {
        const int threads = 256;
        int max_items = max(max(num_sets, max_num_pages), num_blocks);
        const int blocks = (max_items + threads - 1) / threads;
        zero_abg_bitmaps_kernel<<<blocks, threads, 0, stream>>>(
            set_used_mask_ptr,
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_counts_abg,
            num_sets,
            max_num_pages,
            num_blocks
        );
    }

    // Step 2: Pre-partition
    {
        const int threads = 256;
        const int blocks = (max_num_pages + threads - 1) / threads;
        prepartition_pages_abg_kernel<<<blocks, threads, 0, stream>>>(
            src_page_ids.data_ptr<int32_t>(),
            sparse_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            cached_block_page_ids_abg,
            cached_block_page_indices_abg,
            cached_block_counts_abg,
            num_blocks,
            max_pages_per_block
        );
    }

    // Step 3: Block kernel (pass 1 + barrier + pass 2)
    {
        const int threads_per_block = SETS_PER_BLOCK_ABG * kWarpSizeABG;  // 1024
        allocate_pages_abg_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            cpu_to_gpu_slot_map.data_ptr<int32_t>(),
            gpu_to_cpu_page_map.data_ptr<int32_t>(),
            slot_state.data_ptr<uint8_t>(),
            set_used_mask_ptr,
            dst_staging_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_page_ids_abg,
            cached_block_page_indices_abg,
            cached_block_counts_abg,
            MAX_PAGE_ID,
            num_sets,
            num_blocks,
            max_pages_per_block,
            MAX_HASH_ATTEMPTS,
            cache_policy
        );
    }

    // Step 4: Global fallback
    {
        const int warps_per_block = 4;
        const int threads_per_block = warps_per_block * kWarpSizeABG;  // 128
        const int num_warps = (max_num_pages + kWarpSizeABG - 1) / kWarpSizeABG;
        const int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
        allocate_pages_abg_global_evict_kernel<<<blocks, threads_per_block, 0, stream>>>(
            src_page_ids.data_ptr<int32_t>(),
            cpu_to_gpu_slot_map.data_ptr<int32_t>(),
            gpu_to_cpu_page_map.data_ptr<int32_t>(),
            slot_state.data_ptr<uint8_t>(),
            cached_mutexes_abg,
            set_used_mask_ptr,
            dst_staging_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            sparse_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            MAX_PAGE_ID,
            num_sets,
            WAYS_ABG,
            MAX_HASH_ATTEMPTS,
            cache_policy
        );
    }
}
