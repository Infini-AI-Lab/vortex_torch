#include "register.h"
#include "hash_functions.cuh"
#include "warp_mutex.cuh"
#include <cstdlib>

// Constants — identical to lru_block_global_alloc.cu
constexpr int kWarpSizeLruGlobal = 32;
constexpr uint32_t kFullMaskLruGlobal = 0xFFFFFFFFU;

// =============================================================================
// Warp-cooperative age update on global memory
// =============================================================================
__device__ inline void warp_update_ages_global_lg(
    uint8_t* slot_ages, uint32_t set_base, uint32_t target_way,
    uint32_t lane_id, uint32_t WAYS
) {
    uint8_t my_val = slot_ages[set_base + lane_id];
    uint8_t old_val = __shfl_sync(kFullMaskLruGlobal, my_val, target_way);
    if ((int32_t)lane_id == (int32_t)target_way) {
        my_val = WAYS;
    } else if (my_val > old_val && my_val > 0) {
        my_val -= 1;
    }
    slot_ages[set_base + lane_id] = my_val;
}

// =============================================================================
// Kernel 1: Hit check + empty slot (no eviction), device semaphore locks
// =============================================================================
__global__ void allocate_pages_lru_global_no_evict_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    WarpMutexSemaphoreImpl* __restrict__ set_mutexes,
    uint32_t* __restrict__ set_used_mask,
    int32_t* __restrict__ dst_staging_slots,
    bool* __restrict__ owners_bitmap,
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

    for (uint32_t key_idx = global_warp_id; key_idx < (uint32_t)N; key_idx += num_warps) {
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

        // --- HIT CHECK ---
        int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];
        if (existing_slot >= 0) {
            uint32_t hit_set = (uint32_t)(existing_slot / WAYS);
            uint32_t hit_way = existing_slot - hit_set * WAYS;

            bool locked = set_mutexes[hit_set].TryLock(lane_id);
            if (locked) {
                int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
                if (recheck == existing_slot) {
                    warp_update_ages_global_lg(slot_ages, hit_set * WAYS, hit_way, lane_id, WAYS);
                    if ((int32_t)lane_id == (int32_t)hit_way) {
                        atomicOr(&set_used_mask[hit_set], 1u << hit_way);
                    }
                    result_slot = existing_slot;
                    is_owner = false;
                    done = true;
                }
                set_mutexes[hit_set].Unlock(lane_id);
            }
        }

        if (done) {
            if (lane_id == 0) {
                owners_bitmap[key_idx] = is_owner;
                dst_staging_slots[key_idx] = result_slot;
            }
            __syncwarp();
            continue;
        }

        // --- EMPTY SLOT SEARCH ---
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

            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                set_mutexes[target_set].Unlock(lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            int32_t lane_page = gpu_to_cpu_page_map[set_base + lane_id];
            bool lane_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruGlobal, lane_empty);

            if (empty_mask != 0) {
                int insert_way = __ffs(empty_mask) - 1;
                int32_t new_slot = set_base + insert_way;

                if (lane_id == 0) {
                    int32_t old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_slot);
                    if (old_fwd == -1) {
                        gpu_to_cpu_page_map[new_slot] = cpu_slot;
                        atomicOr(&set_used_mask[target_set], 1u << insert_way);
                        result_slot = new_slot;
                        is_owner = true;
                    } else {
                        result_slot = old_fwd;
                        is_owner = false;
                    }
                }
                bool cas_success = __shfl_sync(kFullMaskLruGlobal, is_owner ? 1 : 0, 0);
                if (cas_success) {
                    warp_update_ages_global_lg(slot_ages, set_base, insert_way, lane_id, WAYS);
                }
                done = true;
                set_mutexes[target_set].Unlock(lane_id);
                break;
            }

            set_mutexes[target_set].Unlock(lane_id);
        }

        if (lane_id == 0) {
            if (done) {
                owners_bitmap[key_idx] = is_owner;
                dst_staging_slots[key_idx] = result_slot;
            } else {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;  // needs eviction in kernel 2
            }
        }
        __syncwarp();
    }
}

// =============================================================================
// Kernel 2: Eviction for unresolved pages, with butterfly min-reduction
// =============================================================================
__global__ void allocate_pages_lru_global_evict_kernel(
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
    const int32_t N = sparse_indptr[indptr_last_idx];
    const uint32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_warp_id = global_thread_id / kWarpSizeLruGlobal;
    const uint32_t lane_id = global_thread_id % kWarpSizeLruGlobal;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSizeLruGlobal;

    for (uint32_t key_idx = global_warp_id; key_idx < (uint32_t)N; key_idx += num_warps) {
        // Skip already resolved pages
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

            // Re-check
            int32_t existing_slot = cpu_to_gpu_slot_map[cpu_slot];
            if (existing_slot >= 0) {
                uint32_t hit_set = (uint32_t)(existing_slot / WAYS);
                uint32_t hit_way = existing_slot - hit_set * WAYS;
                if (hit_set == target_set) {
                    warp_update_ages_global_lg(slot_ages, set_base, hit_way, lane_id, WAYS);
                    if ((int32_t)lane_id == (int32_t)hit_way) {
                        atomicOr(&set_used_mask[hit_set], 1u << hit_way);
                    }
                } else {
                    if (lane_id == 0) atomicOr(&set_used_mask[hit_set], 1u << hit_way);
                }
                set_mutexes[target_set].Unlock(lane_id);
                result_slot = existing_slot;
                result_is_owner = false;
                done = true;
                break;
            }

            int32_t lane_page = gpu_to_cpu_page_map[set_base + lane_id];
            uint8_t lane_age_val = slot_ages[set_base + lane_id];
            uint32_t used_mask_val = set_used_mask[target_set];
            bool lane_used = (used_mask_val >> lane_id) & 1;

            bool lane_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruGlobal, lane_empty);

            int32_t candidate_slot = -1;
            int32_t evicted_page = -1;
            int candidate_way = -1;

            if (empty_mask != 0) {
                candidate_way = __ffs(empty_mask) - 1;
                candidate_slot = set_base + candidate_way;
            } else {
                // Butterfly min-reduction for LRU victim
                bool lane_evictable = !lane_used;
                unsigned evictable_mask = __ballot_sync(kFullMaskLruGlobal, lane_evictable);

                if (evictable_mask != 0) {
                    uint32_t my_age = lane_evictable ? (uint32_t)lane_age_val : UINT32_MAX;
                    for (int offset = 16; offset > 0; offset /= 2) {
                        uint32_t other_age = __shfl_xor_sync(kFullMaskLruGlobal, my_age, offset);
                        if (other_age < my_age) my_age = other_age;
                    }
                    uint32_t min_age = __shfl_sync(kFullMaskLruGlobal, my_age, 0);

                    bool has_min = lane_evictable && ((uint32_t)lane_age_val == min_age);
                    unsigned min_mask = __ballot_sync(kFullMaskLruGlobal, has_min);
                    if (min_mask != 0) {
                        candidate_way = __ffs(min_mask) - 1;
                        candidate_slot = set_base + candidate_way;
                        evicted_page = __shfl_sync(kFullMaskLruGlobal, lane_page, candidate_way);
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

            candidate_way = __shfl_sync(kFullMaskLruGlobal, candidate_way, 0);
            bool cas_success = __shfl_sync(kFullMaskLruGlobal, result_is_owner ? 1 : 0, 0);

            if (cas_success) {
                warp_update_ages_global_lg(slot_ages, set_base, candidate_way, lane_id, WAYS);
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
__global__ void zero_allocation_bitmaps_lru_global_kernel(
    uint32_t* __restrict__ set_used_mask,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    int32_t num_sets,
    int32_t max_num_pages
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < num_sets; i += stride) {
        set_used_mask[i] = 0;
    }
    for (int32_t i = tid; i < max_num_pages; i += stride) {
        evicted_cpu_pages[i] = -1;
    }
    if (tid == 0) {
        *overflow_flag = 0;
    }
}

// Static mutex cache
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
// Launcher — now uses set_used_mask (int32/uint32) instead of bool bitmaps
// =============================================================================
void allocate_pages_lru_global(
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
    const int32_t WAYS = 32;
    const int32_t num_sets = gpu_to_cpu_page_map.size(0) / WAYS;

    uint32_t* set_used_mask_ptr = reinterpret_cast<uint32_t*>(set_used_mask.data_ptr<int32_t>());

    WarpMutexSemaphoreImpl* set_mutexes = cached_mutexes_global;

    if (cached_mutexes_global == nullptr || cached_num_sets_global != num_sets) {
        if (cached_mutexes_global != nullptr) cudaFree(cached_mutexes_global);

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
    {
        const int threads = 256;
        int max_items = max(num_sets, max_num_pages);
        const int blocks = (max_items + threads - 1) / threads;
        zero_allocation_bitmaps_lru_global_kernel<<<blocks, threads, 0, stream>>>(
            set_used_mask_ptr,
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            num_sets,
            max_num_pages
        );
    }

    // Step 2: Hit check + empty slot (no eviction)
    allocate_pages_lru_global_no_evict_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        src_page_ids.data_ptr<int32_t>(),
        cpu_to_gpu_slot_map.data_ptr<int32_t>(),
        gpu_to_cpu_page_map.data_ptr<int32_t>(),
        slot_ages.data_ptr<uint8_t>(),
        set_mutexes,
        set_used_mask_ptr,
        dst_staging_slots.data_ptr<int32_t>(),
        owners_bitmap.data_ptr<bool>(),
        sparse_indptr.data_ptr<int32_t>(),
        indptr_last_idx,
        MAX_PAGE_ID,
        num_sets,
        WAYS,
        MAX_HASH_ATTEMPTS
    );

    // Step 3: Eviction for unresolved pages
    allocate_pages_lru_global_evict_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        src_page_ids.data_ptr<int32_t>(),
        cpu_to_gpu_slot_map.data_ptr<int32_t>(),
        gpu_to_cpu_page_map.data_ptr<int32_t>(),
        slot_ages.data_ptr<uint8_t>(),
        set_mutexes,
        set_used_mask_ptr,
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
