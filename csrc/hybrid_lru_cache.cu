#include "register.h"
#include "hash_functions.cuh"

constexpr int kWarpSize = 32;
constexpr uint32_t kFullMask = 0xFFFFFFFFU;

// Atomic load helper for int32
__device__ __forceinline__ int32_t atomic_load(int32_t* ptr) {
    return atomicAdd(ptr, 0);
}

// Atomic load helper for uint32
__device__ __forceinline__ uint32_t atomic_load_u32(uint32_t* ptr) {
    return atomicAdd(ptr, 0);
}

// =============================================================================
// Lock-free warp-cooperative allocation kernel with hybrid lock-free/seqlock
// =============================================================================
__global__ void allocate_pages_hybrid_kernel(
    const int32_t* __restrict__ src_page_ids,
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint32_t* __restrict__ slot_stamps,
    uint32_t* __restrict__ set_clock,
    uint32_t* __restrict__ set_version,
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
    const uint32_t global_warp_id = global_thread_id / kWarpSize;
    const uint32_t lane_id = global_thread_id % kWarpSize;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

    // Each warp processes one page at a time
    for (uint32_t key_idx = global_warp_id; key_idx < N; key_idx += num_warps) {
        const int32_t cpu_slot = src_page_ids[key_idx];

        // =====================================================================
        // Validate cpu_slot (warp-uniform)
        // =====================================================================
        bool valid = (cpu_slot >= 0 && cpu_slot < MAX_PAGE_ID);
        valid = __shfl_sync(kFullMask, valid ? 1 : 0, 0);

        if (!valid) {
            if (lane_id == 0) {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
            }
            __syncwarp();
            continue;
        }

        // Result variables
        int32_t result_slot = -1;
        bool result_is_owner = false;
        int32_t result_evicted = -1;
        bool done = false;

        // =====================================================================
        // PHASE 1: LOCK-FREE CACHE HIT CHECK
        // Fast path - no seqlock needed for reads + atomic timestamp update
        // =====================================================================
        int32_t existing_slot = atomic_load(&cpu_to_gpu_slot_map[cpu_slot]);
        existing_slot = __shfl_sync(kFullMask, existing_slot, 0);

        if (existing_slot >= 0) {
            // Verify the reverse mapping
            int32_t slot_content = atomic_load(&gpu_to_cpu_page_map[existing_slot]);
            slot_content = __shfl_sync(kFullMask, slot_content, 0);

            if (slot_content == cpu_slot) {
                // Cache hit! Update timestamp and mark used (lock-free)
                uint32_t existing_set = existing_slot / WAYS;
                uint32_t existing_way = existing_slot % WAYS;

                if (lane_id == 0) {
                    // Atomic timestamp update - no lock needed
                    uint32_t new_stamp = atomicAdd(&set_clock[existing_set], 1);
                    slot_stamps[existing_slot] = new_stamp;
                    atomicOr(&set_used_mask[existing_set], 1u << existing_way);
                }
                __syncwarp();

                // Re-verify (in case we raced with eviction)
                slot_content = atomic_load(&gpu_to_cpu_page_map[existing_slot]);
                slot_content = __shfl_sync(kFullMask, slot_content, 0);

                if (slot_content == cpu_slot) {
                    result_slot = existing_slot;
                    result_is_owner = false;
                    done = true;
                }
            }
        }

        // Broadcast done status
        done = __shfl_sync(kFullMask, done ? 1 : 0, 0);
        if (done) {
            if (lane_id == 0) {
                owners_bitmap[key_idx] = result_is_owner;
                dst_staging_slots[key_idx] = result_slot;
            }
            __syncwarp();
            continue;
        }

        // =====================================================================
        // PHASE 2: SEQLOCK-PROTECTED ALLOCATION
        // Slow path - need exclusive access to modify slot ownership
        // =====================================================================
        for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
            uint32_t target_set = (uint32_t)(h % (uint64_t)num_sets);
            uint32_t set_base = target_set * WAYS;

            // Quick re-check if someone else allocated it (lock-free)
            existing_slot = atomic_load(&cpu_to_gpu_slot_map[cpu_slot]);
            existing_slot = __shfl_sync(kFullMask, existing_slot, 0);

            if (existing_slot >= 0) {
                int32_t slot_content = atomic_load(&gpu_to_cpu_page_map[existing_slot]);
                slot_content = __shfl_sync(kFullMask, slot_content, 0);

                if (slot_content == cpu_slot) {
                    result_slot = existing_slot;
                    result_is_owner = false;
                    done = true;
                    break;
                }
            }

            // Acquire seqlock for this set
            uint32_t v0 = 0;
            bool got_lock = false;
            if (lane_id == 0) {
                for (int spin = 0; spin < 1000; ++spin) {
                    v0 = atomic_load_u32(&set_version[target_set]);
                    if ((v0 & 1) == 0) {
                        if (atomicCAS(&set_version[target_set], v0, v0 + 1) == v0) {
                            got_lock = true;
                            break;
                        }
                    }
                    __threadfence();
                }
            }
            got_lock = __shfl_sync(kFullMask, got_lock ? 1 : 0, 0);

            if (!got_lock) {
                continue;  // Try next set
            }

            // === CRITICAL SECTION START ===

            // Re-check inside lock
            existing_slot = atomic_load(&cpu_to_gpu_slot_map[cpu_slot]);
            existing_slot = __shfl_sync(kFullMask, existing_slot, 0);

            if (existing_slot >= 0) {
                int32_t slot_content = atomic_load(&gpu_to_cpu_page_map[existing_slot]);
                slot_content = __shfl_sync(kFullMask, slot_content, 0);

                if (slot_content == cpu_slot) {
                    // Release lock and use existing
                    if (lane_id == 0) {
                        atomicAdd(&set_version[target_set], 1);
                    }
                    __syncwarp();
                    result_slot = existing_slot;
                    result_is_owner = false;
                    done = true;
                    break;
                }
            }

            // Read slot data for this set
            int32_t lane_page = gpu_to_cpu_page_map[set_base + lane_id];
            uint32_t lane_stamp = slot_stamps[set_base + lane_id];
            uint32_t used_mask = set_used_mask[target_set];
            bool lane_used = (used_mask >> lane_id) & 1;

            // Find empty slots
            bool lane_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMask, lane_empty);

            int32_t candidate_slot = -1;
            int32_t evicted_page = -1;

            if (empty_mask != 0) {
                int candidate_way = __ffs(empty_mask) - 1;
                candidate_slot = set_base + candidate_way;
            } else {
                // Need to evict: find slot with minimum stamp that isn't used
                bool lane_evictable = !lane_used;
                unsigned evictable_mask = __ballot_sync(kFullMask, lane_evictable);

                if (evictable_mask != 0) {
                    // Warp reduction to find minimum stamp
                    uint32_t my_stamp = lane_evictable ? lane_stamp : UINT32_MAX;

                    for (int offset = 16; offset > 0; offset /= 2) {
                        uint32_t other_stamp = __shfl_xor_sync(kFullMask, my_stamp, offset);
                        if (other_stamp < my_stamp) {
                            my_stamp = other_stamp;
                        }
                    }
                    uint32_t min_stamp = __shfl_sync(kFullMask, my_stamp, 0);

                    // Find which lane has the minimum
                    bool has_min = lane_evictable && (lane_stamp == min_stamp);
                    unsigned min_mask = __ballot_sync(kFullMask, has_min);
                    if (min_mask != 0) {
                        int best_way = __ffs(min_mask) - 1;
                        candidate_slot = set_base + best_way;
                        evicted_page = __shfl_sync(kFullMask, lane_page, best_way);
                    }
                }
            }

            // Broadcast candidate
            candidate_slot = __shfl_sync(kFullMask, candidate_slot, 0);
            evicted_page = __shfl_sync(kFullMask, evicted_page, 0);

            if (candidate_slot < 0) {
                // No available slot, release lock and try next set
                if (lane_id == 0) {
                    atomicAdd(&set_version[target_set], 1);
                }
                __syncwarp();
                continue;
            }

            // Perform the allocation (only lane 0)
            bool alloc_success = false;
            if (lane_id == 0) {
                // Clear old page's forward mapping if evicting
                if (evicted_page >= 0 && evicted_page < MAX_PAGE_ID) {
                    atomicCAS(&cpu_to_gpu_slot_map[evicted_page], candidate_slot, -1);
                }

                // Set our page in the reverse mapping
                gpu_to_cpu_page_map[candidate_slot] = cpu_slot;

                // Set our forward mapping
                int32_t old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, candidate_slot);

                if (old_fwd == -1) {
                    // Success!
                    alloc_success = true;
                    uint32_t new_stamp = atomicAdd(&set_clock[target_set], 1);
                    slot_stamps[candidate_slot] = new_stamp;
                    uint32_t way = candidate_slot - set_base;
                    atomicOr(&set_used_mask[target_set], 1u << way);
                } else {
                    // Someone else allocated - check if valid
                    int32_t other_content = gpu_to_cpu_page_map[old_fwd];
                    if (other_content == cpu_slot) {
                        // Their allocation is valid, revert our slot
                        gpu_to_cpu_page_map[candidate_slot] = evicted_page >= 0 ? evicted_page : -1;
                        if (evicted_page >= 0) {
                            atomicCAS(&cpu_to_gpu_slot_map[evicted_page], -1, candidate_slot);
                        }
                        candidate_slot = old_fwd;
                        evicted_page = -1;
                        alloc_success = false;  // Not owner, but have slot
                    } else {
                        // Their mapping is stale, take it over
                        atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], old_fwd, candidate_slot);
                        alloc_success = true;
                        uint32_t new_stamp = atomicAdd(&set_clock[target_set], 1);
                        slot_stamps[candidate_slot] = new_stamp;
                        uint32_t way = candidate_slot - set_base;
                        atomicOr(&set_used_mask[target_set], 1u << way);
                    }
                }
            }

            // Broadcast results
            alloc_success = __shfl_sync(kFullMask, alloc_success ? 1 : 0, 0);
            candidate_slot = __shfl_sync(kFullMask, candidate_slot, 0);
            evicted_page = __shfl_sync(kFullMask, evicted_page, 0);

            // Release seqlock
            if (lane_id == 0) {
                atomicAdd(&set_version[target_set], 1);
            }
            __syncwarp();

            // === CRITICAL SECTION END ===

            if (candidate_slot >= 0) {
                result_slot = candidate_slot;
                result_is_owner = alloc_success;
                result_evicted = alloc_success ? evicted_page : -1;
                done = true;
            }
        }

        // Broadcast final done status
        done = __shfl_sync(kFullMask, done ? 1 : 0, 0);

        // Write results
        if (lane_id == 0) {
            if (!done) {
                atomicMax(overflow_flag, 1);
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
            } else {
                owners_bitmap[key_idx] = result_is_owner;
                dst_staging_slots[key_idx] = result_slot;
                evicted_cpu_pages[key_idx] = result_evicted;
            }
        }
        __syncwarp();
    }
}

// Kernel to zero per-set data structures
__global__ void zero_hybrid_set_data_kernel(
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

// Kernel to initialize new data structures
__global__ void init_hybrid_structures_kernel(
    uint32_t* __restrict__ slot_stamps,
    uint32_t* __restrict__ set_clock,
    uint32_t* __restrict__ set_version,
    int32_t num_slots,
    int32_t num_sets
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < num_slots; i += stride) {
        slot_stamps[i] = 0;
    }

    for (int32_t i = tid; i < num_sets; i += stride) {
        set_clock[i] = 1;
        set_version[i] = 0;
    }
}

// =============================================================================
// Launcher function
// =============================================================================
void allocate_pages_hybrid(
    at::Tensor src_page_ids,
    at::Tensor sparse_indptr,
    int32_t indptr_last_idx,
    at::Tensor cpu_to_gpu_slot_map,
    at::Tensor gpu_to_cpu_page_map,
    at::Tensor slot_stamps,
    at::Tensor set_clock,
    at::Tensor set_version,
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
    const int32_t num_slots = num_sets * WAYS;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int warps_per_block = 4;
    const int threads_per_block = warps_per_block * kWarpSize;
    const int num_warps = (max_num_pages + kWarpSize - 1) / kWarpSize;
    const int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;

    // Reinterpret int32 tensors as uint32 pointers (bit-compatible)
    uint32_t* slot_stamps_ptr = reinterpret_cast<uint32_t*>(slot_stamps.data_ptr<int32_t>());
    uint32_t* set_clock_ptr = reinterpret_cast<uint32_t*>(set_clock.data_ptr<int32_t>());
    uint32_t* set_version_ptr = reinterpret_cast<uint32_t*>(set_version.data_ptr<int32_t>());
    uint32_t* set_used_mask_ptr = reinterpret_cast<uint32_t*>(set_used_mask.data_ptr<int32_t>());

    // Zero per-round data
    zero_hybrid_set_data_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        set_used_mask_ptr,
        evicted_cpu_pages.data_ptr<int32_t>(),
        overflow_flag.data_ptr<int32_t>(),
        num_sets,
        max_num_pages
    );

    // Main allocation kernel
    allocate_pages_hybrid_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        src_page_ids.data_ptr<int32_t>(),
        cpu_to_gpu_slot_map.data_ptr<int32_t>(),
        gpu_to_cpu_page_map.data_ptr<int32_t>(),
        slot_stamps_ptr,
        set_clock_ptr,
        set_version_ptr,
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

// Helper function to initialize the new Hive structures
void init_hybrid_structures(
    at::Tensor slot_stamps,
    at::Tensor set_clock,
    at::Tensor set_version,
    int32_t num_slots,
    int32_t num_sets
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = (num_slots + threads - 1) / threads;

    uint32_t* slot_stamps_ptr = reinterpret_cast<uint32_t*>(slot_stamps.data_ptr<int32_t>());
    uint32_t* set_clock_ptr = reinterpret_cast<uint32_t*>(set_clock.data_ptr<int32_t>());
    uint32_t* set_version_ptr = reinterpret_cast<uint32_t*>(set_version.data_ptr<int32_t>());

    init_hybrid_structures_kernel<<<blocks, threads, 0, stream>>>(
        slot_stamps_ptr,
        set_clock_ptr,
        set_version_ptr,
        num_slots,
        num_sets
    );
}
