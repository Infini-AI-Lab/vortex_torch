#include "register.h"
#include "hash_functions.cuh"
#include <cstdlib>

// Constants
constexpr int kWarpSizeLruBlock = 32;
constexpr uint32_t kFullMaskLruBlock = 0xFFFFFFFFU;
constexpr int WAYS_LruBlock = 32;
constexpr int SETS_PER_BLOCK_LruBlock = 32;
constexpr int SLOTS_PER_BLOCK_LruBlock = SETS_PER_BLOCK_LruBlock * WAYS_LruBlock;  // 1024

// Shared memory layout per block:
//   int32_t smem_reverse_map[1024]  — 4096 bytes
//   uint8_t smem_ages[1024]         — 1024 bytes
//   bool    smem_used[1024]         — 1024 bytes
//   int32_t smem_lock[32]           — 128 bytes
//   Total: ~6272 bytes per block

// =============================================================================
// Shared memory spin-lock (block-scoped, much cheaper than device semaphore)
// =============================================================================
__device__ inline void smem_lock_acquire(volatile int32_t* lock) {
    while (atomicCAS((int32_t*)lock, 0, 1) != 0) {
        // spin
    }
    __threadfence_block();
}

__device__ inline bool smem_lock_try_acquire(volatile int32_t* lock) {
    bool acquired = (atomicCAS((int32_t*)lock, 0, 1) == 0);
    if (acquired) __threadfence_block();
    return acquired;
}

__device__ inline void smem_lock_release(volatile int32_t* lock) {
    __threadfence_block();
    atomicExch((int32_t*)lock, 0);
}

// Warp-level lock wrappers: only lane 0 does the atomic, result broadcast to all lanes
__device__ inline void warp_smem_lock(volatile int32_t* lock, uint32_t lane_id) {
    if (lane_id == 0) {
        smem_lock_acquire(lock);
    }
    __syncwarp();
}

__device__ inline bool warp_smem_trylock(volatile int32_t* lock, uint32_t lane_id) {
    bool acquired = false;
    if (lane_id == 0) {
        acquired = smem_lock_try_acquire(lock);
    }
    acquired = __shfl_sync(kFullMaskLruBlock, acquired ? 1 : 0, 0);
    __syncwarp();
    return acquired;
}

__device__ inline void warp_smem_unlock(volatile int32_t* lock, uint32_t lane_id) {
    __syncwarp();
    if (lane_id == 0) {
        smem_lock_release(lock);
    }
    __syncwarp();
}

// =============================================================================
// Pre-partition kernel: scatter pages into per-block buckets
// =============================================================================
__global__ void prepartition_pages_lru_block_kernel(
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
// Main v4 kernel: block-local set operations with shared memory
// 32 sets per block, 32 warps (1024 threads) per block
// =============================================================================
__global__ void allocate_pages_lru_lru_block_kernel(
    int32_t* __restrict__ cpu_to_gpu_slot_map,
    int32_t* __restrict__ gpu_to_cpu_page_map,
    uint8_t* __restrict__ slot_ages,
    bool* __restrict__ set_slot_used_bitmap,
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
    __shared__ int32_t smem_reverse_map[SLOTS_PER_BLOCK_LruBlock];
    __shared__ uint8_t smem_ages[SLOTS_PER_BLOCK_LruBlock];
    __shared__ bool    smem_used[SLOTS_PER_BLOCK_LruBlock];
    __shared__ int32_t smem_lock[SETS_PER_BLOCK_LruBlock];

    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_in_block = threadIdx.x / kWarpSizeLruBlock;
    const uint32_t lane_id = threadIdx.x % kWarpSizeLruBlock;
    const uint32_t num_warps_in_block = blockDim.x / kWarpSizeLruBlock;

    const uint32_t set_base_global = block_id * SETS_PER_BLOCK_LruBlock;
    const uint32_t slot_base_global = set_base_global * WAYS_LruBlock;

    // === BLOCK INIT: Load global state into shared memory ===
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_LruBlock; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        smem_reverse_map[i] = gpu_to_cpu_page_map[global_slot];
        smem_ages[i] = slot_ages[global_slot];
        smem_used[i] = set_slot_used_bitmap[global_slot];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_LruBlock; i += blockDim.x) {
        smem_lock[i] = 0;
    }
    __syncthreads();

    const int32_t my_count = block_counts[block_id];
    const int32_t* my_page_ids = block_page_ids + block_id * max_pages_per_block;
    const int32_t* my_page_indices = block_page_indices + block_id * max_pages_per_block;

    // =====================================================================
    // PASS 1: HIT CHECK + EMPTY SLOT ALLOCATION (no eviction)
    // All hits are resolved first, setting used_this_round bits.
    // This prevents the race where a hit page gets evicted before its
    // used bit is set.
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
            uint32_t hit_set_global = (uint32_t)(existing_slot / WAYS_LruBlock);
            uint32_t local_set = hit_set_global - set_base_global;
            uint32_t local_slot_base_hit = local_set * WAYS_LruBlock;

            warp_smem_lock(&smem_lock[local_set], lane_id);

            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck == existing_slot) {
                uint32_t hit_way = existing_slot - hit_set_global * WAYS_LruBlock;

                uint8_t lane_age = smem_ages[local_slot_base_hit + lane_id];
                uint8_t hit_old_age = __shfl_sync(kFullMaskLruBlock, lane_age, hit_way);

                if (lane_id == hit_way) {
                    lane_age = WAYS_LruBlock;
                    smem_used[local_slot_base_hit + lane_id] = true;
                } else if (lane_age > hit_old_age && lane_age > 0) {
                    lane_age -= 1;
                }
                smem_ages[local_slot_base_hit + lane_id] = lane_age;
                __syncwarp();

                result_slot = existing_slot;
                is_owner = false;
                done = true;
            }
            warp_smem_unlock(&smem_lock[local_set], lane_id);
        }

        // --- TRY EMPTY SLOT (no eviction) ---
        if (!done) {
            for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
                uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
                uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_LruBlock);
                uint32_t global_set = set_base_global + local_set;
                uint32_t local_slot_base = local_set * WAYS_LruBlock;
                uint32_t global_slot_base = global_set * WAYS_LruBlock;

                bool locked;
                if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                    locked = warp_smem_trylock(&smem_lock[local_set], lane_id);
                    if (!locked) continue;
                } else {
                    warp_smem_lock(&smem_lock[local_set], lane_id);
                    locked = true;
                }

                // Re-check forward map (another warp may have allocated this page)
                int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
                if (recheck >= 0) {
                    warp_smem_unlock(&smem_lock[local_set], lane_id);
                    result_slot = recheck;
                    is_owner = false;
                    done = true;
                    break;
                }

                // Try empty slot only
                uint8_t lane_age = smem_ages[local_slot_base + lane_id];
                bool lane_is_empty = (lane_age == 0);
                unsigned empty_mask = __ballot_sync(kFullMaskLruBlock, lane_is_empty);

                if (empty_mask != 0) {
                    int insert_way = __ffs((int)empty_mask) - 1;
                    int32_t new_global_slot = global_slot_base + insert_way;

                    bool cas_success = false;
                    int32_t old_fwd = -1;
                    if ((int32_t)lane_id == insert_way) {
                        old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_global_slot);
                        cas_success = (old_fwd == -1);
                    }
                    cas_success = __shfl_sync(kFullMaskLruBlock, cas_success ? 1 : 0, insert_way);
                    old_fwd = __shfl_sync(kFullMaskLruBlock, old_fwd, insert_way);

                    if (cas_success) {
                        if ((int32_t)lane_id == insert_way) {
                            smem_reverse_map[local_slot_base + insert_way] = cpu_slot;
                            lane_age = WAYS_LruBlock;
                            smem_used[local_slot_base + insert_way] = true;
                        } else if (lane_age > 1) {
                            lane_age -= 1;
                        }
                        smem_ages[local_slot_base + lane_id] = lane_age;
                        __syncwarp();

                        warp_smem_unlock(&smem_lock[local_set], lane_id);
                        result_slot = new_global_slot;
                        is_owner = true;
                        done = true;
                        break;
                    } else {
                        warp_smem_unlock(&smem_lock[local_set], lane_id);
                        result_slot = old_fwd;
                        is_owner = false;
                        done = true;
                        break;
                    }
                }

                warp_smem_unlock(&smem_lock[local_set], lane_id);
                // No empty slot in this set, try next hash attempt
            }
        }

        // Write Pass 1 results
        if (lane_id == 0) {
            if (done) {
                owners_bitmap[key_idx] = is_owner;
                dst_staging_slots[key_idx] = result_slot;
                evicted_cpu_pages[key_idx] = -1;
            } else {
                // Mark as needing eviction in Pass 2
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
                evicted_cpu_pages[key_idx] = -1;
            }
        }
        __syncwarp();
    }

    // =====================================================================
    // BARRIER: All hits resolved, used_this_round fully populated.
    // Evictions in Pass 2 will respect used_this_round bits.
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
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_LruBlock);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_LruBlock;
            uint32_t global_slot_base = global_set * WAYS_LruBlock;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock(&smem_lock[local_set], lane_id);
                locked = true;
            }

            // Re-check forward map (may have been allocated by another warp)
            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            // --- Try empty slot first ---
            uint8_t lane_age = smem_ages[local_slot_base + lane_id];
            bool lane_is_empty = (lane_age == 0);
            unsigned empty_mask = __ballot_sync(kFullMaskLruBlock, lane_is_empty);

            if (empty_mask != 0) {
                int insert_way = __ffs((int)empty_mask) - 1;
                int32_t new_global_slot = global_slot_base + insert_way;

                bool cas_success = false;
                int32_t old_fwd = -1;
                if ((int32_t)lane_id == insert_way) {
                    old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_global_slot);
                    cas_success = (old_fwd == -1);
                }
                cas_success = __shfl_sync(kFullMaskLruBlock, cas_success ? 1 : 0, insert_way);
                old_fwd = __shfl_sync(kFullMaskLruBlock, old_fwd, insert_way);

                if (cas_success) {
                    if ((int32_t)lane_id == insert_way) {
                        smem_reverse_map[local_slot_base + insert_way] = cpu_slot;
                        lane_age = WAYS_LruBlock;
                        smem_used[local_slot_base + insert_way] = true;
                    } else if (lane_age > 1) {
                        lane_age -= 1;
                    }
                    smem_ages[local_slot_base + lane_id] = lane_age;
                    __syncwarp();

                    warp_smem_unlock(&smem_lock[local_set], lane_id);
                    result_slot = new_global_slot;
                    is_owner = true;
                    done = true;
                    break;
                } else {
                    warp_smem_unlock(&smem_lock[local_set], lane_id);
                    result_slot = old_fwd;
                    is_owner = false;
                    done = true;
                    break;
                }
            }

            // --- No empty slot: evict LRU ---
            bool lane_not_used = !smem_used[local_slot_base + lane_id];
            int insert_way = -1;

            for (uint8_t target_age = 0; target_age <= WAYS_LruBlock && insert_way == -1; ++target_age) {
                unsigned age_mask = __ballot_sync(kFullMaskLruBlock, lane_age == target_age && lane_not_used);
                if (age_mask != 0) {
                    insert_way = __ffs((int)age_mask) - 1;
                    break;
                }
            }

            if (insert_way == -1) {
                warp_smem_unlock(&smem_lock[local_set], lane_id);
                continue;
            }

            int32_t old_cpu_slot = smem_reverse_map[local_slot_base + insert_way];
            int32_t new_global_slot = global_slot_base + insert_way;

            bool cas_success = false;
            int32_t old_fwd = -1;
            if ((int32_t)lane_id == insert_way) {
                if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
                    atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], new_global_slot, -1);
                }
                old_fwd = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, new_global_slot);
                cas_success = (old_fwd == -1);
            }
            cas_success = __shfl_sync(kFullMaskLruBlock, cas_success ? 1 : 0, insert_way);
            old_fwd = __shfl_sync(kFullMaskLruBlock, old_fwd, insert_way);

            if (cas_success) {
                result_evicted = old_cpu_slot;
                if ((int32_t)lane_id == insert_way) {
                    smem_reverse_map[local_slot_base + insert_way] = cpu_slot;
                    lane_age = WAYS_LruBlock;
                    smem_used[local_slot_base + insert_way] = true;
                } else if (lane_age > 1) {
                    lane_age -= 1;
                }
                smem_ages[local_slot_base + lane_id] = lane_age;
                __syncwarp();

                warp_smem_unlock(&smem_lock[local_set], lane_id);
                result_slot = new_global_slot;
                is_owner = true;
                done = true;
            } else {
                if ((int32_t)lane_id == insert_way) {
                    if (old_cpu_slot >= 0 && old_cpu_slot < MAX_PAGE_ID) {
                        atomicCAS(&cpu_to_gpu_slot_map[old_cpu_slot], -1, new_global_slot);
                    }
                    smem_reverse_map[local_slot_base + insert_way] = old_cpu_slot;
                }
                __syncwarp();

                warp_smem_unlock(&smem_lock[local_set], lane_id);
                result_slot = old_fwd;
                is_owner = false;
                done = true;
            }
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
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_LruBlock; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        gpu_to_cpu_page_map[global_slot] = smem_reverse_map[i];
        slot_ages[global_slot] = smem_ages[i];
        set_slot_used_bitmap[global_slot] = smem_used[i];
    }
}

// Fused kernel to zero bitmaps and initialize outputs
__global__ void zero_allocation_bitmaps_lru_block_kernel(
    bool* __restrict__ set_slot_used_bitmap,
    int32_t* __restrict__ evicted_cpu_pages,
    int32_t* __restrict__ overflow_flag,
    int32_t* __restrict__ block_counts,
    int32_t num_slots,
    int32_t max_num_pages,
    int32_t num_blocks
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = tid; i < num_slots; i += stride) {
        set_slot_used_bitmap[i] = false;
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
// Static temp buffer cache for v4
// =============================================================================
static int32_t* cached_block_page_ids_lru_block = nullptr;
static int32_t* cached_block_page_indices_lru_block = nullptr;
static int32_t* cached_block_counts_lru_block = nullptr;
static int32_t cached_num_blocks_lru_block = 0;
static int32_t cached_max_pages_per_block_lru_block = 0;

static void cleanup_lru_block_cache() {
    if (cached_block_page_ids_lru_block) cudaFree(cached_block_page_ids_lru_block);
    if (cached_block_page_indices_lru_block) cudaFree(cached_block_page_indices_lru_block);
    if (cached_block_counts_lru_block) cudaFree(cached_block_counts_lru_block);
    cached_block_page_ids_lru_block = nullptr;
    cached_block_page_indices_lru_block = nullptr;
    cached_block_counts_lru_block = nullptr;
    cached_num_blocks_lru_block = 0;
    cached_max_pages_per_block_lru_block = 0;
}

// =============================================================================
// Launcher
// =============================================================================
void allocate_pages_lru_block(
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
    const int32_t num_sets = gpu_to_cpu_page_map.size(0) / WAYS_LruBlock;
    const int32_t num_slots = num_sets * WAYS_LruBlock;
    const int32_t num_blocks = num_sets / SETS_PER_BLOCK_LruBlock;

    TORCH_CHECK(num_sets % SETS_PER_BLOCK_LruBlock == 0,
        "num_sets (", num_sets, ") must be divisible by SETS_PER_BLOCK (", SETS_PER_BLOCK_LruBlock, ")");

    // max_pages_per_block: upper bound for pages that hash to one block
    const int32_t max_pages_per_block = max(max_num_pages * 4 / num_blocks, 64);

    // Allocate/reuse temp buffers
    if (cached_num_blocks_lru_block != num_blocks || cached_max_pages_per_block_lru_block != max_pages_per_block) {
        cleanup_lru_block_cache();

        size_t buf_size = (size_t)num_blocks * max_pages_per_block * sizeof(int32_t);
        cudaMalloc(&cached_block_page_ids_lru_block, buf_size);
        cudaMalloc(&cached_block_page_indices_lru_block, buf_size);
        cudaMalloc(&cached_block_counts_lru_block, num_blocks * sizeof(int32_t));

        cached_num_blocks_lru_block = num_blocks;
        cached_max_pages_per_block_lru_block = max_pages_per_block;

        static bool cleanup_registered = false;
        if (!cleanup_registered) {
            std::atexit(cleanup_lru_block_cache);
            cleanup_registered = true;
        }
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Zero bitmaps + block_counts
    {
        const int threads = 256;
        int max_items = max(max(num_slots, max_num_pages), num_blocks);
        const int blocks = (max_items + threads - 1) / threads;
        zero_allocation_bitmaps_lru_block_kernel<<<blocks, threads, 0, stream>>>(
            set_slot_used_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_counts_lru_block,
            num_slots,
            max_num_pages,
            num_blocks
        );
    }

    // Pre-partition
    {
        const int threads = 256;
        const int blocks = (max_num_pages + threads - 1) / threads;
        prepartition_pages_lru_block_kernel<<<blocks, threads, 0, stream>>>(
            src_page_ids.data_ptr<int32_t>(),
            sparse_indptr.data_ptr<int32_t>(),
            indptr_last_idx,
            cached_block_page_ids_lru_block,
            cached_block_page_indices_lru_block,
            cached_block_counts_lru_block,
            num_blocks,
            max_pages_per_block
        );
    }

    // Main kernel: 32 warps (1024 threads) per block
    {
        const int threads_per_block = SETS_PER_BLOCK_LruBlock * kWarpSizeLruBlock;  // 1024
        allocate_pages_lru_lru_block_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            cpu_to_gpu_slot_map.data_ptr<int32_t>(),
            gpu_to_cpu_page_map.data_ptr<int32_t>(),
            slot_ages.data_ptr<uint8_t>(),
            set_slot_used_bitmap.data_ptr<bool>(),
            dst_staging_slots.data_ptr<int32_t>(),
            owners_bitmap.data_ptr<bool>(),
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_page_ids_lru_block,
            cached_block_page_indices_lru_block,
            cached_block_counts_lru_block,
            MAX_PAGE_ID,
            num_sets,
            num_blocks,
            max_pages_per_block,
            MAX_HASH_ATTEMPTS
        );
    }
}
