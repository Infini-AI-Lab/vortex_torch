#include "register.h"
#include "hash_functions.cuh"
#include <cstdlib>

// Constants — identical to lru_block_global_alloc.cu
constexpr int kWarpSizeLruBlock = 32;
constexpr uint32_t kFullMaskLruBlock = 0xFFFFFFFFU;
constexpr int WAYS_LruBlock = 32;
constexpr int SETS_PER_BLOCK_LruBlock = 32;
constexpr int SLOTS_PER_BLOCK_LruBlock = SETS_PER_BLOCK_LruBlock * WAYS_LruBlock;  // 1024

// Shared memory layout per block (same as lru_block_global):
//   int32_t  smem_reverse_map[1024]  — 4096 bytes
//   uint8_t  smem_ages[1024]         — 1024 bytes
//   uint32_t smem_used_mask[32]      — 128 bytes
//   int32_t  smem_lock[32]           — 128 bytes
//   Total: ~5376 bytes per block

// =============================================================================
// Shared memory spin-lock (block-scoped)
// =============================================================================
__device__ inline void smem_lock_acquire_lb(volatile int32_t* lock) {
    while (atomicCAS((int32_t*)lock, 0, 1) != 0) {}
    __threadfence_block();
}

__device__ inline bool smem_lock_try_acquire_lb(volatile int32_t* lock) {
    bool acquired = (atomicCAS((int32_t*)lock, 0, 1) == 0);
    if (acquired) __threadfence_block();
    return acquired;
}

__device__ inline void smem_lock_release_lb(volatile int32_t* lock) {
    __threadfence_block();
    atomicExch((int32_t*)lock, 0);
}

__device__ inline void warp_smem_lock_lb(volatile int32_t* lock, uint32_t lane_id) {
    if (lane_id == 0) smem_lock_acquire_lb(lock);
    __syncwarp();
}

__device__ inline bool warp_smem_trylock_lb(volatile int32_t* lock, uint32_t lane_id) {
    bool acquired = false;
    if (lane_id == 0) acquired = smem_lock_try_acquire_lb(lock);
    acquired = __shfl_sync(kFullMaskLruBlock, acquired ? 1 : 0, 0);
    __syncwarp();
    return acquired;
}

__device__ inline void warp_smem_unlock_lb(volatile int32_t* lock, uint32_t lane_id) {
    __syncwarp();
    if (lane_id == 0) smem_lock_release_lb(lock);
    __syncwarp();
}

// =============================================================================
// Warp-cooperative age update (same as lru_block_global)
// =============================================================================
__device__ inline void warp_update_ages_smem_lb(
    uint8_t* smem_ages_base, uint32_t target_way, uint32_t lane_id
) {
    uint8_t my_age = smem_ages_base[lane_id];
    uint8_t old_age = __shfl_sync(kFullMaskLruBlock, my_age, target_way);
    if (lane_id == target_way) {
        my_age = WAYS_LruBlock;
    } else if (my_age > old_age && my_age > 0) {
        my_age -= 1;
    }
    smem_ages_base[lane_id] = my_age;
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
// Main kernel: block-local only, with all block_global optimizations
// (uint32 bitmask, butterfly reduction, warp-cooperative age update)
// No out-of-block hit handling, no global fallback.
// =============================================================================
__global__ void allocate_pages_lru_lru_block_kernel(
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
    __shared__ int32_t  smem_reverse_map[SLOTS_PER_BLOCK_LruBlock];
    __shared__ uint8_t  smem_ages[SLOTS_PER_BLOCK_LruBlock];
    __shared__ uint32_t smem_used_mask[SETS_PER_BLOCK_LruBlock];
    __shared__ int32_t  smem_lock[SETS_PER_BLOCK_LruBlock];

    const uint32_t block_id = blockIdx.x;
    const uint32_t warp_in_block = threadIdx.x / kWarpSizeLruBlock;
    const uint32_t lane_id = threadIdx.x % kWarpSizeLruBlock;
    const uint32_t num_warps_in_block = blockDim.x / kWarpSizeLruBlock;

    const uint32_t set_base_global = block_id * SETS_PER_BLOCK_LruBlock;
    const uint32_t slot_base_global = set_base_global * WAYS_LruBlock;

    // === BLOCK INIT ===
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_LruBlock; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        smem_reverse_map[i] = gpu_to_cpu_page_map[global_slot];
        smem_ages[i] = slot_ages[global_slot];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_LruBlock; i += blockDim.x) {
        smem_used_mask[i] = 0;
        smem_lock[i] = 0;
    }
    __syncthreads();

    const int32_t my_count = block_counts[block_id];
    const int32_t* my_page_ids = block_page_ids + block_id * max_pages_per_block;
    const int32_t* my_page_indices = block_page_indices + block_id * max_pages_per_block;

    // =====================================================================
    // PASS 1: HIT CHECK + EMPTY SLOT (no eviction)
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
            uint32_t hit_way = existing_slot - hit_set_global * WAYS_LruBlock;

            // Block-only: only handle in-block hits
            if (hit_set_global >= set_base_global &&
                hit_set_global < set_base_global + SETS_PER_BLOCK_LruBlock) {
                uint32_t local_set = hit_set_global - set_base_global;
                warp_smem_lock_lb(&smem_lock[local_set], lane_id);
                warp_update_ages_smem_lb(&smem_ages[local_set * WAYS_LruBlock], hit_way, lane_id);
                if (lane_id == 0) {
                    atomicOr(&smem_used_mask[local_set], 1u << hit_way);
                }
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
            }
            // Out-of-block hits: no global set_used_mask update (block-only limitation)

            if (lane_id == 0) {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = existing_slot;
                evicted_cpu_pages[key_idx] = -1;
            }
            __syncwarp();
            continue;
        }

        // --- TRY EMPTY SLOT ---
        for (int hash_attempt = 0; hash_attempt < MAX_HASH_ATTEMPTS && !done; ++hash_attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, hash_attempt);
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_LruBlock);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_LruBlock;
            uint32_t global_slot_base = global_set * WAYS_LruBlock;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock_lb(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock_lb(&smem_lock[local_set], lane_id);
                locked = true;
            }

            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            int32_t lane_page = smem_reverse_map[local_slot_base + lane_id];
            bool lane_is_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruBlock, lane_is_empty);

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
                cas_success = __shfl_sync(kFullMaskLruBlock, cas_success ? 1 : 0, 0);
                if (cas_success) {
                    warp_update_ages_smem_lb(&smem_ages[local_slot_base], insert_way, lane_id);
                }
                done = true;
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
                break;
            }

            warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
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
    // PASS 2: EVICTION (only for unresolved pages)
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
            uint32_t local_set = (uint32_t)(h % (uint64_t)SETS_PER_BLOCK_LruBlock);
            uint32_t global_set = set_base_global + local_set;
            uint32_t local_slot_base = local_set * WAYS_LruBlock;
            uint32_t global_slot_base = global_set * WAYS_LruBlock;

            bool locked;
            if (hash_attempt < MAX_HASH_ATTEMPTS - 1) {
                locked = warp_smem_trylock_lb(&smem_lock[local_set], lane_id);
                if (!locked) continue;
            } else {
                warp_smem_lock_lb(&smem_lock[local_set], lane_id);
                locked = true;
            }

            int32_t recheck = cpu_to_gpu_slot_map[cpu_slot];
            if (recheck >= 0) {
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
                result_slot = recheck;
                is_owner = false;
                done = true;
                break;
            }

            // Try empty slot first
            int32_t lane_page = smem_reverse_map[local_slot_base + lane_id];
            uint8_t lane_age = smem_ages[local_slot_base + lane_id];
            bool lane_is_empty = (lane_page == -1);
            unsigned empty_mask = __ballot_sync(kFullMaskLruBlock, lane_is_empty);

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
                cas_success = __shfl_sync(kFullMaskLruBlock, cas_success ? 1 : 0, 0);
                if (cas_success) {
                    warp_update_ages_smem_lb(&smem_ages[local_slot_base], insert_way, lane_id);
                }
                done = true;
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
                break;
            }

            // --- Eviction via butterfly min-reduction ---
            uint32_t used_mask = smem_used_mask[local_set];
            bool lane_evictable = !((used_mask >> lane_id) & 1);
            unsigned evictable_mask = __ballot_sync(kFullMaskLruBlock, lane_evictable);

            if (evictable_mask == 0) {
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
                continue;
            }

            // Butterfly reduction to find minimum age
            uint32_t my_age = lane_evictable ? (uint32_t)lane_age : UINT32_MAX;
            for (int offset = 16; offset > 0; offset /= 2) {
                uint32_t other_age = __shfl_xor_sync(kFullMaskLruBlock, my_age, offset);
                if (other_age < my_age) my_age = other_age;
            }
            uint32_t min_age = __shfl_sync(kFullMaskLruBlock, my_age, 0);

            bool has_min = lane_evictable && ((uint32_t)lane_age == min_age);
            unsigned min_mask = __ballot_sync(kFullMaskLruBlock, has_min);

            if (min_mask == 0) {
                warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
                continue;
            }

            int insert_way = __ffs((int)min_mask) - 1;
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
            cas_success = __shfl_sync(kFullMaskLruBlock, cas_success ? 1 : 0, 0);
            if (cas_success) {
                warp_update_ages_smem_lb(&smem_ages[local_slot_base], insert_way, lane_id);
            }
            done = true;
            warp_smem_unlock_lb(&smem_lock[local_set], lane_id);
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

    // === BLOCK END ===
    __syncthreads();
    for (int i = threadIdx.x; i < SLOTS_PER_BLOCK_LruBlock; i += blockDim.x) {
        int32_t global_slot = slot_base_global + i;
        gpu_to_cpu_page_map[global_slot] = smem_reverse_map[i];
        slot_ages[global_slot] = smem_ages[i];
    }
    for (int i = threadIdx.x; i < SETS_PER_BLOCK_LruBlock; i += blockDim.x) {
        set_used_mask_global[set_base_global + i] = smem_used_mask[i];
    }
}

// =============================================================================
// Zero kernel
// =============================================================================
__global__ void zero_allocation_bitmaps_lru_block_kernel(
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
// Launcher — now uses set_used_mask (int32/uint32) instead of bool bitmaps
// =============================================================================
void allocate_pages_lru_block(
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
    const int32_t num_sets = gpu_to_cpu_page_map.size(0) / WAYS_LruBlock;
    const int32_t num_slots = num_sets * WAYS_LruBlock;
    const int32_t num_blocks = num_sets / SETS_PER_BLOCK_LruBlock;

    TORCH_CHECK(num_sets % SETS_PER_BLOCK_LruBlock == 0,
        "num_sets (", num_sets, ") must be divisible by SETS_PER_BLOCK (", SETS_PER_BLOCK_LruBlock, ")");

    const int32_t max_pages_per_block = max(max_num_pages * 4 / num_blocks, 64);

    uint32_t* set_used_mask_ptr = reinterpret_cast<uint32_t*>(set_used_mask.data_ptr<int32_t>());

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
        int max_items = max(max(num_sets, max_num_pages), num_blocks);
        const int blocks = (max_items + threads - 1) / threads;
        zero_allocation_bitmaps_lru_block_kernel<<<blocks, threads, 0, stream>>>(
            set_used_mask_ptr,
            evicted_cpu_pages.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            cached_block_counts_lru_block,
            num_sets,
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

    // Main kernel — block-only, no global fallback
    {
        const int threads_per_block = SETS_PER_BLOCK_LruBlock * kWarpSizeLruBlock;  // 1024
        allocate_pages_lru_lru_block_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            cpu_to_gpu_slot_map.data_ptr<int32_t>(),
            gpu_to_cpu_page_map.data_ptr<int32_t>(),
            slot_ages.data_ptr<uint8_t>(),
            set_used_mask_ptr,
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
