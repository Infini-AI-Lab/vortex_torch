#include "register.h"
#include "hash_functions.cuh"

constexpr int kWarpSize = 32;
constexpr uint32_t kFullMask = 0xFFFFFFFFU;

__device__ __forceinline__ int32_t atomic_load(int32_t* ptr) { return atomicAdd(ptr, 0); }

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
    const uint32_t global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const uint32_t lane_id = threadIdx.x % kWarpSize;
    const uint32_t num_warps = gridDim.x * blockDim.x / kWarpSize;

    // Warp-stride loop
    for (uint32_t key_idx = global_warp_id; key_idx < N; key_idx += num_warps) {
        const int32_t cpu_slot = src_page_ids[key_idx];
        bool valid_req = (cpu_slot >= 0 && cpu_slot < MAX_PAGE_ID);

        // Result state
        int32_t res_slot = -1, res_evict = -1;
        bool res_owner = false, done = !__shfl_sync(kFullMask, valid_req, 0);

        // =====================================================================
        // PHASE 1: OPTIMISTIC CHECK (Optimized: Lane 0 Access Only)
        // =====================================================================
        
        // 1. Initial Map Check (Lane 0 loads, Broadcasts)
        int32_t existing = -1;
        if (lane_id == 0) existing = atomic_load(&cpu_to_gpu_slot_map[cpu_slot]);
        existing = __shfl_sync(kFullMask, existing, 0);

        if (!done && existing >= 0) {
            uint32_t set_idx = existing / WAYS;
            
            // 2. Lock Version Check (Lane 0 loads, Broadcasts)
            uint32_t v1 = 1; // Default to locked/odd to fail safely if not loaded
            if (lane_id == 0) v1 = atomic_load((int32_t*)&set_version[set_idx]);
            v1 = __shfl_sync(kFullMask, v1, 0);
            
            // Only proceed if unlocked (even)
            if ((v1 & 1) == 0) {
                // 3. Verify Reverse Map (Lane 0 loads, Broadcasts)
                int32_t content = -1;
                if (lane_id == 0) content = atomic_load(&gpu_to_cpu_page_map[existing]);
                
                if (__shfl_sync(kFullMask, content, 0) == cpu_slot) {
                    // Optimistic Update (Lane 0 Only)
                    if (lane_id == 0) {
                        atomicAdd(&set_clock[set_idx], 1);
                        // Approximate stamp read is fine here
                        slot_stamps[existing] = atomicAdd(&set_clock[set_idx], 0); 
                        atomicOr(&set_used_mask[set_idx], 1u << (existing % WAYS));
                    }
                    __syncwarp();
                    __threadfence(); // Mandatory memory barrier for race fix

                    // 4. Post-Update Lock Check (Lane 0 loads, Broadcasts)
                    uint32_t v2 = 0;
                    if (lane_id == 0) v2 = atomic_load((int32_t*)&set_version[set_idx]);
                    v2 = __shfl_sync(kFullMask, v2, 0);

                    // If version matched (no eviction occurred during update)
                    if (v1 == v2) {
                        // 5. Final Verify (Lane 0 loads, Broadcasts)
                        if (lane_id == 0) content = atomic_load(&gpu_to_cpu_page_map[existing]);
                        
                        if (__shfl_sync(kFullMask, content, 0) == cpu_slot) {
                            res_slot = existing;
                            done = true;
                        }
                    }
                }
            }
        }

        // =====================================================================
        // PHASE 2: LOCKED ALLOCATION (Optimized)
        // =====================================================================
        for (int attempt = 0; attempt < MAX_HASH_ATTEMPTS && !__shfl_sync(kFullMask, done, 0); ++attempt) {
            uint64_t h = apply_hash((uint64_t)cpu_slot, attempt);
            uint32_t set = (uint32_t)(h % num_sets);
            uint32_t base = set * WAYS;

            // Try Acquire Lock (Lane 0 Only)
            bool locked = false;
            if (lane_id == 0) {
                uint32_t v = atomic_load((int32_t*)&set_version[set]);
                if (!(v & 1) && atomicCAS(&set_version[set], v, v + 1) == v) locked = true;
            }
            if (!__shfl_sync(kFullMask, locked, 0)) continue; // Spin/Retry next hash

            // --- CRITICAL SECTION ---
            
            // 6. Re-check Map inside Lock (Lane 0 loads, Broadcasts)
            // Fixes redundant atomics from all lanes
            if (lane_id == 0) existing = atomic_load(&cpu_to_gpu_slot_map[cpu_slot]);
            existing = __shfl_sync(kFullMask, existing, 0);

            if (existing >= 0) {
                if (lane_id == 0) atomicAdd(&set_version[set], 1); // Release
                res_slot = existing; 
                done = true; 
                continue;
            }

            // 7. Candidate Selection (Coalesced Loads - safe for all lanes)
            int32_t pg = gpu_to_cpu_page_map[base + lane_id];
            bool is_empty = (pg == -1);
            bool is_used  = (set_used_mask[set] >> lane_id) & 1;
            uint32_t stamp = slot_stamps[base + lane_id];
            
            uint32_t min_stamp = is_used ? 0xFFFFFFFF : stamp;
            
            // Warp reduction for LRU
            for (int i = 16; i > 0; i /= 2) min_stamp = min(min_stamp, __shfl_xor_sync(kFullMask, min_stamp, i));
            min_stamp = __shfl_sync(kFullMask, min_stamp, 0);

            // Determine winner
            int winner_lane = -1;
            int empty_mask = __ballot_sync(kFullMask, is_empty);
            if (empty_mask) {
                winner_lane = __ffs(empty_mask) - 1;
            } else {
                int lru_mask = __ballot_sync(kFullMask, !is_used && (stamp == min_stamp));
                if (lru_mask) winner_lane = __ffs(lru_mask) - 1;
            }

            // Shuffle winner data (Safe broadcast, no memory access)
            int32_t cand_slot = (winner_lane >= 0) ? (base + winner_lane) : -1;
            int32_t cand_page = (winner_lane >= 0) ? __shfl_sync(kFullMask, pg, winner_lane) : -1;

            // 8. Perform Swap (Lane 0 Only)
            if (cand_slot != -1 && lane_id == 0) {
                if (cand_page >= 0) atomicCAS(&cpu_to_gpu_slot_map[cand_page], cand_slot, -1);
                
                gpu_to_cpu_page_map[cand_slot] = cpu_slot;
                __threadfence(); // Visibility barrier

                int32_t old = atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], -1, cand_slot);
                
                if (old == -1) {
                    res_owner = true;
                } else {
                    // Collision: Check if we can steal or must revert
                    // Lane 0 Load + no broadcast needed (local logic)
                    int32_t other_content = atomic_load(&gpu_to_cpu_page_map[old]); 
                    
                    if (other_content == cpu_slot) {
                        // Revert
                        gpu_to_cpu_page_map[cand_slot] = cand_page;
                        if (cand_page >= 0) atomicCAS(&cpu_to_gpu_slot_map[cand_page], -1, cand_slot);
                        cand_slot = old; 
                        cand_page = -1;
                    } else {
                        // Steal
                        atomicCAS(&cpu_to_gpu_slot_map[cpu_slot], old, cand_slot);
                        res_owner = true;
                    }
                }
                
                if (res_owner) {
                     slot_stamps[cand_slot] = atomicAdd(&set_clock[set], 1);
                     atomicOr(&set_used_mask[set], 1u << (cand_slot % WAYS));
                }
            }

            // Broadcast final results
            cand_slot = __shfl_sync(kFullMask, cand_slot, 0);
            if (cand_slot != -1) {
                res_slot = cand_slot;
                res_evict = (res_owner) ? cand_page : -1;
                done = true;
            }
            
            // Release Lock (Lane 0)
            if (lane_id == 0) atomicAdd(&set_version[set], 1);
            __syncwarp();
        }

        // =====================================================================
        // WRITE BACK (Lane 0 Only)
        // =====================================================================
        if (lane_id == 0) {
            if (done && res_slot >= 0) {
                owners_bitmap[key_idx] = res_owner;
                dst_staging_slots[key_idx] = res_slot;
                evicted_cpu_pages[key_idx] = res_evict;
            } else if (valid_req) {
                owners_bitmap[key_idx] = false;
                dst_staging_slots[key_idx] = -1;
                atomicMax(overflow_flag, 1);
            }
        }
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

// Helper function to initialize the new hybrid structures
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
