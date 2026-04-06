#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// =============================================================================
// Cache replacement policy abstraction.
// All functions branch on a runtime `policy` parameter.
// Since all threads in a warp use the same policy, there is no divergence.
// =============================================================================

enum CachePolicy : int32_t {
    POLICY_LRU    = 0,
    POLICY_LFU    = 1,
    POLICY_RANDOM = 2,
};

// Number of ways per set (must match allocation kernel constants)
constexpr int POLICY_WAYS = 32;
constexpr uint32_t POLICY_FULL_MASK = 0xFFFFFFFFU;

// =============================================================================
// State update on hit or insertion.
//
// LRU:    promote target to max age, demote siblings with higher age
// LFU:    increment target frequency (cap at 255), no change to others
// RANDOM: no-op (no state to maintain)
// =============================================================================
__device__ inline void policy_update_state(
    int32_t policy,
    uint8_t* smem_state,
    uint32_t target_way,
    uint32_t lane_id
) {
    if (policy == POLICY_LRU) {
        uint8_t my_val = smem_state[lane_id];
        uint8_t old_val = __shfl_sync(POLICY_FULL_MASK, my_val, target_way);
        if (lane_id == target_way) {
            my_val = POLICY_WAYS;
        } else if (my_val > old_val && my_val > 0) {
            my_val -= 1;
        }
        smem_state[lane_id] = my_val;
    } else if (policy == POLICY_LFU) {
        if (lane_id == target_way) {
            uint8_t freq = smem_state[lane_id];
            smem_state[lane_id] = (freq < 255) ? freq + 1 : 255;
        }
    }
    // POLICY_RANDOM: no-op
}

// State update for the global fallback kernel (operates on global memory).
__device__ inline void policy_update_state_global(
    int32_t policy,
    uint8_t* slot_ages,
    uint32_t set_base,
    uint32_t target_way,
    uint32_t lane_id,
    uint32_t WAYS
) {
    if (policy == POLICY_LRU) {
        uint8_t my_val = slot_ages[set_base + lane_id];
        uint8_t old_val = __shfl_sync(POLICY_FULL_MASK, my_val, target_way);
        if ((int32_t)lane_id == (int32_t)target_way) {
            my_val = WAYS;
        } else if (my_val > old_val && my_val > 0) {
            my_val -= 1;
        }
        slot_ages[set_base + lane_id] = my_val;
    } else if (policy == POLICY_LFU) {
        if ((int32_t)lane_id == (int32_t)target_way) {
            uint8_t freq = slot_ages[set_base + lane_id];
            slot_ages[set_base + lane_id] = (freq < 255) ? freq + 1 : 255;
        }
    }
    // POLICY_RANDOM: no-op
}

// =============================================================================
// Victim selection on eviction.
//
// LRU/LFU: butterfly min-reduction — find slot with smallest value
// RANDOM:  pick a random evictable slot
// =============================================================================
__device__ inline int policy_select_victim(
    int32_t policy,
    uint8_t* smem_state,
    uint32_t local_slot_base,
    uint32_t evictable_mask,
    uint32_t lane_id,
    // Extra seed for random policy (e.g., clock64() + blockIdx.x)
    uint64_t rand_seed = 0
) {
    if (policy == POLICY_LRU || policy == POLICY_LFU) {
        // Butterfly min-reduction on state values
        uint8_t lane_val = smem_state[local_slot_base + lane_id];
        bool lane_evictable = (evictable_mask >> lane_id) & 1;
        uint32_t my_val = lane_evictable ? (uint32_t)lane_val : UINT32_MAX;

        for (int offset = 16; offset > 0; offset /= 2) {
            uint32_t other_val = __shfl_xor_sync(POLICY_FULL_MASK, my_val, offset);
            if (other_val < my_val) {
                my_val = other_val;
            }
        }
        uint32_t min_val = __shfl_sync(POLICY_FULL_MASK, my_val, 0);

        bool has_min = lane_evictable && ((uint32_t)lane_val == min_val);
        unsigned min_mask = __ballot_sync(POLICY_FULL_MASK, has_min);

        if (min_mask == 0) return -1;
        return __ffs((int)min_mask) - 1;
    } else {
        // POLICY_RANDOM: pick a random evictable slot
        uint32_t n_evictable = __popc(evictable_mask);
        if (n_evictable == 0) return -1;

        uint32_t rand_idx = (uint32_t)(rand_seed % (uint64_t)n_evictable);

        // Walk evictable_mask to find the rand_idx-th set bit
        // Only lane 0 computes, then broadcasts
        int victim = -1;
        if (lane_id == 0) {
            uint32_t mask = evictable_mask;
            for (uint32_t i = 0; i <= rand_idx; i++) {
                victim = __ffs((int)mask) - 1;
                mask &= ~(1u << victim);
            }
        }
        victim = __shfl_sync(POLICY_FULL_MASK, victim, 0);
        return victim;
    }
}

// Victim selection for the global fallback kernel (operates on global memory).
__device__ inline int policy_select_victim_global(
    int32_t policy,
    uint8_t* slot_ages,
    uint32_t set_base,
    uint32_t evictable_mask,
    uint32_t lane_id,
    uint64_t rand_seed = 0
) {
    if (policy == POLICY_LRU || policy == POLICY_LFU) {
        uint8_t lane_val = slot_ages[set_base + lane_id];
        bool lane_evictable = (evictable_mask >> lane_id) & 1;
        uint32_t my_val = lane_evictable ? (uint32_t)lane_val : UINT32_MAX;

        for (int offset = 16; offset > 0; offset /= 2) {
            uint32_t other_val = __shfl_xor_sync(POLICY_FULL_MASK, my_val, offset);
            if (other_val < my_val) my_val = other_val;
        }
        uint32_t min_val = __shfl_sync(POLICY_FULL_MASK, my_val, 0);

        bool has_min = lane_evictable && ((uint32_t)lane_val == min_val);
        unsigned min_mask = __ballot_sync(POLICY_FULL_MASK, has_min);

        if (min_mask == 0) return -1;
        return __ffs((int)min_mask) - 1;
    } else {
        uint32_t n_evictable = __popc(evictable_mask);
        if (n_evictable == 0) return -1;

        uint32_t rand_idx = (uint32_t)(rand_seed % (uint64_t)n_evictable);
        int victim = -1;
        if (lane_id == 0) {
            uint32_t mask = evictable_mask;
            for (uint32_t i = 0; i <= rand_idx; i++) {
                victim = __ffs((int)mask) - 1;
                mask &= ~(1u << victim);
            }
        }
        victim = __shfl_sync(POLICY_FULL_MASK, victim, 0);
        return victim;
    }
}

// =============================================================================
// Initial state value for newly inserted slots.
//
// LRU:    32 (max age = most recent)
// LFU:    1  (first access)
// RANDOM: 0  (unused)
// =============================================================================
__device__ inline uint8_t policy_init_value(int32_t policy) {
    if (policy == POLICY_LRU) return POLICY_WAYS;   // 32
    if (policy == POLICY_LFU) return 1;
    return 0;  // POLICY_RANDOM
}
