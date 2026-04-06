"""
StagingCache: policy-agnostic GPU staging cache for CPU-offloaded KV pages.

Abstracts the replacement policy (LRU, LFU, Random) behind a unified interface.
All policies use identical state tensor shapes; only the CUDA kernel's internal
state-update and victim-selection hooks differ based on the policy parameter.
"""

import torch
from typing import Optional, Tuple
import vortex_torch.cache as cache_ops


class StagingCache:
    """GPU staging cache with pluggable replacement policy.

    Parameters
    ----------
    policy : str
        Replacement policy: ``"lru"``, ``"lfu"``, or ``"random"``.
    staging_capacity : int
        Total number of GPU staging slots (must be a multiple of 1024).
    max_num_pages : int
        Maximum number of CPU pages (size of the forward map).
    device : torch.device
        GPU device for all tensors.
    """

    POLICY_MAP = {"lru": 0, "lfu": 1, "random": 2}

    def __init__(
        self,
        policy: str,
        staging_capacity: int,
        max_num_pages: int,
        device: torch.device,
    ):
        if policy not in self.POLICY_MAP:
            raise ValueError(
                f"Unknown cache policy '{policy}'. "
                f"Choose from: {list(self.POLICY_MAP.keys())}"
            )

        self.policy = policy
        self.policy_id = self.POLICY_MAP[policy]
        self.staging_capacity = staging_capacity
        self.max_num_pages = max_num_pages
        self.device = device
        self.num_sets = staging_capacity // 32

        # --- Per-layer state tensors (persistent across decode steps) ---
        self.cpu_to_gpu_slot_map = torch.full(
            (max_num_pages,), -1, dtype=torch.int32, device=device
        )
        self.gpu_to_cpu_page_map = torch.full(
            (staging_capacity,), -1, dtype=torch.int32, device=device
        )
        self.slot_state = torch.zeros(
            staging_capacity, dtype=torch.uint8, device=device
        )
        self.set_used_masks = torch.zeros(
            self.num_sets, dtype=torch.int32, device=device
        )

    def allocate(
        self,
        sparse_kv_indices: torch.Tensor,
        sparse_kv_indptr: torch.Tensor,
        dst_gpu_slots: torch.Tensor,
        owners_bitmap: torch.Tensor,
        evicted_cpu_pages: torch.Tensor,
        overflow_flag: torch.Tensor,
        batch_size: int,
        num_kv_heads: int,
        max_hash_attempts: int = 30,
    ) -> None:
        """Run the allocation kernel with the configured policy.

        Parameters
        ----------
        sparse_kv_indices, sparse_kv_indptr : torch.Tensor
            Page requests for the current batch (input).
        dst_gpu_slots, owners_bitmap, evicted_cpu_pages, overflow_flag : torch.Tensor
            Output buffers written in-place by the CUDA kernel.
        batch_size, num_kv_heads : int
            Current batch dimensions.
        max_hash_attempts : int
            Maximum multi-probe hash attempts per miss.
        """
        cache_ops.allocate_pages_block_global(
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_indptr=sparse_kv_indptr,
            cpu_to_gpu_slot_map=self.cpu_to_gpu_slot_map,
            gpu_to_cpu_page_map=self.gpu_to_cpu_page_map,
            slot_state=self.slot_state,
            set_used_mask=self.set_used_masks,
            dst_gpu_slots=dst_gpu_slots,
            owners_bitmap=owners_bitmap,
            evicted_cpu_pages=evicted_cpu_pages,
            overflow_flag=overflow_flag,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            max_num_pages=self.staging_capacity,
            max_hash_attempts=max_hash_attempts,
            cache_policy=self.policy_id,
        )
