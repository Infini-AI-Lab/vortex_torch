from __future__ import annotations
from typing import Any, Final, Union
import torch
from ..abs import ContextBase
from ..utils import UNSET, Mode


class Context(ContextBase):
    """
    Mutable, single-instance context; populate later via .create(...).
    """
    
    __slots__ =  ContextBase.__slots__ + (
        # indices / indptr
        "dense_kv_indices", "sparse_kv_indices", "dense_kv_indptr", "sparse_kv_indptr", "kv_last_page_len", "batch_size",
        # winfo
        "winfo_q_indices", "winfo_kv_offsets", "winfo_kv_lens", "winfo_num_workloads", "winfo_chunk_size", "max_num_workloads",
        # chunk limits
        "max_chunk_size", "min_chunk_size",
        # head / shape
        "group_size", "num_kv_heads", "num_qo_heads", "head_dim",
        # hardware / paging
        "num_sms", "page_size", "max_num_pages", "max_num_pages_per_request",
        # misc
        "indexer_dtype", "topk_val", "page_reserved_bos", "page_reserved_eos", "topk_type",
        "topk_mapping_mode", "topk_mapping_power",
        "topk_histogram_enabled",
        
        # auxilary memory in graph
        "_aux_total_bytes",
        
        # auxilary flops in graph
        "_aux_total_flops"
    )
    
    # --- index tensors ---
    dense_kv_indices: torch.Tensor  #: Dense KV index tensor for mapping keys/values.
    sparse_kv_indices: torch.Tensor  #: Sparse KV index tensor for irregular KV layout.
    dense_kv_indptr: torch.Tensor    #: CSR-style indptr for dense KV segments.
    sparse_kv_indptr: torch.Tensor   #: CSR-style indptr for sparse KV segments.
    kv_last_page_len: int            #: Length of the last KV page.
    batch_size: int                  #: Active batch size.

    # --- workload info (winfo) ---
    winfo_q_indices: torch.Tensor    #: Query indices used in workload scheduling.
    winfo_kv_offsets: torch.Tensor   #: KV offsets per workload.
    winfo_kv_lens: torch.Tensor      #: KV lengths per workload.
    winfo_num_workloads: int         #: Number of workloads in the current batch.
    winfo_chunk_size: int            #: Chunk size for workload partitioning.
    max_num_workloads: int           #: Maximum number of workloads allowed.

    # --- chunk limits ---
    max_chunk_size: int              #: Maximum allowed chunk size.
    min_chunk_size: int              #: Minimum allowed chunk size.

    # --- head / shape configuration ---
    group_size: int                  #: Group size for grouped attention.
    num_kv_heads: int                #: Number of KV heads.
    num_qo_heads: int                #: Number of query/output heads.
    head_dim: int                    #: Dimension per attention head.

    # --- hardware / paging ---
    num_sms: int                     #: Number of streaming multiprocessors (SMs).
    page_size: int                   #: Page size used for memory paging.
    max_num_pages: int               #: Total available pages.
    max_num_pages_per_request: int   #: Page limit per individual request.

    # --- miscellaneous ---
    indexer_dtype: torch.dtype       #: Dtype used by indexer operations.
    topk_val: int                    #: Top-K value used in pruning or selection.
    page_reserved_bos: int           #: Reserved page count for BOS (begin-of-sequence).
    page_reserved_eos: int           #: Reserved page count for EOS (end-of-sequence).
    topk_type: str                   #: TopK kernel type: "naive", "sglang" (unmapped) or "sglang_fused" (remap+topk).
    topk_mapping_mode: int           #: TopK mapping mode for sglang_fused (0=none, 3=power, 4=log, 6=asinh, 7=log1p, 9=erf, 10=tanh, 13=exp_stretch).
    topk_mapping_power: float        #: Hyperparameter (p / alpha / beta) for the active mapping mode.
    topk_histogram_enabled: bool     #: Enable histogram profiling during inference (default False).

    # --- auxiliary ---
    _aux_total_bytes: int            #: Accumulated auxiliary memory in bytes.
    _aux_total_flops: int            #: Accumulated auxiliary flops.
    
    def __init__(self) -> None:
        # Start as an empty shell (no big allocations).
        for name in self.__slots__:
            if name == "_created":
                object.__setattr__(self, name, False)
            elif name == "name":
                object.__setattr__(self, name, "Indexer")
            elif name == "_aux_total_bytes":
                object.__setattr__(self, name, 0)  # start from 0 bytes
            elif name == "_aux_total_flops":
                object.__setattr__(self, name, 0)  # start from 0 flops
            elif name == "batch_size":
                object.__setattr__(self, name, 0)
            elif name == "mode":
                object.__setattr__(self, name, Mode.profile) 
            else:
                object.__setattr__(self, name, UNSET)

    
    def set_batch_size(self, n: int) -> None:
        
        self.batch_size = n
        
    
    def create(self, parent: Any, model_runner: Any, *, overwrite: bool = False) -> "Context":
        """
        Populate this instance once (no locking). Set overwrite=True to allow re-init.
        NOTE: Without locking, concurrent callers may race; call from a single thread.
        """
        if self._created and not overwrite:
            raise RuntimeError("Context.create() already called; pass overwrite=True to reinitialize.")

        sa = model_runner.server_args
        max_pages_per_req = (
            (model_runner.model_config.context_len + sa.page_size - 1) // sa.page_size
            if sa.vortex_max_seq_lens < 0
            else (sa.vortex_max_seq_lens + sa.page_size - 1) // sa.page_size
        )
        max_bs = int(model_runner.req_to_token_pool.size)

        # Backend-known fields
        self.dense_kv_indices = parent.kv_indices_decode[0]
        self.sparse_kv_indices = parent.kv_indices_decode[1]
        self.dense_kv_indptr = parent.kv_indptr_decode[0]
        self.sparse_kv_indptr = parent.kv_indptr_decode[1]
        self.kv_last_page_len = parent.kv_last_page_len_decode

        self.max_chunk_size = sa.vortex_lb_max_chunk_size
        self.min_chunk_size = sa.vortex_lb_min_chunk_size

        self.group_size = parent.group_size
        self.num_kv_heads = parent.num_kv_heads
        self.num_qo_heads = parent.num_qo_heads
        self.head_dim = parent.head_dim

        self.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self.page_size = sa.page_size

        # Capacity model (adjust as needed)
        self.max_num_pages = max_pages_per_req * max_bs * self.num_kv_heads
        self.max_num_pages_per_request = max_pages_per_req

        self.topk_val = sa.vortex_topk_val
        dtype_str = getattr(sa, "vortex_indexer_dtype", "float32")
        if isinstance(dtype_str, str):
            self.indexer_dtype = getattr(torch, dtype_str, torch.float32)
        else:
            self.indexer_dtype = dtype_str
        
        self.page_reserved_bos = sa.vortex_page_reserved_bos
        self.page_reserved_eos = sa.vortex_page_reserved_eos
        self.topk_type = getattr(sa, "vortex_topk_type", "naive")
        self.topk_mapping_mode = getattr(sa, "vortex_topk_mapping_mode", 0)
        self.topk_mapping_power = getattr(sa, "vortex_topk_mapping_power", 0.5)
        self.topk_histogram_enabled = getattr(sa, "vortex_topk_histogram", False)

        device = getattr(model_runner, "device", "cpu")

        self.max_num_workloads = (
            (self.max_num_pages // max(1, sa.vortex_lb_min_chunk_size)) + max_bs * self.num_kv_heads
        )
        self.winfo_q_indices = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_kv_offsets = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_kv_lens = torch.zeros((self.max_num_workloads,), dtype=torch.int32, device=device)
        self.winfo_num_workloads = torch.zeros((1,), dtype=torch.int32, device=device)
        self.winfo_chunk_size = torch.zeros((1,), dtype=torch.int32, device=device)

        self._created = True
        return self



# Module-level singleton (part of the public package API)
ctx: Final[Context] = Context()

def get_ctx() -> Context:
    return ctx

__all__ = ["Context", "ctx", "get_ctx"]
