#include "register.h"
#include <cub/cub.cuh>

struct PairSum {
  __device__ __forceinline__ int2 operator()(const int2& a, const int2& b) const {
    return make_int2(a.x + b.x, a.y + b.y);
  }
};



template <int ITEM_PER_THREAD>
__global__ void Sgl_Decode_Plan_IndptrWorkload_Kernel(
const int*  __restrict__ cached_seq_lens,
int*        __restrict__ dense_kv_indptr,
int*        __restrict__ sparse_kv_indptr,
int*        __restrict__ winfo_q_indices,
int*        __restrict__ winfo_kv_offsets,
int*        __restrict__ winfo_kv_lens,
int*        __restrict__ winfo_num_workload,
int*        __restrict__ winfo_chunk_size,
const int   max_chunk_size,
const int   min_chunk_size,
const int   batch_size,
const int   num_kv_heads,
const int   page_size,
const int   topk_val,
const int   page_reserved_bos,
const int   page_reserved_eos
){

    const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
    const int tx = threadIdx.x;
    const int cached_seq_len = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    const int cached_page_len = (cached_seq_len + page_size - 1) / page_size;
    using BlockScanInt2 = cub::BlockScan<int2, 1024>;
    using BlockScanInt = cub::BlockScan<int, 1024>;

    __shared__ union {
        typename BlockScanInt2::TempStorage scan_int2;
        typename BlockScanInt::TempStorage scan_int;
    } temp;

   
    const int page_cnt = (tx < batch_size) ? cached_page_len : 0;
    const int sparse_cnt = (tx < batch_size) ? min(kv_budget, cached_page_len) : 0;
    
    int2 in  = make_int2(page_cnt, sparse_cnt);
    int2 out;
    BlockScanInt2(temp.scan_int2).InclusiveScan(in, out, PairSum{});

    const int dense_cumsum  = out.x;
    const int sparse_cumsum = out.y;

    if (tx < batch_size){

        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i){
            dense_kv_indptr[num_kv_heads * (tx + 1) - i] = dense_cumsum * num_kv_heads
                - i * cached_page_len;
            
            sparse_kv_indptr[num_kv_heads * (tx + 1) - i] = sparse_cumsum * num_kv_heads
                - i * min(kv_budget, cached_page_len);
        }

    }

    if(tx == 0){
        dense_kv_indptr[0] = 0;
        sparse_kv_indptr[0] = 0;
    }

    __syncthreads();

    const int eff_batch_size = batch_size * num_kv_heads;
    uint16_t page_count[ITEM_PER_THREAD];
    int chunked_page_count_prefix_sum[ITEM_PER_THREAD + 1];
    int tx_offset = tx * ITEM_PER_THREAD;

    chunked_page_count_prefix_sum[0] = 0;

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        int16_t w = ((tx_offset + i) < eff_batch_size) ?
            (dense_kv_indptr[tx_offset+i+1] - dense_kv_indptr[tx_offset+i]
            - page_reserved_bos - page_reserved_eos): 0;

        // See note in Sgl_Decode_Plan_Workload_Kernel: we used to skip slots
        // where w ≤ topk_val, but downstream (GeMV / topK / histogram) has no
        // matching skip, so it read uninitialised scores and silently
        // produced all-zero results. Emit workloads for every slot with w > 0.
        page_count[i] = (w > 0) ? w : 0;
        chunked_page_count_prefix_sum[i + 1] =  int((page_count[i] + max_chunk_size - 1) / max_chunk_size);
    }

    BlockScanInt(temp.scan_int).InclusiveSum(chunked_page_count_prefix_sum, chunked_page_count_prefix_sum);

    if (tx == 1023){
        *winfo_num_workload = chunked_page_count_prefix_sum[ITEM_PER_THREAD];
        *winfo_chunk_size = max_chunk_size;
    }

    
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        if((tx_offset + i) < eff_batch_size){
        const int start = chunked_page_count_prefix_sum[i];
        const int end = chunked_page_count_prefix_sum[i+1];
        int last_len = int(page_count[i] % max_chunk_size);
        if (last_len == 0) last_len = max_chunk_size;
        for (int j = start; j < end; ++j){
                winfo_q_indices[j] = tx_offset + i;
                winfo_kv_lens[j] = (j!=end-1)?(max_chunk_size):(last_len);
                winfo_kv_offsets[j] = dense_kv_indptr[tx_offset + i] + (j - start) * max_chunk_size + page_reserved_bos;
        }

        }

    }

}




__global__ void Sgl_Decode_Plan_Indices_Kernel(
const int*   __restrict__ req_to_token,
const long*  __restrict__ req_indices,
const int*   __restrict__ cache_seq_lens,
const int*   __restrict__ dense_kv_indptr,
const int*   __restrict__ sparse_kv_indptr,
int*         __restrict__ dense_kv_indices,
int*         __restrict__ sparse_kv_indices,
int*         __restrict__ kv_last_page_len,
const int    req_to_token_stride,
const int    page_size,
const int    num_kv_heads,
const int    topk_val,
const int    page_reserved_bos,
const int    page_reserved_eos
){

    const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
    const int block_size = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    const int kv_len = cache_seq_lens[bx];
    const int page_len = (kv_len + page_size - 1) / page_size;
    const int last_len = kv_len % page_size;
    if (tx == 0) {
            kv_last_page_len[bx * num_kv_heads + by] = (last_len == 0) ? page_size : last_len;
    }
    int* dense_output = dense_kv_indices + dense_kv_indptr[bx * num_kv_heads + by];
    int* sparse_output = sparse_kv_indices + sparse_kv_indptr[bx * num_kv_heads + by];

    int pos = tx;
    while(pos < page_len){
            int data = token_indices[pos * page_size];
            dense_output[pos] = (data / page_size) * (num_kv_heads) + by;
            pos += block_size;
    }

    if(page_len <= kv_budget){
        int pos = tx;
        while(pos < page_len){
            sparse_output[pos] = dense_output[pos];
            pos += block_size;
        }
    }else{
        int pos = tx;
        while(pos < page_reserved_bos){
            sparse_output[pos] = dense_output[pos];
            pos += block_size;
        }

        pos = tx;
        while(pos < page_reserved_eos){
            int data = token_indices[(page_len-pos-1) * page_size];
            sparse_output[kv_budget-pos-1] = (data / page_size) * (num_kv_heads) + by;
            pos += block_size;
        }

    }

}



template <int ITEM_PER_THREAD>
__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Workload_Kernel(
const int*  __restrict__ dense_kv_indptr,
int*  __restrict__ winfo_q_indices,
int*  __restrict__ winfo_kv_offsets,
int*  __restrict__ winfo_kv_lens,
int*  __restrict__ winfo_num_workloads,
int*  __restrict__ winfo_chunk_size,
const int max_chunk_size,
const int min_chunk_size,
const int eff_batch_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){  

    const int tx = threadIdx.x;
    using BlockScanInt = cub::BlockScan<int, 1024>;
    __shared__ union {
        typename BlockScanInt::TempStorage scan_int;
    } temp;


    uint16_t page_count[ITEM_PER_THREAD];
    int chunked_page_count_prefix_sum[ITEM_PER_THREAD + 1];
    int tx_offset = tx * ITEM_PER_THREAD;

    chunked_page_count_prefix_sum[0] = 0;

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        int16_t w = ((tx_offset + i) < eff_batch_size) ?
            (dense_kv_indptr[tx_offset+i+1] - dense_kv_indptr[tx_offset+i]
            - page_reserved_bos - page_reserved_eos): 0;

        // Previously: (w > topk_val) ? w : 0, which skipped scoring on slots
        // where the dense page count is already ≤ topk_val. Downstream (GeMV,
        // topK, histogram profiling) does NOT have a matching skip, so it
        // would read uninitialised scores and silently return garbage (all
        // zero). Emit workloads for every slot with w > 0 so scoring always
        // runs; when w ≤ topk_val the topK degenerates to "select all w".
        page_count[i] = (w > 0) ? w : 0;
        chunked_page_count_prefix_sum[i + 1] =  int((page_count[i] + max_chunk_size - 1) / max_chunk_size);
    }

    BlockScanInt(temp.scan_int).InclusiveSum(chunked_page_count_prefix_sum, chunked_page_count_prefix_sum);

    if (tx == 1023){
        *winfo_num_workloads = chunked_page_count_prefix_sum[ITEM_PER_THREAD];
        *winfo_chunk_size = max_chunk_size;
    }

    
    for (int i = 0; i < ITEM_PER_THREAD; ++i){

        if((tx_offset + i) < eff_batch_size){
        const int start = chunked_page_count_prefix_sum[i];
        const int end = chunked_page_count_prefix_sum[i+1];
        int last_len = int(page_count[i] % max_chunk_size);
        if (last_len == 0) last_len = max_chunk_size;
        for (int j = start; j < end; ++j){
                winfo_q_indices[j] = tx_offset + i;
                winfo_kv_lens[j] = (j!=end-1)?(max_chunk_size):(last_len);
                winfo_kv_offsets[j] = dense_kv_indptr[tx_offset + i] + (j - start) * max_chunk_size + page_reserved_bos;
        }

        }

    }

};


__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Indptr_Kernel(
const int*  __restrict__ cached_seq_lens,
int*  __restrict__ dense_kv_indptr,
int*  __restrict__ sparse_kv_indptr,
const int batch_size,
const int num_kv_heads,
const int page_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){

    const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
    const int tx = threadIdx.x;
    const int cached_seq_len = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    const int cached_page_len = (cached_seq_len + page_size - 1) / page_size;
    using BlockScanInt2 = cub::BlockScan<int2, 1024>;

    __shared__ union {
        typename BlockScanInt2::TempStorage scan_int2;
    } temp;

   
    const int page_cnt = (tx < batch_size) ? cached_page_len : 0;
    const int sparse_cnt = (tx < batch_size) ? min(kv_budget, cached_page_len) : 0;
    
    int2 in  = make_int2(page_cnt, sparse_cnt);
    int2 out;
    BlockScanInt2(temp.scan_int2).InclusiveScan(in, out, PairSum{});

    const int dense_cumsum  = out.x;
    const int sparse_cumsum = out.y;

    if (tx < batch_size){

        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i){
            dense_kv_indptr[num_kv_heads * (tx + 1) - i] = dense_cumsum * num_kv_heads
                - i * cached_page_len;
            
            sparse_kv_indptr[num_kv_heads * (tx + 1) - i] = sparse_cumsum * num_kv_heads
                - i * min(kv_budget, cached_page_len);
        }

    }

    if(tx == 0){
        dense_kv_indptr[0] = 0;
        sparse_kv_indptr[0] = 0;
    }

};



void sglang_plan_decode(
const at::Tensor& cached_seq_lens,
at::Tensor&       dense_kv_indptr,
at::Tensor&       dense_kv_indices,
at::Tensor&       sparse_kv_indptr,
at::Tensor&       sparse_kv_indices,
at::Tensor&       kv_last_page_len,
const at::Tensor& req_to_token,
const at::Tensor& req_indices,
at::Tensor&       winfo_q_indices,
at::Tensor&       winfo_kv_offsets,
at::Tensor&       winfo_kv_lens,
at::Tensor&       winfo_num_workloads,
at::Tensor&       winfo_chunk_size,
const int64_t     page_size,
const int64_t     num_kv_heads,
const int64_t     topk_val,
const int64_t     page_reserved_bos,
const int64_t     page_reserved_eos,
const int64_t     max_chunk_size,
const int64_t     min_chunk_size
){

#ifdef VORTEX_DEBUG
    TORCH_CHECK(req_to_token.dtype() == torch::kInt32);
    TORCH_CHECK(req_indices.dtype() == torch::kInt64);
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(sparse_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(sparse_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(kv_last_page_len.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_q_indices.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_kv_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_kv_lens.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_num_workloads.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_chunk_size.dtype() == torch::kInt32);
    TORCH_CHECK(page_size >= 1);
    TORCH_CHECK(num_kv_heads >= 1);
    TORCH_CHECK(topk_val >= 1);
    TORCH_CHECK(page_reserved_bos >= 0);
    TORCH_CHECK(page_reserved_eos >= 1);
    TORCH_CHECK(min_chunk_size >= 1);
    TORCH_CHECK(max_chunk_size > min_chunk_size);
#endif

    const int batch_size = cached_seq_lens.size(0);
    const int eff_batch_size = batch_size * num_kv_heads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Compute Indptr and Workload

    Sgl_Decode_Plan_Indptr_Kernel<<<1, 1024, 0, stream>>>(
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        batch_size,
        num_kv_heads,
        page_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );

    if (eff_batch_size <= 1024){
    Sgl_Decode_Plan_Workload_Kernel<1><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    } else if (eff_batch_size <= 2048){
    Sgl_Decode_Plan_Workload_Kernel<2><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    } else if (eff_batch_size <= 4096){
    Sgl_Decode_Plan_Workload_Kernel<4><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    }  else if (eff_batch_size <= 8192){
    Sgl_Decode_Plan_Workload_Kernel<8><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    }  

    // Fill in indices for dense and sparse (eos and bos)

    const int req_to_token_stride = req_to_token.size(1);
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(512);
    Sgl_Decode_Plan_Indices_Kernel<<<nblks, nthrs, 0, stream>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        kv_last_page_len.data_ptr<int>(),
        req_to_token_stride,
        page_size,
        num_kv_heads,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
};

__launch_bounds__(1024, 1)
__global__ void Sgl_Prefill_Plan_Indptr_Kernel(
const int*  __restrict__ cached_seq_lens,
const int*  __restrict__ input_seq_lens,
int*        __restrict__ kv_indptr,
int*        __restrict__ qo_indptr_ragged,
int*        __restrict__ qo_indptr_paged,
const int   batch_size,
const int   num_kv_heads,
const int   page_size
)
{   

    const int tx = threadIdx.x;
    using BlockScan = cub::BlockScan<int, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int input_seq_cumsum = (tx < batch_size) ? input_seq_lens[tx] : 0;
    int num_cached_pages = (tx < batch_size) ?  ((cached_seq_lens[tx] + page_size - 1) / page_size) : 0;
    int cached_seq_cumsum = (tx < batch_size) ? num_cached_pages : 0;

    BlockScan(temp_storage).InclusiveSum(input_seq_cumsum, input_seq_cumsum);
    __syncthreads();
    BlockScan(temp_storage).InclusiveSum(cached_seq_cumsum, cached_seq_cumsum);
    __syncthreads();

    
    if (tx < batch_size){
        qo_indptr_ragged[tx + 1] = input_seq_cumsum;

        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i){
            qo_indptr_paged[num_kv_heads * (tx + 1) - i] = input_seq_cumsum * num_kv_heads
                - i * input_seq_lens[tx];
            
            kv_indptr[num_kv_heads * (tx + 1) - i] = cached_seq_cumsum * num_kv_heads
                - i * num_cached_pages;
        }

    }

    if(tx == 0){
        qo_indptr_ragged[0] = 0;
        qo_indptr_paged[0] = 0;
        kv_indptr[0] = 0;
    }

}

__global__ void Sgl_Prefill_Plan_IndicesBatchTable_Kernel(
const int*   __restrict__ req_to_token,
const long*  __restrict__ req_indices,
const int*   __restrict__ cache_seq_lens,
const int*   __restrict__ input_seq_lens,
const int*   __restrict__ kv_indptr,
const int*   __restrict__ qo_indptr_ragged,
const int    page_size,
const int    num_kv_heads,
const int    req_to_token_stride,
int*         __restrict__ kv_last_page_len,
int*         __restrict__ kv_indices,
uint16_t*    __restrict__ batch_table
){
    const int block_size = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    const int num_cached_pages = (cache_seq_lens[bx] + page_size - 1) / page_size;
    int* output = kv_indices + kv_indptr[bx * num_kv_heads + by];
    const int last_len = cache_seq_lens[bx] % page_size;
    kv_last_page_len[bx * num_kv_heads + by] = (last_len == 0)? page_size:last_len;

    int pos = tx;
    while(pos < num_cached_pages){
        int data = token_indices[pos * page_size];
        output[pos] = (data / page_size) * (num_kv_heads) + by;
        pos += block_size;
    }

    const int qo_len = input_seq_lens[bx];
    uint16_t* batch_table_output = batch_table + qo_indptr_ragged[bx];
    pos = tx;
    if (by == 0){
        while(pos < qo_len){
        batch_table_output[pos] = static_cast<uint16_t>(bx);
        pos += block_size;
    }
    }

}


void sglang_plan_prefill(
const at::Tensor&  cached_seq_lens,
at::Tensor&        dense_kv_indptr,
at::Tensor&        dense_kv_indices,
const at::Tensor&  input_seq_lens,
at::Tensor&        qo_indptr_ragged,
at::Tensor&        qo_indptr_paged,
at::Tensor&        kv_last_page_len,
const at::Tensor&  req_to_token,
const at::Tensor&  req_indices,
at::Tensor&        batch_table,
const int64_t      page_size,
const int64_t      num_kv_heads
){


    const int batch_size = cached_seq_lens.size(0);
    const int req_to_token_stride = req_to_token.size(1);

#ifdef VORTEX_DEBUG
    TORCH_CHECK(batch_size == input_seq_lens.size(0));
    TORCH_CHECK((batch_size + 1) == qo_indptr_ragged.size(0));
    TORCH_CHECK((batch_size * num_kv_heads + 1) == qo_indptr_paged.size(0));
    TORCH_CHECK((batch_size * num_kv_heads + 1) == dense_kv_indptr.size(0));
    TORCH_CHECK(batch_size <= 1024);
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(input_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(qo_indptr_ragged.dtype() == torch::kInt32);
    TORCH_CHECK(qo_indptr_paged.dtype() == torch::kInt32);
    TORCH_CHECK(kv_last_page_len.dtype() == torch::kInt32);
    TORCH_CHECK(req_to_token.dtype() == torch::kInt32);
    TORCH_CHECK(req_indices.dtype() == torch::kInt64);
#endif

    Sgl_Prefill_Plan_Indptr_Kernel<<<1, 1024>>>(
        cached_seq_lens.data_ptr<int>(),
        input_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        qo_indptr_ragged.data_ptr<int>(),
        qo_indptr_paged.data_ptr<int>(),
        batch_size,
        num_kv_heads,
        page_size
    );


    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(128);
    Sgl_Prefill_Plan_IndicesBatchTable_Kernel<<<nblks, nthrs>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        input_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        qo_indptr_ragged.data_ptr<int>(),
        page_size,
        num_kv_heads,
        req_to_token_stride,
        kv_last_page_len.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>()
    );



}

__global__ void Chunkwise_NH2HN_Transpose_Kernel(
const  __nv_bfloat16*  __restrict__ x,
const int*             __restrict__ indptr,
const uint16_t*        __restrict__ batch_table,
const int              num_kv_heads,
const int              group_size,
const int              head_dim,
__nv_bfloat16*         __restrict__ output
){

const int bx = blockIdx.x;
const int by = blockIdx.y;
const int bz = blockIdx.z;
const int tx = threadIdx.x;
const int batch_idx =  static_cast<int>(batch_table[bx]);

const int batch_offset = indptr[batch_idx];
const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
const int token_offset = bx - indptr[batch_idx];

const __nv_bfloat16* src = x + bx * num_kv_heads * group_size * head_dim +
            by * group_size * head_dim + bz * head_dim;

__nv_bfloat16* dst = output + batch_offset * num_kv_heads * group_size * head_dim +
            + by * batch_q_len * group_size * head_dim + token_offset * group_size * head_dim +
            bz * head_dim;


constexpr int B16_PER_FLOAT2 = 4;
const int nvec = head_dim / B16_PER_FLOAT2;

const float2* __restrict__ src2 = reinterpret_cast<const float2*>(src);
float2* __restrict__ dst2       = reinterpret_cast<float2*>(dst);

// Strided loop so it works for any blockDim.x
for (int i = tx; i < nvec; i += blockDim.x) {
    dst2[i] = src2[i];
}

}


at::Tensor Chunkwise_NH2HN_Transpose(
const at::Tensor&   x,
const at::Tensor&   indptr,
const at::Tensor&   batch_table,
const int64_t       num_qo_heads,
const int64_t       num_kv_heads,
const int64_t       head_dim
){

    const int x_len = x.size(0);
    const int group_size = num_qo_heads / num_kv_heads;
    at::Tensor output = torch::empty(
    {x_len * num_kv_heads, group_size, head_dim}, x.options()
        );

    
    dim3 nblks(x_len, num_kv_heads, group_size);
    dim3 nthrs(head_dim / 4);
    Chunkwise_NH2HN_Transpose_Kernel<<<nblks, nthrs>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        indptr.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>(),
        num_kv_heads,
        group_size,
        head_dim,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>())
    );

    return output;

}


__global__ void Chunkwise_HN2NH_Transpose_Kernel(
const __nv_bfloat16* __restrict__ x,
const float*         __restrict__ y,
const int*           __restrict__ indptr,
const uint16_t*      __restrict__ batch_table,
const int            num_kv_heads,
const int            group_size,
const int            head_dim,
__nv_bfloat16*       __restrict__ x_output,
float*               __restrict__ y_output
){

const int bx = blockIdx.x;
const int by = blockIdx.y;
const int bz = blockIdx.z;
const int tx = threadIdx.x;

const int batch_idx =  static_cast<int>(batch_table[bx]);

const int batch_offset = indptr[batch_idx];
const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
const int token_offset = bx - indptr[batch_idx];

const __nv_bfloat16* src_x = x + batch_offset * num_kv_heads * group_size * head_dim +
            + by * batch_q_len * group_size * head_dim + token_offset * group_size * head_dim +
            bz * head_dim;

__nv_bfloat16* dst_x = x_output + bx * num_kv_heads * group_size * head_dim +
            by * group_size * head_dim + bz * head_dim;


constexpr int B16_PER_FLOAT2 = 4;
const int nvec = head_dim / B16_PER_FLOAT2;

const float2* __restrict__ src2_x = reinterpret_cast<const float2*>(src_x);
float2* __restrict__ dst2_x       = reinterpret_cast<float2*>(dst_x);

for (int i = tx; i < nvec; i += blockDim.x) {
    dst2_x[i] = src2_x[i];
}


const float* src_y = y + batch_offset * num_kv_heads * group_size +
            + by * batch_q_len * group_size + token_offset * group_size + bz;

float* dst_y = y_output + bx * num_kv_heads * group_size +
            by * group_size + bz;


if (tx == 0) {
    *dst_y = *src_y;
}

}

std::tuple<at::Tensor, at::Tensor> Chunkwise_HN2NH_Transpose(
const at::Tensor&   x,
const at::Tensor&   y,
const at::Tensor&   indptr,
const at::Tensor&   batch_table,
const int64_t       num_qo_heads,
const int64_t       num_kv_heads,
const int64_t       head_dim
) {
    
    
    const int x_len = x.size(0) / num_kv_heads;
    const int group_size = x.size(1);


    at::Tensor x_output = torch::empty(
    {x_len, num_qo_heads, head_dim}, x.options()
        );
    
    at::Tensor y_output = torch::empty(
    {x_len, num_qo_heads}, y.options()
        );

    dim3 nblks(x_len, num_kv_heads, group_size);
    dim3 nthrs(head_dim / 4);
    Chunkwise_HN2NH_Transpose_Kernel<<<nblks, nthrs>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        y.data_ptr<float>(),
        indptr.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>(),
        num_kv_heads,
        group_size,
        head_dim,
        reinterpret_cast<__nv_bfloat16*>(x_output.data_ptr<at::BFloat16>()),
        y_output.data_ptr<float>()
    );

    
    return std::make_tuple(x_output, y_output);

}



__global__ void Sgl_Prefill_Plan_PageTable_BatchTable_FA3_Kernel(
const int*   __restrict__ req_to_token,
const long*  __restrict__ req_indices,
const int*   __restrict__ cache_seq_lens,
const int*   __restrict__ cu_seqlens_q,
const int    page_size,
const int    num_kv_heads,
const int    req_to_token_stride,
const int    page_table_stride,
int*         __restrict__ page_table,
uint16_t*    __restrict__ batch_table
){
    const int block_size = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    const int num_cached_pages = (cache_seq_lens[bx * num_kv_heads + by] + page_size - 1) / page_size;
    int* output = page_table + (bx * num_kv_heads + by) * page_table_stride;
   

    int pos = tx;
    while(pos < num_cached_pages){
        int data = token_indices[pos * page_size];
        output[pos] = (data / page_size) * (num_kv_heads) + by;
        pos += block_size;
    }

    const int qo_len = cu_seqlens_q[bx * num_kv_heads + by + 1] - cu_seqlens_q[bx * num_kv_heads + by];
    uint16_t* batch_table_output = batch_table + cu_seqlens_q[bx * num_kv_heads] / num_kv_heads;
    pos = tx;
    if (by == 0){
        while(pos < qo_len){
        batch_table_output[pos] = static_cast<uint16_t>(bx);
        pos += block_size;
    }
    }

}


void sglang_plan_prefill_fa3(
const at::Tensor&  cached_seq_lens,
const at::Tensor&  cu_seqlens_q,
const at::Tensor&  req_to_token,
const at::Tensor&  req_indices,
at::Tensor&        page_table,
at::Tensor&        batch_table,
const int64_t      page_size,
const int64_t      num_kv_heads
){

    const int batch_size = req_indices.size(0);
    const int req_to_token_stride = req_to_token.size(1);
    const int page_table_stride = page_table.size(1);

    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(128);
    
    Sgl_Prefill_Plan_PageTable_BatchTable_FA3_Kernel<<<nblks, nthrs>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        cu_seqlens_q.data_ptr<int>(),
        page_size,
        num_kv_heads,
        req_to_token_stride,
        page_table_stride,
        page_table.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>()
    );
    
}


__global__ void Chunkwise_HN2NH_Transpose_FA3_Kernel(
const __nv_bfloat16* __restrict__ x,
const int*           __restrict__ indptr,
const uint16_t*      __restrict__ batch_table,
const int            num_kv_heads,
const int            group_size,
const int            head_dim,
__nv_bfloat16*       __restrict__ x_output
){

const int bx = blockIdx.x;
const int by = blockIdx.y;
const int bz = blockIdx.z;
const int tx = threadIdx.x;

const int batch_idx =  static_cast<int>(batch_table[bx]);

const int batch_offset = indptr[batch_idx];
const int batch_q_len = indptr[batch_idx + 1] - indptr[batch_idx];
const int token_offset = bx - indptr[batch_idx];

const __nv_bfloat16* src_x = x + batch_offset * num_kv_heads * group_size * head_dim +
            + by * batch_q_len * group_size * head_dim + token_offset * group_size * head_dim +
            bz * head_dim;

__nv_bfloat16* dst_x = x_output + bx * num_kv_heads * group_size * head_dim +
            by * group_size * head_dim + bz * head_dim;


constexpr int B16_PER_FLOAT2 = 4;
const int nvec = head_dim / B16_PER_FLOAT2;

const float2* __restrict__ src2_x = reinterpret_cast<const float2*>(src_x);
float2* __restrict__ dst2_x       = reinterpret_cast<float2*>(dst_x);

for (int i = tx; i < nvec; i += blockDim.x) {
    dst2_x[i] = src2_x[i];
}
}


at::Tensor Chunkwise_HN2NH_Transpose_FA3(
const at::Tensor&   x,
const at::Tensor&   indptr,
const at::Tensor&   batch_table,
const int64_t       num_qo_heads,
const int64_t       num_kv_heads,
const int64_t       head_dim
) {
    
    
    const int x_len = x.size(0) / num_kv_heads;
    const int group_size = x.size(1);


    at::Tensor x_output = torch::empty(
    {x_len, num_qo_heads, head_dim}, x.options()
        );
    
    dim3 nblks(x_len, num_kv_heads, group_size);
    dim3 nthrs(head_dim / 4);
    Chunkwise_HN2NH_Transpose_FA3_Kernel<<<nblks, nthrs>>>(
        reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        indptr.data_ptr<int>(),
        batch_table.data_ptr<uint16_t>(),
        num_kv_heads,
        group_size,
        head_dim,
        reinterpret_cast<__nv_bfloat16*>(x_output.data_ptr<at::BFloat16>())
    );

    
    return x_output;

}


__launch_bounds__(1024, 1)
__global__ void Sgl_Decode_Plan_Indptr_FA3_Kernel(
const int*  __restrict__ cached_seq_lens,
int*  __restrict__ dense_kv_indptr,
int*  __restrict__ sparse_kv_indptr,
int*  __restrict__ dense_cache_seqlens,
int*  __restrict__ sparse_cache_seqlens,
const int batch_size,
const int num_kv_heads,
const int page_size,
const int topk_val,
const int page_reserved_bos,
const int page_reserved_eos
){

    const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
    const int tx = threadIdx.x;
    const int cached_seq_len = (tx < batch_size) ? cached_seq_lens[tx] : 0;
    const int cached_page_len = (cached_seq_len + page_size - 1) / page_size;
    const int r = cached_seq_len % page_size;
    const int kv_last_page_len = (r == 0) ? page_size : r;
    const int sparse_cached_seq_len = (kv_budget - 1) * page_size + kv_last_page_len;
    if(tx < batch_size) {
        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i){
            dense_cache_seqlens[tx * num_kv_heads + i] = cached_seq_len;
            sparse_cache_seqlens[tx * num_kv_heads + i] = min(sparse_cached_seq_len, cached_seq_len);
        }

    }

    using BlockScanInt2 = cub::BlockScan<int2, 1024>;

    __shared__ union {
        typename BlockScanInt2::TempStorage scan_int2;
    } temp;

   
    const int page_cnt = (tx < batch_size) ? cached_page_len : 0;
    const int sparse_cnt = (tx < batch_size) ? min(kv_budget, cached_page_len) : 0;
    
    int2 in  = make_int2(page_cnt, sparse_cnt);
    int2 out;
    BlockScanInt2(temp.scan_int2).InclusiveScan(in, out, PairSum{});

    const int dense_cumsum  = out.x;
    const int sparse_cumsum = out.y;

    if (tx < batch_size){

        #pragma unroll
        for (int i = 0; i < num_kv_heads; ++i){
            dense_kv_indptr[num_kv_heads * (tx + 1) - i] = dense_cumsum * num_kv_heads
                - i * cached_page_len;
            
            sparse_kv_indptr[num_kv_heads * (tx + 1) - i] = sparse_cumsum * num_kv_heads
                - i * min(kv_budget, cached_page_len);
        }

    }

    if(tx == 0){
        dense_kv_indptr[0] = 0;
        sparse_kv_indptr[0] = 0;
    }

};


__global__ void Sgl_Decode_Plan_Indices_FA3_Kernel(
const int*   __restrict__ req_to_token,
const long*  __restrict__ req_indices,
const int*   __restrict__ cache_seq_lens,
const int*   __restrict__ dense_kv_indptr,
const int*   __restrict__ sparse_kv_indptr,
int*         __restrict__ dense_kv_indices,
int*         __restrict__ sparse_kv_indices,
int*         __restrict__ dense_page_table,
int*         __restrict__ sparse_page_table,
const int    req_to_token_stride,
const int    page_table_stride,
const int    page_size,
const int    num_kv_heads,
const int    topk_val,
const int    page_reserved_bos,
const int    page_reserved_eos
){

    const int kv_budget = topk_val + page_reserved_bos + page_reserved_eos;
    const int block_size = blockDim.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int* token_indices = req_to_token + req_indices[bx] * req_to_token_stride;
    const int kv_len = cache_seq_lens[bx];
    const int page_len = (kv_len + page_size - 1) / page_size;
    
    int* dense_output = dense_kv_indices + dense_kv_indptr[bx * num_kv_heads + by];
    int* sparse_output = sparse_kv_indices + sparse_kv_indptr[bx * num_kv_heads + by];
    int* dense_output_pt  = dense_page_table + (bx * num_kv_heads + by) * page_table_stride;
    int* sparse_output_pt  = sparse_page_table + (bx * num_kv_heads + by) * page_table_stride;
    int pos = tx;
    while(pos < page_len){
            int data = token_indices[pos * page_size];
            dense_output[pos] = (data / page_size) * (num_kv_heads) + by;
            dense_output_pt[pos] = (data / page_size) * (num_kv_heads) + by;
            pos += block_size;
    }

    if(page_len <= kv_budget){
        int pos = tx;
        while(pos < page_len){
            sparse_output[pos] = dense_output[pos];
            sparse_output_pt[pos] = dense_output[pos];
            pos += block_size;
        }
    }else{
        int pos = tx;
        while(pos < page_reserved_bos){
            sparse_output[pos] = dense_output[pos];
            sparse_output_pt[pos] = dense_output[pos];
            pos += block_size;
        }

        pos = tx;
        while(pos < page_reserved_eos){
            int data = token_indices[(page_len-pos-1) * page_size];
            sparse_output[kv_budget-pos-1] = (data / page_size) * (num_kv_heads) + by;
            sparse_output_pt[kv_budget-pos-1] = (data / page_size) * (num_kv_heads) + by;
            pos += block_size;
        }

    }

}

void sglang_plan_decode_fa3(
const at::Tensor& cached_seq_lens,
at::Tensor&       dense_kv_indptr,
at::Tensor&       dense_kv_indices,
at::Tensor&       sparse_kv_indptr,
at::Tensor&       sparse_kv_indices,
at::Tensor&       dense_page_table,
at::Tensor&       dense_cache_seqlens,
at::Tensor&       sparse_page_table,
at::Tensor&       sparse_cache_seqlens,
const at::Tensor& req_to_token,
const at::Tensor& req_indices,
at::Tensor&       winfo_q_indices,
at::Tensor&       winfo_kv_offsets,
at::Tensor&       winfo_kv_lens,
at::Tensor&       winfo_num_workloads,
at::Tensor&       winfo_chunk_size,
const int64_t     page_size,
const int64_t     num_kv_heads,
const int64_t     topk_val,
const int64_t     page_reserved_bos,
const int64_t     page_reserved_eos,
const int64_t     max_chunk_size,
const int64_t     min_chunk_size
){

#ifdef VORTEX_DEBUG
    TORCH_CHECK(req_to_token.dtype() == torch::kInt32);
    TORCH_CHECK(req_indices.dtype() == torch::kInt64);
    TORCH_CHECK(cached_seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(dense_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(sparse_kv_indptr.dtype() == torch::kInt32);
    TORCH_CHECK(sparse_kv_indices.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_q_indices.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_kv_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_kv_lens.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_num_workloads.dtype() == torch::kInt32);
    TORCH_CHECK(winfo_chunk_size.dtype() == torch::kInt32);
    TORCH_CHECK(page_size >= 1);
    TORCH_CHECK(num_kv_heads >= 1);
    TORCH_CHECK(topk_val >= 1);
    TORCH_CHECK(page_reserved_bos >= 0);
    TORCH_CHECK(page_reserved_eos >= 1);
    TORCH_CHECK(min_chunk_size >= 1);
    TORCH_CHECK(max_chunk_size > min_chunk_size);
#endif

    const int batch_size = cached_seq_lens.size(0);
    const int eff_batch_size = batch_size * num_kv_heads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Compute Indptr and Workload

    Sgl_Decode_Plan_Indptr_FA3_Kernel<<<1, 1024, 0, stream>>>(
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_cache_seqlens.data_ptr<int>(),
        sparse_cache_seqlens.data_ptr<int>(),
        batch_size,
        num_kv_heads,
        page_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );

    if (eff_batch_size <= 1024){
    Sgl_Decode_Plan_Workload_Kernel<1><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    } else if (eff_batch_size <= 2048){
    Sgl_Decode_Plan_Workload_Kernel<2><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    } else if (eff_batch_size <= 4096){
    Sgl_Decode_Plan_Workload_Kernel<4><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    }  else if (eff_batch_size <= 8192){
    Sgl_Decode_Plan_Workload_Kernel<8><<<1, 1024, 0, stream>>>(
        dense_kv_indptr.data_ptr<int>(),
        winfo_q_indices.data_ptr<int>(),
        winfo_kv_offsets.data_ptr<int>(),
        winfo_kv_lens.data_ptr<int>(),
        winfo_num_workloads.data_ptr<int>(),
        winfo_chunk_size.data_ptr<int>(),
        max_chunk_size,
        min_chunk_size,
        eff_batch_size,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
    }  

    // Fill in indices for dense and sparse (eos and bos)

    const int req_to_token_stride = req_to_token.size(1);
    const int page_table_stride = dense_page_table.size(1);
    dim3 nblks(batch_size, num_kv_heads);
    dim3 nthrs(512);
    Sgl_Decode_Plan_Indices_FA3_Kernel<<<nblks, nthrs, 0, stream>>>(
        req_to_token.data_ptr<int>(),
        req_indices.data_ptr<long>(),
        cached_seq_lens.data_ptr<int>(),
        dense_kv_indptr.data_ptr<int>(),
        sparse_kv_indptr.data_ptr<int>(),
        dense_kv_indices.data_ptr<int>(),
        sparse_kv_indices.data_ptr<int>(),
        dense_page_table.data_ptr<int>(),
        sparse_page_table.data_ptr<int>(),
        req_to_token_stride,
        page_table_stride,
        page_size,
        num_kv_heads,
        topk_val,
        page_reserved_bos,
        page_reserved_eos
    );
};
