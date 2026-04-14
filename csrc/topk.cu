#include "register.h"
#include <cub/cub.cuh>


template <int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void TopKOutput_F32_Kernel(
const float* __restrict__ score,
const int*   __restrict__ dense_kv_indptr,
const int*   __restrict__ sparse_kv_indptr,
const int*   __restrict__ dense_kv_indices,
int*         __restrict__ sparse_kv_indices,
const int    topk_val,
const int    page_reserved_bos,
const int    page_reserved_eos)
{
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const float* __restrict__ score_blk = score + start;
    const int*   __restrict__ idx_blk   = dense_kv_indices + start;
    int*         __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    float key[ITEM_PER_THREAD];
    int   val[ITEM_PER_THREAD];

    using BLF  = cub::BlockLoad<float, NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BLI  = cub::BlockLoad<int,   NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BSI  = cub::BlockStore<int,  NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;

    __shared__ union {
        typename BLF::TempStorage  lf;
        typename BLI::TempStorage  li;
        typename BSI::TempStorage  si;
        typename Sort::TempStorage sort;
    } temp;

    BLF(temp.lf).Load(score_blk, key, nblk, -INFINITY);
    __syncthreads();
    BLI(temp.li).Load(idx_blk,   val, nblk, 0);
    __syncthreads();

    Sort(temp.sort).SortDescending(key, val);
    __syncthreads();

    const int valid_out = min(topk_val, nblk);
    BSI(temp.si).Store(out_blk, /*per-thread regs*/ val, valid_out);
}


template <int NUM_THREADS, int ITEM_PER_THREAD>
__global__ void TopKOutput_BF16_Kernel(
const __nv_bfloat16* __restrict__ score,
const int*           __restrict__ dense_kv_indptr,
const int*           __restrict__ sparse_kv_indptr,
const int*           __restrict__ dense_kv_indices,
int*                 __restrict__ sparse_kv_indices,
const int            topk_val,
const int            page_reserved_bos,
const int            page_reserved_eos)
{
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;

    const int start = dense_kv_indptr[bx] + page_reserved_bos;
    const int end   = dense_kv_indptr[bx + 1] - page_reserved_eos;
    const int nblk  = end - start;
    if (nblk <= topk_val) return;

    const __nv_bfloat16* __restrict__ score_blk = score + start;
    const int*   __restrict__ idx_blk   = dense_kv_indices + start;
    int*         __restrict__ out_blk   = sparse_kv_indices + sparse_kv_indptr[bx] + page_reserved_bos;

    const __nv_bfloat16 ninf_bf16 = __float2bfloat16(-CUDART_INF_F);

    __nv_bfloat16 key_bf16[ITEM_PER_THREAD];
    float key[ITEM_PER_THREAD];
    int   val[ITEM_PER_THREAD];

    using BLF  = cub::BlockLoad<__nv_bfloat16, NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BLI  = cub::BlockLoad<int,   NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BSI  = cub::BlockStore<int,  NUM_THREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using Sort = cub::BlockRadixSort<float, NUM_THREADS, ITEM_PER_THREAD, int>;

    __shared__ union {
        typename BLF::TempStorage  lf;
        typename BLI::TempStorage  li;
        typename BSI::TempStorage  si;
        typename Sort::TempStorage sort;
    } temp;

    BLF(temp.lf).Load(score_blk, key_bf16, nblk, ninf_bf16);

    #pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; ++i){
        key[i] = __bfloat162float(key_bf16[i]);
    }
    __syncthreads();

    BLI(temp.li).Load(idx_blk,   val, nblk, 0);
    __syncthreads();

    Sort(temp.sort).SortDescending(key, val);
    __syncthreads();

    const int valid_out = min(topk_val, nblk);
    BSI(temp.si).Store(out_blk, /*per-thread regs*/ val, valid_out);
}



void topk_output(
const at::Tensor& x,
const at::Tensor& dense_kv_indptr,
const at::Tensor& dense_kv_indices,
const at::Tensor& sparse_kv_indptr,
at::Tensor&       sparse_kv_indices,
const int64_t     eff_batch_size,
const int64_t     topk_val,
const int64_t     reserved_bos,
const int64_t     reserved_eos,
const int64_t     max_num_pages
){


    dim3 nblks(eff_batch_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (max_num_pages <= 128){
        TopKOutput_BF16_Kernel<128, 1><<<nblks, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 256){
        TopKOutput_BF16_Kernel<128, 2><<<nblks, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 512){
        TopKOutput_BF16_Kernel<128, 4><<<nblks, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 1024){
        TopKOutput_BF16_Kernel<256, 4><<<nblks, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 2048){
        TopKOutput_BF16_Kernel<256, 8><<<nblks, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 4096){
        TopKOutput_BF16_Kernel<512, 8><<<nblks, 512, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else if (max_num_pages <= 8192){
        TopKOutput_BF16_Kernel<512, 16><<<nblks, 512, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_kv_indptr.data_ptr<int>(),
            sparse_kv_indptr.data_ptr<int>(),
            dense_kv_indices.data_ptr<int>(),
            sparse_kv_indices.data_ptr<int>(),
            topk_val,
            reserved_bos,
            reserved_eos
        );
    } else {
        TORCH_CHECK(false, "topk_output: max_num_pages=", max_num_pages,
                    " exceeds the supported template ladder (8192).");
    }

}