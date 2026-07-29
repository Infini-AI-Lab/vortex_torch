// Approximate top-k for the trtllm attention backend (indptr-free,
// block-table ABI). bf16 scores only.
//
// Two-pass 8-bit radix selection over the *raw bf16 key*. bf16 is 16 bits
// wide, so two 8-bit rounds resolve the key completely — unlike the
// fp32-promoted flashinfer/CSR sibling, which refines only 2 of 4 bytes
// and is therefore never exact. Here ``tolerate_ratio = 0.0`` yields the
// true top-k (ties between *identical* bf16 values aside, which are
// genuinely interchangeable).
//
//   Pass 1 — histogram the high byte of the key. Reverse-cumsum gives
//            hist[b] = #elements with bin >= b; the threshold bin tbin0
//            is where hist[tbin0] > k >= hist[tbin0+1]. Elements above
//            tbin0 are strict winners: n_strict = hist[tbin0+1].
//
//   Gate — if n_strict >= (1 - tolerate_ratio) * k, stop after one pass:
//          emit the strict winners and fill the k - n_strict slots still
//          owed from the threshold bin in atomic-arrival order. That
//          arrival-order fill is the entire approximation — the dropped
//          candidates tie with the kept ones in the top 8 key bits.
//
//   Pass 2 — otherwise refine: emit the byte-0 strict winners and
//            sub-histogram the low byte of the tbin0 elements, then emit
//            byte-1 strict winners and fill the remainder. After this
//            round all 16 key bits are resolved, so the result is exact.
//
// At tolerate_ratio = 0.0 the gate reduces to n_strict == k, i.e. pass 2
// is skipped only when it would change nothing. At 1.0 it always fires
// (cheapest, coarsest).
//
// Addressing is block-table-shaped, mirroring
// ``topk_output/trtllm/default/kernel.cu``:
//   * score / dense_block_tables / sparse_block_tables are 2D buffers
//     keyed at ``[eff_bs, max_blocks_per_seq]``.
//   * Per-row block counts come from ``dense_seqlens[bx]`` /
//     ``sparse_seqlens[bx]`` (token counts, written by the trtllm
//     planner) and ``block_size``:
//         block_count = (tokens + block_size - 1) / block_size
// There is intentionally **no** ``dense_kv_indptr`` / ``sparse_kv_indptr``
// argument — this kernel never reads a prefix-sum buffer.
//
// Substitutions:
//   __THREADS_PER_BLOCK__  — CTA size.
//   __VORTEX_MAX_TOPK__    — capacity of the shared staging buffer.
//   __TOLERATE_RATIO__     — float in [0.0, 1.0], the single-pass gate.
//
// Note: indices are unsorted within a request (atomic-arrival order).

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreadsPerBlock = __THREADS_PER_BLOCK__;
constexpr int VORTEX_MAX_TOPK  = __VORTEX_MAX_TOPK__;
constexpr float kTolerateRatio = static_cast<float>(__TOLERATE_RATIO__);
constexpr int RADIX = 256;

// bf16 -> total-order uint16 (sign-flip). Higher key == higher score.
// Operates on the raw bf16 bits: no fp32 promotion, no precision loss,
// so the full key is covered by two 8-bit rounds.
__device__ __forceinline__ uint16_t score_to_key16(__nv_bfloat16 x) {
    const uint16_t bits = __bfloat16_as_ushort(x);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits | 0x8000u);
}

__device__ void approx_topk_inner(
    const __nv_bfloat16* __restrict__ input,
    int*                 __restrict__ index,
    const int            length,
    const int            target_k,
    const int            min_strict)   // gate: (1 - tol) * target_k, rounded up
{
    constexpr int BLOCK_SIZE = kThreadsPerBlock;

    alignas(128) __shared__ int hist_buf[2][RADIX + 128];
    alignas(128) __shared__ int s_threshold_bin;
    alignas(128) __shared__ int s_counter;        // strict-winner write head
    alignas(128) __shared__ int s_last_remain;    // atomic-arrival countdown

    auto& hist = hist_buf[0];
    const int tx = threadIdx.x;

    // Reverse inclusive cumulative sum; final result lands in hist_buf[0].
    auto run_cumsum = [&] {
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX);
            if (C10_LIKELY(tx < RADIX)) {
                const int j = 1 << i;
                const int k = i & 1;
                int v = hist_buf[k][tx];
                if (tx < RADIX - j) v += hist_buf[k][tx + j];
                hist_buf[k ^ 1][tx] = v;
            }
            __syncthreads();
        }
    };

    // ---------------- Pass 1: histogram on the high byte ----------------
    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = static_cast<uint32_t>(score_to_key16(input[idx]) >> 8);
        ::atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    run_cumsum();

    // hist[b] == #elements with bin >= b. The crossing bin always exists:
    // hist[0] == length > target_k and hist[RADIX] == 0 <= target_k.
    if (tx < RADIX && hist[tx] > target_k && hist[tx + 1] <= target_k) {
        s_threshold_bin = tx;
        s_counter       = 0;
        s_last_remain   = target_k - hist[tx + 1];   // slots owed by tbin0
    }
    __syncthreads();

    const int tbin0        = s_threshold_bin;
    const int last_remain0 = s_last_remain;
    const int n_strict0    = target_k - last_remain0;   // == hist[tbin0 + 1]

    // ---------------- Gate: enough strict winners already? ----------------
    if (n_strict0 >= min_strict) {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
            const auto bin = static_cast<uint32_t>(score_to_key16(input[idx]) >> 8);
            if (bin > tbin0) {
                const int pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (bin == tbin0) {
                const int pos = ::atomicAdd(&s_last_remain, -1);
                if (pos > 0) {
                    index[target_k - pos] = idx;
                }
            }
        }
        __syncthreads();
        return;
    }

    // ---------------- Pass 2: byte-0 strict + low-byte sub-histogram ----------------
    if (tx < RADIX + 1) hist[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto key16 = score_to_key16(input[idx]);
        const auto bin0  = static_cast<uint32_t>(key16 >> 8);
        if (bin0 > tbin0) {
            const int pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin0 == tbin0) {
            ::atomicAdd(&hist[key16 & 0xFFu], 1);
        }
    }
    __syncthreads();

    run_cumsum();

    // Find the low-byte threshold bin against the leftover budget.
    if (tx < RADIX && hist[tx] > last_remain0 && hist[tx + 1] <= last_remain0) {
        s_threshold_bin = tx;
        s_last_remain   = last_remain0 - hist[tx + 1];
    }
    __syncthreads();

    const int tbin1 = s_threshold_bin;

    // ---------------- Pass 3: low-byte strict + atomic-arrival ----------------
    // All 16 key bits are now resolved; the only arrival-order fill left is
    // between elements whose bf16 values are bit-identical.
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto key16 = score_to_key16(input[idx]);
        const auto bin0  = static_cast<uint32_t>(key16 >> 8);
        if (bin0 != tbin0) continue;
        const auto bin1 = static_cast<uint32_t>(key16 & 0xFFu);
        if (bin1 > tbin1) {
            const int pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin1 == tbin1) {
            const int pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
                index[target_k - pos] = idx;
            }
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(kThreadsPerBlock)
void ApproxTopKOutput_Kernel(
    const __nv_bfloat16* __restrict__ score,
    const int*           __restrict__ dense_seqlens,    // [eff_bs] tokens
    const int*           __restrict__ sparse_seqlens,   // [eff_bs] tokens
    const int*           __restrict__ dense_block_tables,
    int*                 __restrict__ sparse_block_tables,
    const int            row_stride,                    // = max_blocks_per_seq
    const int            block_reserved_bos,
    const int            block_reserved_eos,
    const int            block_size)
{
    const int bx = blockIdx.x;

    // Same derivation as trtllm/default: the
    // ``Sgl_Decode_Plan_Workload_V2_Kernel<IS_TRTLLM=true>`` planner bakes
    // the ``row * row_stride + col`` linear layout into
    // ``winfo_kv_offsets``, so the per-row score base matches here.
    const int dense_block_len  = (dense_seqlens[bx]  + block_size - 1) / block_size;
    const int sparse_block_len = (sparse_seqlens[bx] + block_size - 1) / block_size;
    const int row_offset = bx * row_stride;
    const int target_k = sparse_block_len - block_reserved_bos - block_reserved_eos;
    const int nblk     = dense_block_len  - block_reserved_bos - block_reserved_eos;

    // Nothing to select: the whole dense range already fits in the budget.
    // Matches trtllm/default — the planner has already populated the row.
    if (nblk <= target_k) return;

    // The staging buffer is sized for VORTEX_MAX_TOPK (256, matching the
    // k_256 leaf). k_256 gets this bound from its ``max_topk_val <= 256``
    // dispatch constraint; this leaf cannot use that constraint, because
    // the approx codegen resolves via ``find(..., approx=True)`` without a
    // ``max_topk_val`` kwarg — the constraint would never be satisfied and
    // the flow would silently fall back to the exact CUB leaf, ignoring
    // tolerate_ratio. So enforce the bound here instead, loudly: silently
    // skipping the row would leave stale block ids and corrupt attention.
    CUDA_KERNEL_ASSERT(target_k <= VORTEX_MAX_TOPK
                       && "approx topk (trtllm): topk_val exceeds "
                          "VORTEX_MAX_TOPK (256)");

    // Gate threshold: require at least ceil((1 - tol) * k) strict winners
    // from pass 1 to skip the refinement pass. tol = 0 => min_strict == k
    // (skip only when pass 2 is provably redundant) => exact top-k.
    int min_strict = static_cast<int>(
        ceilf((1.0f - kTolerateRatio) * static_cast<float>(target_k)));
    if (min_strict < 0)        min_strict = 0;
    if (min_strict > target_k) min_strict = target_k;

    const __nv_bfloat16* __restrict__ score_blk =
        score + row_offset + block_reserved_bos;
    const int* __restrict__ idx_blk =
        dense_block_tables + row_offset + block_reserved_bos;
    int* __restrict__ out_blk =
        sparse_block_tables + row_offset + block_reserved_bos;

    __shared__ int s_indices[VORTEX_MAX_TOPK];

    approx_topk_inner(score_blk, s_indices, nblk, target_k, min_strict);
    __syncthreads();

    // Map the selected positions back to block ids via the dense table.
    const int tx = threadIdx.x;
    for (int i = tx; i < target_k; i += kThreadsPerBlock) {
        out_blk[i] = idx_blk[s_indices[i]];
    }
}

}  // namespace

void topk(
    const at::Tensor& x,
    const at::Tensor& dense_seqlens,
    const at::Tensor& sparse_seqlens,
    const at::Tensor& dense_block_tables,
    at::Tensor&       sparse_block_tables,
    const int64_t     eff_batch_size,
    const int64_t     reserved_bos,
    const int64_t     reserved_eos,
    const int64_t     max_blocks_per_seq,
    const int64_t     block_size)
{
    dim3 nblks(eff_batch_size);
    dim3 nthreads(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (x.scalar_type() == at::ScalarType::BFloat16) {
        ApproxTopKOutput_Kernel<<<nblks, nthreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            dense_seqlens.data_ptr<int>(),
            sparse_seqlens.data_ptr<int>(),
            dense_block_tables.data_ptr<int>(),
            sparse_block_tables.data_ptr<int>(),
            static_cast<int>(max_blocks_per_seq),
            static_cast<int>(reserved_bos),
            static_cast<int>(reserved_eos),
            static_cast<int>(block_size));
    } else {
        // bf16-only by design: the 2-pass radix keys on the raw 16-bit bf16
        // value, which is what makes two rounds cover the whole key. Any
        // other score dtype is a config error, not something to silently
        // convert.
        TORCH_CHECK(false,
                    "topk: unsupported dtype ", x.scalar_type(),
                    " — this approx leaf is bf16-only. Set "
                    "vortex_dtype='bfloat16'.");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "topk kernel failed: ", ::cudaGetErrorString(result));
}
