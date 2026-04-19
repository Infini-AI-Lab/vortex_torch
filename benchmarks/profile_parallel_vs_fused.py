"""
Driver for Nsight Compute profiling of the parallel vs fused TopK
kernels. Designed to be launched under `ncu` with --launch-skip and
--launch-count to isolate a specific kernel launch from warmup.

The script does exactly:
    args.warmup matching-kernel launches (skipped by ncu --launch-skip)
    args.iters  matching-kernel launches (captured by ncu --launch-count)

Pair --launch-skip/--launch-count with --kernel-name so unrelated
launches (torch initializers, cublas, etc.) don't pollute the counts.
"""
import argparse
import torch
from vortex_torch_C import (
    topk_output_sglang_fused,
    topk_output_sglang_parallel,
)


def make_inputs(eff_bs: int, pages: int, topk: int):
    reserved = 0
    dense_indptr = torch.arange(
        0, (eff_bs + 1) * pages, pages, dtype=torch.int32, device="cuda"
    )
    sparse_indptr = torch.arange(
        0, (eff_bs + 1) * topk, topk, dtype=torch.int32, device="cuda"
    )
    dense_indices = torch.arange(eff_bs * pages, dtype=torch.int32, device="cuda")
    torch.manual_seed(0)
    x = torch.randn(eff_bs * pages, 1, 1, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(eff_bs * topk, dtype=torch.int32, device="cuda")
    return x, dense_indptr, sparse_indptr, dense_indices, out, reserved


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        choices=["A", "B"],
        required=True,
        help="A: topk=2048 pages=32K ; B: topk=30 pages=2K",
    )
    p.add_argument("--eff-bs", type=int, default=1)
    p.add_argument(
        "--mode", type=int, choices=[15, 16], required=True,
        help="15=MAPPING_SHIFT_POW2, 16=MAPPING_SHIFT_POW3",
    )
    p.add_argument(
        "--power", type=float, default=0.5,
        help="Pivot (p) for the shift_pow transforms. 0.5 matches the "
             "autotune default for Qwen3-1.7B softmax scores.",
    )
    p.add_argument("--num-splits", type=int, default=4)
    p.add_argument("--kernel", choices=["fused", "parallel"], required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=1)
    args = p.parse_args()

    pages, topk = (32768, 2048) if args.config == "A" else (2048, 30)
    x, dense_indptr, sparse_indptr, dense_indices, out, reserved = make_inputs(
        args.eff_bs, pages, topk
    )

    if args.kernel == "fused":
        def call():
            topk_output_sglang_fused(
                x, dense_indptr, sparse_indptr, dense_indices, out,
                args.eff_bs, topk, reserved, reserved, pages,
                args.mode, args.power, None, None,
            )
    else:
        def call():
            topk_output_sglang_parallel(
                x, dense_indptr, sparse_indptr, dense_indices, out,
                args.eff_bs, topk, reserved, reserved, pages,
                args.num_splits, args.mode, args.power, None, None,
            )

    # Warmup: specialised kernel is JIT-instantiated and cudaFuncSetAttribute
    # is cached; these launches dominate the first-call overhead and we want
    # ncu to skip past them.
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()

    # Profiled region. Wrap in NVTX so the same script is also useful under
    # Nsight Systems (nsys) if you prefer a timeline view.
    torch.cuda.nvtx.range_push(
        f"profile-{args.kernel}-mode{args.mode}-cfg{args.config}-eff{args.eff_bs}"
    )
    for _ in range(args.iters):
        call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
