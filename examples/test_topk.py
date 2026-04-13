import torch
import triton
# topk_output_sglang expects sparse_kv_indptr before dense_kv_indices (unlike topk_output).
from vortex_torch_C import topk_output_sglang as topk_output

SEQ_LENS = [4096]
BATCH_SIZES = [256]

K = 32
RESERVE_BOS = 0
RESERVE_EOS = 0
DEVICE = "cuda"


def make_inputs(batch_size, seq_len, k, reserve_bos, reserve_eos, device="cuda"):
    dense_kv_indptr = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device=device
    )

    dense_kv_indices = torch.arange(
        0, batch_size * seq_len, dtype=torch.int32, device=device
    )

    scores = torch.randn(
        batch_size * seq_len, dtype=torch.bfloat16, device=device
    )

    # ✅ Fixed CSR-style sparse indptr
    sparse_kv_indptr = torch.arange(
        0, batch_size * k + 1, k, dtype=torch.int32, device=device
    )

    sparse_kv_indices = torch.empty(
        batch_size * k, dtype=torch.int32, device=device
    )

    return (
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices,
    )


def bench_one(batch_size, seq_len, k, reserve_bos, reserve_eos):
    (
        scores,
        dense_kv_indptr,
        dense_kv_indices,
        sparse_kv_indptr,
        sparse_kv_indices,
    ) = make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        k=k,
        reserve_bos=reserve_bos,
        reserve_eos=reserve_eos,
        device=DEVICE,
    )

    def fn():
        topk_output(
            scores,
            dense_kv_indptr,
            sparse_kv_indptr,
            dense_kv_indices,
            sparse_kv_indices,
            batch_size,
            k,
            reserve_bos,
            reserve_eos,
            seq_len,
        )

    # warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        fn,
        warmup=100,
        rep=1000,
        return_mode="mean",
    )
    return ms


def main():
    torch.cuda.init()

    results = {}

    for bs in BATCH_SIZES:
        results[bs] = {}
        for seq_len in SEQ_LENS:
            ms = bench_one(
                batch_size=bs,
                seq_len=seq_len,
                k=K,
                reserve_bos=RESERVE_BOS,
                reserve_eos=RESERVE_EOS,
            )
            results[bs][seq_len] = ms
            print(f"bs={bs:>3}, seq_len={seq_len:>4} -> {ms:.6f} ms")

    print("\nLatency table (ms):")
    header = "bs\\seq".ljust(10) + "".join(f"{s:>12}" for s in SEQ_LENS)
    print(header)

    for bs in BATCH_SIZES:
        row = f"{bs:<10}" + "".join(f"{results[bs][s]:>12.4f}" for s in SEQ_LENS)
        print(row)


if __name__ == "__main__":
    main()