"""Correctness harness for the trtllm/block-table approxTopK leaf (bf16).

The leaf runs a two-pass 8-bit radix over the *raw bf16 key*. bf16 is 16
bits wide, so two rounds resolve the key completely — which gives two
properties this harness pins down:

1. **Hard invariants** (any tolerate_ratio) — approximation may swap
   near-ties, it may never corrupt the output:
     * exactly ``target_k`` slots written, all inside the row's window
     * every emitted id is a real block id from that row's dense table
     * no duplicates
     * reserved BOS slots and everything past ``target_k`` untouched

2. **tolerate_ratio = 0.0 is EXACT** — the selected score multiset equals
   the true top-k score multiset. (Compared as multisets because equal
   bf16 values are genuinely interchangeable.)

3. **recall >= 1 - tol** — a guarantee, not an observation. The gate only
   fires with >= ceil((1-tol)*k) pass-1 strict winners, and those sit in
   bins strictly above the threshold bin, so they are exactly the top
   ``n_strict`` elements. Scored tie-robustly: a selection counts as a hit
   when its score >= the k-th largest score.

The flashinfer/CSR leaf is run alongside for context only (it keys on
fp32 and resolves 2 of 4 bytes, so it is *not* exact at tol=0); it is
printed, never asserted on.

Run:  python examples/misc/test_approx_topk_trtllm.py
"""
import torch

from vortex_torch.custom_ops import find

BOS, EOS = 1, 2
SENTINEL = -777
DTYPE = torch.bfloat16


def build_case(eff_bs, max_blocks, block_size, k, seed):
    """Synthetic trtllm planner state: 2D [eff_bs, max_blocks] buffers."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dense_blocks = torch.randint(k + BOS + EOS + 8, max_blocks + 1,
                                 (eff_bs,), generator=g, device="cuda")
    # seqlens are TOKEN counts; the kernel ceil-divides by block_size.
    dense_seqlens = (dense_blocks * block_size).int()
    sparse_seqlens = (torch.full_like(dense_blocks, k + BOS + EOS)
                      * block_size).int()
    scores = torch.randn(eff_bs, max_blocks, generator=g,
                         device="cuda", dtype=torch.float32).to(DTYPE)
    # Distinct block ids per row so a wrong row/offset is detectable.
    dense_bt = (torch.arange(eff_bs, device="cuda").unsqueeze(1) * 100000
                + torch.randperm(max_blocks, generator=g,
                                 device="cuda").unsqueeze(0)).int()
    sparse_bt = torch.full((eff_bs, max_blocks), SENTINEL,
                           dtype=torch.int32, device="cuda")
    return (scores, dense_seqlens, sparse_seqlens, dense_bt, sparse_bt,
            dense_blocks)


def run_trtllm(tol, case, eff_bs, max_blocks, block_size):
    scores, dense_seqlens, sparse_seqlens, dense_bt, sparse_bt, _ = case
    fn = find("topk_output", "trtllm", approx=True)(tolerate_ratio=tol,
                                                    verbose=False)
    fn(scores, dense_seqlens, sparse_seqlens, dense_bt, sparse_bt,
       eff_bs, BOS, EOS, max_blocks, block_size)
    torch.cuda.synchronize()
    return sparse_bt


def run_flashinfer(tol, case, eff_bs, k):
    """Same per-row score windows, repacked into the CSR/ragged layout."""
    scores, _, _, dense_bt, _, dense_blocks = case
    counts = dense_blocks.tolist()
    dense_indptr = torch.tensor([0] + list(torch.cumsum(
        torch.tensor(counts), 0)), dtype=torch.int32, device="cuda")
    sparse_indptr = torch.arange(
        0, (eff_bs + 1) * (k + BOS + EOS), k + BOS + EOS,
        dtype=torch.int32, device="cuda")
    packed_scores = torch.cat([scores[r, :counts[r]] for r in range(eff_bs)])
    packed_ids = torch.cat([dense_bt[r, :counts[r]] for r in range(eff_bs)])
    out = torch.full(((k + BOS + EOS) * eff_bs,), SENTINEL,
                     dtype=torch.int32, device="cuda")
    fn = find("topk_output", "flashinfer", approx=True)(tolerate_ratio=tol,
                                                        verbose=False)
    fn(packed_scores, dense_indptr, sparse_indptr, packed_ids, out,
       eff_bs, BOS, EOS, int(dense_indptr[-1]))
    torch.cuda.synchronize()
    return out.view(eff_bs, k + BOS + EOS)


def score_of(selected_ids, row_ids, win):
    """Map emitted block ids back to their scores (order-independent)."""
    pos = (row_ids.unsqueeze(0) == selected_ids.unsqueeze(1)).float().argmax(dim=1)
    return win[pos]


def evaluate(selected_rows, case, eff_bs, k, invariants):
    """Returns (mean tie-robust recall, all-rows-exact flag)."""
    scores, _, _, dense_bt, _, dense_blocks = case
    recalls, exact = [], True
    for r in range(eff_bs):
        nblk = int(dense_blocks[r]) - BOS - EOS
        got = selected_rows[r]
        row_ids = dense_bt[r, BOS:BOS + nblk]
        if invariants:
            assert (got != SENTINEL).all(), f"row {r}: unwritten slot"
            assert torch.isin(got, row_ids).all(), \
                f"row {r}: id outside this row's dense window"
            assert len(torch.unique(got)) == k, f"row {r}: duplicate ids"
        win = scores[r, BOS:BOS + nblk].float()
        true_top = torch.topk(win, k).values
        thresh = true_top.min()
        got_scores = score_of(got, row_ids, win)
        # Tie-robust: anything at or above the k-th largest score is as
        # good as a "true" top-k member.
        recalls.append((got_scores >= thresh).sum().item() / k)
        if not torch.equal(torch.sort(got_scores, descending=True).values,
                           torch.sort(true_top, descending=True).values):
            exact = False
    return sum(recalls) / len(recalls), exact


def check(tol, eff_bs=8, max_blocks=512, block_size=32, k=29, seed=0):
    case = build_case(eff_bs, max_blocks, block_size, k, seed)
    sparse_bt = run_trtllm(tol, case, eff_bs, max_blocks, block_size)
    for r in range(eff_bs):
        assert sparse_bt[r, :BOS].eq(SENTINEL).all(), f"row {r}: BOS clobbered"
        assert sparse_bt[r, BOS + k:].eq(SENTINEL).all(), \
            f"row {r}: wrote past target_k"
    trt_rec, trt_exact = evaluate(sparse_bt[:, BOS:BOS + k], case, eff_bs, k, True)
    fi_rows = run_flashinfer(tol, case, eff_bs, k)
    fi_rec, fi_exact = evaluate(fi_rows[:, BOS:BOS + k], case, eff_bs, k, False)
    return trt_rec, trt_exact, fi_rec, fi_exact


def main():
    print("trtllm approxTopK (bf16, 2-pass radix on the raw bf16 key)")
    print(f"{'tol':>6}{'recall':>9}{'exact?':>8}{'floor 1-tol':>13}"
          f"{'| fi recall':>13}{'fi exact?':>11}")
    for tol in (0.0, 0.05, 0.15, 0.25, 0.45, 0.75, 0.95, 1.0):
        rec, exact, fi_rec, fi_exact = check(tol)
        print(f"{tol:>6}{rec:>9.4f}{str(exact):>8}{1 - tol:>13.4f}"
              f"{fi_rec:>13.4f}{str(fi_exact):>11}")
        # Guaranteed floor: the gate needs ceil((1-tol)*k) pass-1 strict
        # winners, and those are exactly the top n_strict elements.
        assert rec >= (1.0 - tol) - 1e-9, \
            f"recall {rec:.4f} below guaranteed floor {1 - tol:.4f} at tol={tol}"
        if tol == 0.0:
            assert exact, "tolerate_ratio=0.0 must give the exact top-k"

    print("\nedge shapes (bf16):")
    for (eff_bs, max_blocks, bsz, k) in [
        (1, 64, 16, 1), (4, 128, 64, 7), (16, 1024, 32, 128),
        (32, 256, 32, 64), (3, 128, 32, 90),
    ]:
        rec0, exact0, _, _ = check(0.0, eff_bs, max_blocks, bsz, k, seed=7)
        rec, _, _, _ = check(0.15, eff_bs, max_blocks, bsz, k, seed=7)
        print(f"  eff_bs={eff_bs:>3} blocks={max_blocks:>5} bs={bsz:>3} k={k:>4}"
              f" -> tol0 recall {rec0:.4f} exact={exact0}"
              f"   tol.15 recall {rec:.4f}")
        assert exact0, f"tol=0 not exact on shape eff_bs={eff_bs} k={k}"
        assert rec >= 0.85 - 1e-9, f"tol=0.15 recall too low: {rec}"

    # fp32 scores must be rejected loudly, not silently mis-read.
    case = build_case(4, 128, 32, 7, seed=1)
    scores32 = case[0].float()
    fn = find("topk_output", "trtllm", approx=True)(tolerate_ratio=0.15,
                                                    verbose=False)
    try:
        fn(scores32, case[1], case[2], case[3], case[4], 4, BOS, EOS, 128, 32)
        raise AssertionError("fp32 scores should have been rejected")
    except RuntimeError as e:
        assert "bf16-only" in str(e), f"unexpected error text: {e}"
        print("\nfp32 input correctly rejected with a clear message")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
