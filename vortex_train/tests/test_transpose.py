"""The transposed index. Small file, highest-value tests in the repo.

A wrong transpose *silently drops gradient*: dk/dv for the missed (query block,
kv block) pairs are simply never accumulated. Loss still decreases, nothing warns,
and the model trains to a different objective than the one specified. So the
transpose is checked as an exact set equality against the forward pattern, not
merely for plausible-looking shapes.
"""
from __future__ import annotations

import pytest
import torch

from vortex_train.kernels.transpose import transpose_pattern, verify_transpose

from .patterns import ALL_KINDS, make

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPE = dict(b=2, hkv=2, seqlen_q=128, seqlen_kv=128, block_q=32, block_kv=32)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_transpose_is_exact_inverse(kind):
    p = make(kind, **SHAPE, device="cuda")
    off, ind = transpose_pattern(p)
    verify_transpose(p, off, ind)          # exact set equality per kv block


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_csr_totals(kind):
    """Total CSR entries == total valid selections. Catches lost/duplicated writes."""
    p = make(kind, **SHAPE, device="cuda")
    off, ind = transpose_pattern(p)
    ar = torch.arange(p.max_selected, device=p.idx.device, dtype=torch.int32)
    valid = (ar[None, None, None, :] < p.cnt[..., None])
    for b in range(p.batch):
        for h in range(p.num_kv_heads):
            assert int(off[b, h, -1]) == int(valid[b, h].sum()), \
                f"CSR total mismatch at (b={b}, h={h})"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_offsets_monotone(kind):
    p = make(kind, **SHAPE, device="cuda")
    off, _ = transpose_pattern(p)
    assert bool((off[..., 1:] >= off[..., :-1]).all()), "CSR offsets not monotone"
    assert int(off[..., 0].max()) == 0, "CSR must start at 0"


def test_no_host_sync_in_transpose():
    """The transpose must not read a device value onto the host.

    Any `.item()`/`.cpu()` here would serialize the step. Checked by source
    inspection because it is a *structural* property, not a numerical one.
    """
    import inspect

    from vortex_train.kernels import transpose as mod

    src = inspect.getsource(mod.transpose_pattern)
    for banned in (".item()", ".cpu()", ".tolist()", "int(", "float("):
        assert banned not in src, f"transpose_pattern contains host sync: {banned}"


def test_imbalance_is_real():
    """Document the load imbalance the kv-major kernel must tolerate.

    Not a pass/fail on performance — it records *why* dk/dv is a separate kernel
    with its own traversal, so the reason survives in the test suite.
    """
    p = make("random_topk", b=1, hkv=1, seqlen_q=2048, seqlen_kv=2048,
             block_q=32, block_kv=32, device="cuda", topk=8)
    off, _ = transpose_pattern(p)
    work = (off[0, 0, 1:] - off[0, 0, :-1]).float()
    nonzero = work[work > 0]
    ratio = float(nonzero.max() / nonzero.mean())
    print(f"\ndk/dv work per kv block: max={int(nonzero.max())} "
          f"mean={float(nonzero.mean()):.1f} max/mean={ratio:.2f}x")
    assert ratio > 1.0, "expected some imbalance; a perfectly flat pattern is suspicious"
