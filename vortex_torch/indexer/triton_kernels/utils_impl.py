def next_pow2(n: int) -> int:
    """Smallest power of two >= ``n``.

    Triton requires the length of a ``tl.arange`` to be a power of two. Kernels
    that walk a tensor dimension with ``tl.arange`` therefore have to round that
    dimension up to the next power of two and mask off the surplus lanes, while
    still using the real extent for pointer arithmetic.
    """
    assert n >= 1, f"next_pow2 expects n >= 1, got {n}"
    return 1 << (n - 1).bit_length()
