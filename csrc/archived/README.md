# Archived TopK kernels

These files are **not compiled** (not listed in `setup.py`) and are kept only
for historical reference.

- `topk_slgang_ori.cu` — the original SGLang TopK reference kernel (typo in
  the filename is intentional, matches the upstream commit it was adapted
  from). Superseded by the fused `fast_topk_vortex` path in
  `../topk_sglang.cu`.
- `topk_sglang_ori_fastpath.cu` — the `fast_topk_ori` /
  `TopKOutput_Ori_Kernel` / `launch_ori_kernel` code extracted out of
  `../topk_sglang.cu`. It was the "zero mapping overhead" fast path with
  flexible `radix_bits` (4–10). We no longer test it — mode 0 now goes
  through the standard fused kernel with `MAPPING_NONE`, which pays no
  mapping overhead because `mapped_convert_to_uint8` degenerates to
  `convert_to_uint8` in that branch.

If you need to resurrect any of this, add the `.cu` to `setup.py` and
re-export its entry points from `../register.cc` / `../register.h`.
