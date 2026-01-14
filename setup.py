from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vortex_torch',
    version='0.2.0',
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7",
        "lighteval[math]==0.12.2"
    ],
    ext_modules=[
        CUDAExtension(
            name='vortex_torch_C',
            sources=[
                'csrc/register.cc',
                'csrc/utils_sglang.cu',
                'csrc/topk.cu',
                'csrc/unified_reduce.cu',
                'csrc/store_kv_unified.cu',
                'csrc/warp_lru_cache.cu',
                'csrc/copy_sparse_kv.cu'
            ],
            include_dirs=['csrc'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_89,code=sm_89',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_80,code=sm_80'
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
