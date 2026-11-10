import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 相对于本文件的绝对路径，避免依赖调用时的当前工作目录
HERE = os.path.dirname(os.path.abspath(__file__))

setup(
    name='irra_render',
    version='0.0.0',
    ext_modules=[
        CUDAExtension(
            name='irra_render',
            sources=['render.cu'],
            include_dirs=[HERE],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-march=native', '-mtune=native'],
                # 不再写死 -gencode：由 PyTorch 根据当前 GPU 自动选择，
                # 也可用环境变量 TORCH_CUDA_ARCH_LIST 指定（例如 "8.6"）。
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-Xcompiler', '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)}
)
