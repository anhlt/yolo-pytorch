from distutils.core import setup
import numpy as np
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CUDAExtension
import glob
import os

requirements = ["torch", "torchvision"]
print(torch.__version__)

this_dir = os.path.dirname(os.path.abspath(__file__))
main_file = glob.glob(os.path.join(this_dir, "*.cpp"))
source_cpu = glob.glob(os.path.join(this_dir, "cpu", "*.cpp"))
source_cuda = glob.glob(os.path.join(this_dir, "cuda", "*.cu"))

sources = main_file + source_cpu
extension = CppExtension
extra_compile_args = {"cxx": []}
define_macros = []

if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
    extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__"]

sources = [os.path.join(this_dir, s) for s in sources]
include_dirs = [this_dir]

print(sources)

ext_modules = [
    extension(
        "nms_module",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
