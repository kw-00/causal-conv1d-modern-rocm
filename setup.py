# Copyright (c) 2024, Tri Dao.

import sys
import warnings
import os
import re
import shutil
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    HIP_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("CAUSAL_CONV1D_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("CAUSAL_CONV1D_SKIP_CUDA_BUILD", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64"):
            return "linux_aarch64"
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_hip_version(rocm_dir):
    hipcc_bin = "hipcc" if rocm_dir is None else os.path.join(rocm_dir, "bin", "hipcc")
    try:
        raw_output = subprocess.check_output(
            [hipcc_bin, "--version"], universal_newlines=True
        )
    except Exception as e:
        print(
            f"hip installation not found: {e} ROCM_PATH={os.environ.get('ROCM_PATH')}"
        )
        return None, None

    for line in raw_output.split("\n"):
        if "HIP version" in line:
            rocm_version = parse(
                line.split()[-1].replace("-", "+")
            )  # local version is not parsed correctly
            return line, rocm_version

    return None, None


def get_torch_hip_version():
    if torch.version.hip:
        return parse(torch.version.hip.split()[-1].replace("-", "+"))
    else:
        return None


def check_if_hip_home_none() -> None:
    if HIP_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so hipcc won't be necessary
    # in that case.
    warnings.warn(
        f"hipcc was not found.  Are you sure your environment has hipcc available?"
    )

cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []

    check_if_hip_home_none()

    rocm_home = os.getenv("ROCM_PATH")
    _, hip_version = get_hip_version(rocm_home)

    if HIP_HOME is not None:
        if hip_version < Version("6.0"):
            raise RuntimeError(
                f"Only ROCm 6.0 and above are supported."
                " Note: make sure HIP has a supported version by running hipcc --version."
            )
    cc_flag.append("-DBUILD_PYTHON_PACKAGE")

    CPP_VERSION=20
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++20"],
        "nvcc": [
            "-O3",
            "-std=c++20",
            f"--offload-arch={os.getenv('HIP_ARCHITECTURES', 'native')}",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-fgpu-flush-denormals-to-zero",
        ]
        + cc_flag,
    }

    ext_modules.append(
        CUDAExtension(
            name="causal_conv1d_cuda",
            sources=[
                "csrc/causal_conv1d.cpp",
                "csrc/causal_conv1d_fwd.cu",
                "csrc/causal_conv1d_bwd.cu",
                "csrc/causal_conv1d_update.cu",
            ],
            extra_compile_args=extra_compile_args,
            include_dirs=[Path(this_dir) / "csrc"],
        )
    )

setup(
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "causal_conv1d.egg-info",
        )
    ),
    url="https://github.com/kw-00/causal-conv1d-modern-rocm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
