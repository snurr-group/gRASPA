# CPU-only regression test for the CUDA/HIP shim (src_clean/gpu_compat.h).
#
# The AMD/ROCm backend rests on one invariant: under nvcc (no __HIP__ defined)
# gpu_compat.h must be inert -- it must define ZERO cuda*/hip* macros, so the
# CUDA build is unchanged. Under hipcc (__HIP__ defined) it must map the small
# CUDA runtime surface onto HIP.
#
# This guards that invariant with just a C++ preprocessor (no GPU, no CUDA/ROCm
# toolkit), so it runs in the existing CPU-only CI. Stub headers stand in for the
# real <cuda_runtime.h>/<hip/hip_runtime.h> so the preprocessor can resolve them.

import os
import re
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(REPO, "src_clean", "gpu_compat.h")

# The cuda* names the shim is expected to remap under HIP.
EXPECTED_MACROS = [
    "cudaMalloc", "cudaMallocHost", "cudaMallocManaged", "cudaFree",
    "cudaMemcpy", "cudaMemcpyAsync", "cudaMemset", "cudaDeviceSynchronize",
    "cudaGetLastError", "cudaGetErrorString", "cudaError_t", "cudaSuccess",
    "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost",
]


def _compiler():
    for cc in ("g++", "clang++", "c++", "cpp"):
        path = shutil.which(cc)
        if path:
            return path
    return None


def _preprocess_defines(tmp_path, hip):
    """Preprocess gpu_compat.h and return the set of macro names it defines.

    Stub headers are provided on the include path so the preprocessor can resolve
    the CUDA/HIP runtime includes without the real toolkits installed.
    """
    cc = _compiler()
    if cc is None:
        pytest.skip("no C/C++ preprocessor available")
    stub = tmp_path / "stub"
    (stub / "hip").mkdir(parents=True)
    for name in ("cuda_runtime.h", "cuda_fp16.h"):
        (stub / name).write_text("")
    for name in ("hip_runtime.h", "hip_fp16.h"):
        (stub / "hip" / name).write_text("")

    cmd = [cc, "-E", "-dD", "-I", str(stub)]
    if hip:
        cmd.append("-D__HIP__")
    cmd.append(SHIM)
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, f"preprocessing failed:\n{out.stderr}"

    defines = set()
    for line in out.stdout.splitlines():
        m = re.match(r"\s*#define\s+(\w+)", line)
        if m:
            defines.add(m.group(1))
    return defines


def test_shim_exists():
    assert os.path.exists(SHIM), f"missing shim header: {SHIM}"


def test_shim_inert_under_cuda(tmp_path):
    # Under nvcc (no __HIP__) the shim must define no cuda*/hip* macros at all,
    # so the CUDA translation units are unchanged.
    defines = _preprocess_defines(tmp_path, hip=False)
    leaked = sorted(d for d in defines if d.startswith(("cuda", "hip")))
    assert leaked == [], f"shim leaks macros into the CUDA build: {leaked}"


def test_shim_maps_under_hip(tmp_path):
    # Under hipcc (__HIP__) the shim must remap the CUDA runtime surface to HIP.
    defines = _preprocess_defines(tmp_path, hip=True)
    missing = [m for m in EXPECTED_MACROS if m not in defines]
    assert missing == [], f"shim does not remap under HIP: {missing}"
