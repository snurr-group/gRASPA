#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H
// Single-source CUDA/HIP shim.
//
// Under nvcc/nvc++ (no __HIP__ defined) this header reduces to exactly
// <cuda_runtime.h> + <cuda_fp16.h> and defines NO macros, so the CUDA
// build is byte-for-byte unchanged.
//
// Under hipcc (which defines __HIP__) the CUDA runtime/fp16 surface used
// by gRASPA is mapped onto the corresponding HIP entry points. The mapped
// surface is intentionally small: gRASPA uses no streams, textures, device
// atomics intrinsics or warp-level primitives, and all kernel launches use
// the triple-chevron syntax that hipcc accepts directly.
#if defined(__HIP__)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #define cudaMalloc             hipMalloc
  #define cudaMallocHost         hipHostMalloc   // hipMallocHost is a deprecated alias
  #define cudaMallocManaged      hipMallocManaged
  #define cudaFree               hipFree
  #define cudaMemcpy             hipMemcpy
  #define cudaMemcpyAsync        hipMemcpyAsync
  #define cudaMemset             hipMemset
  #define cudaDeviceSynchronize  hipDeviceSynchronize
  #define cudaGetLastError       hipGetLastError
  #define cudaGetErrorString     hipGetErrorString
  #define cudaError_t            hipError_t
  #define cudaSuccess            hipSuccess
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
#endif // defined(__HIP__)
#endif // GPU_COMPAT_H
