#include "GPU_alloc.cuh"
size_t* CUDA_copy_allocate_size_t_array(size_t* x, size_t N)
{
  size_t* device_x;
  cudaMalloc(&device_x, N * sizeof(size_t)); checkCUDAError("Error allocating Malloc");
  cudaMemcpy(device_x, x, N * sizeof(size_t), cudaMemcpyHostToDevice); checkCUDAError("size_t Error Memcpy");
  return device_x;
}

int* CUDA_copy_allocate_int_array(int* x, size_t N)
{
  int* device_x;
  cudaMalloc(&device_x, N * sizeof(int)); checkCUDAError("Error allocating Malloc");
  cudaMemcpy(device_x, x, N * sizeof(int), cudaMemcpyHostToDevice); checkCUDAError("int Error Memcpy");
  return device_x;
}

double* CUDA_copy_allocate_double_array(double* x, size_t N)
{
  double* device_x;
  cudaMalloc(&device_x, N * sizeof(double)); checkCUDAError("Error allocating Malloc");
  cudaMemcpy(device_x, x, N * sizeof(double), cudaMemcpyHostToDevice); checkCUDAError("double Error Memcpy");
  return device_x;
}
double* CUDA_allocate_double_array(size_t N)
{
  double* device_x;
  cudaMalloc(&device_x, N * sizeof(double)); checkCUDAError("Error allocating Malloc");
  return device_x;
}

void CUDA_copy_double_array(double* x, double **device_x, size_t N)
{
  cudaMemcpy(*device_x, x, N * sizeof(double), cudaMemcpyHostToDevice); checkCUDAError("Just double Error Memcpy");
}

