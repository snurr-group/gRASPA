#include "GPU_Reduction.cuh"

size_t* CUDA_copy_allocate_size_t_array(size_t* x, size_t N);
int* CUDA_copy_allocate_int_array(int* x, size_t N);
double* CUDA_copy_allocate_double_array(double* x, size_t N);
double* CUDA_allocate_double_array(size_t N);
void CUDA_copy_double_array(double* x, double **device_x, size_t N);
double* PINNED_CUDA_copy_allocate_double_array(double* x, size_t N);
double* PINNED_CUDA_allocate_double_array(size_t N);

template<typename T>
T* CUDA_allocate_array(size_t N)
{
  T* device_x;
  cudaMalloc(&device_x, N * sizeof(T)); checkCUDAError("Error allocating Malloc");
  return device_x;
}

template<typename T>
T* CUDA_copy_allocate_array(T* x, size_t N)
{
  T* device_x;
  cudaMalloc(&device_x, N * sizeof(T)); checkCUDAError("Error allocating Malloc");
  cudaMemcpy(device_x, x, N * sizeof(T), cudaMemcpyHostToDevice); checkCUDAError("double Error Memcpy");
  return device_x;
}
