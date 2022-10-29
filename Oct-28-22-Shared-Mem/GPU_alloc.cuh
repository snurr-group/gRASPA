#include "GPU_Reduction.cuh"

size_t* CUDA_copy_allocate_size_t_array(size_t* x, size_t N);
int* CUDA_copy_allocate_int_array(int* x, size_t N);
double* CUDA_copy_allocate_double_array(double* x, size_t N);
double* CUDA_allocate_double_array(size_t N);
void CUDA_copy_double_array(double* x, double **device_x, size_t N);

float* CUDA_copy_allocate_downcast_to_float(double* x, size_t N);
