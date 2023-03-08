#include "GPU_alloc.h"
#include <stdio.h> 
///////////////////////////
// for type double array //
///////////////////////////
void gpu_alloc_double(double* v1, size_t N)
{
    #pragma omp target enter data map(alloc:v1[0:N])
};

void gpu_copy_double(double* v1, size_t N)
{
    #pragma omp target update to(v1[:N] )
};

void gpu_free_double(double* v1, size_t N)
{
    #pragma omp target exit data map(delete:v1[0:N])
};
//////////////////////////////
// for type of size_t array //
//////////////////////////////
void gpu_alloc_size_t(size_t* v1, size_t N)
{
    #pragma omp target enter data map(alloc:v1[0:N])
};

void gpu_copy_size_t(size_t* v1, size_t N)
{
    #pragma omp target update to(v1[:N] )
};

void gpu_free_size_t(size_t* v1, size_t N)
{
    #pragma omp target exit data map(delete:v1[0:N])
};
////////////////////////
// for type int array //
////////////////////////
void gpu_alloc_int(int* v1, size_t N)
{
    #pragma omp target enter data map(alloc:v1[0:N])
};

void gpu_copy_int(int* v1, size_t N)
{
    #pragma omp target update to(v1[:N] )
};

void gpu_free_int(int* v1, size_t N)
{
    #pragma omp target exit data map(delete:v1[0:N])
};

///////////////////////////
// for type double value //
///////////////////////////
void gpu_alloc_single_double(double v1)
{
    #pragma omp target enter data map(alloc:v1)
};

void gpu_copy_single_double(double v1)
{
    #pragma omp target update to(v1)
};

void gpu_free_single_double(double v1)
{
    #pragma omp target exit data map(delete:v1)
};

///////////////////////////
// for type size_t value //
///////////////////////////
void gpu_alloc_single_size_t(size_t v1)
{
    #pragma omp target enter data map(alloc:v1)
};

void gpu_copy_single_size_t(size_t v1)
{
    #pragma omp target update to(v1)
};

void gpu_free_single_size_t(size_t v1)
{
    #pragma omp target exit data map(delete:v1)
};

////////////////////////
// for type int value //
////////////////////////
void gpu_alloc_single_int(int v1)
{
    #pragma omp target enter data map(alloc:v1)
};

void gpu_copy_single_int(int v1)
{
    #pragma omp target update to(v1)
};

void gpu_free_single_int(int v1)
{
    #pragma omp target exit data map(delete:v1)
};

/////////////////////////
// for type bool value //
/////////////////////////
void gpu_alloc_single_bool(bool v1)
{
    #pragma omp target enter data map(alloc:v1)
};

void gpu_copy_single_bool(bool v1)
{
    #pragma omp target update to(v1)
};

void gpu_free_single_bool(bool v1)
{
    #pragma omp target exit data map(delete:v1)
};
