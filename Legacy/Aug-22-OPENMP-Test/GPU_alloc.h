#include <stdio.h>
///////////////////////////
// for type double array //
///////////////////////////
void gpu_alloc_double(double*, size_t );
void gpu_copy_double(double* , size_t );
void gpu_free_double(double* , size_t );

///////////////////////////
// for type size_t array //
///////////////////////////
void gpu_alloc_size_t(size_t*, size_t );
void gpu_copy_size_t(size_t* , size_t );
void gpu_free_size_t(size_t* , size_t );

////////////////////////
// for type int array //
////////////////////////
void gpu_alloc_int(int*, size_t );
void gpu_copy_int(int* , size_t );
void gpu_free_int(int* , size_t );

///////////////////////////
// for type double value //
///////////////////////////
void gpu_alloc_single_double(double);
void gpu_copy_single_double(double);
void gpu_free_single_double(double);

///////////////////////////
// for type size_t value //
///////////////////////////
void gpu_alloc_single_size_t(size_t);
void gpu_copy_single_size_t(size_t);
void gpu_free_single_size_t(size_t);

////////////////////////
// for type int value //
////////////////////////
void gpu_alloc_single_int(int);
void gpu_copy_single_int(int);
void gpu_free_single_int(int);

/////////////////////////
// for type bool value //
/////////////////////////
void gpu_alloc_single_bool(bool);
void gpu_copy_single_bool(bool);
void gpu_free_single_bool(bool);
