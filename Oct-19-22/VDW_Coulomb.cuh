#include "data_struct.h"
#include <cuda_fp16.h>  
double Framework_energy_CPU(Boxsize Box, Atoms* Host_System, Atoms* System, ForceField FF, Components SystemComponents);

__device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result);

__device__ void CoulombReal(double* FFParams, const double chargeA, const double chargeB, const double r, const double scaling, double* result);

__device__ void PBC(double* posvec, double* Cell, double* InverseCell, int* OtherParams);

__global__ void Framework_energy_difference_SoA(Boxsize Box, Atoms* System, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, size_t ComponentID, size_t totalthreads, size_t Molsize);

__global__ void Collapse_Framework_Energy(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* y, size_t ComponentID, size_t totalAtoms, size_t totalthreads, size_t trialsize);

__global__ void Collapse_Framework_Energy_OVERLAP(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* y, size_t ComponentID, size_t totalAtoms, size_t* flag, size_t totalthreads, size_t trialsize);

__global__ void Collapse_Framework_Energy_OVERLAP_FLOAT(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* y, size_t ComponentID, size_t totalAtoms, size_t* flag, size_t totalthreads, size_t trialsize, float* y_float);

__global__ void Collapse_Framework_Energy_OVERLAP_HALF(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* y, size_t ComponentID, size_t totalAtoms, size_t* flag, size_t totalthreads, size_t trialsize, half* y_half);

__global__ void one_thread_GPU_test(Boxsize Box, Atoms* d_a, ForceField FF, double* xxx);
