#include "data_struct.h"
#include <cuda_fp16.h>  
double Framework_energy_CPU(Boxsize Box, Atoms* Host_System, Atoms* System, ForceField FF, Components SystemComponents);

__device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result);

__device__ void CoulombReal(const ForceField FF, const double chargeA, const double chargeB, const double r, const double scaling, double* result);

__device__ void PBC(double* posvec, double* Cell, double* InverseCell, bool Cubic);

__global__ void one_thread_GPU_test(Boxsize Box, Atoms* d_a, ForceField FF, double* xxx);

__global__ void Collapse_Framework_Energy_OVERLAP_PARTIAL(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* Blocksum, size_t ComponentID, size_t totalAtoms, bool* flag, size_t totalthreads, size_t chainsize, size_t NblockForTrial);


double CPU_EwaldDifference(Boxsize& Box, Atoms& New, Atoms& Old, ForceField& FF, Components& SystemComponents, size_t SelectedComponent, bool Swap, size_t SelectedTrial);

double GPU_EwaldDifference_Reinsertion(Boxsize& Box, Atoms*& d_a, Atoms& Old, double* tempx, double* tempy, double* tempz, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, size_t UpdateLocation);

void Update_Ewald_Vector(Boxsize& Box, bool CPU, Components& SystemComponents);

double GPU_EwaldDifference_General(Boxsize& Box, Atoms*& d_a, Atoms& New, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location);
