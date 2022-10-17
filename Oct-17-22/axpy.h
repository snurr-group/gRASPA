#include "VDW_Coulomb.cuh"
#include "GPU_alloc.cuh"
#include "RN.h"
#include <vector>
//#include "read_data.h"
/*size_t* CUDA_copy_allocate_size_t_array(size_t* x, size_t N);
int* CUDA_copy_allocate_int_array(int* x, size_t N);
double* CUDA_copy_allocate_double_array(double* x, size_t N);
double* CUDA_allocate_double_array(size_t N);
void CUDA_copy_double_array(double* x, double **device_x, size_t N);*/
//void cufxn(double* device_x, double* device_y, size_t N);
//double cufxn(double* Cell, double* InverseCell, double* Framework, size_t* FrameworkType, double* FF, int* FFType, double* Mol, double* NewMol, size_t* MolType, double* FFParams, int* OtherParams, size_t Frameworksize, size_t Molsize, size_t FFsize, bool noCharges, double* y, double* dUdlambda, int cycles);
//double cuSoA(Boxsize Box, Atoms Framework, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda);
//double cuSoA(Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda);
//double cuSoA(int Cycles, Components SystemComponents, Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda);
//double cuSoA(int Cycles, Components SystemComponents, Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, Move_Statistics MoveStats);
double cuSoA(int Cycles, Components& SystemComponents, Boxsize Box, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, WidomStruct Widom, Units Constants, double init_energy, bool DualPrecision);
