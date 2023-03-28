#include "data_struct.h"
#include <cuda_fp16.h> 
double Framework_energy_CPU(Boxsize Box, Atoms* Host_System, Atoms* System, ForceField FF, Components SystemComponents);

double2 setScale(double lambda);

void setScaleGPU(double lambda, double& scalingVDW, double& scalingCoulomb);

__device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result);

__device__ void CoulombReal(const ForceField FF, const double chargeA, const double chargeB, const double r, const double scaling, double* result, double prefactor, double alpha);

__device__ void PBC(double* posvec, double* Cell, double* InverseCell, bool Cubic);

__global__ void one_thread_GPU_test(Boxsize Box, Atoms* d_a, ForceField FF, double* xxx);

__global__ void Collapse_Framework_Energy_OVERLAP_PARTIAL(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* Blocksum, size_t ComponentID, size_t totalAtoms, bool* flag, size_t totalthreads, size_t chainsize, size_t NblockForTrial);

__global__ void Energy_difference_LambdaChange(Boxsize Box, Atoms* System, Atoms Mol, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, size_t Threadsize, bool* flag, double2 newScale);

double CPU_EwaldDifference(Boxsize& Box, Atoms& New, Atoms& Old, ForceField& FF, Components& SystemComponents, size_t SelectedComponent, bool Swap, size_t SelectedTrial);

double GPU_EwaldDifference_Reinsertion(Boxsize& Box, Atoms*& d_a, Atoms& Old, double* tempx, double* tempy, double* tempz, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, size_t UpdateLocation);

void Update_Ewald_Vector(Boxsize& Box, bool CPU, Components& SystemComponents);

double GPU_EwaldDifference_General(Boxsize& Box, Atoms*& d_a, Atoms& New, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location, double2 proposed_scale);

double GPU_EwaldDifference_LambdaChange(Boxsize& Box, Atoms*& d_a, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, double2 oldScale, double2 newScale, int MoveType);

void Skip_Ewald(Boxsize& Box);

__global__ void TotalVDWCoul(Boxsize Box, Atoms* System, ForceField FF, double* Blocksum, bool* flag, size_t totalthreads, size_t Host_threads, size_t NAds, size_t NFrameworkAtomsPerThread, bool HostHost, bool UseOffset);

void Setup_threadblock_EW(size_t arraysize, size_t *Nblock, size_t *Nthread);

__global__ void Double3_CacheCheck(double* Array, Complex* Vector, size_t totsize);

__global__ void Setup_Ewald_Vector(Boxsize Box, Complex* eik_x, Complex* eik_y, Complex* eik_z, Atoms* System, size_t numberOfAtoms, size_t NumberOfComponents, bool UseOffSet);

__global__ void TotalEwald(Atoms* d_a, Boxsize Box, double* BlockSum, Complex* eik_x, Complex* eik_y, Complex* eik_z, Complex* Eik, size_t totAtom, size_t Ncomp, size_t NAtomPerThread, size_t residueAtoms);

double Ewald_TotalEnergy(Simulations& Sim, Components& SystemComponents, bool UseOffSet);

double Total_VDW_Coulomb_Energy(Simulations& Sim, ForceField FF, size_t totMol, size_t Host_threads, size_t Guest_threads, size_t NFrameworkAtomsPerThread, bool ConsiderHostHost, bool UseOffset);

double CoulombRealCorrection_General(Boxsize& Box, Atoms*& d_a, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType);

double CoulombRealCorrection_Reinsertion(Boxsize& Box, Atoms*& d_a, Atoms& Old, double* tempx, double* tempy, double* tempz, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent);

double TotalTailCorrection(Components& SystemComponents, size_t FFsize, double Volume);

double TailCorrectionDifference(Components& SystemComponents, size_t SelectedComponent, size_t FFsize, double Volume, int MoveType);
