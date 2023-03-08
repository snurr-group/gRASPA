#include "VDW_Coulomb.cuh"
#include "GPU_alloc.cuh"
#include "RN.h"
#include <vector>
//#include "read_data.h"
//void cufxn(double* device_x, double* device_y, size_t N);
//double cufxn(double* Cell, double* InverseCell, double* Framework, size_t* FrameworkType, double* FF, int* FFType, double* Mol, double* NewMol, size_t* MolType, double* FFParams, int* OtherParams, size_t Frameworksize, size_t Molsize, size_t FFsize, bool noCharges, double* y, double* dUdlambda, int cycles);
//double cuSoA(Boxsize Box, Atoms Framework, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda);
//double cuSoA(Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda);
//double cuSoA(int Cycles, Components SystemComponents, Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda);
//double cuSoA(int Cycles, Components SystemComponents, Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, Move_Statistics MoveStats);
double cuSoA(int Cycles, Components& SystemComponents, Boxsize Box, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, WidomStruct Widom, Units Constants, double init_energy, bool DualPrecision);

std::vector<double> Multiple_Simulations(int Cycles, std::vector<Components>& SystemComponents, Boxsize Box, Simulations* Sims, Temp_Atoms* TempAtoms, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, std::vector<WidomStruct>& Widom, Units Constants, std::vector<double> System_Energies, bool DualPrecision);
