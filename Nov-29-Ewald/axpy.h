#include <complex>
#include "VDW_Coulomb.cuh"
//#include "Pairwise_Energy_Diff.cuh"
#include "GPU_alloc.cuh"
#include "RN.h"
#include <vector>
//#include "read_data.h"
double Run_Simulation(int Cycles, Components& SystemComponents, Boxsize Box, Simulations Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, Units Constants, double init_energy, bool DualPrecision, std::vector<size_t>& NumberOfCreateMolecules, bool CreateMolecules);

double Multiple_Sims(int Cycles, std::vector<Components>& SystemComponents, Boxsize Box, Simulations* Sims, ForceField FF, RandomNumber Random, std::vector<WidomStruct>& WidomArray, Units Constants, std::vector<double>& init_energy);

double CPU_GPU_EwaldTotalEnergy(Boxsize& Box, Boxsize& device_Box, Atoms* System, Atoms* d_a, ForceField FF, ForceField device_FF, Components& SystemComponents);

void Calculate_Exclusion_Energy_Rigid(Boxsize& Box, Atoms* System, ForceField FF, Components& SystemComponents);
