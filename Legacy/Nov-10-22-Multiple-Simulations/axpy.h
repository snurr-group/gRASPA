#include "VDW_Coulomb.cuh"
//#include "Pairwise_Energy_Diff.cuh"
#include "GPU_alloc.cuh"
#include "RN.h"
#include <vector>
//#include "read_data.h"
double Run_Simulation(int Cycles, Components& SystemComponents, Boxsize Box, Simulations Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, Units Constants, double init_energy, bool DualPrecision, std::vector<size_t>& NumberOfCreateMolecules, bool CreateMolecules);
