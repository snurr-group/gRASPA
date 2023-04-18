#include <complex>
#include "VDW_Coulomb.cuh"
#include "GPU_alloc.cuh"
//#include "RN.h"
#include <vector>
//#include "read_data.h"
double Run_Simulation(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, double init_energy, std::vector<size_t>& NumberOfCreateMolecules, int SimulationMode, bool AlreadyHasFractionalMolecule = false);

double Multiple_Sims(int Cycles, std::vector<Components>& SystemComponents, Boxsize Box, Simulations* Sims, ForceField FF, RandomNumber& Random, std::vector<WidomStruct>& WidomArray, std::vector<double>& init_energy);

double CPU_GPU_EwaldTotalEnergy(Boxsize& Box, Boxsize& device_Box, Atoms* System, Atoms* d_a, ForceField FF, ForceField device_FF, Components& SystemComponents);

void Calculate_Exclusion_Energy_Rigid(Boxsize& Box, Atoms* System, ForceField FF, Components& SystemComponents);

void Check_WaveVector_CPUGPU(Boxsize& Box, Components& SystemComponents);

double CreateMolecule_InOneBox(Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, bool AlreadyHasFractionalMolecule);

void Run_Simulation_MultipleBoxes(int Cycles, std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, RandomNumber& Random, std::vector<WidomStruct>& Widom, std::vector<SystemEnergies>& Energy, Gibbs& GibbsStatistics, int SimulationMode, bool SetMaxStep, size_t MaxStepPerCycle);

double Run_Simulation_ForOneBox(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, double init_energy, int SimulationMode, bool SetMaxStep, size_t MaxStepPerCycle, Units Constants);

void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread);
