#include <complex>
#include "VDW_Coulomb.cuh"
//#include "RN.h"
#include <vector>
//#include "read_data.h"
//
double Run_Simulation(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, double init_energy, std::vector<size_t>& NumberOfCreateMolecules, int SimulationMode, bool AlreadyHasFractionalMolecule = false);

double Multiple_Sims(int Cycles, std::vector<Components>& SystemComponents, Boxsize Box, Simulations* Sims, ForceField FF, RandomNumber& Random, std::vector<WidomStruct>& WidomArray, std::vector<double>& init_energy);

/*
void CPU_GPU_EwaldTotalEnergy(Boxsize& Box, Boxsize& device_Box, Atoms* System, Atoms* d_a, ForceField FF, ForceField device_FF, Components& SystemComponents, MoveEnergy& E);

void Calculate_Exclusion_Energy_Rigid(Boxsize& Box, Atoms* System, ForceField FF, Components& SystemComponents);

void Check_WaveVector_CPUGPU(Boxsize& Box, Components& SystemComponents);
*/

void RunMoves(Variables& Vars, size_t box_index, int Cycle);

double CreateMolecule_InOneBox(Variables& Vars, size_t systemId, bool AlreadyHasFractionalMolecule);

void Run_Simulation_MultipleBoxes(Variables& Vars);

void Run_Simulation_ForOneBox(Variables& Vars, size_t box_index);

void Setup_threadblock(size_t arraysize, size_t& Nblock, size_t& Nthread);

MoveEnergy SingleBodyMove(Variables& Vars, size_t systemId);

void SingleBody_Prepare(Variables& Vars, size_t systemId);

MoveEnergy SingleBody_Calculation(Variables& Vars, size_t systemId);

void SingleBody_Acceptance(Variables& Vars, size_t systemId, MoveEnergy& tot);

void InitialMCBeforeMoves(Variables& Vars, size_t systemId);

size_t Determine_Number_Of_Steps(Variables& Vars, size_t systemId, size_t current_cycle);

void Select_Box_Component_Molecule(Variables& Vars, size_t box_index);

void GatherStatisticsDuringSimulation(Variables& Vars, size_t systemId, size_t cycle);

void MCEndOfPhaseSummary(Variables& Vars);

void Widom_Move_FirstBead_PARTIAL(Variables& Vars, size_t systemId, CBMC_Variables& CBMC);
void Widom_Move_Chain_PARTIAL(Variables& Vars, size_t systemId, CBMC_Variables& CBMC);

__global__ void StoreNewLocation_Reinsertion(Atoms Mol, Atoms NewMol, double3* temp, size_t SelectedTrial, size_t Moleculesize);
__global__ void Update_Reinsertion_data(Atoms* d_a, double3* temp, size_t SelectedComponent, size_t UpdateLocation);

double GetPrefactor(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, int MoveType);

//void AcceptInsertion(Variables& Vars, size_t systemId, int MoveType, CBMC_Variables& InsertionVariables);
void AcceptInsertion(Variables& Vars, CBMC_Variables& InsertionVariables, size_t systemId, int MoveType);

void AcceptDeletion(Variables& Vars, size_t systemId, int MoveType);
MoveEnergy Insertion_Body(Variables& Vars, size_t systemId, CBMC_Variables& CBMC);
MoveEnergy Deletion_Body(Variables& Vars, size_t systemId, CBMC_Variables& CBMC);
//struct SingleMove;

struct SingleMove
{
  MoveEnergy energy;
  void Prepare(Variables& Vars, size_t systemId)
  {
    SingleBody_Prepare(Vars, systemId);
  }
  void Calculate(Variables& Vars, size_t systemId)
  {
    energy = SingleBody_Calculation(Vars, systemId);
  }
  void Acceptance(Variables& Vars, size_t systemId)
  {
    SingleBody_Acceptance(Vars, systemId, energy);
  }
};

#include "move_struct.h"
