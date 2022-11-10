#include <stdio.h>
#include <math.h>
#include <cstdlib>

#include <array>

#include <omp.h>

#include "axpy.h"

#include "read_data.h"
#include "convert_array.h"
#include "write_lmp_movie.h"
#include "fxn_main.h"

bool DualPrecision = false; //Whether or not to use Dual-Precision CBMC//

int main(void)
{
  size_t  Allocate_space_Adsorbate = 0; //Variable for recording allocate_space on the device for adsorbates //
  // SETUP NUMBER OF SIMULATIONS //
  size_t  NumberOfSimulations = 2;
  size_t  SelectedSim = 0; //Zhao's note: Selected simulation for testing //
  bool    RunSingleSim = false;
  read_number_of_sims_from_input(&NumberOfSimulations, &RunSingleSim);
  ////////////////////////////////////////////////////////////
  // DECLARE BASIC VARIABLES, READ FORCEFIELD and FRAMEWORK //
  ////////////////////////////////////////////////////////////
  size_t NumberOfComponents = 2; //0: Framework; 1: adsorbate
  Atoms System[NumberOfComponents];
  Boxsize Box; Boxsize device_Box;
  PseudoAtomDefinitions PseudoAtom;
  ForceField FF; ForceField device_FF;
  ForceFieldParser(FF, PseudoAtom);
  PseudoAtomParser(FF, PseudoAtom);
  read_FFParams_from_input(FF);

  // FORMULATE WIDOM VARIABLE, and STATISTICS FOR EVERY MOVE //
  WidomStruct Widom; int Cycles;
  Move_Statistics MoveStats; 
  Initialize_Move_Statistics(MoveStats);
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &Cycles, &Widom.NumberWidomTrials, &MoveStats.NumberOfBlocks, &Box.Pressure, &Box.Temperature, &DualPrecision, &Allocate_space_Adsorbate);
  POSCARParser(Box, System[0],PseudoAtom);

  // FORMULATE STATISTICS FOR EVERY COMPONENT (ALWAYS STAYS ON THE HOST) //
  Components TempComponents;
  Update_Components_for_framework(NumberOfComponents, TempComponents, System);

  // READ VALUES FOR EVERY ADSORBATE COMPONENT FROM INPUT FILE //
  std::vector<size_t> NumberOfCreateMolecules(TempComponents.Total_Components, 0); 
  for(size_t i = 0; i < TempComponents.Total_Components; i++)
  {
    if(i == 1)
    { //skip reading the first component, which is the framework
      read_component_values_from_simulation_input(TempComponents, MoveStats, i-1, System[1], PseudoAtom, &NumberOfCreateMolecules[i], Allocate_space_Adsorbate); 
    }
    Widom.NumberWidomTrialsOrientations = 20; //Set a number for the number of trial orientation when doing CBMC
    TempComponents.Moves.push_back(MoveStats);
  }

  // SET UP WIDOM INSERTION //
  Units Constants;
  Setup_System_Units_and_Box(Constants, TempComponents, Box, device_Box);

  // COPY ATOM DATA IN THE SIMULATION BOX FROM HOST TO DEVICE //
  //Copy_Atom_data_to_device(NumberOfComponents, device_System, System);

  // UNIFIED MEMORY FOR DIFFERENT SIMULATIONS //
  Simulations *Sims; cudaMallocManaged(&Sims, NumberOfSimulations*sizeof(Simulations));
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    Atoms device_System[NumberOfComponents];
    cudaMalloc(&Sims[i].d_a, sizeof(Atoms)*NumberOfComponents);
    Copy_Atom_data_to_device(NumberOfComponents, device_System, System);
    cudaMemcpy(Sims[i].d_a, device_System, sizeof(Atoms)*NumberOfComponents, cudaMemcpyHostToDevice);
    // SET UP TEMPORARY ARRAYS //
    Setup_Temporary_Atoms_Structure(Sims[i].Old, System);
    Setup_Temporary_Atoms_Structure(Sims[i].New, System);
    cudaMalloc(&Sims[i].deviceOld, sizeof(Atoms)); cudaMalloc(&Sims[i].deviceNew, sizeof(Atoms));
    cudaMemcpy(Sims[i].deviceOld, &Sims[i].Old, sizeof(Atoms), cudaMemcpyHostToDevice);
    cudaMemcpy(Sims[i].deviceNew, &Sims[i].New, sizeof(Atoms), cudaMemcpyHostToDevice);
  }

  // PREPARE VALUES FOR THE WIDOM STRUCT, DECLARE THE RESULT POINTERS IN WIDOM //
  std::vector<WidomStruct> WidomArray;
  std::vector<Components> SystemComponents;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    WidomArray.push_back(Widom);
    SystemComponents.push_back(TempComponents);
    Prepare_Widom(WidomArray[i], Box, Sims[i], SystemComponents[i], System, MoveStats);
  }
  // PREPARE VALUES FOR THE FORCEFIELD STRUCT //
  Prepare_ForceField(FF, device_FF, PseudoAtom, Box);
  
  // SETUP RANDOM SEEDS //
  RandomNumber Random;
  Setup_RandomNumber(Random, 100000);

  // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
  std::vector<std::vector<size_t>> CreateMolWrapper;
  for(size_t i = 0; i < NumberOfSimulations; i++) 
  {
    CreateMolWrapper.push_back(NumberOfCreateMolecules);
    if(RunSingleSim){if(i != SelectedSim) continue;}
    double Prior_sum = Run_Simulation(0, SystemComponents[i], device_Box, Sims[i], device_FF, Random, WidomArray[i], Constants, 0.0, DualPrecision, CreateMolWrapper[i], true);
  }
  // CALCULATE THE INITIAL ENERGY //
  double sys_energy;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(RunSingleSim){if(i != SelectedSim) continue;}
    sys_energy = Check_Simulation_Energy(Box, device_Box, System, Sims[i].d_a, FF, device_FF, SystemComponents[i], true, i);
  }
  ////////////////
  // RUN CYCLES //
  ////////////////

  double sum[NumberOfSimulations];
  double start = omp_get_wtime(); 
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(RunSingleSim){if(i != SelectedSim) continue;}
    sum[i] = Run_Simulation(Cycles, SystemComponents[i], device_Box, Sims[i], device_FF, Random, WidomArray[i], Constants, sys_energy, DualPrecision, CreateMolWrapper[i], false);
  }
  double end = omp_get_wtime();
  
  // CALCULATE THE FINAL ENERGY //
  double end_sys_energy;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  { 
    if(RunSingleSim){if(i != SelectedSim) continue;}
    end_sys_energy = Check_Simulation_Energy(Box, device_Box, System, Sims[i].d_a, FF, device_FF, SystemComponents[i], false, i);
    printf("Sim: %zu, Difference Energy: %.10f, running_difference: %.10f\n", i, end_sys_energy - sys_energy, sum[i]);
  }
  // PRINT FINAL ENERGIES AND TIME //
  printf("Work took %f seconds\n", end - start);
  //////////////////////
  // PRINT MOVIE FILE //
  //////////////////////
  create_movie_file(0, System, SystemComponents[0], FF, Box, PseudoAtom.Name);
  return 0;
}
