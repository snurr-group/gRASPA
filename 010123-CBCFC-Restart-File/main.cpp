#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <array>
#include <omp.h>

#include <complex>

#include "axpy.h"
#include "read_data.h"
#include "convert_array.h"
#include "write_data.h"
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

  // FORMULATE WIDOM VARIABLE, and STATISTICS FOR EVERY MOVE //
  WidomStruct Widom; 
  int NumberOfInitializationCycles;
  int NumberOfEquilibrationCycles;
  int NumberOfProductionCycles;
  Move_Statistics MoveStats; 
  Initialize_Move_Statistics(MoveStats);
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &NumberOfInitializationCycles, &NumberOfEquilibrationCycles, &NumberOfProductionCycles, &Widom.NumberWidomTrials, &Widom.NumberWidomTrialsOrientations, &MoveStats.NumberOfBlocks, &Box.Pressure, &Box.Temperature, &DualPrecision, &Allocate_space_Adsorbate);
  POSCARParser(Box, System[0],PseudoAtom);

  read_FFParams_from_input(FF, Box);
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
    TempComponents.Moves.push_back(MoveStats);
  }

  // SET UP WIDOM INSERTION //
  Units Constants;
  Setup_System_Units_and_Box(Constants, TempComponents, Box, device_Box);

  // COPY ATOM DATA IN THE SIMULATION BOX FROM HOST TO DEVICE //

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

  //Setup Initial Ewald//
  //size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  //std::vector<std::complex<double>> storedEikBefore(numberOfWaveVectors);
  double InitialEwaldE   = 0.0;
  double FrameworkEwaldE = 0.0;
  //Calculate & Initialize Ewald JUST for the FRAMEWORK//
  if(!FF.noCharges)
  {
    FrameworkEwaldE = CPU_GPU_EwaldTotalEnergy(Box, device_Box, System, Sims[0].d_a, FF, device_FF, SystemComponents[0]);
    Calculate_Exclusion_Energy_Rigid(Box, System, FF, SystemComponents[0]);
  }
  // Allocate/Copy the Ewald Vectors to the GPU //
  Allocate_Copy_Ewald_Vector(device_Box, SystemComponents[0], device_FF);

  // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
  std::vector<std::vector<size_t>> CreateMolWrapper;
  std::vector<double> sys_energy(NumberOfSimulations, 0.0);
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    CreateMolWrapper.push_back(NumberOfCreateMolecules);
    if(RunSingleSim){if(i != SelectedSim) continue;}
    cudaDeviceSynchronize();
    double start = omp_get_wtime();
    
    double Prior_sum = Run_Simulation(0, SystemComponents[i], device_Box, Sims[i], device_FF, Random, WidomArray[i], Constants, sys_energy[i], DualPrecision, CreateMolWrapper[i], CREATE_MOLECULE);
    
    cudaDeviceSynchronize();
    double end = omp_get_wtime();
    printf("Creating Molecules took %.12f secs.\n", end - start);
  }
  // CALCULATE THE INITIAL ENERGY //
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(RunSingleSim){if(i != SelectedSim) continue;}
    sys_energy[i] = Check_Simulation_Energy(Box, device_Box, System, Sims[i].d_a, FF, device_FF, SystemComponents[i], true, i);
  }

  // Calculate Ewald Parameters after creating molecule(s) //
  if(!FF.noCharges)
  {
    InitialEwaldE  = CPU_GPU_EwaldTotalEnergy(Box, device_Box, System, Sims[0].d_a, FF, device_FF, SystemComponents[0]);
    InitialEwaldE -= FrameworkEwaldE;
  }
  printf("After Creating Molecules, Framework Ewald: %.5f, Initial Ewald: %.5f\n", FrameworkEwaldE, InitialEwaldE);
  ////////////////
  // RUN CYCLES //
  ////////////////

  double sum[NumberOfSimulations];
  double start = omp_get_wtime(); 
  ///////////////////////////
  // INITIALIZATION CYCLES //
  ///////////////////////////
  if(!RunSingleSim)
  {
    Multiple_Sims(NumberOfInitializationCycles, SystemComponents, device_Box, Sims, device_FF, Random, WidomArray, Constants, sys_energy);
  }
  else
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i != SelectedSim) continue;
      sum[i] = Run_Simulation(NumberOfInitializationCycles, SystemComponents[i], device_Box, Sims[i], device_FF, Random, WidomArray[i], Constants, sys_energy[i], DualPrecision, CreateMolWrapper[i], INITIALIZATION);
    }
  }
  
  //////////////////////////
  // EQUILIBRATION CYCLES //
  //////////////////////////
  if(!RunSingleSim)
  {
    Multiple_Sims(NumberOfEquilibrationCycles, SystemComponents, device_Box, Sims, device_FF, Random, WidomArray, Constants, sys_energy);
  }
  else
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i != SelectedSim) continue;
      sum[i] += Run_Simulation(NumberOfEquilibrationCycles, SystemComponents[i], device_Box, Sims[i], device_FF, Random, WidomArray[i], Constants, sys_energy[i], DualPrecision, CreateMolWrapper[i], EQUILIBRATION);
    }
  }

  ///////////////////////
  // PRODUCTION CYCLES //
  ///////////////////////
  if(!RunSingleSim)
  {
    Multiple_Sims(NumberOfProductionCycles, SystemComponents, device_Box, Sims, device_FF, Random, WidomArray, Constants, sys_energy);
  }
  else
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i != SelectedSim) continue;
      sum[i] += Run_Simulation(NumberOfProductionCycles, SystemComponents[i], device_Box, Sims[i], device_FF, Random, WidomArray[i], Constants, sys_energy[i], DualPrecision, CreateMolWrapper[i], PRODUCTION);
    }
  }

  double end = omp_get_wtime();
  
  // CALCULATE THE FINAL ENERGY //
  std::vector<double> end_sys_energy(NumberOfSimulations);
  for(size_t i = 0; i < NumberOfSimulations; i++)
  { 
    if(RunSingleSim){if(i != SelectedSim) continue;}
    end_sys_energy[i] = Check_Simulation_Energy(Box, device_Box, System, Sims[i].d_a, FF, device_FF, SystemComponents[i], false, i);
    printf("Sim: %zu, Difference Energy: %.10f, running_difference: %.10f\n", i, end_sys_energy[i] - sys_energy[i], sum[i]);
  }
  // PRINT FINAL ENERGIES AND TIME //
  printf("Work took %f seconds\n", end - start);

  // Calculate Ewald Parameters after creating molecule(s) //
  double FinalEwaldE = 0.0;
  if(!FF.noCharges)
  {
    FinalEwaldE  = CPU_GPU_EwaldTotalEnergy(Box, device_Box, System, Sims[0].d_a, FF, device_FF, SystemComponents[0]);
    FinalEwaldE -= FrameworkEwaldE;
    printf("After Simulations, Framework Ewald: %.5f, Final Ewald: %.5f, DIFF Ewald: %.5f\n", FrameworkEwaldE, FinalEwaldE, FinalEwaldE - InitialEwaldE);
    Check_WaveVector_CPUGPU(device_Box, SystemComponents[0]); //Check WaveVector on the CPU and GPU//
  }
  //Check if the Ewald Diff and running Diff make sense//
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(RunSingleSim){if(i != SelectedSim) continue;}
    double diff = sum[i] - (FinalEwaldE - InitialEwaldE) - end_sys_energy[i] + sys_energy[i]; 
    printf("Final Energy (Running Energy from Simulation): %.5f\n", sum[i] + InitialEwaldE + sys_energy[i]);
    printf("Final Energy (Recalculated by Energy Check)  : %.5f\n", FinalEwaldE + end_sys_energy[i]);
    printf("Drift in Energy: %.5f\n", diff);
  }
  //////////////////////
  // PRINT MOVIE FILE //
  //////////////////////
  create_movie_file(0, System, SystemComponents[0], FF, Box, PseudoAtom.Name);
  create_Restart_file(0, System, SystemComponents[0], FF, Box, PseudoAtom.Name, Sims[0].MaxTranslation, Sims[0].MaxRotation);
  Write_All_Adsorbate_data(0, System, SystemComponents[0], FF, Box, PseudoAtom.Name);
  Write_Lambda(0, SystemComponents[0]);
  return 0;
}
