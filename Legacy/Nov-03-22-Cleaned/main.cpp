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

bool read_from_restart = false; //Zhao's note: The system can be read from def files or from a Restart file in the Restart folder, probably glitchy//
int main(void)
{
  if(read_from_restart){ printf("I am not checking!"); }//check_restart_file();  //Zhao's note: weird function, cannot remove it due to glitches//
  ////////////////////////////////////////////////////////////
  // DECLARE BASIC VARIABLES, READ FORCEFIELD and FRAMEWORK //
  ////////////////////////////////////////////////////////////
  size_t NumberOfComponents = 2; //0: Framework; 1: adsorbate
  Atoms System[NumberOfComponents];
  Atoms device_System[NumberOfComponents];
  Boxsize Box; Boxsize device_Box;
  PseudoAtomDefinitions PseudoAtom;
  ForceField FF; ForceField device_FF;
  ForceFieldParser(FF, PseudoAtom);
  PseudoAtomParser(FF, PseudoAtom);
  FF.FFParams = read_FFParams_from_restart(); //Zhao's note: for some reason, if I remove this line (or use an equivalent function **read_FFParams_from_input()**, the code is 20% slower //
  //FF.FFParams = read_FFParams_from_input();
  std::string FrameworkName; //Zhao's note: to some unknown glitch, this must remain, although it is never called... Dangerous to use std::string//

  // FORMULATE WIDOM VARIABLE, and STATISTICS FOR EVERY MOVE //
  WidomStruct Widom; int Cycles;
  Move_Statistics MoveStats; 
  Initialize_Move_Statistics(MoveStats);
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &Cycles, &Widom.NumberWidomTrials, &MoveStats.NumberOfBlocks, &Box.Pressure, &Box.Temperature, &DualPrecision);
  POSCARParser(Box, System[0],PseudoAtom);
  System[0].Molsize = System[0].size;

  // FORMULATE STATISTICS FOR EVERY COMPONENT (ALWAYS STAYS ON THE HOST) //
  Components SystemComponents;
  Update_Components_for_framework(NumberOfComponents, SystemComponents, System);

  // READ VALUES FOR EVERY ADSORBATE COMPONENT FROM INPUT FILE //
  std::vector<size_t> NumberOfCreateMolecules(SystemComponents.Total_Components, 0); 
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    if(i == 1){ //skip reading the first component, which is the framework
      read_component_values_from_simulation_input(SystemComponents, MoveStats, i-1, System[1], PseudoAtom, &NumberOfCreateMolecules[i]); System[1].Molsize = SystemComponents.Moleculesize[1];}
    Widom.NumberWidomTrialsOrientations = 20; //Set a number for the number of trial orientation when doing CBMC
    SystemComponents.Moves.push_back(MoveStats);
  }

  // SET UP WIDOM INSERTION //
  Units Constants;
  Setup_System_Units_and_Box(Constants, SystemComponents, Box, device_Box);

  // SET UP TEMPORARY ARRAYS //
  Atoms Mol; Atoms NewMol;
  Setup_Temporary_Atoms_Structure(Mol, System);
  Setup_Temporary_Atoms_Structure(NewMol, System);

  // COPY ATOM DATA IN THE SIMULATION BOX FROM HOST TO DEVICE //
  Copy_Atom_data_to_device(NumberOfComponents, device_System, System);
  Atoms *d_a; // Zhao's note: pointer to device_System, accessible only on the device //
  cudaMalloc(&d_a, sizeof(Atoms)*NumberOfComponents);
  cudaMemcpy(d_a, device_System, sizeof(Atoms)*NumberOfComponents, cudaMemcpyHostToDevice);

  // PREPARE VALUES FOR THE WIDOM STRUCT, DECLARE THE RESULT POINTERS IN WIDOM //
  Prepare_Widom(Widom, SystemComponents, System, MoveStats);
  
  // PREPARE VALUES FOR THE FORCEFIELD STRUCT //
  Prepare_ForceField(FF, device_FF, PseudoAtom, Box);
  
  // SETUP RANDOM SEEDS //
  RandomNumber Random;
  Setup_RandomNumber(Random, 100000);

  // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
  double Prior_sum = Run_Simulation(0, SystemComponents, device_Box, device_System, d_a, Mol, NewMol, device_FF, Random, Widom, Constants, 0.0, DualPrecision, NumberOfCreateMolecules, true);
 
  // CALCULATE THE INITIAL ENERGY //
  double sys_energy = Check_Simulation_Energy(Box, device_Box, System, device_System, d_a, FF, device_FF, SystemComponents, true);
  
  ////////////////
  // RUN CYCLES //
  ////////////////
  double start = omp_get_wtime(); 
  double sum = Run_Simulation(Cycles, SystemComponents, device_Box, device_System, d_a, Mol, NewMol, device_FF, Random, Widom, Constants, sys_energy, DualPrecision, NumberOfCreateMolecules, false);
  double end = omp_get_wtime();
  
  // CALCULATE THE FINAL ENERGY //
  double end_sys_energy = Check_Simulation_Energy(Box, device_Box, System, device_System, d_a, FF, device_FF, SystemComponents, false);
  
  // PRINT FINAL ENERGIES AND TIME //
  printf("Difference Energy: %.10f, running_difference: %.10f\n", end_sys_energy - sys_energy, sum);
  printf("Work took %f seconds\n", end - start);
  //////////////////////
  // PRINT MOVIE FILE //
  //////////////////////
  create_movie_file(0, System, SystemComponents, FF, Box, PseudoAtom.Name);
  return 0;
}
