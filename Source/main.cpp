#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <array>
#include <omp.h>

#include <complex>

#include "axpy.h"
#include "read_data.h"
#include "write_data.h"
#include "fxn_main.h"
int main(void)
{
  size_t  Allocate_space_Adsorbate = 0; //Variable for recording allocate_space on the device for adsorbates //

  bool RunOneByOne = false;
  bool RunTogether = true;

  // SETUP NUMBER OF SIMULATIONS //
  size_t  NumberOfSimulations = 2;
  size_t  SelectedSim = 0; //Zhao's note: Selected simulation for testing //
  bool    RunSingleSim = false;
  bool    ReadRestart = false; //Whether we read data from restart file or not//
  read_number_of_sims_from_input(&NumberOfSimulations, &RunSingleSim);
  if(RunSingleSim)
  {
    RunOneByOne = true;
    RunTogether = false;
  }
  // UNIFIED MEMORY FOR DIFFERENT SIMULATIONS //
  Simulations *Sims; cudaMallocManaged(&Sims, NumberOfSimulations*sizeof(Simulations));
  ////////////////////////////////////////////////////////////
  // DECLARE BASIC VARIABLES, READ FORCEFIELD and FRAMEWORK //
  ////////////////////////////////////////////////////////////
  size_t NumberOfComponents = 1; //0: Framework;
  //std::vector<Atoms*> HostSystem(NumberOfSimulations);
  std::vector<Boxsize> Box(NumberOfSimulations); //(Boxsize*) malloc(NumberOfSimulations * sizeof(Boxsize)); //Boxsize device_Box;
  PseudoAtomDefinitions PseudoAtom;
  ForceField FF; ForceField device_FF;
  ForceFieldParser(FF, PseudoAtom);
  PseudoAtomParser(FF, PseudoAtom);

  double EwaldPrecision = 1e-6;
  read_FFParams_from_input(FF, EwaldPrecision);

  // FORMULATE WIDOM VARIABLE, and STATISTICS FOR EVERY MOVE //
  WidomStruct Widom; 
  int NumberOfInitializationCycles;
  int NumberOfEquilibrationCycles;
  int NumberOfProductionCycles;
  int RANDOMSEED;
  Move_Statistics MoveStats; 
  Initialize_Move_Statistics(MoveStats);
  
  bool SameFrameworkEverySimulation = true; //Use the same framework (box) setup for every simulation?//

  double PRESSURE = 0.0; double TEMPERATURE = 0.0; 

  printf("------------------GENERAL SIMULATION SETUP-------------\n");
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &NumberOfInitializationCycles, &NumberOfEquilibrationCycles, &NumberOfProductionCycles, &Widom.NumberWidomTrials, &Widom.NumberWidomTrialsOrientations, &MoveStats.NumberOfBlocks, &PRESSURE, &TEMPERATURE, &Allocate_space_Adsorbate, &ReadRestart, &RANDOMSEED, &SameFrameworkEverySimulation, NumberOfComponents);
  
  Gibbs  GibbsStatistics;
  bool   SetMaxStep = false;
  size_t MaxStepPerCycle = 1;
  read_Gibbs_Stats(GibbsStatistics, SetMaxStep, MaxStepPerCycle);
  printf("-------------------------------------------------------\n");
  // PREPARE VALUES FOR THE FORCEFIELD STRUCT //
  Prepare_ForceField(FF, device_FF, PseudoAtom);

  ////////////////////////
  // SETUP RANDOM SEEDS //
  ////////////////////////
  RandomNumber Random;
  std::srand(RANDOMSEED);
  Setup_RandomNumber(Random, 333334);

  //Physical constants for the simulation//
  Units Constants;
  //if true, then we can simulate the same framework at different T/P//
  //If false, then we can do Gibbs (where a box is empty)//
  std::vector<Components> SystemComponents;
  std::vector<WidomStruct> WidomArray;

  bool samePressure = true; bool sameTemperature = true;

  //Ewald Energy Arrays//
  //Setup Initial Ewald//
  std::vector<SystemEnergies>Energy(NumberOfSimulations);

  for(size_t a = 0; a < NumberOfSimulations; a++)
  {
    //Zhao's note: Hard-coded flag, put it in read_data.cpp later//
    Box[a].ExcludeHostGuestEwald = true;
    if(samePressure)    Box[a].Pressure = PRESSURE;
    //if(sameTemperature) 
    if(!samePressure || !sameTemperature) throw std::runtime_error("Currently not allowing the different Pressure/Temperature!");
    printf("==========================================\n");
    printf("====== Preparing Simulation box %zu ======\n", a);
    printf("==========================================\n");
    //Allocate data on the host for each simulation//
    Components TempComponents; //Temporary component variable//
    TempComponents.HostSystem  = (Atoms*) malloc(NumberOfComponents * sizeof(Atoms));
    TempComponents.ReadRestart = ReadRestart;
    TempComponents.Temperature = TEMPERATURE; 
    /////////////////////////////////////
    // Read and process framework data //
    /////////////////////////////////////
    if(!SameFrameworkEverySimulation || a == 0)
    {
      TempComponents.NumberOfPseudoAtoms.resize(PseudoAtom.Name.size());
      std::fill(TempComponents.NumberOfPseudoAtoms.begin(), TempComponents.NumberOfPseudoAtoms.end(), 0);
      OverWriteFFTerms(TempComponents, FF, PseudoAtom);
      if(a > 0 && !SameFrameworkEverySimulation) printf("Processing %zu, new framework\n", a);
      //Read framework data from cif/poscar file//
      ReadFramework(Box[a], TempComponents.HostSystem[0], PseudoAtom, a, TempComponents);
      read_Ewald_Parameters_from_input(sqrt(FF.CutOffCoul), Box[a], EwaldPrecision);
      Update_Components_for_framework(NumberOfComponents, TempComponents, TempComponents.HostSystem);

      //Initialize Energy Averages//
      double2 tempdou ={0.0, 0.0};
      RosenbluthWeight tempRosen;
      for(size_t i = 0; i < TempComponents.Nblock; i++)
      {  
        TempComponents.EnergyAverage.push_back(tempdou);
        MoveStats.MolAverage.push_back(tempdou);
        MoveStats.Rosen.push_back(tempRosen);
      }
    }
    /////////////////////////////////////
    // Read and process adsorbate data //
    /////////////////////////////////////
    //Zhao's note: different systems will share the SAME components (for adsorbate), only read it once//
    //if(!SameFrameworkEverySimulation || a == 0) //Read adsorbate information//
    //{
      for(size_t comp = 0; comp < TempComponents.Total_Components; comp++)
      {
        if(comp > 0) //0: framework//
        {
          //skip reading the first component, which is the framework
          read_component_values_from_simulation_input(TempComponents, MoveStats, comp-1, TempComponents.HostSystem[comp], PseudoAtom, Allocate_space_Adsorbate);
        }
        TempComponents.Moves.push_back(MoveStats);
      }
      //Int3 for the number of components, consider replace Total_Components with this int3 variable//
      TempComponents.NComponents.x = TempComponents.Total_Components;
      TempComponents.NComponents.y = 1;
      TempComponents.NComponents.z = TempComponents.Total_Components - TempComponents.NComponents.y;

      SystemComponents.push_back(TempComponents);
    //}
    Setup_Box_Temperature_Pressure(Constants, SystemComponents[a], Box[a]);
    Sims[a].Box.Pressure = Box[a].Pressure; Sims[a].Box.Volume = Box[a].Volume;
    Sims[a].Box.Cubic    = Box[a].Cubic;    Sims[a].Box.ReciprocalCutOff = Box[a].ReciprocalCutOff;
    Sims[a].Box.Alpha    = Box[a].Alpha;    Sims[a].Box.Prefactor        = Box[a].Prefactor;
    Sims[a].Box.tol1     = Box[a].tol1;     Sims[a].Box.ExcludeHostGuestEwald = Box[a].ExcludeHostGuestEwald;
    cudaMalloc(&Sims[a].Box.Cell, sizeof(double) * 9); cudaMalloc(&Sims[a].Box.InverseCell, sizeof(double) * 9);
    cudaMemcpy(Sims[a].Box.Cell, Box[a].Cell, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Sims[a].Box.InverseCell, Box[a].InverseCell, 9 * sizeof(double), cudaMemcpyHostToDevice);
    Sims[a].Box.kmax = Box[a].kmax;
    // PREPARE VALUES FOR THE WIDOM STRUCT, DECLARE THE RESULT POINTERS IN WIDOM //
    WidomArray.push_back(Widom);
    Prepare_Widom(WidomArray[a], Box[a], Sims[a], SystemComponents[a], SystemComponents[a].HostSystem, MoveStats.NumberOfBlocks);
    ///////////////////////////////////
    // Read Restart file for Systems //
    ///////////////////////////////////
    //Zhao's note: here think about multiple simulations where we are doing indepedent simulations, each has a fractional molecule, need to be careful//
    bool AlreadyHasFractionalMolecule = false;
    Atoms device_System[NumberOfComponents];
    cudaMalloc(&Sims[a].d_a, sizeof(Atoms)*NumberOfComponents);
    if(RunSingleSim)
    {
      if(a == SelectedSim && ReadRestart) {RestartFileParser(Sims[a], SystemComponents[a].HostSystem, SystemComponents[a]); AlreadyHasFractionalMolecule = true;}
    }
    Copy_Atom_data_to_device(NumberOfComponents, device_System, SystemComponents[a].HostSystem);
    Prepare_TempSystem_On_Host(SystemComponents[a].TempSystem);
    cudaMemcpy(Sims[a].d_a, device_System, sizeof(Atoms)*NumberOfComponents, cudaMemcpyHostToDevice);
    // SET UP TEMPORARY ARRAYS //
    Setup_Temporary_Atoms_Structure(Sims[a].Old, SystemComponents[a].HostSystem);
    Setup_Temporary_Atoms_Structure(Sims[a].New, SystemComponents[a].HostSystem);

    //Test reading Tensorflow model//
    ReadDNNModelSetup(SystemComponents[a]);
    if(SystemComponents[a].UseDNNforHostGuest)
    {
      Read_DNN_Model(SystemComponents[a]);
      Prepare_FeatureMatrix(Sims[a], SystemComponents[a], SystemComponents[a].HostSystem, Box[a]);
    }

    //Prepare detailed Identity Swap statistics if there are more than 1 component//
    for(size_t i = 0; i < SystemComponents.size(); i++)
    if(SystemComponents[i].Total_Components > 2) //Including framework (component 0)
    {
      prepare_MixtureStats(SystemComponents[i]);
    }
    ///////////////////////////////////////////////////////////////////
    // Calculate & Initialize Ewald for the Initial state of the box //
    ///////////////////////////////////////////////////////////////////
    Check_Simulation_Energy(Box[a], SystemComponents[a].HostSystem, FF, device_FF, SystemComponents[a], INITIAL, a, Sims[a], Energy[a]);
    //////////////////////////////////////////////////////////
    // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
    //////////////////////////////////////////////////////////
    Energy[a].running_energy = CreateMolecule_InOneBox(SystemComponents[a], Sims[a], device_FF, Random, Widom, AlreadyHasFractionalMolecule);

    Check_Simulation_Energy(Box[a], SystemComponents[a].HostSystem, FF, device_FF, SystemComponents[a], CREATEMOL, a, Sims[a], Energy[a]);
  }

  ////////////////
  // RUN CYCLES //
  ////////////////
  bool RunSerialSimulations = true;
  double start = omp_get_wtime(); 

  ///////////////////////////
  // INITIALIZATION CYCLES //
  ///////////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue; 
      printf("Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Sims[i].Box.Pressure, SystemComponents[i].Temperature);
      Energy[i].running_energy += Run_Simulation_ForOneBox(NumberOfInitializationCycles, SystemComponents[i], Sims[i], device_FF, Random, WidomArray[i], Energy[i].InitialVDW, INITIALIZATION, SetMaxStep, MaxStepPerCycle, Constants);
    }
  } 
  else if(RunTogether)
  {
    Run_Simulation_MultipleBoxes(NumberOfInitializationCycles, SystemComponents, Sims, device_FF, Random, WidomArray, Energy, GibbsStatistics, INITIALIZATION, SetMaxStep, MaxStepPerCycle);
  }
  //////////////////////////
  // EQUILIBRATION CYCLES //
  //////////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      printf("Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Sims[i].Box.Pressure, SystemComponents[i].Temperature);
      Energy[i].running_energy += Run_Simulation_ForOneBox(NumberOfEquilibrationCycles, SystemComponents[i], Sims[i], device_FF, Random, WidomArray[i], Energy[i].InitialVDW, EQUILIBRATION, SetMaxStep, MaxStepPerCycle, Constants);
    }
  }
  else if(RunTogether)
  {
    Run_Simulation_MultipleBoxes(NumberOfEquilibrationCycles, SystemComponents, Sims, device_FF, Random, WidomArray, Energy, GibbsStatistics, EQUILIBRATION, SetMaxStep, MaxStepPerCycle);
  }
  
  ///////////////////////
  // PRODUCTION CYCLES //
  ///////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      printf("Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Sims[i].Box.Pressure, SystemComponents[i].Temperature);
      Energy[i].running_energy += Run_Simulation_ForOneBox(NumberOfProductionCycles, SystemComponents[i], Sims[i], device_FF, Random, WidomArray[i], Energy[i].InitialVDW, PRODUCTION, SetMaxStep, MaxStepPerCycle, Constants);
    }
  }
  else if(RunTogether)
  {
    Run_Simulation_MultipleBoxes(NumberOfProductionCycles, SystemComponents, Sims, device_FF, Random, WidomArray, Energy, GibbsStatistics, PRODUCTION, SetMaxStep, MaxStepPerCycle);
  }
  
  double end = omp_get_wtime();
  ///////////////////////////////////
  // PRINT FINAL ENERGIES AND TIME //
  ///////////////////////////////////
  printf("Work took %f seconds\n", end - start);

  // CALCULATE THE FINAL ENERGY (VDW + Real) //
  for(size_t i = 0; i < NumberOfSimulations; i++)
  { 
    if(RunSingleSim){if(i != SelectedSim) continue;}
    printf("======================================\n");
    printf("CHECKING FINAL ENERGY FOR SYSTEM [%zu]\n", i);
    printf("======================================\n");
    Check_Simulation_Energy(Box[i], SystemComponents[i].HostSystem, FF, device_FF, SystemComponents[i], FINAL, i, Sims[i], Energy[i]);
    printf("======================================\n");
  }
  /////////////////////////////////////////////////////////
  // Check if the Ewald Diff and running Diff make sense //
  /////////////////////////////////////////////////////////
  ENERGY_SUMMARY(Energy, SystemComponents, Constants);
  printf("Random Numbers Regenerated %zu times, offset: %zu, randomsize: %zu\n", Random.Rounds, Random.offset, Random.randomsize);
  //////////////////////
  // PRINT MOVIE FILE //
  //////////////////////
  GenerateRestartMovies(0, SystemComponents, Sims, FF, Box, PseudoAtom);
  return 0;
}
