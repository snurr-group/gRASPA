#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <array>
#include <omp.h>

#include <complex>

#include <iostream>
#include <filesystem>
#include <fstream>

#include "axpy.h"
#include "read_data.h"
//#include "write_data.h"
#include "equations_of_state.h"
#include "fxn_main.h"

#include <unistd.h>
#include <limits.h>
int main(void)
{
  //Zhao's note: Before everything starts, see if all the lines in Input file can be found in read_data.cpp//
  //An easy way to check if the input file is up-to-date//
  //https://stackoverflow.com/questions/143174/how-do-i-get-the-directory-that-a-program-is-running-from
  char result[ 256 ];
  ssize_t count = readlink( "/proc/self/exe", result, 256 );
  std::string exepath = std::string( result, (count > 0) ? count : 0 );
  std::cout << exepath;
  Check_Inputs_In_read_data_cpp(exepath);
  
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
  ////////////////////////////////////////////////////////////
  // DECLARE BASIC VARIABLES, READ FORCEFIELD and FRAMEWORK //
  ////////////////////////////////////////////////////////////

  int3 NComponents; NComponents.y = 1; //Assuming 1 framework species

  //std::vector<Atoms*> HostSystem(NumberOfSimulations);
  std::vector<Boxsize> Box(NumberOfSimulations); //(Boxsize*) malloc(NumberOfSimulations * sizeof(Boxsize)); //Boxsize device_Box;
  PseudoAtomDefinitions PseudoAtom;
  ForceField FF; ForceField device_FF;
  // read in force field and pseudo atom
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
  
  bool SameFrameworkEverySimulation = true; //Use the same framework (box) setup for every simulation?//

  double PRESSURE = 0.0; double TEMPERATURE = 0.0; 

  printf("------------------GENERAL SIMULATION SETUP-------------\n");
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &NumberOfInitializationCycles, &NumberOfEquilibrationCycles, &NumberOfProductionCycles, &Widom.NumberWidomTrials, &Widom.NumberWidomTrialsOrientations, &PRESSURE, &TEMPERATURE, &Allocate_space_Adsorbate, &ReadRestart, &RANDOMSEED, &SameFrameworkEverySimulation, NComponents);

  printf("Finished Checking Number of Components, There are %d framework, %d Adsorbates, %d total Components\n", NComponents.y, NComponents.z, NComponents.x);
 
  //Zhao's note: if we need to setup DNN models running just on the CPU, we set env variables before the first cuda call//
  //Then set the env variable back//
  std::vector<Components> Comp_for_DNN_Model(NumberOfSimulations);
  //Test reading Tensorflow model//
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    ReadDNNModelSetup(Comp_for_DNN_Model[i]);
    if(Comp_for_DNN_Model[i].UseDNNforHostGuest)
    {
      //###PATCH_LCLIN_MAIN_READMODEL###//
      //###PATCH_ALLEGRO_MAIN_READMODEL###//
    }
  }
  printf("DONE Reading Model Info from simulation.input file\n");
  //setenv("CUDA_VISIBLE_DEVICES", "1", 1); //After setting up tf model, set the GPU as visible again//

 
  // UNIFIED MEMORY FOR DIFFERENT SIMULATIONS //
  Simulations *Sims; cudaMallocManaged(&Sims, NumberOfSimulations*sizeof(Simulations));
  Gibbs  GibbsStatistics;
  bool   SetMaxStep = false;
  size_t MaxStepPerCycle = 1;
  read_Gibbs_Stats(GibbsStatistics, SetMaxStep, MaxStepPerCycle);
  printf("-------------------------------------------------------\n");
  // PREPARE VALUES FOR THE FORCEFIELD STRUCT //
  //file in fxn_main.h//
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
    TempComponents.HostSystem  = (Atoms*) malloc(NComponents.x * sizeof(Atoms));
    TempComponents.ReadRestart = ReadRestart;
    TempComponents.Temperature = TEMPERATURE;
    TempComponents.PseudoAtoms = PseudoAtom;
    TempComponents.FF          = FF;

    //Int3 for the number of components, consider replace Total_Components with this int3 variable//
    TempComponents.NComponents = NComponents; TempComponents.Total_Components = static_cast<size_t>(NComponents.x);
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
      ReadFramework(Box[a], PseudoAtom, a, TempComponents);
      read_Ewald_Parameters_from_input(sqrt(FF.CutOffCoul), Box[a], EwaldPrecision);
      Update_Components_for_framework(TempComponents);
    }
    read_movies_stats_print(TempComponents);
    /////////////////////////////////////
    // Read and process adsorbate data //
    /////////////////////////////////////
    //Zhao's note: different systems will share the SAME components (for adsorbate), only read it once//
      TempComponents.UseDNNforHostGuest = Comp_for_DNN_Model[a].UseDNNforHostGuest;
      TempComponents.UseAllegro         = Comp_for_DNN_Model[a].UseAllegro;
      TempComponents.UseLCLin           = Comp_for_DNN_Model[a].UseLCLin;
      TempComponents.DNNEnergyConversion= Comp_for_DNN_Model[a].DNNEnergyConversion;
      if(TempComponents.UseDNNforHostGuest)
        if(static_cast<int>(TempComponents.UseLCLin) + static_cast<int>(TempComponents.UseAllegro)/* + static_cast<int>(TempComponents.UseDylan)*/ > 1)
          throw std::runtime_error("Currently do not support using more than 1 ML model in gRASPA! Please just use 1 (or none)!!!");

      for(size_t comp = 0; comp < TempComponents.Total_Components; comp++)
      {
        Move_Statistics MoveStats;
        //Initialize Energy Averages//
        double2 tempdou = {0.0, 0.0};
        RosenbluthWeight tempRosen;
        for(size_t i = 0; i < TempComponents.Nblock; i++)
        {
          TempComponents.EnergyAverage.push_back(tempdou);
          MoveStats.MolAverage.push_back(tempdou);
          MoveStats.Rosen.push_back(tempRosen);
        }
        if(comp >= TempComponents.NComponents.y) //0: framework//
        {
          printf("Parsing [%zu] Component\n", comp);
          //skip reading the first component, which is the framework
          read_component_values_from_simulation_input(TempComponents, MoveStats, comp-TempComponents.NComponents.y, TempComponents.HostSystem[comp], PseudoAtom, Allocate_space_Adsorbate);
        }
        ReadFrameworkComponentMoves(MoveStats, TempComponents, comp);
        TempComponents.Moves.push_back(MoveStats);
      }

      SystemComponents.push_back(TempComponents);
    //}
    Setup_Box_Temperature_Pressure(Constants, SystemComponents[a], Box[a]);
    Sims[a].Box.Pressure = Box[a].Pressure; Sims[a].Box.Volume = Box[a].Volume;
    Sims[a].Box.Cubic    = Box[a].Cubic;    Sims[a].Box.ReciprocalCutOff = Box[a].ReciprocalCutOff;
    Sims[a].Box.Alpha    = Box[a].Alpha;    Sims[a].Box.Prefactor        = Box[a].Prefactor;
    Sims[a].Box.tol1     = Box[a].tol1;     Sims[a].Box.ExcludeHostGuestEwald = Box[a].ExcludeHostGuestEwald;

    //Calculate Fugacity Coefficient//
    //Note pressure in Box variable is already converted to internal units//
    ComputeFugacity(SystemComponents[a], PRESSURE, SystemComponents[a].Temperature);
    //throw std::runtime_error("EXIT, just test Fugacity Coefficient\n");
    

    cudaMalloc(&Sims[a].Box.Cell, sizeof(double) * 9); cudaMalloc(&Sims[a].Box.InverseCell, sizeof(double) * 9);
    cudaMemcpy(Sims[a].Box.Cell, Box[a].Cell, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Sims[a].Box.InverseCell, Box[a].InverseCell, 9 * sizeof(double), cudaMemcpyHostToDevice);
    Sims[a].Box.kmax = Box[a].kmax;
    // PREPARE VALUES FOR THE WIDOM STRUCT, DECLARE THE RESULT POINTERS IN WIDOM //
    WidomArray.push_back(Widom);
    Prepare_Widom(WidomArray[a], Box[a], Sims[a], SystemComponents[a], SystemComponents[a].HostSystem);
    ///////////////////////////////////
    // Read Restart file for Systems //
    ///////////////////////////////////
    //Zhao's note: here think about multiple simulations where we are doing indepedent simulations, each has a fractional molecule, need to be careful//
    bool AlreadyHasFractionalMolecule = false;
    Atoms device_System[NComponents.x];
    cudaMalloc(&Sims[a].d_a, sizeof(Atoms)*NComponents.x);
    if(RunSingleSim)
    {
      if(a == SelectedSim && ReadRestart) {RestartFileParser(Sims[a], Box[a], SystemComponents[a]); AlreadyHasFractionalMolecule = true;}
    }
    Copy_Atom_data_to_device((size_t) NComponents.x, device_System, SystemComponents[a].HostSystem);
    Prepare_TempSystem_On_Host(SystemComponents[a].TempSystem);
    cudaMemcpy(Sims[a].d_a, device_System, sizeof(Atoms)*NComponents.x, cudaMemcpyHostToDevice);
    // SET UP TEMPORARY ARRAYS //
    Setup_Temporary_Atoms_Structure(Sims[a].Old, SystemComponents[a].HostSystem);
    Setup_Temporary_Atoms_Structure(Sims[a].New, SystemComponents[a].HostSystem);

    if(SystemComponents[a].UseDNNforHostGuest)
    {
      SystemComponents[a].DNNDrift  = Comp_for_DNN_Model[a].DNNDrift;
      SystemComponents[a].ModelName = Comp_for_DNN_Model[a].ModelName;
      SystemComponents[a].DNNEnergyConversion = Comp_for_DNN_Model[a].DNNEnergyConversion;
      //Zhao's note: Hard-coded component here//
      cudaMallocManaged(&SystemComponents[a].ConsiderThisAdsorbateAtom, sizeof(bool) * SystemComponents[a].Moleculesize[1]);
      for(size_t y = 0; y < SystemComponents[a].Moleculesize[1]; y++)
      {
        printf("Atom %zu, Consider? %s\n", y, SystemComponents[a].ConsiderThisAdsorbateAtom[y] ? "true" : "false");
      }
      //Test reading Tensorflow model//
      //###PATCH_LCLIN_MAIN_PREP###//
      //###PATCH_ALLEGRO_MAIN_PREP###//
    }
    //Prepare detailed Identity Swap statistics if there are more than 1 component//
    for(size_t i = 0; i < SystemComponents.size(); i++)
    if(SystemComponents[i].Total_Components > (SystemComponents[i].NComponents.y + 1)) //NComponent for Framework + 1
    {
      prepare_MixtureStats(SystemComponents[i]);
    }
    ///////////////////////////////////////////////////////////////////
    // Calculate & Initialize Ewald for the Initial state of the box //
    ///////////////////////////////////////////////////////////////////
    Check_Simulation_Energy(Box[a], SystemComponents[a].HostSystem, FF, device_FF, SystemComponents[a], INITIAL, a, Sims[a]);
    //////////////////////////////////////////////////////////
    // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
    //////////////////////////////////////////////////////////
    Energy[a].running_energy = CreateMolecule_InOneBox(SystemComponents[a], Sims[a], device_FF, Random, Widom, AlreadyHasFractionalMolecule);

    Check_Simulation_Energy(Box[a], SystemComponents[a].HostSystem, FF, device_FF, SystemComponents[a], CREATEMOL, a, Sims[a]);
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
    Check_Simulation_Energy(Box[i], SystemComponents[i].HostSystem, FF, device_FF, SystemComponents[i], FINAL, i, Sims[i]);
    printf("======================================\n");
  }
  /////////////////////////////////////////////////////////
  // Check if the Ewald Diff and running Diff make sense //
  /////////////////////////////////////////////////////////
  ENERGY_SUMMARY(SystemComponents, Constants);
  printf("Random Numbers Regenerated %zu times, offset: %zu, randomsize: %zu\n", Random.Rounds, Random.offset, Random.randomsize);

  printf("DNN Feature Preparation Time: %.5f, DNN Prediction Time: %.5f\n", SystemComponents[0].DNNFeatureTime, SystemComponents[0].DNNPredictTime);
   printf("DNN GPU Time: %.5f, DNN Sort Time: %.5f, std::sort Time: %.5f, Featurization Time: %.5f\n", SystemComponents[0].DNNGPUTime, SystemComponents[0].DNNSortTime, SystemComponents[0].DNNstdsortTime, SystemComponents[0].DNNFeaturizationTime);

  //////////////////////
  // PRINT MOVIE FILE //
  //////////////////////
  GenerateSummaryAtEnd(0, SystemComponents, Sims, FF, Box, PseudoAtom);
  /*
  if(SystemComponents[a].UseDNNforHostGuest)
  {
    Free_DNN_Model(SystemComponents[0]);
  }
  */
  return 0;
}
