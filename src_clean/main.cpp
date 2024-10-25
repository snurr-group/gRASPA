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

// // #include <limits.h>

void printMemoryUsage() 
{
  std::cout << "" << "\n";
  std::cout << "==========================="<< std::endl;
  std::cout << "==    END OF PROGRAM!    =="<< std::endl;
  std::cout << "== PRINTING MEMORY USAGE =="<< std::endl;
  std::cout << "==========================="<< std::endl;

  std::ifstream file("/proc/self/statm");
  if (file.is_open()) 
  {
    long totalProgramSize, residentSet, sharedPages, text, data, unused, library;
    file >> totalProgramSize >> residentSet >> sharedPages >> text >> unused >> data >> library;

    // Convert the sizes from pages to bytes
    long pageSize = sysconf(_SC_PAGE_SIZE);
    totalProgramSize *= pageSize;
    residentSet *= pageSize;
    sharedPages *= pageSize;
    text *= pageSize;
    data *= pageSize;

    std::cout << "Total Program Size: " << totalProgramSize / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Resident Set Size: " << residentSet / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Pages: " << sharedPages / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Text (code): " << text / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Data + Stack: " << data / (1024 * 1024) << " MB" << std::endl;
  } 
  else 
  {
    std::cerr << "Unable to open /proc/self/statm" << std::endl;
  }
  file.close();
}

int main(void) //normal cpp
//Variables main(void) //for pybind
{
  //Zhao's note: Before everything starts, see if all the lines in Input file can be found in read_data.cpp//
  //An easy way to check if the input file is up-to-date//
  //https://stackoverflow.com/questions/143174/how-do-i-get-the-directory-that-a-program-is-running-from
  //This breaks when the code is generated to a library, so comment for now..//
  char result[ 256 ];
  ssize_t count = readlink( "/proc/self/exe", result, 256 );
  std::string exepath = std::string( result, (count > 0) ? count : 0 );
  std::cout << exepath;
  //Check_Inputs_In_read_data_cpp(exepath);
 
  //Variable for all important structs// 
  Variables Vars;

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

  int3& NComponents = Vars.TempComponents.NComponents; 
  NComponents.y = 1; //Assuming 1 framework species

  Vars.Box.resize(NumberOfSimulations);
  // read in force field and pseudo atom

  //////////////////////////////////////
  //Process Forcefield and PseudoAtoms//
  //////////////////////////////////////
  read_FFParams_from_input(Vars.Input);
  
  ForceFieldParser(Vars.Input, Vars.PseudoAtoms);
  ForceField_Processing(Vars.Input);
  OverWrite_Mixing_Rule(Vars.Input);
  OverWriteTailCorrection(Vars.Input);
  Copy_InputLoader_Data(Vars);
  Copy_ForceField_to_GPU(Vars);

  PseudoAtomParser(Vars.PseudoAtoms);
  PseudoAtomProcessing(Vars);


  bool SameFrameworkEverySimulation = true; //Use the same framework (box) setup for every simulation?//

  printf("------------------GENERAL SIMULATION SETUP-------------\n");
  read_simulation_input(Vars, &ReadRestart, &SameFrameworkEverySimulation);

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
  //Simulations *Sims; 
  cudaMallocManaged(&Vars.Sims, NumberOfSimulations*sizeof(Simulations));
  read_Gibbs_and_Cycle_Stats(Vars, Vars.SetMaxStep, Vars.MaxStepPerCycle);
  printf("-------------------------------------------------------\n");
  // PREPARE VALUES FOR THE FORCEFIELD STRUCT //
  //file in fxn_main.h//

  //////////////////////////
  // SETUP RANDOM NUMBERS //
  //////////////////////////
  Vars.Random.Setup(333334);

  //if true, then we can simulate the same framework at different T/P//
  //If false, then we can do Gibbs (where a box is empty)//

  bool samePressure = true; bool sameTemperature = true;

  //Ewald Energy Arrays//
  //Setup Initial Ewald//
  std::vector<SystemEnergies>Energy(NumberOfSimulations);

  for(size_t a = 0; a < NumberOfSimulations; a++)
  {
    //if(sameTemperature)
    if(!samePressure || !sameTemperature) throw std::runtime_error("Currently not allowing the different Pressure/Temperature!");
    printf("==========================================\n");
    printf("====== Preparing Simulation box %zu ======\n", a);
    printf("==========================================\n");
    //Allocate data on the host for each simulation//
    Vars.TempComponents.HostSystem  = (Atoms*) malloc(NComponents.x * sizeof(Atoms));
    Vars.TempComponents.ReadRestart = ReadRestart;
    Vars.TempComponents.PseudoAtoms = Vars.PseudoAtoms;
    Vars.TempComponents.FF          = Vars.FF;
   
    //Zhao's note: allocate more space (multiply by the initial structure vector size)//
    Vars.TempComponents.StructureFactor_Multiplier = Vars.StructureFactor_Multiplier;

    /////////////////////////////////////
    // Read and process framework data //
    /////////////////////////////////////
    if(!SameFrameworkEverySimulation || a == 0)
    {
      Vars.TempComponents.NumberOfPseudoAtoms.resize(Vars.TempComponents.PseudoAtoms.Name.size());
      std::fill(Vars.TempComponents.NumberOfPseudoAtoms.begin(), Vars.TempComponents.NumberOfPseudoAtoms.end(), 0);
      if(a > 0 && !SameFrameworkEverySimulation) printf("Processing %zu, new framework\n", a);
      //Read framework data from cif/poscar file//
      ReadFramework(Vars.Box[a], Vars.TempComponents.PseudoAtoms, a, Vars.TempComponents);
      ReadVoidFraction(Vars);

      read_Ewald_Parameters_from_input(sqrt(Vars.FF.CutOffCoul), Vars.Box[a], Vars.Input.EwaldPrecision);
      Update_Components_for_framework(Vars.TempComponents);
    }
    read_movies_stats_print(Vars.TempComponents, a);
    /////////////////////////////////////
    // Read and process adsorbate data //
    /////////////////////////////////////
    //Zhao's note: different systems will share the SAME components (for adsorbate), only read it once//
      Vars.TempComponents.UseDNNforHostGuest = Comp_for_DNN_Model[a].UseDNNforHostGuest;
      Vars.TempComponents.UseAllegro         = Comp_for_DNN_Model[a].UseAllegro;
      Vars.TempComponents.UseLCLin           = Comp_for_DNN_Model[a].UseLCLin;
      Vars.TempComponents.DNNEnergyConversion= Comp_for_DNN_Model[a].DNNEnergyConversion;
      if(Vars.TempComponents.UseDNNforHostGuest)
        if(static_cast<int>(Vars.TempComponents.UseLCLin) + static_cast<int>(Vars.TempComponents.UseAllegro)/* + static_cast<int>(Vars.TempComponents.UseDylan)*/ > 1)
          throw std::runtime_error("Currently do not support using more than 1 ML model in gRASPA! Please just use 1 (or none)!!!");

      for(size_t comp = 0; comp < Vars.TempComponents.NComponents.x; comp++)
      {
        Move_Statistics MoveStats;
        //Initialize Energy Averages//
        double2 tempdou = {0.0, 0.0};
        RosenbluthWeight tempRosen;
	MoveStats.MolSQPerComponent.resize(Vars.TempComponents.NComponents.x, std::vector<double>(Vars.TempComponents.Nblock, 0.0));
        for(size_t i = 0; i < Vars.TempComponents.Nblock; i++)
        {
          MoveStats.MolAverage.push_back(tempdou);
          MoveStats.Rosen.push_back(tempRosen);
        }
        if(comp >= Vars.TempComponents.NComponents.y) //0: framework//
        {
          printf("Parsing [%zu] Component\n", comp);
          //skip reading the first component, which is the framework
          read_component_values_from_simulation_input(Vars, Vars.TempComponents, MoveStats, comp-Vars.TempComponents.NComponents.y, Vars.TempComponents.HostSystem[comp], Vars.TempComponents.PseudoAtoms, Vars.Allocate_space_Adsorbate, a);
        }
        ReadFrameworkComponentMoves(MoveStats, Vars.TempComponents, comp);
        Vars.TempComponents.Moves.push_back(MoveStats);
      }

      Vars.SystemComponents.push_back(Vars.TempComponents);
    
    //Calculate Fugacity Coefficient//
    //Note pressure in Vars.Box variable is already converted to internal units//
    ComputeFugacity(Vars.SystemComponents[a], Vars.SystemComponents[a].Pressure, Vars.SystemComponents[a].Temperature, Vars.Box[a].Volume);
    //throw std::runtime_error("EXIT, just test Fugacity Coefficient\n");
    
    cudaMalloc(&Vars.Sims[a].Box.Cell, sizeof(double) * 9); cudaMalloc(&Vars.Sims[a].Box.InverseCell, sizeof(double) * 9);

    ///////////////////////////////////
    // Read Restart file for Systems //
    ///////////////////////////////////
    //Zhao's note: here think about multiple simulations where we are doing indepedent simulations, each has a fractional molecule, need to be careful//
    bool AlreadyHasFractionalMolecule = false;
    Atoms device_System[NComponents.x];
    cudaMalloc(&Vars.Sims[a].d_a, sizeof(Atoms)*NComponents.x);
    InitializeMaxTranslationRotation(Vars.SystemComponents[a]);
    //Read initial configurations either from restart file or from lammps data file//
    if(RunSingleSim)
    {
      if(a == SelectedSim && ReadRestart)
      { 
        ReadRestartInputFileType(Vars.SystemComponents[a]);
        if(Vars.SystemComponents[a].RestartInputFileType == RASPA_RESTART) 
        {
          RestartFileParser(Vars.Box[a], Vars.SystemComponents[a]); AlreadyHasFractionalMolecule = true;
        }
        else if(Vars.SystemComponents[a].RestartInputFileType == LAMMPS_DATA)
        {
          LMPDataFileParser(Vars.Box[a], Vars.SystemComponents[a]);
        }
      }
    }
    //Zhao's note: move copying cell information to GPU after reading restart
    // PREPARE VALUES FOR THE WIDOM STRUCT, DECLARE THE RESULT POINTERS IN WIDOM //
    Vars.Widom.push_back(Vars.TempWidom);
    Prepare_Widom(Vars.Widom[a], Vars.Box[a], Vars.Sims[a], Vars.SystemComponents[a], Vars.SystemComponents[a].HostSystem);

    Setup_Box_Temperature_Pressure(Vars.Constants, Vars.SystemComponents[a], Vars.Box[a]);
    Vars.Sims[a].Box.UseLAMMPSEwald = Vars.Box[a].UseLAMMPSEwald;
    Vars.Sims[a].Box.Volume = Vars.Box[a].Volume;
    Vars.Sims[a].Box.Cubic    = Vars.Box[a].Cubic;    Vars.Sims[a].Box.ReciprocalCutOff = Vars.Box[a].ReciprocalCutOff;
    Vars.Sims[a].Box.Alpha    = Vars.Box[a].Alpha;    Vars.Sims[a].Box.Prefactor        = Vars.Box[a].Prefactor;
    Vars.Sims[a].Box.tol1     = Vars.Box[a].tol1;

    cudaMemcpy(Vars.Sims[a].Box.Cell, Vars.Box[a].Cell, 9 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Vars.Sims[a].Box.InverseCell, Vars.Box[a].InverseCell, 9 * sizeof(double), cudaMemcpyHostToDevice);
    Vars.Sims[a].Box.kmax = Vars.Box[a].kmax;

    Copy_Atom_data_to_device((size_t) NComponents.x, device_System, Vars.SystemComponents[a].HostSystem);
    Prepare_TempSystem_On_Host(Vars.SystemComponents[a].TempSystem);
    cudaMemcpy(Vars.Sims[a].d_a, device_System, sizeof(Atoms)*NComponents.x, cudaMemcpyHostToDevice);
    // SET UP TEMPORARY ARRAYS //
    Setup_Temporary_Atoms_Structure(Vars.Sims[a].Old, Vars.SystemComponents[a].HostSystem);
    Setup_Temporary_Atoms_Structure(Vars.Sims[a].New, Vars.SystemComponents[a].HostSystem);

    if(Vars.SystemComponents[a].UseDNNforHostGuest)
    {
      Vars.SystemComponents[a].DNNDrift  = Comp_for_DNN_Model[a].DNNDrift;
      Vars.SystemComponents[a].ModelName = Comp_for_DNN_Model[a].ModelName;
      Vars.SystemComponents[a].DNNEnergyConversion = Comp_for_DNN_Model[a].DNNEnergyConversion;
      //Zhao's note: Hard-coded component here//
      //Assuming component `1` is just the 1st adsorbate species//
      std::vector<bool>ConsiderThisAdsorbateAtom(Vars.SystemComponents[a].Moleculesize[1], false);
      for(size_t y = 0; y < Vars.SystemComponents[a].Moleculesize[1]; y++)
      {
        ConsiderThisAdsorbateAtom[y] = Vars.SystemComponents[a].ConsiderThisAdsorbateAtom[y];
      }
      //Declare a new, cuda managed mem (accessible on both CPU/GPU) to overwrite the original  bool mem
      cudaMallocManaged(&Vars.SystemComponents[a].ConsiderThisAdsorbateAtom, sizeof(bool) * Vars.SystemComponents[a].Moleculesize[1]);
      for(size_t y = 0; y < Vars.SystemComponents[a].Moleculesize[1]; y++)
      {
        Vars.SystemComponents[a].ConsiderThisAdsorbateAtom[y] = ConsiderThisAdsorbateAtom[y];
        printf("Atom %zu, Consider? %s\n", y, Vars.SystemComponents[a].ConsiderThisAdsorbateAtom[y] ? "true" : "false");
      }
      //Test reading Tensorflow model//
      //###PATCH_LCLIN_MAIN_PREP###//
      //###PATCH_ALLEGRO_MAIN_PREP###//
    }
    //Prepare detailed Identity Swap statistics if there are more than 1 component//
    for(size_t i = 0; i < Vars.SystemComponents.size(); i++)
    if(Vars.SystemComponents[i].NComponents.x > (Vars.SystemComponents[i].NComponents.y + 1)) //NComponent for Framework + 1
    {
      prepare_MixtureStats(Vars.SystemComponents[i]);
    }
    ///////////////////////////////////////////////////////////////////
    // Calculate & Initialize Ewald for the Initial state of the box //
    ///////////////////////////////////////////////////////////////////
    Check_Simulation_Energy(Vars.Box[a], Vars.SystemComponents[a].HostSystem, Vars.FF, Vars.device_FF, Vars.SystemComponents[a], INITIAL, a, Vars.Sims[a], true);
    //////////////////////////////////////////////////////////
    // CREATE MOLECULES IN THE BOX BEFORE SIMULAITON STARTS //
    //////////////////////////////////////////////////////////
    Energy[a].running_energy = CreateMolecule_InOneBox(Vars.SystemComponents[a], Vars.Sims[a], Vars.device_FF, Vars.Random, Vars.Widom[a], AlreadyHasFractionalMolecule);

    Check_Simulation_Energy(Vars.Box[a], Vars.SystemComponents[a].HostSystem, Vars.FF, Vars.device_FF, Vars.SystemComponents[a], CREATEMOL, a, Vars.Sims[a], true);
  }

  printf("============================================\n");
  printf("== END OF PREPARATION, SIMULATION STARTS! ==\n");
  printf("============================================\n");

  ////////////////
  // RUN CYCLES //
  ////////////////
  double start = omp_get_wtime(); 

  ///////////////////////////
  // INITIALIZATION CYCLES //
  ///////////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue; 
      fprintf(Vars.SystemComponents[i].OUTPUT, "Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Vars.SystemComponents[i].Pressure, Vars.SystemComponents[i].Temperature);
      Vars.SimulationMode = INITIALIZATION; Energy[i].running_energy += Run_Simulation_ForOneBox(Vars, i);
    }
  }
  else if(RunTogether)
  {
    Run_Simulation_MultipleBoxes(Vars, INITIALIZATION);
  }
  //////////////////////////
  // EQUILIBRATION CYCLES //
  //////////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      fprintf(Vars.SystemComponents[i].OUTPUT, "Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Vars.SystemComponents[i].Pressure, Vars.SystemComponents[i].Temperature);
      Vars.SimulationMode = EQUILIBRATION; Energy[i].running_energy += Run_Simulation_ForOneBox(Vars, i);
    }
  }
  else if(RunTogether)
  {
    Run_Simulation_MultipleBoxes(Vars, EQUILIBRATION);
  }
  
  ///////////////////////
  // PRODUCTION CYCLES //
  ///////////////////////
  if(RunOneByOne)
  {
    for(size_t i = 0; i < NumberOfSimulations; i++)
    {
      if(i > 0 && RunSingleSim) continue;
      fprintf(Vars.SystemComponents[i].OUTPUT, "Running Simulation Boxes in SERIAL, currently [%zu] box; pres: %.5f, temp: %.5f\n", i, Vars.SystemComponents[i].Pressure, Vars.SystemComponents[i].Temperature);
      Vars.SimulationMode = PRODUCTION; Energy[i].running_energy += Run_Simulation_ForOneBox(Vars, i);
    }
  }
  else if(RunTogether)
  {
    Run_Simulation_MultipleBoxes(Vars, PRODUCTION);
  }

  printf("========================\n");
  printf("== END OF SIMULATION! ==\n");
  printf("========================\n");
 
  double end = omp_get_wtime();
  ///////////////////////////////////
  // PRINT FINAL ENERGIES AND TIME //
  ///////////////////////////////////

  // CALCULATE THE FINAL ENERGY //
  for(size_t i = 0; i < NumberOfSimulations; i++)
  { 
    fprintf(Vars.SystemComponents[i].OUTPUT, "Work took %f seconds\n", end - start);
    if(RunSingleSim){if(i != SelectedSim) continue;}
    check_energy_wrapper(Vars, i);
    //Report Random Number Summary and DNN statistics//
    fprintf(Vars.SystemComponents[i].OUTPUT, "Random Numbers Regenerated %zu times, offset: %zu, randomsize: %zu\n", Vars.Random.Rounds, Vars.Random.offset, Vars.Random.randomsize);

    fprintf(Vars.SystemComponents[i].OUTPUT, "DNN Feature Preparation Time: %.5f, DNN Prediction Time: %.5f\n", Vars.SystemComponents[0].DNNFeatureTime, Vars.SystemComponents[0].DNNPredictTime);
    fprintf(Vars.SystemComponents[i].OUTPUT, "DNN GPU Time: %.5f, ", Vars.SystemComponents[i].DNNGPUTime);
    fprintf(Vars.SystemComponents[i].OUTPUT, "DNN Sort Time: %.5f, ", Vars.SystemComponents[0].DNNSortTime);
    fprintf(Vars.SystemComponents[i].OUTPUT, "std::sort Time: %.5f, ", Vars.SystemComponents[0].DNNstdsortTime); 
    fprintf(Vars.SystemComponents[i].OUTPUT, "Featurization Time: %.5f\n", Vars.SystemComponents[0].DNNFeaturizationTime);
  }
  /////////////////////////////////////////////////////////
  // Check if the Ewald Diff and running Diff make sense //
  /////////////////////////////////////////////////////////
  ENERGY_SUMMARY(Vars.SystemComponents, Vars.Constants);

  GenerateSummaryAtEnd(0, Vars.SystemComponents, Vars.Sims, Vars.FF, Vars.Box);
  //Check CPU mem used//
  printMemoryUsage();
  /*
  if(Vars.SystemComponents[a].UseDNNforHostGuest)
  {
    Free_DNN_Model(Vars.SystemComponents[0]);
  }
  */

  for(size_t i = 0; i < Vars.SystemComponents.size(); i++)
    if(Vars.SystemComponents[i].OUTPUT != stderr)
      fclose(Vars.SystemComponents[i].OUTPUT);

  return 0;  //normal cpp
  //return Vars; //for pybind
}
