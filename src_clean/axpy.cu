#include "axpy.h"
#include "mc_single_particle.h"
#include "mc_swap_moves.h"
#include "mc_box.h"

#include "write_data.h"

#include "print_statistics.cuh"

//#include "lambda.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <optional>

#include <fstream>

inline void Copy_AtomData_from_Device(Atoms* System, Atoms* d_a, Components& SystemComponents, Boxsize& HostBox, Simulations& Sims)
{
  cudaMemcpy(System, d_a, SystemComponents.Total_Components * sizeof(Atoms), cudaMemcpyDeviceToHost);
  for(size_t ijk=0; ijk < SystemComponents.Total_Components; ijk++)
  {
    // if the host allocate_size is different from the device, allocate more space on the host
    SystemComponents.HostSystem[ijk].pos       = (double3*) malloc(System[ijk].Allocate_size*sizeof(double3));
    SystemComponents.HostSystem[ijk].scale     = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
    SystemComponents.HostSystem[ijk].charge    = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
    SystemComponents.HostSystem[ijk].scaleCoul = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
    SystemComponents.HostSystem[ijk].Type      = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
    SystemComponents.HostSystem[ijk].MolID     = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
    SystemComponents.HostSystem[ijk].size      = System[ijk].size;
    SystemComponents.HostSystem[ijk].Allocate_size = System[ijk].Allocate_size;

    cudaMemcpy(SystemComponents.HostSystem[ijk].pos, System[ijk].pos, sizeof(double3)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SystemComponents.HostSystem[ijk].scale, System[ijk].scale, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SystemComponents.HostSystem[ijk].charge, System[ijk].charge, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SystemComponents.HostSystem[ijk].scaleCoul, System[ijk].scaleCoul, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SystemComponents.HostSystem[ijk].Type, System[ijk].Type, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SystemComponents.HostSystem[ijk].MolID, System[ijk].MolID, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    SystemComponents.HostSystem[ijk].size = System[ijk].size;
  }
  HostBox.Cell = (double*) malloc(9 * sizeof(double));
  HostBox.InverseCell = (double*) malloc(9 * sizeof(double));
  cudaMemcpy(HostBox.Cell,        Sims.Box.Cell,        sizeof(double)*9, cudaMemcpyDeviceToHost);
  cudaMemcpy(HostBox.InverseCell, Sims.Box.InverseCell, sizeof(double)*9, cudaMemcpyDeviceToHost);
  HostBox.Cubic = Sims.Box.Cubic;
}

inline void GenerateRestartMovies(Components& SystemComponents, Simulations& Sims, PseudoAtomDefinitions& PseudoAtom, size_t systemIdx, int SimulationMode)
{
  //Generate Restart file during the simulation, regardless of the phase
  Atoms device_System[SystemComponents.NComponents.x];
  Boxsize HostBox;
  Copy_AtomData_from_Device(device_System, Sims.d_a, SystemComponents, HostBox, Sims);
  create_Restart_file(0, SystemComponents.HostSystem, SystemComponents, SystemComponents.FF, HostBox, PseudoAtom.Name, systemIdx);
  Write_All_Adsorbate_data(0, SystemComponents.HostSystem, SystemComponents, SystemComponents.FF, HostBox, PseudoAtom.Name, systemIdx);
  //Only generate LAMMPS data movie for production phase
  if(SimulationMode == PRODUCTION)  create_movie_file(SystemComponents.HostSystem, SystemComponents, HostBox, PseudoAtom.Name, systemIdx);
}

///////////////////////////////////////////////////////////
// Wrapper for Performing a move for the selected system //
///////////////////////////////////////////////////////////
inline void RunMoves(int Cycle, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, double& Rosenbluth, int SimulationMode)
{
  SystemComponents.CURRENTCYCLE = Cycle;
  //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
  //Zhao's note: The number of atoms can be vulnerable, adding throw error here//
  if(SystemComponents.TotalNumberOfMolecules < SystemComponents.NumberOfFrameworks)
    throw std::runtime_error("There is negative number of adsorbates. Break program!");

  size_t comp = 0; // When selecting components, skip the component 0 (because it is the framework)
  size_t SelectedMolecule = 0; 
  size_t SelectedMolInComponent = 0;

  size_t NumberOfImmobileFrameworkMolecules = 0; size_t ImmobileFrameworkSpecies = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    if(SystemComponents.Moves[i].TotalProb < 1e-10)
    {
      ImmobileFrameworkSpecies++;
      NumberOfImmobileFrameworkMolecules += SystemComponents.NumberOfMolecule_for_Component[i];
    }
  while(SystemComponents.Moves[comp].TotalProb < 1e-10)
  {
    comp = (size_t) (Get_Uniform_Random() * SystemComponents.NComponents.x);
  }
  SelectedMolInComponent = (size_t) (Get_Uniform_Random() * SystemComponents.NumberOfMolecule_for_Component[comp]);

  MoveEnergy DeltaE;
  double RANDOMNUMBER = Get_Uniform_Random();
  //printf("Step %zu, selected Comp %zu, Mol %zu, RANDOM: %.5f", Cycle, comp, SelectedMolInComponent, RANDOMNUMBER);
  if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
  {
    //////////////////////////////
    // PERFORM TRANSLATION MOVE //
    //////////////////////////////
    //printf(" Translation\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
    {
      DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, TRANSLATION);
    }
    else
    {
      SystemComponents.Tmmc[comp].Update(1.0, SystemComponents.NumberOfMolecule_for_Component[comp], TRANSLATION);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) //Rotation
  {
    ///////////////////////////
    // PERFORM ROTATION MOVE //
    ///////////////////////////
    //printf(" Rotation\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
    {
      DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, ROTATION);
    }
    else
    {
      SystemComponents.Tmmc[comp].Update(1.0, SystemComponents.NumberOfMolecule_for_Component[comp], ROTATION);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].SpecialRotationProb) //Special Rotation for Framework Components
  {
    ///////////////////////////////////
    // PERFORM SPECIAL ROTATION MOVE //
    ///////////////////////////////////
    //printf(" Special Rotation\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
      DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SPECIAL_ROTATION);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].WidomProb)
  {
    //////////////////////////////////
    // PERFORM WIDOM INSERTION MOVE //
    //////////////////////////////////
    //printf(" Widom Insertion\n");
    double2 newScale = SystemComponents.Lambda[comp].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0)//
    size_t SelectedTrial=0; bool SuccessConstruction = false; MoveEnergy energy; double StoredR = 0.0;
    double Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, CBMC_INSERTION, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale);
    if(SystemComponents.Moleculesize[comp] > 1 && Rosenbluth > 1e-150)
    {
      size_t SelectedFirstBeadTrial = SelectedTrial;
      Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, CBMC_INSERTION, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); 
      //Zhao's note: need to add widom insertion data back//
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].ReinsertionProb)
  {
    //////////////////////////////
    // PERFORM REINSERTION MOVE //
    //////////////////////////////
    //printf(" Reinsertion\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
    {
      DeltaE = Reinsertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
    }
    else
    {
      SystemComponents.Tmmc[comp].Update(1.0, SystemComponents.NumberOfMolecule_for_Component[comp], REINSERTION);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].IdentitySwapProb)
  {
    //printf(" Identity Swap\n");
    DeltaE = IdentitySwapMove(SystemComponents, Sims, Widom, FF, Random);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].CBCFProb && SystemComponents.hasfractionalMolecule[comp])
  {
    ///////////////////////
    // PERFORM CBCF MOVE //
    ///////////////////////
    //printf(" CBCF\n");
    SelectedMolInComponent = SystemComponents.Lambda[comp].FractionalMoleculeID;
    DeltaE = CBCFMove(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].SwapProb)
  {
    ////////////////////////////
    // PERFORM GCMC INSERTION //
    ////////////////////////////
    if(Get_Uniform_Random() < 0.5)
    {
      //printf(" Swap Insertion\n");
      if(!SystemComponents.SingleSwap)
      {
        DeltaE = Insertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
      }
      else
      {
        DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_INSERTION);
        //DeltaE = SingleSwapMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_INSERTION);
      }
    }
    else
    {
      ///////////////////////////
      // PERFORM GCMC DELETION //
      ///////////////////////////
      //printf(" Swap Deletion\n");
      //Zhao's note: Do not do a deletion if the chosen molecule is a fractional molecule, fractional molecules should go to CBCFSwap moves//
      if(!((SystemComponents.hasfractionalMolecule[comp]) && SelectedMolInComponent == SystemComponents.Lambda[comp].FractionalMoleculeID))
      {
        if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
        {
          if(!SystemComponents.SingleSwap)
          {
            DeltaE = Deletion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
          }
          else
          {
            //DeltaE = SingleSwapMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_DELETION);
            DeltaE = SingleBodyMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, SINGLE_DELETION);
          }
        }
        else
        {
          SystemComponents.Tmmc[comp].Update(0.0, SystemComponents.NumberOfMolecule_for_Component[comp], DELETION);
        }
      }
    }
  }
  
  if(Cycle == 4)
  {
    printf("Cycle [%d], Printing DeltaE\n", Cycle);
    DeltaE.print();
  }
  
  SystemComponents.deltaE += DeltaE;
}

double CreateMolecule_InOneBox(Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, bool AlreadyHasFractionalMolecule)
{
  double running_energy = 0.0;
  // Create Molecules in the Box Before the Simulation //
  for(size_t comp = SystemComponents.NComponents.y; comp < SystemComponents.NComponents.x; comp++)
  {
    size_t CreateFailCount = 0; size_t Created = 0; size_t SelectedMol = 0;
    CreateFailCount = 0;
    printf("Component %zu, Need to create %zu full molecule\n", comp, SystemComponents.NumberOfCreateMolecules[comp]);
    //Create Fractional Molecule first//
    if(SystemComponents.hasfractionalMolecule[comp])
    {
      //Zhao's note: If we need to create fractional molecule, then we initialize WangLandau Histogram//
      size_t FractionalMolToCreate = 1;
      if(AlreadyHasFractionalMolecule) FractionalMolToCreate = 0;
      if(FractionalMolToCreate > 0) Initialize_WangLandauIteration(SystemComponents.Lambda[comp]);
      while(FractionalMolToCreate > 0)
      {
        printf("Creating Fractional Molecule for Component %zu; There are %zu Molecules of that component in the System\n", comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
        SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; 
        //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
        size_t OldVal = SystemComponents.NumberOfMolecule_for_Component[comp];

        size_t NewBin = 5;
        if(SystemComponents.Tmmc[comp].DoTMMC) NewBin = 0;
        double newLambda = static_cast<double>(NewBin) * SystemComponents.Lambda[comp].delta;
        double2 newScale = SystemComponents.Lambda[comp].SET_SCALE(newLambda);
        MoveEnergy DeltaE;
        DeltaE = CreateMolecule(SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, newScale);
        running_energy += DeltaE.total();
        SystemComponents.CreateMoldeltaE += DeltaE;
        if(SystemComponents.NumberOfMolecule_for_Component[comp] == OldVal)
        {
          CreateFailCount ++;
        }
        else
        {
          FractionalMolToCreate --; Created ++; SystemComponents.Lambda[comp].FractionalMoleculeID = SelectedMol;
          SystemComponents.Lambda[comp].currentBin = NewBin;
        }
        if(CreateFailCount > 1000000000) throw std::runtime_error("Bad Insertions When Creating Fractional Molecules!");
      }
    }
    while(SystemComponents.NumberOfCreateMolecules[comp] > 0)
    {
      printf("Creating %zu Molecule for Component %zu; There are %zu Molecules of that component in the System\n", Created, comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
      SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
      size_t OldVal    = SystemComponents.NumberOfMolecule_for_Component[comp];
      double2 newScale = SystemComponents.Lambda[comp].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0)//
      MoveEnergy DeltaE;
      DeltaE = CreateMolecule(SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, newScale);
      //printf("Creating %zu molecule\n", SelectedMol);
      //DeltaE.print();
      running_energy += DeltaE.total();
      SystemComponents.CreateMoldeltaE += DeltaE;
      printf("Delta E in creating molecules:\n"); DeltaE.print();
      if(SystemComponents.NumberOfMolecule_for_Component[comp] == OldVal)
      {CreateFailCount ++;} else {SystemComponents.NumberOfCreateMolecules[comp] --; Created ++;}
      if(CreateFailCount > 10000) throw std::runtime_error("Bad Insertions When Creating Molecules!");
    }
  }
  return running_energy;
}

void Run_Simulation_MultipleBoxes(int Cycles, std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, RandomNumber& Random, std::vector<WidomStruct>& Widom, std::vector<SystemEnergies>& Energy, Gibbs& GibbsStatistics, int SimulationMode, bool SetMaxStep, size_t MaxStepPerCycle)
{
  size_t NumberOfSimulations = SystemComponents.size();
  size_t WLSampled = 0; size_t WLAdjusted = 0;

  std::vector<int> BlockAverageSize(NumberOfSimulations, 1);

  std::string Mode;
  switch(SimulationMode)
  {
    case INITIALIZATION:{Mode = "INITIALIZATION"; break;}
    case EQUILIBRATION: {Mode = "EQUILIBRATION"; break;}
    case PRODUCTION:    {Mode = "PRODUCTION"; break;}
  } 
  /*
  if(SimulationMode == INITIALIZATION && Cycles > 0)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    { 
      SystemComponents[sim].deltaVDWReal = 0.0;
      SystemComponents[sim].deltaEwald   = 0.0;
    }
  }
  */
  if(SimulationMode == PRODUCTION)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      BlockAverageSize[sim] = Cycles / SystemComponents[sim].Nblock;
      if(Cycles % SystemComponents[sim].Nblock != 0)
        printf("Warning! Number of Cycles cannot be divided by Number of blocks. Residue values go to the last block\n");
    }
  }

  std::vector<double> running_Rosenbluth(NumberOfSimulations, 0.0);

  /////////////////////////////////////////////
  // FINALIZE (PRODUCTION) CBCF BIASING TERM //
  /////////////////////////////////////////////
  if(SimulationMode == PRODUCTION)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
      for(size_t icomp = 0; icomp < SystemComponents[sim].Total_Components; icomp++)
        if(SystemComponents[sim].hasfractionalMolecule[icomp] && !SystemComponents[sim].Tmmc[icomp].DoTMMC)
          Finalize_WangLandauIteration(SystemComponents[sim].Lambda[icomp]);
  }

  ///////////////////////////////////////////////////////////////////////
  // FORCE INITIALIZING CBCF BIASING TERM BEFORE INITIALIZATION CYCLES //
  ///////////////////////////////////////////////////////////////////////
  if(SimulationMode == INITIALIZATION && Cycles > 0)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
      for(size_t icomp = 0; icomp < SystemComponents[sim].Total_Components; icomp++)
        if(SystemComponents[sim].hasfractionalMolecule[icomp])
          Initialize_WangLandauIteration(SystemComponents[sim].Lambda[icomp]);
  }
  ///////////////////////////////////////////////////////
  // Run the simulations for different boxes IN SERIAL //
  ///////////////////////////////////////////////////////
  for(size_t i = 0; i < Cycles; i++)
  {
    size_t Steps = 1;
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      if(Steps < SystemComponents[sim].TotalNumberOfMolecules) 
      {
        Steps = SystemComponents[sim].TotalNumberOfMolecules;
      }
    }
    ////////////////////////////////////////
    // Zhao's note: for debugging purpose //
    ////////////////////////////////////////
    if(SetMaxStep && Steps > MaxStepPerCycle) Steps = MaxStepPerCycle;
    for(size_t j = 0; j < Steps; j++)
    {
      //Draw a random number, if fits, run a Gibbs Box move//
      //Zhao's note: if a Gibbs move is performed, skip the cycle//
      double NVTGibbsRN = Get_Uniform_Random();
      bool GoodForNVTGibbs = false;
      //If no framework atoms for both simulation boxes//
      if(NumberOfSimulations == 2 && SystemComponents[0].Moleculesize[0] == 0 && SystemComponents[1].Moleculesize[0] == 0) GoodForNVTGibbs = true;
      if(GibbsStatistics.DoGibbs && GoodForNVTGibbs)
        if(NVTGibbsRN < GibbsStatistics.GibbsBoxProb) //Zhao's note: for the test, do it at the last step//
        {
          double start = omp_get_wtime();
          printf("Cycle [%zu], Step [%zu], Perform Gibbs Volume Move\n", i, j);
          NVTGibbsMove(SystemComponents, Sims, FF, Energy, GibbsStatistics);
          double end = omp_get_wtime();
          GibbsStatistics.GibbsTime += end - start;
          continue;
        }
      double GibbsXferRN = Get_Uniform_Random();
      if(GibbsStatistics.DoGibbs && GoodForNVTGibbs)
        if(GibbsXferRN < GibbsStatistics.GibbsXferProb)
        {
          //Do a Gibbs Particle Transfer move//
          size_t SelectedComponent = 1;
          printf("Cycle [%zu], Step [%zu], Perform Gibbs Particle Move\n", i, j);
          GibbsParticleTransfer(SystemComponents, Sims, FF, Random, Widom, Energy, SelectedComponent, GibbsStatistics);
          continue;
        }
      for(size_t sim = 0; sim < NumberOfSimulations; sim++)
      {
        RunMoves(i, SystemComponents[sim], Sims[sim], FF, Random, Widom[sim], running_Rosenbluth[sim], SimulationMode);
      }
    }
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      //////////////////////////////////////////////
      // SAMPLE (EQUILIBRATION) CBCF BIASING TERM //
      //////////////////////////////////////////////
      if(SimulationMode == EQUILIBRATION && i%50==0)
      {
        for(size_t icomp = 0; icomp < SystemComponents[sim].Total_Components; icomp++)
        { 
          //Try to sample it if there are more CBCF moves performed//
          if(SystemComponents[sim].hasfractionalMolecule[icomp] && !SystemComponents[sim].Tmmc[icomp].DoTMMC)
          {
            Sample_WangLandauIteration(SystemComponents[sim].Lambda[icomp]);
            WLSampled++;
          }
        }
      }

      if(i%500==0)
      {
        for(size_t comp = 0; comp < SystemComponents[sim].Total_Components; comp++)
          if(SystemComponents[sim].Moves[comp].TranslationTotal > 0)
            Update_Max_Translation(SystemComponents[sim], comp);
        for(size_t comp = 0; comp < SystemComponents[sim].Total_Components; comp++)
          if(SystemComponents[sim].Moves[comp].RotationTotal > 0)
            Update_Max_Rotation(SystemComponents[sim], comp);
        for(size_t comp = 0; comp < SystemComponents[sim].Total_Components; comp++)
          if(SystemComponents[sim].Moves[comp].SpecialRotationTotal > 0)
            Update_Max_SpecialRotation(SystemComponents[sim], comp);
      }
      
      if(i % SystemComponents[sim].PrintStatsEvery == 0) Print_Cycle_Statistics(i, SystemComponents[sim], Mode);
      ////////////////////////////////////////////////
      // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
      ////////////////////////////////////////////////
      if(i%5000==0 && SimulationMode == EQUILIBRATION)
      {
        for(size_t icomp = 0; icomp < SystemComponents[sim].Total_Components; icomp++)
        if(SystemComponents[sim].hasfractionalMolecule[icomp] && !SystemComponents[sim].Tmmc[icomp].DoTMMC)
        {  
          Adjust_WangLandauIteration(SystemComponents[sim].Lambda[icomp]); 
          WLAdjusted++;
        }
      }
      if(SimulationMode == PRODUCTION)
      {
        //Record values for energy//
        Gather_Averages_Types(SystemComponents[sim].EnergyAverage, Energy[sim].InitialEnergy, Energy[sim].running_energy, i, BlockAverageSize[sim], SystemComponents[sim].Nblock);
        //Record values for Number of atoms//
        for(size_t comp = 0; comp < SystemComponents[sim].Total_Components; comp++)
          Gather_Averages_Types(SystemComponents[sim].Moves[comp].MolAverage, SystemComponents[sim].NumberOfMolecule_for_Component[comp], 0.0, i, BlockAverageSize[sim], SystemComponents[sim].Nblock);
      }
    }
  }
  //print statistics
  if(Cycles > 0)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      if(SimulationMode == EQUILIBRATION) printf("Sampled %zu WangLandau, Adjusted WL %zu times\n", WLSampled, WLAdjusted);
      PrintAllStatistics(SystemComponents[sim], Sims[sim], Cycles, SimulationMode, BlockAverageSize[sim]);
    }
    if(GibbsStatistics.DoGibbs)
    {
      PrintGibbs(GibbsStatistics);
    }
  }
}

double Run_Simulation_ForOneBox(int Cycles, Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, double init_energy, int SimulationMode, bool SetMaxStep, size_t MaxStepPerCycle, Units Constants)
{
  std::vector<size_t>CBCFPerformed(SystemComponents.Total_Components);
  size_t WLSampled = 0; size_t WLAdjusted = 0;

  int BlockAverageSize = 1;
  if(SimulationMode == PRODUCTION)
  {
    BlockAverageSize = Cycles / SystemComponents.Nblock;
    if(Cycles % SystemComponents.Nblock != 0)
      printf("Warning! Number of Cycles cannot be divided by Number of blocks. Residue values go to the last block\n");
  }

  printf("Number of Frameworks: %zu\n", SystemComponents.NumberOfFrameworks);
 
  if(SimulationMode == EQUILIBRATION) //Rezero the TMMC stats at the beginning of the Equilibration cycles//
  {
    //Clear TMMC data in the collection matrix//
    for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
      SystemComponents.Tmmc[comp].ClearCMatrix();
  }
  //Clear Rosenbluth weight statistics after Initialization//
  if(SimulationMode == EQUILIBRATION)
  {
    for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
      for(size_t i = 0; i < SystemComponents.Nblock; i++)
        SystemComponents.Moves[comp].ClearRosen(i);
  }
  double running_energy = 0.0;
  double running_Rosenbluth = 0.0;
  /////////////////////////////////////////////
  // FINALIZE (PRODUCTION) CBCF BIASING TERM //
  /////////////////////////////////////////////
  if(SimulationMode == PRODUCTION)
  {
    for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp] && !SystemComponents.Tmmc[icomp].DoTMMC)
        Finalize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }

  ///////////////////////////////////////////////////////////////////////
  // FORCE INITIALIZING CBCF BIASING TERM BEFORE INITIALIZATION CYCLES //
  ///////////////////////////////////////////////////////////////////////
  if(SimulationMode == INITIALIZATION && Cycles > 0)
  {
    for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp])
        Initialize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }

  std::string Mode;
  switch(SimulationMode)
  {
    case INITIALIZATION:{Mode = "INITIALIZATION"; break;}
    case EQUILIBRATION: {Mode = "EQUILIBRATION"; break;}
    case PRODUCTION:    {Mode = "PRODUCTION"; break;}
  }

  for(size_t i = 0; i < Cycles; i++)
  {
    size_t Steps = 20;
    if(Steps < SystemComponents.TotalNumberOfMolecules)
    {
      Steps = SystemComponents.TotalNumberOfMolecules;
    }
    //Determine BlockID//
    for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++){
      BlockAverageSize = Cycles / SystemComponents.Nblock;
      if(BlockAverageSize > 0) SystemComponents.Moves[comp].BlockID = i/BlockAverageSize; 
      if(SystemComponents.Moves[comp].BlockID >= SystemComponents.Nblock) SystemComponents.Moves[comp].BlockID--;   }
    ////////////////////////////////////////
    // Zhao's note: for debugging purpose //
    ////////////////////////////////////////
    if(SetMaxStep && Steps > MaxStepPerCycle) Steps = MaxStepPerCycle;
    for(size_t j = 0; j < Steps; j++)
    {
      RunMoves(i, SystemComponents, Sims, FF, Random, Widom, running_Rosenbluth, SimulationMode);
    }
    //////////////////////////////////////////////
    // SAMPLE (EQUILIBRATION) CBCF BIASING TERM //
    //////////////////////////////////////////////
    if(SimulationMode == EQUILIBRATION && i%50==0)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      { //Try to sample it if there are more CBCF moves performed//
        if(SystemComponents.hasfractionalMolecule[icomp] && !SystemComponents.Tmmc[icomp].DoTMMC)
        {
          Sample_WangLandauIteration(SystemComponents.Lambda[icomp]);
          CBCFPerformed[icomp] = SystemComponents.Moves[icomp].CBCFTotal; WLSampled++;
        }
      }
    }

    if(i%500==0)
    {
      for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
      {  
        if(SystemComponents.Moves[comp].TranslationTotal > 0)
          Update_Max_Translation(SystemComponents, comp);
        if(SystemComponents.Moves[comp].RotationTotal > 0)
          Update_Max_Rotation(SystemComponents, comp);
        if(SystemComponents.Moves[comp].SpecialRotationTotal > 0)
          Update_Max_SpecialRotation(SystemComponents, comp);
      }
    }
    if(i%SystemComponents.PrintStatsEvery==0) Print_Cycle_Statistics(i, SystemComponents, Mode);
    ////////////////////////////////////////////////
    // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
    ////////////////////////////////////////////////
    if(i%5000==0 && SimulationMode == EQUILIBRATION)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
        if(SystemComponents.hasfractionalMolecule[icomp] && !SystemComponents.Tmmc[icomp].DoTMMC)//Try not to use CBCFC + TMMC//
        {  Adjust_WangLandauIteration(SystemComponents.Lambda[icomp]); WLAdjusted++;}
    }
    if(SimulationMode == PRODUCTION)
    {
      //Record values for energy//
      Gather_Averages_Types(SystemComponents.EnergyAverage, init_energy, running_energy, i, BlockAverageSize, SystemComponents.Nblock);
      //Record values for Number of atoms//
      for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
        Gather_Averages_Types(SystemComponents.Moves[comp].MolAverage, SystemComponents.NumberOfMolecule_for_Component[comp], 0.0, i, BlockAverageSize, SystemComponents.Nblock);
    }
    if(SimulationMode != INITIALIZATION && i > 0)
    {
      for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
        if(i % SystemComponents.Tmmc[comp].UpdateTMEvery == 0)
          SystemComponents.Tmmc[comp].AdjustTMBias();
    }
    if(i % SystemComponents.MoviesEvery == 0)//Generate restart file and movies 
      GenerateRestartMovies(SystemComponents, Sims, SystemComponents.PseudoAtoms, 0, SimulationMode);
  }
  //print statistics
  if(Cycles > 0)
  {
    if(SimulationMode == EQUILIBRATION) printf("Sampled %zu WangLandau, Adjusted WL %zu times\n", WLSampled, WLAdjusted);
    PrintAllStatistics(SystemComponents, Sims, Cycles, SimulationMode, BlockAverageSize);
    //Print_Widom_Statistics(SystemComponents, Sims.Box, Constants, 1);
  }
  //At the end of the sim, print a last-step restart and last-step movie
  GenerateRestartMovies(SystemComponents, Sims, SystemComponents.PseudoAtoms, 0, SimulationMode);
  return running_energy;
}
