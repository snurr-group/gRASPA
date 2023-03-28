#include "axpy.h"
#include "print_statistics.h"
#include "mc_translation.h"
#include "mc_insertion_deletion.h"
#include "mc_box.h"
#include "Ewald.h"
//#include "lambda.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>

///////////////////////////////////////////////////////////
// Wrapper for Performing a move for the selected system //
///////////////////////////////////////////////////////////
inline void RunMoves(int Cycle, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, double& running_energy, double& Rosenbluth, int SimulationMode)
{
  //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
  //Zhao's note: The number of atoms can be vulnerable, adding throw error here//
  if(SystemComponents.TotalNumberOfMolecules < SystemComponents.NumberOfFrameworks)
    throw std::runtime_error("There is negative number of adsorbates. Break program!");
  size_t SelectedMolecule = (size_t) (get_random_from_zero_to_one()*(SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks));
  //Zhao's note add a protection so that the selected Molecule do not exceed (or equal to) the total number of molecules//
  if(SelectedMolecule > 0 && SelectedMolecule == (SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks))
    SelectedMolecule --;
  size_t comp = SystemComponents.NumberOfFrameworks; // When selecting components, skip the component 0 (because it is the framework)
  size_t SelectedMolInComponent = SelectedMolecule; size_t totalsize= 0;
  for(size_t ijk = SystemComponents.NumberOfFrameworks; ijk < SystemComponents.Total_Components; ijk++) //Assuming Framework atoms are the top in the Atoms array
  {
    if(SelectedMolInComponent == 0) break;
    totalsize += SystemComponents.NumberOfMolecule_for_Component[ijk];
    if(SelectedMolInComponent >= totalsize)
    {
      comp++;
      SelectedMolInComponent -= SystemComponents.NumberOfMolecule_for_Component[ijk];
    }
  }
  if(SystemComponents.NumberOfMolecule_for_Component[comp] == 0)
  { //no molecule in the system for this species
    running_energy += Insertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
    return;
  }
  double RANDOMNUMBER = get_random_from_zero_to_one();
  if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
  {
    //////////////////////////////
    // PERFORM TRANSLATION MOVE //
    //////////////////////////////
    running_energy += SingleMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, 0);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) //Rotation
  {
    ///////////////////////////
    // PERFORM ROTATION MOVE //
    ///////////////////////////
    running_energy += SingleMove(SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp, 1);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].WidomProb)
  {
    //////////////////////////////////
    // PERFORM WIDOM INSERTION MOVE //
    //////////////////////////////////
    double2 newScale = SystemComponents.Lambda[comp].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0)//
    size_t SelectedTrial=0; bool SuccessConstruction = false; double energy = 0.0; double StoredR = 0.0;
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
    running_energy += Reinsertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].CBCFProb && SystemComponents.hasfractionalMolecule[comp])
  {
    ///////////////////////
    // PERFORM CBCF MOVE //
    ///////////////////////
    SelectedMolInComponent = SystemComponents.Lambda[comp].FractionalMoleculeID;
    running_energy += CBCFMove(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
  }
  else
  {
    ////////////////////////////
    // PERFORM GCMC INSERTION //
    ////////////////////////////
    if(get_random_from_zero_to_one() < 0.5)
    {
      running_energy += Insertion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
    }
    else
    {
      ///////////////////////////
      // PERFORM GCMC DELETION //
      ///////////////////////////
      //Zhao's note: Do not do a deletion if the chosen molecule is a fractional molecule, fractional molecules should go to CBCFSwap moves//
      if(!((SystemComponents.hasfractionalMolecule[comp]) && SelectedMolInComponent == SystemComponents.Lambda[comp].FractionalMoleculeID))
      {
        running_energy += Deletion(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp);
      }
    }
  }
}

double CreateMolecule_InOneBox(Components& SystemComponents, Simulations& Sims, ForceField FF, RandomNumber& Random, WidomStruct Widom, bool AlreadyHasFractionalMolecule)
{
  double running_energy = 0.0;
  // Create Molecules in the Box Before the Simulation //
  size_t CreateFailCount = 0; size_t Created = 0; size_t SelectedMol = 0;
  for(size_t comp = 1; comp < SystemComponents.Total_Components; comp++)
  {
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
        SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
        size_t OldVal = SystemComponents.NumberOfMolecule_for_Component[comp];

        size_t NewBin = 5;
        double newLambda = static_cast<double>(NewBin) * SystemComponents.Lambda[comp].delta;
        double2 newScale = SystemComponents.Lambda[comp].SET_SCALE(newLambda);
        running_energy  += CreateMolecule(SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, newScale);
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
      running_energy  += CreateMolecule(SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, newScale);
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
        if(SystemComponents[sim].hasfractionalMolecule[icomp])
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
    size_t Steps = 20;
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
      double NVTGibbsRN = get_random_from_zero_to_one();
      bool GoodForNVTGibbs = false;
      //If no framework atoms for both simulation boxes//
      if(NumberOfSimulations == 2 && SystemComponents[0].Moleculesize[0] == 0 && SystemComponents[1].Moleculesize[0] == 0) GoodForNVTGibbs = true;
      if(GibbsStatistics.DoGibbs && GoodForNVTGibbs)
        if(NVTGibbsRN < GibbsStatistics.GibbsBoxProb) //Zhao's note: for the test, do it at the last step//
        {
          NVTGibbsMove(SystemComponents, Sims, FF, Energy, GibbsStatistics);
          continue;
        }
      double GibbsXferRN = get_random_from_zero_to_one();
      if(GibbsStatistics.DoGibbs && GoodForNVTGibbs)
        if(GibbsXferRN < GibbsStatistics.GibbsXferProb)
        {
          //Do a Gibbs Particle Transfer move//
          size_t SelectedComponent = 1;
          GibbsParticleTransfer(SystemComponents, Sims, FF, Random, Widom, Energy, SelectedComponent, GibbsStatistics);
          continue;
        }
      for(size_t sim = 0; sim < NumberOfSimulations; sim++)
      {
        RunMoves(i, SystemComponents[sim], Sims[sim], FF, Random, Widom[sim], Energy[sim].running_energy, running_Rosenbluth[sim], SimulationMode);
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
          if(SystemComponents[sim].hasfractionalMolecule[icomp])
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
            Update_Max_Translation(SystemComponents[sim].Moves[comp], Sims[sim]);
        for(size_t comp = 0; comp < SystemComponents[sim].Total_Components; comp++)
          if(SystemComponents[sim].Moves[comp].RotationTotal > 0)
            Update_Max_Rotation(SystemComponents[sim].Moves[comp], Sims[sim]);
      }
      if(i%5000==0)
        printf("Cycle: %zu, %zu Adsorbate Molecules, Total Energy: %.5f\n", i, SystemComponents[sim].TotalNumberOfMolecules - SystemComponents[sim].NumberOfFrameworks, Energy[sim].running_energy + Energy[sim].InitialEnergy);
      ////////////////////////////////////////////////
      // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
      ////////////////////////////////////////////////
      if(i%5000==0 && SimulationMode == EQUILIBRATION)
      {
        for(size_t icomp = 0; icomp < SystemComponents[sim].Total_Components; icomp++)
        if(SystemComponents[sim].hasfractionalMolecule[icomp])
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
      PrintAllStatistics(SystemComponents[sim], Sims[sim], Cycles, SimulationMode, Energy[sim].running_energy, Energy[sim].InitialEnergy, BlockAverageSize[sim]);
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
  if(SimulationMode == INITIALIZATION && Cycles > 0)
  {
    SystemComponents.deltaVDWReal = 0.0;
    SystemComponents.deltaEwald   = 0.0;
  }
  if(SimulationMode == PRODUCTION)
  {
    BlockAverageSize = Cycles / SystemComponents.Nblock;
    if(Cycles % SystemComponents.Nblock != 0)
      printf("Warning! Number of Cycles cannot be divided by Number of blocks. Residue values go to the last block\n");
  }
 
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
      if(SystemComponents.hasfractionalMolecule[icomp])
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
      SystemComponents.Moves[comp].BlockID = i/BlockAverageSize;
      if(SystemComponents.Moves[comp].BlockID >= SystemComponents.Nblock) SystemComponents.Moves[comp].BlockID--;   }
    ////////////////////////////////////////
    // Zhao's note: for debugging purpose //
    ////////////////////////////////////////
    if(SetMaxStep && Steps > MaxStepPerCycle) Steps = MaxStepPerCycle;
    for(size_t j = 0; j < Steps; j++)
    {
      RunMoves(i, SystemComponents, Sims, FF, Random, Widom, running_energy, running_Rosenbluth, SimulationMode);
    }
    //////////////////////////////////////////////
    // SAMPLE (EQUILIBRATION) CBCF BIASING TERM //
    //////////////////////////////////////////////
    if(SimulationMode == EQUILIBRATION && i%50==0)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      { //Try to sample it if there are more CBCF moves performed//
        if(SystemComponents.hasfractionalMolecule[icomp])
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
          Update_Max_Translation(SystemComponents.Moves[comp], Sims);
        if(SystemComponents.Moves[comp].RotationTotal > 0)
          Update_Max_Rotation(SystemComponents.Moves[comp], Sims);
      }
    }
    if(i%50==0)
      printf("Cycle: %zu, %zu Adsorbate Molecules, Total Energy: %.5f\n", i, SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks, running_energy + init_energy);
    ////////////////////////////////////////////////
    // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
    ////////////////////////////////////////////////
    if(i%5000==0 && SimulationMode == EQUILIBRATION)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
        if(SystemComponents.hasfractionalMolecule[icomp])
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
  }
  //print statistics
  if(Cycles > 0)
  {
    if(SimulationMode == EQUILIBRATION) printf("Sampled %zu WangLandau, Adjusted WL %zu times\n", WLSampled, WLAdjusted);
    PrintAllStatistics(SystemComponents, Sims, Cycles, SimulationMode, running_energy, init_energy, BlockAverageSize);
    Print_Widom_Statistics(SystemComponents, Sims.Box, Constants, 1);
  }
  return running_energy;
}
