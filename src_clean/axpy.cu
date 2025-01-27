
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

//#include <format>

inline void Copy_AtomData_from_Device(Atoms* System, Atoms* d_a, Components& SystemComponents, Boxsize& HostBox, Simulations& Sims)
{
  cudaMemcpy(System, d_a, SystemComponents.NComponents.x * sizeof(Atoms), cudaMemcpyDeviceToHost);
  for(size_t ijk=0; ijk < SystemComponents.NComponents.x; ijk++)
  {
    if(SystemComponents.HostSystem[ijk].Allocate_size != System[ijk].Allocate_size)
    {
      // if the host allocate_size is different from the device, allocate more space on the host
      SystemComponents.HostSystem[ijk].pos       = (double3*) malloc(System[ijk].Allocate_size*sizeof(double3));
      SystemComponents.HostSystem[ijk].scale     = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      SystemComponents.HostSystem[ijk].charge    = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      SystemComponents.HostSystem[ijk].scaleCoul = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      SystemComponents.HostSystem[ijk].Type      = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      SystemComponents.HostSystem[ijk].MolID     = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      SystemComponents.HostSystem[ijk].Allocate_size = System[ijk].Allocate_size;
    }
  
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

inline void GenerateRestartMovies(Variables& Vars, size_t systemId, PseudoAtomDefinitions& PseudoAtom, int SimulationMode)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims = Vars.Sims[systemId];
  Boxsize& HostBox = Vars.Box[systemId];
  //Generate Restart file during the simulation, regardless of the phase
  Atoms device_System[SystemComponents.NComponents.x];
  Copy_AtomData_from_Device(device_System, Sims.d_a, SystemComponents, HostBox, Sims);
  create_Restart_file(0, SystemComponents.HostSystem, SystemComponents, SystemComponents.FF, HostBox, PseudoAtom.Name, systemId);
  Write_All_Adsorbate_data(0, SystemComponents.HostSystem, SystemComponents, SystemComponents.FF, HostBox, PseudoAtom.Name, systemId);
  //Only generate LAMMPS data movie for production phase
  if(SimulationMode == PRODUCTION)  create_movie_file(SystemComponents.HostSystem, SystemComponents, HostBox, PseudoAtom.Name, systemId);
}

///////////////////////////////////////////////////////////
// Wrapper for Performing a move for the selected system //
///////////////////////////////////////////////////////////
void Select_Box_Component_Molecule(Variables& Vars, size_t box_index)
{
  Components& SystemComponents = Vars.SystemComponents[box_index];
  WidomStruct& Widom = Vars.Widom[box_index];
  SystemComponents.TempVal.Initialize();
  size_t& comp                   = SystemComponents.TempVal.component;
  size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
  
  //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
  //Zhao's note: The number of atoms can be vulnerable, adding throw error here//
  if(SystemComponents.TotalNumberOfMolecules < SystemComponents.NumberOfFrameworks)
    throw std::runtime_error("There is negative number of adsorbates. Break program!");

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

  Vars.RandomNumber = Get_Uniform_Random();
}
void RunMoves(Variables& Vars, size_t box_index, int Cycle)
{
  MC_MOVES MOVES;

  Components& SystemComponents = Vars.SystemComponents[box_index];
  Simulations& Sims = Vars.Sims[box_index];
  ForceField& FF = Vars.device_FF;
  //RandomNumber& Random = Vars.Random;
  WidomStruct& Widom = Vars.Widom[box_index];

  //variables that affects the selection of a move, written into TempVal//
  Select_Box_Component_Molecule(Vars, box_index);
  double& RANDOMNUMBER = Vars.RandomNumber;
  size_t& comp         = SystemComponents.TempVal.component;
  size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
  //printf("Step %zu, selected Comp %zu, Mol %zu, RANDOM: %.5f", Cycle, comp, SelectedMolInComponent, RANDOMNUMBER);

  MoveEnergy DeltaE;
  int& MoveType = SystemComponents.TempVal.MoveType;
  if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
  {
    MoveType = TRANSLATION;
    //////////////////////////////
    // PERFORM TRANSLATION MOVE //
    //////////////////////////////
    //printf(" Translation\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
    {
      DeltaE = SingleBodyMove(Vars, box_index);
    }
    else
    {
      SystemComponents.Tmmc[comp].Update(1.0, SystemComponents.NumberOfMolecule_for_Component[comp], TRANSLATION);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) //Rotation
  {
    MoveType = ROTATION;
    ///////////////////////////
    // PERFORM ROTATION MOVE //
    ///////////////////////////
    //printf(" Rotation\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
    {
      DeltaE = SingleBodyMove(Vars, box_index);
    }
    else
    {
      SystemComponents.Tmmc[comp].Update(1.0, SystemComponents.NumberOfMolecule_for_Component[comp], ROTATION);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].SpecialRotationProb) //Special Rotation for Framework Components
  {
    MoveType = SPECIAL_ROTATION;
    ///////////////////////////////////
    // PERFORM SPECIAL ROTATION MOVE //
    ///////////////////////////////////
    //printf(" Special Rotation\n");
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
      DeltaE = SingleBodyMove(Vars, box_index);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].WidomProb)
  {
    MoveType = WIDOM;
    //////////////////////////////////
    // PERFORM WIDOM INSERTION MOVE //
    //////////////////////////////////
    //printf(" Widom Insertion\n");
    double2& newScale = SystemComponents.TempVal.Scale; 
    newScale = SystemComponents.Lambda[comp].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0)//
    double Rosenbluth = MOVES.INSERTION.WidomMove(Vars, box_index);
    SystemComponents.Moves[comp].RecordRosen(Rosenbluth, WIDOM);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].ReinsertionProb)
  {
    //////////////////////////////
    // PERFORM REINSERTION MOVE //
    //////////////////////////////
    //printf(" Reinsertion\n");
    MoveType = REINSERTION;
    if(SystemComponents.NumberOfMolecule_for_Component[comp] > 0)
    {
      //DeltaE = Reinsertion(Vars, box_index);
      DeltaE = MOVES.REINSERTION.Run(Vars, box_index);
    }
    else
    {
      SystemComponents.Tmmc[comp].Update(1.0, SystemComponents.NumberOfMolecule_for_Component[comp], REINSERTION);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].IdentitySwapProb)
  {
    MoveType = IDENTITY_SWAP;
    //printf(" Identity Swap\n");
    DeltaE = IdentitySwapMove(Vars, box_index);
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].CBCFProb && SystemComponents.hasfractionalMolecule[comp])
  {
    ///////////////////////
    // PERFORM CBCF MOVE //
    ///////////////////////
    //printf(" CBCF\n");
    SelectedMolInComponent = SystemComponents.Lambda[comp].FractionalMoleculeID;
    DeltaE = CBCFMove(Vars, box_index);
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
        MoveType = INSERTION;
        DeltaE = MOVES.INSERTION.Run(Vars, box_index);
        //DeltaE = Insertion(Vars, box_index);
      }
      else
      {
        MoveType = SINGLE_INSERTION;
        DeltaE = SingleBodyMove(Vars, box_index);
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
            MoveType = DELETION;
            DeltaE = MOVES.DELETION.Run(Vars, box_index);
            //DeltaE = Deletion(Vars, box_index);
          }
          else
          {
            MoveType = SINGLE_DELETION;
            DeltaE = SingleBodyMove(Vars, box_index);
          }
        }
        else
        {
          MoveType = DELETION;
          SystemComponents.Tmmc[comp].Update(0.0, SystemComponents.NumberOfMolecule_for_Component[comp], DELETION);
        }
      }
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].VolumeMoveProb)
  {
    //printf(" VOLUME MOVE\n");
    double start = omp_get_wtime();
    ForceField& FF = Vars.device_FF;
    VolumeMove(SystemComponents, Sims, FF);
    double end = omp_get_wtime();
    SystemComponents.VolumeMoveTime += end - start;
  }
  //Gibbs Xfer//
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].GibbsSwapProb)
  {
    //if(Vars.GibbsStatistics.DoGibbs)
    //printf(" Gibbs SWAP\n");
    if(Vars.SystemComponents.size() == 2)
    {
      //GibbsParticleTransfer(Vars, comp, Vars.GibbsStatistics);
      MOVES.GIBBS_PARTICLE_XFER.Run(Vars, box_index, Vars.GibbsStatistics);
    }
  }
  else if(RANDOMNUMBER < SystemComponents.Moves[comp].GibbsVolumeMoveProb)
  {
    //printf(" Gibbs VOLUME\n");
    if(Vars.SystemComponents.size() == 2)
      NVTGibbsMove(Vars.SystemComponents, Vars.Sims, FF, Vars.GibbsStatistics);
  }
  SystemComponents.deltaE += DeltaE;
}

double CreateMolecule_InOneBox(Variables& Vars, size_t systemId, bool AlreadyHasFractionalMolecule)
{
  MC_MOVES MOVES; 
  Components& SystemComponents = Vars.SystemComponents[systemId];
  //Simulations& Sims = Vars.Sims[systemId];
  //ForceField& FF = Vars.device_FF;
  //RandomNumber& Random = Vars.Random;
  //WidomStruct& Widom = Vars.Widom[systemId];
  double running_energy = 0.0;
  // Create Molecules in the Box Before the Simulation //
  for(size_t comp = SystemComponents.NComponents.y; comp < SystemComponents.NComponents.x; comp++)
  {
    size_t CreateFailCount = 0; size_t Created = 0; size_t SelectedMol = 0;
    CreateFailCount = 0;
    fprintf(SystemComponents.OUTPUT, "Component %zu, Need to create %zu full molecule\n", comp, SystemComponents.NumberOfCreateMolecules[comp]);
    //Create Fractional Molecule first//
    if(SystemComponents.hasfractionalMolecule[comp])
    {
      //Zhao's note: If we need to create fractional molecule, then we initialize WangLandau Histogram//
      size_t FractionalMolToCreate = 1;
      if(AlreadyHasFractionalMolecule) FractionalMolToCreate = 0;
      if(FractionalMolToCreate > 0) Initialize_WangLandauIteration(SystemComponents.Lambda[comp]);
      while(FractionalMolToCreate > 0)
      {
        fprintf(SystemComponents.OUTPUT, "Creating Fractional Molecule for Component %zu; There are %zu Molecules of that component in the System\n", comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
        SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; 
        //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
        size_t OldVal = SystemComponents.NumberOfMolecule_for_Component[comp];

        size_t NewBin = 5;
        MoveEnergy DeltaE;
        if(SystemComponents.Tmmc[comp].DoTMMC) NewBin = 0;
        double newLambda = static_cast<double>(NewBin) * SystemComponents.Lambda[comp].delta;
        SystemComponents.TempVal.Initialize();
	SystemComponents.TempVal.Scale = SystemComponents.Lambda[comp].SET_SCALE(newLambda);
        SystemComponents.TempVal.MoveType = INSERTION;
        SystemComponents.TempVal.component = comp;
        SystemComponents.TempVal.molecule  = SelectedMol;
        DeltaE = MOVES.INSERTION.CreateMolecule(Vars, systemId);
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
        if(CreateFailCount > 1e20) throw std::runtime_error("Bad Insertions When Creating Fractional Molecules!");
      }
    }
    while(SystemComponents.NumberOfCreateMolecules[comp] > 0)
    {
      fprintf(SystemComponents.OUTPUT, "Creating %zu Molecule for Component %zu; There are %zu Molecules of that component in the System\n", Created, comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
      SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
      size_t OldVal    = SystemComponents.NumberOfMolecule_for_Component[comp];
      MoveEnergy DeltaE;
      SystemComponents.TempVal.Initialize();
      SystemComponents.TempVal.Scale = SystemComponents.Lambda[comp].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0)//
      SystemComponents.TempVal.MoveType = INSERTION;
      SystemComponents.TempVal.component = comp;
      SystemComponents.TempVal.molecule  = SelectedMol;
      DeltaE = MOVES.INSERTION.CreateMolecule(Vars, systemId);
      //printf("Creating %zu molecule\n", SelectedMol);
      //DeltaE.print();
      running_energy += DeltaE.total();
      SystemComponents.CreateMoldeltaE += DeltaE;
      fprintf(SystemComponents.OUTPUT, "Delta E in creating molecules:\n"); DeltaE.print();
      if(SystemComponents.NumberOfMolecule_for_Component[comp] == OldVal)
      {CreateFailCount ++;} else {SystemComponents.NumberOfCreateMolecules[comp] --; Created ++;}
      if(CreateFailCount > 1e10) throw std::runtime_error("Bad Insertions When Creating Molecules!");
    }
  }
  return running_energy;
}

void GatherStatisticsDuringSimulation(Variables& Vars, size_t systemId, size_t cycle)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations&  Sims           = Vars.Sims[systemId];
  size_t& i = cycle;
  int& BlockAverageSize        = Vars.BlockAverageSize;
  int& SimulationMode          = Vars.SimulationMode;
  std::string& Mode = Vars.Mode;
  //////////////////////////////////////////////
  // SAMPLE (EQUILIBRATION) CBCF BIASING TERM //
  //////////////////////////////////////////////
  if(SimulationMode == EQUILIBRATION && i%50==0)
  {
    for(size_t icomp = 0; icomp < SystemComponents.NComponents.x; icomp++)
    { //Try to sample it if there are more CBCF moves performed//
      if(SystemComponents.hasfractionalMolecule[icomp] && !SystemComponents.Tmmc[icomp].DoTMMC)
      {
        Sample_WangLandauIteration(SystemComponents.Lambda[icomp]);
        SystemComponents.CBCFPerformed[icomp] = SystemComponents.Moves[icomp].CBCFTotal; 
        SystemComponents.WLSampled++;
      }
    }
  }

  if(i%500==0)
  {
    for(size_t comp = 0; comp < SystemComponents.NComponents.x; comp++)
    {  
      if(SystemComponents.Moves[comp].TranslationTotal > 0)
        Update_Max_Translation(SystemComponents, comp);
      if(SystemComponents.Moves[comp].RotationTotal > 0)
        Update_Max_Rotation(SystemComponents, comp);
      if(SystemComponents.Moves[comp].SpecialRotationTotal > 0)
        Update_Max_SpecialRotation(SystemComponents, comp);
      if(SystemComponents.VolumeMoveAttempts > 0) Update_Max_VolumeChange(SystemComponents);
    }
  }
  if(i%SystemComponents.PrintStatsEvery==0) Print_Cycle_Statistics(i, SystemComponents, Mode);
  ////////////////////////////////////////////////
  // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
  ////////////////////////////////////////////////
  if(i%5000==0 && SimulationMode == EQUILIBRATION)
  {
    for(size_t icomp = 0; icomp < SystemComponents.NComponents.x; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp] && !SystemComponents.Tmmc[icomp].DoTMMC)//Try not to use CBCFC + TMMC//
      {  Adjust_WangLandauIteration(SystemComponents.Lambda[icomp]); SystemComponents.WLAdjusted++;}
  }
  if(SimulationMode == PRODUCTION)
  {
    //Record values for Number of atoms//
    for(size_t comp = 0; comp < SystemComponents.NComponents.x; comp++)
    {
      Gather_Averages_Types(SystemComponents.Moves[comp].MolAverage, SystemComponents.NumberOfMolecule_for_Component[comp], 0.0, i, BlockAverageSize, SystemComponents.Nblock);
      //Gather total energy * number of molecules for each adsorbate component//
      if(comp >= SystemComponents.NComponents.y)
      {
        double deltaE_Adsorbate = SystemComponents.deltaE.total() - SystemComponents.deltaE.HHVDW - SystemComponents.deltaE.HHEwaldE - SystemComponents.deltaE.HHReal;
        double ExN = SystemComponents.createmol_energy + deltaE_Adsorbate * SystemComponents.NumberOfMolecule_for_Component[comp];
        Gather_Averages_double(SystemComponents.EnergyTimesNumberOfMolecule[comp], ExN, i, BlockAverageSize, SystemComponents.Nblock);
        //Calculate Average Excess Loading//
        //AmountOfExcessMolecules only be resized during EOS calculation, don't have that? then no excess loading because excess loading needs compressibility from EOS//
        if(SystemComponents.AmountOfExcessMolecules.size() > 0)
          Gather_Averages_Types(SystemComponents.ExcessLoading[comp], SystemComponents.NumberOfMolecule_for_Component[comp] - SystemComponents.AmountOfExcessMolecules[comp], 0.0, i, BlockAverageSize, SystemComponents.Nblock);
      }
      for(size_t compj = 0; compj < SystemComponents.NComponents.x; compj++)
      {
        if(comp >= SystemComponents.NComponents.y && compj >= SystemComponents.NComponents.y)
        {
          double NxNj = SystemComponents.NumberOfMolecule_for_Component[comp] * SystemComponents.NumberOfMolecule_for_Component[compj];
          Gather_Averages_double(SystemComponents.Moves[comp].MolSQPerComponent[compj], NxNj, i, BlockAverageSize, SystemComponents.Nblock);
          Gather_Averages_Types(SystemComponents.DensityPerComponent[comp], SystemComponents.NumberOfMolecule_for_Component[comp] / Sims.Box.Volume, 0.0, i, BlockAverageSize, SystemComponents.Nblock);
        }
      }
    }
    Gather_Averages_Types(SystemComponents.VolumeAverage, Sims.Box.Volume, 0.0, i, BlockAverageSize, SystemComponents.Nblock);
    Gather_Averages_MoveEnergy(SystemComponents, i, BlockAverageSize, SystemComponents.deltaE);
  }
  if(SimulationMode != INITIALIZATION && i > 0)
  {
    for(size_t comp = 0; comp < SystemComponents.NComponents.x; comp++)
      if(i % SystemComponents.Tmmc[comp].UpdateTMEvery == 0)
        SystemComponents.Tmmc[comp].AdjustTMBias();
  }
  if(i % SystemComponents.MoviesEvery == 0)//Generate restart file and movies 
    GenerateRestartMovies(Vars, systemId, SystemComponents.PseudoAtoms, SimulationMode);
}

void InitialMCBeforeMoves(Variables& Vars, size_t systemId)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  size_t NumberOfSimulations   = Vars.SystemComponents.size();
  int&   BlockAverageSize      = Vars.BlockAverageSize;
  int&   Cycles                = Vars.Cycles;

  SystemComponents.CBCFPerformed.resize(SystemComponents.NComponents.x);
  SystemComponents.WLSampled = 0; SystemComponents.WLAdjusted = 0;

  fprintf(SystemComponents.OUTPUT, "==================================\n");
  std::string& Mode   = Vars.Mode;
  int& SimulationMode = Vars.SimulationMode;
  switch(SimulationMode)
  {
    case INITIALIZATION: {Mode = "INITIALIZATION"; fprintf(SystemComponents.OUTPUT, "== RUNNING INITIALIZATION PHASE ==\n"); Cycles = Vars.NumberOfInitializationCycles; break;}
    case EQUILIBRATION:  {Mode = "EQUILIBRATION";  fprintf(SystemComponents.OUTPUT, "== RUNNING EQUILIBRATION PHASE ==\n");  Cycles = Vars.NumberOfEquilibrationCycles; break;}
    case PRODUCTION:     {Mode = "PRODUCTION";     fprintf(SystemComponents.OUTPUT, "==  RUNNING PRODUCTION PHASE   ==\n");     Cycles = Vars.NumberOfProductionCycles; break;}
  }
  fprintf(SystemComponents.OUTPUT, "==================================\n");

  fprintf(SystemComponents.OUTPUT, "CBMC Uses %zu trial positions and %zu trial orientations\n", Vars.Widom[systemId].NumberWidomTrials, Vars.Widom[systemId].NumberWidomTrialsOrientations);

  if(SimulationMode == INITIALIZATION)
  {
    fprintf(SystemComponents.OUTPUT, "Box %zu, Volume: %.5f\n", systemId, Vars.Sims[systemId].Box.Volume);
    Vars.GibbsStatistics.TotalVolume += Vars.Sims[systemId].Box.Volume;

    fprintf(SystemComponents.OUTPUT, "Total Volume: %.5f\n", Vars.GibbsStatistics.TotalVolume);
  }
  // Kaihang Shi: Record initial energy but exclude the host-host Ewald
  SystemComponents.createmol_energy = SystemComponents.CreateMol_Energy.total() - SystemComponents.CreateMol_Energy.HHVDW - SystemComponents.CreateMol_Energy.HHEwaldE - SystemComponents.CreateMol_Energy.HHReal;

  if(SimulationMode == PRODUCTION)
  {
    BlockAverageSize = Cycles / SystemComponents.Nblock;
    if(Cycles % SystemComponents.Nblock != 0)
      fprintf(SystemComponents.OUTPUT, "Warning! Number of Cycles cannot be divided by Number of blocks. Residue values go to the last block\n");
    SystemComponents.BookKeepEnergy.resize(SystemComponents.Nblock);
    SystemComponents.BookKeepEnergy_SQ.resize(SystemComponents.Nblock);
    //Initialize vectors for energy * N for each component//
    //initialize for each component, start with zero//
    std::vector<double>FILL(SystemComponents.Nblock, 0.0);
    SystemComponents.EnergyTimesNumberOfMolecule.resize(SystemComponents.NComponents.x, FILL);
    SystemComponents.VolumeAverage.resize(SystemComponents.Nblock, {0.0, 0.0});
    SystemComponents.DensityPerComponent.resize(SystemComponents.NComponents.x, std::vector<double2>(SystemComponents.Nblock, {0.0, 0.0}));
    if(SystemComponents.AmountOfExcessMolecules.size() > 0)
      SystemComponents.ExcessLoading.resize(SystemComponents.NComponents.x, std::vector<double2>(SystemComponents.Nblock, {0.0, 0.0}));
  }

  /////////////////////////////////////////////
  // FINALIZE (PRODUCTION) CBCF BIASING TERM //
  /////////////////////////////////////////////
  if(SimulationMode == PRODUCTION)
  {
    for(size_t icomp = 0; icomp < SystemComponents.NComponents.x; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp] && !SystemComponents.Tmmc[icomp].DoTMMC)
        Finalize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }

  ///////////////////////////////////////////////////////////////////////
  // FORCE INITIALIZING CBCF BIASING TERM BEFORE INITIALIZATION CYCLES //
  ///////////////////////////////////////////////////////////////////////
  if(SimulationMode == INITIALIZATION && Cycles > 0)
  {
    for(size_t icomp = 0; icomp < SystemComponents.NComponents.x; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp])
        Initialize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }
  
  if(SimulationMode == EQUILIBRATION) //Rezero the TMMC stats at the beginning of the Equilibration cycles//
  {
    for(size_t comp = 0; comp < SystemComponents.NComponents.x; comp++)
    {
      //Clear TMMC data in the collection matrix//
      SystemComponents.Tmmc[comp].ClearCMatrix();
      //Clear Rosenbluth weight statistics after Initialization//
      for(size_t i = 0; i < SystemComponents.Nblock; i++)
        SystemComponents.Moves[comp].ClearRosen(i);
    }
  }
}

inline void MCEndOfPhaseSummary(Variables& Vars)
{
  std::vector<Components>& SystemComponents = Vars.SystemComponents;
  Simulations*&  Sims   = Vars.Sims;
  Units& Constants = Vars.Constants;
  std::string& Mode = Vars.Mode;

  size_t NumberOfSimulations = SystemComponents.size();
  int& Cycles = Vars.Cycles;

  //print statistics
  if(Cycles > 0)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      if(Vars.SimulationMode == EQUILIBRATION) fprintf(SystemComponents[sim].OUTPUT, "Sampled %zu WangLandau, Adjusted WL %zu times\n", SystemComponents[sim].WLSampled, SystemComponents[sim].WLAdjusted);
      PrintAllStatistics(SystemComponents[sim], Sims[sim], Cycles, Vars.SimulationMode, Vars.BlockAverageSize, Constants);
      if(Vars.SimulationMode == PRODUCTION)
        Calculate_Overall_Averages_MoveEnergy(SystemComponents[sim], Vars.BlockAverageSize, Cycles);
    }
    PrintSystemMoves(Vars);
  }
  for(size_t i = 0; i < Vars.SystemComponents.size(); i++)
  {
    fprintf(SystemComponents[i].OUTPUT, "===============================\n");
    fprintf(SystemComponents[i].OUTPUT, "== %s PHASE ENDS ==\n", Mode.c_str());
    fprintf(SystemComponents[i].OUTPUT, "===============================\n");
  }
}
//Default is 20 steps per cycle//
//If # of molecules > 20, use # of molecules//
//If a max limit is imposed, use the max limit if it exceeds//
size_t Determine_Number_Of_Steps(Variables& Vars, size_t systemId, size_t current_cycle)
{ 
  //Record current step//
  Vars.SystemComponents[systemId].CURRENTCYCLE = current_cycle;
  size_t Steps = 20;
  if(Steps < Vars.SystemComponents[systemId].TotalNumberOfMolecules)
  {
    Steps = Vars.SystemComponents[systemId].TotalNumberOfMolecules;
  }
  if(Vars.SetMaxStep && Steps > Vars.MaxStepPerCycle) Steps = Vars.MaxStepPerCycle;
  return Steps;
}

void Run_Simulation_MultipleBoxes(Variables& Vars)
{
  std::vector<Components>&   SystemComponents = Vars.SystemComponents;
  size_t NumberOfSimulations = SystemComponents.size();

  for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    InitialMCBeforeMoves(Vars, sim);

  ///////////////////////////////////////////////////////
  // Run the simulations for different boxes IN SERIAL //
  ///////////////////////////////////////////////////////
  for(size_t i = 0; i < Vars.Cycles; i++)
  {
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      double RNM = Get_Uniform_Random();
      size_t selectedSim = static_cast<size_t>(RNM * static_cast<double>(NumberOfSimulations));
      size_t Steps = Determine_Number_Of_Steps(Vars, selectedSim, i);
      //printf("STEPS: %zu, RNM: %.5f, selectedSim: %zu\n", Steps, RNM, selectedSim);
      for(size_t j = 0; j < Steps; j++)
      {
        RunMoves(Vars, selectedSim, i);
      }
    }
    for(size_t sim = 0; sim < NumberOfSimulations; sim++)
    {
      GatherStatisticsDuringSimulation(Vars, sim, i);
    }
    if(i > 0 && i % 500 == 0)
      Update_Max_GibbsVolume(Vars.GibbsStatistics);
  }
  MCEndOfPhaseSummary(Vars);
}

void Run_Simulation_ForOneBox(Variables& Vars, size_t box_index)
{
  InitialMCBeforeMoves(Vars, box_index);

  for(size_t i = 0; i < Vars.Cycles; i++)
  {
    size_t Steps = Determine_Number_Of_Steps(Vars, box_index, i);
    for(size_t j = 0; j < Steps; j++)
    {
      RunMoves(Vars, box_index, i);
    }
    GatherStatisticsDuringSimulation(Vars, box_index, i);
  }
  MCEndOfPhaseSummary(Vars);
}
