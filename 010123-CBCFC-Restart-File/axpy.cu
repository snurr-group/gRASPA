#include "axpy.h"
#include "print_statistics.h"
#include "mc_translation.h"
#include "mc_insertion_deletion.h"
#include "Ewald.h"
//#include "lambda.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>

__global__ void get_random_trial_position_firstbead_MultiSims(Boxsize Box, Simulations* Sims, double* random, size_t offset, size_t SelectedComponent, size_t chainsize)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = i*3*chainsize + offset;
  const Atoms AllData = Sims[i].d_a[SelectedComponent];
  const size_t real_pos = Sims[i].start_position;
  const double3 Max = Sims[i].MaxTranslation;
  for(size_t j = 0; j < chainsize; j++)
  {  
    size_t random_ij       = random_i + j * chainsize;
    const double x         = AllData.x[real_pos];
    const double y         = AllData.y[real_pos];
    const double z         = AllData.z[real_pos];
    const double scale     = AllData.scale[real_pos];
    const double charge    = AllData.charge[real_pos];
    const double scaleCoul = AllData.scaleCoul[real_pos];
    const size_t Type      = AllData.Type[real_pos];
    const size_t MolID     = AllData.MolID[real_pos];
    Sims[i].Old.x[j] = x; Sims[i].Old.y[j] = y; Sims[i].Old.z[j] = z;
    Sims[i].New.x[j] = x + Max.x * 2.0 * (random[random_ij] - 0.5);
    Sims[i].New.y[j] = y + Max.y * 2.0 * (random[random_ij+1] - 0.5);
    Sims[i].New.z[j] = z + Max.z * 2.0 * (random[random_ij+2] - 0.5);
    Sims[i].Old.scale[j] = scale; Sims[i].Old.charge[j] = charge; Sims[i].Old.scaleCoul[j] = scaleCoul; Sims[i].Old.Type[j] = Type; Sims[i].Old.MolID[j] = MolID;
    Sims[i].New.scale[j] = scale; Sims[i].New.charge[j] = charge; Sims[i].New.scaleCoul[j] = scaleCoul; Sims[i].New.Type[j] = Type; Sims[i].New.MolID[j] = MolID;
    //printf("RandomPos: Sim: %lu, Old xyz: %.5f %.5f %.5f, New xyz: %.5f %.5f %.5f\n", i, Sims[i].Old.x[j], Sims[i].Old.y[j], Sims[i].Old.z[j], Sims[i].New.x[j], Sims[i].New.y[j], Sims[i].New.z[j]);
  }
  Sims[i].device_flag[0] = false; // Single Trial, Set the first element of flags to false //
  Sims[i].AcceptedFlag   = false;
}

static inline void SelectMoleculeInComponent(Components& SystemComponents, size_t SelectedComponent, size_t *SelectedMolInComponent)
{
  size_t TempMolecule = (size_t) (get_random_from_zero_to_one()* SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  *SelectedMolInComponent = TempMolecule;
}

static inline size_t CountAtomInSimulation(Components& SystemComponents)
{
  size_t Tempsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Tempsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  return Tempsize;
}

static inline double Energy_Sum(Simulations Sims, Components& SystemComponents)
{
  double Sum = 0.0;
  size_t NBlocks = Sims.Nblocks; double Energies[NBlocks];
  cudaMemcpy(Energies, Sims.Blocksum, NBlocks * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < NBlocks; i++) Sum += Energies[i];
  return Sum;
}

__global__ void update_translation_position_MULTIPLE(Simulations* Sims, size_t SelectedComponent, size_t chainsize)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("updating, i: %lu\n", i);
  if(Sims[i].AcceptedFlag)
  {
    size_t start_pos = Sims[i].start_position;
    for(size_t a = 0; a < chainsize; a++)
    {
      //printf("i: %lu, startpos: %lu, Old pos: %.5f %.5f %.5f, d_a pos: %.5f %.5f %.5f\n", i, start_pos, Sims[i].Old.x[a], Sims[i].Old.y[a], Sims[i].Old.z[a], Sims[i].d_a[SelectedComponent].x[start_pos+a], Sims[i].d_a[SelectedComponent].y[start_pos+a], Sims[i].d_a[SelectedComponent].z[start_pos+a]);
 
      Sims[i].d_a[SelectedComponent].x[start_pos+a] = Sims[i].New.x[a];
      Sims[i].d_a[SelectedComponent].y[start_pos+a] = Sims[i].New.y[a];
      Sims[i].d_a[SelectedComponent].z[start_pos+a] = Sims[i].New.z[a];
      
      //printf("i: %lu, startpos: %lu, New: %.5f %.5f %.5f, updated xyz: %.5f %.5f %.5f\n", i, start_pos, Sims[i].New.x[a], Sims[i].New.y[a], Sims[i].New.z[a], Sims[i].d_a[SelectedComponent].x[start_pos+a], Sims[i].d_a[SelectedComponent].y[start_pos+a], Sims[i].d_a[SelectedComponent].z[start_pos+a]);
    }
  }
}

double Multiple_Sims(int Cycles, std::vector<Components>& SystemComponents, Boxsize Box, Simulations* Sims, ForceField FF, RandomNumber Random, std::vector<WidomStruct>& WidomArray, Units Constants, std::vector<double>& init_energy)
{
  // Try to Get Multiple Random Positions for Translation //
  size_t SelectedComponent = 1;
  size_t NumberOfSimulations = WidomArray.size();
  std::vector<size_t> SelectedMolecules(NumberOfSimulations);
  std::vector<double> Energies(NumberOfSimulations);
  for(size_t cycle = 0; cycle < Cycles; cycle++)
  {
  //printf("===================================\n");
  if(cycle % 500 == 0) printf("cycle: %zu\n", cycle);
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    SelectMoleculeInComponent(SystemComponents[i], SelectedComponent, &SelectedMolecules[i]);
    //printf("Selected Molecule for Sim is %zu\n", SelectedMolecules[i]);
    Sims[i].start_position = SelectedMolecules[i] * SystemComponents[i].Moleculesize[SelectedComponent];
  }
  // REAL MC TRANSLATION MOVE //
  size_t  chainsize = SystemComponents[0].Moleculesize[SelectedComponent];
  //size_t* Atomsize; cudaMallocManaged(&Atomsize, NumberOfSimulations * sizeof(size_t));
  std::vector<size_t> SimBlocks(NumberOfSimulations); std::vector<size_t> SimThreads(NumberOfSimulations);
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    SystemComponents[i].Moves[SelectedComponent].TranslationTotal ++;
    Sims[i].TotalAtoms = CountAtomInSimulation(SystemComponents[i]);
    // Zhao's note: because different simulations have different number of atoms, the number of blocks for every simulation is different //
    Setup_threadblock(Sims[i].TotalAtoms * chainsize, &SimBlocks[i], &SimThreads[i]);
    //printf("Sim %zu, Block: %zu, Thread: %zu\n", i, SimBlocks[i], SimThreads[i]);
    Sims[i].Nblocks = SimBlocks[i];
    //Zhao's note: later, we are going to assume that every simulation has the same amount of thread per block, only the number of blocks is different for every simulation//
  }
  size_t totalblocks = std::accumulate(SimBlocks.begin(), SimBlocks.end(), decltype(SimBlocks)::value_type(0));
  size_t maxthread   = DEFAULTTHREAD; //Zhao's note: this is kind of lazy to assume they all have DEFAULTTHREAD, correct way is to use a std::max_element//
  //size_t offset = 0;
  if((Random.offset + 3*chainsize*NumberOfSimulations) >= Random.randomsize) Random.offset = 0;
  get_random_trial_position_firstbead_MultiSims<<<1,NumberOfSimulations>>>(Box, Sims, Random.device_random, Random.offset, SelectedComponent, chainsize);
  Random.offset += 3*chainsize*NumberOfSimulations;
  Energy_difference_PARTIAL_MULTIPLE<<<totalblocks, maxthread, maxthread*sizeof(double)>>>(Box, Sims, FF, SelectedComponent, chainsize, NumberOfSimulations);
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    double tot = Energy_Sum(Sims[i], SystemComponents[i]);
    //printf("tot is %.10f for Simulation %zu\n", tot, i);
    if(get_random_from_zero_to_one() < std::exp(-SystemComponents[i].Beta * tot))
    {
      Sims[i].AcceptedFlag = true;
      SystemComponents[i].Moves[SelectedComponent].TranslationAccepted ++;
      Energies[i] += tot;
    }
  }
  // Update Atom position data for those that are accepted //
  update_translation_position_MULTIPLE<<<1,NumberOfSimulations>>>(Sims, SelectedComponent, chainsize);
  cudaDeviceSynchronize();
  }
  for(size_t i = 0; i < NumberOfSimulations; i++)
    printf("Simulation %zu, running total: %.10f\n", i, init_energy[i]+Energies[i]);
}

double Run_Simulation(int Cycles, Components& SystemComponents, Boxsize Box, Simulations Sims, ForceField FF, RandomNumber Random, WidomStruct Widom, Units Constants, double& init_energy, bool DualPrecision, std::vector<size_t>& NumberOfCreateMolecules, int SimulationMode)
{
  if(SimulationMode == INITIALIZATION)
  {
    SystemComponents.deltaVDWReal = 0.0;
    SystemComponents.deltaEwald   = 0.0;
  }
  double running_energy = 0.0;

  size_t WidomCount = 0;

  bool DEBUG = false;

  // Create Molecules in the Box Before the Simulation //
  if(SimulationMode == CREATE_MOLECULE)
  {
    size_t CreateFailCount = 0; size_t Created = 0; size_t SelectedMol = 0;
    for(size_t comp = 1; comp < SystemComponents.Total_Components; comp++)
    {
      CreateFailCount = 0;
      printf("Component %zu, Need to create %zu full molecule\n", comp, NumberOfCreateMolecules[comp]);
      //Create Fractional Molecule first//
      if(SystemComponents.hasfractionalMolecule[comp])
      {
        //Zhao's note: If we need to create fractional molecule, then we initialize WangLandau Histogram//
        Initialize_WangLandauIteration(SystemComponents.Lambda[comp]);
        size_t FractionalMolToCreate = 1;
        while(FractionalMolToCreate > 0)
        {
          printf("Creating Fractional Molecule for Component %zu; There are %zu Molecules of that component in the System\n", comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
          SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
          size_t OldVal = SystemComponents.NumberOfMolecule_for_Component[comp];

          size_t NewBin = 5;
          double newLambda = static_cast<double>(NewBin) * SystemComponents.Lambda[comp].delta;
          double2 newScale = setScale(newLambda);
          running_energy  += CreateMolecule(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, DualPrecision, newScale);
          if(SystemComponents.NumberOfMolecule_for_Component[comp] == OldVal)
          {
            CreateFailCount ++;
          } 
          else 
          {
            FractionalMolToCreate --; Created ++; SystemComponents.Lambda[comp].FractionalMoleculeID = SelectedMol; 
            SystemComponents.Lambda[comp].currentBin = NewBin;
          }
          if(CreateFailCount > 10) throw std::runtime_error("Bad Insertions When Creating Fractional Molecules!");
        }
      }
      while(NumberOfCreateMolecules[comp] > 0)
      {
        printf("Creating %zu Molecule for Component %zu; There are %zu Molecules of that component in the System\n", Created, comp, SystemComponents.NumberOfMolecule_for_Component[comp]);
        SelectedMol = Created; if(Created > 0) SelectedMol = Created - 1; //Zhao's note: this is a little confusing, but when number of molecule for that species = 0 or 1, the chosen molecule is zero. This is creating from zero loading, need to change in the future, when we read from restart file//
        size_t OldVal    = SystemComponents.NumberOfMolecule_for_Component[comp];
        double2 newScale = setScale(1.0); //Set scale for full molecule (lambda = 1.0)//
        running_energy  += CreateMolecule(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMol, comp, DualPrecision, newScale);
        if(SystemComponents.NumberOfMolecule_for_Component[comp] == OldVal)
        {CreateFailCount ++;} else {NumberOfCreateMolecules[comp] --; Created ++;}
        if(CreateFailCount > 10) throw std::runtime_error("Bad Insertions When Creating Molecules!");
      }
    }
    return running_energy;
  }
  //printf("There are %zu Molecules, %zu Frameworks\n",SystemComponents.TotalNumberOfMolecules, SystemComponents.NumberOfFrameworks);

  /////////////////////////////////////////////
  // FINALIZE (PRODUCTION) CBCF BIASING TERM //
  /////////////////////////////////////////////
  if(SimulationMode == PRODUCTION)
  {
    for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
      if(SystemComponents.hasfractionalMolecule[icomp])
        Finalize_WangLandauIteration(SystemComponents.Lambda[icomp]);
  }

  for(size_t i = 0; i < Cycles; i++)
  {
    //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
    //Zhao's note: The number of atoms can be vulnerable, adding throw error here//
    if(SystemComponents.TotalNumberOfMolecules < SystemComponents.NumberOfFrameworks) 
      throw std::runtime_error("There is negative number of adsorbates. Break program!");
    size_t SelectedMolecule = (size_t) (get_random_from_zero_to_one()*(SystemComponents.TotalNumberOfMolecules-SystemComponents.NumberOfFrameworks));
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

    //printf("Selected Comp: %zu, SelectedMol: %zu, Num: %zu, Original_selected Molecule: %zu, TotalMol: %zu\n", comp, SelectedMolInComponent, SystemComponents.NumberOfMolecule_for_Component[comp], SelectedMolecule, SystemComponents.TotalNumberOfMolecules);
    if(SystemComponents.NumberOfMolecule_for_Component[comp] == 0)
    { //no molecule in the system for this species
      running_energy += Insertion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
      continue;
    }

    double RANDOMNUMBER = get_random_from_zero_to_one();
    if(RANDOMNUMBER < SystemComponents.Moves[comp].TranslationProb)
    {
      //////////////////////////////
      // PERFORM TRANSLATION MOVE //
      //////////////////////////////
      running_energy += Translation_Move(Box, SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp);
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].RotationProb) //Rotation
    {
      ///////////////////////////
      // PERFORM ROTATION MOVE //
      ///////////////////////////
      running_energy += Rotation_Move(Box, SystemComponents, Sims, Widom, FF, Random, SelectedMolInComponent, comp);
      //printf("After Translation: running energy: %.10f\n", running_energy);
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].WidomProb)
    {
      //////////////////////////////////
      // PERFORM WIDOM INSERTION MOVE //
      //////////////////////////////////
      WidomCount ++;
      double2 newScale = setScale(1.0); //Set scale for full molecule (lambda = 1.0)//
      size_t SelectedTrial=0; bool SuccessConstruction = false; double energy = 0.0; double StoredR = 0.0;
      double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, CBMC_INSERTION, StoredR, &SelectedTrial, &SuccessConstruction, &energy, false, newScale);
      if(SystemComponents.Moleculesize[comp] > 1 && Rosenbluth > 1e-150)
      {
        size_t SelectedFirstBeadTrial = SelectedTrial; 
        Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, CBMC_INSERTION, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, false, newScale); //false is for using Dual-Precision. For Widom Insertion, don't use it.//
      }
      //Assume 5 blocks
      size_t BlockIDX = i/(Cycles/SystemComponents.Moves[comp].NumberOfBlocks); //printf("BlockIDX=%zu\n", BlockIDX);
      Widom.Rosenbluth[BlockIDX]+= Rosenbluth;
      Widom.RosenbluthSquared[BlockIDX]+= Rosenbluth*Rosenbluth;
      Widom.RosenbluthCount[BlockIDX]++;
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].ReinsertionProb)
    {
      //////////////////////////////
      // PERFORM REINSERTION MOVE //
      //////////////////////////////
      running_energy += Reinsertion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
    }
    else if(RANDOMNUMBER < SystemComponents.Moves[comp].CBCFProb && SystemComponents.hasfractionalMolecule[comp])
    {
      ///////////////////////
      // PERFORM CBCF MOVE //
      ///////////////////////
      SelectedMolInComponent = SystemComponents.Lambda[comp].FractionalMoleculeID;
      running_energy += CBCFMove(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, false);
    }
    else
    {
      ////////////////////////////
      // PERFORM GCMC INSERTION //
      ////////////////////////////
      if(get_random_from_zero_to_one() < 0.5)
      { 
        running_energy += Insertion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
      }
      else
      {
        ///////////////////////////
        // PERFORM GCMC DELETION //
        ///////////////////////////
        //Zhao's note: Do not do a deletion if the chosen molecule is a fractional molecule, fractional molecules should go to CBCFSwap moves//
        if(!((SystemComponents.hasfractionalMolecule[comp]) && SelectedMolInComponent == SystemComponents.Lambda[comp].FractionalMoleculeID))
        {
          running_energy += Deletion(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, comp, DualPrecision);
        }
      }
    }
    //////////////////////////////////////////////
    // SAMPLE (EQUILIBRATION) CBCF BIASING TERM //
    //////////////////////////////////////////////
    if(SimulationMode == EQUILIBRATION)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
        if(SystemComponents.hasfractionalMolecule[icomp])
          Sample_WangLandauIteration(SystemComponents.Lambda[icomp]);
    }

    if(i%500==0 &&(SystemComponents.Moves[comp].TranslationTotal > 0))
    {
      printf("i: %zu\n", i);
      Update_Max_Translation(SystemComponents.Moves[comp], Sims);
    }
    ////////////////////////////////////////////////
    // ADJUST CBCF BIASING FACTOR (EQUILIBRATION) //
    ////////////////////////////////////////////////
    if(i%50==0 && SimulationMode == EQUILIBRATION)
    {
      for(size_t icomp = 0; icomp < SystemComponents.Total_Components; icomp++)
        if(SystemComponents.hasfractionalMolecule[icomp])
          Adjust_WangLandauIteration(SystemComponents.Lambda[icomp]);
    }
  }
  //print statistics
  for(size_t comp = SystemComponents.NumberOfFrameworks; comp < SystemComponents.Total_Components; comp++)
  {
    Print_Translation_Statistics(SystemComponents.Moves[comp], Sims.MaxTranslation);
    Print_Rotation_Statistics(SystemComponents.Moves[comp], Sims.MaxRotation);
    Print_Widom_Statistics(Widom, SystemComponents.Moves[comp], SystemComponents.Beta, Constants.energy_to_kelvin);
    Print_Swap_Statistics(Widom, SystemComponents.Moves[comp]);
    if(SystemComponents.hasfractionalMolecule[comp]) Print_CBCF_Statistics(SystemComponents.Moves[comp]);
    printf("running total: %.10f, DeltaVDW+Real: %.5f, DeltaEwald: %.5f\n", running_energy + init_energy, SystemComponents.deltaVDWReal, SystemComponents.deltaEwald);
  }
  return running_energy;
}
