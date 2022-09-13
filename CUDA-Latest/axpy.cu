#include "axpy.h"
#include "print_statistics.h"
#include "mc_translation.h"
#include "mc_insertion_deletion.h"
//#include "write_lmp_movie.h"
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>

__global__ void rezero_result_array(double* x)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] = 0.0;
}

__global__ void Update_deletion_data_firstbead(Atoms* d_a, Atoms NewMol, size_t SelectedTrial, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  Atoms AllData = d_a[SelectedComponent];
  
  //UpdateLocation should be the molecule that needs to be deleted
  //Then move the atom at the last position to the location of the deleted molecule
  //**Zhao's note** MolID of the deleted molecule should not be changed
  //**Zhao's note** if Molecule deleted is the last molecule, then nothing is copied, just change the size later.
  if(UpdateLocation != LastLocation)
  {
    d_a[SelectedComponent].x[UpdateLocation]         = d_a[SelectedComponent].x[LastLocation];
    d_a[SelectedComponent].y[UpdateLocation]         = d_a[SelectedComponent].y[LastLocation];
    d_a[SelectedComponent].z[UpdateLocation]         = d_a[SelectedComponent].z[LastLocation];
    d_a[SelectedComponent].scale[UpdateLocation]     = d_a[SelectedComponent].scale[LastLocation];
    d_a[SelectedComponent].charge[UpdateLocation]    = d_a[SelectedComponent].charge[LastLocation];
    d_a[SelectedComponent].scaleCoul[UpdateLocation] = d_a[SelectedComponent].scaleCoul[LastLocation];
    d_a[SelectedComponent].Type[UpdateLocation]      = d_a[SelectedComponent].Type[LastLocation];
  }
  //printf("NewMol data: %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %lu, %lu\n", NewMol.x[SelectedTrial], NewMol.y[SelectedTrial], NewMol.z[SelectedTrial], NewMol.scale[SelectedTrial], NewMol.charge[SelectedTrial], NewMol.scaleCoul[SelectedTrial], NewMol.Type[SelectedTrial], NewMol.MolID[SelectedTrial]);
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  -= Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
    //printf("x: %.10f, Alldata.size: %lu, d_a.size: %lu\n", AllData.x[UpdateLocation], AllData.size, d_a[SelectedComponent].size);
  }
}
__global__ void CHECK_d_a(Atoms* d_a, Atoms Sys, size_t location)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  //d_a[1].size = Sys.size;
  printf("d_a size value: %lu, Sys size: %lu\n", d_a[1].size, Sys.size);
  printf("d_a pos: %.10f, Sys pos: %.10f\n", d_a[1].x[location], Sys.x[location]);
}

static inline double Deletion(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //if(System[SelectedComponent].size == 0){printf("No Molecule to delete\n"); return 0.0;} //cannot delete molecule when there is no molecule
  Widom.DeletionTotal ++;
  bool Insertion = false; 
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0;
  double Rosenbluth=Widom_Move_FirstBead(Box, SystemComponents, System, d_a, NewMol, FF, MoveStats, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, &SelectedTrial, &SuccessConstruction, &energy);
  if(!SuccessConstruction)
    return 0.0;
  //printf("Selected Trial is %zu, Rosenbluth is %.10f\n", SelectedTrial, Rosenbluth);
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  //printf("MolFraction: %.10f, IdealRosen: %.10f, Fugacoeff: %.10f, NumMol: %.10f, Pressure: %.10f\n", MolFraction, IdealRosen, FugacityCoefficient, NumberOfMolecules, Box.Pressure);
  double preFactor = (NumberOfMolecules) / (FF.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume);
  double RRR = get_random_from_zero_to_one();
  //printf("RRR: %.10f, prefactor: %.10f, Rosenbluth: %.10f, idealrosen: %.10f\n", RRR, preFactor, Rosenbluth, IdealRosen);
  //if(RRR < preFactor * IdealRosen / Rosenbluth)
  if(RRR < preFactor * IdealRosen / Rosenbluth) // for bebug: always accept
  { // accept the move
    Widom.DeletionAccepted ++;
    size_t UpdateLocation = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    // Get the starting position of the last molecule in the array
    size_t LastLocation = (SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1)*SystemComponents.Moleculesize[SelectedComponent];
    //printf("Accepted, UpdateLocation: %zu, LastLocation: %zu, energy: %.10f\n", UpdateLocation, LastLocation, energy);
    //if(UpdateLocation == LastLocation){printf("Deleting the LAST molecule\n");}
    Update_deletion_data_firstbead<<<1,1>>>(d_a, NewMol, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
    Update_NumberOfMolecules(SystemComponents, System, d_a, SelectedComponent, Insertion);
    return -energy;
    //CHECK_d_a<<<1,1>>>(d_a, System[1], UpdateLocation);
  }
  return 0.0;
}

double cuSoA(int Cycles, Components SystemComponents, Boxsize Box, Atoms* System, Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, double* y, double* dUdlambda, RandomNumber Random, Move_Statistics MoveStats, WidomStruct Widom, Units Constants, double init_energy)
{
  
  double tot = 0.0;

  double running_energy = 0.0;

  size_t WidomCount = 0;

  bool DEBUG = false;
  size_t transCount=0;
  printf("There are %zu Molecules, %zu Frameworks\n",SystemComponents.TotalNumberOfMolecules, SystemComponents.NumberOfFrameworks);

  for(size_t i = 0; i < Cycles; i++)
  {
    //printf("CYCLE: %zu\n", i);
    //Randomly Select an Adsorbate Molecule and determine its Component: MoleculeID --> Component
    //if((SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks) == 0)
    //  continue;
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

    //printf("Selected Comp: %zu, SelectedMol: %zu, Num: %zu\n", comp, SelectedMolInComponent, SystemComponents.NumberOfMolecule_for_Component[comp]);
    if(SystemComponents.NumberOfMolecule_for_Component[comp] == 0){ //no molecule in the system for this species
      //printf("Doing insertion since there is no molecule for this species\n");
      running_energy += Insertion(Box, SystemComponents, System, d_a, NewMol, FF, MoveStats, Random, Widom, SelectedMolInComponent, comp);
      continue;
    }

    double RANDOMNUMBER = get_random_from_zero_to_one();
    if(RANDOMNUMBER < MoveStats.TranslationProb)
    {
      transCount++;
      //PERFORM TRANSLATION MOVE//
      running_energy += Translation_Move(Box, SystemComponents, System, d_a, Mol, NewMol, FF, MoveStats, y, dUdlambda, Random, SelectedMolInComponent, comp);
      if(DEBUG){printf("After Translation: running energy: %.10f\n", running_energy);}
    }
    else if(RANDOMNUMBER < MoveStats.WidomProb + MoveStats.TranslationProb)
    {
      WidomCount ++;
      //printf("Performing Widom\n");
      size_t SelectedTrial=0; bool SuccessConstruction = false; double energy = 0.0;
      double Rosenbluth=Widom_Move_FirstBead(Box, SystemComponents, System, d_a, NewMol, FF, MoveStats, Random, Widom, SelectedMolInComponent, comp, Insertion, &SelectedTrial, &SuccessConstruction, &energy);
      //Assume 5 blocks
      size_t BlockIDX = i/(Cycles/MoveStats.NumberOfBlocks); //printf("BlockIDX=%zu\n", BlockIDX);
      Widom.Rosenbluth[BlockIDX]+= Rosenbluth;
      Widom.RosenbluthSquared[BlockIDX]+= Rosenbluth*Rosenbluth;
      Widom.RosenbluthCount[BlockIDX]++;
    }
    else
    {
      // DO GCMC INSERTION //
      //if((DEBUG) && (i < 20)){
      if(get_random_from_zero_to_one() < 0.5){
        running_energy += Insertion(Box, SystemComponents, System, d_a, NewMol, FF, MoveStats, Random, Widom, SelectedMolInComponent, comp);}
      else{
        if(DEBUG){printf("Cycle: %zu, DOING DELETION\n", i);}
        running_energy += Deletion(Box, SystemComponents, System, d_a, NewMol, FF, MoveStats, Random, Widom, SelectedMolInComponent, comp);}
    }
    if(i%500==0 &&(MoveStats.TranslationTotal > 0))
    {
      printf("i: %zu\n", i);
      Update_Max_Translation(FF, MoveStats);
    }
    if(DEBUG)
    {
      printf("After MOVE: Sum energies\n");
      double* xxx; xxx = (double*) malloc(sizeof(double)*2);
      double* device_xxx = CUDA_copy_allocate_double_array(xxx, 2);
      one_thread_GPU_test<<<1,1>>>(Box, d_a, FF, device_xxx); cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
      printf("Current Total Energy (1 thread GPU): %.10f, running total: %.10f\n", xxx[0], init_energy+running_energy);
      cudaDeviceSynchronize();
      if(abs(xxx[0] - (init_energy+running_energy)) > 0.1) //means that there is an energy drift
      {
        printf("THere is an energy drift at cycle %zu\n", i);
      }
    }
  }
  //print statistics
  Print_Translation_Statistics(MoveStats, FF);
  Print_Widom_Statistics(Widom, MoveStats, FF, Constants.energy_to_kelvin);
  Print_Swap_Statistics(Widom, MoveStats);
  printf("TransCount: %zu\n", transCount);
  printf("total-deltaU: %.10f\n", running_energy);
  //Generate Movie
  return running_energy;
}
