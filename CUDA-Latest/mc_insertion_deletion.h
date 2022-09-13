#include "mc_widom.h"

__global__ void AllocateMoreSpace_CopyToTemp(Atoms* d_a, Atoms temp, size_t Space, size_t SelectedComponent)
{
  for(size_t i = 0; i < Space; i++)
  {
    temp.x[i]         = d_a[SelectedComponent].x[i];
    temp.y[i]         = d_a[SelectedComponent].y[i];
    temp.z[i]         = d_a[SelectedComponent].z[i];
    temp.scale[i]     = d_a[SelectedComponent].scale[i];
    temp.charge[i]    = d_a[SelectedComponent].charge[i];
    temp.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[i];
    temp.Type[i]      = d_a[SelectedComponent].Type[i];
    temp.MolID[i]     = d_a[SelectedComponent].MolID[i];
  }
}

__global__ void AllocateMoreSpace_CopyBack(Atoms* d_a, Atoms System_comp, Atoms temp, size_t Space, size_t Newspace, size_t SelectedComponent)
{
  d_a[SelectedComponent].Allocate_size = Newspace;
  for(size_t i = 0; i < Space; i++)
  {
    d_a[SelectedComponent].x[i]         = temp.x[i];
    d_a[SelectedComponent].y[i]         = temp.y[i];
    d_a[SelectedComponent].z[i]         = temp.z[i];
    d_a[SelectedComponent].scale[i]     = temp.scale[i];
    d_a[SelectedComponent].charge[i]    = temp.charge[i];
    d_a[SelectedComponent].scaleCoul[i] = temp.scaleCoul[i];
    d_a[SelectedComponent].Type[i]      = temp.Type[i];
    d_a[SelectedComponent].MolID[i]     = temp.MolID[i];
  
    System_comp.x[i]         = temp.x[i];
    System_comp.y[i]         = temp.y[i];
    System_comp.z[i]         = temp.z[i];
    System_comp.scale[i]     = temp.scale[i];
    System_comp.charge[i]    = temp.charge[i];
    System_comp.scaleCoul[i] = temp.scaleCoul[i];
    System_comp.Type[i]      = temp.Type[i];
    System_comp.MolID[i]     = temp.MolID[i];
  }
  //test the new allocation//
  d_a[SelectedComponent].x[Newspace-1] = 0.0;
  /*if(d_a[SelectedComponent].x[Newspace-1] = 0.0)
  {
    printf("Copy works\n");
  }*/
}



void AllocateMoreSpace(Atoms*& System, Atoms*& d_a, size_t SelectedComponent)
{
  Atoms temp; // allocate a struct on the device for copying data.
  size_t Copysize=System[SelectedComponent].Allocate_size;
  size_t Morespace = 1024;
  size_t Newspace = Copysize+Morespace;
  //Allocate space on the temporary struct
  cudaMalloc(&temp.x, Copysize * sizeof(double));
  cudaMalloc(&temp.y, Copysize * sizeof(double));
  cudaMalloc(&temp.z, Copysize * sizeof(double));
  cudaMalloc(&temp.scale, Copysize * sizeof(double));
  cudaMalloc(&temp.charge, Copysize * sizeof(double));
  cudaMalloc(&temp.scaleCoul, Copysize * sizeof(double));
  cudaMalloc(&temp.Type, Copysize * sizeof(size_t));
  cudaMalloc(&temp.MolID, Copysize * sizeof(size_t));
  // Copy data to temp
  AllocateMoreSpace_CopyToTemp<<<1,1>>>(d_a, temp, Copysize, SelectedComponent);
  // Allocate more space on the device pointers
  cudaMalloc(&System[SelectedComponent].x, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].y, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].z, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].scale, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].charge, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].scaleCoul, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].Type, Newspace * sizeof(size_t));
  cudaMalloc(&System[SelectedComponent].MolID, Newspace * sizeof(size_t));
  // Copy data from temp back to the new pointers
  AllocateMoreSpace_CopyBack<<<1,1>>>(d_a, System[SelectedComponent], temp, Copysize, Newspace, SelectedComponent); 
  System[SelectedComponent].Allocate_size = Newspace;
}

double Insertion(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent);

void Update_NumberOfMolecules(Components& SystemComponents, Atoms*& System, Atoms*& d_a, size_t SelectedComponent, bool Insertion);

static inline void Update_NumberOfMolecules(Components& SystemComponents, Atoms*& System, Atoms*& d_a, size_t SelectedComponent, bool Insertion)
{
  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //change in atom number counts
  int NumMol = -1; //default: deletion; Insertion: +1, Deletion: -1, size_t is never negative
  if(Insertion) NumMol = 1;
  //Update Components
  SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] += NumMol;
  SystemComponents.TotalNumberOfMolecules += NumMol;
  //if(!Insertion) printf("After deletion: Num: %zu\n", SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  //Update Atoms
  if(Insertion){
    System[SelectedComponent].size += Molsize;} //size is the number of atoms for one component, so update with number of atoms in a molecule
  else //deletion
    {System[SelectedComponent].size -= Molsize;}
  
  //check size
  if(System[SelectedComponent].size > System[SelectedComponent].Allocate_size)
  {
    //printf("Trying to allocate more space\n");
    AllocateMoreSpace(System, d_a, SelectedComponent);
    //throw std::runtime_error("Need to allocate more space, not implemented\n");
  }
}

__global__ void Update_insertion_data_firstbead(Atoms* d_a, Atoms NewMol, size_t SelectedTrial, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  Atoms AllData = d_a[SelectedComponent];
  //UpdateLocation should be the last position of the dataset
  //Need to check if Allocate_size is smaller than size
  d_a[SelectedComponent].x[UpdateLocation]         = NewMol.x[SelectedTrial];
  d_a[SelectedComponent].y[UpdateLocation]         = NewMol.y[SelectedTrial];
  d_a[SelectedComponent].z[UpdateLocation]         = NewMol.z[SelectedTrial];
  d_a[SelectedComponent].scale[UpdateLocation]     = NewMol.scale[SelectedTrial];
  d_a[SelectedComponent].charge[UpdateLocation]    = NewMol.charge[SelectedTrial];
  d_a[SelectedComponent].scaleCoul[UpdateLocation] = NewMol.scaleCoul[SelectedTrial];
  d_a[SelectedComponent].Type[UpdateLocation]      = NewMol.Type[SelectedTrial];
  d_a[SelectedComponent].MolID[UpdateLocation]     = NewMol.MolID[SelectedTrial];
  //printf("NewMol data: %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %lu, %lu\n", NewMol.x[SelectedTrial], NewMol.y[SelectedTrial], NewMol.z[SelectedTrial], NewMol.scale[SelectedTrial], NewMol.charge[SelectedTrial], NewMol.scaleCoul[SelectedTrial], NewMol.Type[SelectedTrial], NewMol.MolID[SelectedTrial]);
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  += Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
    //printf("x: %.10f, Alldata.size: %lu, d_a.size: %lu\n", AllData.x[UpdateLocation], AllData.size, d_a[SelectedComponent].size);
  }
}

static inline double Insertion(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  Widom.InsertionTotal ++;
  bool Insertion = true;
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
  double preFactor = FF.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume / (1.0+NumberOfMolecules);
  double RRR = get_random_from_zero_to_one();
  //printf("RRR: %.10f, prefactor: %.10f, Rosenbluth: %.10f, idealrosen: %.10f\n", RRR, preFactor, Rosenbluth, IdealRosen);
  if(RRR < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    Widom.InsertionAccepted ++;
    size_t UpdateLocation = System[SelectedComponent].size;
    //printf("Accepted, UpdateLocation: %zu\n", UpdateLocation);
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data_firstbead<<<1,1>>>(d_a, NewMol, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    Update_NumberOfMolecules(SystemComponents, System, d_a, SelectedComponent, Insertion);
    return energy;
    //CHECK_d_a<<<1,1>>>(d_a, System[1], UpdateLocation);
  }
  return 0.0;
}
