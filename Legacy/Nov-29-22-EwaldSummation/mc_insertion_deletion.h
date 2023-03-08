#include "mc_widom.h"
//#include "ewald_kernel.h"
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

__global__ void AllocateMoreSpace_CopyBack(Atoms* d_a, Atoms temp, size_t Space, size_t Newspace, size_t SelectedComponent)
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
  /*
    System_comp.x[i]         = temp.x[i];
    System_comp.y[i]         = temp.y[i];
    System_comp.z[i]         = temp.z[i];
    System_comp.scale[i]     = temp.scale[i];
    System_comp.charge[i]    = temp.charge[i];
    System_comp.scaleCoul[i] = temp.scaleCoul[i];
    System_comp.Type[i]      = temp.Type[i];
    System_comp.MolID[i]     = temp.MolID[i];
  */
  }
  //test the new allocation//
  //d_a[SelectedComponent].x[Newspace-1] = 0.0;
  /*if(d_a[SelectedComponent].x[Newspace-1] = 0.0)
  {
    printf("Copy works\n");
  }*/
}



void AllocateMoreSpace(Atoms*& d_a, size_t SelectedComponent, Components& SystemComponents)
{
  printf("Allocating more space on device\n");
  Atoms temp; // allocate a struct on the device for copying data.
  Atoms tempSystem[SystemComponents.Total_Components];
  size_t Copysize=SystemComponents.Allocate_size[SelectedComponent];
  size_t Morespace = 1024;
  size_t Newspace = Copysize+Morespace;
  //Allocate space on the temporary struct
  cudaMalloc(&temp.x,         Copysize * sizeof(double));
  cudaMalloc(&temp.y,         Copysize * sizeof(double));
  cudaMalloc(&temp.z,         Copysize * sizeof(double));
  cudaMalloc(&temp.scale,     Copysize * sizeof(double));
  cudaMalloc(&temp.charge,    Copysize * sizeof(double));
  cudaMalloc(&temp.scaleCoul, Copysize * sizeof(double));
  cudaMalloc(&temp.Type,      Copysize * sizeof(size_t));
  cudaMalloc(&temp.MolID,     Copysize * sizeof(size_t));
  // Copy data to temp
  AllocateMoreSpace_CopyToTemp<<<1,1>>>(d_a, temp, Copysize, SelectedComponent);
  // Allocate more space on the device pointers

  Atoms System[SystemComponents.Total_Components]; cudaMemcpy(System, d_a, SystemComponents.Total_Components * sizeof(Atoms), cudaMemcpyDeviceToHost);

  cudaMalloc(&System[SelectedComponent].x,         Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].y,         Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].z,         Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].scale,     Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].charge,    Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].scaleCoul, Newspace * sizeof(double));
  cudaMalloc(&System[SelectedComponent].Type,      Newspace * sizeof(size_t));
  cudaMalloc(&System[SelectedComponent].MolID,     Newspace * sizeof(size_t));
  // Copy data from temp back to the new pointers
  AllocateMoreSpace_CopyBack<<<1,1>>>(d_a, temp, Copysize, Newspace, SelectedComponent);
  //System[SelectedComponent].Allocate_size = Newspace;
}

double Insertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision);

double Reinsertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision);

void Update_NumberOfMolecules(Components& SystemComponents, Atoms*& d_a, size_t SelectedComponent, bool Insertion);

double CreateMolecule(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision);

static inline void Update_NumberOfMolecules(Components& SystemComponents, Atoms*& d_a, size_t SelectedComponent, bool Insertion)
{
  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //change in atom number counts
  int NumMol = -1; //default: deletion; Insertion: +1, Deletion: -1, size_t is never negative
  if(Insertion) NumMol = 1;
  //Update Components
  SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] += NumMol;
  SystemComponents.TotalNumberOfMolecules += NumMol;
  
  // check System size //
  size_t size = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(size > SystemComponents.Allocate_size[SelectedComponent])
  {
    //printf("Trying to allocate more space\n");
    AllocateMoreSpace(d_a, SelectedComponent, SystemComponents);
    //throw std::runtime_error("Need to allocate more space, not implemented\n");
  }
}

__global__ void Update_insertion_data(Atoms* d_a, Atoms Mol, Atoms NewMol, size_t SelectedTrial, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  //UpdateLocation should be the last position of the dataset
  //Need to check if Allocate_size is smaller than size
  if(Moleculesize == 1) //Only first bead is inserted, first bead data is stored in NewMol
  {
    d_a[SelectedComponent].x[UpdateLocation]         = NewMol.x[SelectedTrial];
    d_a[SelectedComponent].y[UpdateLocation]         = NewMol.y[SelectedTrial];
    d_a[SelectedComponent].z[UpdateLocation]         = NewMol.z[SelectedTrial];
    d_a[SelectedComponent].scale[UpdateLocation]     = NewMol.scale[SelectedTrial];
    d_a[SelectedComponent].charge[UpdateLocation]    = NewMol.charge[SelectedTrial];
    d_a[SelectedComponent].scaleCoul[UpdateLocation] = NewMol.scaleCoul[SelectedTrial];
    d_a[SelectedComponent].Type[UpdateLocation]      = NewMol.Type[SelectedTrial];
    d_a[SelectedComponent].MolID[UpdateLocation]     = NewMol.MolID[SelectedTrial];
    //printf("First bead updated: %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %lu, %lu\n", NewMol.x[SelectedTrial], NewMol.y[SelectedTrial], NewMol.z[SelectedTrial], NewMol.scale[SelectedTrial], NewMol.charge[SelectedTrial], NewMol.scaleCoul[SelectedTrial], NewMol.Type[SelectedTrial], NewMol.MolID[SelectedTrial]);
  }
  else //Multiple beads: first bead + trial orientations
  {
  //Update the first bead, first bead data stored in position 0 of Mol //
  d_a[SelectedComponent].x[UpdateLocation]         = Mol.x[0];
  d_a[SelectedComponent].y[UpdateLocation]         = Mol.y[0];
  d_a[SelectedComponent].z[UpdateLocation]         = Mol.z[0];
  d_a[SelectedComponent].scale[UpdateLocation]     = Mol.scale[0];
  d_a[SelectedComponent].charge[UpdateLocation]    = Mol.charge[0];
  d_a[SelectedComponent].scaleCoul[UpdateLocation] = Mol.scaleCoul[0];
  d_a[SelectedComponent].Type[UpdateLocation]      = Mol.Type[0];
  d_a[SelectedComponent].MolID[UpdateLocation]     = Mol.MolID[0];
  
  size_t chainsize = Moleculesize - 1; // FOr trial orientations //
  for(size_t i = 0; i < chainsize; i++) //Update the selected orientations//
  {
    size_t selectsize = SelectedTrial*chainsize+i;
    d_a[SelectedComponent].x[UpdateLocation+i+1]         = NewMol.x[selectsize];
    d_a[SelectedComponent].y[UpdateLocation+i+1]         = NewMol.y[selectsize];
    d_a[SelectedComponent].z[UpdateLocation+i+1]         = NewMol.z[selectsize];
    d_a[SelectedComponent].scale[UpdateLocation+i+1]     = NewMol.scale[selectsize];
    d_a[SelectedComponent].charge[UpdateLocation+i+1]    = NewMol.charge[selectsize];
    d_a[SelectedComponent].scaleCoul[UpdateLocation+i+1] = NewMol.scaleCoul[selectsize];
    d_a[SelectedComponent].Type[UpdateLocation+i+1]      = NewMol.Type[selectsize];
    d_a[SelectedComponent].MolID[UpdateLocation+i+1]     = NewMol.MolID[selectsize];
  
  //printf("Updated: NewMol data: %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %lu, %lu\n", NewMol.x[selectsize], NewMol.y[selectsize], NewMol.z[selectsize], NewMol.scale[selectsize], NewMol.charge[selectsize], NewMol.scaleCoul[selectsize], NewMol.Type[selectsize], NewMol.MolID[selectsize]);
  }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  += Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
    //printf("x: %.10f, Alldata.size: %lu, d_a.size: %lu\n", AllData.x[UpdateLocation], AllData.size, d_a[SelectedComponent].size);
  }
}

static inline double Insertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  SystemComponents.Moves[SelectedComponent].InsertionTotal ++;
  bool Insertion = true;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0; double StoredR = 0.0;
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, false, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision); //Not reinsertion, not Retrace//
  if(!SuccessConstruction)
    return 0.0;
  //printf("First bead energy: %.10f, Rosenbluth: %.10f, Molsize: %zu\n", energy, Rosenbluth, SystemComponents.Moleculesize[SelectedComponent]);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    //printf("Doing trial orientations\n");
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, false, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision);
    if(!SuccessConstruction){ return 0.0;}
    energy += temp_energy;
    //printf("Trial Orientation energy: %.10f, Rosenbluth: %.10f\n", energy, Rosenbluth);
  }
  //printf("Selected Trial is %zu, Rosenbluth is %.10f\n", SelectedTrial, Rosenbluth);
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  //printf("MolFraction: %.10f, IdealRosen: %.10f, Fugacoeff: %.10f, NumMol: %.10f, Pressure: %.10f\n", MolFraction, IdealRosen, FugacityCoefficient, NumberOfMolecules, Box.Pressure);
  double preFactor = SystemComponents.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume / (1.0+NumberOfMolecules);
  if(!FF.noCharges)
  {
    double EwaldE = 0.0;
    //EwaldE = GPU_EwaldDifference(Box, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, true, SelectedTrial);
    EwaldE = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 1, SelectedTrial);
    //printf("After Insertion Move, EwaldDeltaE: %.5f\n", EwaldE);
    preFactor *= std::exp(-SystemComponents.Beta * EwaldE);
    energy    += EwaldE;
  }
  double RRR = get_random_from_zero_to_one();
  //printf("RRR: %.10f, prefactor: %.10f, Rosenbluth: %.10f, idealrosen: %.10f\n", RRR, preFactor, Rosenbluth, IdealRosen);
  if(RRR < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //printf("Accepted, UpdateLocation: %zu\n", UpdateLocation);
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, Insertion);
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Box, false, SystemComponents);
    }
    return energy;
  }
  return 0.0;
}

__global__ void StoreNewLocation_Reinsertion(Atoms Mol, Atoms NewMol, double* tempx, double* tempy, double* tempz, size_t SelectedTrial, size_t Moleculesize)
{
  if(Moleculesize == 1) //Only first bead is inserted, first bead data is stored in NewMol
  {
    tempx[0] = NewMol.x[SelectedTrial];
    tempy[0] = NewMol.y[SelectedTrial];
    tempz[0] = NewMol.z[SelectedTrial];
  }
  else //Multiple beads: first bead + trial orientations
  {
    //Update the first bead, first bead data stored in position 0 of Mol //
    tempx[0] = Mol.x[0];
    tempy[0] = Mol.y[0];
    tempz[0] = Mol.z[0];
   
    size_t chainsize = Moleculesize - 1; // FOr trial orientations //
    for(size_t i = 0; i < chainsize; i++) //Update the selected orientations//
    {
      size_t selectsize = SelectedTrial*chainsize+i;
      tempx[i+1] = NewMol.x[selectsize];
      tempy[i+1] = NewMol.y[selectsize];
      tempz[i+1] = NewMol.z[selectsize];
    }
  }
}

__global__ void Update_Reinsertion_data(Atoms* d_a, double* tempx, double* tempy, double* tempz, size_t SelectedComponent, size_t UpdateLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t realLocation = UpdateLocation + i;
  d_a[SelectedComponent].x[realLocation] = tempx[i];
  d_a[SelectedComponent].y[realLocation] = tempy[i];
  d_a[SelectedComponent].z[realLocation] = tempz[i];
  //printf("temp xyz: %.5f %.5f %.5f, Updated xyz: %.5f %.5f %.5f\n", tempx[i], tempy[i], tempz[i], d_a[SelectedComponent].x[realLocation], d_a[SelectedComponent].y[realLocation], d_a[SelectedComponent].z[realLocation]);
}

static inline double Reinsertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  SystemComponents.Moves[SelectedComponent].ReinsertionTotal ++;
  bool Insertion = true;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0; double old_energy = 0.0; double StoredR = 0.0;
 
  ///////////////
  // INSERTION //
  ///////////////
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, true, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision); //Not reinsertion, not Retrace//
  if(!SuccessConstruction)
    return 0.0;
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, true, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision); //True for doing insertion for reinsertion, different in MoleculeID//
    if(!SuccessConstruction){ return 0.0;}
    energy += temp_energy;
  }
  //Store The New Locations//
  double *tempx; double *tempy; double *tempz;
  cudaMalloc(&tempx, sizeof(double) * SystemComponents.Moleculesize[SelectedComponent]);
  cudaMalloc(&tempy, sizeof(double) * SystemComponents.Moleculesize[SelectedComponent]);
  cudaMalloc(&tempz, sizeof(double) * SystemComponents.Moleculesize[SelectedComponent]);
  StoreNewLocation_Reinsertion<<<1,1>>>(Sims.Old, Sims.New, tempx, tempy, tempz, SelectedTrial, SystemComponents.Moleculesize[SelectedComponent]);
  /////////////
  // RETRACE //
  /////////////
  double Old_Rosen=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, false, true, true, StoredR, &SelectedTrial, &SuccessConstruction, &old_energy, DualPrecision);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = old_energy;
    Old_Rosen*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, false, false, &SelectedTrial, &SuccessConstruction, &old_energy, SelectedFirstBeadTrial, DualPrecision);
    old_energy += temp_energy;
  } 
 
  //Calculate Ewald//
  double EwaldE = 0.0;
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
  if(!FF.noCharges)
  {
    EwaldE = GPU_EwaldDifference_Reinsertion(Box, Sims.d_a, Sims.Old, tempx, tempy, tempz, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);
    Rosenbluth *= std::exp(-SystemComponents.Beta * EwaldE);
    energy     += EwaldE;
  }

  //Determine whether to accept or reject the insertion
  double RRR = get_random_from_zero_to_one();
  if(RRR < Rosenbluth / Old_Rosen)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].ReinsertionAccepted ++;
    //size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
    Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, tempx, tempy, tempz, SelectedComponent, UpdateLocation); checkCUDAError("error Updating Reinsertion data");
    if(!FF.noCharges) Update_Ewald_Vector(Box, false, SystemComponents);
    return energy - old_energy;
  }
  else
  cudaFree(tempx); cudaFree(tempy); cudaFree(tempz);
  return 0.0;
}

__global__ void Update_deletion_data(Atoms* d_a, Atoms NewMol, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  //UpdateLocation should be the molecule that needs to be deleted
  //Then move the atom at the last position to the location of the deleted molecule
  //**Zhao's note** MolID of the deleted molecule should not be changed
  //**Zhao's note** if Molecule deleted is the last molecule, then nothing is copied, just change the size later.
  if(UpdateLocation != LastLocation)
  {
    for(size_t i = 0; i < Moleculesize; i++)
    {
      d_a[SelectedComponent].x[UpdateLocation+i]         = d_a[SelectedComponent].x[LastLocation+i];
      d_a[SelectedComponent].y[UpdateLocation+i]         = d_a[SelectedComponent].y[LastLocation+i];
      d_a[SelectedComponent].z[UpdateLocation+i]         = d_a[SelectedComponent].z[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
    }
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

static inline double Deletion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF,  RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  //if(System[SelectedComponent].size == 0){printf("No Molecule to delete\n"); return 0.0;} //cannot delete molecule when there is no molecule
  SystemComponents.Moves[SelectedComponent].DeletionTotal ++;
  bool Insertion = false;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0;
  double StoredR = 0.0; //Don't use this for Deletion//
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, false, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, false, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision); //The false is for Reinsertion//
    energy += temp_energy;
  }
  if(!SuccessConstruction)
    return 0.0;
  //printf("Selected Trial is %zu, Rosenbluth is %.10f\n", SelectedTrial, Rosenbluth);
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  //printf("Deletion: MolFraction: %.10f, IdealRosen: %.10f, Fugacoeff: %.10f, NumMol: %.10f, Pressure: %.10f\n", MolFraction, IdealRosen, FugacityCoefficient, NumberOfMolecules, Box.Pressure);
  double preFactor = (NumberOfMolecules) / (SystemComponents.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume);
  size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
  double EwaldE = 0.0;
  if(!FF.noCharges)
  {
    //EwaldE = GPU_EwaldDifference_Deletion(Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);
    EwaldE = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 2, UpdateLocation);
    preFactor *= std::exp(-SystemComponents.Beta * EwaldE);
    energy    -= EwaldE;
  }
  double RRR = get_random_from_zero_to_one();
  //printf("RRR: %.10f, prefactor: %.10f, Rosenbluth: %.10f, idealrosen: %.10f\n", RRR, preFactor, Rosenbluth, IdealRosen);
  //if(RRR < preFactor * IdealRosen / Rosenbluth)
  if(RRR < preFactor * IdealRosen / Rosenbluth) // for bebug: always accept
  { // accept the move
    SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    // Get the starting position of the last molecule in the array
    size_t LastLocation = (SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1)*SystemComponents.Moleculesize[SelectedComponent];
    //printf("Accepted, UpdateLocation: %zu, LastLocation: %zu, energy: %.10f\n", UpdateLocation, LastLocation, energy);
    //if(UpdateLocation == LastLocation){printf("Deleting the LAST molecule\n");}
    Update_deletion_data<<<1,1>>>(Sims.d_a, Sims.New, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, Insertion);
    if(!FF.noCharges) Update_Ewald_Vector(Box, false, SystemComponents);
    return -energy;
  }
  return 0.0;
}

static inline double CreateMolecule(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  bool Insertion = true;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0; double StoredR = 0.0;
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, false, false, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision); //Not reinsertion, not Retrace//
  if(!SuccessConstruction)
    return 0.0;
  //printf("First bead energy: %.10f, Rosenbluth: %.10f, Molsize: %zu\n", energy, Rosenbluth, SystemComponents.Moleculesize[SelectedComponent]);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    //printf("Doing trial orientations\n");
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Insertion, false, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision);
    if(!SuccessConstruction){ return 0.0;}
    energy += temp_energy;
    //printf("Trial Orientation energy: %.10f, Rosenbluth: %.10f\n", energy, Rosenbluth);
  }
  //printf("Selected Trial is %zu, Rosenbluth is %.10f\n", SelectedTrial, Rosenbluth);
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  //printf("MolFraction: %.10f, IdealRosen: %.10f, Fugacoeff: %.10f, NumMol: %.10f, Pressure: %.10f\n", MolFraction, IdealRosen, FugacityCoefficient, NumberOfMolecules, Box.Pressure);
  double preFactor = SystemComponents.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume / (1.0+NumberOfMolecules);

  double EwaldCorrection = 1.0; //Start with no Ewald Correction//
  //Ewald Correction, done on HOST (CPU) //
  bool EwaldCPU = false;
  if(!FF.noCharges) 
  {
    double EwaldE = 0.0;
    if(EwaldCPU) 
    {
      EwaldE = CPU_EwaldDifference(Box, Sims.New, Sims.Old, FF, SystemComponents, SelectedComponent, true, SelectedTrial);
    }
    else
    {
      //EwaldE = CPU_EwaldDifference(Box, Sims.New, Sims.Old, FF, SystemComponents, SelectedComponent, true, SelectedTrial);
      //EwaldE = GPU_EwaldDifference(Box, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, true, SelectedTrial);
      EwaldE = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 1, SelectedTrial);
    }
    EwaldCorrection = std::exp(-SystemComponents.Beta * EwaldE);
    printf("EwaldDeltaE: %.5f, EwaldCorrection is %.5f\n", EwaldE, EwaldCorrection);
    preFactor *= EwaldCorrection;
  }

  double RRR = 1e-10;
  if(RRR < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //printf("Accepted, UpdateLocation: %zu\n", UpdateLocation);
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Box, EwaldCPU, SystemComponents);
    }
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, Insertion);
    return energy;
  }
  return 0.0;
}
