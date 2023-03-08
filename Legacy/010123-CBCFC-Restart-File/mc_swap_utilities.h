static inline double Insertion_Body(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision, double& Rosenbluth, bool& SuccessConstruction, size_t& SelectedTrial, double& preFactor, bool previous_step, double2 newScale)
{
  double energy = 0.0; double StoredR = 0.0;
  int CBMCType = CBMC_INSERTION; //Insertion//
  Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision, newScale); //Not reinsertion, not Retrace//
  if(!SuccessConstruction)
    return 0.0;
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision, newScale);
    if(!SuccessConstruction){ return 0.0;}
    energy += temp_energy;
  }
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);

  //If component has fractional molecule, subtract the number of molecules by 1.//
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]){NumberOfMolecules-=1.0;}
  if(NumberOfMolecules < 0.0) NumberOfMolecules = 0.0;

  preFactor = SystemComponents.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume / (1.0+NumberOfMolecules);

  SystemComponents.tempdeltaVDWReal += energy;

  //Ewald Correction, done on HOST (CPU) //
  bool EwaldCPU = false;
  if(!FF.noCharges)
  {
    double EwaldE = 0.0;
    int MoveType = INSERTION; //Normal Insertion, including fractional insertion, no previous step (do not use temprorary totalEik)//
    if(previous_step) //Fractional Insertion after a lambda change move that makes the old fractional molecule full//
    {
      MoveType = CBCF_INSERTION;  // CBCF fractional insertion //
    }
    if(EwaldCPU)
    {
      EwaldE = CPU_EwaldDifference(Box, Sims.New, Sims.Old, FF, SystemComponents, SelectedComponent, true, SelectedTrial);
    }
    else
    {
      EwaldE = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType, SelectedTrial, newScale);
      SystemComponents.tempdeltaEwald += EwaldE;
    }
    preFactor *= std::exp(-SystemComponents.Beta * EwaldE);
    energy    += EwaldE;
  }
  return energy;
}

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
  }*/
}

void AllocateMoreSpace(Atoms*& d_a, size_t SelectedComponent, Components& SystemComponents)
{
  printf("Allocating more space on device\n");
  Atoms temp; // allocate a struct on the device for copying data.
  //Atoms tempSystem[SystemComponents.Total_Components];
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

  }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  += Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
  }
}
