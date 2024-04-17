static inline MoveEnergy Insertion_Body(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, double& Rosenbluth, bool& SuccessConstruction, size_t& SelectedTrial, double& preFactor, bool previous_step, double2 newScale)
{
  MoveEnergy energy; double StoredR = 0.0;
  int CBMCType = CBMC_INSERTION; //Insertion//
  Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale); //Not reinsertion, not Retrace//
  if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
  if(!SuccessConstruction) 
  {
    //printf("Early return FirstBead\n");
    energy.zero();
    return energy;
  }
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale);
    if(Rosenbluth <= 1e-150) SuccessConstruction = false;
    if(!SuccessConstruction) 
    { 
      //printf("Early return Chain\n");
      energy.zero();
      return energy;
    }
    SystemComponents.SumDeltaE(energy, temp_energy, ADD);
  }
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);

  //If component has fractional molecule, subtract the number of molecules by 1.//
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]){NumberOfMolecules-=1.0;}
  if(NumberOfMolecules < 0.0) NumberOfMolecules = 0.0;

  preFactor = SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume / (1.0+NumberOfMolecules);

  //Ewald Correction, done on HOST (CPU) //
  bool EwaldCPU = false;
  int MoveType = INSERTION; //Normal Insertion, including fractional insertion, no previous step (do not use temprorary totalEik)//
  if(previous_step) //Fractional Insertion after a lambda change move that makes the old fractional molecule full//
  {
    MoveType = CBCF_INSERTION;  // CBCF fractional insertion //
  }
  if(!FF.noCharges)
  {
    double EwaldE = 0.0;
    if(EwaldCPU)
    {
      EwaldE = CPU_EwaldDifference(Sims.Box, Sims.New, Sims.Old, FF, SystemComponents, SelectedComponent, true, SelectedTrial);
    }
    else
    {
      EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType, SelectedTrial, newScale);
    }
    energy.EwaldE = EwaldE;
    energy.HGEwaldE=SystemComponents.tempdeltaHGEwald;
    double CoulRealE = 0.0;
    if(!FF.VDWRealBias)
    {
      CoulRealE = CoulombRealCorrection_General(Sims.Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType);
    }
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE + SystemComponents.tempdeltaHGEwald + CoulRealE));
    energy.GGVDWReal += CoulRealE;
  }
  double TailE = 0.0;
  TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, MoveType);
  Rosenbluth *= std::exp(-SystemComponents.Beta * TailE);
  energy.TailE= TailE;
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    double DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, DNN_INSERTION);
    energy.DNN_E   = DNN_New;
    double correction = energy.DNN_Correction();
    if(fabs(correction) > 1000.0) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
      //printf("INSERTION: Bad Prediction, reject the move!!!\n"); 
      SystemComponents.InsertionDNNReject ++;
      SuccessConstruction = false;
      energy.zero();
      return energy;
    }
    Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
  }
  return energy;
}
/*
static inline void AcceptInsertion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, size_t SelectedTrial, bool noCharges)
{
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
  Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
  Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, true); //true = Insertion//
  if(noCharges)
  {
    Update_Ewald_Vector(Sims.Box, false, SystemComponents);
  }
  SystemComponents.deltaVDWReal += SystemComponents.tempdeltaVDWReal;
  SystemComponents.deltaEwald   += SystemComponents.tempdeltaEwald;
}

static inline void AcceptDeletion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, size_t UpdateLocation, bool noCharges)
{
  size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1;
  size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[SelectedComponent];
  Update_deletion_data<<<1,1>>>(Sims.d_a, Sims.New, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);

  Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, false); //false = Deletion//
  if(noCharges) 
  {
    Update_Ewald_Vector(Sims.Box, false, SystemComponents);
  }
  SystemComponents.deltaVDWReal += SystemComponents.tempdeltaVDWReal;
  SystemComponents.deltaEwald   += SystemComponents.tempdeltaEwald;
}
*/
static inline MoveEnergy Deletion_Body(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, size_t& UpdateLocation, double& Rosenbluth, bool& SuccessConstruction, double& preFactor, double2 newScale)
{
  size_t SelectedTrial = 0;
  MoveEnergy energy; 
  
  double StoredR = 0.0; //Don't use this for Deletion//
  int CBMCType = CBMC_DELETION; //Deletion//
  //Zhao's note: Deletion_body will be part the GibbsParticleTransferMove, and the fractional molecule might be selected, so newScale will not be 1.0//
  //double2 newScale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0), Zhao's note: This is not used in deletion, just set to 1//
  Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); //The false is for Reinsertion//
    SystemComponents.SumDeltaE(energy, temp_energy, ADD);
  }
  if(!SuccessConstruction)
  {
    energy.zero();
    return energy;
  }
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);

  //If component has fractional molecule, subtract the number of molecules by 1.//
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]){NumberOfMolecules-=1.0;}

  preFactor = (NumberOfMolecules) / (SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume);
  UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
  double EwaldE = 0.0;
  if(!FF.noCharges)
  {
    EwaldE      = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, DELETION, UpdateLocation, newScale);
    double CoulRealE = 0.0;
    if(!FF.VDWRealBias)
    {
      CoulRealE += CoulombRealCorrection_General(Sims.Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, DELETION);
      SystemComponents.tempdeltaVDWReal += CoulRealE;
    }
    Rosenbluth /= std::exp(-SystemComponents.Beta * (EwaldE + SystemComponents.tempdeltaHGEwald + CoulRealE));
    energy.EwaldE= -1.0 * (EwaldE);
    energy.HGEwaldE= -1.0*SystemComponents.tempdeltaHGEwald; //Becareful with the sign here, you need a HG sum, but HGVDWReal and HGEwaldE here have opposite signs???//
  }
  double TailE = 0.0;
  TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, DELETION);
  Rosenbluth /= std::exp(-SystemComponents.Beta * TailE);
  energy.TailE = -TailE;
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    double DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, DNN_DELETION);
    energy.DNN_E   = DNN_New;
    double correction = energy.DNN_Correction();
    if(fabs(correction) > 1000.0) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
      //printf("DELETION: Bad Prediction, reject the move!!!\n"); 
      SystemComponents.DeletionDNNReject ++;
      SuccessConstruction = false;
      energy.zero();
      return energy;
    }
    Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
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
  if(Insertion)  SystemComponents.UpdatePseudoAtoms(INSERTION, SelectedComponent);
  if(!Insertion) SystemComponents.UpdatePseudoAtoms(DELETION,  SelectedComponent);
  if(size > SystemComponents.Allocate_size[SelectedComponent])
  {
    AllocateMoreSpace(d_a, SelectedComponent, SystemComponents);
    throw std::runtime_error("Need to allocate more space, not implemented\n");
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
    for(size_t j = 0; j < chainsize; j++) //Update the selected orientations//
    {
      size_t selectsize = SelectedTrial*chainsize+j;
      d_a[SelectedComponent].x[UpdateLocation+j+1]         = NewMol.x[selectsize];
      d_a[SelectedComponent].y[UpdateLocation+j+1]         = NewMol.y[selectsize];
      d_a[SelectedComponent].z[UpdateLocation+j+1]         = NewMol.z[selectsize];
      d_a[SelectedComponent].scale[UpdateLocation+j+1]     = NewMol.scale[selectsize];
      d_a[SelectedComponent].charge[UpdateLocation+j+1]    = NewMol.charge[selectsize];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+j+1] = NewMol.scaleCoul[selectsize];
      d_a[SelectedComponent].Type[UpdateLocation+j+1]      = NewMol.Type[selectsize];
      d_a[SelectedComponent].MolID[UpdateLocation+j+1]     = NewMol.MolID[selectsize];
    }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  += Moleculesize;
    /*
    for(size_t j = 0; j < Moleculesize; j++)
      printf("xyz: %.5f %.5f %.5f, scale/charge/scaleCoul: %.5f %.5f %.5f, Type: %lu, MolID: %lu\n", d_a[SelectedComponent].x[UpdateLocation+j], d_a[SelectedComponent].y[UpdateLocation+j], d_a[SelectedComponent].z[UpdateLocation+j], d_a[SelectedComponent].scale[UpdateLocation+j], d_a[SelectedComponent].charge[UpdateLocation+j], d_a[SelectedComponent].scaleCoul[UpdateLocation+j], d_a[SelectedComponent].Type[UpdateLocation+j], d_a[SelectedComponent].MolID[UpdateLocation+j]);
    */
  }
}
