__global__ void Prepare_LambdaChange(Atoms* d_a, Atoms Mol, Simulations Sims, ForceField FF, size_t start_position, size_t SelectedComponent, bool* device_flag)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t real_pos = start_position + i;

  const double x         = d_a[SelectedComponent].x[real_pos];
  const double y         = d_a[SelectedComponent].y[real_pos];
  const double z         = d_a[SelectedComponent].z[real_pos];
  const double scale     = d_a[SelectedComponent].scale[real_pos];
  const double charge    = d_a[SelectedComponent].charge[real_pos];
  const double scaleCoul = d_a[SelectedComponent].scaleCoul[real_pos];
  const size_t Type      = d_a[SelectedComponent].Type[real_pos];
  const size_t MolID     = d_a[SelectedComponent].MolID[real_pos];

  //Atoms Temp = Sims.Old; Atoms TempNEW = Sims.New;

  Mol.x[i] = x;
  Mol.y[i] = y;
  Mol.z[i] = z;
  Mol.scale[i] = scale; Mol.charge[i] = charge; Mol.scaleCoul[i] = scaleCoul; Mol.Type[i] = Type; Mol.MolID[i] = MolID;
  device_flag[i] = false;
}

static inline MoveEnergy CBCF_LambdaChange(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, double2 oldScale, double2 newScale, size_t& start_position, int MoveType, bool& SuccessConstruction)
{
  MoveEnergy tot;
  //double result = 0.0;
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  size_t chainsize;
  //Set up Old position and New position arrays
  chainsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  if(chainsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  Prepare_LambdaChange<<<1, chainsize>>>(Sims.d_a, Sims.Old, Sims, FF, start_position, SelectedComponent, Sims.device_flag);

  // Setup for the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(Atomsize * chainsize, &Nblock, &Nthread); //Zhao's note: add back dUdlambda later //
  Energy_difference_LambdaChange<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SelectedComponent, Atomsize, chainsize, Nthread, Sims.device_flag, newScale);
  cudaMemcpy(Sims.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  if(!Sims.flag[0])
  {
    SuccessConstruction = true;
    double BlockResult[Nblock];
    cudaMemcpy(BlockResult, Sims.Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < Nblock; i++) tot.HGVDWReal += BlockResult[i]; //Zhao's note: this will be fixed after I divide the HG and GG in Energy_difference_LambdaChange//
    // Calculate Ewald //
    if(!FF.noCharges)
    {
      //Zhao's note: since we changed it from using translation/rotation functions to its own, this needs to be changed as well//
      tot.EwaldE = GPU_EwaldDifference_LambdaChange(Sims.Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, oldScale, newScale, MoveType);
    }
    tot.TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, MoveType);
  }
  else
  {
    SuccessConstruction = false;
  }
  return tot;
}

__global__ void Revert_CBCF_Insertion(Atoms* d_a, size_t SelectedComponent, size_t start_position, double2 oldScale)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[SelectedComponent].scale[start_position+i] = oldScale.x;
  d_a[SelectedComponent].scaleCoul[start_position+i] = oldScale.y;
}

/////////////////////////////////////

__global__ void Update_deletion_data_fractional(Atoms* d_a, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  //UpdateLocation should be the molecule that needs to be deleted
  //Then move the atom at the last position to the location of the deleted molecule
  //**Zhao's note** MolID of the deleted molecule should not be changed
  //**Zhao's note** if Molecule deleted is the last molecule, then nothing is copied, just change the size later.
  //**Zhao's note** for fractional molecule, since we still need the data of the deleted fractional molecule (for reversion if the move is rejected), store the old data at the LastLocation of d_a
  if(UpdateLocation != LastLocation)
  {
    for(size_t i = 0; i < Moleculesize; i++)
    {
      double x = d_a[SelectedComponent].x[UpdateLocation+i]; double y = d_a[SelectedComponent].y[UpdateLocation+i];
      double z = d_a[SelectedComponent].z[UpdateLocation+i]; double scale = d_a[SelectedComponent].scale[UpdateLocation+i];
      double charge = d_a[SelectedComponent].charge[UpdateLocation+i];
      double scaleCoul = d_a[SelectedComponent].scaleCoul[UpdateLocation+i]; double Type = d_a[SelectedComponent].Type[UpdateLocation+i];
      d_a[SelectedComponent].x[UpdateLocation+i]         = d_a[SelectedComponent].x[LastLocation+i];
      d_a[SelectedComponent].y[UpdateLocation+i]         = d_a[SelectedComponent].y[LastLocation+i];
      d_a[SelectedComponent].z[UpdateLocation+i]         = d_a[SelectedComponent].z[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
      //Put the data of the old fractional molecule to LastLocation//
      d_a[SelectedComponent].x[LastLocation+i] = x; d_a[SelectedComponent].y[LastLocation+i] = y;
      d_a[SelectedComponent].z[LastLocation+i] = z; d_a[SelectedComponent].scale[LastLocation+i] = scale;
      d_a[SelectedComponent].charge[LastLocation+i] = charge; d_a[SelectedComponent].scaleCoul[LastLocation+i] = scaleCoul;
      d_a[SelectedComponent].Type[LastLocation+i] = Type;
    }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  if(i==0)
  {
    d_a[SelectedComponent].size  -= Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
  }
}

__global__ void Revert_CBCF_Deletion(Atoms* d_a, Atoms NewMol, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation)
{
  //Zhao's note: this is literally the same function as Update_deletion_data_fractional
  //UpdateLocation should be the molecule that needs to be deleted
  //Then move the atom at the last position to the location of the deleted molecule
  //**Zhao's note** MolID of the deleted molecule should not be changed
  //**Zhao's note** if Molecule deleted is the last molecule, then nothing is copied, just change the size later.
  //**Zhao's note** for fractional molecule, since we still need the data of the deleted fractional molecule (for reversion if the move is rejected), store the old data at the LastLocation of d_a
  if(UpdateLocation != LastLocation)
  {
    for(size_t i = 0; i < Moleculesize; i++)
    {
      double x = d_a[SelectedComponent].x[UpdateLocation+i]; double y = d_a[SelectedComponent].y[UpdateLocation+i];
      double z = d_a[SelectedComponent].z[UpdateLocation+i]; double scale = d_a[SelectedComponent].scale[UpdateLocation+i];
      double charge = d_a[SelectedComponent].charge[UpdateLocation+i];
      double scaleCoul = d_a[SelectedComponent].scaleCoul[UpdateLocation+i]; double Type = d_a[SelectedComponent].Type[UpdateLocation+i];
      d_a[SelectedComponent].x[UpdateLocation+i]         = d_a[SelectedComponent].x[LastLocation+i];
      d_a[SelectedComponent].y[UpdateLocation+i]         = d_a[SelectedComponent].y[LastLocation+i];
      d_a[SelectedComponent].z[UpdateLocation+i]         = d_a[SelectedComponent].z[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
      //Put the data of the old fractional molecule to LastLocation//
      d_a[SelectedComponent].x[LastLocation+i] = x; d_a[SelectedComponent].y[LastLocation+i] = y;
      d_a[SelectedComponent].z[LastLocation+i] = z; d_a[SelectedComponent].scale[LastLocation+i] = scale;
      d_a[SelectedComponent].charge[LastLocation+i] = charge; d_a[SelectedComponent].scaleCoul[LastLocation+i] = scaleCoul;
      d_a[SelectedComponent].Type[LastLocation+i] = Type;
    }
  }
  //Zhao's note: the single values in d_a and System are pointing to different locations//
  //d_a is just device (cannot access on host), while System is shared (accessible on host), need to update d_a values here
  //there are two of these values: size and Allocate_size
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i==0)
  {
    d_a[SelectedComponent].size  += Moleculesize; //Zhao's special note: AllData.size doesn't work... So single values are painful, need to consider pointers for single values
  }
}

__global__ void update_CBCF_scale(Atoms* d_a, size_t start_position, size_t SelectedComponent, double2 newScale)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[SelectedComponent].scale[start_position+i]     = newScale.x;
  d_a[SelectedComponent].scaleCoul[start_position+i] = newScale.y;
}

////////////////////////////////////////





MoveEnergy CBCFMove(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent);

static inline MoveEnergy CBCFMove(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;

  int    MoveType;
  double TMMCPacc = 0.0;

  SystemComponents.Moves[SelectedComponent].CBCFTotal ++;
  //First draw a New Lambda, from the Bin//
  LAMBDA lambda = SystemComponents.Lambda[SelectedComponent];
  MoveEnergy final_energy;
  size_t oldBin = lambda.currentBin;
  size_t nbin   = lambda.binsize;
  double delta  = lambda.delta;
  double oldLambda = delta * static_cast<double>(oldBin);

  //printf("Old Bin is %zu, Old Lambda is %.5f\n", oldBin, oldLambda);

  SystemComponents.tempdeltaVDWReal = 0.0; SystemComponents.tempdeltaEwald = 0.0;

  MoveEnergy energy;

  int SelectednewBin = selectNewBin(SystemComponents.Lambda[SelectedComponent]);
  //Zhao's note: if the new bin == oldbin, the scale is the same, no need for a move, so select a new lambda
  //Zhao's note: Do not delete the fractional molecule when there is only one molecule, the system needs to have at least one fractional molecule for each species
  while(SelectednewBin == oldBin || (SelectednewBin < 0 && SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] == 1))
    SelectednewBin = selectNewBin(SystemComponents.Lambda[SelectedComponent]);
  //////////////////
  //INSERTION MOVE//
  //////////////////
  if(SelectednewBin > static_cast<int>(nbin))
  {
    MoveType = CBCF_INSERTION;
    //return 0.0;
    SystemComponents.Moves[SelectedComponent].CBCFInsertionTotal ++;
    int newBin = SelectednewBin - static_cast<int>(nbin);
    double  newLambda = delta * static_cast<double>(newBin);
    //Zhao's note: Set the Scaling factors for VDW and Coulombic Interaction depending on the mode//
    double2 oldScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(oldLambda);
    double2 InterScale= SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
    double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(newLambda);
    //////////////////////////////////////////////////////////////////////////////
    //First step: Increase the scale of the *current* fractional molecule to 1.0//
    //////////////////////////////////////////////////////////////////////////////
    size_t start_position = 1;
    bool   SuccessConstruction = false;
    energy = CBCF_LambdaChange(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, oldScale, InterScale, start_position, 5, SuccessConstruction);
    if(!SuccessConstruction) 
    {
      //If unsuccessful move (Overlap), Pacc = 0//
      SystemComponents.Tmmc[SelectedComponent].Update(0.0, NMol, CBCF_INSERTION);
      energy.zero();
      return energy;
    }
    MoveEnergy first_step_energy = energy;
    //Pretend the first step is accepted//
    update_CBCF_scale<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, start_position, SelectedComponent, InterScale);
           SuccessConstruction = false;
    double Rosenbluth          = 0.0;
    size_t SelectedTrial       = 0;
    double preFactor           = 0.0;
    bool   Accepted            = false;
    /////////////////////////////////////////////////////
    //Second step: Insertion of the fractional molecule//
    /////////////////////////////////////////////////////
    MoveEnergy second_step_energy = Insertion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, true, newScale);
    if(SuccessConstruction)
    {
      SystemComponents.SumDeltaE(energy, second_step_energy, ADD);
      //Account for the biasing terms of different lambdas//
      double biasTerm = SystemComponents.Lambda[SelectedComponent].biasFactor[newBin] - SystemComponents.Lambda[SelectedComponent].biasFactor[oldBin];
      preFactor *= std::exp(-SystemComponents.Beta * first_step_energy.total() + biasTerm);
      double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
      double RRR = get_random_from_zero_to_one();
      TMMCPacc = preFactor * Rosenbluth / IdealRosen; //Unbiased Acceptance//
      //Apply the bias according to the macrostate//
      SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, CBCF_INSERTION);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, CBCF_INSERTION);

      if(RRR < preFactor * Rosenbluth / IdealRosen) Accepted = true;
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accepted, NMol, CBCF_INSERTION);

      if(Accepted)
      { // accept the move
        SystemComponents.Moves[SelectedComponent].CBCFInsertionAccepted ++;
        SystemComponents.Moves[SelectedComponent].CBCFAccepted ++;
        size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
        //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
        Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
        Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, CBCF_INSERTION);
        //Update the ID of the fractional molecule on the host//
        SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] - 1;
        SystemComponents.Lambda[SelectedComponent].currentBin = newBin;
        if(!FF.noCharges)
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents);
        }
        final_energy = energy;
      }
    }
    if(!Accepted) Revert_CBCF_Insertion<<<1, SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, SelectedComponent, start_position, oldScale);
  }
  else if(SelectednewBin < 0) //Deletion Move//
  {
    MoveType = CBCF_DELETION;
    //return 0.0; //for debugging
    SystemComponents.Moves[SelectedComponent].CBCFDeletionTotal ++;
    int newBin = SelectednewBin + nbin;
    double newLambda = delta * static_cast<double>(newBin);
    //Zhao's note: Set the Scaling factors for VDW and Coulombic Interaction depending on the mode//
    double2 oldScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(oldLambda);
    double2 InterScale= SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
    double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(newLambda);
    //////////////////////////////////////////////
    //First step: delete the fractional molecule//
    //////////////////////////////////////////////
    bool SuccessConstruction = false;
    size_t SelectedTrial = 0;
    MoveEnergy energy;
    double StoredR = 0.0; //Don't use this for Deletion//
    int MoveType = CBMC_DELETION; //Deletion//
    double Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, MoveType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, oldScale);
    if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
    {
      size_t SelectedFirstBeadTrial = SelectedTrial; 
      MoveEnergy temp_energy = energy;
      Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, MoveType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, oldScale); //The false is for Reinsertion//
      SystemComponents.SumDeltaE(energy, temp_energy, ADD);
    }
    if(!SuccessConstruction) 
    {
      //If unsuccessful move (Overlap), Pacc = 0//
      SystemComponents.Tmmc[SelectedComponent].Update(0.0, NMol, CBCF_DELETION);
      energy.zero();
      return energy;
    }
    //Determine whether to accept or reject the insertion
    double MolFraction = SystemComponents.MolFraction[SelectedComponent];
    double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
    double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
    double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
 
    //If component has fractional molecule, subtract the number of molecules by 1.//
    if(SystemComponents.hasfractionalMolecule[SelectedComponent]){NumberOfMolecules-=1.0;}
 
    double preFactor = (NumberOfMolecules) / (SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume);
    size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
    double EwaldE = 0.0;
    if(!FF.noCharges)
    {
      if(oldLambda < 1e-100) //Skip Ewald Summation if the old fractional molecule has a scale of zero//
      {
        Skip_Ewald(Sims.Box);
      }
      else
      {
        //EwaldE = GPU_EwaldDifference_Deletion(Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);
        EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 6, UpdateLocation, oldScale);
      }
      preFactor *= std::exp(-SystemComponents.Beta * EwaldE);
      energy.EwaldE = -EwaldE;
      SystemComponents.tempdeltaEwald += EwaldE;
    }
    double TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, CBCF_DELETION);
    preFactor   *= std::exp(-SystemComponents.Beta * TailE);
    SystemComponents.tempdeltaTailE += TailE;
    energy.TailE = -TailE;
    MoveEnergy first_step_energy = energy;
    //Pretend the deletion move is accepted//
    size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1;
    size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[SelectedComponent];
    Update_deletion_data_fractional<<<1,1>>>(Sims.d_a, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
    //Record the old fractionalmolecule ID//
    size_t OldFracMolID = SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID;
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, CBCF_DELETION);
    ///////////////////////////////////////////////////////////////////////////////////
    //Second step: randomly choose a new fractional molecule, perform the lambda move//
    ///////////////////////////////////////////////////////////////////////////////////
    SelectedMolInComponent = static_cast<size_t>(get_random_from_zero_to_one() * static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]));
    SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SelectedMolInComponent;


    bool Accepted = false;
    if(SuccessConstruction)
    {
      size_t start_position = 1;
      MoveEnergy second_step_energy;
      
      SuccessConstruction = false;
      second_step_energy = CBCF_LambdaChange(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, InterScale, newScale, start_position, 6, SuccessConstruction); //Fractional deletion, lambda change is the second step (that uses the temporary totalEik vector)
      
      //Account for the biasing terms of different lambdas//
      double biasTerm = SystemComponents.Lambda[SelectedComponent].biasFactor[newBin] - SystemComponents.Lambda[SelectedComponent].biasFactor[oldBin];
      preFactor *= std::exp(-SystemComponents.Beta * second_step_energy.total() + biasTerm);
      TMMCPacc = preFactor * IdealRosen / Rosenbluth; //Unbiased Acceptance//
      //Apply the bias according to the macrostate//
      SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, CBCF_DELETION);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, CBCF_DELETION);

      if(get_random_from_zero_to_one() < preFactor * IdealRosen / Rosenbluth) Accepted = true;
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accepted, NMol, CBCF_DELETION);

      if(Accepted)
      {
        SystemComponents.Moves[SelectedComponent].CBCFDeletionAccepted ++;
        SystemComponents.Moves[SelectedComponent].CBCFAccepted ++;
        //update_translation_position<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
        update_CBCF_scale<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, start_position, SelectedComponent, newScale);
        SystemComponents.Lambda[SelectedComponent].currentBin = newBin;
        if(!FF.noCharges)
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents);
        }
        energy.HGVDWReal *= -1.0; energy.GGVDWReal *= -1.0; 
        energy.EwaldE    *= -1.0; energy.TailE     *= -1.0;
        energy.DNN_E     *= -1.0;
        SystemComponents.SumDeltaE(energy, second_step_energy, MINUS);
        final_energy = energy;
      }
    }
    if(!Accepted)
    {
      Revert_CBCF_Deletion<<<1,1>>>(Sims.d_a, Sims.New, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
      SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = OldFracMolID;
      SystemComponents.TotalNumberOfMolecules ++;
      SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] ++;
    }
  }
  else //Lambda Move//
  {
    MoveType = CBCF_LAMBDACHANGE;
    SystemComponents.Moves[SelectedComponent].CBCFLambdaTotal ++;
    int newBin = static_cast<size_t>(SelectednewBin);
    double newLambda = delta * static_cast<double>(newBin);
    //Zhao's note: Set the Scaling factors for VDW and Coulombic Interaction depending on the mode//
    double2 oldScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(oldLambda);
    double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(newLambda);
    size_t start_position = 1;
    bool   SuccessConstruction = false;
    energy = CBCF_LambdaChange(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, oldScale, newScale, start_position, CBCF_LAMBDACHANGE, SuccessConstruction);
    if(!SuccessConstruction) 
    {
      //If unsuccessful move (Overlap), Pacc = 0//
      SystemComponents.Tmmc[SelectedComponent].Update(0.0, NMol, MoveType);
      energy.zero();
      return energy;
    }
    //Account for the biasing terms of different lambdas//
    double biasTerm = SystemComponents.Lambda[SelectedComponent].biasFactor[newBin] - SystemComponents.Lambda[SelectedComponent].biasFactor[oldBin];
    TMMCPacc = 1.0; //Zhao's note: for a lambda change, it is not changing the macrostate, set Pacc to 1.0
    if (get_random_from_zero_to_one() < std::exp(-SystemComponents.Beta * energy.total() + biasTerm))
    {
      SystemComponents.Moves[SelectedComponent].CBCFLambdaAccepted ++;
      //update_translation_position<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
      update_CBCF_scale<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, start_position, SelectedComponent, newScale);
      SystemComponents.Moves[SelectedComponent].CBCFAccepted ++;
      SystemComponents.Lambda[SelectedComponent].currentBin = newBin;
      if(!FF.noCharges)
      {
        Update_Ewald_Vector(Sims.Box, false, SystemComponents);
      }
      final_energy = energy;
      SystemComponents.deltaVDWReal += SystemComponents.tempdeltaVDWReal; SystemComponents.deltaEwald += SystemComponents.tempdeltaEwald;
      SystemComponents.deltaTailE   += SystemComponents.tempdeltaTailE;
    }
  }
  SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, MoveType);
  return final_energy;
}
