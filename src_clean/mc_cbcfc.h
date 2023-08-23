__global__ void Prepare_LambdaChange(Atoms* d_a, Atoms Mol, Simulations Sims, ForceField FF, size_t start_position, size_t SelectedComponent, bool* device_flag)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t real_pos = start_position + i;

  const double3 pos       = d_a[SelectedComponent].pos[real_pos];
  const double  scale     = d_a[SelectedComponent].scale[real_pos];
  const double  charge    = d_a[SelectedComponent].charge[real_pos];
  const double  scaleCoul = d_a[SelectedComponent].scaleCoul[real_pos];
  const size_t  Type      = d_a[SelectedComponent].Type[real_pos];
  const size_t  MolID     = d_a[SelectedComponent].MolID[real_pos];

  //Atoms Temp = Sims.Old; Atoms TempNEW = Sims.New;

  Mol.pos[i] = pos;
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
  size_t Molsize;
  //Set up Old position and New position arrays
  Molsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  if(Molsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  Prepare_LambdaChange<<<1, Molsize>>>(Sims.d_a, Sims.Old, Sims, FF, start_position, SelectedComponent, Sims.device_flag);
  
  // Setup for the pairwise calculation //
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];


  size_t HH_Nthread=0; size_t HH_Nblock=0; Setup_threadblock(NHostAtom  * Molsize, &HH_Nblock, &HH_Nthread);
  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom  * Molsize, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * Molsize, &GG_Nblock, &GG_Nthread);

  size_t CrossTypeNthread = 0;
  if(SelectedComponent < SystemComponents.NComponents.y) //Framework-Framework + Framework-Adsorbate//
  {GG_Nthread = 0; GG_Nblock = 0; CrossTypeNthread = HH_Nthread; }
  else //Framework-Adsorbate + Adsorbate-Adsorbate//
  {HH_Nthread = 0; HH_Nblock = 0; CrossTypeNthread = GG_Nthread; }

  size_t Nthread = std::max(CrossTypeNthread, HG_Nthread);
  size_t Total_Nblock  = HH_Nblock + HG_Nblock + GG_Nblock;

  int3 NBlocks = {(int) HH_Nblock, (int) HG_Nblock, (int) GG_Nblock}; //x: HH_Nblock, y: HG_Nblock, z: GG_Nblock;
  bool Do_New = true; bool Do_Old = true;

  Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal_LambdaChange<<<Total_Nblock, Nthread, Nthread * 2 * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Molsize, Sims.device_flag, NBlocks, Do_New, Do_Old, SystemComponents.NComponents, newScale);

  cudaMemcpy(SystemComponents.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);

  if(!SystemComponents.flag[0])
  {
    SuccessConstruction = true;
    double BlockResult[Total_Nblock + Total_Nblock];
    cudaMemcpy(BlockResult, Sims.Blocksum, 2 * Total_Nblock * sizeof(double), cudaMemcpyDeviceToHost);

    //VDW Part and Real Part Coulomb//
    for(size_t i = 0; i < HH_Nblock; i++)
    {
      tot.HHVDW += BlockResult[i];
      tot.HHReal+= BlockResult[i + Total_Nblock];
      //if(MoveType == SPECIAL_ROTATION) printf("HH Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }
    for(size_t i = HH_Nblock; i < HH_Nblock + HG_Nblock; i++)
    {
      tot.HGVDW += BlockResult[i];
      tot.HGReal+= BlockResult[i + Total_Nblock];
      //if(SystemComponents.CURRENTCYCLE == 25) printf("HG Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }
    for(size_t i = HH_Nblock + HG_Nblock; i < Total_Nblock; i++)
    {
      tot.GGVDW += BlockResult[i];
      tot.GGReal+= BlockResult[i + Total_Nblock];
      //printf("GG Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }

    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      //Zhao's note: since we changed it from using translation/rotation functions to its own, this needs to be changed as well//
      double2 EwaldE = GPU_EwaldDifference_LambdaChange(Sims.Box, Sims.d_a, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, oldScale, newScale, MoveType);
      if(HH_Nblock == 0)
      {
        tot.GGEwaldE = EwaldE.x;
        tot.HGEwaldE = EwaldE.y;
      }
      else
      {
        tot.HHEwaldE = EwaldE.x;
        tot.HGEwaldE = EwaldE.y;
        //printf("HHEwald: %.5f, HGEwald: %.5f\n", tot.HHEwaldE, tot.HGEwaldE);
      }

    }
    tot.TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, MoveType);
  }
  else
  {
    SuccessConstruction = false;
  }
  //printf("Cycle: %zu, Newscale: %.5f/%.5f, Oldscale: %.5f/%.5f, Construction: %s, CBCF Lambda E", SystemComponents.CURRENTCYCLE, newScale.x, newScale.y, oldScale.x, oldScale.y, SuccessConstruction ? "SUCCESS" : "BAD"); tot.print();
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
      double3 pos = d_a[SelectedComponent].pos[UpdateLocation+i]; 
      double  scale = d_a[SelectedComponent].scale[UpdateLocation+i];
      double  charge = d_a[SelectedComponent].charge[UpdateLocation+i];
      double  scaleCoul = d_a[SelectedComponent].scaleCoul[UpdateLocation+i]; double Type = d_a[SelectedComponent].Type[UpdateLocation+i];
      d_a[SelectedComponent].pos[UpdateLocation+i]       = d_a[SelectedComponent].pos[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
      //Put the data of the old fractional molecule to LastLocation//
      d_a[SelectedComponent].pos[LastLocation+i] = pos; 
      d_a[SelectedComponent].scale[LastLocation+i] = scale;
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
      double3 pos = d_a[SelectedComponent].pos[UpdateLocation+i]; 
      double  scale = d_a[SelectedComponent].scale[UpdateLocation+i];
      double  charge = d_a[SelectedComponent].charge[UpdateLocation+i];
      double  scaleCoul = d_a[SelectedComponent].scaleCoul[UpdateLocation+i]; double Type = d_a[SelectedComponent].Type[UpdateLocation+i];
      d_a[SelectedComponent].pos[UpdateLocation+i]       = d_a[SelectedComponent].pos[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
      //Put the data of the old fractional molecule to LastLocation//
      d_a[SelectedComponent].pos[LastLocation+i] = pos; 
      d_a[SelectedComponent].scale[LastLocation+i] = scale;
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

  MoveEnergy energy;

  int SelectednewBin = selectNewBinTMMC(SystemComponents.Lambda[SelectedComponent]);
  //int SelectednewBin = 1;
  int Binchange = 1; //change for updating/getting bias for TMMC + CBCFC//
  //Zhao's note: if the new bin == oldbin, the scale is the same, no need for a move, so select a new lambda
  //Zhao's note: Do not delete the fractional molecule when there is only one molecule, the system needs to have at least one fractional molecule for each species
  while(SelectednewBin == oldBin || (SelectednewBin < 0 && SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] == 1))
  {
    if(SystemComponents.Tmmc[SelectedComponent].DoTMMC) //CBCFC + TMMC, can only move to adjacent bin//
    {
      //Binchange = 1;
      //if(Get_Uniform_Random() < 0.5) SelectednewBin = -1;//Binchange = -1;
      //SelectednewBin = Binchange + oldBin;
      SelectednewBin = selectNewBinTMMC(SystemComponents.Lambda[SelectedComponent]);
      //printf("in while, SelectednewBin: %d, oldBin: %zu\n", SelectednewBin, oldBin);
    }
    else //non-TMMC//
    {
      SelectednewBin = selectNewBin(SystemComponents.Lambda[SelectedComponent]);
    }
  }
  Binchange = SelectednewBin - oldBin;
  //printf("Binchange: %d, NewBin: %d, oldBin: %zu\n", Binchange, SelectednewBin, oldBin);
  //////////////////
  //INSERTION MOVE//
  //////////////////
  if(SelectednewBin > static_cast<int>(nbin))
  {
    MoveType = CBCF_INSERTION;
    
    //return 0.0;
    SystemComponents.Moves[SelectedComponent].CBCFInsertionTotal ++;
    int newBin = SelectednewBin - static_cast<int>(nbin); //printf("INSERTION, newBin: %zu\n", newBin);
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
    energy = CBCF_LambdaChange(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, oldScale, InterScale, start_position, CBCF_INSERTION, SuccessConstruction);
    if(!SuccessConstruction) 
    {
      //If unsuccessful move (Overlap), Pacc = 0//
      SystemComponents.Tmmc[SelectedComponent].UpdateCBCF(0.0, NMol, Binchange);
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
      energy += second_step_energy;
      //Account for the biasing terms of different lambdas//
      double biasTerm = SystemComponents.Lambda[SelectedComponent].biasFactor[newBin] - SystemComponents.Lambda[SelectedComponent].biasFactor[oldBin];
      preFactor *= std::exp(-SystemComponents.Beta * first_step_energy.total() + biasTerm);
      double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
      TMMCPacc = preFactor * Rosenbluth / IdealRosen; //Unbiased Acceptance//
      //Apply the bias according to the macrostate//
      SystemComponents.Tmmc[SelectedComponent].ApplyWLBiasCBCF(preFactor, NMol, Binchange);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBiasCBCF(preFactor, NMol, Binchange);

      if(Get_Uniform_Random() < preFactor * Rosenbluth / IdealRosen) Accepted = true;
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBoundCBCF(Accepted, NMol, Binchange);

      //printf("Insertion E: %.5f, Acc: %s\n", energy.total(), Accepted ? "Accept" : "Reject");

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
        if(SystemComponents.Tmmc[SelectedComponent].DoTMMC)
          SystemComponents.Tmmc[SelectedComponent].currentBin = newBin;
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents, SelectedComponent);
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
    int newBin = SelectednewBin + nbin; //printf("DELETION, newBin: %zu\n", newBin);
    double newLambda = delta * static_cast<double>(newBin);
    //Zhao's note: Set the Scaling factors for VDW and Coulombic Interaction depending on the mode//
    double2 oldScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(oldLambda);
    double2 InterScale= SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
    double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(newLambda);
    //////////////////////////////////////////////
    //First step: delete the fractional molecule//
    //////////////////////////////////////////////
    bool SuccessConstruction = false;
    MoveEnergy energy;
    double Rosenbluth = 0.0;
    double preFactor  = 0.0;
    size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
    energy = Deletion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, UpdateLocation, Rosenbluth, SuccessConstruction, preFactor, oldScale);

    //printf("Deletion First Step E: "); energy.print();
 
    //MoveEnergy first_step_energy = energy;
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
    SelectedMolInComponent = static_cast<size_t>(Get_Uniform_Random() * static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]));
    SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SelectedMolInComponent;

    bool Accepted = false;
    if(SuccessConstruction)
    {
      size_t start_position = 1;
      MoveEnergy second_step_energy;
      
      SuccessConstruction = false;
      second_step_energy = CBCF_LambdaChange(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, InterScale, newScale, start_position, CBCF_DELETION, SuccessConstruction); //Fractional deletion, lambda change is the second step (that uses the temporary tempEik vector)
      
      //Account for the biasing terms of different lambdas//
      double biasTerm = SystemComponents.Lambda[SelectedComponent].biasFactor[newBin] - SystemComponents.Lambda[SelectedComponent].biasFactor[oldBin];
      preFactor *= std::exp(-SystemComponents.Beta * second_step_energy.total() + biasTerm);
      double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
      TMMCPacc = preFactor * IdealRosen / Rosenbluth; //Unbiased Acceptance//
      //Apply the bias according to the macrostate//
      //SystemComponents.Tmmc[SelectedComponent].ApplyWLBiasCBCF(preFactor, NMol, Binchange);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBiasCBCF(preFactor, NMol, Binchange);

      if(Get_Uniform_Random() < preFactor * IdealRosen / Rosenbluth) Accepted = true;
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBoundCBCF(Accepted, NMol, Binchange);

      if(Accepted)
      {
        SystemComponents.Moves[SelectedComponent].CBCFDeletionAccepted ++;
        SystemComponents.Moves[SelectedComponent].CBCFAccepted ++;
        //update_translation_position<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
        update_CBCF_scale<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, start_position, SelectedComponent, newScale);
        SystemComponents.Lambda[SelectedComponent].currentBin = newBin;
        if(SystemComponents.Tmmc[SelectedComponent].DoTMMC)
          SystemComponents.Tmmc[SelectedComponent].currentBin = newBin;
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents, SelectedComponent);
        }
        energy.take_negative();
        energy += second_step_energy;
        final_energy = energy;
      }
    }
    if(!Accepted)
    {
      Revert_CBCF_Deletion<<<1,1>>>(Sims.d_a, Sims.New, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
      SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = OldFracMolID;
      SystemComponents.TotalNumberOfMolecules ++;
      SystemComponents.NumberOfMolecule_for_Component[SelectedComponent] ++;
      SystemComponents.UpdatePseudoAtoms(CBCF_INSERTION,  SelectedComponent);
    }
    //if(SystemComponents.CURRENTCYCLE == 917) printf("Deletion E: %.5f, Acc: %s\n", energy.total(), Accepted ? "Accept": "Reject");
  }
  else //Lambda Move//
  {
    MoveType = CBCF_LAMBDACHANGE;
    SystemComponents.Moves[SelectedComponent].CBCFLambdaTotal ++;
    int newBin = static_cast<size_t>(SelectednewBin); //printf("LAMBDACHANGE, newBin: %zu\n", newBin);
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
      SystemComponents.Tmmc[SelectedComponent].UpdateCBCF(0.0, NMol, Binchange);
      energy.zero();
      return energy;
    }
    //Account for the biasing terms of different lambdas//
    double biasTerm = SystemComponents.Lambda[SelectedComponent].biasFactor[newBin] - SystemComponents.Lambda[SelectedComponent].biasFactor[oldBin];
    TMMCPacc = std::exp(-SystemComponents.Beta * energy.total() + biasTerm);
    double preFactor = 1.0;
    SystemComponents.Tmmc[SelectedComponent].ApplyTMBiasCBCF(preFactor, NMol, Binchange);

    bool Accepted = false;
    if (Get_Uniform_Random() < preFactor * std::exp(-SystemComponents.Beta * energy.total() + biasTerm)) Accepted = true;
    SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBoundCBCF(Accepted, NMol, Binchange);

    if(Accepted)
    {
      SystemComponents.Moves[SelectedComponent].CBCFLambdaAccepted ++;
      //update_translation_position<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
      update_CBCF_scale<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, start_position, SelectedComponent, newScale);
      SystemComponents.Moves[SelectedComponent].CBCFAccepted ++;
      SystemComponents.Lambda[SelectedComponent].currentBin = newBin;
      if(SystemComponents.Tmmc[SelectedComponent].DoTMMC)
          SystemComponents.Tmmc[SelectedComponent].currentBin = newBin;
      if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
      {
        Update_Ewald_Vector(Sims.Box, false, SystemComponents, SelectedComponent);
      }
      final_energy = energy;
    }
  }
  SystemComponents.Tmmc[SelectedComponent].UpdateCBCF(TMMCPacc, NMol, Binchange);
  return final_energy;
}
