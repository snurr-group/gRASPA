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
    energy += temp_energy;
  }
  //Determine whether to accept or reject the insertion
  preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, INSERTION);
  //Ewald Correction, done on HOST (CPU) //
  bool EwaldCPU = false;
  int MoveType = INSERTION; //Normal Insertion, including fractional insertion, no previous step (do not use temprorary totalEik)//
  if(previous_step) //Fractional Insertion after a lambda change move that makes the old fractional molecule full//
  {
    MoveType = CBCF_INSERTION;  // CBCF fractional insertion //
  }
  bool EwaldPerformed = false;
  if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
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
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE + energy.HGEwaldE));
    EwaldPerformed = true;
  }
  double TailE = 0.0;
  TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, MoveType);
  Rosenbluth *= std::exp(-SystemComponents.Beta * TailE);
  energy.TailE= TailE;
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents, SelectedComponent, MoveType, SelectedTrial);
    double DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, DNN_INSERTION);
    energy.DNN_E   = DNN_New;
    double correction = energy.DNN_Correction();
    if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
      //printf("INSERTION: Bad Prediction, reject the move!!!\n"); 
      SystemComponents.InsertionDNNReject ++;
      SuccessConstruction = false;
      WriteOutliers(SystemComponents, Sims, DNN_INSERTION, energy, correction);
      energy.zero();
      return energy;
    }
    SystemComponents.InsertionDNNDrift += fabs(correction);
    Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
  }
  //printf("Insertion energy summary: "); energy.print();
  return energy;
}

static inline MoveEnergy Deletion_Body(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, size_t& UpdateLocation, double& Rosenbluth, bool& SuccessConstruction, double& preFactor, double2 newScale)
{
  size_t SelectedTrial = 0;
  MoveEnergy energy; 
  

  double StoredR = 0.0; //Don't use this for Deletion//
  int CBMCType = CBMC_DELETION; //Deletion//
  //Zhao's note: Deletion_body will be part the GibbsParticleTransferMove, and the fractional molecule might be selected, so newScale will not be 1.0//
  //double2 newScale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0), Zhao's note: This is not used in deletion, just set to 1//
  Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale);
  if(Rosenbluth <= 1e-150) SuccessConstruction = false;
  if(!SuccessConstruction)
  {
    energy.zero();
    return energy;
  }  

  //DEBUG//
  /*
  if(SystemComponents.CURRENTCYCLE == 28981)
  {
    printf("DELETION FIRST BEAD ENERGY: "); energy.print();
    printf("EXCLUSION LIST: %d %d\n", Sims.ExcludeList[0].x, Sims.ExcludeList[0].y);
  }
  */

  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); //The false is for Reinsertion//
    energy += temp_energy;
  }
  if(Rosenbluth <= 1e-150) SuccessConstruction = false;
  if(!SuccessConstruction)
  {
    energy.zero();
    return energy;
  }

  preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, DELETION);

  UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
  double EwaldE = 0.0;
  bool EwaldPerformed = false;
  if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
  {
    EwaldE      = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, DELETION, UpdateLocation, newScale);
    Rosenbluth /= std::exp(-SystemComponents.Beta * (EwaldE + SystemComponents.tempdeltaHGEwald));
    energy.EwaldE= -1.0 * (EwaldE);
    energy.HGEwaldE= -1.0*SystemComponents.tempdeltaHGEwald; //Becareful with the sign here, you need a HG sum, but HGVDWReal and HGEwaldE here have opposite signs???//
    EwaldPerformed = true;
  }
  double TailE = 0.0;
  TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, DELETION);
  Rosenbluth /= std::exp(-SystemComponents.Beta * TailE);
  energy.TailE = -TailE;
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents, SelectedComponent, DELETION, UpdateLocation);
    double DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, DNN_DELETION);
    energy.DNN_E   = DNN_New;
    double correction = energy.DNN_Correction();
    if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
      //printf("DELETION: Bad Prediction, reject the move!!!\n"); 
      SystemComponents.DeletionDNNReject ++;
      SuccessConstruction = false;
      WriteOutliers(SystemComponents, Sims, DNN_DELETION, energy, correction);
      energy.zero();
      return energy;
    }
    SystemComponents.DeletionDNNDrift += fabs(correction);
    Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
  }
  return energy;
}
