struct InsertionMove
{
  MoveEnergy energy;
  double preFactor = 0.0;
  double Pacc = 0.0;
  bool Accept = false;
  bool CreateMoleculePhase = false;
  CBMC_Variables InsertionVariables;
  void Initialize()
  {
    InsertionVariables.clear();
    energy.zero(); preFactor = 0.0; Pacc = 0.0; Accept = false;
    CreateMoleculePhase = false;
  }
  void Calculate(Variables& Vars, size_t systemId)
  {
    Simulations& Sims            = Vars.Sims[systemId];
    Components& SystemComponents = Vars.SystemComponents[systemId];
    size_t& SelectedComponent = SystemComponents.TempVal.component;
    //CBMC_Variables& InsertionVariables = SystemComponents.CBMC_New[0]; InsertionVariables.clear();
    energy = Insertion_Body(Vars, systemId, InsertionVariables);

    double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
    preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, INSERTION);
    Pacc = preFactor * InsertionVariables.Rosenbluth / IdealRosen;
  }
  void Acceptance(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];

    double RANDOM = 1e-100;
    if(!CreateMoleculePhase) RANDOM = Get_Uniform_Random();
    if(RANDOM < Pacc) Accept = true;

    if(Accept)
    {
      //SystemComponents.CBMC_New[0].selectedTrial = InsertionVariables.selectedTrial;
      //SystemComponents.CBMC_New[0].selectedTrialOrientation = InsertionVariables.selectedTrialOrientation;
      AcceptInsertion(Vars, InsertionVariables, systemId, INSERTION);
    }
    else
      energy.zero();
  }
  //Wrapper that combines Calculate and Acceptance//
  MoveEnergy Run(Variables& Vars, size_t systemId)
  {
    size_t& SelectedComponent = Vars.SystemComponents[systemId].TempVal.component;

    Initialize();
    Vars.SystemComponents[systemId].Moves[SelectedComponent].Record_Move_Total(INSERTION);
    Calculate(Vars, systemId);
    if(!InsertionVariables.SuccessConstruction)
    {
      Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Pacc, INSERTION);
      return energy;
    }
    //TMMC Adjust Pacc and record C-matrix//
    Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Pacc, INSERTION);

    Acceptance(Vars, systemId);
    if(Accept) Vars.SystemComponents[systemId].Moves[SelectedComponent].Record_Move_Accept(INSERTION);
    return energy;
  }
  double WidomMove(Variables& Vars, size_t systemId)
  {
    Initialize();
    Calculate(Vars, systemId);
    return InsertionVariables.Rosenbluth;
  }
  MoveEnergy CreateMolecule(Variables& Vars, size_t systemId)
  {
    Initialize();
    CreateMoleculePhase = true;
    Calculate(Vars, systemId);
    if(!InsertionVariables.SuccessConstruction)
    {
      return energy;
    }
    Acceptance(Vars, systemId);
    return energy;
  }
};

struct DeletionMove
{
  MoveEnergy energy;
  double preFactor = 0.0;
  double Pacc = 0.0;
  bool Accept = false;
  CBMC_Variables DeletionVariables;
  void Initialize()
  {
    DeletionVariables.clear();
    energy.zero(); preFactor = 0.0; Pacc = 0.0; Accept = false;
  }
  void Calculate(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    Simulations& Sims            = Vars.Sims[systemId];
    size_t& SelectedComponent    = SystemComponents.TempVal.component;

    energy = Deletion_Body(Vars, systemId, DeletionVariables);
    double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
    preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, DELETION);
    Pacc = preFactor * IdealRosen / DeletionVariables.Rosenbluth;
  }
  void Acceptance(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    size_t& SelectedComponent    = SystemComponents.TempVal.component;
    double RANDOM = Get_Uniform_Random();
    if(RANDOM < Pacc) Accept = true;
    if(Accept)
    { // accept the move
      SystemComponents.Moves[SelectedComponent].Record_Move_Accept(DELETION);
      AcceptDeletion(Vars, systemId, DELETION);
      energy.take_negative();
    }
    else
      energy.zero();
  }
  MoveEnergy Run(Variables& Vars, size_t systemId)
  {
    size_t& SelectedComponent = Vars.SystemComponents[systemId].TempVal.component;

    Initialize();
    Vars.SystemComponents[systemId].Moves[SelectedComponent].Record_Move_Total(DELETION);
    Calculate(Vars, systemId);
    if(!DeletionVariables.SuccessConstruction)
    {
      //If unsuccessful move (Overlap), Pacc = 0//
      Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Pacc, DELETION);
      energy.zero();
      return energy;
    }
    Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Pacc, DELETION);

    Acceptance(Vars, systemId);

    return energy;
  }
};

struct ReinsertionMove
{
  MoveEnergy energy;
  MoveEnergy old_energy;
  double preFactor = 0.0;
  double Pacc = 0.0;
  bool Accept = false;
  CBMC_Variables InsertionVariables;
  CBMC_Variables DeletionVariables;
  void Initialize()
  {
    InsertionVariables.clear(); DeletionVariables.clear();
    energy.zero(); old_energy.zero(); preFactor = 0.0; Pacc = 0.0; Accept = false;
  }
  void Calculate_Insertion(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    Simulations& Sims            = Vars.Sims[systemId];
    size_t& SelectedComponent = SystemComponents.TempVal.component;

    InsertionVariables.MoveType = REINSERTION_INSERTION; //Reinsertion-Insertion, storing first bead Rosenbluth for retrace (next step)// 
    SystemComponents.TempVal.Scale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Zhao's note: not used in reinsertion, just set to 1.0//
    double& Rosenbluth = InsertionVariables.Rosenbluth;
    Widom_Move_FirstBead_PARTIAL(Vars, systemId, InsertionVariables); //Not reinsertion, not Retrace//

    if(Rosenbluth <= 1e-150) InsertionVariables.SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer

    if(!InsertionVariables.SuccessConstruction)
    {
      return;
    }

    if(SystemComponents.Moleculesize[SelectedComponent] > 1)
    {
      Widom_Move_Chain_PARTIAL(Vars, systemId, InsertionVariables); //True for doing insertion for reinsertion, different in MoleculeID//
      if(Rosenbluth <= 1e-150) InsertionVariables.SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
    }
    //Store the inserted molecule//
    if(InsertionVariables.SuccessConstruction)
    {
      size_t SelectedTrial = InsertionVariables.selectedTrial;
      if(SystemComponents.Moleculesize[SelectedComponent] > 1) SelectedTrial = InsertionVariables.selectedTrialOrientation;

      //Store The New Locations//
      StoreNewLocation_Reinsertion<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.Old, Sims.New, SystemComponents.tempMolStorage, SelectedTrial, SystemComponents.Moleculesize[SelectedComponent]);

      energy = InsertionVariables.FirstBeadEnergy + InsertionVariables.ChainEnergy;
    }
  }
  void Calculate_Deletion(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    size_t& SelectedComponent = SystemComponents.TempVal.component;
  
    DeletionVariables.MoveType = REINSERTION_RETRACE; //Reinsertion-Retrace//
    //Transfer StoredR, the rosenbluth weights already calculated for the first bead during insertion//
    DeletionVariables.StoredR = InsertionVariables.StoredR;
    Widom_Move_FirstBead_PARTIAL(Vars, systemId, DeletionVariables);

    if(SystemComponents.Moleculesize[SelectedComponent] > 1)
    {
      Widom_Move_Chain_PARTIAL(Vars, systemId, DeletionVariables);
    }
    old_energy = DeletionVariables.FirstBeadEnergy + DeletionVariables.ChainEnergy;
  }
  void Calculate_AdjustRosenbluth(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    Simulations& Sims            = Vars.Sims[systemId];
    ForceField& FF               = Vars.device_FF;
    size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
    size_t& SelectedComponent = SystemComponents.TempVal.component;
  
    double& Rosenbluth = InsertionVariables.Rosenbluth;
  
    //Calculate Ewald//
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
    
    bool EwaldPerformed = false;
    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      double2 EwaldE = GPU_EwaldDifference_General(Sims, FF, SystemComponents, SelectedComponent, REINSERTION, UpdateLocation, {1.0, 1.0});
      //double2 EwaldE = GPU_EwaldDifference_Reinsertion(Sims.Box, Sims.d_a, Sims.Old, SystemComponents.tempMolStorage, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);

      energy.GGEwaldE = EwaldE.x;
      energy.HGEwaldE = EwaldE.y;
      Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y));
      EwaldPerformed = true;
    }
    //Calculate DNN//
    //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
    if(SystemComponents.UseDNNforHostGuest)
    {
      if(!EwaldPerformed)
        Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents.tempMolStorage, SystemComponents, SelectedComponent, REINSERTION, UpdateLocation);
      //Prepare_DNN_InitialPositions_Reinsertion(Sims.d_a, Sims.Old, SystemComponents.tempMolStorage, SystemComponents, SelectedComponent, UpdateLocation);
      energy.DNN_E = DNN_Prediction_Reinsertion(SystemComponents, Sims, SelectedComponent, SystemComponents.tempMolStorage);
      //Correction of DNN - HostGuest energy to the Rosenbluth weight//
      energy.DNN_Replace_Energy();
      double correction = energy.DNN_Correction();
      if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
      {
          SystemComponents.ReinsertionDNNReject++;
          //WriteOutliers(SystemComponents, Sims, REINSERTION_NEW, energy, correction);
          //WriteOutliers(SystemComponents, Sims, REINSERTION_OLD, energy, correction);
          energy.zero();
      }
      SystemComponents.ReinsertionDNNDrift += fabs(correction);
      Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
    }
  }
  void Acceptance(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    Simulations& Sims            = Vars.Sims[systemId];
    ForceField& FF               = Vars.device_FF;
    //size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
    size_t& SelectedComponent = SystemComponents.TempVal.component;

    double RANDOM = Get_Uniform_Random();
    Pacc = InsertionVariables.Rosenbluth / DeletionVariables.Rosenbluth;

    if(RANDOM < Pacc)
    { // accept the move
      SystemComponents.Moves[SelectedComponent].Record_Move_Accept(REINSERTION);
      size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.TempVal.molecule;
      Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, SystemComponents.tempMolStorage, SelectedComponent, UpdateLocation); checkCUDAError("error Updating Reinsertion data");

      if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        Update_Vector_Ewald(Sims.Box, false, SystemComponents, SelectedComponent);
      //energy.print();
    }
    else
    energy.zero();
  }
  MoveEnergy Run(Variables& Vars, size_t systemId)
  {
    Components& SystemComponents = Vars.SystemComponents[systemId];
    size_t& SelectedComponent = SystemComponents.TempVal.component;

    Initialize();

    SystemComponents.Moves[SelectedComponent].Record_Move_Total(REINSERTION);
    Calculate_Insertion(Vars, systemId);
    if(!InsertionVariables.SuccessConstruction) 
    {
      Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Pacc, REINSERTION);
      return MoveEnergy();
    }

    //RETRACE//
    Calculate_Deletion(Vars, systemId);
    //if(!DeletionVariables.SuccessConstruction) return MoveEnergy();
    energy -= old_energy;

    Calculate_AdjustRosenbluth(Vars, systemId);
 
    Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Pacc, REINSERTION);
 
    Acceptance(Vars, systemId);

    if(Accept) SystemComponents.Moves[SelectedComponent].Record_Move_Accept(REINSERTION);

    return energy;
  }
};

struct GibbsParticleXferMove
{
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  
  MoveEnergy energy;
  MoveEnergy old_energy;
  double Insertion_preFactor = 0.0;
  double Deletion_preFactor = 0.0;
  double Pacc = 0.0;
  bool Accept = false;
  CBMC_Variables InsertionVariables;
  CBMC_Variables DeletionVariables;

  void Initialize()
  {
    SelectedBox = 0; OtherBox = 1;
    energy.zero(); old_energy.zero();
    Insertion_preFactor = 0.0; Deletion_preFactor = 0.0;
    Pacc = 0.0; Accept = false;
    InsertionVariables.clear(); DeletionVariables.clear();
  }
  void Prepare_SelectBoxMolecule(Variables& Vars, size_t systemId)
  {
    std::vector<Components>& SystemComponents = Vars.SystemComponents;
    size_t SelectedComponent = SystemComponents[systemId].TempVal.component;
    for(size_t i = 0; i < SystemComponents.size(); i++) SystemComponents[i].TempVal.component = SelectedComponent;

    if(Get_Uniform_Random() > 0.5)
    {
      SelectedBox = 1;
      OtherBox    = 0;
    }
    if(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent] < 1) return;
    size_t InsertionSelectedMol = (size_t) (Get_Uniform_Random()*(SystemComponents[SelectedBox].NumberOfMolecule_for_Component[SelectedComponent]));
    size_t DeletionSelectedMol  = (size_t) (Get_Uniform_Random()*(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent]));

    SystemComponents[SelectedBox].TempVal.molecule = InsertionSelectedMol;
    SystemComponents[OtherBox].TempVal.molecule    = DeletionSelectedMol;
  }
  void Calculate_Insertion(Variables& Vars, size_t systemId)
  {
    energy = Insertion_Body(Vars, SelectedBox, InsertionVariables);
  }
  void Calculate_Deletion(Variables& Vars, size_t systemId)
  {
    old_energy = Deletion_Body(Vars, OtherBox, DeletionVariables);
  }
  void Acceptance(Variables& Vars, size_t systemId)
  {
    std::vector<Components>& SystemComponents = Vars.SystemComponents;
    Simulations*& Sims                        = Vars.Sims;
    size_t NMolA = 0;
    size_t NMolB = 0;
    for(size_t comp = SystemComponents[SelectedBox].NComponents.y; comp < SystemComponents[SelectedBox].NComponents.x; comp++)
      NMolA += SystemComponents[SelectedBox].NumberOfMolecule_for_Component[comp];

    for(size_t comp = SystemComponents[OtherBox].NComponents.y; comp < SystemComponents[OtherBox].NComponents.x; comp++)
      NMolB += SystemComponents[OtherBox].NumberOfMolecule_for_Component[comp];

    //Minus the fractional molecule//
    for(size_t comp = 0; comp < SystemComponents[SelectedBox].NComponents.x; comp++)
    {
      if(SystemComponents[SelectedBox].hasfractionalMolecule[comp])
      {
        NMolA-=1;
      }
    }
    for(size_t comp = 0; comp < SystemComponents[OtherBox].NComponents.x; comp++)
    {
      if(SystemComponents[OtherBox].hasfractionalMolecule[comp])
      {
        NMolB-=1;
      }
    }

    Pacc = (InsertionVariables.Rosenbluth * static_cast<double>(NMolB) * Sims[SelectedBox].Box.Volume) / (DeletionVariables.Rosenbluth * static_cast<double>(NMolA + 1) * Sims[OtherBox].Box.Volume);
    if(Get_Uniform_Random()< Pacc) Accept = true;
    if(Accept)
    {
      AcceptInsertion(Vars, InsertionVariables, SelectedBox, INSERTION);
      AcceptDeletion(Vars, OtherBox, DELETION);
      SystemComponents[SelectedBox].deltaE += energy;
      SystemComponents[OtherBox].deltaE    -= old_energy;
    }
  }
  void Run(Variables& Vars, size_t systemId, Gibbs& GibbsStatistics)
  {
    GibbsStatistics.GibbsXferStats.x += 1;

    Initialize();

    Prepare_SelectBoxMolecule(Vars, systemId);
    
    Calculate_Insertion(Vars, systemId);
    if(!InsertionVariables.SuccessConstruction) return;
 
    Calculate_Deletion(Vars, systemId);
    if(!DeletionVariables.SuccessConstruction) return;

    Acceptance(Vars, systemId);
    
    if(Accept) GibbsStatistics.GibbsXferStats.y += 1;
  }
};

struct MC_MOVES
{
  InsertionMove   INSERTION;
  DeletionMove    DELETION;
  ReinsertionMove REINSERTION;
  
  GibbsParticleXferMove GIBBS_PARTICLE_XFER;
};
