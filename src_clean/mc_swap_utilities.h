#include "mc_utilities.h"  // For CheckBlockedPosition

inline MoveEnergy Insertion_Body(Variables& Vars, size_t systemId, CBMC_Variables& CBMC)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims            = Vars.Sims[systemId];
  ForceField& FF               = Vars.device_FF;

  MoveEnergy energy;
  CBMC.clear();
  CBMC.MoveType = CBMC_INSERTION; //Insertion//
 
  double& preFactor = SystemComponents.TempVal.preFactor;
 
  size_t& SelectedComponent = SystemComponents.TempVal.component;
  
  double& Rosenbluth = CBMC.Rosenbluth;
  
  Widom_Move_FirstBead_PARTIAL(Vars, systemId, CBMC); //Not reinsertion, not Retrace//

  if(Rosenbluth <= 1e-150) CBMC.SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
  if(!CBMC.SuccessConstruction)
  {
    // CRITICAL: If construction failed (blocked or high energy), ensure Rosenbluth is 0
    CBMC.Rosenbluth = 0.0;
    return energy;
  }
  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    Widom_Move_Chain_PARTIAL(Vars, systemId, CBMC);

    if(Rosenbluth <= 1e-150) CBMC.SuccessConstruction = false;
    if(!CBMC.SuccessConstruction)
    {
      // CRITICAL: If construction failed (blocked or high energy), ensure Rosenbluth is 0
      CBMC.Rosenbluth = 0.0;
      return energy;
    }
    
    // NOTE: Blocking check for CBMC INSERTION is done in Widom_Move_FirstBead_PARTIAL
    // Only first bead is checked (center-only, like RASPA2)
    // RASPA2 does NOT check all atoms after chain growth for regular CBMC INSERTION
    // Only REINSERTION checks all atoms after chain growth (line 4487-4491)
  }
  else if(SystemComponents.Moleculesize[SelectedComponent] == 1)
  {
    // For single atom molecules, blocking is already checked in Widom_Move_FirstBead_PARTIAL
    // No need to check again here
  }
  energy = CBMC.FirstBeadEnergy + CBMC.ChainEnergy;

  double2& Scale = SystemComponents.TempVal.Scale;
  //Determine whether to accept or reject the insertion
  preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, INSERTION);
  //Ewald Correction, done on HOST (CPU) //
  size_t SelectedTrial = CBMC.selectedTrial;
  if(SystemComponents.Moleculesize[SelectedComponent] > 1) SelectedTrial = CBMC.selectedTrialOrientation;
  bool EwaldPerformed = false;
  if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
  {
    double2 EwaldE = GPU_EwaldDifference_General(Sims, FF, SystemComponents, SelectedComponent, INSERTION, SelectedTrial, Scale);

    energy.GGEwaldE = EwaldE.x;
    energy.HGEwaldE = EwaldE.y;
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y));
    EwaldPerformed = true;
  }
  double TailE = 0.0;
  TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, SystemComponents.TempVal.MoveType);
  Rosenbluth *= std::exp(-SystemComponents.Beta * TailE);
  energy.TailE= TailE;
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents.tempMolStorage, SystemComponents, SelectedComponent, SystemComponents.TempVal.MoveType, SelectedTrial);
    double DNN_New = DNN_Prediction_Move(SystemComponents, Sims, SelectedComponent, INSERTION);
    energy.DNN_E   = DNN_New;

    energy.DNN_Replace_Energy();
    double correction = energy.DNN_Correction();
    if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
      //printf("INSERTION: Bad Prediction, reject the move!!!\n"); 
      SystemComponents.InsertionDNNReject ++;
      CBMC.SuccessConstruction = false;
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

inline MoveEnergy Deletion_Body(Variables& Vars, size_t systemId, CBMC_Variables& CBMC)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims            = Vars.Sims[systemId];
  ForceField& FF               = Vars.device_FF;
  //RandomNumber& Random         = Vars.Random;
  //WidomStruct& Widom           = Vars.Widom[systemId];

  size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
  size_t& SelectedComponent = SystemComponents.TempVal.component;

  //size_t SelectedTrial = 0;
  MoveEnergy energy; 
  
  CBMC.clear();
  CBMC.MoveType = CBMC_DELETION; //Deletion//
  double& Rosenbluth = CBMC.Rosenbluth;
  double& preFactor  = SystemComponents.TempVal.preFactor;
  //Zhao's note: Deletion_body will be part the GibbsParticleTransferMove, and the fractional molecule might be selected, so Scale will not be 1.0//
  //double2 Scale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0), Zhao's note: This is not used in deletion, just set to 1//
  double2& Scale = SystemComponents.TempVal.Scale;
  Widom_Move_FirstBead_PARTIAL(Vars, systemId, CBMC);
  if(Rosenbluth <= 1e-150) CBMC.SuccessConstruction = false;
  if(!CBMC.SuccessConstruction)
  {
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
    Widom_Move_Chain_PARTIAL(Vars, systemId, CBMC); //The false is for Reinsertion//
  }
  if(Rosenbluth <= 1e-150) CBMC.SuccessConstruction = false;
  if(!CBMC.SuccessConstruction)
  {
    return energy;
  }
  energy = CBMC.FirstBeadEnergy + CBMC.ChainEnergy;
  preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, DELETION);

  size_t& UpdateLocation = SystemComponents.TempVal.UpdateLocation; 
  UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
  bool EwaldPerformed = false;
  if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
  {
    int MoveType = DELETION; if(Scale.y < 1.0) MoveType = CBCF_DELETION;

    double2 EwaldE = GPU_EwaldDifference_General(Sims, FF, SystemComponents, SelectedComponent, MoveType, UpdateLocation, Scale);
    Rosenbluth /= std::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y));
    energy.GGEwaldE = -1.0 * EwaldE.x;
    energy.HGEwaldE = -1.0 * EwaldE.y; //Becareful with the sign here, you need a HG sum, but HGVDWReal and HGEwaldE here have opposite signs???//
    EwaldPerformed  = true;
  }
  double TailE = 0.0;
  TailE = TailCorrectionDifference(SystemComponents, SelectedComponent, FF.size, Sims.Box.Volume, DELETION);
  Rosenbluth /= std::exp(-SystemComponents.Beta * TailE);
  energy.TailE = -TailE;
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents.tempMolStorage, SystemComponents, SelectedComponent, DELETION, UpdateLocation);
    //Deletion positions stored in Old//
    double DNN_New = DNN_Prediction_Move(SystemComponents, Sims, SelectedComponent, DELETION);
    energy.DNN_E   = DNN_New;

    energy.DNN_Replace_Energy();
    double correction = energy.DNN_Correction();
    if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
      //printf("DELETION: Bad Prediction, reject the move!!!\n"); 
      SystemComponents.DeletionDNNReject ++;
      CBMC.SuccessConstruction = false;
      WriteOutliers(SystemComponents, Sims, DNN_DELETION, energy, correction);
      energy.zero();
      return energy;
    }
    SystemComponents.DeletionDNNDrift += fabs(correction);
    Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
  }
  return energy;
}
