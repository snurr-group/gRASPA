#include "mc_utilities.h"

////////////////////////////////////////////////
// Generalized function for single Body moves //
////////////////////////////////////////////////
static inline MoveEnergy SingleBodyMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;

  bool Do_New = false;
  bool Do_Old = false;

  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  //Set up Old position and New position arrays
  if(Molsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];

  SystemComponents.Moves[SelectedComponent].Record_Move_Total(MoveType);
  double3 MaxChange = {0.0, 0.0, 0.0};
  bool CheckOverlap = true;
  switch (MoveType)
  {
    case TRANSLATION:
    {
      Do_New = true; Do_Old = true;
      MaxChange = SystemComponents.MaxTranslation[SelectedComponent];
      break;
    }
    case ROTATION:
    {
      Do_New = true; Do_Old = true;
      MaxChange = SystemComponents.MaxRotation[SelectedComponent];
      break;
    }
    case SPECIAL_ROTATION:
    {
      Do_New = true; Do_Old = true;
      MaxChange = SystemComponents.MaxSpecialRotation[SelectedComponent];
      //Zhao's note: if we separate framework components, there might be lots of overlaps between different species (node and linker overlaps), we can turn this Overlap flag off//
      CheckOverlap = false;
      //printf("Performing move on %zu comp, %zu mol\n", SelectedComponent, SelectedMolInComponent);
      break;
    }
    case SINGLE_INSERTION:
    {
      Do_New = true;
      start_position = 0;
      break;
    } 
    case SINGLE_DELETION:
    {
      Do_Old = true;
      break;
    }
  }
  if(!Do_New && !Do_Old) throw std::runtime_error("Doing Nothing For Single Particle Move?\n");

  //Zhao's note: possible bug, you may only need 3 instead of 3 * N random numbers//
  Random.Check(Molsize);
  get_new_position<<<1, Molsize>>>(Sims, FF, start_position, SelectedComponent, MaxChange, Random.device_random, Random.offset, MoveType);
  Random.Update(Molsize);

  // Setup for the pairwise calculation //
  // New Features: divide the Blocks into two parts: Host-Guest + Guest-Guest //
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];


  size_t HH_Nthread=0; size_t HH_Nblock=0; Setup_threadblock(NHostAtom *  Molsize, &HH_Nblock, &HH_Nthread);
  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom *  Molsize, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * Molsize, &GG_Nblock, &GG_Nthread);
  
  size_t CrossTypeNthread = 0;
  if(SelectedComponent < SystemComponents.NComponents.y) //Framework-Framework + Framework-Adsorbate//
  {GG_Nthread = 0; GG_Nblock = 0; CrossTypeNthread = HH_Nthread; }
  else //Framework-Adsorbate + Adsorbate-Adsorbate//
  {HH_Nthread = 0; HH_Nblock = 0; CrossTypeNthread = GG_Nthread; }

  size_t Nthread = std::max(CrossTypeNthread, HG_Nthread);
  size_t Total_Nblock  = HH_Nblock + HG_Nblock + GG_Nblock;

  int3 NBlocks = {(int) HH_Nblock, (int) HG_Nblock, (int) GG_Nblock}; //x: HH_Nblock, y: HG_Nblock, z: GG_Nblock;
  //printf("Total_Comp: %zu, Host Comp: %zu, Adsorbate Comp: %zu\n", SystemComponents.NComponents.x, SystemComponents.NComponents.y, SystemComponents.NComponents.z);
  //printf("NHostAtom: %zu, HH_Nblock: %zu, HG_Nblock: %zu, NGuestAtom: %zu, GG_Nblock: %zu\n", NHostAtom, HH_Nblock, HG_Nblock, NGuestAtom, GG_Nblock);

  Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal<<<Total_Nblock, Nthread, Nthread * 2 * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Molsize, Sims.device_flag, NBlocks, Do_New, Do_Old, SystemComponents.NComponents);

  cudaMemcpy(SystemComponents.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);

  MoveEnergy tot; bool Accept = false; double Pacc = 0.0;
  if(!SystemComponents.flag[0] || !CheckOverlap)
  {
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
      //printf("HG Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }
    for(size_t i = HH_Nblock + HG_Nblock; i < Total_Nblock; i++)
    {
      tot.GGVDW += BlockResult[i];
      tot.GGReal+= BlockResult[i + Total_Nblock];
      //printf("GG Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }

    /*
    printf("HG_NBlock: %zu\n", Total_Nblock);
    printf("Separated VDW : %.5f (HH), %.5f (HG), %.5f (GG)\n", tot.HHVDW,  tot.HGVDW , tot.GGVDW);
    printf("Separated Real: %.5f (HH), %.5f (HG), %.5f (GG)\n", tot.HHReal, tot.HGReal, tot.GGReal);
    */

    // Calculate Ewald //
    bool EwaldPerformed = false;
    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
      double2 EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType, 0, newScale);
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
      EwaldPerformed = true;
    }
    if(SystemComponents.UseDNNforHostGuest)
    {
      //Calculate DNN//
      if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents, SelectedComponent, MoveType, 0);
      tot.DNN_E = DNN_Prediction_Move(SystemComponents, Sims, SelectedComponent, MoveType);
      double correction = tot.DNN_Correction(); //If use DNN, HGVDWReal and HGEwaldE are zeroed//
      if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
      {
        //printf("TRANSLATION/ROTATION: Bad Prediction, reject the move!!!\n");
        switch(MoveType)
        {
          case TRANSLATION: case ROTATION:
          {
            SystemComponents.TranslationRotationDNNReject ++; break;
          }
          case SINGLE_INSERTION: case SINGLE_DELETION:
          {
            SystemComponents.SingleSwapDNNReject ++; break;
          }
        }
        WriteOutliers(SystemComponents, Sims, NEW, tot, correction); //Print New Locations//
        WriteOutliers(SystemComponents, Sims, OLD, tot, correction); //Print Old Locations//
        tot.zero();
        return tot;
      }
      SystemComponents.SingleMoveDNNDrift += fabs(correction);
    }

    double preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, MoveType);
    Pacc = preFactor * std::exp(-SystemComponents.Beta * tot.total());

    //Apply the bias according to the macrostate//
    if(MoveType == SINGLE_INSERTION || MoveType == SINGLE_DELETION)
    {
      SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, NMol, MoveType);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, NMol, MoveType);
    }
    //if(MoveType == SINGLE_INSERTION) printf("SINGLE INSERTION, tot: %.5f, preFactor: %.5f, Pacc: %.5f\n", tot.total(), preFactor, Pacc);
    //if(MoveType == SINGLE_DELETION)  printf("SINGLE DELETION,  tot: %.5f, preFactor: %.5f, Pacc: %.5f\n", tot.total(), preFactor, Pacc);
    if(Get_Uniform_Random() < preFactor * std::exp(-SystemComponents.Beta * tot.total())) Accept = true;
  }

  //if(MoveType == SPECIAL_ROTATION)
  //  printf("Framework Component Move, %zu cycle, Molecule: %zu, Energy: %.5f, %s\n", SystemComponents.CURRENTCYCLE, SelectedMolInComponent, tot.total(), Accept ? "Accept" : "Reject");
  //if(MoveType == SPECIAL_ROTATION) Accept = true;

  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: case SPECIAL_ROTATION:
    { 
      if(Accept)
      {
        update_translation_position<<<1,Molsize>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
        SystemComponents.Moves[SelectedComponent].Record_Move_Accept(MoveType);
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents, SelectedComponent);
        }
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
      break;
    }
    case SINGLE_INSERTION:
    {
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, MoveType);
      if(Accept)
      {
        SystemComponents.Moves[SelectedComponent].Record_Move_Accept(MoveType);
        AcceptInsertion(SystemComponents, Sims, SelectedComponent, 0, FF.noCharges, SINGLE_INSERTION); //0: selectedTrial//
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(Pacc, NMol, MoveType);
      break;
    }
    case SINGLE_DELETION:
    {
      SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, MoveType);
      if(Accept)
      {
        SystemComponents.Moves[SelectedComponent].Record_Move_Accept(MoveType);
        size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
        AcceptDeletion(SystemComponents, Sims, SelectedComponent, UpdateLocation, SelectedMolInComponent, FF.noCharges);
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(Pacc, NMol, MoveType);
      break;
    }
  }
  //if(MoveType == SINGLE_INSERTION) {printf("Cycle %zu, ENERGY: ", SystemComponents.CURRENTCYCLE); tot.print();}
  return tot;
}
