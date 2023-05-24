#include "mc_utilities.h"

__device__ void get_nsim(Simulations* Sims, size_t *nsim, size_t *blockbefore, size_t NumberOfSimulations, size_t BlockID)
{
  size_t totalblock = 0; size_t tempsim = 0;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(BlockID >= (totalblock + Sims[i].Nblocks))
    {
      tempsim++;
      totalblock += Sims[i].Nblocks;
    }
  }
  *nsim        = tempsim;
  *blockbefore = totalblock;
}

__device__ void get_position(const Atoms* System, size_t *posi, size_t *comp, size_t i, size_t NumberComp)
{
  size_t temppos = i; size_t totalsize = 0; size_t tempcomp = 0;
  for(size_t ijk = 0; ijk < NumberComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(temppos >= totalsize)
    {
      tempcomp++;
      temppos -= System[ijk].size;
    }
  }
  *posi = temppos;
  *comp = tempcomp;
}

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

  switch (MoveType)
  {
    case TRANSLATION:
    {
      SystemComponents.Moves[SelectedComponent].TranslationTotal++;
      Do_New = true; Do_Old = true;
      break;
    }
    case ROTATION:
    {
      SystemComponents.Moves[SelectedComponent].RotationTotal++;
      Do_New = true; Do_Old = true;
      break;
    }
    case SINGLE_INSERTION:
    {
      SystemComponents.Moves[SelectedComponent].InsertionTotal++;
      Do_New = true;
      start_position = 0;
      break;
    } 
    case SINGLE_DELETION:
    {
      SystemComponents.Moves[SelectedComponent].DeletionTotal++;
      Do_Old = true;
      break;
    }
  }
  if(!Do_New && !Do_Old) throw std::runtime_error("Doing Nothing For Single Particle Move?\n");

  //Zhao's note: possible bug, you may only need 3 instead of 3 * N random numbers//
  Random.Check(Molsize);
  get_new_position<<<1, Molsize>>>(Sims, FF, start_position, SelectedComponent, Random.device_random, Random.offset, MoveType);
  Random.Update(Molsize);

  // Setup for the pairwise calculation //
  // New Features: divide the Blocks into two parts: Host-Guest + Guest-Guest //
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom *  Molsize, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * Molsize, &GG_Nblock, &GG_Nthread);
  size_t HGGG_Nthread = std::max(HG_Nthread, GG_Nthread);
  size_t HGGG_Nblock  = HG_Nblock + GG_Nblock;
  //printf("Total_Comp: %zu, Host Comp: %zu, Adsorbate Comp: %zu\n", SystemComponents.NComponents.x, SystemComponents.NComponents.y, SystemComponents.NComponents.z);
  //printf("NHostAtom: %zu, HG_Nblock: %zu, NGuestAtom: %zu, GG_Nblock: %zu\n", NHostAtom, HG_Nblock, NGuestAtom, GG_Nblock);
  Calculate_Single_Body_Energy_SEPARATE_HostGuest_VDWReal<<<HGGG_Nblock, HGGG_Nthread, HGGG_Nthread * 2 * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Molsize, Sims.device_flag, HG_Nblock, GG_Nblock, Do_New, Do_Old, SystemComponents.NComponents);
  cudaMemcpy(Sims.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);

  MoveEnergy tot; bool Accept = false; double Pacc = 0.0;
  if(!Sims.flag[0])
  {
    double HGGG_BlockResult[HGGG_Nblock + HGGG_Nblock];
    cudaMemcpy(HGGG_BlockResult, Sims.Blocksum, 2 * HGGG_Nblock * sizeof(double), cudaMemcpyDeviceToHost);
    double hg_vdw = 0.0; double gg_vdw = 0.0;
    double hg_real= 0.0; double gg_real= 0.0;
    for(size_t i = 0; i < HG_Nblock; i++)           hg_vdw += HGGG_BlockResult[i];
    for(size_t i = HG_Nblock; i < HGGG_Nblock; i++) gg_vdw += HGGG_BlockResult[i];

    for(size_t i = HGGG_Nblock; i < HG_Nblock + HGGG_Nblock; i++)
      hg_real += HGGG_BlockResult[i];
    for(size_t i = HGGG_Nblock + HG_Nblock; i < HGGG_Nblock + HGGG_Nblock; i++)
      gg_real += HGGG_BlockResult[i];
    /*
    printf("HG_NBlock: %zu\n", HGGG_Nblock);
    printf("Separated VDW : %.5f (HG), %.5f (GG)\n", hg_vdw , gg_vdw);
    printf("Separated Real: %.5f (HG), %.5f (GG)\n", hg_real, gg_real);
    */
    tot.HGVDW = hg_vdw; tot.HGReal = hg_real; tot.GGVDW = gg_vdw; tot.GGReal = gg_real;

    // Calculate Ewald //
    bool EwaldPerformed = false;
    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
      tot.EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType, 0, newScale);
      tot.HGEwaldE=SystemComponents.tempdeltaHGEwald;
      EwaldPerformed = true;
    }
    if(SystemComponents.UseDNNforHostGuest)
    {
      //Calculate DNN//
      if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents, SelectedComponent, MoveType, 0);
      double DNN_New = 0.0; double DNN_Old = 0.0;
      if(Do_New) DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, NEW);
      if(Do_Old) DNN_Old = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, OLD);

      tot.DNN_E = DNN_New - DNN_Old;
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
      SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, MoveType);
      SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, MoveType);
    }
    //if(MoveType == SINGLE_INSERTION) printf("SINGLE INSERTION, tot: %.5f, preFactor: %.5f, Pacc: %.5f\n", tot.total(), preFactor, Pacc);
    //if(MoveType == SINGLE_DELETION)  printf("SINGLE DELETION,  tot: %.5f, preFactor: %.5f, Pacc: %.5f\n", tot.total(), preFactor, Pacc);
    if(Get_Uniform_Random() < preFactor * std::exp(-SystemComponents.Beta * tot.total())) Accept = true;
  }

  switch(MoveType)
  {
    case TRANSLATION:
    { 
      if(Accept)
      {
        update_translation_position<<<1,Molsize>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
        SystemComponents.Moves[SelectedComponent].TranslationAccepted ++;
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents);
        }
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
      break;
    }
    case ROTATION:
    {
      if(Accept)
      {
        update_translation_position<<<1,Molsize>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
        SystemComponents.Moves[SelectedComponent].RotationAccepted ++;
        if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
        {
          Update_Ewald_Vector(Sims.Box, false, SystemComponents);
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
        SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
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
        SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
        size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
        AcceptDeletion(SystemComponents, Sims, SelectedComponent, UpdateLocation, SelectedMolInComponent, FF.noCharges);
      }
      else {tot.zero(); };
      SystemComponents.Tmmc[SelectedComponent].Update(Pacc, NMol, MoveType);
      break;
    }
  }
  return tot;
}
