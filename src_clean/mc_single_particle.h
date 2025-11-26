#include "mc_utilities.h"
#include "read_data.h"

////////////////////////////////////////////////
// Generalized function for single Body moves //
////////////////////////////////////////////////

//Zhao's note: decompose the single body move into different sections: preparation, calculation, and acceptance //
//For easier manipulation of moves, to use for, for example, MLP //
inline void SingleBody_Prepare(Variables& Vars, size_t systemId)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims            = Vars.Sims[systemId];
  ForceField& FF               = Vars.device_FF;
  RandomNumber& Random         = Vars.Random;
  WidomStruct& Widom           = Vars.Widom[systemId];

  size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
  size_t& SelectedComponent      = SystemComponents.TempVal.component;
  int&    MoveType               = SystemComponents.TempVal.MoveType;

  bool& Do_New  = SystemComponents.TempVal.Do_New;
  bool& Do_Old  = SystemComponents.TempVal.Do_Old;
  Do_New = false;
  Do_Old = false;

  SystemComponents.TempVal.CheckOverlap = true;

  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  //Set up Old position and New position arrays
  if(Molsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t& start_position = SystemComponents.TempVal.start_position; 
  start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];

  SystemComponents.Moves[SelectedComponent].Record_Move_Total(MoveType);
  double3 MaxChange = {0.0, 0.0, 0.0};
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
      SystemComponents.TempVal.CheckOverlap = false;
      //Zhao's note: if we separate framework components, there might be lots of overlaps between different species (node and linker overlaps), we can turn this Overlap flag off//
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
  
  // Check block pockets for all moves that create new positions (matching RASPA2)
  // RASPA2 checks BlockedPocket for all atoms in TrialPosition for translation, rotation, etc.
  if(Do_New && 
     SelectedComponent < SystemComponents.UseBlockPockets.size() && 
     SystemComponents.UseBlockPockets[SelectedComponent])
  {
    // Ensure statistics vectors are large enough
    if(SelectedComponent >= SystemComponents.BlockPocketTotalAttempts.size())
    {
      SystemComponents.BlockPocketTotalAttempts.resize(SelectedComponent + 1, 0);
      SystemComponents.BlockPocketBlockedCount.resize(SelectedComponent + 1, 0);
    }
    
    // Check all atoms in the molecule (matching RASPA2: for(i=0;i<nr_atoms;i++) if(BlockedPocket(TrialPosition[i])) return 0;)
    std::vector<double3> trial_positions(Molsize);
    cudaMemcpy(trial_positions.data(), &Sims.New.pos[start_position], Molsize * sizeof(double3), cudaMemcpyDeviceToHost);
    
    for(size_t i = 0; i < Molsize; i++)
    {
      if(BlockedPocket(SystemComponents, SelectedComponent, trial_positions[i], Sims.Box))
      {
        // Block the move (set flag to indicate overlap/blocked)
        bool blocked = true;
        cudaMemcpy(Sims.device_flag, &blocked, sizeof(bool), cudaMemcpyHostToDevice);
        return; // Early return, matching RASPA2 behavior
      }
    }
  }
}

inline MoveEnergy SingleBody_Calculation(Variables& Vars, size_t systemId)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims            = Vars.Sims[systemId];
  ForceField& FF               = Vars.device_FF;
  WidomStruct& Widom           = Vars.Widom[systemId];

  //size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
  size_t& SelectedComponent      = SystemComponents.TempVal.component;
  int&    MoveType               = SystemComponents.TempVal.MoveType;

  bool& CheckOverlap = SystemComponents.TempVal.CheckOverlap; //CheckOverlap = true;
  bool& Do_New  = SystemComponents.TempVal.Do_New;
  bool& Do_Old  = SystemComponents.TempVal.Do_Old;

  size_t Molsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  // Setup for the pairwise calculation //
  // New Features: divide the Blocks into two parts: Host-Guest + Guest-Guest //
  
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  //Zhao's note: Cross term, if the selected species is host atom, the crossAtom = guest, vice versa//
  size_t NCrossAtom = NHostAtom;
  if(SelectedComponent < SystemComponents.NComponents.y) //Framework component//
    NCrossAtom = NGuestAtom;
  size_t HH_Nthread=0; size_t HH_Nblock=0; Setup_threadblock(NHostAtom *  Molsize, HH_Nblock, HH_Nthread);
  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NCrossAtom * Molsize, HG_Nblock, HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * Molsize, GG_Nblock, GG_Nthread);

  size_t SameTypeNthread = 0;
  if(SelectedComponent < SystemComponents.NComponents.y) //Framework-Framework + Framework-Adsorbate//
  {GG_Nthread = 0; GG_Nblock = 0; SameTypeNthread = HH_Nthread; }
  else //Framework-Adsorbate + Adsorbate-Adsorbate//
  {HH_Nthread = 0; HH_Nblock = 0; SameTypeNthread = GG_Nthread; }

  size_t Nthread = std::max(SameTypeNthread, HG_Nthread);
  size_t Total_Nblock  = HH_Nblock + HG_Nblock + GG_Nblock;

  int3 NBlocks = {(int) HH_Nblock, (int) HG_Nblock, (int) GG_Nblock}; //x: HH_Nblock, y: HG_Nblock, z: GG_Nblock;
  //printf("Total_Comp: %zu, Host Comp: %zu, Adsorbate Comp: %zu\n", SystemComponents.NComponents.x, SystemComponents.NComponents.y, SystemComponents.NComponents.z);
  //printf("NHostAtom: %zu, HH_Nblock: %zu, HG_Nblock: %zu, NGuestAtom: %zu, GG_Nblock: %zu\n", NHostAtom, HH_Nblock, HG_Nblock, NGuestAtom, GG_Nblock);
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.NComponents.x; ijk++)
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];

  if(Atomsize != 0)
  {
    Calculate_Single_Body_Energy_VDWReal<<<Total_Nblock, Nthread, Nthread * 2 * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Molsize, Sims.device_flag, NBlocks, Do_New, Do_Old, SystemComponents.NComponents);

    SystemComponents.flag = Sims.device_flag;
    cudaDeviceSynchronize();
  }
  MoveEnergy tot; 
  if(!SystemComponents.flag[0] || !CheckOverlap)
  {
    //VDW Part and Real Part Coulomb//
    for(size_t i = 0; i < HH_Nblock; i++) 
    {
      tot.HHVDW += Sims.Blocksum[i];
      tot.HHReal+= Sims.Blocksum[i + Total_Nblock];
      //if(MoveType == SPECIAL_ROTATION) printf("HH Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }
    for(size_t i = HH_Nblock; i < HH_Nblock + HG_Nblock; i++) 
    {
      tot.HGVDW += Sims.Blocksum[i];
      tot.HGReal+= Sims.Blocksum[i + Total_Nblock];
      //printf("HG Block %zu, VDW: %.5f, Real: %.5f\n", i, BlockResult[i], BlockResult[i + Total_Nblock]);
    }
    for(size_t i = HH_Nblock + HG_Nblock; i < Total_Nblock; i++)
    {
      tot.GGVDW += Sims.Blocksum[i];
      tot.GGReal+= Sims.Blocksum[i + Total_Nblock];
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
      double2 EwaldE = GPU_EwaldDifference_General(Sims, FF, SystemComponents, SelectedComponent, MoveType, 0, newScale);
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
      if(!EwaldPerformed) Prepare_DNN_InitialPositions(Sims.d_a, Sims.New, Sims.Old, SystemComponents.tempMolStorage, SystemComponents, SelectedComponent, MoveType, 0);
      tot.DNN_E = DNN_Prediction_Move(SystemComponents, Sims, SelectedComponent, MoveType);
      tot.DNN_Replace_Energy();
    }
    double& preFactor = SystemComponents.TempVal.preFactor;
    double& Pacc      = SystemComponents.TempVal.Pacc;
    preFactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, MoveType);
    Pacc      = preFactor * std::exp(-SystemComponents.Beta * tot.total());
    SystemComponents.TempVal.Pacc = Pacc;
  }
  return tot;
}

inline void SingleBody_Acceptance(Variables& Vars, size_t systemId, MoveEnergy& tot)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  //Simulations& Sims            = Vars.Sims[systemId];
  //ForceField& FF               = Vars.device_FF;
  WidomStruct& Widom           = Vars.Widom[systemId];

  //size_t& SelectedMolInComponent = SystemComponents.TempVal.molecule;
  size_t& SelectedComponent      = SystemComponents.TempVal.component;
  int&    MoveType               = SystemComponents.TempVal.MoveType;

  //Get Number of Molecules for this component (For updating TMMC)//

  double& Pacc      = SystemComponents.TempVal.Pacc;
  bool& Accept = SystemComponents.TempVal.Accept;
  Accept = false;
  //no overlap, or don't check overlap (for special moves)//
  if(!SystemComponents.flag[0] || !SystemComponents.TempVal.CheckOverlap)
  {
    double Random = Get_Uniform_Random();
    if(Random < Pacc) Accept = true;
    //printf("Random: %.5f, Accept: %s\n", Accept ? "True" : "False");
  }

  if(Accept)
  {
    switch(MoveType)
    {
      case TRANSLATION: case ROTATION: case SPECIAL_ROTATION:
      {
        AcceptTranslation(Vars, systemId);
        break;
      }
      case SINGLE_INSERTION:
      {
        AcceptInsertion(Vars, Vars.SystemComponents[systemId].CBMC_New[0], systemId, SINGLE_INSERTION);
        break;
      }
      case SINGLE_DELETION:
      {
        AcceptDeletion(Vars, systemId, SINGLE_DELETION);
        break;
      }
    }
    SystemComponents.Moves[SelectedComponent].Record_Move_Accept(MoveType);
  }
  else
  {
    tot.zero();
  }
}

MoveEnergy SingleBodyMove(Variables& Vars, size_t systemId)
{
  SingleBody_Prepare(Vars, systemId);
  //Calculates prefactor, and Pacc (unbiased)//
  MoveEnergy tot = SingleBody_Calculation(Vars, systemId);
  //DNN: Check for DNN correction, if drifts too much from classical (typically bad prediction), reject//
  if(Vars.SystemComponents[systemId].UseDNNforHostGuest)
  { 
    bool DNN_Drift = Check_DNN_Drift(Vars, systemId, tot);
    if(DNN_Drift) 
    {
      tot.zero();
      return tot;
    }
  }
  //TMMC: Apply the bias according to the macrostate and check for out of bound, apply bias to Pacc//
  Vars.SystemComponents[systemId].ApplyTMMCBias_UpdateCMatrix(Vars.SystemComponents[systemId].TempVal.Pacc, Vars.SystemComponents[systemId].TempVal.MoveType);
  SingleBody_Acceptance(Vars, systemId, tot);
  return tot;
}
