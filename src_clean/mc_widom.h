#include <algorithm>
#include <omp.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
static inline size_t SelectTrialPosition(std::vector <double> LogBoltzmannFactors) //In Zhao's code, LogBoltzmannFactors = Rosen
{
    std::vector<double> ShiftedBoltzmannFactors(LogBoltzmannFactors.size());

    // Energies are always bounded from below [-U_max, infinity>
    // Find the lowest energy value, i.e. the largest value of (-Beta U)
    double largest_value = *std::max_element(LogBoltzmannFactors.begin(), LogBoltzmannFactors.end());

    // Standard trick: shift the Boltzmann factors down to avoid numerical problems
    // The largest value of 'ShiftedBoltzmannFactors' will be 1 (which corresponds to the lowest energy).
    double SumShiftedBoltzmannFactors = 0.0;
    for (size_t i = 0; i < LogBoltzmannFactors.size(); ++i)
    {
        ShiftedBoltzmannFactors[i] = exp(LogBoltzmannFactors[i] - largest_value);
        SumShiftedBoltzmannFactors += ShiftedBoltzmannFactors[i];
    }

    // select the Boltzmann factor
    size_t selected = 0;
    double cumw = ShiftedBoltzmannFactors[0];
    double ws = Get_Uniform_Random() * SumShiftedBoltzmannFactors;
    while (cumw < ws)
        cumw += ShiftedBoltzmannFactors[++selected];

    return selected;
}

template<typename T>
inline void Host_sum_Widom_HGGG_SEPARATE(size_t NumberWidomTrials, double Beta, T* energy_array, bool* flag, size_t HG_Nblock, size_t HGGG_Nblock, std::vector<MoveEnergy>& energies, std::vector<size_t>& Trialindex, std::vector<double>& Rosen, size_t Cycle, bool VDWRealBiasing)
{
  std::vector<size_t>reasonable_trials;
  for(size_t i = 0; i < NumberWidomTrials; i++)
  {
    if(!flag[i])
    {
      reasonable_trials.push_back(i);
    }
    //else printf("THERE IS OVERLAP FOR TRIAL %zu, flag = %s\n", i, flag[i] ? "true" : "false");
  }
  T host_array[HGGG_Nblock * 2];
  for(size_t i = 0; i < reasonable_trials.size(); i++)
  {
    size_t trial = reasonable_trials[i];
    cudaMemcpy(host_array, &energy_array[trial*HGGG_Nblock * 2], 2 * HGGG_Nblock*sizeof(T), cudaMemcpyDeviceToHost);
    T HG_vdw = 0.0; T HG_real = 0.0;
    T GG_vdw = 0.0; T GG_real = 0.0;
    //Zhao's note: If during the pairwise interaction, there is no overlap, then don't check overlap in the summation//
    //Otherwise, it will cause issues when using the fractional molecule (energy drift in retrace)//
    //So translation/rotation summation of energy are not checking the overlaps, so this is the reason for the discrepancy//
    for(size_t ijk=0; ijk < HG_Nblock; ijk++)           HG_vdw+=host_array[ijk];
    for(size_t ijk=HG_Nblock; ijk < HGGG_Nblock; ijk++) GG_vdw+=host_array[ijk];
    
    for(size_t ijk=HGGG_Nblock; ijk < HG_Nblock + HGGG_Nblock; ijk++) HG_real+=host_array[ijk];
    for(size_t ijk=HG_Nblock + HGGG_Nblock; ijk < HGGG_Nblock + HGGG_Nblock; ijk++) GG_real+=host_array[ijk];

    double tot = static_cast<double>(HG_vdw + GG_vdw);
    if(VDWRealBiasing) tot += static_cast<double>(HG_real + GG_real);
   
    MoveEnergy E;
    E.HGVDW = static_cast<double>(HG_vdw); 
    E.HGReal= static_cast<double>(HG_real);
    E.GGVDW = static_cast<double>(GG_vdw);
    E.GGReal= static_cast<double>(GG_real);
    //printf("trial: %zu, HG: %.5f, GG: %.5f, tot: %.5f\n", i, HG_vdw + HG_real, GG_vdw + GG_real, tot);
    energies.push_back(E);
    Trialindex.push_back(trial);
    Rosen.push_back(-Beta*tot);
    //printf("Trial %zu ", trial); E.print();
  }
}

__global__ void get_random_trial_position(Boxsize Box, Atoms* d_a, Atoms NewMol, bool* device_flag, RandomNumber Random, size_t start_position, size_t SelectedComponent, size_t MolID, int MoveType, double2 proposed_scale)
{
  double3* random = Random.device_random;
  size_t   offset = Random.offset;

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t random_index = i + offset;
  const Atoms AllData = d_a[SelectedComponent];
  double scale=0.0; double scaleCoul = 0.0;
 
  switch(MoveType)
  {
    case CBMC_INSERTION: //Insertion (whole and fractional molecule)//
    {
      scale = proposed_scale.x; scaleCoul = proposed_scale.y;
      double3 BoxLength = {Box.Cell[0], Box.Cell[4], Box.Cell[8]};
      NewMol.pos[i] = BoxLength * random[random_index];
      break;
    }
    case CBMC_DELETION: //Deletion (whole and fractional molecule)//
    {
      scale = AllData.scale[start_position]; scaleCoul = AllData.scaleCoul[start_position];
      if(i==0) //if deletion, the first trial position is the old position of the selected molecule//
      {
        NewMol.pos[i] = AllData.pos[start_position];
      }
      else
      {
        double3 BoxLength = {Box.Cell[0], Box.Cell[4], Box.Cell[8]};
        NewMol.pos[i] = BoxLength * random[random_index];
      }
      //printf("FIRSTBEAD: trial: %lu, xyz: %.5f %.5f %.5f\n", i, NewMol.pos[i].x, NewMol.pos[i].y, NewMol.pos[i].z);
      //if(i == 0) printf("i=0, start_position: %lu\n", start_position);
      break;
    }
    case REINSERTION_INSERTION: //Reinsertion-Insertion//
    {
      scale     = AllData.scale[start_position];
      scaleCoul = AllData.scaleCoul[start_position];
      double3 BoxLength = {Box.Cell[0], Box.Cell[4], Box.Cell[8]};
      NewMol.pos[i] = BoxLength * random[random_index];
      break;
    }
    case REINSERTION_RETRACE: //Reinsertion-Retrace//
    {
      scale = AllData.scale[start_position]; scaleCoul = AllData.scaleCoul[start_position];
      if(i==0) //if deletion, the first trial position is the old position of the selected molecule//
      {
        NewMol.pos[i] = AllData.pos[start_position];
      }
      else //Zhao's note: not necessarily correct, retrace only needs 1 trial, no else statement needed//
      {
        double3 BoxLength = {Box.Cell[0], Box.Cell[4], Box.Cell[8]};
        NewMol.pos[i] = BoxLength * random[random_index];
      }
      break;
    }
    case IDENTITY_SWAP_NEW: //IDENTITY-SWAP, NEW CONFIGURATION//
    {
      //xyz = the first bead of the molecule being identity swapped, already taken care of outside of the function//
      scale = proposed_scale.x; scaleCoul = proposed_scale.y;
      break;
    }
    case IDENTITY_SWAP_OLD:
    {
      scale = AllData.scale[start_position]; scaleCoul = AllData.scaleCoul[start_position];
      if(i==0) //if deletion, the first trial position is the old position of the selected molecule//
      {
        NewMol.pos[i] = AllData.pos[start_position];
      }
    }
  }
  NewMol.scale[i] = scale; 
  NewMol.charge[i] = AllData.charge[start_position]; 
  NewMol.scaleCoul[i] = scaleCoul; 
  NewMol.Type[i] = AllData.Type[start_position]; 
  NewMol.MolID[i] = MolID;
  /*
  if(MoveType == IDENTITY_SWAP_NEW)
  {
    printf("scale: %.5f charge: %.5f scaleCoul: %.5f, Type: %lu, MolID: %lu\n", NewMol.scale[i], NewMol.charge[i],  NewMol.scaleCoul[i] , NewMol.Type[i], NewMol.MolID[i] );
  }
  */
  /*
  if(MoveType == CBMC_INSERTION)
  {
    printf("NEWPOS: %.5f %.5f %.5f, scale: %.5f charge: %.5f scaleCoul: %.5f, Type: %lu, MolID: %lu\n", NewMol.pos[i].x, NewMol.pos[i].y, NewMol.pos[i].z, NewMol.scale[i], NewMol.charge[i],  NewMol.scaleCoul[i] , NewMol.Type[i], NewMol.MolID[i]);
  }
  */

  device_flag[i] = false;
}

__global__ void get_random_trial_orientation(Boxsize Box, Atoms* d_a, Atoms Mol, Atoms NewMol, bool* device_flag, double3* random, size_t offset, size_t FirstBeadTrial, size_t start_position, size_t SelectedComponent, size_t MolID, size_t chainsize, int MoveType, double2 proposed_scale, size_t Cycle)
{
  //Zhao's note: for trial orientations, each orientation may have more than 1 atom, do a for loop. So the threads are for different trial orientations, rather than different atoms in different orientations//
  //Zhao's note: chainsize is the size of the molecule excluding the first bead(-1).
  //Zhao's note: caveat: I am assuming the first bead of insertion is the first bead defined in def file. Need to relax this assumption in the future//
  //Zhao's note: First bead position stored in Mol, at position zero
  //Insertion/Widom: MolID = NewValue (Component.NumberOfMolecule_for_Component[SelectedComponent])
  //Deletion: MolID = selected ID
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ double3 OldPos;
  __shared__ double  Chosenscale;
  __shared__ double  ChosenscaleCoul;
  //Record First Bead Information//
  //For all orientations//
  if(i == 0)
  {
    Mol.charge[0]    = NewMol.charge[FirstBeadTrial];
    Chosenscale      = NewMol.scale[FirstBeadTrial];
    ChosenscaleCoul  = NewMol.scaleCoul[FirstBeadTrial];
    Mol.scale[0]     = Chosenscale;
    Mol.scaleCoul[0] = ChosenscaleCoul;
    Mol.Type[0]      = NewMol.Type[FirstBeadTrial];
    Mol.MolID[0]     = NewMol.MolID[FirstBeadTrial];
    OldPos           = NewMol.pos[FirstBeadTrial];
    Mol.pos[0]       = OldPos;
  }
  __syncthreads();
  size_t trial = i / chainsize;
  size_t a     = i % chainsize;
  //Quaternions uses 3 random seeds//
  size_t random_index = trial + offset;
  const Atoms& AllData = d_a[SelectedComponent];
  //different from translation (where we copy a whole molecule), here we duplicate the properties of the first bead of a molecule
  // so use start_position, not real_pos
  //Zhao's note: when there are zero molecule for the species, we need to use some preset values
  //the first values always have some numbers. The xyz are not correct, but type and charge are correct. Use those.
  //double scale = 0.0; double scaleCoul = 0.0;
  //for(size_t a = 0; a < chainsize; a++)
  //{
    double3 Vec;
    Vec = AllData.pos[1+a] - AllData.pos[0];
    switch(MoveType)
    {
      /////////////////////////////////////////////////////////////////////////////////////////////////
      //Zhao's note: It depends on whether Identity_swap needs to be operated on fractional molecules//
      //FOR NOW, JUST PUT IT ALONGSIDE WITH CBMC_INSERTION                                           //
      /////////////////////////////////////////////////////////////////////////////////////////////////
      case CBMC_INSERTION: case IDENTITY_SWAP_NEW: case REINSERTION_INSERTION: //Insertion (whole/fractional Molecule)//
      {
        //scale = proposed_scale.x; scaleCoul = proposed_scale.y;
        Rotate_Quaternions(Vec, random[random_index]);
        NewMol.pos[i] = OldPos + Vec;
        break;
      }
      /*
      case REINSERTION_INSERTION: //Reinsertion-Insertion//
      {
        //scale = AllData.scale[start_position+a]; scaleCoul = AllData.scaleCoul[start_position+a];
        Rotate_Quaternions(Vec, random[random_index]);
        NewMol.pos[i] = OldPos + Vec;
        break;
      }
      */
      case CBMC_DELETION: //Deletion (whole/fractional molecule)//
      case REINSERTION_RETRACE: case IDENTITY_SWAP_OLD: //Reinsertion-Retrace, but also works for Identity swap (old molecule) for multiple orientations (not first bead) //
      {
        //scale = AllData.scale[start_position+a]; scaleCoul = AllData.scaleCoul[start_position+a];
        if(trial==0) //if deletion, the first trial position is the old position of the selected molecule//
        {
          NewMol.pos[i] = AllData.pos[start_position+a];
        }
        else
        {
          Rotate_Quaternions(Vec, random[random_index]);
          NewMol.pos[i] = OldPos + Vec;
        }
        break;
      }
    }
    NewMol.scale[i]     = Chosenscale;
    NewMol.charge[i]    = AllData.charge[start_position+a];
    NewMol.scaleCoul[i] = ChosenscaleCoul;
    NewMol.Type[i]      = AllData.Type[start_position+a];
    NewMol.MolID[i]     = MolID;
  //}
  if(a == 0)
    device_flag[trial] = false;
}


static inline double Widom_Move_FirstBead_PARTIAL(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType, double &StoredR, size_t *REAL_Selected_Trial, bool *SuccessConstruction, MoveEnergy *energy, double2 proposed_scale)
{

  bool Goodconstruction = false; size_t SelectedTrial = 0; double Rosenbluth = 0.0;
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.NComponents.x; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  std::vector<double>Rosen; std::vector<MoveEnergy>energies; std::vector<size_t>Trialindex;
 
  //Three variables to settle: NumberOfTrials, start_position, SelectedMolID
  //start_position: where to copy data from
  //SelectedMolID : MolID to assign to the new molecule 
  size_t NumberOfTrials = Widom.NumberWidomTrials;
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  size_t SelectedMolID  = SelectedMolInComponent;
  switch(MoveType)
  {
    case CBMC_INSERTION:
    {
      start_position = 0;
      SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
      break; 
    }
    case CBMC_DELETION: case REINSERTION_INSERTION: 
    {
      break; //Go With Default//
    }
    case REINSERTION_RETRACE: case IDENTITY_SWAP_OLD:
    {
      NumberOfTrials = 1;
      
      break;
    }
    case IDENTITY_SWAP_NEW:
    {
      NumberOfTrials = 1;
      start_position = 0;
      SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
      break;
    }
  }
  size_t threadsNeeded = Atomsize * NumberOfTrials;

  Random.Check(NumberOfTrials);

  //printf("MoveType: %d, Ntrial: %zu, SelectedMolID: %zu, start_position: %zu, selectedComponent: %zu\n", MoveType, NumberOfTrials, SelectedMolID, start_position, SelectedComponent);

  get_random_trial_position<<<1,NumberOfTrials>>>(Sims.Box, Sims.d_a, Sims.New, Sims.device_flag, Random, start_position, SelectedComponent, SelectedMolID, MoveType, proposed_scale); checkCUDAError("error getting random trials");
  Random.Update(NumberOfTrials);

  //printf("Selected Component: %zu, Selected Molecule: %zu (%zu), Total in Component: %zu\n", SelectedComponent, SelectedMolID, SelectedMolInComponent, SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);

  // Setup the pairwise calculation //
  // Setup Number of Blocks and threads for separated HG + GG calculations //
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom  * 1, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * 1, &GG_Nblock, &GG_Nthread);
  size_t HGGG_Nthread = std::max(HG_Nthread, GG_Nthread);
  size_t HGGG_Nblock  = HG_Nblock + GG_Nblock;
  if(Atomsize != 0)
  {
    Calculate_Multiple_Trial_Energy_VDWReal<<<HGGG_Nblock * NumberOfTrials, HGGG_Nthread, 2 * HGGG_Nthread * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Sims.device_flag, threadsNeeded,1, HGGG_Nblock, HG_Nblock, SystemComponents.NComponents, Sims.ExcludeList); checkCUDAError("Error calculating energies (PARTIAL SUM HGGG)");
    cudaMemcpy(SystemComponents.flag, Sims.device_flag, NumberOfTrials*sizeof(bool), cudaMemcpyDeviceToHost);
  }
  //printf("OldNBlock: %zu, HG_Nblock: %zu, GG_Nblock: %zu, HGGG_Nblock: %zu\n", Nblock, HG_Nblock, GG_Nblock, HGGG_Nblock);

  //printf("FIRST BEAD ENERGIES\n");
  Host_sum_Widom_HGGG_SEPARATE(NumberOfTrials, SystemComponents.Beta, Sims.Blocksum, SystemComponents.flag, HG_Nblock, HGGG_Nblock, energies, Trialindex, Rosen, SystemComponents.CURRENTCYCLE, FF.VDWRealBias);

  double averagedRosen = 0.0;
  size_t REALselected  = 0;
  //Zhao's note: The final part of the CBMC seems complicated, using a switch may help understand how each case is processed//
  switch(MoveType)
  {
    case CBMC_INSERTION: case REINSERTION_INSERTION:
    {
      if(Rosen.size() == 0) break;
      SelectedTrial = SelectTrialPosition(Rosen);
      for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
      Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
      if(Rosenbluth < 1e-150) break;
      Goodconstruction = true;
      break;
    }
    case IDENTITY_SWAP_NEW:
    {
      if(Rosen.size() == 0) break;
      SelectedTrial = 0;
      for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
      Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
      if(Rosenbluth < 1e-150) break;
      Goodconstruction = true;
      break;
    }
    case CBMC_DELETION: case REINSERTION_RETRACE: case IDENTITY_SWAP_OLD:
    {
      SelectedTrial = 0;
      for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
      Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
      Goodconstruction = true;
      break;
    }
  }

  if(!Goodconstruction) return 0.0;
  REALselected = Trialindex[SelectedTrial];

  if(MoveType == REINSERTION_INSERTION) StoredR = Rosenbluth - Rosen[SelectedTrial];
  if(MoveType == REINSERTION_RETRACE) Rosenbluth += StoredR;
  //For 1st bead for identity swap, there is only 1 bead used for insertion/retrace, don't divide.//
  averagedRosen = Rosenbluth;
  if(MoveType != IDENTITY_SWAP_OLD && MoveType != IDENTITY_SWAP_NEW) 
    averagedRosen /= double(Widom.NumberWidomTrials);

  //Calculate Rosenbluth correction if real-charge is not included when selecting the trials//
  if(!FF.VDWRealBias)
  {
    double Real_Energy = energies[SelectedTrial].HGReal + energies[SelectedTrial].GGReal;
    averagedRosen*=std::exp(-SystemComponents.Beta*Real_Energy);
  }

  *REAL_Selected_Trial = REALselected;
  *SuccessConstruction = Goodconstruction;
  *energy = energies[SelectedTrial];
  return averagedRosen;
}

static inline double Widom_Move_Chain_PARTIAL(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType, size_t *REAL_Selected_Trial, bool *SuccessConstruction, MoveEnergy *energy, size_t FirstBeadTrial, double2 proposed_scale)
{
  MoveEnergy TEMP;
  *energy = TEMP;
  //printf("DOING RANDOM ORIENTAITONS\n");
  size_t Atomsize = 0;
  size_t chainsize = SystemComponents.Moleculesize[SelectedComponent]-1; //size for the data of trial orientations are the number of trial orientations times the size of molecule excluding the first bead//
  bool Goodconstruction = false; size_t SelectedTrial = 0; double Rosenbluth = 0.0;

  // here, the Atomsize is the total number of atoms in the system
  for(size_t ijk = 0; ijk < SystemComponents.NComponents.x; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }

  std::vector<double>Rosen; std::vector<MoveEnergy>energies; std::vector<size_t>Trialindex;

  //Two variables to settle: start_position, SelectedMolID
  //start_position: where to copy data from
  //SelectedMolID : MolID to assign to the new molecule
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent] + 1;
  size_t SelectedMolID  = SelectedMolInComponent;
  switch(MoveType)
  {
    case CBMC_INSERTION:
    {
      start_position = 1;
      SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
      break;
    }
    case CBMC_DELETION: case REINSERTION_INSERTION: case REINSERTION_RETRACE: case IDENTITY_SWAP_OLD:
    {
      break; //Go With Default//
    }
    case IDENTITY_SWAP_NEW:
    {
      start_position = 1;
      SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
      break;
    }
  }
  // number of threads needed = Atomsize*Widom.NumberWidomTrialsOrientations;
  size_t threadsNeeded = Atomsize*Widom.NumberWidomTrialsOrientations*chainsize;

  Random.Check(Widom.NumberWidomTrialsOrientations);
  //Get the first bead positions, and setup trial orientations//
  size_t trialO_Nthread=0; size_t trialO_Nblock=0; Setup_threadblock(Widom.NumberWidomTrialsOrientations*chainsize, &trialO_Nblock, &trialO_Nthread);
  get_random_trial_orientation<<<trialO_Nblock,trialO_Nthread>>>(Sims.Box, Sims.d_a, Sims.Old, Sims.New, Sims.device_flag, Random.device_random, Random.offset, FirstBeadTrial, start_position, SelectedComponent, SelectedMolID, chainsize, MoveType, proposed_scale, SystemComponents.CURRENTCYCLE); checkCUDAError("error getting random trials orientations");

  Random.Update(Widom.NumberWidomTrialsOrientations);

  // Setup the pairwise calculation //
  // Setup Number of Blocks and threads for separated HG + GG calculations //
  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom  * chainsize, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * chainsize, &GG_Nblock, &GG_Nthread);
  size_t HGGG_Nthread = std::max(HG_Nthread, GG_Nthread);
  size_t HGGG_Nblock  = HG_Nblock + GG_Nblock;

  //Setup calculation for separated HG + GG interactions//
  if(Atomsize != 0)
  {
    Calculate_Multiple_Trial_Energy_VDWReal<<<HGGG_Nblock * Widom.NumberWidomTrialsOrientations, HGGG_Nthread, 2 * HGGG_Nthread * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Sims.device_flag, threadsNeeded, chainsize, HGGG_Nblock, HG_Nblock, SystemComponents.NComponents, Sims.ExcludeList); checkCUDAError("Error calculating energies (PARTIAL SUM HGGG Orientation)");

    cudaMemcpy(SystemComponents.flag, Sims.device_flag, Widom.NumberWidomTrialsOrientations*sizeof(bool), cudaMemcpyDeviceToHost);
  }
  //printf("CHAIN ENERGIES\n");

  Host_sum_Widom_HGGG_SEPARATE(Widom.NumberWidomTrialsOrientations, SystemComponents.Beta, Sims.Blocksum, SystemComponents.flag, HG_Nblock, HGGG_Nblock, energies, Trialindex, Rosen, SystemComponents.CURRENTCYCLE, FF.VDWRealBias);

  double averagedRosen= 0.0; 
  size_t REALselected = 0;

  //Zhao's note: The final part of the CBMC seems complicated, using a switch may help understand how each case is processed//
  switch(MoveType)
  {
    case CBMC_INSERTION: case REINSERTION_INSERTION: case IDENTITY_SWAP_NEW:
    {
      if(Rosen.size() == 0) break;
      SelectedTrial = SelectTrialPosition(Rosen); 

      for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
      Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
      if(Rosenbluth < 1e-150) break;
      Goodconstruction = true;
      break;
    }
    case CBMC_DELETION: case REINSERTION_RETRACE: case IDENTITY_SWAP_OLD:
    {
      SelectedTrial = 0;
      for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
      Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
      Goodconstruction = true;
      break;
    }
  }
  if(!Goodconstruction) return 0.0;
  REALselected = Trialindex[SelectedTrial];

  averagedRosen = Rosenbluth/double(Widom.NumberWidomTrialsOrientations);
  //Calculate Rosenbluth correction if real-charge is not included when selecting the trials//
  if(!FF.VDWRealBias)
  {
    double Real_Energy = energies[SelectedTrial].HGReal + energies[SelectedTrial].GGReal;
    averagedRosen*=std::exp(-SystemComponents.Beta*Real_Energy);
  }
  *REAL_Selected_Trial = REALselected;
  *SuccessConstruction = Goodconstruction;
  *energy = energies[SelectedTrial];
  //if(SystemComponents.CURRENTCYCLE == 687347) energies[SelectedTrial].print();
  return averagedRosen;
}
