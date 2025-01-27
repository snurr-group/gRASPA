#include "mc_widom.h"
#include "mc_swap_utilities.h"
#include "lambda.h"
#include "mc_cbcfc.h"
/*
__global__ void StoreNewLocation_Reinsertion(Atoms Mol, Atoms NewMol, double3* temp, size_t SelectedTrial, size_t Moleculesize)
{
  if(Moleculesize == 1) //Only first bead is inserted, first bead data is stored in NewMol
  {
    temp[0] = NewMol.pos[SelectedTrial];
  }
  else //Multiple beads: first bead + trial orientations
  {
    //Update the first bead, first bead data stored in position 0 of Mol //
    temp[0] = Mol.pos[0];
   
    size_t chainsize = Moleculesize - 1; // FOr trial orientations //
    for(size_t i = 0; i < chainsize; i++) //Update the selected orientations//
    {
      size_t selectsize = SelectedTrial*chainsize+i;
      temp[i+1] = NewMol.pos[selectsize];
    }
  }
}
*/
__global__ void StoreNewLocation_Reinsertion(Atoms Mol, Atoms NewMol, double3* temp, size_t SelectedTrial, size_t Moleculesize)
{
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i == 0)
  {
    if(Moleculesize == 1) temp[0] = NewMol.pos[SelectedTrial];
    else temp[0] = Mol.pos[0];
  }
  else
  {
    size_t chainsize = Moleculesize - 1; // FOr trial orientations //
    size_t selectsize = SelectedTrial*chainsize+(i-1);
    temp[i] = NewMol.pos[selectsize];
  }
}


__global__ void Update_Reinsertion_data(Atoms* d_a, double3* temp, size_t SelectedComponent, size_t UpdateLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t realLocation = UpdateLocation + i;
  d_a[SelectedComponent].pos[realLocation] = temp[i];
}

static inline void GibbsParticleTransfer(Variables& Vars, size_t SelectedComponent, Gibbs& GibbsStatistics)
{
  std::vector<Components>& SystemComponents = Vars.SystemComponents;
  Simulations*& Sims                        = Vars.Sims;
  //RandomNumber& Random                      = Vars.Random;
  //std::vector<WidomStruct>& Widom           = Vars.Widom;

  size_t NBox = SystemComponents.size();
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  GibbsStatistics.GibbsXferStats.x += 1;

  if(Get_Uniform_Random() > 0.5)
  {
    SelectedBox = 1;
    OtherBox    = 0;
  }
  
  double2& Scale = SystemComponents[SelectedBox].TempVal.Scale;
  Scale = SystemComponents[SelectedBox].Lambda[SelectedComponent].SET_SCALE(1.0);
  bool TransferFractionalMolecule = false;
  //Randomly Select a Molecule from the OtherBox in the SelectedComponent//
  if(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent] < 1) return;
  size_t InsertionSelectedMol = (size_t) (Get_Uniform_Random()*(SystemComponents[SelectedBox].NumberOfMolecule_for_Component[SelectedComponent]));
  size_t DeletionSelectedMol  = (size_t) (Get_Uniform_Random()*(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent]));
  //Special treatment for transfering the fractional molecule//
  if(SystemComponents[OtherBox].hasfractionalMolecule[SelectedComponent] && SystemComponents[OtherBox].Lambda[SelectedComponent].FractionalMoleculeID == DeletionSelectedMol)
  {
    double oldBin    = SystemComponents[OtherBox].Lambda[SelectedComponent].currentBin;
    double delta     = SystemComponents[OtherBox].Lambda[SelectedComponent].delta;
    double oldLambda = delta * static_cast<double>(oldBin);
    Scale            = SystemComponents[OtherBox].Lambda[SelectedComponent].SET_SCALE(oldLambda);
    TransferFractionalMolecule = true;
  }

  //printf("Passing FIRST return\n");
  SystemComponents[SelectedBox].TempVal.molecule = InsertionSelectedMol;
  SystemComponents[SelectedBox].TempVal.component= SelectedComponent;

  //Perform Insertion on the selected System, then deletion on the other system//

  /////////////////////////////////////////////////
  // PERFORMING INSERTION ON THE SELECTED SYSTEM //
  /////////////////////////////////////////////////
  double& InsertionPrefactor = SystemComponents[SelectedBox].TempVal.preFactor;
  CBMC_Variables& InsertionVariables = SystemComponents[SelectedBox].CBMC_New[0];
  InsertionVariables.clear();
  //size_t InsertionSelectedMol   = 0; //It is safer to ALWAYS choose the first atom as the template for CBMC_INSERTION//
  MoveEnergy InsertionEnergy = Insertion_Body(Vars, SelectedBox, InsertionVariables);

  //printf("InsertionRosen: %.5f, InsertionSuccess: %s, RN offset: %zu\n", InsertionRosen, InsertionSuccess ? "true" : "false", Random.offset);

  if(!InsertionVariables.SuccessConstruction) return;

  //printf("Passing SECOND return\n");
  /////////////////////////////////////////////
  // PERFORMING DELETION ON THE OTHER SYSTEM //
  /////////////////////////////////////////////
  SystemComponents[OtherBox].TempVal.molecule = DeletionSelectedMol;
  SystemComponents[OtherBox].TempVal.component= SelectedComponent;

  CBMC_Variables& DeletionVariables = SystemComponents[OtherBox].CBMC_Old[0];
  DeletionVariables.clear();
  MoveEnergy DeletionEnergy = Deletion_Body(Vars, OtherBox, DeletionVariables);
  if(!DeletionVariables.SuccessConstruction) return;

  bool Accept = false;

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

  double PAcc = (InsertionVariables.Rosenbluth * static_cast<double>(NMolB) * Sims[SelectedBox].Box.Volume) / (DeletionVariables.Rosenbluth * static_cast<double>(NMolA + 1) * Sims[OtherBox].Box.Volume);
  //This assumes that the two boxes share the same temperature, it might not be true//
  if(Get_Uniform_Random()< PAcc) Accept = true;

  //printf("CYCLE: %zu, Insertion box: %zu, delete box: %zu, delete molecule: %zu, InsertionRosen: %.5f, DeletionRosen: %.5f, PAcc: %.5f\n", SystemComponents[0].CURRENTCYCLE, SelectedBox, OtherBox, DeletionSelectedMol, InsertionRosen, DeletionRosen, PAcc);

  if(Accept)
  {
    GibbsStatistics.GibbsXferStats.y += 1;
    // Zhao's note: the assumption for the below implementation is that the component index are the same for both systems //
    // for example, methane in box A is component 1, it has to be component 1 in box B //
    //For the box that is deleting the molecule, update the recorded fractional molecule ID//
    // Update System information regarding the fractional molecule, if the fractional molecule is being transfered //
    if(TransferFractionalMolecule)
    {
      SystemComponents[SelectedBox].hasfractionalMolecule[SelectedComponent] = true;
      SystemComponents[OtherBox].hasfractionalMolecule[SelectedComponent]    = false;
      SystemComponents[SelectedBox].Lambda[SelectedComponent].currentBin = SystemComponents[OtherBox].Lambda[SelectedComponent].currentBin;
    }
    ///////////////////////////////////////////
    // UPDATE INSERTION FOR THE SELECTED BOX //
    ///////////////////////////////////////////
    AcceptInsertion(Vars, Vars.SystemComponents[SelectedBox].CBMC_New[0], SelectedBox, INSERTION);

    SystemComponents[SelectedBox].deltaE += InsertionEnergy;

    ///////////////////////////////////////
    // UPDATE DELETION FOR THE OTHER BOX //
    ///////////////////////////////////////
    //AcceptDeletion(SystemComponents[OtherBox], Sims[OtherBox], SelectedComponent, SystemComponents[OtherBox].TempVal.UpdateLocation, DeletionSelectedMol, FF.noCharges);
    AcceptDeletion(Vars, OtherBox, DELETION);
    SystemComponents[OtherBox].deltaE -= DeletionEnergy;
  }
}

__global__ void copy_firstbead_to_new(Atoms NEW, Atoms* d_a, size_t comp, size_t position)
{
  NEW.pos[0] = d_a[comp].pos[position];
}

__global__ void Update_IdentitySwap_Insertion_data(Atoms* d_a, double3* temp, size_t NEWComponent, size_t UpdateLocation, size_t MolID, size_t Molsize)
{
  //Zhao's note: assuming not working with fractional molecule for Identity swap//
  for(size_t i = 0; i < Molsize; i++)
  {
    size_t realLocation = UpdateLocation + i;
    d_a[NEWComponent].pos[realLocation]   = temp[i];
    d_a[NEWComponent].scale[realLocation] = 1.0;
    d_a[NEWComponent].charge[realLocation]= d_a[NEWComponent].charge[i];
    d_a[NEWComponent].scaleCoul[realLocation]=1.0;
    d_a[NEWComponent].Type[realLocation] = d_a[NEWComponent].Type[i];
    d_a[NEWComponent].MolID[realLocation] = MolID;
  }
  d_a[NEWComponent].size += Molsize;
}

static inline MoveEnergy IdentitySwapMove(Variables& Vars, size_t systemId)
{
  Components& SystemComponents = Vars.SystemComponents[systemId];
  Simulations& Sims            = Vars.Sims[systemId];
  ForceField& FF               = Vars.device_FF;
  //RandomNumber& Random         = Vars.Random;
  //WidomStruct& Widom           = Vars.Widom[systemId];
  //Identity Swap is sort-of Reinsertion//
  //The difference is that the component of the molecule are different//
  //Selected Molecule is the molecule that is being identity-swapped//

   //Zhao's note: If CO2/CH4 mixture, since CH4 doesn't have rotation moves, identity swap will be performed more times for CH4 than for CO2. Add this switch of Old/NewComponent to avoid this issue.
  size_t NEWComponent = 0;
  size_t OLDComponent = 0;
  size_t NOld = 0;
  //It seems that identity swap can swap back to its own species, so need to relax the current criterion//
  //while(NEWComponent == OLDComponent || NEWComponent == 0 || NEWComponent >= SystemComponents.NComponents.x)
  //Must select adsorbate species, cannot exceed number of species in sim, oldcomponent number of mol > 0//
  if((SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks) == 0) //No adsorbates
  {
    MoveEnergy Empty;
    return Empty;
  }
  while(OLDComponent == 0 || OLDComponent >= SystemComponents.NComponents.x ||
        NEWComponent == 0 || NEWComponent >= SystemComponents.NComponents.x || 
        NOld == 0)
  {
    OLDComponent = (size_t) (Get_Uniform_Random()*(SystemComponents.NComponents.x - SystemComponents.NComponents.y)) + SystemComponents.NComponents.y;
    NEWComponent = (size_t) (Get_Uniform_Random()*(SystemComponents.NComponents.x - SystemComponents.NComponents.y)) + SystemComponents.NComponents.y;
    NOld = SystemComponents.NumberOfMolecule_for_Component[OLDComponent];
    /*
    if(Get_Uniform_Random() < 0.5)
    {
      size_t tempComponent = OLDComponent;
      OLDComponent = NEWComponent;
      NEWComponent = tempComponent;
    } 
    */
  }

  size_t NEWMolInComponent = SystemComponents.NumberOfMolecule_for_Component[NEWComponent];
  size_t OLDMolInComponent = (size_t) (Get_Uniform_Random() * SystemComponents.NumberOfMolecule_for_Component[OLDComponent]);

  double NNEWMol = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[NEWComponent]);
  double NOLDMol = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[OLDComponent]);
  if(SystemComponents.hasfractionalMolecule[NEWComponent]) NNEWMol -= 1.0;
  if(SystemComponents.hasfractionalMolecule[OLDComponent]) NOLDMol -= 1.0;

  //FOR DEBUG//
  //OLDMolInComponent = 52;

  SystemComponents.TempVal.molecule = NEWMolInComponent;
  SystemComponents.TempVal.component= NEWComponent;

  SystemComponents.Moves[OLDComponent].IdentitySwapRemoveTotal ++;
  SystemComponents.Moves[NEWComponent].IdentitySwapAddTotal ++;
  SystemComponents.Moves[OLDComponent].IdentitySwap_Total_TO[NEWComponent] ++;

  MoveEnergy energy; MoveEnergy old_energy;

  ///////////////
  // INSERTION //
  ///////////////
  CBMC_Variables& InsertionVariables = SystemComponents.CBMC_New[0];
  InsertionVariables.clear();
  InsertionVariables.MoveType = IDENTITY_SWAP_NEW; //Reinsertion-Insertion//
  //Take care of the frist bead location HERE//
  copy_firstbead_to_new<<<1,1>>>(Sims.New, Sims.d_a, OLDComponent, OLDMolInComponent * SystemComponents.Moleculesize[OLDComponent]);
  //WRITE THE component and molecule ID THAT IS GOING TO BE IGNORED DURING THE ENERGY CALC//
  Sims.ExcludeList[0] = {OLDComponent, OLDMolInComponent};
  SystemComponents.TempVal.Scale  = SystemComponents.Lambda[NEWComponent].SET_SCALE(1.0); //Zhao's note: not used in reinsertion, just set to 1.0//
  double& Rosenbluth = InsertionVariables.Rosenbluth;
  Widom_Move_FirstBead_PARTIAL(Vars, systemId, InsertionVariables);

  if(Rosenbluth <= 1e-150) InsertionVariables.SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer

  if(!InsertionVariables.SuccessConstruction)
  {
    Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored
    energy.zero();
    return energy;
  }

  if(SystemComponents.Moleculesize[NEWComponent] > 1 && Rosenbluth > 1e-150)
  {
    Widom_Move_Chain_PARTIAL(Vars, systemId, InsertionVariables); //True for doing insertion for reinsertion, different in MoleculeID//
    if(Rosenbluth <= 1e-150) InsertionVariables.SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
    if(!InsertionVariables.SuccessConstruction)
    {
      Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored
      energy.zero();
      return energy;
    }
  }
  energy = InsertionVariables.FirstBeadEnergy + InsertionVariables.ChainEnergy;

  size_t SelectedTrial = InsertionVariables.selectedTrial;
  if(SystemComponents.Moleculesize[NEWComponent] > 1) SelectedTrial = InsertionVariables.selectedTrialOrientation;
  // Store The New Locations //
  StoreNewLocation_Reinsertion<<<1,SystemComponents.Moleculesize[NEWComponent]>>>(Sims.Old, Sims.New, SystemComponents.tempMolStorage, SelectedTrial, SystemComponents.Moleculesize[NEWComponent]);

  /////////////
  // RETRACE //
  /////////////
  CBMC_Variables& DeletionVariables = SystemComponents.CBMC_Old[0]; 
  DeletionVariables.clear();
  Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored

  SystemComponents.TempVal.component = OLDComponent;
  SystemComponents.TempVal.molecule  = OLDMolInComponent;

  DeletionVariables.MoveType = IDENTITY_SWAP_OLD; //Identity_Swap for the old configuration//
  double& Old_Rosen= DeletionVariables.Rosenbluth;
  Widom_Move_FirstBead_PARTIAL(Vars, systemId, DeletionVariables);
  if(SystemComponents.Moleculesize[OLDComponent] > 1)
  {
    Widom_Move_Chain_PARTIAL(Vars, systemId, DeletionVariables);
  }
  old_energy = DeletionVariables.FirstBeadEnergy + DeletionVariables.ChainEnergy;
  energy -= old_energy;

  //Calculate Ewald//
  size_t UpdateLocation = SystemComponents.Moleculesize[OLDComponent] * OLDMolInComponent;
  if(!FF.noCharges)
  {
    double2 EwaldE = GPU_EwaldDifference_IdentitySwap(Sims.Box, Sims.d_a, Sims.Old, SystemComponents.tempMolStorage, FF, Sims.Blocksum, SystemComponents, OLDComponent, NEWComponent, UpdateLocation);
    energy.GGEwaldE = EwaldE.x;
    energy.HGEwaldE = EwaldE.y;
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y));
  }

  energy.TailE = TailCorrectionIdentitySwap(SystemComponents, NEWComponent, OLDComponent, FF.size, Sims.Box.Volume);
  InsertionVariables.Rosenbluth *= std::exp(-SystemComponents.Beta * energy.TailE);

  //NO DEEP POTENTIAL FOR THIS MOVE!//
  if(SystemComponents.UseDNNforHostGuest) throw std::runtime_error("NO DEEP POTENTIAL FOR IDENTITY SWAP!");

  /////////////////////////////
  // PREPARE ACCEPTANCE RULE //
  /////////////////////////////
  double NEWIdealRosen = SystemComponents.IdealRosenbluthWeight[NEWComponent];
  double OLDIdealRosen = SystemComponents.IdealRosenbluthWeight[OLDComponent];

  double preFactor  = GetPrefactor(SystemComponents, Sims, NEWComponent, INSERTION); 
         preFactor *= GetPrefactor(SystemComponents, Sims, OLDComponent, DELETION);

  double Pacc = preFactor * (Rosenbluth / NEWIdealRosen) / (Old_Rosen / OLDIdealRosen);

  //Determine whether to accept or reject the insertion
  double RANDOM = Get_Uniform_Random();
  bool Accept = false;
  if(RANDOM < Pacc) Accept = true;

  if(Accept)
  { // accept the move
    SystemComponents.Moves[NEWComponent].IdentitySwapAddAccepted ++;
    SystemComponents.Moves[OLDComponent].IdentitySwapRemoveAccepted ++;

    SystemComponents.Moves[OLDComponent].IdentitySwap_Acc_TO[NEWComponent] ++;
    if(NEWComponent != OLDComponent)
    {
      size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[OLDComponent]-1;
      size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[OLDComponent];
      Update_deletion_data<<<1,1>>>(Sims.d_a, OLDComponent, UpdateLocation, (int) SystemComponents.Moleculesize[OLDComponent], LastLocation);
   
      //The function below will only be processed if the system has a fractional molecule and the transfered molecule is NOT the fractional one //
      if((SystemComponents.hasfractionalMolecule[OLDComponent])&&(LastMolecule == SystemComponents.Lambda[OLDComponent].FractionalMoleculeID))
      {
        //Since the fractional molecule is moved to the place of the selected deleted molecule, update fractional molecule ID on host
        SystemComponents.Lambda[OLDComponent].FractionalMoleculeID = OLDMolInComponent;
      }
 
      UpdateLocation = SystemComponents.Moleculesize[NEWComponent] * NEWMolInComponent;
      Update_IdentitySwap_Insertion_data<<<1,1>>>(Sims.d_a, SystemComponents.tempMolStorage, NEWComponent, UpdateLocation, NEWMolInComponent, SystemComponents.Moleculesize[NEWComponent]); checkCUDAError("error Updating Identity Swap Insertion data");
    
      Update_NumberOfMolecules(SystemComponents, Sims.d_a, NEWComponent, INSERTION);
      Update_NumberOfMolecules(SystemComponents, Sims.d_a, OLDComponent, DELETION);
    }
    else //If they are the same species, just update in the reinsertion way (just the locations)//
    {
      //Regarding the UpdateLocation, in the code it is the old molecule position//
      //if the OLDComponent = NEWComponent, The new location will be filled into the old position//
      //So just use what is already in the code//
      Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[OLDComponent]>>>(Sims.d_a, SystemComponents.tempMolStorage, OLDComponent, UpdateLocation);
    }
    //Zhao's note: BUG!!!!, Think about if OLD/NEW Component belong to different type (framework/adsorbate)//
    if(!FF.noCharges && ((SystemComponents.hasPartialCharge[NEWComponent]) ||(SystemComponents.hasPartialCharge[OLDComponent])))
      Update_Vector_Ewald(Sims.Box, false, SystemComponents, NEWComponent);
    //energy.print();
    return energy;
  }
  else
  {
    energy.zero();
    return energy;
  }
}
