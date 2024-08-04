#include "mc_widom.h"
#include "mc_swap_utilities.h"
#include "lambda.h"
#include "mc_cbcfc.h"

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
  /*
  for(size_t i = 0; i < Moleculesize; i++)
    printf("i: %lu, xyz: %.5f %.5f %.5f\n", i, temp[i].x, temp[i].y, temp[i].z);
  */
}

__global__ void Update_Reinsertion_data(Atoms* d_a, double3* temp, size_t SelectedComponent, size_t UpdateLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t realLocation = UpdateLocation + i;
  d_a[SelectedComponent].pos[realLocation] = temp[i];
}

static inline MoveEnergy Reinsertion(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;

  SystemComponents.Moves[SelectedComponent].ReinsertionTotal ++;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  MoveEnergy energy; MoveEnergy old_energy; double StoredR = 0.0;
 
  ///////////////
  // INSERTION //
  ///////////////
  int CBMCType = REINSERTION_INSERTION; //Reinsertion-Insertion//
  double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Zhao's note: not used in reinsertion, just set to 1.0//
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale); //Not reinsertion, not Retrace//

  if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer

  if(!SuccessConstruction)
  {
    SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION);
    energy.zero();
    return energy;
  }

  //DEBUG//
  /*
  if(SystemComponents.CURRENTCYCLE == 28)
  {printf("REINSERTION MOVE (comp: %zu, Mol: %zu) FIRST BEAD INSERTION ENERGY: ", SelectedComponent, SelectedMolInComponent); energy.print();
   printf("Rosen: %.5f\n", Rosenbluth);
  }
  */
 
  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; 
    MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); //True for doing insertion for reinsertion, different in MoleculeID//
    if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
    if(!SuccessConstruction)
    { 
      SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION);
      energy.zero();
      return energy;
    }
    energy += temp_energy;
  }

  /*
  if(SystemComponents.CURRENTCYCLE == 28) 
  {printf("REINSERTION MOVE, INSERTION ENERGY: "); energy.print();
   printf("Rosen: %.5f\n", Rosenbluth);
  }
  */

  //Store The New Locations//
  double3* temp;
  cudaMalloc(&temp, sizeof(double3) * SystemComponents.Moleculesize[SelectedComponent]);
  StoreNewLocation_Reinsertion<<<1,1>>>(Sims.Old, Sims.New, temp, SelectedTrial, SystemComponents.Moleculesize[SelectedComponent]);
  /////////////
  // RETRACE //
  /////////////
  CBMCType = REINSERTION_RETRACE; //Reinsertion-Retrace//
  double Old_Rosen=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &old_energy, newScale);

  /*
  if(SystemComponents.CURRENTCYCLE == 28)
  {printf("REINSERTION MOVE (comp: %zu, Mol: %zu) FIRST BEAD DELETION ENERGY: ", SelectedComponent, SelectedMolInComponent); old_energy.print();
   printf("Rosen: %.5f\n", Old_Rosen);
  }
  */


  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial;
    MoveEnergy temp_energy = old_energy;
    Old_Rosen*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &old_energy, SelectedFirstBeadTrial, newScale);
    old_energy += temp_energy;
  } 

  /*
  if(SystemComponents.CURRENTCYCLE == 28)
  {printf("REINSERTION MOVE, DELETION ENERGY: "); old_energy.print();
   printf("Rosen: %.5f\n", Old_Rosen);
  }
  */

  energy -= old_energy;

  //Calculate Ewald//
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;

  bool EwaldPerformed = false;
  if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
  {
    double2 EwaldE = GPU_EwaldDifference_Reinsertion(Sims.Box, Sims.d_a, Sims.Old, temp, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);

    energy.GGEwaldE = EwaldE.x;
    energy.HGEwaldE = EwaldE.y;
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y));
    EwaldPerformed = true;
  }
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    if(!EwaldPerformed) Prepare_DNN_InitialPositions_Reinsertion(Sims.d_a, Sims.Old, temp, SystemComponents, SelectedComponent, UpdateLocation);
    energy.DNN_E = DNN_Prediction_Reinsertion(SystemComponents, Sims, SelectedComponent, temp);
    //printf("DNN Delta Reinsertion: %.5f\n", energy.DNN_E);
    //Correction of DNN - HostGuest energy to the Rosenbluth weight//
    double correction = energy.DNN_Correction();
    if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
    {
        //printf("REINSERTION: Bad Prediction, reject the move!!!\n"); 
        SystemComponents.ReinsertionDNNReject++;
        //WriteOutliers(SystemComponents, Sims, REINSERTION_NEW, energy, correction);
        //WriteOutliers(SystemComponents, Sims, REINSERTION_OLD, energy, correction);
        energy.zero();
        return energy;
    }
    SystemComponents.ReinsertionDNNDrift += fabs(correction);
    Rosenbluth *= std::exp(-SystemComponents.Beta * correction);
  }

  //Determine whether to accept or reject the insertion
  double RANDOM = Get_Uniform_Random();
  //printf("RANDOM: %.5f, Rosenbluth / Old_Rosen: %.5f\n", RANDOM, Rosenbluth / Old_Rosen);
  if(RANDOM < Rosenbluth / Old_Rosen)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].ReinsertionAccepted ++;
    //size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
    Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, temp, SelectedComponent, UpdateLocation); checkCUDAError("error Updating Reinsertion data");
    cudaFree(temp); 
    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent]) 
      Update_Ewald_Vector(Sims.Box, false, SystemComponents, SelectedComponent);
    SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION); //Update for TMMC, since Macrostate not changed, just add 1.//
    //energy.print();
    return energy;
  }
  else
  {
    cudaFree(temp); 
    SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION); //Update for TMMC, since Macrostate not changed, just add 1.//
    energy.zero();
    return energy;
  }
}

//Zhao's note: added feature for creating fractional molecules//
static inline double WidomMove(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, double2 newScale)
{
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;
  
  //Zhao's note: For creating the fractional molecule, there is no previous step, so set the flag to false//
  MoveEnergy energy = Insertion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale);
  return Rosenbluth;
}


//Zhao's note: added feature for creating fractional molecules//
static inline MoveEnergy CreateMolecule(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, double2 newScale)
{
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;
  
  //Zhao's note: For creating the fractional molecule, there is no previous step, so set the flag to false//
  MoveEnergy energy = Insertion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale); 
  if(!SuccessConstruction) 
  {
    energy.zero();
    return energy;
  }
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RANDOM = 1e-100;
  if(RANDOM < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    if(!FF.noCharges && SystemComponents.hasPartialCharge[SelectedComponent])
    {
      Update_Ewald_Vector(Sims.Box, false, SystemComponents, SelectedComponent);
    }
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, INSERTION);
    return energy;
  }
  energy.zero();
  return energy;
}
//Zhao's note: This insertion only takes care of the full (not fractional) molecules//
static inline MoveEnergy Insertion(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  //This is the OLD STATE//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;
  double TMMCPacc = 0.0;

  SystemComponents.Moves[SelectedComponent].InsertionTotal ++;
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;

  double2 newScale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0)//
  MoveEnergy energy = Insertion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale); 
  if(!SuccessConstruction) 
  {
    //If unsuccessful move (Overlap), Pacc = 0//
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, INSERTION);
    SystemComponents.Moves[SelectedComponent].RecordRosen(0.0, INSERTION);
    energy.zero();
    return energy;
  }
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RANDOM = Get_Uniform_Random();
  TMMCPacc = preFactor * Rosenbluth / IdealRosen; //Unbiased Acceptance//
  //Apply the bias according to the macrostate//
  SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, NMol, INSERTION);
  SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, NMol, INSERTION);

  bool Accept = false;
  if(RANDOM < preFactor * Rosenbluth / IdealRosen) Accept = true;
  SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, INSERTION);
  SystemComponents.Moves[SelectedComponent].RecordRosen(Rosenbluth, INSERTION);

  if(Accept)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
    AcceptInsertion(SystemComponents, Sims, SelectedComponent, SelectedTrial, FF.noCharges, INSERTION);
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, INSERTION);
    return energy;
  }
  //else
  //Zhao's note: Even if the move is rejected by acceptance rule, still record the Pacc//
  SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, INSERTION);
  energy.zero();
  return energy;
}

static inline MoveEnergy Deletion(Components& SystemComponents, Simulations& Sims, ForceField& FF,  RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;
  double TMMCPacc = 0.0;

  SystemComponents.Moves[SelectedComponent].DeletionTotal ++;
 
  double preFactor = 0.0;
  bool SuccessConstruction = false;
  MoveEnergy energy;
  double Rosenbluth = 0.0;
  size_t UpdateLocation = 0;
  int CBMCType = CBMC_DELETION; //Deletion//
  double2 Scale = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0); //Set scale for full molecule (lambda = 1.0), Zhao's note: This is not used in deletion, just set to 1//
  //Wrapper for the deletion move//
  energy = Deletion_Body(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, UpdateLocation, Rosenbluth, SuccessConstruction, preFactor, Scale);
  if(!SuccessConstruction)
  {
    //If unsuccessful move (Overlap), Pacc = 0//
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, DELETION);
    SystemComponents.Moves[SelectedComponent].RecordRosen(0.0, DELETION);
    energy.zero();
    return energy;
  }
  //DEBUG//
  /*
  if(SystemComponents.CURRENTCYCLE == 28981)
  { printf("Selected Molecule: %zu\n", SelectedMolInComponent);
    printf("DELETION MOVE ENERGY: "); energy.print();
  }
  */
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RANDOM = Get_Uniform_Random();
  TMMCPacc = preFactor * IdealRosen / Rosenbluth; //Unbiased Acceptance//
  //Apply the bias according to the macrostate//
  SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, NMol, DELETION);
  SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, NMol, DELETION);

  bool Accept = false;
  if(RANDOM < preFactor * IdealRosen / Rosenbluth) Accept = true;
  SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, DELETION);

  if(Accept)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
    AcceptDeletion(SystemComponents, Sims, SelectedComponent, UpdateLocation, SelectedMolInComponent, FF.noCharges);
    //If unsuccessful move (Overlap), Pacc = 0//
    SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, DELETION);
    energy.take_negative();
    return energy;
  }
  //Zhao's note: Even if the move is rejected by acceptance rule, still record the Pacc//
  SystemComponents.Tmmc[SelectedComponent].Update(TMMCPacc, NMol, DELETION);
  energy.zero();
  return energy;
}

static inline void GibbsParticleTransfer(std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, RandomNumber Random, std::vector<WidomStruct>& Widom, std::vector<SystemEnergies>& Energy, size_t SelectedComponent, Gibbs& GibbsStatistics)
{
  size_t NBox = SystemComponents.size();
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  GibbsStatistics.GibbsXferStats.x += 1.0;

  if(Get_Uniform_Random() > 0.5)
  {
    SelectedBox = 1;
    OtherBox    = 0;
  }
  
  //printf("Performing Gibbs Particle Transfer Move! on Box[%zu]\n", SelectedBox);
  double2 Scale = {1.0, 1.0};
  bool TransferFractionalMolecule = false;
  //Randomly Select a Molecule from the OtherBox in the SelectedComponent//
  if(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent] == 0) return;
  size_t DeletionSelectedMol = (size_t) (Get_Uniform_Random()*(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent]));
  //Special treatment for transfering the fractional molecule//
  if(SystemComponents[OtherBox].hasfractionalMolecule[SelectedComponent] && SystemComponents[OtherBox].Lambda[SelectedComponent].FractionalMoleculeID == DeletionSelectedMol)
  {
    double oldBin    = SystemComponents[OtherBox].Lambda[SelectedComponent].currentBin;
    double delta     = SystemComponents[OtherBox].Lambda[SelectedComponent].delta;
    double oldLambda = delta * static_cast<double>(oldBin);
    Scale            = SystemComponents[OtherBox].Lambda[SelectedComponent].SET_SCALE(oldLambda);
    TransferFractionalMolecule = true;
  }
  //Perform Insertion on the selected System, then deletion on the other system//

  /////////////////////////////////////////////////
  // PERFORMING INSERTION ON THE SELECTED SYSTEM //
  /////////////////////////////////////////////////
  size_t InsertionSelectedTrial = 0;
  double InsertionPrefactor     = 0.0;
  double InsertionRosen         = 0.0;
  bool   InsertionSuccess       = false;
  size_t InsertionSelectedMol   = 0; //It is safer to ALWAYS choose the first atom as the template for CBMC_INSERTION//
  MoveEnergy InsertionEnergy = Insertion_Body(SystemComponents[SelectedBox], Sims[SelectedBox], FF, Random, Widom[SelectedBox], InsertionSelectedMol, SelectedComponent, InsertionRosen, InsertionSuccess, InsertionSelectedTrial, InsertionPrefactor, false, Scale);
  printf("Gibbs Particle Insertion energy: "); InsertionEnergy.print();
  if(!InsertionSuccess) return;

  /////////////////////////////////////////////
  // PERFORMING DELETION ON THE OTHER SYSTEM //
  /////////////////////////////////////////////
  size_t DeletionUpdateLocation = 0;
  double DeletionPrefactor      = 0.0;
  double DeletionRosen          = 0.0;
  bool   DeletionSuccess        = false;
  MoveEnergy DeletionEnergy = Deletion_Body(SystemComponents[OtherBox], Sims[OtherBox], FF, Random, Widom[OtherBox], DeletionSelectedMol, SelectedComponent, DeletionUpdateLocation, DeletionRosen, DeletionSuccess, DeletionPrefactor, Scale);
  printf("Gibbs Particle Deletion energy: "); DeletionEnergy.print();
  if(!DeletionSuccess) return;

  bool Accept = false;

  double NMolA= static_cast<double>(SystemComponents[SelectedBox].TotalNumberOfMolecules - SystemComponents[SelectedBox].NumberOfFrameworks);
  double NMolB= static_cast<double>(SystemComponents[OtherBox].TotalNumberOfMolecules - SystemComponents[OtherBox].NumberOfFrameworks);
  //Minus the fractional molecule//
  for(size_t comp = 0; comp < SystemComponents[SelectedBox].Total_Components; comp++)
  {
    if(SystemComponents[SelectedBox].hasfractionalMolecule[comp])
    {
      NMolA-=1.0;
    }
  }
  for(size_t comp = 0; comp < SystemComponents[OtherBox].Total_Components; comp++)
  {
    if(SystemComponents[OtherBox].hasfractionalMolecule[comp])
    {
      NMolB-=1.0;
    }
  }

  //This assumes that the two boxes share the same temperature, it might not be true//
  if(Get_Uniform_Random()< (InsertionRosen * NMolB * Sims[SelectedBox].Box.Volume) / (DeletionRosen * (NMolA + 1) * Sims[OtherBox].Box.Volume)) Accept = true;

  //printf("SelectedBox: %zu, OtherBox: %zu, InsertionEnergy: %.5f(%.5f %.5f), DeletionEnergy: %.5f(%.5f %.5f)\n", SelectedBox, OtherBox, InsertionEnergy, SystemComponents[SelectedBox].tempdeltaVDWReal, SystemComponents[SelectedBox].tempdeltaEwald, DeletionEnergy, SystemComponents[OtherBox].tempdeltaVDWReal, SystemComponents[OtherBox].tempdeltaEwald);

  if(Accept)
  {
    GibbsStatistics.GibbsXferStats.y += 1.0;
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
    AcceptInsertion(SystemComponents[SelectedBox], Sims[SelectedBox], SelectedComponent, InsertionSelectedTrial, FF.noCharges, INSERTION);

    SystemComponents[SelectedBox].deltaE += InsertionEnergy;
    //Energy[SelectedBox].running_energy += InsertionEnergy.total();
    //printf("Insert Box: %zu, Insertion Energy: %.5f\n", SelectedBox, InsertionEnergy);

    ///////////////////////////////////////
    // UPDATE DELETION FOR THE OTHER BOX //
    ///////////////////////////////////////
    AcceptDeletion(SystemComponents[OtherBox], Sims[OtherBox], SelectedComponent, DeletionUpdateLocation, DeletionSelectedMol, FF.noCharges);
    Energy[OtherBox].running_energy -= DeletionEnergy.total();
    SystemComponents[OtherBox].deltaE -= DeletionEnergy;
    //printf("Delete Box: %zu, Insertion Energy: %.5f\n", OtherBox, DeletionEnergy);
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

static inline MoveEnergy IdentitySwapMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random)
{
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

  size_t OLDMolInComponent = (size_t) (Get_Uniform_Random() * SystemComponents.NumberOfMolecule_for_Component[OLDComponent]);

  //printf("Cycle: %zu, OLDComp: %zu, NEWComp: %zu\n", SystemComponents.CURRENTCYCLE, OLDComponent, NEWComponent);
  //JUST FOR DEBUG//
  //NEWComponent = 1; OLDComponent = 1;
  //

  double NNEWMol = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[NEWComponent]);
  double NOLDMol = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[OLDComponent]);
  if(SystemComponents.hasfractionalMolecule[NEWComponent]) NNEWMol -= 1.0;
  if(SystemComponents.hasfractionalMolecule[OLDComponent]) NOLDMol -= 1.0;

  //FOR DEBUG//
  //OLDMolInComponent = 52;

  size_t NEWMolInComponent = SystemComponents.NumberOfMolecule_for_Component[NEWComponent];

  SystemComponents.Moves[OLDComponent].IdentitySwapRemoveTotal ++;
  SystemComponents.Moves[NEWComponent].IdentitySwapAddTotal ++;
  SystemComponents.Moves[OLDComponent].IdentitySwap_Total_TO[NEWComponent] ++;

  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  MoveEnergy energy; MoveEnergy old_energy; double StoredR = 0.0;

  ///////////////
  // INSERTION //
  ///////////////
  int CBMCType = IDENTITY_SWAP_NEW; //Reinsertion-Insertion//
  //Take care of the frist bead location HERE//
  copy_firstbead_to_new<<<1,1>>>(Sims.New, Sims.d_a, OLDComponent, OLDMolInComponent * SystemComponents.Moleculesize[OLDComponent]);
  //WRITE THE component and molecule ID THAT IS GOING TO BE IGNORED DURING THE ENERGY CALC//
  Sims.ExcludeList[0] = {OLDComponent, OLDMolInComponent};
  double2 newScale  = SystemComponents.Lambda[NEWComponent].SET_SCALE(1.0); //Zhao's note: not used in reinsertion, just set to 1.0//
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, NEWMolInComponent, NEWComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, newScale);

  if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer

  if(!SuccessConstruction)
  {
    energy.zero(); Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored
    return energy;
  }

  if(SystemComponents.Moleculesize[NEWComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial;
    MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, NEWMolInComponent, NEWComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); //True for doing insertion for reinsertion, different in MoleculeID//
    if(Rosenbluth <= 1e-150) SuccessConstruction = false; //Zhao's note: added this protection bc of weird error when testing GibbsParticleXfer
    if(!SuccessConstruction)
    {
      energy.zero(); Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored
      return energy;
    }
    energy += temp_energy;
  }
  //printf("NEW MOLECULE ENERGY:"); energy.print();

  // Store The New Locations //
  double3* temp;
  cudaMalloc(&temp, sizeof(double3) * SystemComponents.Moleculesize[NEWComponent]);
  StoreNewLocation_Reinsertion<<<1,1>>>(Sims.Old, Sims.New, temp, SelectedTrial, SystemComponents.Moleculesize[NEWComponent]);

  /////////////
  // RETRACE //
  /////////////
  
  Sims.ExcludeList[0] = {-1, -1}; //Set to negative so that excludelist is ignored

  CBMCType = IDENTITY_SWAP_OLD; //Identity_Swap for the old configuration//
  double Old_Rosen=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, OLDMolInComponent, OLDComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &old_energy, newScale);
  if(SystemComponents.Moleculesize[OLDComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial;
    MoveEnergy temp_energy = old_energy;
    Old_Rosen*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, OLDMolInComponent, OLDComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &old_energy, SelectedFirstBeadTrial, newScale);
    old_energy += temp_energy;
  }

  //printf("OLD MOLECULE ENERGY:"); old_energy.print();
  energy -= old_energy;

  //Calculate Ewald//
  size_t UpdateLocation = SystemComponents.Moleculesize[OLDComponent] * OLDMolInComponent;
  if(!FF.noCharges)
  {
    double2 EwaldE = GPU_EwaldDifference_IdentitySwap(Sims.Box, Sims.d_a, Sims.Old, temp, FF, Sims.Blocksum, SystemComponents, OLDComponent, NEWComponent, UpdateLocation);
    energy.GGEwaldE = EwaldE.x;
    energy.HGEwaldE = EwaldE.y;
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y));
    //printf("Ewald PreFactor: %.5f\n", ::exp(-SystemComponents.Beta * (EwaldE.x + EwaldE.y)));
  }

  energy.TailE = TailCorrectionIdentitySwap(SystemComponents, NEWComponent, OLDComponent, FF.size, Sims.Box.Volume);
  Rosenbluth *= std::exp(-SystemComponents.Beta * energy.TailE);

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

  //printf("Rosenbluth Weights %.5f (New), %.5f (Old)\n", Rosenbluth, Old_Rosen);
  //printf("Partial Fugacities: %.5f (New), %.5f (Old)\n", PartialFugacityNew, PartialFugacityOld);
  //printf("NNEW: %.5f, NOLD: %.5f\n", NNEWMol, NOLDMol);
  //printf("PAcc: %.5f\n", Pacc);

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
      Update_IdentitySwap_Insertion_data<<<1,1>>>(Sims.d_a, temp, NEWComponent, UpdateLocation, NEWMolInComponent, SystemComponents.Moleculesize[NEWComponent]); checkCUDAError("error Updating Identity Swap Insertion data");
    
      Update_NumberOfMolecules(SystemComponents, Sims.d_a, NEWComponent, INSERTION);
      Update_NumberOfMolecules(SystemComponents, Sims.d_a, OLDComponent, DELETION);
    }
    else //If they are the same species, just update in the reinsertion way (just the locations)//
    {
      //Regarding the UpdateLocation, in the code it is the old molecule position//
      //if the OLDComponent = NEWComponent, The new location will be filled into the old position//
      //So just use what is already in the code//
      Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[OLDComponent]>>>(Sims.d_a, temp, OLDComponent, UpdateLocation);
    }
    cudaFree(temp);
    //Zhao's note: BUG!!!!, Think about if OLD/NEW Component belong to different type (framework/adsorbate)//
    if(!FF.noCharges && ((SystemComponents.hasPartialCharge[NEWComponent]) ||(SystemComponents.hasPartialCharge[OLDComponent])))
      Update_Ewald_Vector(Sims.Box, false, SystemComponents, NEWComponent);
    //energy.print();
    return energy;
  }
  else
  {
    cudaFree(temp); 
    energy.zero();
    return energy;
  }
}
