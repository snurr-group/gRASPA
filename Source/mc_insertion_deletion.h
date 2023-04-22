#include "mc_widom.h"
#include "mc_swap_utilities.h"
#include "lambda.h"
#include "mc_cbcfc.h"
MoveEnergy Insertion(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent);

MoveEnergy Reinsertion(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent);

MoveEnergy CreateMolecule(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, double2 newScale);

__global__ void StoreNewLocation_Reinsertion(Atoms Mol, Atoms NewMol, double* tempx, double* tempy, double* tempz, size_t SelectedTrial, size_t Moleculesize)
{
  if(Moleculesize == 1) //Only first bead is inserted, first bead data is stored in NewMol
  {
    tempx[0] = NewMol.x[SelectedTrial];
    tempy[0] = NewMol.y[SelectedTrial];
    tempz[0] = NewMol.z[SelectedTrial];
  }
  else //Multiple beads: first bead + trial orientations
  {
    //Update the first bead, first bead data stored in position 0 of Mol //
    tempx[0] = Mol.x[0];
    tempy[0] = Mol.y[0];
    tempz[0] = Mol.z[0];
   
    size_t chainsize = Moleculesize - 1; // FOr trial orientations //
    for(size_t i = 0; i < chainsize; i++) //Update the selected orientations//
    {
      size_t selectsize = SelectedTrial*chainsize+i;
      tempx[i+1] = NewMol.x[selectsize];
      tempy[i+1] = NewMol.y[selectsize];
      tempz[i+1] = NewMol.z[selectsize];
    }
  }
}

__global__ void Update_Reinsertion_data(Atoms* d_a, double* tempx, double* tempy, double* tempz, size_t SelectedComponent, size_t UpdateLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t realLocation = UpdateLocation + i;
  d_a[SelectedComponent].x[realLocation] = tempx[i];
  d_a[SelectedComponent].y[realLocation] = tempy[i];
  d_a[SelectedComponent].z[realLocation] = tempz[i];
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
  
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; 
    MoveEnergy temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, newScale); //True for doing insertion for reinsertion, different in MoleculeID//
    if(!SuccessConstruction)
    { 
      SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION);
      energy.zero();
      return energy;
    }
    SystemComponents.SumDeltaE(energy, temp_energy, ADD);
  }

  //Store The New Locations//
  double *tempx; double *tempy; double *tempz;
  cudaMalloc(&tempx, sizeof(double) * SystemComponents.Moleculesize[SelectedComponent]);
  cudaMalloc(&tempy, sizeof(double) * SystemComponents.Moleculesize[SelectedComponent]);
  cudaMalloc(&tempz, sizeof(double) * SystemComponents.Moleculesize[SelectedComponent]);
  StoreNewLocation_Reinsertion<<<1,1>>>(Sims.Old, Sims.New, tempx, tempy, tempz, SelectedTrial, SystemComponents.Moleculesize[SelectedComponent]);
  /////////////
  // RETRACE //
  /////////////
  CBMCType = REINSERTION_RETRACE; //Reinsertion-Retrace//
  double Old_Rosen=Widom_Move_FirstBead_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &old_energy, newScale);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial;
    MoveEnergy temp_energy = old_energy;
    Old_Rosen*=Widom_Move_Chain_PARTIAL(SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &old_energy, SelectedFirstBeadTrial, newScale);
    SystemComponents.SumDeltaE(old_energy, temp_energy, ADD);
  } 
 
  SystemComponents.SumDeltaE(energy, old_energy, MINUS);

  //Calculate Ewald//
  double EwaldE = 0.0;
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
  if(!FF.noCharges)
  {
    EwaldE = GPU_EwaldDifference_Reinsertion(Sims.Box, Sims.d_a, Sims.Old, tempx, tempy, tempz, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);
    double CoulRealE = 0.0;
    if(!FF.VDWRealBias)
    {
      CoulRealE += CoulombRealCorrection_Reinsertion(Sims.Box, Sims.d_a, Sims.Old, tempx, tempy, tempz, FF, Sims.Blocksum, SystemComponents, SelectedComponent);
    }
    Rosenbluth *= std::exp(-SystemComponents.Beta * (EwaldE + CoulRealE));
    energy.EwaldE = EwaldE;
    energy.HGVDWReal += CoulRealE;
    energy.HGEwaldE=SystemComponents.tempdeltaHGEwald;
  }
  //Calculate DNN//
  //Put it after Ewald summation, the required positions are already in place (done by the preparation parts of Ewald Summation)//
  if(SystemComponents.UseDNNforHostGuest)
  {
    double DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, REINSERTION_NEW);
    double DNN_Old = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, REINSERTION_OLD);
    energy.DNN_E = DNN_New - DNN_Old;
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
  double RRR = get_random_from_zero_to_one();
  if(RRR < Rosenbluth / Old_Rosen)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].ReinsertionAccepted ++;
    //size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
    Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, tempx, tempy, tempz, SelectedComponent, UpdateLocation); checkCUDAError("error Updating Reinsertion data");
    cudaFree(tempx); cudaFree(tempy); cudaFree(tempz);
    if(!FF.noCharges) Update_Ewald_Vector(Sims.Box, false, SystemComponents);
    SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION); //Update for TMMC, since Macrostate not changed, just add 1.//
    //energy.print();
    return energy;
  }
  else
  {
    cudaFree(tempx); cudaFree(tempy); cudaFree(tempz);
    SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, REINSERTION); //Update for TMMC, since Macrostate not changed, just add 1.//
    energy.zero();
    return energy;
  }
}

__global__ void Update_deletion_data(Atoms* d_a, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  //UpdateLocation should be the molecule that needs to be deleted
  //Then move the atom at the last position to the location of the deleted molecule
  //**Zhao's note** MolID of the deleted molecule should not be changed
  //**Zhao's note** if Molecule deleted is the last molecule, then nothing is copied, just change the size later.
  if(UpdateLocation != LastLocation)
  {
    for(size_t i = 0; i < Moleculesize; i++)
    {
      d_a[SelectedComponent].x[UpdateLocation+i]         = d_a[SelectedComponent].x[LastLocation+i];
      d_a[SelectedComponent].y[UpdateLocation+i]         = d_a[SelectedComponent].y[LastLocation+i];
      d_a[SelectedComponent].z[UpdateLocation+i]         = d_a[SelectedComponent].z[LastLocation+i];
      d_a[SelectedComponent].scale[UpdateLocation+i]     = d_a[SelectedComponent].scale[LastLocation+i];
      d_a[SelectedComponent].charge[UpdateLocation+i]    = d_a[SelectedComponent].charge[LastLocation+i];
      d_a[SelectedComponent].scaleCoul[UpdateLocation+i] = d_a[SelectedComponent].scaleCoul[LastLocation+i];
      d_a[SelectedComponent].Type[UpdateLocation+i]      = d_a[SelectedComponent].Type[LastLocation+i];
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

//Zhao's note: added feature for creating fractional molecules//
static inline MoveEnergy CreateMolecule(Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, double2 newScale)
{
  bool Insertion = true;
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
  double RRR = 1e-100;
  if(RRR < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Sims.Box, false, SystemComponents);
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
  bool Insertion = true;
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
  double RRR = get_random_from_zero_to_one();
  TMMCPacc = preFactor * Rosenbluth / IdealRosen; //Unbiased Acceptance//
  //Apply the bias according to the macrostate//
  SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, INSERTION);
  SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, INSERTION);

  bool Accept = false;
  if(RRR < preFactor * Rosenbluth / IdealRosen) Accept = true;
  SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, INSERTION);
  SystemComponents.Moves[SelectedComponent].RecordRosen(Rosenbluth, INSERTION);

  if(Accept)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, INSERTION);
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Sims.Box, false, SystemComponents);
    }
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
 
  bool Insertion = false;
  double preFactor = 0.0;
  bool SuccessConstruction = false;
  MoveEnergy energy;
  double Rosenbluth = 0.0;
  double StoredR = 0.0; //Don't use this for Deletion//
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
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RRR = get_random_from_zero_to_one();
  TMMCPacc = preFactor * IdealRosen / Rosenbluth; //Unbiased Acceptance//
  //Apply the bias according to the macrostate//
  SystemComponents.Tmmc[SelectedComponent].ApplyWLBias(preFactor, SystemComponents.Beta, NMol, DELETION);
  SystemComponents.Tmmc[SelectedComponent].ApplyTMBias(preFactor, SystemComponents.Beta, NMol, DELETION);

  bool Accept = false;
  if(RRR < preFactor * IdealRosen / Rosenbluth) Accept = true;
  SystemComponents.Tmmc[SelectedComponent].TreatAccOutofBound(Accept, NMol, DELETION);
  SystemComponents.Moves[SelectedComponent].RecordRosen(Rosenbluth, DELETION);

  if(Accept)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    // Get the starting position of the last molecule in the array
    size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1;
    size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[SelectedComponent];
    Update_deletion_data<<<1,1>>>(Sims.d_a, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
    //Zhao's note: the last molecule can be the fractional molecule, (fractional molecule ID is stored on the host), we need to update it as well (at least check it)//
    if((SystemComponents.hasfractionalMolecule[SelectedComponent])&&(LastMolecule == SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID))
    {
      //Since the fractional molecule is moved to the place of the selected deleted molecule, update fractional molecule ID on host
      SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SelectedMolInComponent;
    }
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, DELETION);
    if(!FF.noCharges) Update_Ewald_Vector(Sims.Box, false, SystemComponents);
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

static inline void AcceptInsertion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, size_t SelectedTrial, bool noCharges)
{
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  //printf("AccInsertion, SelectedTrial: %zu, UpdateLocation: %zu\n", SelectedTrial, UpdateLocation);
  //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
  if(!SystemComponents.SingleSwap) 
  {  
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
  }
  else
  {
    Update_SINGLE_INSERTION_data<<<1,1>>>(Sims.d_a, Sims.New, SelectedComponent);
  }
  Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, INSERTION); //true = Insertion//
  if(!noCharges)
  {
    Update_Ewald_Vector(Sims.Box, false, SystemComponents);
  }
}

static inline void AcceptDeletion(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, size_t UpdateLocation, size_t SelectedMol, bool noCharges)
{
  size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1;
  size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[SelectedComponent];
  Update_deletion_data<<<1,1>>>(Sims.d_a, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);

  Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, DELETION); //false = Deletion//
  if(!noCharges)
  {
    Update_Ewald_Vector(Sims.Box, false, SystemComponents);
  }
  //Zhao's note: the last molecule can be the fractional molecule, (fractional molecule ID is stored on the host), we need to update it as well (at least check it)//
  //The function below will only be processed if the system has a fractional molecule and the transfered molecule is NOT the fractional one //
  if((SystemComponents.hasfractionalMolecule[SelectedComponent])&&(LastMolecule == SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID))
  {
    //Since the fractional molecule is moved to the place of the selected deleted molecule, update fractional molecule ID on host
    SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SelectedMol;
  }
}

static inline void GibbsParticleTransfer(std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField FF, RandomNumber Random, std::vector<WidomStruct>& Widom, std::vector<SystemEnergies>& Energy, size_t SelectedComponent, Gibbs& GibbsStatistics)
{
  size_t NBox = SystemComponents.size();
  size_t SelectedBox = 0;
  size_t OtherBox    = 1;
  GibbsStatistics.GibbsXferStats.x += 1.0;

  if(get_random_from_zero_to_one() > 0.5)
  {
    SelectedBox = 1;
    OtherBox    = 0;
  }
  
  SystemComponents[SelectedBox].tempdeltaVDWReal = 0.0;
  SystemComponents[SelectedBox].tempdeltaEwald   = 0.0;

  SystemComponents[OtherBox].tempdeltaVDWReal = 0.0;
  SystemComponents[OtherBox].tempdeltaEwald   = 0.0;

  //printf("Performing Gibbs Particle Transfer Move! on Box[%zu]\n", SelectedBox);
  double2 Scale = {1.0, 1.0};
  bool TransferFractionalMolecule = false;
  //Randomly Select a Molecule from the OtherBox in the SelectedComponent//
  if(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent] == 0) return;
  size_t DeletionSelectedMol = (size_t) (get_random_from_zero_to_one()*(SystemComponents[OtherBox].NumberOfMolecule_for_Component[SelectedComponent]));
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
  if(!InsertionSuccess) return;

  /////////////////////////////////////////////
  // PERFORMING DELETION ON THE OTHER SYSTEM //
  /////////////////////////////////////////////
  size_t DeletionUpdateLocation = 0;
  double DeletionPrefactor      = 0.0;
  double DeletionRosen          = 0.0;
  bool   DeletionSuccess        = false;
  MoveEnergy DeletionEnergy = Deletion_Body(SystemComponents[OtherBox], Sims[OtherBox], FF, Random, Widom[OtherBox], DeletionSelectedMol, SelectedComponent, DeletionUpdateLocation, DeletionRosen, DeletionSuccess, DeletionPrefactor, Scale);
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
  if(get_random_from_zero_to_one()< (InsertionRosen * NMolB * Sims[SelectedBox].Box.Volume) / (DeletionRosen * (NMolA + 1) * Sims[OtherBox].Box.Volume)) Accept = true;

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
    AcceptInsertion(SystemComponents[SelectedBox], Sims[SelectedBox], SelectedComponent, InsertionSelectedTrial, FF.noCharges);
    Energy[SelectedBox].running_energy += InsertionEnergy.total();
    //printf("Insert Box: %zu, Insertion Energy: %.5f\n", SelectedBox, InsertionEnergy);

    ///////////////////////////////////////
    // UPDATE DELETION FOR THE OTHER BOX //
    ///////////////////////////////////////
    AcceptDeletion(SystemComponents[OtherBox], Sims[OtherBox], SelectedComponent, DeletionUpdateLocation, DeletionSelectedMol, FF.noCharges);
    Energy[OtherBox].running_energy -= DeletionEnergy.total();
    //printf("Delete Box: %zu, Insertion Energy: %.5f\n", OtherBox, DeletionEnergy);
  }
}

static inline MoveEnergy SingleSwapMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;

  bool Do_New = false; 
  bool Do_Old = false;

  if(MoveType == SINGLE_INSERTION)
  {
    Do_New = true;
    SystemComponents.Moves[SelectedComponent].InsertionTotal++;
  }
  else if(MoveType == SINGLE_DELETION)
  {
    Do_Old = true;
    SystemComponents.Moves[SelectedComponent].DeletionTotal++;
  }
  if(!Do_New && !Do_Old) throw std::runtime_error("Doing Nothing For Single Particle Move?\n");

  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  size_t chainsize;
  //Set up Old position and New position arrays
  chainsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  if(chainsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  Random.Check(3 * chainsize);
  get_new_position<<<1, chainsize>>>(Sims, FF, start_position, SelectedComponent, Random.device_random, Random.offset, MoveType);
  Random.Update(3 * chainsize);

  // Setup for the pairwise calculation //
  // New Features: divide the Blocks into two parts: Host-Guest + Guest-Guest //
  size_t NHostAtom = SystemComponents.Moleculesize[0] * SystemComponents.NumberOfMolecule_for_Component[0];
  size_t NGuestAtom= SystemComponents.Moleculesize[1] * SystemComponents.NumberOfMolecule_for_Component[1];
  size_t HG_Nthread=0; size_t HG_Nblock=0; Setup_threadblock(NHostAtom *  chainsize, &HG_Nblock, &HG_Nthread);
  size_t GG_Nthread=0; size_t GG_Nblock=0; Setup_threadblock(NGuestAtom * chainsize, &GG_Nblock, &GG_Nthread);
  size_t HGGG_Nthread = std::max(HG_Nthread, GG_Nthread);
  size_t HGGG_Nblock  = HG_Nblock + GG_Nblock;
  Energy_difference_PARTIAL_FLAG_HGGG<<<HGGG_Nblock, HGGG_Nthread, HGGG_Nthread * sizeof(double)>>>(Sims.Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, chainsize, Sims.device_flag, HG_Nblock, GG_Nblock, Do_New, Do_Old);
  cudaMemcpy(Sims.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);

  MoveEnergy tot;
  if(!Sims.flag[0])
  {
    double HGGG_BlockResult[HGGG_Nblock];
    cudaMemcpy(HGGG_BlockResult, Sims.Blocksum, HGGG_Nblock * sizeof(double), cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < HG_Nblock; i++) tot.HGVDWReal += HGGG_BlockResult[i];
    for(size_t i = HG_Nblock; i < HGGG_Nblock; i++) tot.GGVDWReal += HGGG_BlockResult[i];
    // Calculate Ewald //
    if(!FF.noCharges)
    {
      double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
      tot.EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, MoveType, 0, newScale);
    }
    tot.HGEwaldE=SystemComponents.tempdeltaHGEwald;
    if(SystemComponents.UseDNNforHostGuest)
    {
      //Calculate DNN//
      double DNN = 0.0;
      if(MoveType == SINGLE_INSERTION) DNN = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, NEW);
      if(MoveType == SINGLE_DELETION)  DNN = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, OLD) * -1.0;
      tot.DNN_E = DNN;
      /*
      if(MoveType == SINGLE_INSERTION)
      { printf("SINGLE INSERTION: "); tot.print();}
      if(MoveType == SINGLE_DELETION)
      { printf("SINGLE DELETION: "); tot.print();}
      */
      double correction = tot.DNN_Correction(); //If use DNN, HGVDWReal and HGEwaldE are zeroed//
      SystemComponents.SingleSwapDNNDrift += fabs(correction);
      if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
      {
        //printf("TRANSLATION/ROTATION: Bad Prediction, reject the move!!!\n");
        SystemComponents.SingleSwapDNNReject ++;
        tot.zero();
        return tot;
      }
    }
  }
  else
  {
   tot.zero();
   return tot;
  }

  double prefactor = GetPrefactor(SystemComponents, Sims, SelectedComponent, MoveType);
  double Pacc = prefactor * std::exp(-SystemComponents.Beta * tot.total());
  if (get_random_from_zero_to_one() < Pacc)
  {
    if(MoveType == SINGLE_INSERTION)
    {
      SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
      AcceptInsertion(SystemComponents, Sims, SelectedComponent, 0, FF.noCharges); //0: selectedTrial//
    }
    else if(MoveType == SINGLE_DELETION)
    {
      SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
      size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
      AcceptDeletion(SystemComponents, Sims, SelectedComponent, UpdateLocation, SelectedMolInComponent, FF.noCharges);
    }
    return tot;
  }
  tot.zero();
  return tot;
}
