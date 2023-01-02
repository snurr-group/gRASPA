#include "mc_widom.h"
#include "mc_swap_utilities.h"
#include "lambda.h"
#include "mc_cbcfc.h"
double Insertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision);

double Reinsertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision);

void Update_NumberOfMolecules(Components& SystemComponents, Atoms*& d_a, size_t SelectedComponent, bool Insertion);

double CreateMolecule(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision, double2 newScale);

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

static inline double Reinsertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  SystemComponents.Moves[SelectedComponent].ReinsertionTotal ++;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0; double old_energy = 0.0; double StoredR = 0.0;
 
  ///////////////
  // INSERTION //
  ///////////////
  int CBMCType = REINSERTION_INSERTION; //Reinsertion-Insertion//
  double2 newScale  = setScale(1.0); //Zhao's note: not used in reinsertion, just set to 1.0//
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision, newScale); //Not reinsertion, not Retrace//
  if(!SuccessConstruction)
    return 0.0;
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision, newScale); //True for doing insertion for reinsertion, different in MoleculeID//
    if(!SuccessConstruction){ return 0.0;}
    energy += temp_energy;
    SystemComponents.tempdeltaVDWReal = energy;
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
  double Old_Rosen=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &old_energy, DualPrecision, newScale);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = old_energy;
    Old_Rosen*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &old_energy, SelectedFirstBeadTrial, DualPrecision, newScale);
    old_energy += temp_energy;
    SystemComponents.tempdeltaVDWReal -= old_energy;
  } 
 
  //Calculate Ewald//
  double EwaldE = 0.0;
  size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
  if(!FF.noCharges)
  {
    EwaldE = GPU_EwaldDifference_Reinsertion(Box, Sims.d_a, Sims.Old, tempx, tempy, tempz, FF, Sims.Blocksum, SystemComponents, SelectedComponent, UpdateLocation);
    Rosenbluth *= std::exp(-SystemComponents.Beta * EwaldE);
    energy     += EwaldE;
    SystemComponents.tempdeltaEwald = EwaldE;
  }

  //Determine whether to accept or reject the insertion
  double RRR = get_random_from_zero_to_one();
  if(RRR < Rosenbluth / Old_Rosen)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].ReinsertionAccepted ++;
    //size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SelectedMolInComponent;
    Update_Reinsertion_data<<<1,SystemComponents.Moleculesize[SelectedComponent]>>>(Sims.d_a, tempx, tempy, tempz, SelectedComponent, UpdateLocation); checkCUDAError("error Updating Reinsertion data");
    cudaFree(tempx); cudaFree(tempy); cudaFree(tempz);
    SystemComponents.deltaVDWReal   += SystemComponents.tempdeltaVDWReal;
    SystemComponents.deltaEwald     += SystemComponents.tempdeltaEwald;
    if(!FF.noCharges) Update_Ewald_Vector(Box, false, SystemComponents);
    return energy - old_energy;
  }
  else
  cudaFree(tempx); cudaFree(tempy); cudaFree(tempz);
  return 0.0;
}

__global__ void Update_deletion_data(Atoms* d_a, Atoms NewMol, size_t SelectedComponent, size_t UpdateLocation, int Moleculesize, size_t LastLocation)
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

static inline double Deletion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF,  RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  SystemComponents.Moves[SelectedComponent].DeletionTotal ++;
  bool Insertion = false;
  bool SuccessConstruction = false;
  size_t SelectedTrial = 0;
  double energy = 0.0;
  double StoredR = 0.0; //Don't use this for Deletion//
  int CBMCType = CBMC_DELETION; //Deletion//
  double2 newScale = setScale(1.0); //Set scale for full molecule (lambda = 1.0), Zhao's note: This is not used in deletion, just set to 1//
  double Rosenbluth=Widom_Move_FirstBead_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, StoredR, &SelectedTrial, &SuccessConstruction, &energy, DualPrecision, newScale);
  if(SystemComponents.Moleculesize[SelectedComponent] > 1 && Rosenbluth > 1e-150)
  {
    size_t SelectedFirstBeadTrial = SelectedTrial; double temp_energy = energy;
    Rosenbluth*=Widom_Move_Chain_PARTIAL(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, CBMCType, &SelectedTrial, &SuccessConstruction, &energy, SelectedFirstBeadTrial, DualPrecision, newScale); //The false is for Reinsertion//
    energy += temp_energy;
  }
  if(!SuccessConstruction)
    return 0.0;
  SystemComponents.tempdeltaVDWReal = -energy;
  //Determine whether to accept or reject the insertion
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);
  double preFactor = (NumberOfMolecules) / (SystemComponents.Beta * MolFraction * Box.Pressure * FugacityCoefficient * Box.Volume);
  size_t UpdateLocation = SelectedMolInComponent * SystemComponents.Moleculesize[SelectedComponent];
  double EwaldE = 0.0;
  if(!FF.noCharges)
  {
    EwaldE     = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, DELETION, UpdateLocation, newScale);
    preFactor *= std::exp(-SystemComponents.Beta * EwaldE);
    energy    -= EwaldE;
    SystemComponents.tempdeltaEwald = EwaldE;
  }
  double RRR = get_random_from_zero_to_one();
  //if(RRR < preFactor * IdealRosen / Rosenbluth)
  if(RRR < preFactor * IdealRosen / Rosenbluth) // for bebug: always accept
  { // accept the move
    SystemComponents.Moves[SelectedComponent].DeletionAccepted ++;
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    // Get the starting position of the last molecule in the array
    size_t LastMolecule = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]-1;
    size_t LastLocation = LastMolecule*SystemComponents.Moleculesize[SelectedComponent];
    Update_deletion_data<<<1,1>>>(Sims.d_a, Sims.New, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent], LastLocation);
    //Zhao's note: the last molecule can be the fractional molecule, (fractional molecule ID is stored on the host), we need to update it as well (at least check it)//
    if((SystemComponents.hasfractionalMolecule[SelectedComponent])&&(LastMolecule == SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID))
    {
      //Since the fractional molecule is moved to the place of the selected deleted molecule, update fractional molecule ID on host
      SystemComponents.Lambda[SelectedComponent].FractionalMoleculeID = SelectedMolInComponent;
    }
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, Insertion);
    if(!FF.noCharges) Update_Ewald_Vector(Box, false, SystemComponents);
    SystemComponents.deltaVDWReal += SystemComponents.tempdeltaVDWReal; SystemComponents.deltaEwald += SystemComponents.tempdeltaEwald;
    return -energy;
  }
  return 0.0;
}
//Zhao's note: added feature for creating fractional molecules//
static inline double CreateMolecule(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision, double2 newScale)
{
  bool Insertion = true;
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;
  
  //Zhao's note: For creating the fractional molecule, there is no previous step, so set the flag to false//
  double energy = Insertion_Body(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, DualPrecision, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale); if(!SuccessConstruction) return 0.0;
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RRR = 1e-10;
  if(RRR < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Box, false, SystemComponents);
    }
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, Insertion);
    return energy;
  }
  return 0.0;
}
//Zhao's note: This insertion only takes care of the full (not fractional) molecules//
static inline double Insertion(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool DualPrecision)
{
  SystemComponents.Moves[SelectedComponent].InsertionTotal ++;
  bool Insertion = true;
  bool SuccessConstruction = false;
  double Rosenbluth = 0.0;
  size_t SelectedTrial = 0;
  double preFactor = 0.0;

  SystemComponents.tempdeltaVDWReal = 0.0; SystemComponents.tempdeltaEwald = 0.0;

  double2 newScale = setScale(1.0); //Set scale for full molecule (lambda = 1.0)//
  double energy = Insertion_Body(Box, SystemComponents, Sims, FF, Random, Widom, SelectedMolInComponent, SelectedComponent, DualPrecision, Rosenbluth, SuccessConstruction, SelectedTrial, preFactor, false, newScale); if(!SuccessConstruction) return 0.0;
  double IdealRosen = SystemComponents.IdealRosenbluthWeight[SelectedComponent];
  double RRR = get_random_from_zero_to_one();
  if(RRR < preFactor * Rosenbluth / IdealRosen)
  { // accept the move
    SystemComponents.Moves[SelectedComponent].InsertionAccepted ++;
    size_t UpdateLocation = SystemComponents.Moleculesize[SelectedComponent] * SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
    //Zhao's note: here needs more consideration: need to update after implementing polyatomic molecule
    Update_insertion_data<<<1,1>>>(Sims.d_a, Sims.Old, Sims.New, SelectedTrial, SelectedComponent, UpdateLocation, (int) SystemComponents.Moleculesize[SelectedComponent]);
    Update_NumberOfMolecules(SystemComponents, Sims.d_a, SelectedComponent, Insertion);
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Box, false, SystemComponents);
    }
    SystemComponents.deltaVDWReal += SystemComponents.tempdeltaVDWReal; SystemComponents.deltaEwald += SystemComponents.tempdeltaEwald;
    return energy;
  }
  //else
  return 0.0;
}
