#include <algorithm>
#include <omp.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
size_t SelectTrialPosition(std::vector <double> LogBoltzmannFactors); //In Zhao's code, LogBoltzmannFactors = Rosen

double Widom_Move_FirstBead_PARTIAL(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, bool Retrace, double &StoredR, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, bool DualPrecision);

static inline double Widom_Move_Chain_PARTIAL(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, size_t FirstBeadTrial, bool DualPrecision);

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
    double ws = get_random_from_zero_to_one() * SumShiftedBoltzmannFactors;
    while (cumw < ws)
        cumw += ShiftedBoltzmannFactors[++selected];

    return selected;
}

__global__ void get_random_trial_position_firstbead(Boxsize Box, Atoms* d_a, Atoms NewMol, bool* device_flag, double* random, size_t offset, size_t start_position, size_t SelectedComponent, size_t MolID, bool Deletion)
{
  //Insertion/Widom: MolID = NewValue (Component.NumberOfMolecule_for_Component[SelectedComponent]
  //Deletion: MolID = selected ID
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = i*3 + offset;
  const Atoms AllData = d_a[SelectedComponent];
  //different from translation (where we copy a whole molecule), here we duplicate the properties of the first bead of a molecule
  // so use start_position, not real_pos
  //the first values always have some numbers. The xyz are not correct, but type and charge are correct. Use those.
  double scale=0.0; double charge = 0.0; double scaleCoul = 0.0; size_t Type = 0;
  if((!Deletion) && AllData.size == 0)
  {
    scale     = AllData.scale[0];
    charge    = AllData.charge[0];
    scaleCoul = AllData.scaleCoul[0];
    Type      = AllData.Type[0];
    //printf("i: %lu, scale: %.10f, charge: %.10f, scaleCoul: %.10f, Type: %lu\n", i, scale, charge, scaleCoul, Type);
  }
  else
  {
    scale     = AllData.scale[start_position];
    charge    = AllData.charge[start_position];
    scaleCoul = AllData.scaleCoul[start_position];
    Type      = AllData.Type[start_position];
  }
  if((i==0) && (Deletion)) //if deletion, the first trial position is the old position of the selected molecule//
  {
    NewMol.x[i] = AllData.x[start_position];
    NewMol.y[i] = AllData.y[start_position];
    NewMol.z[i] = AllData.z[start_position];
  }
  else
  {
    NewMol.x[i] = Box.Cell[0] * random[random_i];
    NewMol.y[i] = Box.Cell[4] * random[random_i+1];
    NewMol.z[i] = Box.Cell[8] * random[random_i+2];
  }
  NewMol.scale[i] = scale; NewMol.charge[i] = charge; NewMol.scaleCoul[i] = scaleCoul; NewMol.Type[i] = Type; NewMol.MolID[i] = MolID;
  device_flag[i] = false;
}

template<typename T>
inline void GPU_Sum_Widom(size_t NumberWidomTrials, double Beta, double OverlapCriteria, T* energy_array, bool* flag, size_t arraysize, std::vector<double>& energies, std::vector<size_t>& Trialindex, std::vector<double>& Rosen)
{
  for(size_t i = 0; i < NumberWidomTrials; i++)
  {
    if(!flag[i])
    {
      double tot = 0.0;
      T float_tot = GPUReduction<BLOCKSIZE>(&energy_array[i*arraysize], arraysize);
      tot = static_cast<double>(float_tot);
      if(tot < OverlapCriteria)
      {
        energies.push_back(tot);
        Trialindex.push_back(i);
        Rosen.push_back((-Beta*tot));
      }
    }
  }
}
template<typename T>
inline void Host_sum_Widom(size_t NumberWidomTrials, double Beta, T Overlap, T* energy_array, bool* flag, size_t arraysize, std::vector<double>& energies, std::vector<size_t>& Trialindex, std::vector<double>& Rosen)
{
  std::vector<size_t>reasonable_trials;
  for(size_t i = 0; i < NumberWidomTrials; i++){
    if(!flag[i]){
      reasonable_trials.push_back(i);}}
  T host_array[arraysize];
  for(size_t i = 0; i < reasonable_trials.size(); i++)
  {
    size_t trial = reasonable_trials[i];
    cudaMemcpy(host_array, &energy_array[trial*arraysize], arraysize*sizeof(T), cudaMemcpyDeviceToHost);
    T float_tot = 0.0;
    for(size_t ijk=0; ijk < arraysize; ijk++)
    {
      float_tot+=host_array[ijk];
      if(float_tot > Overlap)
        break;
    }
    if(float_tot < Overlap)
    {
      double tot = static_cast<double>(float_tot);
      energies.push_back(tot);
      Trialindex.push_back(trial);
      Rosen.push_back(-Beta*tot);
    }
  }
}

__device__ void RotationAroundXAxis_trialOrientations(double* Vec, double theta)
{
  double w,s,c,rot[3*3];

  c=cos(theta);
  s=sin(theta);

  rot[0*3+0]=1.0; rot[1*3+0]=0.0;  rot[2*3+0]=0.0;
  rot[0*3+1]=0.0; rot[1*3+1]=c;    rot[2*3+1]=-s;
  rot[0*3+2]=0.0; rot[1*3+2]=s;    rot[2*3+2]=c;

  w=Vec[0]*rot[0*3+0]+Vec[1]*rot[0*3+1]+Vec[2]*rot[0*3+2];
  s=Vec[0]*rot[1*3+0]+Vec[1]*rot[1*3+1]+Vec[2]*rot[1*3+2];
  c=Vec[0]*rot[2*3+0]+Vec[1]*rot[2*3+1]+Vec[2]*rot[2*3+2];
  Vec[0]=w;
  Vec[1]=s;
  Vec[2]=c;
}

__device__ void RotationAroundYAxis_trialOrientations(double* Vec, double theta)
{
  double w,s,c,rot[3*3];

  c=cos(theta);
  s=sin(theta);

  rot[0*3+0]=c;   rot[1*3+0]=0;    rot[2*3+0]=s;
  rot[0*3+1]=0;   rot[1*3+1]=1.0;  rot[2*3+1]=0;
  rot[0*3+2]=-s;  rot[1*3+2]=0;    rot[2*3+2]=c;

  w=Vec[0]*rot[0*3+0]+Vec[1]*rot[0*3+1]+Vec[2]*rot[0*3+2];
  s=Vec[0]*rot[1*3+0]+Vec[1]*rot[1*3+1]+Vec[2]*rot[1*3+2];
  c=Vec[0]*rot[2*3+0]+Vec[1]*rot[2*3+1]+Vec[2]*rot[2*3+2];
  Vec[0]=w;
  Vec[1]=s;
  Vec[2]=c;
}

__device__ void RotationAroundZAxis_trialOrientations(double* Vec, double theta)
{
  double w,s,c,rot[3*3];

  c=cos(theta);
  s=sin(theta);

  rot[0*3+0]=c;   rot[1*3+0]=-s;   rot[2*3+0]=0;
  rot[0*3+1]=s;   rot[1*3+1]=c;    rot[2*3+1]=0;
  rot[0*3+2]=0;   rot[1*3+2]=0;    rot[2*3+2]=1.0;

  w=Vec[0]*rot[0*3+0]+Vec[1]*rot[0*3+1]+Vec[2]*rot[0*3+2];
  s=Vec[0]*rot[1*3+0]+Vec[1]*rot[1*3+1]+Vec[2]*rot[1*3+2];
  c=Vec[0]*rot[2*3+0]+Vec[1]*rot[2*3+1]+Vec[2]*rot[2*3+2];
  Vec[0]=w;
  Vec[1]=s;
  Vec[2]=c;
}
/*
//Zhao's note: this is the modified algorithm based on Marsaglia's, might not be correct//
__device__ void Rotate_Quaternions(double* Vec, double* random, size_t random_i)
{
  double q0   = 0.0; double q1    = 0.0;
  double q2   = 0.0; double q3    = 0.0;
  double x1   = 0.0; double x2    = 0.0;
  double x3   = 0.0; double x4    = 0.0;
  double xisq = 1.0; double xisq2 = 1.0;
  double residue_range = 0.0; double xi = 0.0;
  //Zhao's note: my undergrad fortran code used while loops for generating quaternions//
  //Here, we do something else, here first select x1, then based on the value of x1, select x2//
  //Zhao's note: might be tricky, do they sample the same distribution?//
  x1 = 2.0 * (random[random_i]   - 0.5);
  residue_range = sqrt(1.0 - x1 * x1);
  x2 = residue_range * 2.0 * (random[random_i+1] - 0.5);
  xisq = x1 * x1 + x2 * x2;
  
  x3 = 2.0 * (random[random_i+2] - 0.5);
  residue_range = sqrt(1.0 - x2 * x2);
  x4 = residue_range * 2.0 * (random[random_i+3] - 0.5);
  xisq2 = x3 * x3 + x4 * x4;
  xi = sqrt((1 - xisq)/xisq2);
  
  q0 = x1; q1 = x2; q2 = x3 * xi; q3 = x4 * xi;
  double rot[3*3];
  double a01=q0*q1; double a02=q0*q2; double a03=q0*q3;
  double a11=q1*q1; double a12=q1*q2; double a13=q1*q3;
  double a22=q2*q2; double a23=q2*q3; double a33=q3*q3;

  rot[0]=1.0-2.0*(a22+a33);
  rot[1]=2.0*(a12-a03);
  rot[2]=2.0*(a13+a02);
  rot[3]=2.0*(a12+a03);
  rot[4]=1.0-2.0*(a11+a33);
  rot[5]=2.0*(a23-a01);
  rot[6]=2.0*(a13-a02);
  rot[7]=2.0*(a23+a01);
  rot[8]=1.0-2.0*(a11+a22);
  double w=Vec[0]*rot[0*3+0]+Vec[1]*rot[0*3+1]+Vec[2]*rot[0*3+2];
  double s=Vec[0]*rot[1*3+0]+Vec[1]*rot[1*3+1]+Vec[2]*rot[1*3+2];
  double c=Vec[0]*rot[2*3+0]+Vec[1]*rot[2*3+1]+Vec[2]*rot[2*3+2];
  Vec[0]=w;
  Vec[1]=s;
  Vec[2]=c;
}
*/

__device__ void Rotate_Quaternions(double* Vec, double* random, size_t random_i)
{
  //https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
  //https://stackoverflow.com/questions/38978441/creating-uniform-random-quaternion-and-multiplication-of-two-quaternions
  //James J. Kuffner 2004//
  //Zhao's note: this one needs three random numbers//
  double u = random[random_i]; double v = random[random_i + 1]; double w = random[random_i + 2];
  double pi=3.14159265358979323846;//Zhao's note: consider using M_PI
  double q0 = sqrt(1-u) * std::sin(2*pi*v);
  double q1 = sqrt(1-u) * std::cos(2*pi*v);
  double q2 = sqrt(u)   * std::sin(2*pi*w);
  double q3 = sqrt(u)   * std::cos(2*pi*w);

  double rot[3*3];
  double a01=q0*q1; double a02=q0*q2; double a03=q0*q3;
  double a11=q1*q1; double a12=q1*q2; double a13=q1*q3;
  double a22=q2*q2; double a23=q2*q3; double a33=q3*q3;

  rot[0]=1.0-2.0*(a22+a33);
  rot[1]=2.0*(a12-a03);
  rot[2]=2.0*(a13+a02);
  rot[3]=2.0*(a12+a03);
  rot[4]=1.0-2.0*(a11+a33);
  rot[5]=2.0*(a23-a01);
  rot[6]=2.0*(a13-a02);
  rot[7]=2.0*(a23+a01);
  rot[8]=1.0-2.0*(a11+a22);
  double r=Vec[0]*rot[0*3+0]+Vec[1]*rot[0*3+1]+Vec[2]*rot[0*3+2];
  double s=Vec[0]*rot[1*3+0]+Vec[1]*rot[1*3+1]+Vec[2]*rot[1*3+2];
  double c=Vec[0]*rot[2*3+0]+Vec[1]*rot[2*3+1]+Vec[2]*rot[2*3+2];
  Vec[0]=r;
  Vec[1]=s;
  Vec[2]=c;
  
}
__global__ void get_random_trial_position_chain(Boxsize Box, Atoms* d_a, Atoms Mol, Atoms NewMol, bool* device_flag, double* random, size_t offset, size_t FirstBeadTrial, size_t start_position, size_t SelectedComponent, size_t MolID, bool Deletion, size_t chainsize)
{
  //Zhao's note: for trial orientations, each orientation may have more than 1 atom, do a for loop. So the threads are for different trial orientations, rather than different atoms in different orientations//
  //Zhao's note: chainsize is the size of the molecule excluding the first bead(-1).
  //Zhao's note: caveat: I am assuming the first bead of insertion is the first bead defined in def file. Need to relax this assumption in the future//
  //Zhao's note: First bead position stored in Mol, at position zero
  //Insertion/Widom: MolID = NewValue (Component.NumberOfMolecule_for_Component[SelectedComponent])
  //Deletion: MolID = selected ID
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i == 0)
  {
    Mol.x[0]         = NewMol.x[FirstBeadTrial];
    Mol.y[0]         = NewMol.y[FirstBeadTrial];
    Mol.z[0]         = NewMol.z[FirstBeadTrial];
    Mol.scale[0]     = NewMol.scale[FirstBeadTrial];
    Mol.charge[0]    = NewMol.charge[FirstBeadTrial];
    Mol.scaleCoul[0] = NewMol.scaleCoul[FirstBeadTrial];
    Mol.Type[0]      = NewMol.Type[FirstBeadTrial];
    Mol.MolID[0]     = NewMol.MolID[FirstBeadTrial];
  }

  size_t random_i = i*3 + offset;
  const Atoms AllData = d_a[SelectedComponent];
  //different from translation (where we copy a whole molecule), here we duplicate the properties of the first bead of a molecule
  // so use start_position, not real_pos
  //Zhao's note: when there are zero molecule for the species, we need to use some preset values
  //the first values always have some numbers. The xyz are not correct, but type and charge are correct. Use those.
  for(size_t a = 0; a < chainsize; a++)
  {
    double scale=0.0; double charge = 0.0; double scaleCoul = 0.0; size_t Type = 0;
    double XAngle = 3.1415 * 2.0 * (random[random_i] - 0.5);
    double YAngle = 3.1415 * 2.0 * (random[random_i+1] - 0.5);
    double ZAngle = 3.1415 * 2.0 * (random[random_i+2] - 0.5);
    if((!Deletion) && AllData.size == 0)
    {
      scale     = AllData.scale[1+a];
      charge    = AllData.charge[1+a];
      scaleCoul = AllData.scaleCoul[1+a];
      Type      = AllData.Type[1+a];
    }
    else
    {
      scale     = AllData.scale[start_position+a];
      charge    = AllData.charge[start_position+a];
      scaleCoul = AllData.scaleCoul[start_position+a];
      Type      = AllData.Type[start_position+a];
    }
    if((i==0) && (Deletion)) //if deletion, the first trial position is the old position of the selected molecule//
    {
      NewMol.x[i*chainsize+a] = AllData.x[start_position+a];
      NewMol.y[i*chainsize+a] = AllData.y[start_position+a];
      NewMol.z[i*chainsize+a] = AllData.z[start_position+a];
    }
    else //assign new positions based on new orientations//
    {
      double Vec[3]; 
      Vec[0] = AllData.x[1+a] - AllData.x[0]; //Uses the first atom as the template, Zhao's note: caveat: assuming first bead is the first in def file//
      Vec[1] = AllData.y[1+a] - AllData.y[0];
      Vec[2] = AllData.z[1+a] - AllData.z[0];
      //RotationAroundXAxis_trialOrientations(Vec, XAngle); RotationAroundYAxis_trialOrientations(Vec, YAngle); RotationAroundZAxis_trialOrientations(Vec, ZAngle);
      NewMol.x[i*chainsize+a] = Mol.x[0]+Vec[0];
      NewMol.y[i*chainsize+a] = Mol.y[0]+Vec[1];
      NewMol.z[i*chainsize+a] = Mol.z[0]+Vec[2];
    }
    NewMol.scale[i*chainsize+a] = scale; NewMol.charge[i*chainsize+a] = charge; NewMol.scaleCoul[i*chainsize+a] = scaleCoul;
    NewMol.Type[i*chainsize+a] = Type; NewMol.MolID[i*chainsize+a] = MolID;
  }
  device_flag[i] = false;
}


__global__ void get_random_trial_position_chain_Quaternion(Boxsize Box, Atoms* d_a, Atoms Mol, Atoms NewMol, bool* device_flag, double* random, size_t offset, size_t FirstBeadTrial, size_t start_position, size_t SelectedComponent, size_t MolID, bool Deletion, size_t chainsize)
{
  //Zhao's note: for trial orientations, each orientation may have more than 1 atom, do a for loop. So the threads are for different trial orientations, rather than different atoms in different orientations//
  //Zhao's note: chainsize is the size of the molecule excluding the first bead(-1).
  //Zhao's note: caveat: I am assuming the first bead of insertion is the first bead defined in def file. Need to relax this assumption in the future//
  //Zhao's note: First bead position stored in Mol, at position zero
  //Insertion/Widom: MolID = NewValue (Component.NumberOfMolecule_for_Component[SelectedComponent])
  //Deletion: MolID = selected ID
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i == 0)
  {
    Mol.x[0]         = NewMol.x[FirstBeadTrial];
    Mol.y[0]         = NewMol.y[FirstBeadTrial];
    Mol.z[0]         = NewMol.z[FirstBeadTrial];
    Mol.scale[0]     = NewMol.scale[FirstBeadTrial];
    Mol.charge[0]    = NewMol.charge[FirstBeadTrial];
    Mol.scaleCoul[0] = NewMol.scaleCoul[FirstBeadTrial];
    Mol.Type[0]      = NewMol.Type[FirstBeadTrial];
    Mol.MolID[0]     = NewMol.MolID[FirstBeadTrial];
  }
  //Quaternions uses 4 random seeds//
  size_t random_i = i*4 + offset;
  const Atoms AllData = d_a[SelectedComponent];
  //different from translation (where we copy a whole molecule), here we duplicate the properties of the first bead of a molecule
  // so use start_position, not real_pos
  //Zhao's note: when there are zero molecule for the species, we need to use some preset values
  //the first values always have some numbers. The xyz are not correct, but type and charge are correct. Use those.
  for(size_t a = 0; a < chainsize; a++)
  {
    double scale=0.0; double charge = 0.0; double scaleCoul = 0.0; size_t Type = 0;
    if((!Deletion) && AllData.size == 0)
    {
      scale     = AllData.scale[1+a];
      charge    = AllData.charge[1+a];
      scaleCoul = AllData.scaleCoul[1+a];
      Type      = AllData.Type[1+a];
    }
    else
    {
      scale     = AllData.scale[start_position+a];
      charge    = AllData.charge[start_position+a];
      scaleCoul = AllData.scaleCoul[start_position+a];
      Type      = AllData.Type[start_position+a];
    }
    if((i==0) && (Deletion)) //if deletion, the first trial position is the old position of the selected molecule//
    {
      NewMol.x[i*chainsize+a] = AllData.x[start_position+a];
      NewMol.y[i*chainsize+a] = AllData.y[start_position+a];
      NewMol.z[i*chainsize+a] = AllData.z[start_position+a];
    }
    else //assign new positions based on new orientations//
    {
      double Vec[3]; 
      Vec[0] = AllData.x[1+a] - AllData.x[0]; //Uses the first atom as the template, Zhao's note: caveat: assuming first bead is the first in def file//
      Vec[1] = AllData.y[1+a] - AllData.y[0];
      Vec[2] = AllData.z[1+a] - AllData.z[0];
      Rotate_Quaternions(Vec, random, random_i);
      NewMol.x[i*chainsize+a] = Mol.x[0]+Vec[0];
      NewMol.y[i*chainsize+a] = Mol.y[0]+Vec[1];
      NewMol.z[i*chainsize+a] = Mol.z[0]+Vec[2];
    }
    NewMol.scale[i*chainsize+a] = scale; NewMol.charge[i*chainsize+a] = charge; NewMol.scaleCoul[i*chainsize+a] = scaleCoul;
    NewMol.Type[i*chainsize+a] = Type; NewMol.MolID[i*chainsize+a] = MolID;
  }
  device_flag[i] = false;
}
static inline double Widom_Move_FirstBead_PARTIAL(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, bool Retrace, double &StoredR, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, bool DualPrecision)
{
  double OverlapCriteria = 1e5;
  size_t Atomsize = 0;
  bool Deletion = true;
  if(Insertion) Deletion = false;
  if(Retrace) Deletion = true; //Retrace is like Deletion, but Retrace only uses 1 trial position//

  bool Goodconstruction = false; size_t SelectedTrial = 0; double Rosenbluth = 0.0;
  //Depending on the mode of the move insertion/deletion vs. reinsertion-retrace. different number of trials is needed//
  //Retrace only needs 1 trial (the current position of the molecule)//
  size_t NumberOfTrials = Widom.NumberWidomTrials;
  if(Reinsertion && Retrace) NumberOfTrials = 1; //The Retrace move uses StoredR, so we only need 1 trial position for retrace//
  // here, the Atomsize is the total number of atoms in the system
  // number of threads needed = Atomsize*NumberOfTrials;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  size_t threadsNeeded = Atomsize * NumberOfTrials;
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];

  std::vector<double>Rosen; std::vector<double>energies; std::vector<size_t>Trialindex;
  
  //Zhao's note: consider doing the first beads first, might consider doing the whole molecule in the future.
  // Determine the MolID
  size_t SelectedMolID = 0;
  if(!Reinsertion && Insertion) //Pure insertion for Widom insertion or GCMC Insertion//
  {
    SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  }
  else if(Deletion || Reinsertion)
  {
    SelectedMolID = SelectedMolInComponent;
  }
  if((Random.offset+NumberOfTrials*3) > Random.randomsize) Random.offset = 0;
  get_random_trial_position_firstbead<<<1,NumberOfTrials>>>(Box, Sims.d_a, Sims.New, Sims.device_flag, Random.device_random, Random.offset, start_position, SelectedComponent, SelectedMolID, Deletion); checkCUDAError("error getting random trials");
  Random.offset += NumberOfTrials*3;

  // Setup the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0; Setup_threadblock(Atomsize, &Nblock, &Nthread); //for partial reduction, Nblock is number of blocks for **each trial**//
  //printf("Atomsize: %zu, Nblock: %zu, Nthread: %zu\n", Atomsize, Nblock, Nthread);
  Collapse_Framework_Energy_OVERLAP_PARTIAL<<<Nblock*NumberOfTrials, Nthread, Nthread * sizeof(double)>>>(Box, Sims.d_a, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Sims.device_flag, threadsNeeded,1, Nblock); checkCUDAError("Error calculating energies (PARTIAL SUM)");
  cudaMemcpy(Sims.flag, Sims.device_flag, NumberOfTrials*sizeof(bool), cudaMemcpyDeviceToHost);

  ///////////////////////
  //     Reduction     //
  ///////////////////////
  Host_sum_Widom(NumberOfTrials, SystemComponents.Beta, OverlapCriteria, Sims.Blocksum, Sims.flag, Nblock, energies, Trialindex, Rosen);
  if(!Retrace && Rosen.size() == 0) return 0.0;
  if(Insertion) SelectedTrial = SelectTrialPosition(Rosen);
  for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
  Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
  if(!Retrace && Rosenbluth < 1e-150) return 0.0;
  if(Rosen.size() > 0) Goodconstruction = true; //Whether the insertion/deletion construction return valid trials

  if(Reinsertion && Insertion) StoredR = Rosenbluth - Rosen[SelectedTrial]; //For insertion in Reinsertion//
  if(Reinsertion && Retrace) Rosenbluth += StoredR;  //For retrace in Reinsertion, just use the StoredR//
  double averagedRosen = Rosenbluth/double(Widom.NumberWidomTrials);
  size_t REALselected = 0;

  if(Goodconstruction)
  {
    REALselected = Trialindex[SelectedTrial];
  }
  else
  {
    return 0.0;
  }
  *REAL_Selected_Trial = REALselected;
  *SuccessConstruction = Goodconstruction;
  *energy = energies[SelectedTrial];
  return averagedRosen;
}

/*
double test_out_Quaternion(double* random, size_t offset)
{
  double x1; double x2; double x3; double x4;
  for(size_t i = 0; i < 20; i++)
  {
    //size_t random_i = offset + i;
    //x1   = 2.0 * (random[random_i]   - 0.5);
    //double x1sq = x1 * x1;
    //double residue_range = sqrt(1.0 - x1sq);
    //x2   = residue_range * 2.0 * (random[random_i + 1] - 0.5);
    
    double u = random[random_i]; double v = random[random_i + 1]; double w = random[random_i + 2];
    double pi=3.14159265358979323846;//Zhao's note: consider using M_PI 
    q1 = sqrt(1-u) * std::sin(2*pi*v);
    q2 = sqrt(1-u) * std::cos(2*pi*v);
    q3 = sqrt(u)   * std::sin(2*pi*w);
    q4 = sqrt(u)   * std::cos(2*pi*w);
    printf("x1: %.5f, x2: %.5f, x3: %.5f, x4: %.5f, length1: %.5f, length2: %.5f\n", x1, x2, x3, x4, x1 * x1 + x2 * x2, x3 * x3 + x4 * x4);
  }
}
*/
static inline double Widom_Move_Chain_PARTIAL(Boxsize& Box, Components& SystemComponents, Simulations& Sims, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, size_t FirstBeadTrial, bool DualPrecision)
{
  //printf("DOING RANDOM ORIENTAITONS\n");
  double OverlapCriteria = 1e5;
  size_t Atomsize = 0;
  size_t chainsize = SystemComponents.Moleculesize[SelectedComponent]-1; //size for the data of trial orientations are the number of trial orientations times the size of molecule excluding the first bead//
  bool Deletion = true;
  if(Insertion) Deletion = false;
  bool Goodconstruction = false; size_t SelectedTrial = 0; double Rosenbluth = 0.0;

  // here, the Atomsize is the total number of atoms in the system
  // number of threads needed = Atomsize*Widom.NumberWidomTrialsOrientations;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  size_t threadsNeeded = Atomsize*Widom.NumberWidomTrialsOrientations*chainsize;
 
  //Zhao's note: start position for chain (not the first atom) is the second atom in the molecule, might change in the future//
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent]+1;

  std::vector<double>Rosen;
  std::vector<double>energies; std::vector<size_t>Trialindex;
  // Determine the MolID
  size_t SelectedMolID = 0;
  if(!Reinsertion && Insertion)
  {
    SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  }
  else if(Reinsertion && Insertion)
  {
    SelectedMolID = SelectedMolInComponent;
  }
  else if(Deletion)
  {
    SelectedMolID = SelectedMolInComponent;
  }
  if((Random.offset+Widom.NumberWidomTrialsOrientations*3) > Random.randomsize) Random.offset = 0;
  //Get the first bead positions, and setup trial orientations//
  //get_random_trial_position_chain<<<1,Widom.NumberWidomTrialsOrientations>>>(Box, Sims.d_a, Sims.Old, Sims.New, Sims.device_flag, Random.device_random, Random.offset, FirstBeadTrial, start_position, SelectedComponent, SelectedMolID, Deletion, chainsize); checkCUDAError("error getting random trials orientations");
  get_random_trial_position_chain_Quaternion<<<1,Widom.NumberWidomTrialsOrientations>>>(Box, Sims.d_a, Sims.Old, Sims.New, Sims.device_flag, Random.device_random, Random.offset, FirstBeadTrial, start_position, SelectedComponent, SelectedMolID, Deletion, chainsize); checkCUDAError("error getting random trials orientations");
  //test_out_Quaternion(Random.host_random, Random.offset);
  Random.offset += Widom.NumberWidomTrialsOrientations*3;

  // Setup the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(Atomsize * chainsize, &Nblock, &Nthread); // use threadsNeeded for setting up the threads and blocks

  Collapse_Framework_Energy_OVERLAP_PARTIAL<<<Nblock*Widom.NumberWidomTrialsOrientations, Nthread, Nthread * sizeof(double)>>>(Box, Sims.d_a, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, Sims.device_flag, threadsNeeded, chainsize, Nblock); checkCUDAError("Error calculating energies (PARTIAL SUM, Orientations)");

  cudaMemcpy(Sims.flag, Sims.device_flag, Widom.NumberWidomTrialsOrientations*sizeof(bool), cudaMemcpyDeviceToHost);
  ////////////////////
  //    Reduction   //
  ////////////////////
  Host_sum_Widom(Widom.NumberWidomTrialsOrientations, SystemComponents.Beta, OverlapCriteria, Sims.Blocksum, Sims.flag, Nblock, energies, Trialindex, Rosen);
  
  if(Rosen.size() == 0) return 0.0;
  if(Insertion) SelectedTrial = SelectTrialPosition(Rosen);
  for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
  Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
  if(Rosenbluth < 1e-150) return 0.0;
  
  double averagedRosen = Rosenbluth/double(Widom.NumberWidomTrialsOrientations);
  size_t REALselected = 0;

  if(Rosen.size() > 0) Goodconstruction = true; //Whether the insertion/deletion construction return valid trials
  if(Goodconstruction)
  {
    REALselected = Trialindex[SelectedTrial];
  }
  else
  {
    return 0.0;
  }
  *REAL_Selected_Trial = REALselected;
  *SuccessConstruction = Goodconstruction;
  *energy = energies[SelectedTrial];
  return averagedRosen;
}
