#include <algorithm>
#include <omp.h>
#include <cuda_fp16.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
size_t SelectTrialPosition(std::vector <double> LogBoltzmannFactors); //In Zhao's code, LogBoltzmannFactors = Rosen
double Widom_Move_FirstBead(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, bool Retrace, double &StoredR, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, bool DualPrecision);

double Widom_Move_Chain(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, size_t FirstBeadTrial, bool DualPrecision);

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

__global__ void get_random_trial_position_firstbead(Boxsize Box, Atoms* d_a, Atoms NewMol, double* random, size_t offset, size_t start_position, size_t SelectedComponent, size_t MolID, bool Deletion)
{
  //Insertion/Widom: MolID = NewValue (Component.NumberOfMolecule_for_Component[SelectedComponent]
  //Deletion: MolID = selected ID
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = i*3 + offset;
  const Atoms AllData = d_a[SelectedComponent];
  const size_t real_pos = start_position + i;
  //different from translation (where we copy a whole molecule), here we duplicate the properties of the first bead of a molecule
  // so use start_position, not real_pos
  //Zhao's note: when there are zero molecule for the species, we need to use some preset values
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
  //printf("i: %lu, scale: %.10f, charge: %.10f, scaleCoul: %.10f, Type: %lu, MolID: %lu\n", i, scale, charge, scaleCoul, Type, MolID);
}

template<typename T>
inline void GPU_Sum_Widom(size_t NumberWidomTrials, double Beta, double OverlapCriteria, T* energy_array, size_t* flag, size_t arraysize, std::vector<double>& energies, std::vector<size_t>& Trialindex, std::vector<double>& Rosen)
{
  for(size_t i = 0; i < NumberWidomTrials; i++)
  {
    if(flag[i]==0)
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
inline void Host_sum_Widom(size_t NumberWidomTrials, double Beta, T Overlap, T* energy_array, size_t* flag, size_t arraysize, std::vector<double>& energies, std::vector<size_t>& Trialindex, std::vector<double>& Rosen)
{
  std::vector<size_t>reasonable_trials;
  for(size_t i = 0; i < NumberWidomTrials; i++){
    if(flag[i] ==0){
      reasonable_trials.push_back(i);}}
  T host_array[reasonable_trials.size() * arraysize];

  const size_t num_streams = 32;
  /*cudaStream_t streams[num_streams];
  for (int j=0; j<num_streams; j++)
    cudaStreamCreateWithFlags(&streams[j],cudaStreamNonBlocking);
  */
  for(size_t i = 0; i < reasonable_trials.size(); i++)
  {
    size_t trial = reasonable_trials[i];
    //int id = i % num_streams;
    //cudaMemcpyAsync(&host_array[i*arraysize], &energy_array[trial*arraysize], arraysize*sizeof(T), cudaMemcpyDeviceToHost, streams[id]);
    cudaMemcpy(&host_array[i*arraysize], &energy_array[trial*arraysize], arraysize*sizeof(T), cudaMemcpyDeviceToHost);
    T float_tot = 0.0;
    for(size_t ijk=0; ijk < arraysize; ijk++)
    {
      float_tot+=host_array[i*arraysize + ijk];
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
  //for (int j=0; j<num_streams; j++) cudaStreamDestroy(streams[j]);
}

inline void Update_Dual_Precision_Rosenbluth(WidomStruct& Widom, double Beta, size_t arraysize, size_t SelectedTrial, size_t REALselected, std::vector<double>& energies, double averagedRosen)
{
  double tot = 0.0;
  if(Widom.UseGPUReduction){
    tot = GPUReduction<BLOCKSIZE>(&Widom.WidomFirstBeadResult[REALselected*arraysize], arraysize);}
  else{
    tot = 0.0; double host_array[arraysize]; cudaMemcpy(host_array, &Widom.WidomFirstBeadResult[REALselected*arraysize], arraysize*sizeof(double), cudaMemcpyDeviceToHost);
   for(size_t i = 0; i < arraysize; i++) tot += host_array[i];
  }
  //Don't pass this delta-E to the later function, merge with the Rosenbluth Weight//
  double float_tot = energies[SelectedTrial];
  double energy_diff_float_double = (tot-float_tot);
  //Update the Energy of the SelectedTrial//
  energies[SelectedTrial] += energy_diff_float_double;
  //Update Rosenbluth Weight//
  averagedRosen *= std::exp(-Beta * energy_diff_float_double);
}

static inline double Widom_Move_FirstBead(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, bool Reinsertion, bool Retrace, double &StoredR, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy, bool DualPrecision)
{
  double tot = 0.0;
  double start;
  double end;
  double OverlapCriteria = 1e6;
  size_t arraysize = 0;
  bool Deletion = true;
  if(Insertion) Deletion = false;
  if(Retrace) Deletion = true; //Retrace is like Deletion, but Retrace only uses 1 trial position//

  float *widomfloat; float float_tot;

  bool Goodconstruction = false; size_t SelectedTrial = 0; double Rosenbluth = 0.0;
  //Depending on the mode of the move insertion/deletion vs. reinsertion-retrace. different number of trials is needed//
  //Retrace only needs 1 trial (the current position of the molecule)//
  size_t NumberOfTrials = Widom.NumberWidomTrials;
  if(Reinsertion && Retrace) NumberOfTrials = 1; //The Retrace move uses StoredR, so we only need 1 trial position for retrace//
  // here, the arraysize is the total number of atoms in the system
  // number of threads needed = arraysize*NumberOfTrials;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    arraysize += System[ijk].size;
  }
  size_t threadsNeeded = arraysize * NumberOfTrials;
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];

  if(DualPrecision) cudaMalloc(&widomfloat, sizeof(float)*threadsNeeded);

  std::vector<double>Rosen; std::vector<double>energies; std::vector<size_t>Trialindex;
  NewMol.size = NumberOfTrials;
  size_t flag[NewMol.size]; for(size_t i=0;i<NewMol.size;i++){flag[i]=0;}
  size_t* device_flag;
  cudaMalloc(&device_flag, NewMol.size * sizeof(size_t));
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
  if((Random.offset+NewMol.size*3) > Random.randomsize) Random.offset = 0;
  get_random_trial_position_firstbead<<<1,NewMol.size>>>(Box, d_a, NewMol, Random.device_random, Random.offset, start_position, SelectedComponent, SelectedMolID, Deletion); checkCUDAError("error getting random trials");
  Random.offset += NewMol.size*3;

  // Setup the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(threadsNeeded, &Nblock, &Nthread); // use threadsNeeded for setting up the threads and blocks
  if(!Widom.Useflag)
  {
    Collapse_Framework_Energy<<<Nblock, Nthread>>>(Box, d_a, NewMol, FF, Widom.WidomFirstBeadResult, SelectedComponent, arraysize, threadsNeeded,1); //arraysize is the total number of atoms in the system
  }
  else
  {
    cudaMemset(device_flag, 0, NewMol.size*sizeof(size_t));
    if(DualPrecision)
    {
      Collapse_Framework_Energy_OVERLAP_FLOAT<<<Nblock, Nthread>>>(Box, d_a, NewMol, FF, Widom.WidomFirstBeadResult, SelectedComponent, arraysize, device_flag, threadsNeeded,1, widomfloat); checkCUDAError("Error calculating energies (float)");
    }
    else
    {
      Collapse_Framework_Energy_OVERLAP<<<Nblock, Nthread>>>(Box, d_a, NewMol, FF, Widom.WidomFirstBeadResult, SelectedComponent, arraysize, device_flag, threadsNeeded,1); checkCUDAError("Error calculating energies");
    }
    cudaMemcpy(flag, device_flag, NewMol.size*sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  ///////////////////////
  //     Reduction     //
  ///////////////////////
  if(Widom.UseGPUReduction) //Perform Reduction on the GPU//
  {
    if(DualPrecision && NumberOfTrials > 1) //If there is only 1 trial position, there is no need to do Dual-Precision CBMC//
    {
      GPU_Sum_Widom(NumberOfTrials, FF.Beta, OverlapCriteria, widomfloat, flag, arraysize, energies, Trialindex, Rosen);
    }
    else
    {
      GPU_Sum_Widom(NumberOfTrials, FF.Beta, OverlapCriteria, Widom.WidomFirstBeadResult, flag, arraysize, energies, Trialindex, Rosen);
    }
  }
  else //Perform Reduction on the CPU//
  {
    if(DualPrecision && NumberOfTrials > 1){ //If there is only 1 trial position, there is no need to do Dual-Precision CBMC//
      Host_sum_Widom(NumberOfTrials, FF.Beta, static_cast<float>(OverlapCriteria), widomfloat, flag, arraysize, energies, Trialindex, Rosen);}
    else{
      Host_sum_Widom(NumberOfTrials, FF.Beta, OverlapCriteria, Widom.WidomFirstBeadResult, flag, arraysize, energies, Trialindex, Rosen);}
  }
  if(!Retrace && Rosen.size() == 0) return;
  for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
  Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
  if(!Retrace && Rosenbluth < 1e-150) return;
  if(Rosen.size() > 0) Goodconstruction = true; //Whether the insertion/deletion construction return valid trials

  if(Insertion) SelectedTrial = SelectTrialPosition(Rosen);
  if(Reinsertion && Insertion) StoredR = Rosenbluth - Rosen[SelectedTrial]; //For insertion in Reinsertion//
  if(Reinsertion && Retrace) Rosenbluth += StoredR;  //For retrace in Reinsertion, just use the StoredR//
  double averagedRosen = Rosenbluth/double(Widom.NumberWidomTrials);
  size_t REALselected = 0;

  cudaFree(device_flag);
  if(DualPrecision) cudaFree(widomfloat);
  if(Goodconstruction)
  {
    REALselected = Trialindex[SelectedTrial];
    if(DualPrecision && NumberOfTrials > 1)
    {
      Update_Dual_Precision_Rosenbluth(Widom, FF.Beta, arraysize, SelectedTrial, REALselected, energies, averagedRosen);
    }
  }
  else
  {
    return;
  }
 
 
  *REAL_Selected_Trial = REALselected;
  *SuccessConstruction = Goodconstruction;
  *energy = energies[SelectedTrial];
  return averagedRosen;
}
