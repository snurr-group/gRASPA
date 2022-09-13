#include <algorithm>
size_t SelectTrialPosition(std::vector <double> LogBoltzmannFactors); //In Zhao's code, LogBoltzmannFactors = Rosen
double Widom_Move_FirstBead(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy);

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

static inline double Widom_Move_FirstBead(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, RandomNumber& Random, WidomStruct& Widom, size_t SelectedMolInComponent, size_t SelectedComponent, bool Insertion, size_t *REAL_Selected_Trial, bool *SuccessConstruction, double *energy)
{
  double tot = 0.0;
  double start;
  double end;
  double OverlapCriteria = 1e6;
  size_t arraysize = 0;
  bool Deletion = true;
  if(Insertion) Deletion = false;
  bool Goodconstruction = false; size_t SelectedTrial = 0; double Rosenbluth = 0.0;
  // here, the arraysize is the total number of atoms in the system
  // number of threads needed = arraysize*Widom.NumberWidomTrials;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    arraysize += System[ijk].size;
  }
  size_t threadsNeeded = arraysize*Widom.NumberWidomTrials;
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];

  std::vector<double>Rosen;
  std::vector<double>SINGLERosen; double SINGLERosenbluth=0.0;
  std::vector<double>energies; std::vector<size_t>Trialindex;
  NewMol.size = Widom.NumberWidomTrials;
  size_t flag[NewMol.size]; for(size_t i=0;i<NewMol.size;i++){flag[i]=0;}
  size_t* device_flag;
  cudaMalloc(&device_flag, NewMol.size * sizeof(size_t));
  //consider doing the first beads first, might consider doing the whole molecule in the future.
  // Determine the MolID
  size_t SelectedMolID = 0;
  if(Insertion)
  {
    SelectedMolID = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  }
  else if(Deletion)
  {
    SelectedMolID = SelectedMolInComponent;
  }
  if((Random.offset+NewMol.size*3) > Random.randomsize) Random.offset = 0;
  get_random_trial_position_firstbead<<<1,NewMol.size>>>(Box, d_a, NewMol, Random.device_random, Random.offset, start_position, SelectedComponent, SelectedMolID, Deletion); checkCUDAError("error getting random trials");
  Random.offset += NewMol.size*3;
  //printf("SelectedComponent: %zu, SelectedMolID: %zu, start_position: %zu\n", SelectedComponent, SelectedMolID, start_position);

  // Setup the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(threadsNeeded, &Nblock, &Nthread); // use threadsNeeded for setting up the threads and blocks
  if(!Widom.Useflag)
  {
    Collapse_Framework_Energy<<<Nblock, Nthread>>>(Box, d_a, NewMol, FF, Widom.WidomFirstBeadResult, SelectedComponent, arraysize, threadsNeeded); //arraysize is the total number of atoms in the system
  }
  else
  {
    cudaMemset(device_flag, 0, NewMol.size*sizeof(size_t));
    Collapse_Framework_Energy_OVERLAP<<<Nblock, Nthread>>>(Box, d_a, NewMol, FF, Widom.WidomFirstBeadResult, SelectedComponent, arraysize, device_flag, threadsNeeded); checkCUDAError("Error calculating energies");
    cudaMemcpy(flag, device_flag, NewMol.size*sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  if(Widom.UseGPUReduction)
  {
    for(size_t i = 0; i < Widom.NumberWidomTrials; i++)
    {
      if(flag[i]==0)
      {
        tot = 0.0;
        tot = GPUReduction<BLOCKSIZE>(&Widom.WidomFirstBeadResult[i*arraysize], arraysize);
        if(tot < OverlapCriteria)
        {
          energies.push_back(tot);
          Trialindex.push_back(i);
          Rosen.push_back((-FF.Beta*tot));
          //printf("%zu trial E (collapsed, GPU): %.10f\n", i, tot);
        }
      }
    }
  }
  else
  {
    size_t SIZE = arraysize*Widom.NumberWidomTrials;
    double host_array[SIZE];
    cudaMemcpy(host_array, Widom.WidomFirstBeadResult, SIZE*sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t trial = 0; trial < Widom.NumberWidomTrials; trial++)
    {
      if(flag[trial] ==0)
      {
        int count = 0; tot = 0.0;
        for(size_t ijk=0; ijk < arraysize; ijk++)
        {
          size_t idx = trial*arraysize+ijk;
          tot+=host_array[idx];
          count++;
          if(tot > OverlapCriteria)
            break;
        }
        if(tot < OverlapCriteria)
        {
          energies.push_back(tot);
          Trialindex.push_back(trial);
          //Rosen.push_back(std::exp(-FF.Beta*tot));
          Rosen.push_back(-FF.Beta*tot);
          //printf("%zu trial E (collapsed): %.10f\n", trial, tot);
        }
      }
    }
    
    // Zhao's note: Tried OMP, does not help 
    /*double E_array[Widom.NumberWidomTrials]; size_t Trial_array[Widom.NumberWidomTrials]; double Rosen_array[Widom.NumberWidomTrials];
    bool good_array[Widom.NumberWidomTrials];
    #pragma omp parallel for 
    for(size_t trial = 0; trial < Widom.NumberWidomTrials; trial++)
    {
      E_array[trial] = 0.0;
      good_array[trial] = false;
      Trial_array[trial] = 0;
      Rosen_array[trial] = 0.0;
      if(flag[trial] ==0)
      {
        int count = 0; tot = 0.0;
        for(size_t ijk=0; ijk < arraysize; ijk++)
        {
          size_t idx = trial*arraysize+ijk;
          tot+=host_array[idx];
          count++;
          //if(tot > OverlapCriteria)
          //  break;
        }
        if(tot < OverlapCriteria)
        {
          E_array[trial] = tot;
          Trial_array[trial] = trial;
          //Rosen.push_back(std::exp(-FF.Beta*tot));
          Rosen_array[trial] = -FF.Beta*tot;
          good_array[trial] = true;
          //printf("%zu trial E (collapsed): %.10f\n", trial, tot);
        }
      }
    }
    //copy these array value to vector
    for(size_t trial = 0; trial < Widom.NumberWidomTrials; trial++)
    {
      if(good_array[trial])
      {
        energies.push_back(E_array[trial]);
        Trialindex.push_back(trial);
        Rosen.push_back(Rosen_array[trial]);
      }
    }*/
  }
  if(Rosen.size() == 0) return;
  if(Insertion) SelectedTrial = SelectTrialPosition(Rosen);
  for(size_t a = 0; a < Rosen.size(); a++) Rosen[a] = std::exp(Rosen[a]);
  Rosenbluth =std::accumulate(Rosen.begin(), Rosen.end(), decltype(Rosen)::value_type(0));
  if(Rosenbluth < 1e-150) return;
  //printf("accumulate Rosen: %.10f, SelectedTrial: %zu, realtrial: %zu\n", Rosenbluth, SelectedTrial, Trialindex[SelectedTrial]);
  double averagedRosen = Rosenbluth/double(Widom.NumberWidomTrials);
  size_t REALselected = 0;
  

  if(Rosen.size() > 0) Goodconstruction = true; //Whether the insertion/deletion construction return valid trials
  cudaFree(device_flag);
  if(Insertion and Goodconstruction)
  {
    //select a trial position
    //for(size_t xx = 0; xx < Rosen.size(); xx++){printf("Trial: %zu, Energy: %.10f, Rosenbluth: %.10f\n", Trialindex[xx], energies[xx], Rosen[xx]);}
  }
  else if(Deletion)
  {
    //for deletion, the selected trial position is always the old position, which is the first one in the NewMol array

  }
  if(Goodconstruction)
  {
    REALselected = Trialindex[SelectedTrial];
    //printf("SelectedTrial: %zu\n", SelectedTrial);
    //printf("RealSelected: %zu\n", REALselected);
    //printf("Selected Trial: %zu, Energy: %.10f, Rosenbluth: %.10f\n", Trialindex[SelectedTrial], energies[SelectedTrial], Rosen[SelectedTrial]);
  }
  else
  {
    //printf("No good trials\n");
    return;
  }
  *REAL_Selected_Trial = REALselected;
  *SuccessConstruction = Goodconstruction;
  *energy = energies[SelectedTrial];
  return averagedRosen;
}
