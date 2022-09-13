double Translation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, double* y, double* dUdlambda, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent);

void Update_Max_Translation(ForceField FF, Move_Statistics MoveStats);

void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread);

__global__ void update_device_Max_Translation(ForceField FF, double scale)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  FF.MaxTranslation[i] *= scale;
}

static inline void Update_Max_Translation(ForceField FF, Move_Statistics MoveStats)
{
  MoveStats.TranslationAccRatio = static_cast<double>(MoveStats.TranslationAccepted)/MoveStats.TranslationTotal;
  //printf("AccRatio is %.10f\n", MoveStats.TranslationAccRatio);
  if(MoveStats.TranslationAccRatio > 0.5)
  {
    update_device_Max_Translation<<<1,3>>>(FF, 1.05);
  }
  else
  {
    update_device_Max_Translation<<<1,3>>>(FF, 0.95);
  }
  MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal=0;
  printf("Translation Updated\n");
}

static inline void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  size_t value = arraysize;
  if(value >= DEFAULTTHREAD) value = DEFAULTTHREAD;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  *Nthread = value;
  *Nblock = blockValue;
}

__global__ void get_new_translation_position(Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, size_t start_position, size_t SelectedComponent, double* random, size_t offset)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = i + offset;
  size_t real_pos = start_position + i;
  const Atoms AllData = d_a[SelectedComponent];
  const double scale = AllData.scale[real_pos];
  const double charge = AllData.charge[real_pos];
  const double scaleCoul = AllData.scaleCoul[real_pos];
  const size_t Type = AllData.Type[real_pos];
  const size_t MolID = AllData.MolID[real_pos];
  Mol.x[i] = AllData.x[real_pos];
  Mol.y[i] = AllData.y[real_pos];
  Mol.z[i] = AllData.z[real_pos];
  NewMol.x[i] = Mol.x[i] + FF.MaxTranslation[0] * 2.0 * (random[random_i] - 0.5);
  NewMol.y[i] = Mol.y[i] + FF.MaxTranslation[1] * 2.0 * (random[random_i+1] - 0.5);
  NewMol.z[i] = Mol.z[i] + FF.MaxTranslation[2] * 2.0 * (random[random_i+2] - 0.5);
  Mol.scale[i] = scale; Mol.charge[i] = charge; Mol.scaleCoul[i] = scaleCoul; Mol.Type[i] = Type; Mol.MolID[i] = MolID;
  NewMol.scale[i] = scale; NewMol.charge[i] = charge; NewMol.scaleCoul[i] = scaleCoul; NewMol.Type[i] = Type; NewMol.MolID[i] = MolID;
}

__global__ void update_translation_position(Atoms* d_a, Atoms NewMol, size_t start_position, size_t SelectedComponent)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[SelectedComponent].x[start_position+i] = NewMol.x[i];
  d_a[SelectedComponent].y[start_position+i] = NewMol.y[i];
  d_a[SelectedComponent].z[start_position+i] = NewMol.z[i];
}

static inline double Translation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, double* y, double* dUdlambda, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  double tot = 0.0;
  //double result = 0.0;
  MoveStats.TranslationTotal++;
  bool use_curand = true;
  size_t arraysize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    arraysize += System[ijk].size;
  }

  //Set up Old position and New position arrays
  Mol.size = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  NewMol.size = Mol.size;
  if(Mol.size >= Mol.Allocate_size)
  {
    throw std::runtime_error("restart file' not found\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  if((Random.offset + 3*Mol.size) >= Random.randomsize) Random.offset = 0;
  get_new_translation_position<<<1, Mol.size>>>(d_a, Mol, NewMol, FF, start_position, SelectedComponent, Random.device_random, Random.offset); Random.offset += 3*Mol.size;

  // Setup for the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(arraysize, &Nblock, &Nthread);
  //printf("Nthread: %zu, Nblock: %zu\n", Nthread, Nblock);
  Framework_energy_difference_SoA<<<Nblock, Nthread>>>(Box, d_a, Mol, NewMol, FF, y, dUdlambda,SelectedComponent, arraysize);
  tot = GPUReduction<BLOCKSIZE>(y, arraysize);
  if (get_random_from_zero_to_one() < std::exp(-FF.Beta * tot))
  {
    //printf("tot: %.10f\n", tot);
    update_translation_position<<<1,Mol.size>>>(d_a, NewMol, start_position, SelectedComponent);
    MoveStats.TranslationAccepted ++;
    return tot;
  }
  return 0.0;
}
