//Translation Move//
double Translation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent);

void Update_Max_Translation(ForceField FF, Move_Statistics MoveStats);

void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread);

//Rotation Move//
double Rotation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent);

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
  size_t random_i = offset; // all atoms in the molecule are moving in the same direction, so no i in this variable
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

static inline double Translation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  double tot = 0.0;
  //double result = 0.0;
  SystemComponents.Moves[SelectedComponent].TranslationTotal++;
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += System[ijk].size;
  }
  size_t chainsize;
  //Set up Old position and New position arrays
  chainsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  Mol.size = chainsize;
  NewMol.size = Mol.size;
  if(Mol.size >= Mol.Allocate_size)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  if((Random.offset + 3*Mol.size) >= Random.randomsize) Random.offset = 0;
  get_new_translation_position<<<1, Mol.size>>>(d_a, Mol, NewMol, FF, start_position, SelectedComponent, Random.device_random, Random.offset); Random.offset += 3*Mol.size;

  // Setup for the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(Atomsize * chainsize, &Nblock, &Nthread);
  //double* BlockdU; Zhao's note: add back dUdlambda later //
  Energy_difference_PARTIAL<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, d_a, Mol, NewMol, FF, Widom.Blocksum, SelectedComponent, Atomsize, chainsize, Nthread);
  double BlockResult[Nblock];
  cudaMemcpy(BlockResult, Widom.Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++) tot += BlockResult[i];
  if (get_random_from_zero_to_one() < std::exp(-SystemComponents.Beta * tot))
  {
    update_translation_position<<<1,Mol.size>>>(d_a, NewMol, start_position, SelectedComponent);
    SystemComponents.Moves[SelectedComponent].TranslationAccepted ++;
    return tot;
  }
  return 0.0;
}

__device__ void RotationAroundXAxis(Atoms Mol,size_t i,double theta)
{
  double w,s,c,rot[3*3];

  c=cos(theta);
  s=sin(theta);

  rot[0*3+0]=1.0; rot[1*3+0]=0.0;  rot[2*3+0]=0.0;
  rot[0*3+1]=0.0; rot[1*3+1]=c;    rot[2*3+1]=-s;
  rot[0*3+2]=0.0; rot[1*3+2]=s;    rot[2*3+2]=c;

  w=Mol.x[i]*rot[0*3+0]+Mol.y[i]*rot[0*3+1]+Mol.z[i]*rot[0*3+2];
  s=Mol.x[i]*rot[1*3+0]+Mol.y[i]*rot[1*3+1]+Mol.z[i]*rot[1*3+2];
  c=Mol.x[i]*rot[2*3+0]+Mol.y[i]*rot[2*3+1]+Mol.z[i]*rot[2*3+2];
  Mol.x[i]=w;
  Mol.y[i]=s;
  Mol.z[i]=c;
}

__device__ void RotationAroundYAxis(Atoms Mol,size_t i,double theta)
{
  double w,s,c,rot[3*3];

  c=cos(theta);
  s=sin(theta);

  rot[0*3+0]=c;   rot[1*3+0]=0;    rot[2*3+0]=s;
  rot[0*3+1]=0;   rot[1*3+1]=1.0;  rot[2*3+1]=0;
  rot[0*3+2]=-s;  rot[1*3+2]=0;    rot[2*3+2]=c;

  w=Mol.x[i]*rot[0*3+0]+Mol.y[i]*rot[0*3+1]+Mol.z[i]*rot[0*3+2];
  s=Mol.x[i]*rot[1*3+0]+Mol.y[i]*rot[1*3+1]+Mol.z[i]*rot[1*3+2];
  c=Mol.x[i]*rot[2*3+0]+Mol.y[i]*rot[2*3+1]+Mol.z[i]*rot[2*3+2];
  Mol.x[i]=w;
  Mol.y[i]=s;
  Mol.z[i]=c;
}

__device__ void RotationAroundZAxis(Atoms Mol,size_t i,double theta)
{
  double w,s,c,rot[3*3];

  c=cos(theta);
  s=sin(theta);

  rot[0*3+0]=c;   rot[1*3+0]=-s;   rot[2*3+0]=0;
  rot[0*3+1]=s;   rot[1*3+1]=c;    rot[2*3+1]=0;
  rot[0*3+2]=0;   rot[1*3+2]=0;    rot[2*3+2]=1.0;

  w=Mol.x[i]*rot[0*3+0]+Mol.y[i]*rot[0*3+1]+Mol.z[i]*rot[0*3+2];
  s=Mol.x[i]*rot[1*3+0]+Mol.y[i]*rot[1*3+1]+Mol.z[i]*rot[1*3+2];
  c=Mol.x[i]*rot[2*3+0]+Mol.y[i]*rot[2*3+1]+Mol.z[i]*rot[2*3+2];
  Mol.x[i]=w;
  Mol.y[i]=s;
  Mol.z[i]=c;
}

__global__ void get_new_rotation_position(Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, size_t start_position, size_t SelectedComponent, double* random, size_t offset)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = offset; // all atoms in the molecule are moving in the same direction, so no i in this variable
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
  double XAngle = FF.MaxRotation[0] * 2.0 * (random[random_i] - 0.5);
  double YAngle = FF.MaxRotation[1] * 2.0 * (random[random_i+1] - 0.5);
  double ZAngle = FF.MaxRotation[2] * 2.0 * (random[random_i+2] - 0.5);
  //Old Distance, stored temporarily in NewMol// 
  //Zhao's note: the start position is the position of the first atom in the molecule. Here we assume that the origin of the rotation is the first atom. This is not necessarily true, might need to add a new value later//
  NewMol.x[i] = Mol.x[i] - AllData.x[start_position]; NewMol.y[i] = Mol.y[i] - AllData.y[start_position]; NewMol.z[i] = Mol.z[i] - AllData.z[start_position];
  //Rotation around X,Y, and Z// //Zhao's note: think about quaternions in the future//
  RotationAroundXAxis(NewMol, i, XAngle);
  RotationAroundYAxis(NewMol, i, YAngle);
  RotationAroundZAxis(NewMol, i, ZAngle);
  //Update NewMol//
  NewMol.x[i] += AllData.x[start_position]; NewMol.y[i] += AllData.y[start_position]; NewMol.z[i] += AllData.z[start_position];
  Mol.scale[i] = scale; Mol.charge[i] = charge; Mol.scaleCoul[i] = scaleCoul; Mol.Type[i] = Type; Mol.MolID[i] = MolID;
  NewMol.scale[i] = scale; NewMol.charge[i] = charge; NewMol.scaleCoul[i] = scaleCoul; NewMol.Type[i] = Type; NewMol.MolID[i] = MolID;
}

static inline double Rotation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  double tot = 0.0;
  SystemComponents.Moves[SelectedComponent].RotationTotal++;
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += System[ijk].size;
  }
  size_t chainsize;
  //Set up Old position and New position arrays
  chainsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  Mol.size = chainsize; //Get the size of the selected Molecule
  NewMol.size = Mol.size;
  if(Mol.size >= Mol.Allocate_size)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  //Determine the axis of rotation//
  if((Random.offset + 3*Mol.size) >= Random.randomsize) Random.offset = 0;
  get_new_rotation_position<<<1,Mol.size>>>(d_a, Mol, NewMol, FF, start_position, SelectedComponent, Random.device_random, Random.offset);Random.offset += 3*Mol.size;
  
  // Setup for the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(Atomsize * chainsize, &Nblock, &Nthread);
  //double* BlockdU; Zhao's note: add back dUdlambda later //
  Energy_difference_PARTIAL<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, d_a, Mol, NewMol, FF, Widom.Blocksum, SelectedComponent, Atomsize, chainsize, Nthread);
  double BlockResult[Nblock];
  cudaMemcpy(BlockResult, Widom.Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost); 
  for(size_t i = 0; i < Nblock; i++) tot += BlockResult[i];
  if (get_random_from_zero_to_one() < std::exp(-SystemComponents.Beta * tot))
  {
    //printf("accepted, tot: %.10f\n", tot);
    update_translation_position<<<1,Mol.size>>>(d_a, NewMol, start_position, SelectedComponent);
    SystemComponents.Moves[SelectedComponent].RotationAccepted ++; 
    return tot;
  }
  return 0.0;
}
