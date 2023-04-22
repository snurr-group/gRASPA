__device__ void get_nsim(Simulations* Sims, size_t *nsim, size_t *blockbefore, size_t NumberOfSimulations, size_t BlockID)
{
  size_t totalblock = 0; size_t tempsim = 0;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    if(BlockID >= (totalblock + Sims[i].Nblocks))
    {
      tempsim++;
      totalblock += Sims[i].Nblocks;
    }
  }
  *nsim        = tempsim;
  *blockbefore = totalblock;
}

__device__ void get_position(const Atoms* System, size_t *posi, size_t *comp, size_t i, size_t NumberComp)
{
  size_t temppos = i; size_t totalsize = 0; size_t tempcomp = 0;
  for(size_t ijk = 0; ijk < NumberComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(temppos >= totalsize)
    {
      tempcomp++;
      temppos -= System[ijk].size;
    }
  }
  *posi = temppos;
  *comp = tempcomp;
}

__global__ void Energy_difference_PARTIAL_FLAG_HGGG(Boxsize Box, Atoms* System, Atoms Old, Atoms New, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, bool* flag, size_t HG_Nblock, size_t GG_Nblock, bool Do_New, bool Do_Old)
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum, energy + dUdlambda//
  int cache_id = threadIdx.x;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; //sdata[ij_within_block].y = 0.0;
  //Initialize Blocky and BlockdUdlambda//
  BlockEnergy[blockIdx.x] = 0.0; //BlockdUdlambda[blockIdx.x] = 0.0;

  __shared__ bool Blockflag = false;

  size_t ij = total_ij;
  if(blockIdx.x >= HG_Nblock)
    ij -= HG_Nblock * blockDim.x; //It belongs to the Guest-Guest Interaction//

  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; //+ ij/totalAtoms; // position in Old and New

  const size_t NTotalComp = 2; //Zhao's note: need to change here for multicomponent (Nguest comp > 1)
  const size_t NHostComp  = 1;
  const size_t NGuestComp = 1;
  size_t posi = i; size_t totalsize= 0;
  size_t startComp = 0; size_t endComp = 0;
  if(blockIdx.x < HG_Nblock) //Host-Guest Interaction//
  { 
    startComp = 0; endComp = NHostComp;
  }
  else //Guest-Guest Interaction//
  {
    startComp = NHostComp; endComp = NTotalComp;
  }
  size_t comp = startComp;
  for(size_t ijk = startComp; ijk < endComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= totalsize)
    {
      comp++;
      posi -= System[ijk].size;
    }
  }

  if(posi < System[comp].size)
  {

  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double tempy = 0.0; double tempdU = 0.0;
  if(!((MoleculeID == New.MolID[0]) &&(comp == ComponentID)) && Do_New) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    double posvec[3] = {Component.x[posi] - New.x[j], Component.y[posi] - New.y[j], Component.z[posi] - New.z[j]};

    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = New.Type[j];
      const double scaleB = New.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      if(result[0] > FF.OverlapCriteria) { Blockflag = true; flag[0] = true; }
      if(rr_dot < 0.01)                  { Blockflag = true; flag[0] = true; } //DistanceCheck//
      tempy  += result[0];
      tempdU += result[1];
    }

    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = New.charge[j];
      const double scalingCoulombB = New.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy += result[0]; //prefactor merged in the CoulombReal function
    }
  }
  if(!((MoleculeID == Old.MolID[0]) &&(comp == ComponentID)) && Do_Old) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  OLD  //
    ///////////
    double posvec[3] = {Component.x[posi] - Old.x[j], Component.y[posi] - Old.y[j], Component.z[posi] - Old.z[j]};
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Old.Type[j];
      const double scaleB = Old.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      tempy  -= result[0];
      tempdU -= result[1];
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Old.charge[j];
      const double scalingCoulombB = Old.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy -= result[0]; //prefactor merged in the CoulombReal function
    }
  }
  sdata[ij_within_block] = tempy; //sdata[ij_within_block].y = tempdU;
  }
  __syncthreads();
  //Partial block sum//
  if(!Blockflag)
  {
    int i=blockDim.x / 2;
    while(i != 0)
    {
      if(cache_id < i)
      {
        sdata[cache_id] += sdata[cache_id + i]; //sdata[cache_id].y += sdata[cache_id + i].y;
      }
      __syncthreads();
      i /= 2;
    }
    if(cache_id == 0)
    {
      BlockEnergy[blockIdx.x] = sdata[0]; //BlockdUdlambda[blockIdx.x] = sdata[0].y;
    }
  }
  /*else
  {
    flag[0]=true;
  }*/
}


static inline MoveEnergy SingleMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType);

void Update_Max_Translation(Move_Statistics MoveStats, Simulations& Sims);

void Update_Max_Rotation(Move_Statistics MoveStats, Simulations& Sims);

void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread);

static inline void Update_Max_Translation(Move_Statistics MoveStats, Simulations& Sims)
{
  MoveStats.TranslationAccRatio = static_cast<double>(MoveStats.TranslationAccepted)/MoveStats.TranslationTotal;
  //printf("AccRatio is %.10f\n", MoveStats.TranslationAccRatio);
  if(MoveStats.TranslationAccRatio > 0.5)
  {
    Sims.MaxTranslation.x *= 1.05; Sims.MaxTranslation.y *= 1.05; Sims.MaxTranslation.z *= 1.05;
  }
  else
  {
    Sims.MaxTranslation.x *= 0.95; Sims.MaxTranslation.y *= 0.95; Sims.MaxTranslation.z *= 0.95;
  }
  if(Sims.MaxTranslation.x < 0.01) Sims.MaxTranslation.x = 0.01;
  if(Sims.MaxTranslation.y < 0.01) Sims.MaxTranslation.y = 0.01;
  if(Sims.MaxTranslation.z < 0.01) Sims.MaxTranslation.z = 0.01;

  if(Sims.MaxTranslation.x > 5.0) Sims.MaxTranslation.x = 5.0;
  if(Sims.MaxTranslation.y > 5.0) Sims.MaxTranslation.y = 5.0;
  if(Sims.MaxTranslation.z > 5.0) Sims.MaxTranslation.z = 5.0;
  MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal=0;
}

static inline void Update_Max_Rotation(Move_Statistics MoveStats, Simulations& Sims)
{
  MoveStats.RotationAccRatio = static_cast<double>(MoveStats.RotationAccepted)/MoveStats.RotationTotal;
  //printf("AccRatio is %.10f\n", MoveStats.TranslationAccRatio);
  if(MoveStats.RotationAccRatio > 0.5)
  {
    Sims.MaxRotation.x *= 1.05; Sims.MaxRotation.y *= 1.05; Sims.MaxRotation.z *= 1.05;
  }
  else
  {
    Sims.MaxRotation.x *= 0.95; Sims.MaxRotation.y *= 0.95; Sims.MaxRotation.z *= 0.95;
  }
  if(Sims.MaxRotation.x < 0.01) Sims.MaxRotation.x = 0.01;
  if(Sims.MaxRotation.y < 0.01) Sims.MaxRotation.y = 0.01;
  if(Sims.MaxRotation.z < 0.01) Sims.MaxRotation.z = 0.01;

  if(Sims.MaxRotation.x > 3.14) Sims.MaxRotation.x = 3.14;
  if(Sims.MaxRotation.y > 3.14) Sims.MaxRotation.y = 3.14;
  if(Sims.MaxRotation.z > 3.14) Sims.MaxRotation.z = 3.14;
  MoveStats.RotationAccepted = 0; MoveStats.RotationTotal=0;
}

static inline void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  if(arraysize == 0)  return;
  size_t value = arraysize;
  if(value >= DEFAULTTHREAD) value = DEFAULTTHREAD;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  //Zhao's note: Default thread should always be 64, 128, 256, 512, ...
  // This is because we are using partial sums, if arraysize is smaller than defaultthread, we need to make sure that 
  //while Nthread is dividing by 2, it does not generate ODD NUMBER (for example, 5/2 = 2, then element 5 will be ignored)//
  *Nthread = DEFAULTTHREAD;
  *Nblock = blockValue;
}

__global__ void update_translation_position(Atoms* d_a, Atoms NewMol, size_t start_position, size_t SelectedComponent)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[SelectedComponent].x[start_position+i] = NewMol.x[i];
  d_a[SelectedComponent].y[start_position+i] = NewMol.y[i];
  d_a[SelectedComponent].z[start_position+i] = NewMol.z[i];
  d_a[SelectedComponent].scale[start_position+i] = NewMol.scale[i];
  d_a[SelectedComponent].charge[start_position+i] = NewMol.charge[i];
  d_a[SelectedComponent].scaleCoul[start_position+i] = NewMol.scaleCoul[i];
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

__global__ void get_new_position(Simulations& Sim, ForceField FF, size_t start_position, size_t SelectedComponent, double* random, size_t offset, int MoveType)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = offset; // all atoms in the molecule are moving in the same direction, so no i in this variable
  size_t real_pos = start_position + i;

  const double x         = Sim.d_a[SelectedComponent].x[real_pos];
  const double y         = Sim.d_a[SelectedComponent].y[real_pos];
  const double z         = Sim.d_a[SelectedComponent].z[real_pos];
  const double scale     = Sim.d_a[SelectedComponent].scale[real_pos];
  const double charge    = Sim.d_a[SelectedComponent].charge[real_pos];
  const double scaleCoul = Sim.d_a[SelectedComponent].scaleCoul[real_pos];
  const size_t Type      = Sim.d_a[SelectedComponent].Type[real_pos];
  const size_t MolID     = Sim.d_a[SelectedComponent].MolID[real_pos];

  switch(MoveType)
  {
    case TRANSLATION://TRANSLATION:
    {
      Sim.New.x[i] = x + Sim.MaxTranslation.x * 2.0 * (random[random_i] - 0.5);
      Sim.New.y[i] = y + Sim.MaxTranslation.y * 2.0 * (random[random_i+1] - 0.5);
      Sim.New.z[i] = z + Sim.MaxTranslation.z * 2.0 * (random[random_i+2] - 0.5);
      Sim.New.scale[i] = scale; Sim.New.charge[i] = charge; Sim.New.scaleCoul[i] = scaleCoul; Sim.New.Type[i] = Type; Sim.New.MolID[i] = MolID;

      Sim.Old.x[i] = x; Sim.Old.y[i] = y; Sim.Old.z[i] = z;
      Sim.Old.scale[i] = scale; Sim.Old.charge[i] = charge; Sim.Old.scaleCoul[i] = scaleCoul; Sim.Old.Type[i] = Type; Sim.Old.MolID[i] = MolID;
      break;
    }
    case ROTATION://ROTATION:
    {
      Sim.New.x[i] = x - Sim.d_a[SelectedComponent].x[start_position];
      Sim.New.y[i] = y - Sim.d_a[SelectedComponent].y[start_position];
      Sim.New.z[i] = z - Sim.d_a[SelectedComponent].z[start_position];
      double XAngle = Sim.MaxRotation.x * 2.0 * (random[random_i] - 0.5);
      double YAngle = Sim.MaxRotation.y * 2.0 * (random[random_i+1] - 0.5);
      double ZAngle = Sim.MaxRotation.z * 2.0 * (random[random_i+2] - 0.5);
      RotationAroundXAxis(Sim.New, i, XAngle);
      RotationAroundYAxis(Sim.New, i, YAngle);
      RotationAroundZAxis(Sim.New, i, ZAngle);
      Sim.New.x[i] += Sim.d_a[SelectedComponent].x[start_position];
      Sim.New.y[i] += Sim.d_a[SelectedComponent].y[start_position];
      Sim.New.z[i] += Sim.d_a[SelectedComponent].z[start_position];
      Sim.New.scale[i] = scale; Sim.New.charge[i] = charge; Sim.New.scaleCoul[i] = scaleCoul; Sim.New.Type[i] = Type; Sim.New.MolID[i] = MolID;

      Sim.Old.x[i] = x; Sim.Old.y[i] = y; Sim.Old.z[i] = z;
      Sim.Old.scale[i] = scale; Sim.Old.charge[i] = charge; Sim.Old.scaleCoul[i] = scaleCoul; Sim.Old.Type[i] = Type; Sim.Old.MolID[i] = MolID;
      break;
    }
    case SINGLE_INSERTION: //TRANSLATION + ROTATION:
    { //First ROTATION//
      Sim.New.x[i] = x - Sim.d_a[SelectedComponent].x[start_position];
      Sim.New.y[i] = y - Sim.d_a[SelectedComponent].y[start_position];
      Sim.New.z[i] = z - Sim.d_a[SelectedComponent].z[start_position];
      double XAngle = Sim.MaxTranslation.x * 2.0 * (random[random_i] - 0.5);
      double YAngle = Sim.MaxTranslation.y * 2.0 * (random[random_i+1] - 0.5);
      double ZAngle = Sim.MaxTranslation.z * 2.0 * (random[random_i+2] - 0.5);
      RotationAroundXAxis(Sim.New, i, XAngle);
      RotationAroundYAxis(Sim.New, i, YAngle);
      RotationAroundZAxis(Sim.New, i, ZAngle);
      //Then TRANSLATION//
      Sim.New.x[i] += Sim.MaxRotation.x * 2.0 * (random[random_i] - 0.5);
      Sim.New.y[i] += Sim.MaxRotation.y * 2.0 * (random[random_i+1] - 0.5);
      Sim.New.z[i] += Sim.MaxRotation.z * 2.0 * (random[random_i+2] - 0.5);
      Sim.New.scale[i] = scale; Sim.New.charge[i] = charge; Sim.New.scaleCoul[i] = scaleCoul; Sim.New.Type[i] = Type; 
      Sim.New.MolID[i] = Sim.d_a[SelectedComponent].size / Sim.d_a[SelectedComponent].Molsize;
      break;
    }
    case SINGLE_DELETION: //Just Copy the old positions//
    {
      Sim.Old.x[i] = x; Sim.Old.y[i] = y; Sim.Old.z[i] = z;
      Sim.Old.scale[i] = scale; Sim.Old.charge[i] = charge; Sim.Old.scaleCoul[i] = scaleCoul; Sim.Old.Type[i] = Type; Sim.Old.MolID[i] = MolID;
    }
  }
  Sim.device_flag[i] = false;
}
////////////////////////////////////////////////////////
//Translation or Rotation. could add another move here//
////////////////////////////////////////////////////////
static inline MoveEnergy SingleMove(Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent, int MoveType)
{
  //Get Number of Molecules for this component (For updating TMMC)//
  double NMol = SystemComponents.NumberOfMolecule_for_Component[SelectedComponent];
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]) NMol--;

  bool Do_New = false;
  bool Do_Old = false;

  if(MoveType == TRANSLATION) //Translation//
  {
    SystemComponents.Moves[SelectedComponent].TranslationTotal++;
    Do_New = true; Do_Old = true;
  }
  else if(MoveType == ROTATION) //Rotation//
  {
    SystemComponents.Moves[SelectedComponent].RotationTotal++;
    Do_New = true; Do_Old = true;
  }
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
    //printf("New method tot: %.5f (HG), %.5f (GG)\n", hg_tot, gg_tot);
    // Calculate Ewald //
    if(!FF.noCharges) 
    {
      double2 newScale  = SystemComponents.Lambda[SelectedComponent].SET_SCALE(1.0);
      tot.EwaldE = GPU_EwaldDifference_General(Sims.Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 0, 0, newScale);
    }
    tot.HGEwaldE=SystemComponents.tempdeltaHGEwald;
    if(SystemComponents.UseDNNforHostGuest)
    {
      //Calculate DNN//
      double DNN_New = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, NEW);
      double DNN_Old = Predict_From_FeatureMatrix_Move(Sims, SystemComponents, OLD);
      tot.DNN_E = DNN_New - DNN_Old;
      double correction = tot.DNN_Correction(); //If use DNN, HGVDWReal and HGEwaldE are zeroed//
      if(fabs(correction) > SystemComponents.DNNDrift) //If there is a huge drift in the energy correction between DNN and Classical HostGuest//
      {
        //printf("TRANSLATION/ROTATION: Bad Prediction, reject the move!!!\n"); 
        SystemComponents.TranslationRotationDNNReject ++;
        //WriteOutliers(SystemComponents, Sims, NEW, tot, correction); //Print New Locations//
        //WriteOutliers(SystemComponents, Sims, OLD, tot, correction); //Print Old Locations//
        tot.zero(); 
        return tot;
      }
      SystemComponents.SingleMoveDNNDrift += fabs(correction);
    }
  }
  else
  {
   SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
   tot.zero();
   return tot;
  }
  if (get_random_from_zero_to_one() < std::exp(-SystemComponents.Beta * tot.total()))
  {
    update_translation_position<<<1,chainsize>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
    if(MoveType == 0) //Translation//
    { 
      SystemComponents.Moves[SelectedComponent].TranslationAccepted ++;
    }
    else if(MoveType == 1) //Rotation//
    {
      SystemComponents.Moves[SelectedComponent].RotationAccepted ++;
    }
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Sims.Box, false, SystemComponents);
    }

    SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
    return tot;
    
  }
  SystemComponents.Tmmc[SelectedComponent].Update(1.0, NMol, MoveType);
  tot.zero();
  return tot;
}

//Determine whether to accept or reject the insertion
double GetPrefactor(Components& SystemComponents, Simulations& Sims, size_t SelectedComponent, int MoveType)
{
  double MolFraction = SystemComponents.MolFraction[SelectedComponent];
  double FugacityCoefficient = SystemComponents.FugacityCoeff[SelectedComponent];
  double NumberOfMolecules = static_cast<double>(SystemComponents.NumberOfMolecule_for_Component[SelectedComponent]);

  //If component has fractional molecule, subtract the number of molecules by 1.//
  if(SystemComponents.hasfractionalMolecule[SelectedComponent]){NumberOfMolecules-=1.0;}
  if(NumberOfMolecules < 0.0) NumberOfMolecules = 0.0;

  double preFactor = 0.0;

  switch(MoveType)
  {
    case INSERTION: case SINGLE_INSERTION: 
    {  
      preFactor = SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume / (1.0+NumberOfMolecules); 
      break;
    }
    case DELETION: case SINGLE_DELETION:
    {
      preFactor = (NumberOfMolecules) / (SystemComponents.Beta * MolFraction * Sims.Box.Pressure * FugacityCoefficient * Sims.Box.Volume); 
      break;
    }
  }
  return preFactor;
}
