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

__global__ void Energy_difference_PARTIAL_MULTIPLE(Boxsize Box, Simulations* Sims, ForceField FF, size_t ComponentID, size_t chainsize, size_t NumberOfSimulations) // Consider to change this for polyatomic
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum, energy + dUdlambda//
  int cache_id = threadIdx.x;
  size_t ijk = blockIdx.x * blockDim.x + threadIdx.x;

  size_t ijk_within_block = ijk - blockIdx.x * blockDim.x;

  sdata[ijk_within_block] = 0.0; //sdata[ijk_within_block].y = 0.0;

  __shared__ bool Blockflag = false;

  size_t nsim = 0; size_t blockbefore = 0;
  get_nsim(Sims, &nsim, &blockbefore, NumberOfSimulations, blockIdx.x);
  size_t ijk_within_sim = ijk - blockbefore * blockDim.x;

  size_t BlockID_in_sim = blockIdx.x - blockbefore;

  if(ijk_within_sim < Sims[nsim].TotalAtoms * chainsize)
  {
    Sims[nsim].Blocksum[BlockID_in_sim] = 0.0; //BlockdUdlambda[BlockID_in_sim] = 0.0;
    // Manually fusing/collapsing the loop //
    size_t i = ijk_within_sim / chainsize;
    size_t j = ijk_within_sim % chainsize; //+ ij/totalAtoms; // position in Mol and NewMol

    const size_t NumberComp = 2; //Zhao's note: need to change here for multicomponent
    const Atoms* System = Sims[nsim].d_a;
    const Atoms  Old    = Sims[nsim].Old;
    const Atoms  New    = Sims[nsim].New;
    
    size_t posi = i;      size_t comp = 0;
    get_position(System, &posi, &comp, i, NumberComp);
    //printf("thread: %lu, block: %lu, ijk_within_block: %lu, ijk_within_sim: %lu, blockbefore: %lu, nsim: %lu, comp: %lu, posi: %lu\n", ijk, blockIdx.x, ijk_within_block, ijk_within_sim, blockbefore, nsim, comp, posi);


    const Atoms Component        = System[comp];
    const double scaleA          = Component.scale[posi];
    const double chargeA         = Component.charge[posi];
    const double scalingCoulombA = Component.scaleCoul[posi];
    const size_t typeA           = Component.Type[posi];
    const size_t MoleculeID      = Component.MolID[posi];
    double tempy = 0.0;          double tempdU = 0.0;
    if(!((MoleculeID == New.MolID[0]) &&(comp == ComponentID))) //ComponentID: Component ID for the molecule being translated
    {
      ///////////
      //  NEW  //
      ///////////
      double posvec[3] = {Component.x[posi] - New.x[j], Component.y[posi] - New.y[j], Component.z[posi] - New.z[j]};
      //printf("posvec: %.10f, %.10f, %.10f; Box[0]: %.10f\n", posvec[0], posvec[1], posvec[2], Box.Cell[0]/*, Box.InverseCell[0], Box.Cubic ? "true":"false"*/);
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);

      double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      double result[2] = {0.0, 0.0};
      if(rr_dot < FF.CutOffVDW)
      {
        const size_t typeB    = New.Type[j];
        const double scaleB   = New.scale[j];
        const double scaling  = scaleA * scaleB;
        const size_t row      = typeA*FF.size+typeB;
        const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
        VDW(FFarg, rr_dot, scaling, result);
        tempy                += result[0];
        tempdU               += result[1];
      }

      if (!FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB         = New.charge[j];
        const double scalingCoulombB = New.scaleCoul[j];
        const double r               = sqrt(rr_dot);
        const double scalingCoul     = scalingCoulombA * scalingCoulombB;
        CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result);
        tempy                       += result[0]; //prefactor merged in the CoulombReal function
      }
      ///////////
      //  OLD  //
      ///////////
      posvec[0] = Component.x[posi] - Old.x[j]; posvec[1] = Component.y[posi] - Old.y[j]; posvec[2] = Component.z[posi] - Old.z[j];
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      if(rr_dot < FF.CutOffVDW)
      {
        const size_t typeB    = Old.Type[j];
        const double scaleB   = Old.scale[j];
        const double scaling  = scaleA * scaleB;
        const size_t row      = typeA*FF.size+typeB;
        const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
        VDW(FFarg, rr_dot, scaling, result);
        tempy                -= result[0];
        tempdU               -= result[1];
      }
      if (!FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB         = Old.charge[j];
        const double scalingCoulombB = Old.scaleCoul[j];
        const double r               = sqrt(rr_dot);
        const double scalingCoul     = scalingCoulombA * scalingCoulombB;
        CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result);
        tempy                       -= result[0]; //prefactor merged in the CoulombReal function
      }

    }
    sdata[ijk_within_block] = tempy; //sdata[ijk_within_block].y = tempdU;

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
      Sims[nsim].Blocksum[BlockID_in_sim] = sdata[0]; //BlockdUdlambda[blockIdx.x] = sdata[0].y;
    }
  }

}

__global__ void Energy_difference_PARTIAL(Boxsize Box, Atoms* System, Atoms Mol, Atoms NewMol, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, size_t Threadsize) // Consider to change this for polyatomic
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum, energy + dUdlambda//
  int cache_id = threadIdx.x;
  size_t ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t ij_within_block = ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; //sdata[ij_within_block].y = 0.0;
  //Initialize Blocky and BlockdUdlambda//
  BlockEnergy[blockIdx.x] = 0.0; //BlockdUdlambda[blockIdx.x] = 0.0;

  __shared__ bool Blockflag = false;

  if(ij < totalAtoms * chainsize)
  {
  BlockEnergy[blockIdx.x] = 0.0; //BlockdUdlambda[blockIdx.x] = 0.0;
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; //+ ij/totalAtoms; // position in Mol and NewMol

  size_t comp = 0;
  const size_t NumberComp = 2; //Zhao's note: need to change here for multicomponent
  size_t posi = i; size_t totalsize= 0;
  for(size_t ijk = 0; ijk < NumberComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= totalsize)
    {
      comp++;
      posi -= System[ijk].size;
    }
  }
  //printf("thread: %lu, comp: %lu, posi: %lu\n", i,comp, posi);

  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double tempy = 0.0; double tempdU = 0.0;
  if(!((MoleculeID == NewMol.MolID[0]) &&(comp == ComponentID))) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    double posvec[3] = {Component.x[posi] - NewMol.x[j], Component.y[posi] - NewMol.y[j], Component.z[posi] - NewMol.z[j]};

    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = NewMol.Type[j];
      const double scaleB = NewMol.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      tempy += result[0];
      tempdU += result[1];
    }

    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = NewMol.charge[j];
      const double scalingCoulombB = NewMol.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result);
      tempy += result[0]; //prefactor merged in the CoulombReal function
    }
    ///////////
    //  OLD  //
    ///////////
    posvec[0] = Component.x[posi] - Mol.x[j]; posvec[1] = Component.y[posi] - Mol.y[j]; posvec[2] = Component.z[posi] - Mol.z[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Mol.Type[j];
      const double scaleB = Mol.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      tempy -= result[0];
      tempdU -= result[1];
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Mol.charge[j];
      const double scalingCoulombB = Mol.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result);
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
}

__global__ void Energy_difference_PARTIAL_FLAG(Boxsize Box, Atoms* System, Atoms Mol, Atoms NewMol, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, size_t Threadsize, bool* flag) // Consider to change this for polyatomic
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum, energy + dUdlambda//
  int cache_id = threadIdx.x;
  size_t ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t ij_within_block = ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; //sdata[ij_within_block].y = 0.0;
  //Initialize Blocky and BlockdUdlambda//
  BlockEnergy[blockIdx.x] = 0.0; //BlockdUdlambda[blockIdx.x] = 0.0;

  __shared__ bool Blockflag = false;

  if(ij < totalAtoms * chainsize)
  {
  BlockEnergy[blockIdx.x] = 0.0; //BlockdUdlambda[blockIdx.x] = 0.0;
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; //+ ij/totalAtoms; // position in Mol and NewMol

  size_t comp = 0;
  const size_t NumberComp = 2; //Zhao's note: need to change here for multicomponent
  size_t posi = i; size_t totalsize= 0;
  for(size_t ijk = 0; ijk < NumberComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= totalsize)
    {
      comp++;
      posi -= System[ijk].size;
    }
  }
  //printf("thread: %lu, comp: %lu, posi: %lu\n", i,comp, posi);

  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double tempy = 0.0; double tempdU = 0.0;
  if(!((MoleculeID == NewMol.MolID[0]) &&(comp == ComponentID))) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    double posvec[3] = {Component.x[posi] - NewMol.x[j], Component.y[posi] - NewMol.y[j], Component.z[posi] - NewMol.z[j]};

    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    double result[2] = {0.0, 0.0};
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = NewMol.Type[j];
      const double scaleB = NewMol.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      tempy += result[0];
      tempdU += result[1];
    }

    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = NewMol.charge[j];
      const double scalingCoulombB = NewMol.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result);
      tempy += result[0]; //prefactor merged in the CoulombReal function
    }
    ///////////
    //  OLD  //
    ///////////
    posvec[0] = Component.x[posi] - Mol.x[j]; posvec[1] = Component.y[posi] - Mol.y[j]; posvec[2] = Component.z[posi] - Mol.z[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Mol.Type[j];
      const double scaleB = Mol.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result);
      if(result[0] > FF.OverlapCriteria){ Blockflag = true; }
      tempy -= result[0];
      tempdU -= result[1];
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Mol.charge[j];
      const double scalingCoulombB = Mol.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result);
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
  else
  {
    flag[0]=true;
  }
}

//Translation Move//
double Translation_Move(Boxsize& Box, Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent);

void Update_Max_Translation(Move_Statistics MoveStats, Simulations& Sims);

void Setup_threadblock(size_t arraysize, size_t *Nblock, size_t *Nthread);

//Rotation Move//
double Rotation_Move(Boxsize& Box, Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent);

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
  MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal=0;
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

__global__ void get_new_translation_position(Atoms* d_a, Atoms Mol, Atoms NewMol, Simulations Sims, ForceField FF, size_t start_position, size_t SelectedComponent, double* random, size_t offset, double3 Max, bool* device_flag)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t random_i = offset; // all atoms in the molecule are moving in the same direction, so no i in this variable
  size_t real_pos = start_position + i;

  const double x         = d_a[SelectedComponent].x[real_pos];
  const double y         = d_a[SelectedComponent].y[real_pos];
  const double z         = d_a[SelectedComponent].z[real_pos];
  const double scale     = d_a[SelectedComponent].scale[real_pos];
  const double charge    = d_a[SelectedComponent].charge[real_pos];
  const double scaleCoul = d_a[SelectedComponent].scaleCoul[real_pos];
  const size_t Type      = d_a[SelectedComponent].Type[real_pos];
  const size_t MolID     = d_a[SelectedComponent].MolID[real_pos];

  //Atoms Temp = Sims.Old; Atoms TempNEW = Sims.New;

  Mol.x[i] = x;
  Mol.y[i] = y;
  Mol.z[i] = z;
  NewMol.x[i] = Mol.x[i] + Max.x * 2.0 * (random[random_i] - 0.5);
  NewMol.y[i] = Mol.y[i] + Max.y * 2.0 * (random[random_i+1] - 0.5);
  NewMol.z[i] = Mol.z[i] + Max.z * 2.0 * (random[random_i+2] - 0.5);
  Mol.scale[i] = scale; Mol.charge[i] = charge; Mol.scaleCoul[i] = scaleCoul; Mol.Type[i] = Type; Mol.MolID[i] = MolID;
  NewMol.scale[i] = scale; NewMol.charge[i] = charge; NewMol.scaleCoul[i] = scaleCoul; NewMol.Type[i] = Type; NewMol.MolID[i] = MolID;
  device_flag[i] = false;
}

__global__ void update_translation_position(Atoms* d_a, Atoms NewMol, size_t start_position, size_t SelectedComponent)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  d_a[SelectedComponent].x[start_position+i] = NewMol.x[i];
  d_a[SelectedComponent].y[start_position+i] = NewMol.y[i];
  d_a[SelectedComponent].z[start_position+i] = NewMol.z[i];
}

static inline double Translation_Move(Boxsize& Box, Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  bool FLAG = false;
  double tot = 0.0;
  //double result = 0.0;
  SystemComponents.Moves[SelectedComponent].TranslationTotal++;
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  size_t chainsize;
  //Set up Old position and New position arrays
  chainsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  if(chainsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  if((Random.offset + 3*chainsize) >= Random.randomsize) Random.offset = 0;
  get_new_translation_position<<<1, chainsize>>>(Sims.d_a, Sims.Old, Sims.New, Sims, FF, start_position, SelectedComponent, Random.device_random, Random.offset, Sims.MaxTranslation, Sims.device_flag); Random.offset += 3*chainsize;

  // Setup for the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(Atomsize * chainsize, &Nblock, &Nthread); //Zhao's note: add back dUdlambda later //
  Energy_difference_PARTIAL_FLAG<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, chainsize, Nthread, Sims.device_flag);
  //Energy_difference_PARTIAL<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, chainsize, Nthread);
  cudaMemcpy(Sims.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);
  double EwaldE = 0.0; 
  if(!FLAG)
  {
    double BlockResult[Nblock];
    cudaMemcpy(BlockResult, Sims.Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < Nblock; i++) tot += BlockResult[i];
    // Calculate Ewald //
    if(!FF.noCharges) 
    {
      EwaldE = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 0, 0);
      tot   += EwaldE;
      //printf("EwaldE (GPU) is %.5f\n", EwaldE);
    }
  }
  else
   return 0.0;
  if (get_random_from_zero_to_one() < std::exp(-SystemComponents.Beta * tot))
  {
    update_translation_position<<<1,chainsize>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
    SystemComponents.Moves[SelectedComponent].TranslationAccepted ++;
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Box, false, SystemComponents);
    }
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

__global__ void get_new_rotation_position(Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, size_t start_position, size_t SelectedComponent, double* random, size_t offset, double3 MaxRotation)
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
  double XAngle = MaxRotation.x * 2.0 * (random[random_i] - 0.5);
  double YAngle = MaxRotation.y * 2.0 * (random[random_i+1] - 0.5);
  double ZAngle = MaxRotation.z * 2.0 * (random[random_i+2] - 0.5);
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

static inline double Rotation_Move(Boxsize& Box, Components& SystemComponents, Simulations& Sims, WidomStruct& Widom, ForceField& FF, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent)
{
  double tot = 0.0;
  SystemComponents.Moves[SelectedComponent].RotationTotal++;
  size_t Atomsize = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
  {
    Atomsize += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  }
  size_t chainsize;
  //Set up Old position and New position arrays
  chainsize = SystemComponents.Moleculesize[SelectedComponent]; //Get the size of the selected Molecule
  if(chainsize >= 1024)
  {
    throw std::runtime_error("Molecule size is greater than allocated size, Why so big?\n");
  }
  size_t start_position = SelectedMolInComponent*SystemComponents.Moleculesize[SelectedComponent];
  //Determine the axis of rotation//
  if((Random.offset + 3*chainsize) >= Random.randomsize) Random.offset = 0;
  get_new_rotation_position<<<1,chainsize>>>(Sims.d_a, Sims.Old, Sims.New, FF, start_position, SelectedComponent, Random.device_random, Random.offset, Sims.MaxRotation);Random.offset += 3*chainsize;
  
  // Setup for the pairwise calculation //
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock(Atomsize * chainsize, &Nblock, &Nthread);
  //double* BlockdU; Zhao's note: add back dUdlambda later //
  Energy_difference_PARTIAL<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Sims.d_a, Sims.Old, Sims.New, FF, Sims.Blocksum, SelectedComponent, Atomsize, chainsize, Nthread);
  double BlockResult[Nblock];
  cudaMemcpy(BlockResult, Sims.Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++) tot += BlockResult[i];
  // Calculate Ewald //
  if(!FF.noCharges)
  {
    double EwaldE = GPU_EwaldDifference_General(Box, Sims.d_a, Sims.New, Sims.Old, FF, Sims.Blocksum, SystemComponents, SelectedComponent, 0, 0);
    tot   += EwaldE;
    //printf("EwaldE (GPU) is %.5f\n", EwaldE);
  }
  if (get_random_from_zero_to_one() < std::exp(-SystemComponents.Beta * tot))
  {
    update_translation_position<<<1,chainsize>>>(Sims.d_a, Sims.New, start_position, SelectedComponent);
    SystemComponents.Moves[SelectedComponent].RotationAccepted ++;
    if(!FF.noCharges)
    {
      Update_Ewald_Vector(Box, false, SystemComponents);
    }
    return tot;
  }
  return 0.0;
}
