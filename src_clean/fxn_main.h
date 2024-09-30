void Copy_Atom_data_to_device(size_t NumberOfComponents, Atoms* device_System, Atoms* System);
void Update_Components_for_framework(Components& SystemComponents);

void Setup_Temporary_Atoms_Structure(Atoms& TempMol, Atoms* System);  

void Setup_Box_Temperature_Pressure(Units& Constants, Components& SystemComponents, Boxsize& device_Box);

void Prepare_ForceField(ForceField& FF, ForceField& device_FF, PseudoAtomDefinitions PseudoAtom);

void Prepare_Widom(WidomStruct& Widom, Boxsize Box, Simulations& Sims, Components SystemComponents, Atoms* System, Move_Statistics MoveStats);

template<typename T>
T* CUDA_allocate_array(size_t N, T InitVal)
{
  T* device_x;
  cudaMalloc(&device_x, N * sizeof(T)); checkCUDAError("Error allocating Malloc");
  T array[N];
  for(size_t i = 0; i < N; i++) array[i] = InitVal;
  cudaMemcpy(device_x, array, N * sizeof(T), cudaMemcpyHostToDevice);
  //cudaMemset(device_x, (T) InitVal, N * sizeof(T));
  return device_x;
}

template<typename T>
T* CUDA_copy_allocate_array(T* x, size_t N)
{
  T* device_x;
  cudaMalloc(&device_x, N * sizeof(T)); checkCUDAError("Error allocating Malloc");
  cudaMemcpy(device_x, x, N * sizeof(T), cudaMemcpyHostToDevice); checkCUDAError("double Error Memcpy");
  return device_x;
}

void Prepare_TempSystem_On_Host(Atoms& TempSystem)
{
    size_t Allocate_size = 1024;
    TempSystem.pos           =(double3*) malloc(Allocate_size*sizeof(double3));
    TempSystem.scale         = (double*) malloc(Allocate_size*sizeof(double));
    TempSystem.charge        = (double*) malloc(Allocate_size*sizeof(double));
    TempSystem.scaleCoul     = (double*) malloc(Allocate_size*sizeof(double));
    TempSystem.Type          = (size_t*) malloc(Allocate_size*sizeof(size_t));
    TempSystem.MolID         = (size_t*) malloc(Allocate_size*sizeof(size_t));
    TempSystem.size          = 0;
    TempSystem.Allocate_size = Allocate_size;
}

inline void Copy_Atom_data_to_device(size_t NumberOfComponents, Atoms* device_System, Atoms* System)
{
  size_t required_size = 0;
  for(size_t i = 0; i < NumberOfComponents; i++)
  {
    if(i == 0){ required_size = System[i].size;} else { required_size = System[i].Allocate_size; }
    device_System[i].pos           = CUDA_copy_allocate_array(System[i].pos,       required_size);
    device_System[i].scale         = CUDA_copy_allocate_array(System[i].scale,     required_size);
    device_System[i].charge        = CUDA_copy_allocate_array(System[i].charge,    required_size);
    device_System[i].scaleCoul     = CUDA_copy_allocate_array(System[i].scaleCoul, required_size);
    device_System[i].Type          = CUDA_copy_allocate_array(System[i].Type,      required_size);
    device_System[i].MolID         = CUDA_copy_allocate_array(System[i].MolID,     required_size);
    device_System[i].size          = System[i].size;
    device_System[i].Molsize       = System[i].Molsize;
    device_System[i].Allocate_size = System[i].Allocate_size;
  }
}

inline void Update_Components_for_framework(Components& SystemComponents)
{
  //Fill in Non-Important values for the framework components//
  //Other values should be filled in CheckFrameworkCIF function//
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
  {
    SystemComponents.MolFraction.push_back(1.0);
    SystemComponents.IdealRosenbluthWeight.push_back(1.0);
    SystemComponents.FugacityCoeff.push_back(1.0);
    SystemComponents.Tc.push_back(0.0);        //Tc for framework is set to zero
    SystemComponents.Pc.push_back(0.0);        //Pc for framework is set to zero
    SystemComponents.Accentric.push_back(0.0); //Accentric factor for framework is set to zero
    //Zhao's note: for now, assume the framework is rigid//
    SystemComponents.rigid.push_back(true);
    SystemComponents.hasfractionalMolecule.push_back(false); //No fractional molecule for the framework//
    SystemComponents.NumberOfCreateMolecules.push_back(0); //Create zero molecules for the framework//
    LAMBDA lambda;
    lambda.newBin = 0; lambda.delta = static_cast<double>(1.0/(lambda.binsize)); lambda.WangLandauScalingFactor = 0.0; //Zhao's note: in raspa3, delta is 1/(nbin - 1)
    lambda.FractionalMoleculeID = 0;
    SystemComponents.Lambda.push_back(lambda);

    TMMC tmmc;
    SystemComponents.Tmmc.push_back(tmmc); //Just use default values for tmmc for the framework, it will do nothing//
  }
  //Add PseudoAtoms from the Framework to the total PseudoAtoms array//
  //SystemComponents.UpdatePseudoAtoms(INSERTION, 0);
}

inline void Setup_Temporary_Atoms_Structure(Atoms& TempMol, Atoms* System)
{
  //Set up MolArrays//
  size_t Allocate_size_Temporary=1024; //Assign 1024 empty slots for the temporary structures//
  //OLD//
  TempMol.pos       = CUDA_allocate_array<double3> (Allocate_size_Temporary, {0.0, 0.0, 0.0});
  TempMol.scale     = CUDA_allocate_array<double>  (Allocate_size_Temporary, 0.0);
  TempMol.charge    = CUDA_allocate_array<double>  (Allocate_size_Temporary, 0.0);
  TempMol.scaleCoul = CUDA_allocate_array<double>  (Allocate_size_Temporary, 0.0);
  TempMol.Type      = CUDA_allocate_array<size_t>  (Allocate_size_Temporary, 0.0);
  TempMol.MolID     = CUDA_allocate_array<size_t>  (Allocate_size_Temporary, 0.0);
  TempMol.size      = 0;
  TempMol.Molsize   = 0;
  TempMol.Allocate_size = Allocate_size_Temporary;
}

inline void Setup_Box_Temperature_Pressure(Units& Constants, Components& SystemComponents, Boxsize& device_Box)
{
  SystemComponents.Beta = 1.0/(Constants.BoltzmannConstant/(Constants.MassUnit*pow(Constants.LengthUnit,2)/pow(Constants.TimeUnit,2))*SystemComponents.Temperature);
  //Convert pressure from pascal
  SystemComponents.Pressure/=(Constants.MassUnit/(Constants.LengthUnit*pow(Constants.TimeUnit,2)));
  printf("------------------- SIMULATION BOX PARAMETERS -----------------\n");
  printf("Pressure:        %.5f\n", SystemComponents.Pressure);
  printf("Box Volume:      %.5f\n", device_Box.Volume);
  printf("Box Beta:        %.5f\n", SystemComponents.Beta);
  printf("Box Temperature: %.5f\n", SystemComponents.Temperature);
  printf("---------------------------------------------------------------\n");
}

void Copy_ForceField_to_GPU(Variables& Vars)
{
  // COPY DATA TO DEVICE POINTER //
  //device_FF.FFParams      = CUDA_copy_allocate_array(FF.FFParams, 5);
  Vars.device_FF.OverlapCriteria = Vars.FF.OverlapCriteria;
  Vars.device_FF.CutOffVDW       = Vars.FF.CutOffVDW;
  Vars.device_FF.CutOffCoul      = Vars.FF.CutOffCoul;
  //device_FF.Prefactor       = FF.Prefactor;
  //device_FF.Alpha           = FF.Alpha;
  size_t FF_size = Vars.FF.size;
  Vars.device_FF.epsilon         = CUDA_copy_allocate_array(Vars.FF.epsilon, FF_size*FF_size);
  Vars.device_FF.sigma           = CUDA_copy_allocate_array(Vars.FF.sigma,   FF_size*FF_size);
  Vars.device_FF.z               = CUDA_copy_allocate_array(Vars.FF.z,       FF_size*FF_size);
  Vars.device_FF.shift           = CUDA_copy_allocate_array(Vars.FF.shift,   FF_size*FF_size);
  Vars.device_FF.FFType          = CUDA_copy_allocate_array(Vars.FF.FFType,  FF_size*FF_size);
  Vars.device_FF.noCharges       = Vars.FF.noCharges;
  Vars.device_FF.size            = FF_size;
  Vars.device_FF.VDWRealBias     = Vars.FF.VDWRealBias;
}

inline void InitializeMaxTranslationRotation(Components& SystemComponents)
{
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    double3 MaxTranslation = {1.0, 1.0, 1.0};
    double3 MaxRotation    = {30.0/(180/3.1415), 30.0/(180/3.1415), 30.0/(180/3.1415)};
    SystemComponents.MaxTranslation.push_back(MaxTranslation);
    SystemComponents.MaxRotation.push_back(MaxRotation);
    SystemComponents.MaxSpecialRotation.push_back(MaxRotation);
  }
}

inline void Prepare_Widom(WidomStruct& Widom, Boxsize Box, Simulations& Sims, Components& SystemComponents, Atoms* System)
{
  //Zhao's note: NumberWidomTrials is for first bead. NumberWidomTrialsOrientations is for the rest, here we consider single component, not mixture //

  size_t MaxTrialsize = max(Widom.NumberWidomTrials, Widom.NumberWidomTrialsOrientations*(SystemComponents.Moleculesize[1]-1));

  //Zhao's note: The previous way yields a size for blocksum that can be smaller than the number of kpoints
  //This is a problem when you need to do parallel Ewald summation for the whole system//
  //Might be good to add a flag or so//
  //size_t MaxResultsize = MaxTrialsize*(System[0].Allocate_size+System[1].Allocate_size);
  size_t MaxAllocatesize = max(System[0].Allocate_size, System[1].Allocate_size);
  size_t MaxResultsize = MaxTrialsize * SystemComponents.NComponents.x * MaxAllocatesize * 5; //For Volume move, it really needs a lot of blocks//


  printf("----------------- MEMORY ALLOCAION STATUS -----------------\n");
  //Compare Allocate sizes//
  printf("System allocate_sizes are: %zu, %zu\n", System[0].Allocate_size, System[1].Allocate_size); 
  printf("Component allocate_sizes are: %zu, %zu\n", SystemComponents.Allocate_size[0], SystemComponents.Allocate_size[1]);

  SystemComponents.flag        = (bool*)malloc(MaxTrialsize * sizeof(bool));
  cudaMallocHost(&Sims.device_flag,          MaxTrialsize * sizeof(bool));
 
  size_t vdw_real_size = (MaxResultsize/DEFAULTTHREAD + 1);
  size_t blocksum_size = vdw_real_size;
  size_t fourier_size  = SystemComponents.EikAllocateSize;
  if(fourier_size > vdw_real_size) blocksum_size = fourier_size;

  cudaMallocHost(&Sims.Blocksum, blocksum_size*sizeof(double));

  cudaMallocManaged(&Sims.ExcludeList,        10 * sizeof(int2));
  for(size_t i = 0; i < 10; i++) Sims.ExcludeList[i] = {-1, -1}; //Initialize with negative # so that nothing is ignored//
  //cudaMalloc(&Sims.Blocksum,             (MaxResultsize/DEFAULTTHREAD + 1)*sizeof(double));

  printf("Allocated Blocksum size: %zu, vdw_real size: %zu, fourier_size: %zu\n", blocksum_size, vdw_real_size, fourier_size);
 
  //cudaMalloc(&Sims.Blocksum,             (MaxResultsize/DEFAULTTHREAD + 1)*sizeof(double));
  Sims.Nblocks = blocksum_size;

  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    double3 MaxTranslation = {Box.Cell[0]*0.1, Box.Cell[4]*0.1, Box.Cell[8]*0.1};
    //double3 MaxTranslation = {1.0, 1.0, 1.0};
    double3 MaxRotation    = {30.0/(180/3.1415), 30.0/(180/3.1415), 30.0/(180/3.1415)};
    SystemComponents.MaxTranslation[i]    =MaxTranslation;
    SystemComponents.MaxRotation[i]       =MaxRotation;
    SystemComponents.MaxSpecialRotation[i]=MaxRotation;
  }
  Sims.start_position = 0;

  Widom.WidomFirstBeadAllocatesize = MaxResultsize/DEFAULTTHREAD;
  printf("------------------------------------------------------------\n");
}

inline void Allocate_Copy_Ewald_Vector(Boxsize& device_Box, Components& SystemComponents)
{
  printf("******   Allocating Ewald WaveVectors (INITIAL STAGE ONLY)   ******\n");
  //Zhao's note: This only works if the box size is not changed, eik_xy might not be useful if box size is not changed//
  size_t eikx_size     = SystemComponents.eik_x.size() * 2;
  size_t eiky_size     = SystemComponents.eik_y.size() * 2; //added times 2 for box volume move//
  size_t eikz_size     = SystemComponents.eik_z.size() * 2;
  printf("Allocated %zu %zu %zu space for eikxyz\n", eikx_size, eiky_size, eikz_size);
  //size_t eikxy_size    = SystemComponents.eik_xy.size();
  size_t AdsorbateEiksize = SystemComponents.AdsorbateEik.size() * SystemComponents.StructureFactor_Multiplier; //added X times for box volume move//
  SystemComponents.EikAllocateSize     = AdsorbateEiksize;
  SystemComponents.tempEikAllocateSize = AdsorbateEiksize;

  cudaMalloc(&device_Box.eik_x,     eikx_size     * sizeof(Complex));
  cudaMalloc(&device_Box.eik_y,     eiky_size     * sizeof(Complex));
  cudaMalloc(&device_Box.eik_z,     eikz_size     * sizeof(Complex));
  //cudaMalloc(&device_Box.eik_xy,    eikxy_size    * sizeof(Complex));
  cudaMalloc(&device_Box.AdsorbateEik,     AdsorbateEiksize * sizeof(Complex));
  cudaMalloc(&device_Box.tempEik,          AdsorbateEiksize * sizeof(Complex));
  cudaMalloc(&device_Box.tempFrameworkEik, AdsorbateEiksize * sizeof(Complex));
  cudaMalloc(&device_Box.FrameworkEik,     AdsorbateEiksize * sizeof(Complex));

  Complex AdsorbateEik[AdsorbateEiksize]; //Temporary Complex struct on the host//
  Complex FrameworkEik[AdsorbateEiksize];
  //for(size_t i = 0; i < SystemComponents.AdsorbateEik.size(); i++)
  for(size_t i = 0; i < AdsorbateEiksize; i++)
  {
    if(i < SystemComponents.AdsorbateEik.size())
    {
      AdsorbateEik[i].real = SystemComponents.AdsorbateEik[i].real();
      AdsorbateEik[i].imag = SystemComponents.AdsorbateEik[i].imag();
      
      FrameworkEik[i].real = SystemComponents.FrameworkEik[i].real();
      FrameworkEik[i].imag = SystemComponents.FrameworkEik[i].imag();
    }
    else
    {
      AdsorbateEik[i].real    = 0.0; AdsorbateEik[i].imag    = 0.0;
      FrameworkEik[i].real = 0.0; FrameworkEik[i].imag = 0.0;
    }
    if(i < 10) printf("Wave Vector %zu is %.5f %.5f\n", i, AdsorbateEik[i].real, AdsorbateEik[i].imag);
  }
  cudaMemcpy(device_Box.AdsorbateEik,    AdsorbateEik,    AdsorbateEiksize * sizeof(Complex), cudaMemcpyHostToDevice); checkCUDAError("error copying Complex");
  cudaMemcpy(device_Box.FrameworkEik, FrameworkEik, AdsorbateEiksize * sizeof(Complex), cudaMemcpyHostToDevice); checkCUDAError("error copying Complex");
  printf("****** DONE Allocating Ewald WaveVectors (INITIAL STAGE ONLY) ******\n");
}

inline void Check_Simulation_Energy(Boxsize& Box, Atoms* System, ForceField FF, ForceField device_FF, Components& SystemComponents, int SIMULATIONSTAGE, size_t Numsim, Simulations& Sim, bool UseGPU)
{
  std::string STAGE; 
  switch(SIMULATIONSTAGE)
  {
    case INITIAL:
    { STAGE = "INITIAL"; break;}
    case CREATEMOL:
    { STAGE = "CREATE_MOLECULE"; break;}
    case FINAL:
    { STAGE = "FINAL"; break;}
  }
  printf("======================== CALCULATING %s STAGE ENERGY ========================\n", STAGE.c_str());
  MoveEnergy ENERGY;

  Atoms device_System[SystemComponents.NComponents.x];
  cudaMemcpy(device_System, Sim.d_a, SystemComponents.NComponents.x * sizeof(Atoms), cudaMemcpyDeviceToHost);
  cudaMemcpy(Box.Cell,        Sim.Box.Cell,        9 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Box.InverseCell, Sim.Box.InverseCell, 9 * sizeof(double), cudaMemcpyDeviceToHost); 
  //Update every value that can be changed during a volume move//
  Box.Volume = Sim.Box.Volume;
  Box.ReciprocalCutOff = Sim.Box.ReciprocalCutOff;
  Box.Cubic = Sim.Box.Cubic;
  Box.kmax  = Sim.Box.kmax;

  double start = omp_get_wtime();
  VDWReal_Total_CPU(Box, System, device_System, FF, SystemComponents, ENERGY);
  double end = omp_get_wtime();
  double CPUSerialTime = end - start;
         start = omp_get_wtime();


  if(!device_FF.noCharges)
  {
    double EwStart = omp_get_wtime();
    CPU_GPU_EwaldTotalEnergy(Box, Sim.Box, System, Sim.d_a, FF, device_FF, SystemComponents, ENERGY);
    ENERGY.GGEwaldE -= SystemComponents.FrameworkEwald;
    double EwEnd  = omp_get_wtime();
    printf("Ewald Summation (total energy) on the CPU took %.5f secs\n", EwEnd - EwStart);
    //Zhao's note: if it is in the initial stage, calculate the intra and self exclusion energy for ewald summation//
    if(SIMULATIONSTAGE == INITIAL) Calculate_Exclusion_Energy_Rigid(Box, System, FF, SystemComponents);

    cudaDeviceSynchronize();
    //Zhao's note: if doing initial energy, initialize and copy host Ewald to device// 
    if(SIMULATIONSTAGE == INITIAL) Allocate_Copy_Ewald_Vector(Sim.Box, SystemComponents);
    Check_WaveVector_CPUGPU(Sim.Box, SystemComponents); //Check WaveVector on the CPU and GPU//
    cudaDeviceSynchronize();
  }
  //Calculate Tail Correction Energy//
  //This is only on CPU, not GPU//
  ENERGY.TailE     = TotalTailCorrection(SystemComponents, FF.size, Sim.Box.Volume);

  //This energy uses GPU, but lets copy it as well, no need to compute 2times//
  if(SystemComponents.UseDNNforHostGuest) ENERGY.DNN_E     = DNN_Prediction_Total(SystemComponents, Sim);
  if(SystemComponents.UseDNNforHostGuest) double Correction = ENERGY.DNN_Correction();
 
  if(SIMULATIONSTAGE == INITIAL) SystemComponents.Initial_Energy = ENERGY;
  else if(SIMULATIONSTAGE == CREATEMOL)
  {
    SystemComponents.CreateMol_Energy = ENERGY;
  }
  else
  { 
    SystemComponents.Final_Energy = ENERGY;
  }

  if(UseGPU)
  {
    MoveEnergy GPU_Energy;
    bool   UseOffset = false;
    double start = omp_get_wtime();
    GPU_Energy += Total_VDW_Coulomb_Energy(Sim, SystemComponents, device_FF, UseOffset);
    cudaDeviceSynchronize();
    double end = omp_get_wtime();
    printf("VDW + Real on the GPU took %.5f secs\n", end - start);

    /*
    //SINGLE-THREAD GPU VDW + Real, use just for debugging!!!//
    double* xxx; xxx = (double*) malloc(sizeof(double)*2);
    double* device_xxx; device_xxx = CUDA_copy_allocate_array(xxx, 2);
    //Zhao's note: if the serial GPU energy test is too slow, comment it out//
    //one_thread_GPU_test<<<1,1>>>(Sim.Box, Sim.d_a, device_FF, device_xxx);
    cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
           end = omp_get_wtime();
    cudaDeviceSynchronize();
    */

    if(!device_FF.noCharges)
    {
      start = omp_get_wtime();
      GPU_Energy  += Ewald_TotalEnergy(Sim, SystemComponents, UseOffset);
      end = omp_get_wtime();
      double GPUEwaldTime = end - start;
      printf("Ewald Summation (total energy) on the GPU took %.5f secs\n", GPUEwaldTime);
    }
    GPU_Energy.TailE = TotalTailCorrection(SystemComponents, FF.size, Sim.Box.Volume);
    if(SystemComponents.UseDNNforHostGuest) GPU_Energy.DNN_E = ENERGY.DNN_E;
    if(SystemComponents.UseDNNforHostGuest) double GPU_Correction = GPU_Energy.DNN_Correction();

    printf("Total GPU Energy: \n"); GPU_Energy.print();
    if(SIMULATIONSTAGE == FINAL) SystemComponents.GPU_Energy = GPU_Energy;
  }

  printf("====================== DONE CALCULATING %s STAGE ENERGY ======================\n", STAGE.c_str());
}

MoveEnergy check_energy_wrapper(Variables& Var, int SimulationIndex)
{
  // CALCULATE THE FINAL ENERGY (VDW + Real) //
  int PHASE = FINAL;

  //for(size_t i = 0; i < NumberOfSimulations; i++)
  //{
    size_t i = static_cast<size_t>(SimulationIndex);
    printf("======================================\n");
    printf("CHECKING FINAL ENERGY FOR SYSTEM [%zu]\n", i);
    printf("======================================\n");
    bool UseGPU = true;
    Check_Simulation_Energy(Var.Box[i], Var.SystemComponents[i].HostSystem, Var.FF, Var.device_FF, Var.SystemComponents[i], PHASE, i, Var.Sims[i], UseGPU);
    printf("======================================\n");
  //}
  return Var.SystemComponents[i].GPU_Energy;
}

inline void Copy_AtomData_from_Device(Atoms* System, Atoms* Host_System, Atoms* d_a, Components& SystemComponents)
{
  cudaMemcpy(System, d_a, SystemComponents.NComponents.x * sizeof(Atoms), cudaMemcpyDeviceToHost);
  //printval<<<1,1>>>(System[0]);
  //printvald_a<<<1,1>>>(d_a);
  for(size_t ijk=0; ijk < SystemComponents.NComponents.x; ijk++)
  {
    // if the host allocate_size is different from the device, allocate more space on the host
    size_t current_allocated_size = System[ijk].Allocate_size;
    if(current_allocated_size != Host_System[ijk].Allocate_size) //Need to update host
    {
      Host_System[ijk].pos       = (double3*) malloc(System[ijk].Allocate_size*sizeof(double3));
      Host_System[ijk].scale     = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].charge    = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].scaleCoul = (double*)  malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].Type      = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].MolID     = (size_t*)  malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].Allocate_size = System[ijk].Allocate_size;
    }
    Host_System[ijk].size      = System[ijk].size; //Zhao's note: no matter what, the size (not allocated size) needs to be updated

    cudaMemcpy(Host_System[ijk].pos, System[ijk].pos, sizeof(double3)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Host_System[ijk].scale, System[ijk].scale, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Host_System[ijk].charge, System[ijk].charge, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Host_System[ijk].scaleCoul, System[ijk].scaleCoul, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Host_System[ijk].Type, System[ijk].Type, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Host_System[ijk].MolID, System[ijk].MolID, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
    Host_System[ijk].size = System[ijk].size;
  }
}

inline void PRINT_ENERGY_AT_STAGE(Components& SystemComponents, int stage, Units& Constants)
{
  std::string stage_name;
  MoveEnergy  E;
  switch(stage)
  {
    case INITIAL:               {stage_name = "INITIAL STAGE";         E = SystemComponents.Initial_Energy;   break;}
    case CREATEMOL:             {stage_name = "CREATE MOLECULE STAGE"; E = SystemComponents.CreateMol_Energy; break;}
    case FINAL:                 {stage_name = "FINAL STAGE";           E = SystemComponents.Final_Energy;     break;}
    case CREATEMOL_DELTA:       {stage_name = "RUNNING DELTA_E (CREATE MOLECULE - INITIAL)"; E = SystemComponents.CreateMoldeltaE; break;}
    case DELTA:                 {stage_name = "RUNNING DELTA_E (FINAL - CREATE MOLECULE)";   E = SystemComponents.deltaE; break;}
    case CREATEMOL_DELTA_CHECK: {stage_name = "CHECK DELTA_E (CREATE MOLECULE - INITIAL)"; E = SystemComponents.CreateMol_Energy - SystemComponents.Initial_Energy; break;}
    case DELTA_CHECK: {stage_name = "CHECK DELTA_E (RUNNING FINAL - CREATE MOLECULE)"; E = SystemComponents.Final_Energy - SystemComponents.CreateMol_Energy; break;}
    case DRIFT: {stage_name = "ENERGY DRIFT (CPU FINAL - RUNNING FINAL)"; E = SystemComponents.CreateMol_Energy + SystemComponents.deltaE - SystemComponents.Final_Energy; break;}
    case GPU_DRIFT: {stage_name = "GPU DRIFT (GPU FINAL - CPU FINAL)"; E = SystemComponents.Final_Energy - SystemComponents.GPU_Energy; break;}
    case AVERAGE: {stage_name = "PRODUCTION PHASE AVERAGE ENERGY"; E = SystemComponents.AverageEnergy;break;}
    case AVERAGE_ERR: {stage_name = "PRODUCTION PHASE AVERAGE ENERGY ERRORBAR"; E = SystemComponents.AverageEnergy_Errorbar; break;}
  }
  printf(" *** %s *** \n", stage_name.c_str());
  printf("========================================================================\n");
  printf("VDW [Host-Host]:            %.5f (%.5f [K])\n", E.HHVDW, E.HHVDW * Constants.energy_to_kelvin);
  printf("VDW [Host-Guest]:           %.5f (%.5f [K])\n", E.HGVDW, E.HGVDW * Constants.energy_to_kelvin);
  printf("VDW [Guest-Guest]:          %.5f (%.5f [K])\n", E.GGVDW, E.GGVDW * Constants.energy_to_kelvin);
  printf("Real Coulomb [Host-Host]:   %.5f (%.5f [K])\n", E.HHReal, E.HHReal * Constants.energy_to_kelvin);
  printf("Real Coulomb [Host-Guest]:  %.5f (%.5f [K])\n", E.HGReal, E.HGReal * Constants.energy_to_kelvin);
  printf("Real Coulomb [Guest-Guest]: %.5f (%.5f [K])\n", E.GGReal, E.GGReal * Constants.energy_to_kelvin);
  printf("Ewald [Host-Host]:          %.5f (%.5f [K])\n", E.HHEwaldE, E.HHEwaldE * Constants.energy_to_kelvin);
  printf("Ewald [Host-Guest]:         %.5f (%.5f [K])\n", E.HGEwaldE, E.HGEwaldE * Constants.energy_to_kelvin);
  printf("Ewald [Guest-Guest]:        %.5f (%.5f [K])\n", E.GGEwaldE, E.GGEwaldE * Constants.energy_to_kelvin);
  printf("DNN Energy:                 %.5f (%.5f [K])\n", E.DNN_E, E.DNN_E * Constants.energy_to_kelvin);
  if(SystemComponents.UseDNNforHostGuest)
  {
    printf(" --> Stored Classical Host-Guest Interactions: \n");
    printf("     VDW:             %.5f (%.5f [K])\n", E.storedHGVDW, E.storedHGVDW * Constants.energy_to_kelvin);
    printf("     Real Coulomb:    %.5f (%.5f [K])\n", E.storedHGReal, E.storedHGReal * Constants.energy_to_kelvin);
    printf("     Ewald:           %.5f (%.5f [K])\n", E.storedHGEwaldE, E.storedHGEwaldE * Constants.energy_to_kelvin);
    printf("     Total:           %.5f (%.5f [K])\n", E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE, (E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE) * Constants.energy_to_kelvin);
    printf(" --> DNN - Classical: %.5f (%.5f [K])\n", E.DNN_E - (E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE), (E.DNN_E - (E.storedHGVDW + E.storedHGReal + E.storedHGEwaldE)) * Constants.energy_to_kelvin);
  }
  printf("Tail Correction Energy:     %.5f (%.5f [K])\n", E.TailE, E.TailE * Constants.energy_to_kelvin);
  printf("Total Energy:               %.5f (%.5f [K])\n", E.total(), E.total() * Constants.energy_to_kelvin);
  printf("========================================================================\n");
}
inline void ENERGY_SUMMARY(std::vector<Components>& SystemComponents, Units& Constants)
{
  size_t NumberOfSimulations = SystemComponents.size();
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    printf("======================== ENERGY SUMMARY (Simulation %zu) =========================\n", i);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], INITIAL, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], CREATEMOL, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], CREATEMOL_DELTA, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], CREATEMOL_DELTA_CHECK, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], FINAL, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], DELTA, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], DELTA_CHECK, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], DRIFT, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], GPU_DRIFT, Constants);
    printf("================================================================================\n");
    printf("======================== PRODUCTION PHASE AVERAGE ENERGIES (Simulation %zu) =========================\n", i);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], AVERAGE, Constants);
    PRINT_ENERGY_AT_STAGE(SystemComponents[i], AVERAGE_ERR, Constants);
    printf("================================================================================\n");

    printf("DNN Rejection Summary:\nTranslation+Rotation: %zu\nReinsertion: %zu\nInsertion: %zu\nDeletion: %zu\nSingleSwap: %zu\n", SystemComponents[i].TranslationRotationDNNReject, SystemComponents[i].ReinsertionDNNReject, SystemComponents[i].InsertionDNNReject, SystemComponents[i].DeletionDNNReject, SystemComponents[i].SingleSwapDNNReject);
    printf("DNN Drift Summary:\nTranslation+Rotation: %.5f\nReinsertion: %.5f\nInsertion: %.5f\nDeletion: %.5f\nSingleSwap: %.5f\n", SystemComponents[i].SingleMoveDNNDrift, SystemComponents[i].ReinsertionDNNDrift, SystemComponents[i].InsertionDNNDrift, SystemComponents[i].DeletionDNNDrift, SystemComponents[i].SingleSwapDNNDrift);
  }
}

static inline void Write_Lambda(size_t Cycle, Components SystemComponents, size_t SystemIndex)
{
  std::ofstream textrestartFile{};
  std::filesystem::path cwd = std::filesystem::current_path();

  std::string dirname="Lambda/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "Lambda_Histogram.data";

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  textrestartFile = std::ofstream(fileName, std::ios::out);
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    if(SystemComponents.hasfractionalMolecule[i])
    {
      textrestartFile << "Component " << i << ": " << SystemComponents.MoleculeName[i] << '\n';
      textrestartFile << "BIN SIZE : " << SystemComponents.Lambda[i].binsize << '\n';
      textrestartFile << "BIN WIDTH: " << SystemComponents.Lambda[i].delta << '\n';
      textrestartFile << "WL SCALING FACTOR: " << SystemComponents.Lambda[i].WangLandauScalingFactor << '\n';
      textrestartFile << "FRACTIONAL MOLECULE ID: " << SystemComponents.Lambda[i].FractionalMoleculeID << '\n';
      textrestartFile << "CURRENT BIN: " << SystemComponents.Lambda[i].currentBin << '\n';
      textrestartFile << "BINS: ";
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << j << " ";
      textrestartFile << "\nHistogram: ";
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << SystemComponents.Lambda[i].Histogram[j] << " ";
      textrestartFile << "\nBIAS FACTOR: ";
      for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
        textrestartFile << SystemComponents.Lambda[i].biasFactor[j] << " ";
    }
  }
  textrestartFile.close();
}

static inline void Write_TMMC(size_t Cycle, Components SystemComponents, size_t SystemIndex)
{
  std::ofstream textTMMCFile{};
  std::filesystem::path cwd = std::filesystem::current_path();

  std::string dirname="TMMC/System_" + std::to_string(SystemIndex) + "/";
  std::string fname  = dirname + "/" + "TMMC_Statistics.data";

  std::filesystem::path directoryName = cwd /dirname;
  std::filesystem::path fileName = cwd /fname;
  std::filesystem::create_directories(directoryName);
  textTMMCFile = std::ofstream(fileName, std::ios::out);
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    if(SystemComponents.Tmmc[i].DoTMMC)
    {
      textTMMCFile << "Component " << i << ": " << SystemComponents.MoleculeName[i] << " -> Updated " << SystemComponents.Tmmc[i].TMUpdateTimes << " times \n";
      textTMMCFile << "Min Macrostate : " << SystemComponents.Tmmc[i].MinMacrostate << '\n';
      textTMMCFile << "Max Macrostate : " << SystemComponents.Tmmc[i].MaxMacrostate << '\n';
      textTMMCFile << "Wang-Landau Factor : " << SystemComponents.Tmmc[i].WLFactor << '\n';
      textTMMCFile << "N NMol Bin CM[-1] CM[0] CM[1] WLBias ln_g TMBias lnpi Forward_lnpi Reverse_lnpi Histogram" << '\n';
      for(size_t j = 0; j < SystemComponents.Tmmc[i].Histogram.size(); j++)
      {
        size_t N   = j / SystemComponents.Tmmc[i].nbinPerMacrostate;
        size_t bin = j % SystemComponents.Tmmc[i].nbinPerMacrostate;
        textTMMCFile << j << " " << N << " " << bin << " "; 
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].CMatrix[j].x << " " ; 
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].CMatrix[j].y << " " ;
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].CMatrix[j].z << " " ;
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].WLBias[j] << " " ; 
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].ln_g[j] << " "; 
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].TMBias[j] << " " ;
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].lnpi[j] << " "; 
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].forward_lnpi[j] << " " ;
        textTMMCFile /*<< std::setprecision (15)*/ << SystemComponents.Tmmc[i].reverse_lnpi[j] << " " ; 
        textTMMCFile << SystemComponents.Tmmc[i].Histogram[j] << '\n';
      }
    }
  }
  textTMMCFile.close();
}

inline void GenerateSummaryAtEnd(int Cycle, std::vector<Components>& SystemComponents, Simulations*& Sims, ForceField& FF, std::vector<Boxsize>& Box)
{
  
  size_t NumberOfSimulations = SystemComponents.size();
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    printf("System %zu\n", i);
    Atoms device_System[SystemComponents[i].NComponents.x];
    Copy_AtomData_from_Device(device_System, SystemComponents[i].HostSystem, Sims[i].d_a, SystemComponents[i]);

    Write_Lambda(Cycle, SystemComponents[i], i);
    Write_TMMC(Cycle, SystemComponents[i], i);
    //Print Number of Pseudo Atoms//
    for(size_t j = 0; j < SystemComponents[i].NumberOfPseudoAtoms.size(); j++) printf("PseudoAtom Type: %s[%zu], #: %zu\n", SystemComponents[i].PseudoAtoms.Name[j].c_str(), j, SystemComponents[i].NumberOfPseudoAtoms[j]);
  }
}

inline void prepare_MixtureStats(Components& SystemComponents)
{
  double tot = 0.0;
  printf("================= MOL FRACTIONS =================\n");
  for(size_t j = 0; j < SystemComponents.NComponents.x; j++)
  {
    SystemComponents.Moves[j].IdentitySwap_Total_TO.resize(SystemComponents.NComponents.x, 0);
    SystemComponents.Moves[j].IdentitySwap_Acc_TO.resize(SystemComponents.NComponents.x,   0);
    if(j != 0) tot += SystemComponents.MolFraction[j];
  }
  //Prepare MolFraction for adsorbate components//
  for(size_t j = 1; j < SystemComponents.NComponents.x; j++)
  {
    SystemComponents.MolFraction[j] /= tot;
    printf("Component [%zu] (%s), Mol Fraction: %.5f\n", j, SystemComponents.MoleculeName[j].c_str(), SystemComponents.MolFraction[j]);
  }
  printf("=================================================\n");
}
