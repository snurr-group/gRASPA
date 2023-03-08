void Setup_RandomNumber(RandomNumber& Random, size_t SIZE);
void Copy_Atom_data_to_device(size_t NumberOfComponents, Atoms* device_System, Atoms* System);
void Update_Components_for_framework(size_t NumberOfComponents, Components& SystemComponents, Atoms* System);
void Setup_Temporary_Atoms_Structure(Atoms& device_Mol, Atoms* System);
void Initialize_Move_Statistics(Move_Statistics& MoveStats);

void Setup_System_Units_and_Box(Units& Constants, Components& SystemComponents, Boxsize& Box, Boxsize& device_Box);

void Prepare_ForceField(ForceField& FF, ForceField& device_FF, PseudoAtomDefinitions PseudoAtom, Boxsize Box);

void Prepare_Widom(WidomStruct& Widom, Components SystemComponents, Atoms* System, Move_Statistics MoveStats);

double Check_Simulation_Energy(Boxsize Box, Boxsize device_Box, Atoms* System, Atoms* device_System, Atoms* d_a, ForceField FF, ForceField device_FF, Components SystemComponents, bool initial);

inline void Setup_RandomNumber(RandomNumber& Random, size_t SIZE)
{
  Random.randomsize = SIZE; Random.offset = 0;
  std::vector<double> array_random(Random.randomsize);
  for (size_t i = 0; i < Random.randomsize; i++)
  {
    array_random[i] = get_random_from_zero_to_one();
  }
  Random.host_random = Doubleconvert1DVectortoArray(array_random);
  Random.device_random = CUDA_copy_allocate_double_array(Random.host_random, Random.randomsize);
}

inline void Copy_Atom_data_to_device(size_t NumberOfComponents, Atoms* device_System, Atoms* System)
{
  size_t required_size = 0;
  for(size_t i = 0; i < NumberOfComponents; i++)
  {
    if(i == 0){ required_size = System[i].size;} else { required_size = System[i].Allocate_size; }
    device_System[i].x             = CUDA_copy_allocate_array(System[i].x,         required_size);
    device_System[i].y             = CUDA_copy_allocate_array(System[i].y,         required_size);
    device_System[i].z             = CUDA_copy_allocate_array(System[i].z,         required_size);
    device_System[i].scale         = CUDA_copy_allocate_array(System[i].scale,     required_size);
    device_System[i].charge        = CUDA_copy_allocate_array(System[i].charge,    required_size);
    device_System[i].scaleCoul     = CUDA_copy_allocate_array(System[i].scaleCoul, required_size);
    device_System[i].Type          = CUDA_copy_allocate_array(System[i].Type,      required_size);
    device_System[i].MolID         = CUDA_copy_allocate_array(System[i].MolID,     required_size);
    device_System[i].size          = System[i].size;
    device_System[i].Molsize       = System[i].Molsize;
    device_System[i].Allocate_size = System[i].Allocate_size;
    printf("DONE device_system[%zu]\n", i);
  }
}

inline void Update_Components_for_framework(size_t NumberOfComponents, Components& SystemComponents, Atoms* System)
{
  SystemComponents.Total_Components = NumberOfComponents; //Framework + 1 adsorbate
  SystemComponents.TotalNumberOfMolecules = 0;
  SystemComponents.NumberOfFrameworks = 1; //Just one framework
  SystemComponents.MoleculeName.push_back("MOF"); //Name of the framework
  SystemComponents.Moleculesize.push_back(System[0].size);
  SystemComponents.NumberOfMolecule_for_Component.push_back(1);
  SystemComponents.MolFraction.push_back(1.0);
  SystemComponents.IdealRosenbluthWeight.push_back(1.0);
  SystemComponents.FugacityCoeff.push_back(1.0);
  SystemComponents.Tc.push_back(0.0);        //Tc for framework is set to zero
  SystemComponents.Pc.push_back(0.0);        //Pc for framework is set to zero
  SystemComponents.Accentric.push_back(0.0); //Accentric factor for framework is set to zero
}

inline void Setup_Temporary_Atoms_Structure(Atoms& device_Mol, Atoms* System)
{
  //Set up MolArrays//
  size_t Allocate_size_Temporary=1024; //Assign 1024 empty slots for the temporary structures//

  device_Mol.x         = CUDA_allocate_array<double> (Allocate_size_Temporary);
  device_Mol.y         = CUDA_allocate_array<double> (Allocate_size_Temporary);
  device_Mol.z         = CUDA_allocate_array<double> (Allocate_size_Temporary);
  device_Mol.scale     = CUDA_allocate_array<double> (Allocate_size_Temporary);
  device_Mol.charge    = CUDA_allocate_array<double> (Allocate_size_Temporary);
  device_Mol.scaleCoul = CUDA_allocate_array<double> (Allocate_size_Temporary);
  device_Mol.Type      = CUDA_allocate_array<size_t> (Allocate_size_Temporary);
  device_Mol.size      = System[1].size;
  device_Mol.Allocate_size = Allocate_size_Temporary;
  device_Mol.MolID     = CUDA_allocate_array<size_t> (Allocate_size_Temporary);
}

inline void Initialize_Move_Statistics(Move_Statistics& MoveStats)
{
  MoveStats.TranslationProb = 0.0; MoveStats.RotationProb = 0.0; MoveStats.WidomProb = 0.0; MoveStats.SwapProb = 0.0; MoveStats.ReinsertionProb = 0.0;
  MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal = 0; MoveStats.TranslationAccRatio = 0.0;
  MoveStats.RotationAccepted = 0; MoveStats.RotationTotal = 0; MoveStats.RotationAccRatio = 0.0;
  MoveStats.InsertionTotal=0;   MoveStats.InsertionAccepted=0;
  MoveStats.DeletionTotal=0;    MoveStats.DeletionAccepted=0;
  MoveStats.ReinsertionTotal=0; MoveStats.ReinsertionAccepted=0;
}
inline void Setup_System_Units_and_Box(Units& Constants, Components& SystemComponents, Boxsize& Box, Boxsize& device_Box)
{
  Constants.MassUnit = 1.6605402e-27; Constants.TimeUnit = 1e-12; Constants.LengthUnit = 1e-10;
  Constants.energy_to_kelvin = 1.2027242847; Constants.BoltzmannConstant = 1.38e-23;
  SystemComponents.Beta = 1.0/(Constants.BoltzmannConstant/(Constants.MassUnit*pow(Constants.LengthUnit,2)/pow(Constants.TimeUnit,2))*Box.Temperature);
  //Convert pressure from pascal
  Box.Pressure/=(Constants.MassUnit/(Constants.LengthUnit*pow(Constants.TimeUnit,2)));
  printf("Pressure: %.10f\n", Box.Pressure);
  device_Box.Pressure = Box.Pressure;

  // READ OTHER DATA //
  device_Box.Cell        = CUDA_copy_allocate_double_array(Box.Cell,9);
  device_Box.InverseCell = CUDA_copy_allocate_double_array(Box.InverseCell,9);
  device_Box.Volume      = Box.Volume;
}

inline void Prepare_ForceField(ForceField& FF, ForceField& device_FF, PseudoAtomDefinitions PseudoAtom, Boxsize Box)
{
  std::vector<int> OtherVector(2, 0);
  FF.OtherParams = Intconvert1DVectortoArray(OtherVector);
  FF.OtherParams[0] = 0; FF.OtherParams[1] = 0;
  std::vector<double> MaxTranslation = {Box.Cell[0]*0.1, Box.Cell[4]*0.1, Box.Cell[8]*0.1};
  std::vector<double> MaxRotation    = {30.0/(180/3.1415), 30.0/(180/3.1415), 30.0/(180/3.1415)};
  FF.MaxTranslation = Doubleconvert1DVectortoArray(MaxTranslation); FF.MaxRotation = Doubleconvert1DVectortoArray(MaxRotation);

  // COPY DATA TO DEVICE POINTER //
  device_FF.FFParams      = CUDA_copy_allocate_double_array(FF.FFParams, 5);
  device_FF.OtherParams   = CUDA_copy_allocate_int_array(FF.OtherParams,2);

  device_FF.epsilon        = CUDA_copy_allocate_double_array(FF.epsilon, FF.size*FF.size);
  device_FF.sigma          = CUDA_copy_allocate_double_array(FF.sigma, FF.size*FF.size);
  device_FF.z              = CUDA_copy_allocate_double_array(FF.z, FF.size*FF.size);
  device_FF.shift          = CUDA_copy_allocate_double_array(FF.shift, FF.size*FF.size);
  device_FF.FFType         = CUDA_copy_allocate_int_array(FF.FFType, FF.size*FF.size);
  device_FF.noCharges      = FF.noCharges;
  device_FF.size           = FF.size;
  device_FF.MaxTranslation = CUDA_copy_allocate_double_array(FF.MaxTranslation, 3);
  device_FF.MaxRotation    = CUDA_copy_allocate_double_array(FF.MaxRotation, 3);
  //Formulate Component statistics on the host
  //ForceFieldParser(FF, PseudoAtom);
  //PseudoAtomParser(FF, PseudoAtom);
}

inline void Prepare_Widom(WidomStruct& Widom, Components SystemComponents, Atoms* System, Move_Statistics MoveStats)
{
  //Zhao's note: NumberWidomTrials is for first bead. NumberWidomTrialsOrientations is for the rest, here we consider single component, not mixture //

  size_t MaxTrialsize = max(Widom.NumberWidomTrials, Widom.NumberWidomTrialsOrientations*(SystemComponents.Moleculesize[1]-1));

  size_t MaxResultsize = MaxTrialsize*(System[0].Allocate_size+System[1].Allocate_size);

  Widom.flag        = (bool*)malloc(MaxTrialsize * sizeof(bool));

  cudaMalloc(&Widom.device_flag,          MaxTrialsize * sizeof(bool));
  cudaMalloc(&Widom.floatResult,          MaxResultsize*sizeof(float));
  cudaMalloc(&Widom.WidomFirstBeadResult, MaxResultsize*sizeof(double));
  cudaMalloc(&Widom.Blocksum,             MaxResultsize/DEFAULTTHREAD*sizeof(double));

  Widom.WidomFirstBeadAllocatesize = MaxResultsize;
  for(size_t i = 0; i < MoveStats.NumberOfBlocks; i++)
  {
    Widom.Rosenbluth.push_back(0.0);
    Widom.RosenbluthSquared.push_back(0.0);
    Widom.ExcessMu.push_back(0.0);
    Widom.ExcessMuSquared.push_back(0.0);
    Widom.RosenbluthCount.push_back(0);
  }
  printf("Preparing device widom\n");
}

inline double Check_Simulation_Energy(Boxsize Box, Boxsize device_Box, Atoms* System, Atoms* device_System, Atoms* d_a, ForceField FF, ForceField device_FF, Components SystemComponents, bool initial)
{
  double ewaldEnergy = 0.0;
  double sys_energy = Framework_energy_CPU(Box, System, device_System, FF, SystemComponents);
  double* xxx; xxx = (double*) malloc(sizeof(double)*2);
  double* device_xxx; device_xxx = CUDA_copy_allocate_double_array(xxx, 2);
  one_thread_GPU_test<<<1,1>>>(device_Box, d_a, device_FF, device_xxx);
  cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  if(initial){ printf("INITIAL STATE: System Energy (CPU): %.10f, Ewald (CPU): %.10f, (1 thread GPU): %.10f\n", sys_energy, ewaldEnergy, xxx[0]);}
  else{        printf("FINAL STATE: System Energy (CPU): %.10f, Ewald (CPU): %.10f, (1 thread GPU): %.10f\n", sys_energy, ewaldEnergy, xxx[0]);}
  return sys_energy;
}
