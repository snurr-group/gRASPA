#include <stdio.h>
#include <math.h>
#include <cstdlib>

#include <array>

#include <omp.h>

#include "axpy.h"

#include "read_data.h"
#include "convert_array.h"
#include "write_lmp_movie.h"
//#include "ewald.h"

inline void Setup_threadblock_main(size_t arraysize, size_t *Nblock, size_t *Nthread) //Zhao's note: need to delete it//
{
  size_t value = arraysize;
  if(value >= DEFAULTTHREAD) value = DEFAULTTHREAD;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  *Nthread = value;
  *Nblock = blockValue;
}

bool DualPrecision = false; //Whether or not to use Dual-Precision CBMC//

bool read_from_restart = false; //The system can be read from def files or from a Restart file in the Restart folder
bool DEBUG=true;
int main(void)
{
  if(read_from_restart){
  check_restart_file();
  }
  size_t NumberOfComponents = 2; //0: Framework; 1: adsorbate (methane)
  Atoms System[NumberOfComponents];
  Atoms device_System[NumberOfComponents];
  Boxsize Box; Boxsize device_Box;
  PseudoAtomDefinitions PseudoAtom;
  ForceField FF; ForceField device_FF;
  if(read_from_restart)
  {
    read_framework_atoms_from_restart_SoA(&System[0].size, &System[0].Allocate_size, &System[0].x, &System[0].y, &System[0].z, &System[0].scale, &System[0].charge, &System[0].scaleCoul, &System[0].Type, &System[0].MolID);
    Read_Cell_wrapper(Box);
    bool shift = false; //Zhao's note: shifted or not, need to add it in simulation.input
    read_force_field_from_restart_SoA(&FF.size, &FF.epsilon, &FF.sigma, &FF.z, &FF.shift, &FF.FFType, shift);
  }else{
    // READ THE Forcefield, then PseudoAtoms
    ForceFieldParser(FF, PseudoAtom);
    PseudoAtomParser(FF, PseudoAtom);
    WidomStruct Widom; int Cycles; std::string FrameworkName;
  Move_Statistics MoveStats; MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal = 0; MoveStats.TranslationAccRatio = 0.0;
                             MoveStats.RotationAccepted = 0; MoveStats.RotationTotal = 0; MoveStats.RotationAccRatio = 0.0;
  MoveStats.TranslationProb = 0.0; MoveStats.RotationProb = 0.0; MoveStats.WidomProb = 0.0; MoveStats.SwapProb = 0.0; MoveStats.ReinsertionProb = 0.0;
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &Cycles, &Widom.NumberWidomTrials, &MoveStats.NumberOfBlocks, &Box.Pressure, &Box.Temperature, &DualPrecision);
    POSCARParser(Box, System[0],PseudoAtom);
    System[0].Molsize = System[0].size;
  }
  if(DEBUG)
  {
    printf("FF.ep: %.10f, FF.sig: %.10f\n", FF.epsilon[0], FF.sigma[0], FF.shift[0]);
    printf("Framework Positions: %.10f, %.10f, %.10f\n", System[0].x[0], System[0].y[0], System[0].z[0]);
    printf("Volume is %.10f\n", Box.Volume);
    //std::vector<std::string>Names={"H", "N", "C", "Zn", "CH4_sp3", "C_co2", "O_co2"};
    printf("Inversed cell: %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", Box.InverseCell[0], Box.InverseCell[1], Box.InverseCell[2], Box.InverseCell[3], Box.InverseCell[4], Box.InverseCell[5], Box.InverseCell[6], Box.InverseCell[7], Box.InverseCell[8]);
  }

  //Formulate Component statistics on the host
  Components SystemComponents;
  SystemComponents.Total_Components = 2; //Framework + 1 adsorbate
  std::vector<double> MoleculeONE_vector(SystemComponents.Total_Components, 1.0); //MolFractions start with one, idealrosenbluth also one, fugacity coeff also start from one
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
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
    SystemComponents.TotalNumberOfMolecules += SystemComponents.NumberOfMolecule_for_Component[i];

  //For each component, there is a move statistics//
  WidomStruct Widom; int Cycles;
  Move_Statistics MoveStats; MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal = 0; MoveStats.TranslationAccRatio = 0.0;
                             MoveStats.RotationAccepted = 0; MoveStats.RotationTotal = 0; MoveStats.RotationAccRatio = 0.0;
  MoveStats.TranslationProb = 0.0; MoveStats.RotationProb = 0.0; MoveStats.WidomProb = 0.0; MoveStats.SwapProb = 0.0; MoveStats.ReinsertionProb = 0.0;
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &FF.noCharges, &Cycles, &Widom.NumberWidomTrials, &MoveStats.NumberOfBlocks, &Box.Pressure, &Box.Temperature, &DualPrecision);
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    if(i == 1){ //skip reading the first component, which is the framework
      read_component_values_from_simulation_input(SystemComponents, MoveStats, i-1, System[1], PseudoAtom); System[1].Molsize = SystemComponents.Moleculesize[1];}
    Widom.NumberWidomTrialsOrientations = 20; //Set a number like this
    SystemComponents.Moves.push_back(MoveStats);
  }

  // SET UP WIDOM INSERTION //
  Units Constants;
  Constants.MassUnit = 1.6605402e-27; Constants.TimeUnit = 1e-12; Constants.LengthUnit = 1e-10;
  Constants.energy_to_kelvin = 1.2027242847; Constants.BoltzmannConstant = 1.38e-23;
  FF.Beta = 1.0/(Constants.BoltzmannConstant/(Constants.MassUnit*pow(Constants.LengthUnit,2)/pow(Constants.TimeUnit,2))*Box.Temperature);
  device_FF.Beta = FF.Beta; printf("Beta: %.10f\n", FF.Beta);
  //Convert pressure from pascal
  Box.Pressure/=(Constants.MassUnit/(Constants.LengthUnit*pow(Constants.TimeUnit,2)));
  printf("Pressure: %.10f\n", Box.Pressure);
  device_Box.Pressure = Box.Pressure;

  FF.FFParams = read_FFParams_from_restart();
  std::vector<int> OtherVector(2, 0);
  FF.OtherParams = Intconvert1DVectortoArray(OtherVector);
  FF.OtherParams[0] = 0; FF.OtherParams[1] = 0;
  std::vector<double> MaxTranslation(3);
  FF.MaxTranslation = Doubleconvert1DVectortoArray(MaxTranslation); FF.MaxRotation = Doubleconvert1DVectortoArray(MaxTranslation);
  FF.MaxTranslation[0] = Box.Cell[0]*0.1; FF.MaxTranslation[1] = Box.Cell[4]*0.1; FF.MaxTranslation[2] = Box.Cell[8]*0.1;
  FF.MaxRotation[0] = 30.0/(180/3.1415); FF.MaxRotation[1] = 30.0/(180/3.1415); FF.MaxRotation[2] = 30.0/(180/3.1415);

  double* device_y; double* device_dUdlambda; // result pointer, may need to move in the cuda function //
  // COPY DATA TO DEVICE POINTER //
  device_FF.FFParams      = CUDA_copy_allocate_double_array(FF.FFParams, 5);
  device_FF.OtherParams   = CUDA_copy_allocate_int_array(FF.OtherParams,2);

  device_System[0].x         = CUDA_copy_allocate_double_array(System[0].x, System[0].size);
  device_System[0].y         = CUDA_copy_allocate_double_array(System[0].y, System[0].size);
  device_System[0].z         = CUDA_copy_allocate_double_array(System[0].z, System[0].size);
  device_System[0].scale     = CUDA_copy_allocate_double_array(System[0].scale, System[0].size);
  device_System[0].charge    = CUDA_copy_allocate_double_array(System[0].charge, System[0].size);
  device_System[0].scaleCoul = CUDA_copy_allocate_double_array(System[0].scaleCoul, System[0].size);
  device_System[0].Type      = CUDA_copy_allocate_size_t_array(System[0].Type, System[0].size);
  device_System[0].size      = System[0].size;
  device_System[0].Molsize   = System[0].Molsize;
  device_System[0].MolID     = CUDA_copy_allocate_size_t_array(System[0].MolID, System[0].size);
  printf("DONE device_system[0]\n");
  // READ OTHER DATA //
  device_Box.Cell        = CUDA_copy_allocate_double_array(Box.Cell,9);
  device_Box.InverseCell = CUDA_copy_allocate_double_array(Box.InverseCell,9);

  //Downcasting Boxsize to floats//
  //device_Box.float_Cell         = CUDA_copy_allocate_downcast_to_float(Box.Cell, 9);
  //device_Box.float_InverseCell  = CUDA_copy_allocate_downcast_to_float(Box.InverseCell, 9);
  

  device_Box.Volume      = Box.Volume;

  Atoms Mol; Atoms NewMol; Atoms device_Mol; Atoms device_NewMol;
  
  //read atom definitions from either restart file, or the def file
  if(read_from_restart)
  {
    read_adsorbate_atoms_from_restart_SoA(0, &System[1].size, &System[1].Allocate_size, &System[1].x, &System[1].y, &System[1].z, &System[1].scale, &System[1].charge, &System[1].scaleCoul, &System[1].Type, &System[1].MolID);
  }

  //print values for SystemComp

  //Zhao's note: NumberWidomTrials is for first bead. NumberWidomTrialsOrientations is for the rest, here we consider single component, not mixture //
  size_t MaxTrialsize = max(Widom.NumberWidomTrials, Widom.NumberWidomTrialsOrientations*(SystemComponents.Moleculesize[1]-1));

  size_t MaxResultsize = MaxTrialsize*(System[0].Allocate_size+System[1].Allocate_size);
  
  Widom.flag        = (bool*)malloc(MaxTrialsize * sizeof(bool));

  cudaMalloc(&Widom.device_flag,          MaxTrialsize * sizeof(bool));
  cudaMalloc(&Widom.floatResult,          MaxResultsize*sizeof(float));
  cudaMalloc(&Widom.WidomFirstBeadResult, MaxResultsize*sizeof(double));
  cudaMalloc(&Widom.Blocksum,             MaxResultsize/DEFAULTTHREAD*sizeof(double)); 

  Widom.WidomFirstBeadAllocatesize = MaxResultsize;
  std::vector<double> Temp(MoveStats.NumberOfBlocks);
  std::vector<int> TempInt(MoveStats.NumberOfBlocks);
  Widom.Rosenbluth = Doubleconvert1DVectortoArray(Temp);
  Widom.RosenbluthSquared = Doubleconvert1DVectortoArray(Temp);
  Widom.ExcessMu = Doubleconvert1DVectortoArray(Temp);
  Widom.ExcessMuSquared = Doubleconvert1DVectortoArray(Temp);
  Widom.RosenbluthCount = Intconvert1DVectortoArray(TempInt);
  Widom.InsertionTotal=0; Widom.InsertionAccepted=0; Widom.DeletionTotal=0;Widom.DeletionAccepted=0; 
  Widom.ReinsertionTotal=0; Widom.ReinsertionAccepted=0;

  printf("Preparing device widom\n");

  //Set up MolArrays//
  size_t Molsize = 1;
  size_t max_component_size = Molsize; if(max_component_size == 1) max_component_size++;
  std::vector<double> MolVector(max_component_size, 0.0);
  std::vector<size_t> MolTypeVector(max_component_size, 0);
  //Mol and NewMol are temporary arrays//
  Mol.x         = Doubleconvert1DVectortoArray(MolVector);
  Mol.y         = Doubleconvert1DVectortoArray(MolVector);
  Mol.z         = Doubleconvert1DVectortoArray(MolVector);
  Mol.scale     = Doubleconvert1DVectortoArray(MolVector);
  Mol.charge    = Doubleconvert1DVectortoArray(MolVector);
  Mol.scaleCoul = Doubleconvert1DVectortoArray(MolVector);
  Mol.Type      = Size_tconvert1DVectortoArray(MolTypeVector);
  Mol.MolID     = Size_tconvert1DVectortoArray(MolTypeVector);
  Mol.size      = System[1].size;
  Mol.Allocate_size = 1024;

  NewMol.x         = Doubleconvert1DVectortoArray(MolVector);
  NewMol.y         = Doubleconvert1DVectortoArray(MolVector);
  NewMol.z         = Doubleconvert1DVectortoArray(MolVector);
  NewMol.scale     = Doubleconvert1DVectortoArray(MolVector);
  NewMol.charge    = Doubleconvert1DVectortoArray(MolVector);
  NewMol.scaleCoul = Doubleconvert1DVectortoArray(MolVector);
  NewMol.Type      = Size_tconvert1DVectortoArray(MolTypeVector);
  NewMol.MolID     = Size_tconvert1DVectortoArray(MolTypeVector);
  NewMol.size      = System[1].size;
  NewMol.Allocate_size = 1024;
  
  //Initialize Mol values//
  //Mol.x[0] = 19.54630; Mol.y[0] = 8.82521; Mol.z[0] = 11.18792;
  Mol.x[0] = 19.54630; Mol.y[0] = 23.13125; Mol.z[0] = 19.17187;
  Mol.scale[0] = 1.0;      Mol.charge[0] = 0.0;      Mol.scaleCoul[0] = 1.0; Mol.MolID[0] = 0;
  NewMol.x[0] = 19.69; NewMol.y[0] = 23.13125; NewMol.z[0] = 19.17187;
  NewMol.scale[0] = 1.0;      NewMol.charge[0] = 0.0;      NewMol.scaleCoul[0] = 1.0; NewMol.MolID[0] = 0;
  Mol.Type[0] = 0; NewMol.Type[0] = 0;

  //Copy data to device_Mol and device_NewMol//
  device_System[1].x         = CUDA_copy_allocate_double_array(System[1].x, System[1].Allocate_size);
  device_System[1].y         = CUDA_copy_allocate_double_array(System[1].y, System[1].Allocate_size);
  device_System[1].z         = CUDA_copy_allocate_double_array(System[1].z, System[1].Allocate_size);
  device_System[1].scale     = CUDA_copy_allocate_double_array(System[1].scale, System[1].Allocate_size);
  device_System[1].charge    = CUDA_copy_allocate_double_array(System[1].charge, System[1].Allocate_size);
  device_System[1].scaleCoul = CUDA_copy_allocate_double_array(System[1].scaleCoul, System[1].Allocate_size);
  device_System[1].Type      = CUDA_copy_allocate_size_t_array(System[1].Type, System[1].Allocate_size);
  device_System[1].size      = System[1].size;
  device_System[1].Molsize   = System[1].Molsize;
  device_System[1].Allocate_size = System[1].Allocate_size;
  device_System[1].MolID     = CUDA_copy_allocate_size_t_array(System[1].MolID, System[1].Allocate_size);
  printf("DONE device_system[1]\n");
  
 
  device_Mol.x         = CUDA_copy_allocate_double_array(Mol.x, Mol.Allocate_size);
  device_Mol.y         = CUDA_copy_allocate_double_array(Mol.y, Mol.Allocate_size);
  device_Mol.z         = CUDA_copy_allocate_double_array(Mol.z, Mol.Allocate_size);
  device_Mol.scale     = CUDA_copy_allocate_double_array(Mol.scale, Mol.Allocate_size);
  device_Mol.charge    = CUDA_copy_allocate_double_array(Mol.charge, Mol.Allocate_size);
  device_Mol.scaleCoul = CUDA_copy_allocate_double_array(Mol.scaleCoul, Mol.Allocate_size);
  device_Mol.Type      = CUDA_copy_allocate_size_t_array(Mol.Type, Mol.Allocate_size);
  device_Mol.size      = Mol.size;
  device_Mol.Allocate_size = Mol.Allocate_size;
  device_Mol.MolID     = CUDA_copy_allocate_size_t_array(Mol.MolID, Mol.Allocate_size); 
  printf("DONE device_Mol\n"); 
 
  device_NewMol.x         = CUDA_copy_allocate_double_array(NewMol.x, NewMol.Allocate_size);
  device_NewMol.y         = CUDA_copy_allocate_double_array(NewMol.y, NewMol.Allocate_size);
  device_NewMol.z         = CUDA_copy_allocate_double_array(NewMol.z, NewMol.Allocate_size);
  device_NewMol.scale     = CUDA_copy_allocate_double_array(NewMol.scale, NewMol.Allocate_size);
  device_NewMol.charge    = CUDA_copy_allocate_double_array(NewMol.charge, NewMol.Allocate_size);
  device_NewMol.scaleCoul = CUDA_copy_allocate_double_array(NewMol.scaleCoul, NewMol.Allocate_size);
  device_NewMol.Type      = CUDA_copy_allocate_size_t_array(NewMol.Type, NewMol.Allocate_size);
  device_NewMol.size      = NewMol.size;
  device_NewMol.Allocate_size = NewMol.Allocate_size;
  device_NewMol.MolID     = CUDA_copy_allocate_size_t_array(NewMol.MolID, NewMol.Allocate_size);
  printf("DONE device_NewMol\n");

  device_FF.epsilon   = CUDA_copy_allocate_double_array(FF.epsilon, FF.size*FF.size);
  device_FF.sigma     = CUDA_copy_allocate_double_array(FF.sigma, FF.size*FF.size);
  device_FF.z         = CUDA_copy_allocate_double_array(FF.z, FF.size*FF.size);
  device_FF.shift     = CUDA_copy_allocate_double_array(FF.shift, FF.size*FF.size);
  device_FF.FFType    = CUDA_copy_allocate_int_array(FF.FFType, FF.size*FF.size);
  device_FF.noCharges = FF.noCharges;
  device_FF.size      = FF.size;
  device_FF.MaxTranslation = CUDA_copy_allocate_double_array(FF.MaxTranslation, 3);
  device_FF.MaxRotation    = CUDA_copy_allocate_double_array(FF.MaxRotation, 3);

  //Setup result array//
  device_y = CUDA_allocate_double_array(System[0].Allocate_size+System[1].Allocate_size);
  device_dUdlambda     = CUDA_allocate_double_array(System[0].size+System[1].Allocate_size);

  Atoms *d_a;
  cudaMalloc(&d_a, sizeof(Atoms)*2);
  cudaMemcpy(d_a, device_System, sizeof(Atoms)*2, cudaMemcpyHostToDevice);

  // Setup random number //
  RandomNumber Random;
  Random.randomsize = 100000; Random.offset = 0;
  std::vector<double> array_random(Random.randomsize);
  for (size_t i = 0; i < Random.randomsize; i++)
  {
    array_random[i] = get_random_from_zero_to_one();
    //printf("random: %.10f\n", host_random[i]);
  }
  //printf("Random: %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", host_random[0], host_random[1], host_random[2], host_random[3], host_random[4], host_random[5]);
 
  Random.host_random = Doubleconvert1DVectortoArray(array_random); 
  Random.device_random = CUDA_copy_allocate_double_array(Random.host_random, Random.randomsize);

  // SET UP MolArray //
  // DECLARE DEVICE POINTERS //
  // CALL THE CUDA FUNCTION //
  double start;
  double end;
  int count = 0;
  double sum = 0.0;
  double Sumsum = 0.0;
  // Get total energy of system //
  double sys_energy = Framework_energy_CPU(Box, System, device_System, FF, SystemComponents);


/*
  start = omp_get_wtime();
  double ewaldEnergy = Ewald_Total(Box, System,FF, SystemComponents);

  end = omp_get_wtime(); double CPU_ewald_time = end-start;

  GPU_ewald_alloc(Box, device_Box, System, FF, SystemComponents.Total_Components);
  size_t numberOfAtoms = 0;
  for(size_t i=0; i < SystemComponents.Total_Components; i++) //Skip the first one(framework)
  {
    numberOfAtoms  += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  }
  //Initialize the Ewald Vectors//
  if(!FF.noCharges){
  start = omp_get_wtime();
  Setup_Ewald_Vector<<<1,1>>>(device_Box, d_a, numberOfAtoms, SystemComponents.Total_Components);
  size_t Nthread=0; size_t Nblock=0;  Setup_threadblock_main(9*17*17*numberOfAtoms, &Nblock, &Nthread);
  Complex *ewaldE; cudaMalloc(&ewaldE, sizeof(Complex)*9*17*17*numberOfAtoms);
  Complex host_E[9*17*17*numberOfAtoms];
  GPU_Ewald_Total_Box<<<Nblock,Nthread>>>(device_Box, d_a, device_FF, SystemComponents.Total_Components, ewaldE, numberOfAtoms);
  double host_tot=0.0; 

  size_t num_streams = 9*17*17;
  cudaStream_t streams[num_streams];
  Complex *temp_hostE; temp_hostE = (Complex*) malloc(9*17*17*numberOfAtoms*sizeof(Complex));
  //Complex temp_hostE[numberOfAtoms];
  //#pragma omp parallel for reduction(+:host_tot)
  for(size_t i = 0; i < 9*17*17; i++) //kx, ky, kz
  {
    //cudaStreamCreate(&streams[i]);
    //Complex temp_hostE[numberOfAtoms];
    cudaMemcpy(&temp_hostE[i*numberOfAtoms], &ewaldE[i*numberOfAtoms], sizeof(Complex)*numberOfAtoms, cudaMemcpyDeviceToHost);
    //cudaMemcpyAsync(&temp_hostE[i*numberOfAtoms], &ewaldE[i*numberOfAtoms], sizeof(Complex)*numberOfAtoms, cudaMemcpyDeviceToHost, streams[i]);
    Complex tempsum; tempsum.real = 0.0; tempsum.imag = 0.0;
    for(size_t j = 0; j < numberOfAtoms; j++)
    {
      tempsum.real += temp_hostE[i * numberOfAtoms + j].real; tempsum.imag += temp_hostE[i * numberOfAtoms + j].imag;
      //tempsum.real += temp_hostE[j].real; tempsum.imag += temp_hostE[j].imag;
    }
    host_tot += tempsum.real * tempsum.real + tempsum.imag * tempsum.imag;
    //cudaStreamDestroy(streams[i]);
  }
  end = omp_get_wtime();
  printf("Pre-exclusion GPU_threaded ewald sum: %.10f, CPU took: %.10f seconds, GPU took %.10f seconds\n", host_tot, CPU_ewald_time, end-start); 


  //Determine the number of threads needed for the exclusion//
  size_t threadNeeeded = 0;
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    size_t NumMol = System[i].size/System[i].Molsize;
    //For each molecule, we do the pairwise count inside of the molecule//
    //For a molecule of size 5, it needs 5*4*3*2*1 threads//
    size_t MolThreads = System[i].Molsize * (System[i].Molsize + 1)/2; 
    threadNeeeded += NumMol * MolThreads;
  }
  printf("For the whole system, we need %zu threads\n", threadNeeeded);
  
  Setup_threadblock_main(threadNeeeded, &Nblock, &Nthread);
  double *excludeE; cudaMalloc(&excludeE, sizeof(double)*threadNeeeded);
  Ewald_Total_Exclusion_Energy<<<Nblock, Nthread>>>(device_Box, d_a, SystemComponents.Total_Components, numberOfAtoms, device_FF, excludeE);
  double host_Exclude[threadNeeeded];
  cudaMemcpy(host_Exclude, excludeE, sizeof(double)*threadNeeeded, cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < threadNeeeded; i++) //kx, ky, kz
  {
    host_tot += host_Exclude[i];
  }
  end = omp_get_wtime();
  printf("Post-exclusion GPU_threaded ewald sum: %.10f, total GPU took: %.10f seconds\n", host_tot, end-start);
  }
  //GPU_Ewald_Total_singlethread<<<1,1>>>(device_Box, d_a, device_FF, SystemComponents.Total_Components);
*/
  double ewaldEnergy = 0.0;
  double* xxx; xxx = (double*) malloc(sizeof(double)*2);
  double* device_xxx; device_xxx = CUDA_copy_allocate_double_array(xxx, 2);
  one_thread_GPU_test<<<1,1>>>(device_Box, d_a, device_FF, device_xxx);
  cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("init: System Energy (CPU): %.10f, Ewald (CPU): %.10f, (1 thread GPU): %.10f\n", sys_energy, ewaldEnergy, xxx[0]);
  //TRY TO CALCULATE THE INITIAL ENERGY//
  start = omp_get_wtime(); 
  sum = cuSoA(Cycles, SystemComponents, device_Box, device_System, d_a, device_Mol, device_NewMol, device_FF, device_y, device_dUdlambda, Random, Widom, Constants, sys_energy, DualPrecision);
  end = omp_get_wtime();
  double end_sys_energy = Framework_energy_CPU(Box, System, device_System, FF, SystemComponents);
  xxx[0] = 0.0; one_thread_GPU_test<<<1,1>>>(device_Box, d_a, device_FF, device_xxx);
  cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("end: System Energy: %.10f, (1 thread GPU): %.10f\n", end_sys_energy, xxx[0]);
  printf("Difference Energy: %.10f, 1 thread GPU Difference: %.10f, running_difference: %.10f\n", end_sys_energy - sys_energy, xxx[0] - sys_energy, sum);
  printf("Work took %f seconds\n", end - start);
  //print data
  //char* AtomNames[FF.size] = {"H", "N", "C", "Zn", "CH4_sp3", "CH3_sp3", "C_co2", "O_co2"}; 
  create_movie_file(0, System, SystemComponents, FF, Box, PseudoAtom.Name);
  return 0;
}
