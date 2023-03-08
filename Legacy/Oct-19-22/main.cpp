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

inline void Copy_Simulation_data_to_device(Atoms* device_System, Atoms* System)
{
  device_System[0].x         = CUDA_copy_allocate_array(System[0].x, System[0].size);
  device_System[0].y         = CUDA_copy_allocate_array(System[0].y, System[0].size);
  device_System[0].z         = CUDA_copy_allocate_array(System[0].z, System[0].size);
  device_System[0].scale     = CUDA_copy_allocate_array(System[0].scale, System[0].size);
  device_System[0].charge    = CUDA_copy_allocate_array(System[0].charge, System[0].size);
  device_System[0].scaleCoul = CUDA_copy_allocate_array(System[0].scaleCoul, System[0].size);
  device_System[0].Type      = CUDA_copy_allocate_array(System[0].Type, System[0].size);
  device_System[0].size      = System[0].size;
  device_System[0].Molsize   = System[0].Molsize;
  device_System[0].MolID     = CUDA_copy_allocate_array(System[0].MolID, System[0].size);
  //printf("DONE device_system[0]\n");
  device_System[1].x         = CUDA_copy_allocate_array(System[1].x, System[1].Allocate_size);
  device_System[1].y         = CUDA_copy_allocate_array(System[1].y, System[1].Allocate_size);
  device_System[1].z         = CUDA_copy_allocate_array(System[1].z, System[1].Allocate_size);
  device_System[1].scale     = CUDA_copy_allocate_array(System[1].scale, System[1].Allocate_size);
  device_System[1].charge    = CUDA_copy_allocate_array(System[1].charge, System[1].Allocate_size);
  device_System[1].scaleCoul = CUDA_copy_allocate_array(System[1].scaleCoul, System[1].Allocate_size);
  device_System[1].Type      = CUDA_copy_allocate_array(System[1].Type, System[1].Allocate_size);
  device_System[1].size      = System[1].size;
  device_System[1].Molsize   = System[1].Molsize;
  device_System[1].Allocate_size = System[1].Allocate_size;
  device_System[1].MolID     = CUDA_copy_allocate_array(System[1].MolID, System[1].Allocate_size);
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
  for(size_t j = 0; j < SystemComponents.Total_Components; j++){
    SystemComponents.TotalNumberOfMolecules += SystemComponents.NumberOfMolecule_for_Component[j];}
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

inline void Setup_Widom_For_Simulations(std::vector<WidomStruct>& WidomVector, WidomStruct TempWidom, size_t NumberOfSimulations, Atoms* System, Move_Statistics MoveStats, Components TemporaryComponents)//Zhao's note: as simulation goes more complicated, different simulations may use different number of trials, need to change these later.
{
  //Zhao's note: NumberWidomTrials is for first bead. NumberWidomTrialsOrientations is for the rest, here we consider single component, not mixture //
  size_t MaxTrialsize = max(TempWidom.NumberWidomTrials, TempWidom.NumberWidomTrialsOrientations*(TemporaryComponents.Moleculesize[1]-1));
  std::vector<double> Temp(MoveStats.NumberOfBlocks);
  std::vector<int> TempInt(MoveStats.NumberOfBlocks);
  TempWidom.WidomFirstBeadAllocatesize = MaxTrialsize*(System[0].Allocate_size+System[1].Allocate_size);
  TempWidom.Rosenbluth = Temp;
  TempWidom.RosenbluthSquared = Temp;
  TempWidom.ExcessMu = Temp;
  TempWidom.ExcessMuSquared = Temp;
  TempWidom.RosenbluthCount = TempInt;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    WidomVector.push_back(TempWidom);
    WidomVector[i].WidomFirstBeadResult = CUDA_allocate_array<double> (MaxTrialsize*(System[0].Allocate_size+System[1].Allocate_size));
  }
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

  // Unified Memory Pointer for Different Simulations //
  size_t NumberOfSimulations = 2;
  Simulations *Sims; 
  cudaMallocManaged(&Sims, NumberOfSimulations*sizeof(Simulations)); checkCUDAError("Error allocating Malloc"); //Let's see if one system is OK, use unified memory//
  printf("Done unified mem\n");
  for(size_t i = 0; i < NumberOfSimulations; i++)
    cudaMalloc(&Sims[i].d_a, sizeof(Atoms)*2);

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
  Move_Statistics MoveStats;
  Initialize_Move_Statistics(MoveStats);
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
  Components TemporaryComponents;
  Update_Components_for_framework(NumberOfComponents, TemporaryComponents, System); //Update the components data for all simulations
  TemporaryComponents.Allocate_size.push_back(System[0].Allocate_size);


  //For each component, there is a move statistics//
  WidomStruct TempWidom; int Cycles;
  Move_Statistics MoveStats;
  Initialize_Move_Statistics(MoveStats);
  read_simulation_input(&TempWidom.UseGPUReduction, &TempWidom.Useflag, &FF.noCharges, &Cycles, &TempWidom.NumberWidomTrials, &MoveStats.NumberOfBlocks, &Box.Pressure, &Box.Temperature, &DualPrecision);
  for(size_t i = 0; i < TemporaryComponents.Total_Components; i++)
  {
    if(i == 1)
    { //skip reading the first component, which is the framework
      read_component_values_from_simulation_input(TemporaryComponents, MoveStats, i-1, System[1], PseudoAtom); System[1].Molsize = TemporaryComponents.Moleculesize[1];
      TemporaryComponents.Allocate_size.push_back(System[1].Allocate_size);
    }
    TempWidom.NumberWidomTrialsOrientations = 20; //Set a number like this
    TemporaryComponents.Moves.push_back(MoveStats);
  }
  // SET UP UNITS //
  Units Constants;
  Constants.MassUnit = 1.6605402e-27; Constants.TimeUnit = 1e-12; Constants.LengthUnit = 1e-10;
  Constants.energy_to_kelvin = 1.2027242847; Constants.BoltzmannConstant = 1.38e-23;
  TemporaryComponents.Beta = 1.0/(Constants.BoltzmannConstant/(Constants.MassUnit*pow(Constants.LengthUnit,2)/pow(Constants.TimeUnit,2))*Box.Temperature);
  printf("Beta: %.10f\n", TemporaryComponents.Beta);
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
  device_FF.FFParams      = CUDA_copy_allocate_array(FF.FFParams, 5);
  device_FF.OtherParams   = CUDA_copy_allocate_array(FF.OtherParams,2);

  // READ OTHER DATA //
  device_Box.Cell        = CUDA_copy_allocate_array(Box.Cell,9);
  device_Box.InverseCell = CUDA_copy_allocate_array(Box.InverseCell,9);
  device_Box.Volume      = Box.Volume;

  //Atoms device_Mol; Atoms device_NewMol;
  
  //read atom definitions from either restart file, or the def file
  if(read_from_restart)
  {
    read_adsorbate_atoms_from_restart_SoA(0, &System[1].size, &System[1].Allocate_size, &System[1].x, &System[1].y, &System[1].z, &System[1].scale, &System[1].charge, &System[1].scaleCoul, &System[1].Type, &System[1].MolID);
  }


  std::vector<WidomStruct>WidomVector;
  Setup_Widom_For_Simulations(WidomVector, TempWidom, NumberOfSimulations, System, MoveStats, TemporaryComponents);


  printf("DONE Preparing widom\n");

  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    Copy_Simulation_data_to_device(device_System, System); 
    cudaMemcpy(Sims[i].d_a, device_System, sizeof(Atoms)*2, cudaMemcpyHostToDevice);
    printf("DONE device_system %zu\n", i);
  } 

  //Set up MolArrays for Every Simulation//

  Temp_Atoms* TempAtoms; cudaMallocManaged(&TempAtoms, NumberOfSimulations*sizeof(Temp_Atoms)); checkCUDAError("Error allocating Malloc");
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    Setup_Temporary_Atoms_Structure(TempAtoms[i].Mol, System);
    Setup_Temporary_Atoms_Structure(TempAtoms[i].NewMol, System);
  }
  printf("DONE Setting Up Device Atom Structures\n");

  device_FF.epsilon   = CUDA_copy_allocate_array(FF.epsilon, FF.size*FF.size);
  device_FF.sigma     = CUDA_copy_allocate_array(FF.sigma, FF.size*FF.size);
  device_FF.z         = CUDA_copy_allocate_array(FF.z, FF.size*FF.size);
  device_FF.shift     = CUDA_copy_allocate_array(FF.shift, FF.size*FF.size);
  device_FF.FFType    = CUDA_copy_allocate_array(FF.FFType, FF.size*FF.size);
  device_FF.noCharges = FF.noCharges;
  device_FF.size      = FF.size;
  device_FF.MaxTranslation = CUDA_copy_allocate_array(FF.MaxTranslation, 3);
  device_FF.MaxRotation    = CUDA_copy_allocate_array(FF.MaxRotation, 3);

  //Setup result array//
  device_y = CUDA_allocate_array<double> (System[0].Allocate_size+System[1].Allocate_size);
  device_dUdlambda     = CUDA_allocate_array<double> (System[0].size+System[1].Allocate_size);

  printf("Done alloc d_a\n");

  // Setup random number //
  RandomNumber Random;
  Random.randomsize = 100000; Random.offset = 0;
  std::vector<double> array_random(Random.randomsize);
  for (size_t i = 0; i < Random.randomsize; i++)
  {
    array_random[i] = get_random_from_zero_to_one();
  }
 
  Random.host_random = Doubleconvert1DVectortoArray(array_random); 
  Random.device_random = CUDA_copy_allocate_array(Random.host_random, Random.randomsize);

  printf("Done alloc Random\n");

  //Add TemporaryComponents to SystemComponents, as an element of the vector//
  std::vector<Components> SystemComponents;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    SystemComponents.push_back(TemporaryComponents);
  }
  double start;
  double end;
  int count = 0;
  double sum = 0.0;
  double Sumsum = 0.0;

  // Get total energy of system on the CPU (serial)//
  std::vector<double> System_Energies(NumberOfSimulations);
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    cudaMemcpy(device_System, Sims[i].d_a, 2*sizeof(Atoms), cudaMemcpyDeviceToHost);
    double sys_energy = Framework_energy_CPU(Box, System, device_System, FF, SystemComponents[i]);
    start = omp_get_wtime();

    //Get total energy (initial) on the GPU (serial), this will test if the memory on the CPU and GPU are the same//
    double* xxx; xxx = (double*) malloc(sizeof(double)*2);
    double* device_xxx; device_xxx = CUDA_copy_allocate_array(xxx, 2); 
    one_thread_GPU_test<<<1,1>>>(device_Box, Sims[i].d_a, device_FF, device_xxx); 
    cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("init: System Energy (CPU): %.10f, (1 thread GPU): %.10f\n", sys_energy, xxx[0]);
    System_Energies[i] = sys_energy;
  }

  //Perform Moves//
  start = omp_get_wtime(); 
  //sum = cuSoA(Cycles, SystemComponents[1], device_Box, Sims[1].d_a, device_Mol, device_NewMol, device_FF, device_y, device_dUdlambda, Random, Widom, Constants, System_Energies[1], DualPrecision);
  std::vector<double> Running_Energies = Multiple_Simulations(Cycles, SystemComponents, device_Box, Sims, TempAtoms, device_FF, device_y, device_dUdlambda, Random, WidomVector, Constants, System_Energies, DualPrecision);
  end = omp_get_wtime();
  printf("Work took %f seconds\n", end - start);

  //TRY TO CALCULATE THE FINAL ENERGY on the CPU and GPU, both serial//
  double end_sys_energy = 0.0;
  for(size_t i = 0; i < NumberOfSimulations; i++)
  {
    cudaMemcpy(device_System, Sims[i].d_a, 2*sizeof(Atoms), cudaMemcpyDeviceToHost);
    end_sys_energy = Framework_energy_CPU(Box, System, device_System, FF, SystemComponents[i]);
    double* xxx; xxx = (double*) malloc(sizeof(double)*2);
    double* device_xxx; device_xxx = CUDA_copy_allocate_array(xxx, 2);
    one_thread_GPU_test<<<1,1>>>(device_Box, Sims[i].d_a, device_FF, device_xxx);
    cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("end: Simulation %zu, System Energy: %.10f, (1 thread GPU): %.10f\n", i, end_sys_energy, xxx[0]);
  }
  //Create Movies For Systems//
  create_movie_file(0, System, SystemComponents[0], FF, Box, PseudoAtom.Name);
  return 0;
}
