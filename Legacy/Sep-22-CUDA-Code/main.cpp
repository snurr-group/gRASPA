#include <stdio.h>
#include <math.h>
#include <cstdlib>

#include <array>

#include <omp.h>

#include "axpy.h"

//#include "data_struct.h"
#include "read_data.h"
#include "convert_array.h"
#include "write_lmp_movie.h"
double matrix_determinant(double* x) //9*1 array
{
  double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
  double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
  double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
  double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
  return determinant;
}

int main(void)
{
  check_restart_file();
  
  size_t NumberOfComponents = 2; //0: Framework; 1: adsorbate (methane)
  Atoms System[NumberOfComponents];

  Atoms device_System[NumberOfComponents];

  read_framework_atoms_from_restart_SoA(&System[0].size, &System[0].Allocate_size, &System[0].x, &System[0].y, &System[0].z, &System[0].scale, &System[0].charge, &System[0].scaleCoul, &System[0].Type, &System[0].MolID);
  printf("Framework Positions: %.10f, %.10f, %.10f\n", System[0].x[0], System[0].y[0], System[0].z[0]);
  // READ MOLECULE ATOMS //
  
  Boxsize Box; Boxsize device_Box;
  Box.Cell = read_Cell_from_restart(2); //CellArray is the first line in the section
  Box.InverseCell = read_Cell_from_restart(3); //InverseCellArray is the second line in the section  
  Box.Volume = matrix_determinant(Box.Cell);
  printf("Volume is %.10f\n", Box.Volume);


  ForceField FF; ForceField device_FF;
  bool shift = false; //Zhao's note: shifted or not, need to add it in simulation.input
  read_force_field_from_restart_SoA(&FF.size, &FF.epsilon, &FF.sigma, &FF.z, &FF.shift, &FF.FFType, shift);

  FF.FFParams = read_FFParams_from_restart();
  std::vector<int> OtherVector(2, 0);
  FF.OtherParams = Intconvert1DVectortoArray(OtherVector);
  FF.OtherParams[0] = 0; FF.OtherParams[1] = 0;
  std::vector<double> MaxTranslation(3);
  FF.MaxTranslation = Doubleconvert1DVectortoArray(MaxTranslation);
  FF.MaxTranslation[0] = Box.Cell[0]*0.1; FF.MaxTranslation[1] = Box.Cell[4]*0.1; FF.MaxTranslation[2] = Box.Cell[8]*0.1;

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
  device_System[0].MolID     = CUDA_copy_allocate_size_t_array(System[0].MolID, System[0].size);
  printf("DONE device_system[0]\n");
  // READ OTHER DATA //
  device_Box.Cell        = CUDA_copy_allocate_double_array(Box.Cell,9);
  device_Box.InverseCell = CUDA_copy_allocate_double_array(Box.InverseCell,9);
  device_Box.Volume      = Box.Volume;

  Atoms Mol; Atoms NewMol; Atoms device_Mol; Atoms device_NewMol;
  read_adsorbate_atoms_from_restart_SoA(0, &System[1].size, &System[1].Allocate_size, &System[1].x, &System[1].y, &System[1].z, &System[1].scale, &System[1].charge, &System[1].scaleCoul, &System[1].Type, &System[1].MolID);
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
  device_FF.noCharges = true;
  device_FF.size      = FF.size;
  device_FF.MaxTranslation = CUDA_copy_allocate_double_array(FF.MaxTranslation, 3);

  Move_Statistics MoveStats; MoveStats.TranslationAccepted = 0; MoveStats.TranslationTotal = 0; MoveStats.TranslationAccRatio = 0.0;
  MoveStats.TranslationProb = 0.0; MoveStats.WidomProb = 0.0; MoveStats.SwapProb = 0.0;

  WidomStruct Widom; int Cycles; 
  Units Constants;
  Constants.MassUnit = 1.6605402e-27; Constants.TimeUnit = 1e-12; Constants.LengthUnit = 1e-10;
  Constants.energy_to_kelvin = 1.2027242847; Constants.BoltzmannConstant = 1.38e-23;
  read_simulation_input(&Widom.UseGPUReduction, &Widom.Useflag, &Cycles, &Widom.NumberWidomTrials, &MoveStats.NumberOfBlocks, &MoveStats.TranslationProb, &MoveStats.WidomProb, &MoveStats.SwapProb, &Box.Pressure, &Box.Temperature);
  FF.Beta = 1.0/(Constants.BoltzmannConstant/(Constants.MassUnit*pow(Constants.LengthUnit,2)/pow(Constants.TimeUnit,2))*Box.Temperature);
  device_FF.Beta = FF.Beta; printf("Beta: %.10f\n", FF.Beta);
  //Convert pressure from pascal
  Box.Pressure/=(Constants.MassUnit/(Constants.LengthUnit*pow(Constants.TimeUnit,2)));
  printf("Pressure: %.10f\n", Box.Pressure);
  device_Box.Pressure = Box.Pressure;

  // SET UP WIDOM INSERTION //
  Widom.WidomFirstBeadResult = CUDA_allocate_double_array(Widom.NumberWidomTrials*(System[0].Allocate_size+System[1].Allocate_size));
  Widom.WidomFirstBeadAllocatesize = Widom.NumberWidomTrials*(System[0].Allocate_size+System[1].Allocate_size);
  std::vector<double> Temp(MoveStats.NumberOfBlocks);
  std::vector<int> TempInt(MoveStats.NumberOfBlocks);
  Widom.Rosenbluth = Doubleconvert1DVectortoArray(Temp);
  Widom.RosenbluthSquared = Doubleconvert1DVectortoArray(Temp);
  Widom.ExcessMu = Doubleconvert1DVectortoArray(Temp);
  Widom.ExcessMuSquared = Doubleconvert1DVectortoArray(Temp);
  Widom.RosenbluthCount = Intconvert1DVectortoArray(TempInt);
  Widom.InsertionTotal=0; Widom.InsertionAccepted=0; Widom.DeletionTotal=0;Widom.DeletionAccepted=0;

  device_y = CUDA_allocate_double_array(System[0].Allocate_size+System[1].Allocate_size);
  device_dUdlambda     = CUDA_allocate_double_array(System[0].size+System[1].Allocate_size);

  Atoms *d_a;
  cudaMalloc(&d_a, sizeof(Atoms)*2);
  cudaMemcpy(d_a, device_System, sizeof(Atoms)*2, cudaMemcpyHostToDevice);

  //Formulate Component statistics on the host
  Components SystemComponents;
  SystemComponents.Total_Components = 2; //Framework + 1 adsorbate
  std::vector<size_t> Moleculesize_vector(SystemComponents.Total_Components);
  Moleculesize_vector[0] = System[0].size; Moleculesize_vector[1] = 1; //Assume methane here
  std::vector<size_t> MoleculeNumber_vector(SystemComponents.Total_Components);
  std::vector<double> MoleculeONE_vector(SystemComponents.Total_Components, 1.0); //MolFractions start with one, idealrosenbluth also one, fugacity coeff also start from one
  MoleculeNumber_vector[0] = 1; //Just one framework
  MoleculeNumber_vector[1] = System[1].size/Moleculesize_vector[1]; //Assume methane here
  SystemComponents.TotalNumberOfMolecules = 0;
  SystemComponents.NumberOfFrameworks = 1; //Just one framework
  SystemComponents.Moleculesize = Size_tconvert1DVectortoArray(Moleculesize_vector);
  SystemComponents.NumberOfMolecule_for_Component = Size_tconvert1DVectortoArray(MoleculeNumber_vector);
  SystemComponents.MolFraction           = Doubleconvert1DVectortoArray(MoleculeONE_vector);
  SystemComponents.IdealRosenbluthWeight = Doubleconvert1DVectortoArray(MoleculeONE_vector);
  SystemComponents.FugacityCoeff         = Doubleconvert1DVectortoArray(MoleculeONE_vector);
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
    SystemComponents.TotalNumberOfMolecules += SystemComponents.NumberOfMolecule_for_Component[i];

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
  double* xxx; xxx = (double*) malloc(sizeof(double)*2);
  double* device_xxx; device_xxx = CUDA_copy_allocate_double_array(xxx, 2);
  one_thread_GPU_test<<<1,1>>>(device_Box, d_a, device_FF, device_xxx);
  cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("init: System Energy (CPU): %.10f, (1 thread GPU): %.10f\n", sys_energy, xxx[0]);
  
  //TRY TO CALCULATE THE INITIAL ENERGY//
  start = omp_get_wtime(); 
  sum = cuSoA(Cycles, SystemComponents, device_Box, device_System, d_a, device_Mol, device_NewMol, device_FF, device_y, device_dUdlambda, Random, MoveStats, Widom, Constants, sys_energy);
  end = omp_get_wtime();
  double end_sys_energy = Framework_energy_CPU(Box, System, device_System, FF, SystemComponents);
  xxx[0] = 0.0; one_thread_GPU_test<<<1,1>>>(device_Box, d_a, device_FF, device_xxx);
  cudaMemcpy(xxx, device_xxx, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("end: System Energy: %.10f, (1 thread GPU): %.10f\n", end_sys_energy, xxx[0]);
  printf("Difference Energy: %.10f, 1 thread GPU Difference: %.10f, running_difference: %.10f\n", end_sys_energy - sys_energy, xxx[0] - sys_energy, sum);
  printf("Work took %f seconds\n", end - start);

  //print data
  char* AtomNames[FF.size] = {"H", "N", "C", "Zn", "CH4_sp3", "C_CO2", "O_CO2"}; 
  create_movie_file(0, System, SystemComponents, FF, Box, AtomNames);
  
  return 0;
}
