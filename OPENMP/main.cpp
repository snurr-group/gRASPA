#include <stdio.h>
#include <math.h>
#include <cstdlib>

#include <omp.h>

#include "GPU_alloc.h"
#include "read_data.h"
#include "convert_array.h"
#include "framework_energy_difference.h"
int main(void)
{
  check_restart_file();
  size_t  Frameworksize = 0; size_t FFsize = 0;
  double* FrameworkArray; size_t* FrameworkTypeArray; double* FFArray;
  int*    FFTypeArray; double* CellArray; double* InverseCellArray; double* FFParams; int OtherParams[2];
  FrameworkArray = read_framework_atoms_from_restart(&Frameworksize);
  FrameworkTypeArray = read_framework_atoms_types_from_restart();
  FFArray = read_force_field_from_restart(&FFsize);
  FFTypeArray = read_force_field_type_from_restart();
  CellArray = read_Cell_from_restart(2); //CellArray is the first line in the section
  InverseCellArray = read_Cell_from_restart(3); //InverseCellArray is the second line in the section
  FFParams = read_FFParams_from_restart();
  OtherParams[0] = 0; OtherParams[1] = 0;
  //Set up MolArrays//
  size_t max_component_size = 1; if(max_component_size == 1) max_component_size++;
  std::vector<double> MolVector(max_component_size*6, 0.0);
  std::vector<double> NewMolVector(max_component_size*6, 0.0);
  std::vector<size_t> MolTypeVector(max_component_size, 0);
  double* MolArray;
  MolArray = Doubleconvert1DVectortoArray(MolVector);
  double* NewMolArray;
  NewMolArray = Doubleconvert1DVectortoArray(NewMolVector);
  size_t* MolTypeArray;
  MolTypeArray = Size_tconvert1DVectortoArray(MolTypeVector);
  //Initialize Mol values//
  MolArray[0] = 19.54630; MolArray[1] = 23.13125; MolArray[2] = 19.17187; 
  MolArray[3] = 1.0;      MolArray[4] = 0.0;      MolArray[5] = 1.0;
  NewMolArray[0] = 19.69; NewMolArray[1] = 23.13125; NewMolArray[2] = 19.17187;
  NewMolArray[3] = 1.0;      NewMolArray[4] = 0.0;      NewMolArray[5] = 1.0;
  MolTypeArray[0] = 0;
  gpu_alloc_double(MolArray, max_component_size*6);
  gpu_copy_double(MolArray, max_component_size*6);

  gpu_alloc_double(NewMolArray, max_component_size*6);
  gpu_copy_double(NewMolArray, max_component_size*6);

  gpu_alloc_size_t(MolTypeArray, max_component_size);
  gpu_copy_size_t(MolTypeArray, max_component_size);
  //Set up other values//
  bool noCharges = true;
  // ALLOCATE MEMORY ON GPU //

  gpu_alloc_double(FrameworkArray, Frameworksize*6);
  gpu_copy_double(FrameworkArray, Frameworksize*6);
  
  gpu_alloc_size_t(FrameworkTypeArray, Frameworksize);
  gpu_copy_size_t(FrameworkTypeArray, Frameworksize);

  gpu_alloc_double(FFArray, FFsize*FFsize*4);
  gpu_copy_double(FFArray, FFsize*FFsize*4);

  gpu_alloc_int(FFTypeArray, FFsize*FFsize);
  gpu_copy_int(FFTypeArray, FFsize*FFsize);

  gpu_alloc_double(CellArray, 9);
  gpu_copy_double(CellArray, 9);
  gpu_alloc_double(InverseCellArray, 9);
  gpu_copy_double(InverseCellArray, 9);


  gpu_alloc_double(FFParams, 5);
  gpu_copy_double(FFParams, 5);

  gpu_alloc_int(OtherParams, 2);
  gpu_copy_int(OtherParams, 2);

  gpu_alloc_single_size_t(Frameworksize);
  gpu_copy_single_size_t(Frameworksize);

  gpu_alloc_single_size_t(FFsize);
  gpu_copy_single_size_t(FFsize);

  gpu_alloc_single_bool(noCharges);
  gpu_copy_single_bool(noCharges);

  //////////////////////////
  // PERFORM CALCULATIONS //
  //////////////////////////
  size_t Molsize = 1;
  int Cycles = 200000; //20000
  std::array<double,3> result_energy; 
  // START TIMING //
  double start; 
  double end; 
  start = omp_get_wtime(); 

  for (int i = 0; i < Cycles; i++)
  {
    result_energy = computeFrameworkMoleculeEnergyDifferenceGPU(CellArray, InverseCellArray, FrameworkArray, FrameworkTypeArray, FFArray, FFTypeArray, MolArray, MolTypeArray, NewMolArray, FFParams, OtherParams, Frameworksize, Molsize, FFsize, noCharges);
  }
  end = omp_get_wtime(); 
  printf("Work took %f seconds\n", end - start);

  printf("Energy: %.5f\n", result_energy[0]);
  //////////////////////////
  // THEN FREE THE MEMORY //
  //////////////////////////
  gpu_free_double(FrameworkArray, Frameworksize*6);
  gpu_free_size_t(FrameworkTypeArray, Frameworksize);
  printf("GOOD0\n");
  gpu_free_double(FFArray, FFsize*FFsize*4);
  gpu_free_int(FFTypeArray, FFsize*FFsize);
  printf("GOOD1\n");
  gpu_free_double(CellArray, 9);
  gpu_free_double(InverseCellArray, 9);
  printf("GOOD2\n");
  gpu_free_double(FFParams, 5);
  gpu_free_int(OtherParams, 2);
  printf("GOOD3\n");
  return 0;
}
