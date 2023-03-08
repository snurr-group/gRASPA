#include "RN.h"
#include "GPU_Reduction.cuh"
//#include "data_struct.h"
#include <vector>
#include "VDW_Coulomb.cuh"
#include "mc_translation.h"
/*
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
__global__ void get_new_translation_position_CURAND(Atoms* d_a, Atoms Mol, Atoms NewMol, ForceField FF, size_t start_position, size_t SelectedComponent, size_t offset)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  //curand_init((unsigned long long)clock() + i, 0, 0, &state);  //Using random seed from clock
  curand_init(i, 0, offset, &state);
  const double rand1 = curand_uniform_double(&state);
  const double rand2 = curand_uniform_double(&state);
  const double rand3 = curand_uniform_double(&state);
  size_t real_pos = start_position + i;
  const Atoms AllData = d_a[SelectedComponent];
  const double scale = AllData.scale[real_pos];
  const double charge = AllData.charge[real_pos];
  const double scaleCoul = AllData.scaleCoul[real_pos];
  const double Type = AllData.Type[real_pos];
  const size_t MolID = AllData.MolID[real_pos];
  Mol.x[i] = AllData.x[real_pos];
  Mol.y[i] = AllData.y[real_pos];
  Mol.z[i] = AllData.z[real_pos];
  NewMol.x[i] = Mol.x[i] + FF.MaxTranslation[0] * 2.0 * (rand1 - 0.5);
  NewMol.y[i] = Mol.y[i] + FF.MaxTranslation[1] * 2.0 * (rand2 - 0.5);
  NewMol.z[i] = Mol.z[i] + FF.MaxTranslation[2] * 2.0 * (rand3 - 0.5);
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

double Translation_Move(Boxsize& Box, Components& SystemComponents, Atoms*& System, Atoms*& d_a, Atoms& Mol, Atoms& NewMol, ForceField& FF, Move_Statistics& MoveStats, double* y, double* dUdlambda, RandomNumber& Random, size_t SelectedMolInComponent, size_t SelectedComponent)
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
  Framework_energy_difference_SoA<<<Nblock, Nthread>>>(Box, d_a, Mol, NewMol, FF, y, dUdlambda,SelectedComponent);
  tot = GPUReduction<BLOCKSIZE>(y, arraysize);
  if (get_random_from_zero_to_one() < std::exp(-FF.Beta * tot))
  {
    printf("tot: %.10f\n", tot);
    update_translation_position<<<1,Mol.size>>>(d_a, NewMol, start_position, SelectedComponent);
    MoveStats.TranslationAccepted ++;
    return tot;
  }
  return 0.0;
}
*/
