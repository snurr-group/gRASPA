//##include "data_struct.h"
#include "VDW_Coulomb.cuh"
#include <cuda_fp16.h>
inline void VDW_CPU(const double* FFarg, const double rr_dot, const double scaling, double* result) //Lennard-Jones 12-6
{
  double arg1 = 4.0 * FFarg[0];
  double arg2 = FFarg[1] * FFarg[1];
  double arg3 = FFarg[3]; //the third element of the 3rd dimension of the array
  double temp = (rr_dot / arg2);
  double temp3 = temp * temp * temp;
  double rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
  double rri6 = rri3 * rri3;
  double term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
  double dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
  result[0] = scaling * term; result[1] = scaling < 1.0 ? term + (1.0 - scaling) * dlambda_term : 0.0;
}

inline void CoulombReal_CPU(const ForceField FF, const double chargeA, const double chargeB, const double r, const double scaling, double* result) //energy = -q1*q2/r
{
  double prefactor = FF.Prefactor;
  double alpha = FF.Alpha;
  double term = chargeA * chargeB * std::erfc(alpha * r);
  result[0] = prefactor * scaling * term / r;
}

inline void PBC_CPU(double* posvec, double* Cell, double* InverseCell, bool Cubic)
{
  if(Cubic)//cubic/cuboid
  {
    posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCell[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
    posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCell[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
    posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCell[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
  }
  else
  {
    double s[3] = {0.0, 0.0, 0.0};
    s[0]=InverseCell[0*3+0]*posvec[0]+InverseCell[1*3+0]*posvec[1]+InverseCell[2*3+0]*posvec[2];
    s[1]=InverseCell[0*3+1]*posvec[0]+InverseCell[1*3+1]*posvec[1]+InverseCell[2*3+1]*posvec[2];
    s[2]=InverseCell[0*3+2]*posvec[0]+InverseCell[1*3+2]*posvec[1]+InverseCell[2*3+2]*posvec[2];

    s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
    s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
    s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
    // convert from abc to xyz
    posvec[0]=Cell[0*3+0]*s[0]+Cell[1*3+0]*s[1]+Cell[2*3+0]*s[2];
    posvec[1]=Cell[0*3+1]*s[0]+Cell[1*3+1]*s[1]+Cell[2*3+1]*s[2];
    posvec[2]=Cell[0*3+2]*s[0]+Cell[1*3+2]*s[1]+Cell[2*3+2]*s[2];
  }
}

double Framework_energy_CPU(Boxsize Box, Atoms* Host_System, Atoms* System, ForceField FF, Components SystemComponents)
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  //Copy Adsorbate to host//
  for(size_t ijk=1; ijk < SystemComponents.Total_Components; ijk++) //Skip the first one(framework)
  {
    //if(Host_System[ijk].Allocate_size != System[ijk].Allocate_size)
    //{
      // if the host allocate_size is different from the device, allocate more space on the host
      Host_System[ijk].x         = (double*) malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].y         = (double*) malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].z         = (double*) malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].scale     = (double*) malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].charge    = (double*) malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].scaleCoul = (double*) malloc(System[ijk].Allocate_size*sizeof(double));
      Host_System[ijk].Type      = (size_t*) malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].MolID     = (size_t*) malloc(System[ijk].Allocate_size*sizeof(size_t));
      Host_System[ijk].size      = System[ijk].size; 
      Host_System[ijk].Allocate_size = System[ijk].Allocate_size;
    //}
  
    //if(Host_System[ijk].Allocate_size = System[ijk].Allocate_size) //means there is no more space allocated on the device than host, otherwise, allocate more on host
    //{
      cudaMemcpy(Host_System[ijk].x, System[ijk].x, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].y, System[ijk].y, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].z, System[ijk].z, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].scale, System[ijk].scale, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].charge, System[ijk].charge, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].scaleCoul, System[ijk].scaleCoul, sizeof(double)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].Type, System[ijk].Type, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(Host_System[ijk].MolID, System[ijk].MolID, sizeof(size_t)*System[ijk].Allocate_size, cudaMemcpyDeviceToHost);
      Host_System[ijk].size = System[ijk].size;
      //printf("CPU CHECK: comp: %zu, Host Allocate_size: %zu, Allocate_size: %zu\n", ijk, Host_System[ijk].Allocate_size, System[ijk].Allocate_size);
    //}
  }
  double Total_energy = 0.0; size_t count = 0; size_t cutoff_count=0;
  for(size_t compi=0; compi < SystemComponents.Total_Components; compi++) 
  {
    const Atoms Component=Host_System[compi];
    //printf("compi: %zu, size: %zu\n", compi, Component.size);
    for(size_t i=0; i<Component.size; i++)
    {
      //printf("comp: %zu, i: %zu, x: %.10f\n", compi, i, Component.x[i]);
      const double scaleA = Component.scale[i];
      const double chargeA = Component.charge[i];
      const double scalingCoulombA = Component.scaleCoul[i];
      const size_t typeA = Component.Type[i];
      const size_t MoleculeID = Component.MolID[i];
      for(size_t compj=0; compj < SystemComponents.Total_Components; compj++)
      {
        if(!((compi == 0) && (compj == 0))) //ignore fraemwrok-framework interaction
        {
          const Atoms Componentj=Host_System[compj];
          for(size_t j=0; j<Componentj.size; j++)
          {
            const double scaleB = Componentj.scale[j];
            const double chargeB = Componentj.charge[j];
            const double scalingCoulombB = Componentj.scaleCoul[j];
            const size_t typeB = Componentj.Type[j];
            const size_t MoleculeIDB = Componentj.MolID[j];
            if(!((MoleculeID == MoleculeIDB) &&(compi == compj)))
            {
              count++;
              double posvec[3] = {Component.x[i] - Componentj.x[j], Component.y[i] - Componentj.y[j], Component.z[i] - Componentj.z[j]};
              PBC_CPU(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
              const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
              //printf("i: %zu, j: %zu, rr_dot: %.10f\n", i,j,rr_dot);
              double result[2] = {0.0, 0.0};
              if(rr_dot < FF.CutOffVDW)
              {
                cutoff_count++;
                const double scaling = scaleA * scaleB;
                const size_t row = typeA*FF.size+typeB;
                const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
                VDW_CPU(FFarg, rr_dot, scaling, result);
                Total_energy += 0.5*result[0];
              }
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal_CPU(FF, chargeA, chargeB, r, scalingCoul, resultCoul);
                Total_energy += 0.5*resultCoul[0]; //prefactor merged in the CoulombReal function
              }
            }
          }
        }
      }
    }  
  }
  //printf("%zu interactions, within cutoff: %zu, energy: %.10f\n", count, Total_energy, cutoff_count);
  return Total_energy;
}

////////////////////////////// GPU CODE //////////////////////////

__device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result) //Lennard-Jones 12-6
{
  double arg1 = 4.0 * FFarg[0];
  double arg2 = FFarg[1] * FFarg[1];
  double arg3 = FFarg[3]; //the third element of the 3rd dimension of the array
  double temp = (rr_dot / arg2);
  double temp3 = temp * temp * temp;
  double rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
  double rri6 = rri3 * rri3;
  double term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
  double dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
  result[0] = scaling * term; result[1] = scaling < 1.0 ? term + (1.0 - scaling) * dlambda_term : 0.0;
}

__device__ void CoulombReal(const ForceField FF, const double chargeA, const double chargeB, const double r, const double scaling, double* result) //energy = -q1*q2/r
{
  double prefactor = FF.Prefactor;
  double alpha = FF.Alpha;
  double term = chargeA * chargeB * std::erfc(alpha * r);
  result[0] = prefactor * scaling * term / r;
}

__device__ void PBC(double* posvec, double* Cell, double* InverseCell, bool Cubic)
{
  if(Cubic)//cubic/cuboid
  {
    posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCell[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
    posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCell[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
    posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCell[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
  }
  else
  {
    double s[3] = {0.0, 0.0, 0.0};
    s[0]=InverseCell[0*3+0]*posvec[0]+InverseCell[1*3+0]*posvec[1]+InverseCell[2*3+0]*posvec[2];
    s[1]=InverseCell[0*3+1]*posvec[0]+InverseCell[1*3+1]*posvec[1]+InverseCell[2*3+1]*posvec[2];
    s[2]=InverseCell[0*3+2]*posvec[0]+InverseCell[1*3+2]*posvec[1]+InverseCell[2*3+2]*posvec[2];

    s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
    s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
    s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
    // convert from abc to xyz
    posvec[0]=Cell[0*3+0]*s[0]+Cell[1*3+0]*s[1]+Cell[2*3+0]*s[2];
    posvec[1]=Cell[0*3+1]*s[0]+Cell[1*3+1]*s[1]+Cell[2*3+1]*s[2];
    posvec[2]=Cell[0*3+2]*s[0]+Cell[1*3+2]*s[1]+Cell[2*3+2]*s[2];
  }
}

__device__ void VDW_float(const float* FFarg, const float rr_dot, const float scaling, float* result) //Lennard-Jones 12-6
{
  float arg1 = 4.0 * FFarg[0];
  float arg2 = FFarg[1] * FFarg[1];
  float arg3 = FFarg[3]; //the third element of the 3rd dimension of the array
  float temp = (rr_dot / arg2);
  float temp3 = temp * temp * temp;
  float rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
  float rri6 = rri3 * rri3;
  float term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
  float dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
  result[0] = scaling * term; result[1] = scaling < 1.0 ? term + (1.0 - scaling) * dlambda_term : 0.0;
}

__device__ void CoulombReal_float(const ForceField FF, const float chargeA, const float chargeB, const float r, const float scaling, float* result) //energy = -q1*q2/r
{
  float prefactor = __double2float_rd(FF.Prefactor);
  float alpha     = __double2float_rd(FF.Alpha);
  float term = chargeA * chargeB * std::erfc(alpha * r);
  result[0] = prefactor * scaling * term / r;
}

__device__ void PBC_float(float* posvec, float* Cell, float* InverseCell, bool Cubic)
{
  if(Cubic)//cubic/cuboid
  {
    posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCell[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
    posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCell[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
    posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCell[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
  }
  else
  {
    float s[3] = {0.0, 0.0, 0.0};
    s[0]=InverseCell[0*3+0]*posvec[0]+InverseCell[1*3+0]*posvec[1]+InverseCell[2*3+0]*posvec[2];
    s[1]=InverseCell[0*3+1]*posvec[0]+InverseCell[1*3+1]*posvec[1]+InverseCell[2*3+1]*posvec[2];
    s[2]=InverseCell[0*3+2]*posvec[0]+InverseCell[1*3+2]*posvec[1]+InverseCell[2*3+2]*posvec[2];
    s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
    s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
    s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
    // convert from abc to xyz
    posvec[0]=Cell[0*3+0]*s[0]+Cell[1*3+0]*s[1]+Cell[2*3+0]*s[2];
    posvec[1]=Cell[0*3+1]*s[0]+Cell[1*3+1]*s[1]+Cell[2*3+1]*s[2];
    posvec[2]=Cell[0*3+2]*s[0]+Cell[1*3+2]*s[1]+Cell[2*3+2]*s[2];
  }
}


__global__ void one_thread_GPU_test(Boxsize Box, Atoms* System, ForceField FF, double* xxx)
{
  bool DEBUG=false;
  //Zhao's note: added temp_xxx values for checking individual energy for each molecule//
  double temp_energy = 0.0; double temp_firstbead = 0.0; double temp_chain = 0.0; int temp_count = -1;
  double Total_energy = 0.0; size_t count = 0; size_t cutoff_count=0;
  for(size_t compi=0; compi < 2; compi++) //Zhao's note: hard coded component, change
  {
    const Atoms Component=System[compi];
    //printf("GPU CHECK: Comp: %lu, Comp size: %lu, Allocate size: %lu\n", compi, Component.size, Component.Allocate_size);
    for(size_t i=0; i<Component.size; i++)
    {
      //printf("comp: %lu, i: %lu, x: %.10f\n", compi, i, Component.x[i]);
      const double scaleA = Component.scale[i];
      const double chargeA = Component.charge[i];
      const double scalingCoulombA = Component.scaleCoul[i];
      const size_t typeA = Component.Type[i];
      const size_t MoleculeID = Component.MolID[i];
      if(DEBUG){if(MoleculeID == 5) temp_count++;} //For testing individual molecule energy//
      for(size_t compj=0; compj < 2; compj++) //Zhao's note: hard coded component, change
      {
        if(!((compi == 0) && (compj == 0))) //ignore fraemwrok-framework interaction
        {
          const Atoms Componentj=System[compj];
          for(size_t j=0; j<Componentj.size; j++)
          {
            const double scaleB = Componentj.scale[j];
            const double chargeB = Componentj.charge[j];
            const double scalingCoulombB = Componentj.scaleCoul[j];
            const size_t typeB = Componentj.Type[j];
            const size_t MoleculeIDB = Componentj.MolID[j];
            if(!((MoleculeID == MoleculeIDB) &&(compi == compj)))
            {
              count++;
              double posvec[3] = {Component.x[i] - Componentj.x[j], Component.y[i] - Componentj.y[j], Component.z[i] - Componentj.z[j]};
              PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
              const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
              double result[2] = {0.0, 0.0};
              if(rr_dot < FF.CutOffVDW)
              {
                cutoff_count++;
                const double scaling = scaleA * scaleB;
                const size_t row = typeA*FF.size+typeB;
                const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
                VDW(FFarg, rr_dot, scaling, result);
                Total_energy += 0.5*result[0];
                if(DEBUG){if(MoleculeID == 5)
                { 
                  temp_energy += result[0];
                  if(temp_count == 0){temp_firstbead += result[0];}
                  else {temp_chain += result[0];}
                } 
              }}
              //  printf("SPECIEL CHECK: compi: %lu, i: %lu, compj: %lu, j: %lu, pos: %.5f, %.5f, %.5f, rr_dot: %.10f, energy: %.10f\n", compi,i,compj,j,Component.x[i], Component.y[i], Component.z[i], rr_dot, result[0]);
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal(FF, chargeA, chargeB, r, scalingCoul, resultCoul);
                Total_energy += 0.5*resultCoul[0]; //prefactor merged in the CoulombReal function
              }
            }
          }
        }
      }
    }
  }
  if(DEBUG) printf("For Molecule 5, energy: %.10f, firstbead: %.10f, chain: %.10f\n", temp_energy, temp_firstbead, temp_chain);
  xxx[0] = Total_energy;
  //printf("xxx: %.10f\n", Total_energy);
}
/*
__device__ __forceinline__ 
double fast_float2double (float a)
{
    unsigned int ia = __float_as_int (a);
    return __hiloint2double ((((ia >> 3) ^ ia) & 0x07ffffff) ^ ia, ia << 29);
}
*/
__global__ void Collapse_Framework_Energy_OVERLAP_PARTIAL(Boxsize Box, Atoms* System, Atoms NewMol, ForceField FF, double* Blocksum, size_t ComponentID, size_t totalAtoms, bool* flag, size_t totalthreads, size_t chainsize, size_t NblockForTrial)
{
  // TEST THE SPEED OF THIS //
  // CHANGED HOW THE ith element of framework positions and y are written/accessed //
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum//
  //__shared__ double sdata[128];
  int cache_id = threadIdx.x; 
  size_t trial = blockIdx.x/NblockForTrial;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;
  size_t ij = total_ij - trial * NblockForTrial * blockDim.x;

  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; 
  //Initialize Blocksum//
  if(cache_id == 0) Blocksum[blockIdx.x] = 0.0; 

  __shared__ bool Blockflag = false;

  if(ij < totalAtoms * chainsize)
  {
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize; //ij is the thread id within the trial, just divide by chainsize to get the true i (atom id)
  size_t j = trial*chainsize + ij%chainsize; //+ ij/totalAtoms; // position in NewMol
  //printf("ij: %lu, i: %lu, j: %lu, trial: %lu, totalAtoms: %lu, totalthreads: %lu\n", ij,i,j,k,totalAtoms, totalthreads);
  size_t comp = 0;
  const size_t NumberComp = 2; //Zhao's note: need to change here for multicomponent
  size_t posi = i; size_t totalsize= 0;
  //printf("%lu, %lu\n", System[0].size, System[1].size);
  for(size_t ijk = 0; ijk < NumberComp; ijk++)
  {
    totalsize += System[ijk].size;
    if(posi >= totalsize)
    {
      comp++;
      posi -= System[ijk].size;
    }
  }

  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];

  //printf("i: %lu, posi: %lu, size1: %lu, size2: %lu\n", i, posi, System[0].size, System[1].size);

  double Pos[3] = {Component.x[posi], Component.y[posi], Component.z[posi]};
  double tempy = 0.0;
  //if(j == 6) printf("PAIR CHECK: i: %lu, j: %lu, MoleculeID: %lu, NewMol.MolID: %lu\n", i,j,MoleculeID, NewMol.MolID[0]);
  if(!((MoleculeID == NewMol.MolID[0]) &&(comp == ComponentID)))
  {
    double posvec[3] = {Pos[0] - NewMol.x[j], Pos[1] - NewMol.y[j], Pos[2] - NewMol.z[j]};
    //printf("thread: %lu, i:%lu, j:%lu, comp: %lu, posi: %lu\n", ij,i,j,comp, posi);

    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    if(rr_dot < FF.CutOffVDW)
    {
      double result[2] = {0.0, 0.0};
      const size_t typeB = NewMol.Type[j];
      const double scaleB = NewMol.scale[j];
      const double scaling = scaleA * scaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result); 
      if(result[0] > FF.OverlapCriteria){ flag[trial]=true; Blockflag = true; }
      tempy += result[0];
    }

    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = NewMol.charge[j];
      const double scalingCoulombB = NewMol.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      double resultCoul[2] = {0.0, 0.0};
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, resultCoul);
      tempy += resultCoul[0]; //prefactor merged in the CoulombReal function
    }
  }
  sdata[ij_within_block] = tempy;
  }
  __syncthreads();
  //Partial block sum//
  if(!Blockflag)
  {
    int i=blockDim.x / 2;
    while(i != 0) 
    {
      if(cache_id < i) {sdata[cache_id] += sdata[cache_id + i];}
      __syncthreads();
      i /= 2;
    }
    if(cache_id == 0) {Blocksum[blockIdx.x] = sdata[0];}
  }
}
