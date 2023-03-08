#include <complex>
#include "VDW_Coulomb.cuh"
#include <cuda_fp16.h>
#include <omp.h>
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

inline void matrix_multiply_by_vector(double* a, double* b, double* c) //3x3(9*1) matrix (a) times 3x1(3*1) vector (b), a*b=c//
{
  c[0]=a[0*3+0]*b[0]+a[1*3+0]*b[1]+a[2*3+0]*b[2];
  c[1]=a[0*3+1]*b[0]+a[1*3+1]*b[1]+a[2*3+1]*b[2];
  c[2]=a[0*3+2]*b[0]+a[1*3+2]*b[1]+a[2*3+2]*b[2];
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

double CPU_EwaldDifference(Boxsize& Box, Atoms& New, Atoms& Old, ForceField& FF, Components& SystemComponents, size_t SelectedComponent, bool Swap, size_t SelectedTrial)
{
  double recip_cutoff = Box.ReciprocalCutOff;
  int kx_max = Box.kmax.x;
  int ky_max = Box.kmax.y;
  int kz_max = Box.kmax.z;
  if(FF.noCharges) return 0.0;
  double alpha = FF.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = FF.Prefactor * (2.0 * M_PI / Box.Volume);

  double ewaldE = 0.0;

  Boxsize Host_Box;
  Host_Box.Cell        = (double*) malloc(9 * sizeof(double)); cudaMemcpy(Host_Box.Cell, Box.Cell, 9*sizeof(double), cudaMemcpyDeviceToHost);
  Host_Box.InverseCell = (double*) malloc(9 * sizeof(double)); cudaMemcpy(Host_Box.InverseCell, Box.InverseCell, 9*sizeof(double), cudaMemcpyDeviceToHost);
  Host_Box.Cubic       = Box.Cubic;

  double ax[3] = {Host_Box.InverseCell[0], Host_Box.InverseCell[3], Host_Box.InverseCell[6]}; //printf("ax: %.10f, %.10f, %.10f\n", Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]);
  double ay[3] = {Host_Box.InverseCell[1], Host_Box.InverseCell[4], Host_Box.InverseCell[7]}; //printf("ay: %.10f, %.10f, %.10f\n", Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]);
  double az[3] = {Host_Box.InverseCell[2], Host_Box.InverseCell[5], Host_Box.InverseCell[8]}; //printf("az: %.10f, %.10f, %.10f\n", Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]);

  size_t Oldsize = 0; size_t Newsize = SystemComponents.Moleculesize[SelectedComponent];

  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  //Zhao's note: translation/rotation/reinsertion involves new + old states. Insertion/Deletion only has the new state.
  if(!Swap)
  {
    numberOfAtoms  += SystemComponents.Moleculesize[SelectedComponent];
    Oldsize         = SystemComponents.Moleculesize[SelectedComponent];
  }
  //std::vector<std::complex<double>>eik_x(numberOfAtoms * (kx_max + 1));
  //std::vector<std::complex<double>>eik_y(numberOfAtoms * (ky_max + 1));
  //std::vector<std::complex<double>>eik_z(numberOfAtoms * (kz_max + 1));
  //std::vector<std::complex<double>>eik_xy(numberOfAtoms);
  //std::vector<std::complex<double>>totalEik(SystemComponents.storedEik.size());
  size_t numberOfWaveVectors = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);
  SystemComponents.totalEik.resize(numberOfWaveVectors);
  //Create Temporary Struct for storing values//
  Atoms TempAtoms;
  TempAtoms.x         = (double*) malloc(numberOfAtoms * sizeof(double));
  TempAtoms.y         = (double*) malloc(numberOfAtoms * sizeof(double));
  TempAtoms.z         = (double*) malloc(numberOfAtoms * sizeof(double));
  TempAtoms.scale     = (double*) malloc(numberOfAtoms * sizeof(double));
  TempAtoms.charge    = (double*) malloc(numberOfAtoms * sizeof(double));
  TempAtoms.scaleCoul = (double*) malloc(numberOfAtoms * sizeof(double));
  TempAtoms.Type      = (size_t*) malloc(numberOfAtoms * sizeof(size_t));
  TempAtoms.MolID     = (size_t*) malloc(numberOfAtoms * sizeof(size_t));
  if(Swap)
  {
    //Copy the NEW values, first bead first, First bead stored in Old, at the first element//
    cudaMemcpy(TempAtoms.x,         Old.x,         sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.y,         Old.y,         sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.z,         Old.z,         sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scale,     Old.scale,     sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.charge,    Old.charge,    sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scaleCoul, Old.scaleCoul, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.Type,      Old.Type,      sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.MolID,     Old.MolID,     sizeof(size_t), cudaMemcpyDeviceToHost);
    //Copy the NEW Orientation for the selected Trial, stored in New//
    size_t chainsize  = SystemComponents.Moleculesize[SelectedComponent] - 1;
    size_t selectsize = chainsize * SelectedTrial;
    cudaMemcpy(&TempAtoms.x[1],         &New.x[selectsize],         chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.y[1],         &New.y[selectsize],         chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.z[1],         &New.z[selectsize],         chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scale[1],     &New.scale[selectsize],     chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.charge[1],    &New.charge[selectsize],    chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scaleCoul[1], &New.scaleCoul[selectsize], chainsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.Type[1],      &New.Type[selectsize],      chainsize * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.MolID[1],     &New.MolID[selectsize],     chainsize * sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  else //Translation/Rotation//
  {
    cudaMemcpy(TempAtoms.x,         Old.x,         Oldsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.y,         Old.y,         Oldsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.z,         Old.z,         Oldsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scale,     Old.scale,     Oldsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.charge,    Old.charge,    Oldsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.scaleCoul, Old.scaleCoul, Oldsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.Type,      Old.Type,      Oldsize * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(TempAtoms.MolID,     Old.MolID,     Oldsize * sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(&TempAtoms.x[Oldsize],         New.x,         Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.y[Oldsize],         New.y,         Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.z[Oldsize],         New.z,         Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scale[Oldsize],     New.scale,     Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.charge[Oldsize],    New.charge,    Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.scaleCoul[Oldsize], New.scaleCoul, Newsize * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.Type[Oldsize],      New.Type,      Newsize * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&TempAtoms.MolID[Oldsize],     New.MolID,     Newsize * sizeof(size_t), cudaMemcpyDeviceToHost);
  }
  for(size_t i=0; i < numberOfAtoms; i++) printf("TempAtoms: %.5f %.5f %.5f\n", TempAtoms.x[i], TempAtoms.y[i], TempAtoms.z[i]);
  double start = omp_get_wtime();  
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  size_t count=0;
  //Old//
  for(size_t posi=0; posi < Oldsize; ++posi)
  {
    //determine the component for i
    double pos[3] = {TempAtoms.x[posi], TempAtoms.y[posi], TempAtoms.z[posi]};
    SystemComponents.eik_x[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_y[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_z[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double s[3]; matrix_multiply_by_vector(Host_Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    SystemComponents.eik_x[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[0]), std::sin(s[0]));
    SystemComponents.eik_y[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[1]), std::sin(s[1]));
    SystemComponents.eik_z[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[2]), std::sin(s[2]));
    count++; 
  }
  //New//
  for(size_t posi=Oldsize; posi < Oldsize + Newsize; ++posi)
  {
    //determine the component for i
    double pos[3] = {TempAtoms.x[posi], TempAtoms.y[posi], TempAtoms.z[posi]};
    SystemComponents.eik_x[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_y[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    SystemComponents.eik_z[posi + 0 * numberOfAtoms] = std::complex<double>(1.0, 0.0);
    double s[3]; matrix_multiply_by_vector(Host_Box.InverseCell, pos, s); for(size_t j = 0; j < 3; j++) s[j]*=2*M_PI;
    SystemComponents.eik_x[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[0]), std::sin(s[0]));
    SystemComponents.eik_y[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[1]), std::sin(s[1]));
    SystemComponents.eik_z[posi + 1 * numberOfAtoms] = std::complex<double>(std::cos(s[2]), std::sin(s[2]));
    count++;
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      SystemComponents.eik_x[i + kx * numberOfAtoms] = SystemComponents.eik_x[i + (kx - 1) * numberOfAtoms] * SystemComponents.eik_x[i + 1 * numberOfAtoms];
    }
  }
  for(size_t ky = 2; ky <= ky_max; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      SystemComponents.eik_y[i + ky * numberOfAtoms] = SystemComponents.eik_y[i + (ky - 1) * numberOfAtoms] * SystemComponents.eik_y[i + 1 * numberOfAtoms];
    }
  }
  for(size_t kz = 2; kz <= kz_max; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      SystemComponents.eik_z[i + kz * numberOfAtoms] = SystemComponents.eik_z[i + (kz - 1) * numberOfAtoms] * SystemComponents.eik_z[i + 1 * numberOfAtoms];
    }
  }
 
  size_t nvec = 0;
  std::complex<double> cksum_old(0.0, 0.0);
  std::complex<double> cksum_new(0.0, 0.0);
  //for debugging
  for(std::make_signed_t<std::size_t> kx = 0; kx <= kx_max; ++kx)
  {
    double kvec_x[3]; for(size_t j = 0; j < 3; j++) kvec_x[j] = 2.0 * M_PI * static_cast<double>(kx) * ax[j];
    // Only positive kx are used, the negative kx are taken into account by the factor of two
    double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);

    for(std::make_signed_t<std::size_t> ky = -ky_max; ky <= ky_max; ++ky)
    {
      double kvec_y[3]; for(size_t j = 0; j < 3; j++) kvec_y[j] = 2.0 * M_PI * static_cast<double>(ky) * ay[j];
      // Precompute and store eik_x * eik_y outside the kz-loop
      // OLD //
      for(size_t i = 0; i != numberOfAtoms; ++i)
      {
        std::complex<double> eiky_temp = SystemComponents.eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eiky_temp.imag(ky>=0 ? eiky_temp.imag() : -eiky_temp.imag());
        SystemComponents.eik_xy[i] = SystemComponents.eik_x[i + numberOfAtoms * static_cast<size_t>(kx)] * eiky_temp;
      }

      for(std::make_signed_t<std::size_t> kz = -kz_max; kz <= kz_max; ++kz)
      {
        // Ommit kvec==0
        size_t ksqr = kx * kx + ky * ky + kz * kz;
        if((ksqr != 0) && (static_cast<double>(ksqr) < recip_cutoff))
        {
          double kvec_z[3]; for(size_t j = 0; j < 3; j++) kvec_z[j] = 2.0 * M_PI * static_cast<double>(kz) * az[j];
          cksum_old = std::complex<double>(0.0, 0.0); cksum_new = std::complex<double>(0.0, 0.0);
          count=0;
          //OLD//
          for(size_t posi=0; posi<Oldsize; ++posi)
          {
            std::complex<double> eikz_temp = SystemComponents.eik_z[posi + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            double charge  = TempAtoms.charge[posi];
            double scaling = TempAtoms.scaleCoul[posi];
            cksum_old     += scaling * charge * (SystemComponents.eik_xy[posi] * eikz_temp);
            count++;
          }
          //NEW//
          for(size_t posi=Oldsize; posi<Oldsize + Newsize; ++posi)
          {
            std::complex<double> eikz_temp = SystemComponents.eik_z[posi + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
            eikz_temp.imag(kz>=0 ? eikz_temp.imag() : -eikz_temp.imag());
            double charge  = TempAtoms.charge[posi];
            double scaling = TempAtoms.scaleCoul[posi];
            cksum_new     += scaling * charge * (SystemComponents.eik_xy[posi] * eikz_temp);
            count++;
          }
          //double rksq = (kvec_x + kvec_y + kvec_z).length_squared();
          double tempkvec[3] = {kvec_x[0]+kvec_y[0]+kvec_z[0], kvec_x[1]+kvec_y[1]+kvec_z[1], kvec_x[2]+kvec_y[2]+kvec_z[2]};
          double rksq = tempkvec[0]*tempkvec[0] + tempkvec[1]*tempkvec[1] + tempkvec[2]*tempkvec[2];
          double temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
          //std::complex<double> cksum_sub = SystemComponents.storedEik[nvec];
          //std::complex<double> cksum_add = SystemComponents.storedEik[nvec] + cksum_new - cksum_old;
          //double tempsum_add = temp * (cksum_add.real() * cksum_add.real() + cksum_add.imag() * cksum_add.imag());
          //double tempsum_sub = temp * (cksum_sub.real() * cksum_sub.real() + cksum_sub.imag() * cksum_sub.imag());
          //ewaldE += tempsum_add;
          //ewaldE -= tempsum_sub;
          std::complex<double> cksum_add = SystemComponents.storedEik[nvec] + cksum_new - cksum_old;
          std::complex<double> cksum = cksum_new - cksum_old;
          //double tempE;
          //tempE  += temp * std::norm(SystemComponents.storedEik[nvec] + cksum_new - cksum_old);
          //tempE  -= temp * std::norm(SystemComponents.storedEik[nvec]);
          ewaldE += temp * std::norm(SystemComponents.storedEik[nvec] + cksum_new - cksum_old);
          ewaldE -= temp * std::norm(SystemComponents.storedEik[nvec]);
          //ewaldE += tempE;
          SystemComponents.totalEik[nvec] = SystemComponents.storedEik[nvec] + cksum_new - cksum_old;
          //++nvec;
          //printf("GPU kx/ky/kz: %d %d %d temp: %.5f, tempE: %.5f\n", kx, ky, kz, temp, tempE);
          //printf("CPU kx/ky/kz: %d %d %d, new Vector: %.5f %.5f, cksum: %.5f %.5f, stored: %.5f %.5f\n", kx, ky, kz, cksum_add.real(), cksum_add.imag(), cksum.real(), cksum.imag(), SystemComponents.storedEik[nvec].real(), SystemComponents.storedEik[nvec].imag());
          //printf("CPU kx/ky/kz: %d %d %d, new Vector: %.5f %.5f, cknew: %.5f %.5f, ckold: %.5f %.5f\n", kx, ky, kz, cksum_add.real(), cksum_add.imag(), cksum_new.real(), cksum_new.imag(), cksum_old.real(), cksum_old.imag());
        } 
        ++nvec;
      }
    }
  }

  double end = omp_get_wtime();
  printf("CPU Fourier took: %.12f sec, Post-Fourier (CPU) energy is %.5f\n", end - start, ewaldE);

  ///////////////////////////////
  // Subtract exclusion-energy // Zhao's note: taking out the pairs of energies that belong to the same molecule
  ///////////////////////////////
  double SelfEOld = 0.0; double SelfENew = 0.0;
  count=0;
  //OLD//
  for(size_t posi=0; posi<Oldsize; posi++)
  {
    double charge  = TempAtoms.charge[posi];
    double scaling = TempAtoms.scaleCoul[posi];
    double factorA = charge * scaling;
    double posA[3] = {TempAtoms.x[posi], TempAtoms.y[posi], TempAtoms.z[posi]};
    for(size_t posj=posi+1; posj < Oldsize; posj++)
    {
      double charge  = TempAtoms.charge[posj];
      double scaling = TempAtoms.scaleCoul[posj];
      double factorB = charge * scaling;
      double posB[3] = {TempAtoms.x[posj], TempAtoms.y[posj], TempAtoms.z[posj]}; 
      double posvec[3] = {posA[0]-posB[0], posA[1]-posB[1], posA[2]-posB[2]};
      PBC_CPU(posvec, Host_Box.Cell, Host_Box.InverseCell, Host_Box.Cubic);
      double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      double r = std::sqrt(rr_dot);
      SelfEOld      += FF.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
    }
  }
  //NEW//
  for(size_t posi=Oldsize; posi!=Oldsize + Newsize; posi++)
  {
    double charge  = TempAtoms.charge[posi];
    double scaling = TempAtoms.scaleCoul[posi];
    double factorA = charge * scaling;
    double posA[3] = {TempAtoms.x[posi], TempAtoms.y[posi], TempAtoms.z[posi]};
    for(size_t posj=posi+1; posj != Oldsize + Newsize; posj++)
    {
      double charge  = TempAtoms.charge[posj];
      double scaling = TempAtoms.scaleCoul[posj];
      double factorB = charge * scaling;
      double posB[3] = {TempAtoms.x[posj], TempAtoms.y[posj], TempAtoms.z[posj]};
      double posvec[3] = {posA[0]-posB[0], posA[1]-posB[1], posA[2]-posB[2]};
      PBC_CPU(posvec, Host_Box.Cell, Host_Box.InverseCell, Host_Box.Cubic);
      double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      double r = std::sqrt(rr_dot);
      SelfENew      += FF.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
    }
  }
  ewaldE -= SelfENew - SelfEOld;

  //////////////////////////
  // Subtract self-energy //
  //////////////////////////
  double prefactor_self = FF.Prefactor * alpha / std::sqrt(M_PI);
  double SelfExcludeOld = 0.0; double SelfExcludeNew = 0.0;
  //OLD//
  for(size_t i = 0; i != Oldsize; ++i)
  {
    double charge   = TempAtoms.charge[i];
    double scale    = TempAtoms.scale[i];
    SelfExcludeOld += prefactor_self * charge * charge * scale * scale;
  }
  //NEW//
  size_t j_count = 0;
  for(size_t i = Oldsize; i != Oldsize + Newsize; ++i)
  {
    double charge   = TempAtoms.charge[i];
    double scale    = TempAtoms.scale[i];
    SelfExcludeNew += prefactor_self * charge * charge * scale * scale;
    j_count++;
  }
  ewaldE -= SelfExcludeNew - SelfExcludeOld;

  return ewaldE;
}
 

__device__ void GPU_matrix_multiply_by_vector(double* a, double3 b, double3 *c) //3x3(9*1) matrix (a) times 3x1(3*1) vector (b), a*b=c//
{
  double3 temp;
  temp.x=a[0*3+0]*b.x+a[1*3+0]*b.y+a[2*3+0]*b.z;
  temp.y=a[0*3+1]*b.x+a[1*3+1]*b.y+a[2*3+1]*b.z;
  temp.z=a[0*3+2]*b.x+a[1*3+2]*b.y+a[2*3+2]*b.z;
  *c    = temp;
}

__device__ Complex multiply(Complex a, Complex b) //a*b = c for complex numbers//
{
  Complex c;
  c.real = a.real*b.real - a.imag*b.imag;
  c.imag = a.real*b.imag + a.imag*b.real;
  return c;
}

__device__ void Initialize_Vectors(Boxsize Box, size_t Oldsize, size_t Newsize, Atoms Old, size_t numberOfAtoms, int3 kmax)
{
  int kx_max = kmax.x;
  int ky_max = kmax.y;
  int kz_max = kmax.z;
  //Old//
  Complex tempcomplex; tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
  for(size_t posi=0; posi < Oldsize; ++posi)
  {
    tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
    double3 pos; pos.x = Old.x[posi]; pos.y = Old.y[posi]; pos.z = Old.z[posi];
    Box.eik_x[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_y[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_z[posi + 0 * numberOfAtoms] = tempcomplex;
    double3 s; GPU_matrix_multiply_by_vector(Box.InverseCell, pos, &s); s.x*=2*M_PI; s.y*=2*M_PI; s.z*=2*M_PI;
    tempcomplex.real = std::cos(s.x); tempcomplex.imag = std::sin(s.x); Box.eik_x[posi + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.y); tempcomplex.imag = std::sin(s.y); Box.eik_y[posi + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.z); tempcomplex.imag = std::sin(s.z); Box.eik_z[posi + 1 * numberOfAtoms] = tempcomplex;
  }
  //New//
  for(size_t posi=Oldsize; posi < Oldsize + Newsize; ++posi)
  {
    tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
    double3 pos; pos.x = Old.x[posi]; pos.y = Old.y[posi]; pos.z = Old.z[posi];
    Box.eik_x[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_y[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_z[posi + 0 * numberOfAtoms] = tempcomplex;
    double3 s ; GPU_matrix_multiply_by_vector(Box.InverseCell, pos, &s); s.x*=2*M_PI; s.y*=2*M_PI; s.z*=2*M_PI;
    tempcomplex.real = std::cos(s.x); tempcomplex.imag = std::sin(s.x); Box.eik_x[posi + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.y); tempcomplex.imag = std::sin(s.y); Box.eik_y[posi + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.z); tempcomplex.imag = std::sin(s.z); Box.eik_z[posi + 1 * numberOfAtoms] = tempcomplex;
  }
  // Calculate remaining positive kx, ky and kz by recurrence
  for(size_t kx = 2; kx <= kx_max; ++kx)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      Box.eik_x[i + kx * numberOfAtoms] = multiply(Box.eik_x[i + (kx - 1) * numberOfAtoms], Box.eik_x[i + 1 * numberOfAtoms]);
    }
  }
  for(size_t ky = 2; ky <= ky_max; ++ky)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      Box.eik_y[i + ky * numberOfAtoms] = multiply(Box.eik_y[i + (ky - 1) * numberOfAtoms], Box.eik_y[i + 1 * numberOfAtoms]);
    }
  }
  for(size_t kz = 2; kz <= kz_max; ++kz)
  {
    for(size_t i = 0; i != numberOfAtoms; ++i)
    {
      Box.eik_z[i + kz * numberOfAtoms] = multiply(Box.eik_z[i + (kz - 1) * numberOfAtoms], Box.eik_z[i + 1 * numberOfAtoms]);
    }
  }
}

__global__ void Initialize_EwaldVector_General(Boxsize Box, int3 kmax, Atoms* d_a, Atoms New, Atoms Old, size_t Oldsize, size_t Newsize, size_t SelectedComponent, size_t Location, size_t chainsize, size_t numberOfAtoms, int MoveType)
{
  //Zhao's note: need to think about changing this boolean to switch//
  if(MoveType == 0) // Translation/Rotation //
  { 
    //For Translation/Rotation, the Old positions are already in the Old struct, just need to put the New positions into Old, after the Old positions//
    for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
    {
      Old.x[i]             = New.x[i - Oldsize];
      Old.y[i]             = New.y[i - Oldsize];
      Old.z[i]             = New.z[i - Oldsize];
      Old.scale[i]         = New.scale[i - Oldsize];
      Old.charge[i]        = New.charge[i - Oldsize];
      Old.scaleCoul[i]     = New.scaleCoul[i - Oldsize];
      //printf("Old Trans: %zu, %.5f %.5f %.5f\n", i, Old.x[i], Old.y[i], Old.z[i]);
    }
  }
  else if(MoveType == 1) // Insertion //
  {
    //Put the trial orientations in New to Old, right after the first bead position//
    for(size_t i = 0; i < chainsize; i++)
    {
      Old.x[i + 1]         = New.x[Location * chainsize + i];
      Old.y[i + 1]         = New.y[Location * chainsize + i];
      Old.z[i + 1]         = New.z[Location * chainsize + i];
      Old.scale[i + 1]     = New.scale[Location * chainsize + i];
      Old.charge[i + 1]    = New.charge[Location * chainsize + i];
      Old.scaleCoul[i + 1] = New.scaleCoul[Location * chainsize + i];
      //printf("New Insertion: %lu, %.5f %.5f %.5f\n", i, Old.x[i+1], Old.y[i+1], Old.z[i+1]);
    }
  }
  else if(MoveType == 2) // Deletion //
  {
    for(size_t i = 0; i < Oldsize; i++)
    {
      // For deletion, Location = UpdateLocation, see Deletion Move //
      Old.x[i]             = d_a[SelectedComponent].x[Location + i];
      Old.y[i]             = d_a[SelectedComponent].y[Location + i];
      Old.z[i]             = d_a[SelectedComponent].z[Location + i];
      Old.scale[i]         = d_a[SelectedComponent].scale[Location + i];
      Old.charge[i]        = d_a[SelectedComponent].charge[Location + i];
      Old.scaleCoul[i]     = d_a[SelectedComponent].scaleCoul[Location + i];
      //printf("Old Deletion: %lu, %.5f %.5f %.5f\n", i, Old.x[i], Old.y[i], Old.z[i]);
    }
  }
  Initialize_Vectors(Box, Oldsize, Newsize, Old, numberOfAtoms, kmax);
}

__device__ double length_squared(double3 a)
{
  return a.x * a.x + a.y * a.y + a.z * a.z;
}
__device__ double ComplexNorm(Complex a)
{
  return a.real * a.real + a.imag * a.imag;
}

__device__ double3 a_mult_double3(double a, double3 b)
{
  double3 c;
  c.x = a * b.x; c.y = a * b.y; c.z = a * b.z; 
  return c;
}

__device__ double3 double3_add_3(double3 a, double3 b, double3 c)
{
  double3 d;
  d.x = a.x + b.x + c.x; d.y = a.y + b.y + c.y; d.z = a.z + b.z + c.z;
  return d;
}

__global__ void Fourier_Ewald_Diff(Boxsize Box, Atoms Old, double alpha_squared, double prefactor, int3 kmax, double recip_cutoff, size_t Oldsize, size_t Newsize, double* Blocksum)
{
  extern __shared__ double sdata[]; //shared memory for partial sum//
  size_t kxyz           = blockIdx.x * blockDim.x + threadIdx.x;
  int    cache_id       = threadIdx.x;
  size_t i_within_block = kxyz - blockIdx.x * blockDim.x; //for recording the position of the thread within a block
  double tempE = 0.0;
  size_t    kx_max  = kmax.x; 
  size_t    ky_max  = kmax.y; 
  size_t    kz_max  = kmax.z;
  size_t    nvec    = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);
  if(kxyz < nvec)
  {
    Box.totalEik[kxyz] = Box.storedEik[kxyz];
    sdata[i_within_block] = 0.0;
    int       kz      = kxyz%(2 * kz_max + 1) - kz_max;
    int       kxy     = kxyz/(2 * kz_max + 1);
    int       kx      = kxy /(2 * ky_max + 1); 
    int       ky      = kxy %(2 * ky_max + 1) - ky_max;
    size_t    ksqr    = kx * kx + ky * ky + kz * kz;
  
    if((ksqr != 0) && (static_cast<double>(ksqr) < recip_cutoff))
    {
      double3 ax; ax.x = Box.InverseCell[0]; ax.y = Box.InverseCell[3]; ax.z = Box.InverseCell[6];
      double3 ay; ay.x = Box.InverseCell[1]; ay.y = Box.InverseCell[4]; ay.z = Box.InverseCell[7];
      double3 az; az.x = Box.InverseCell[2]; az.y = Box.InverseCell[5]; az.z = Box.InverseCell[8];
      size_t numberOfAtoms = Oldsize + Newsize;
      Complex cksum_old; cksum_old.real = 0.0; cksum_old.imag = 0.0;
      Complex cksum_new; cksum_new.real = 0.0; cksum_new.imag = 0.0;
      double3 kvec_x; kvec_x = a_mult_double3(2.0 * M_PI * (double) kx, ax);
      double3 kvec_y; kvec_y = a_mult_double3(2.0 * M_PI * (double) ky, ay);
      double3 kvec_z; kvec_z = a_mult_double3(2.0 * M_PI * (double) kz, az);
      double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
      //OLD//
      for(size_t i=0; i<Oldsize + Newsize; ++i)
      {
        Complex eik_temp  = Box.eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eik_temp.imag     = ky>=0 ? eik_temp.imag : -eik_temp.imag;
        Complex eik_xy    = multiply(Box.eik_x[i + numberOfAtoms * static_cast<size_t>(kx)], eik_temp);

        eik_temp          = Box.eik_z[i + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
        eik_temp.imag     = kz>=0 ? eik_temp.imag : -eik_temp.imag;
        double charge     = Old.charge[i];
        double scaling    = Old.scaleCoul[i];
        Complex tempi     = multiply(eik_xy, eik_temp);
        if(i < Oldsize)
        {
          cksum_old.real += scaling * charge * tempi.real;
          cksum_old.imag += scaling * charge * tempi.imag;
        }
        else
        {
          cksum_new.real += scaling * charge * tempi.real;
          cksum_new.imag += scaling * charge * tempi.imag;
        }
      }
      double3 tempkvec   = double3_add_3(kvec_x, kvec_y, kvec_z);
      double  rksq       = length_squared(tempkvec);
      double  temp       = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
      Complex newV;
      newV.real          = Box.storedEik[kxyz].real + cksum_new.real - cksum_old.real;
      newV.imag          = Box.storedEik[kxyz].imag + cksum_new.imag - cksum_old.imag;
      tempE             += temp * ComplexNorm(newV);
      tempE             -= temp * ComplexNorm(Box.storedEik[kxyz]);
      Box.totalEik[kxyz] = newV;
    }
  }
  sdata[i_within_block] = tempE;
  __syncthreads();
  //Partial block sum//
  int i=blockDim.x / 2;
  while(i != 0)
  {
    if(cache_id < i) {sdata[cache_id] += sdata[cache_id + i];}
    __syncthreads();
    i /= 2;
  }
  if(cache_id == 0) {Blocksum[blockIdx.x] = sdata[0];}
}

static inline void Setup_threadblock_EW(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  size_t value = arraysize;
  if(value >= 128) value = 128;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  *Nthread = value;
  *Nblock = blockValue;
}

__global__ void Initialize_EwaldVector_Reinsertion(Boxsize Box, int3 kmax, double* tempx, double* tempy, double* tempz, Atoms* d_a, Atoms Old, size_t Oldsize, size_t Newsize, size_t realpos, size_t numberOfAtoms, size_t SelectedComponent)
{
  for(size_t i = 0; i < Oldsize; i++)
  {
    Old.x[i]         = d_a[SelectedComponent].x[realpos + i];
    Old.y[i]         = d_a[SelectedComponent].y[realpos + i];
    Old.z[i]         = d_a[SelectedComponent].z[realpos + i];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i];
  }
  //Reinsertion New Positions stored in three arrays, other data are the same as the Old molecule information in d_a//
  for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
  {
    Old.x[i]         = tempx[i - Oldsize];
    Old.y[i]         = tempy[i - Oldsize];
    Old.z[i]         = tempz[i - Oldsize];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i - Oldsize];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i - Oldsize];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i - Oldsize];
  }
  Initialize_Vectors(Box, Oldsize, Newsize, Old, numberOfAtoms, kmax);
}

__global__ void Update_Ewald_Stored(Boxsize Box, size_t nvec)
{
  size_t i         = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nvec) Box.storedEik[i] = Box.totalEik[i];
}
void Update_Ewald_Vector(Boxsize& Box, bool CPU, Components& SystemComponents)
{
  if(CPU) //Update on the CPU//
  {
    SystemComponents.storedEik = SystemComponents.totalEik;
  }
  else    //Update on the GPU//
  {
    size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
    size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_EW(numberOfWaveVectors, &Nblock, &Nthread);
    Update_Ewald_Stored<<<Nblock, Nthread>>>(Box, numberOfWaveVectors);
  }
}

//Zhao's note: Currently this only works for INSERTION, not even tested with Deletion//
double GPU_EwaldDifference_Reinsertion(Boxsize& Box, Atoms*& d_a, Atoms& Old, double* tempx, double* tempy, double* tempz, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, size_t UpdateLocation)
{
  if(FF.noCharges) return 0.0;
  double alpha = FF.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = FF.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  size_t Oldsize = 0; size_t Newsize = numberOfAtoms;
  //Zhao's note: translation/rotation/reinsertion involves new + old states. Insertion/Deletion only has the new state.
  Oldsize         = SystemComponents.Moleculesize[SelectedComponent];
  numberOfAtoms  += Oldsize;
  //Copy the NEW Orientation for the selected Trial, stored in New//
  size_t chainsize  = SystemComponents.Moleculesize[SelectedComponent];

  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  Initialize_EwaldVector_Reinsertion<<<1,1>>>(Box, Box.kmax, tempx, tempy, tempz, d_a, Old, Oldsize, Newsize, UpdateLocation, numberOfAtoms, SelectedComponent);

  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_EW(numberOfWaveVectors, &Nblock, &Nthread);
  Fourier_Ewald_Diff<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Old, alpha_squared, prefactor, Box.kmax, Box.ReciprocalCutOff, Oldsize, Newsize, Blocksum);
  double Host_sum[Nblock]; double tot = 0.0;
  cudaMemcpy(Host_sum, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++){tot += Host_sum[i];}
  return tot;
}

//Zhao's note: Currently this only works for INSERTION, not even tested with Deletion//
double GPU_EwaldDifference_General(Boxsize& Box, Atoms*& d_a, Atoms& New, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location)
{
  if(FF.noCharges) return 0.0;
  //cudaDeviceSynchronize();
  double start = omp_get_wtime();
  double alpha = FF.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = FF.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t Oldsize = 0; size_t Newsize = 0; size_t chainsize = 0;
  switch(MoveType)
  {
    case 0: // Translation/Rotation Move //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent];
      break;
    }
    case 1: // Insertion //
    {
      Oldsize   = 0;
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case 2: // Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1; 
      break;
    }
    case 3: // Reinsertion //
    {
      throw std::runtime_error("Use the Special Function for Reinsertion");
      break;
    }
  }
  //printf("Old: %zu, New: %zu, chain: %zu\n", Oldsize, Newsize, chainsize);
  size_t numberOfAtoms = Oldsize + Newsize;

  Initialize_EwaldVector_General<<<1,1>>>(Box, Box.kmax, d_a, New, Old, Oldsize, Newsize, SelectedComponent, Location, chainsize, numberOfAtoms, MoveType); //checkCUDAError("error Initializing Ewald Vectors");

  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_EW(numberOfWaveVectors, &Nblock, &Nthread);
  Fourier_Ewald_Diff<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Old, alpha_squared, prefactor, Box.kmax, Box.ReciprocalCutOff, Oldsize, Newsize, Blocksum); //checkCUDAError("error Doing Fourier");
  double Host_sum[Nblock]; double tot = 0.0;
  cudaMemcpy(Host_sum, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++){tot += Host_sum[i];}
  //Zhao's note: when adding fractional molecules, this might not be correct//
  if(SystemComponents.rigid[SelectedComponent])
  {
    if(MoveType == 1) // Insertion //
    {
      tot -= SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent];
    }
    else if(MoveType == 2) // Deletion //
    {
      tot += SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent];
    }
  }
  //printf("GPU Fourier Energy: %.12f\n", tot);
  //cudaDeviceSynchronize();
  double end = omp_get_wtime();
  //printf("GPU Fourier took %.12f sec\n", end - start);
  return tot;
}
