#include <complex>
#include "VDW_Coulomb.cuh"
#include "Ewald_Energy_Functions.h"
#include "TailCorrection_Energy_Functions.h"
#include <cuda_fp16.h>
#include <omp.h>
/*
inline void VDW_CPU(const double* FFarg, const double rr_dot, const double scaling, double* result) //Lennard-Jones 12-6
{
  double arg1 = 4.0 * FFarg[0];      // epsilon //
  double arg2 = FFarg[1] * FFarg[1]; //  sigma  //
  double arg3 = FFarg[3]; //the third element of the 3rd dimension of the array, shift
  double temp = (rr_dot / arg2);
  double temp3 = temp * temp * temp;
  double rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
  double rri6 = rri3 * rri3;
  double term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
  double dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
  result[0] = scaling * term; result[1] = scaling < 1.0 ? term + (1.0 - scaling) * dlambda_term : 0.0;
}
*/
inline void CoulombReal_CPU(const ForceField FF, const double chargeA, const double chargeB, const double r, const double scaling, double* result, double prefactor, double alpha) //energy = -q1*q2/r
{
  double term      = chargeA * chargeB * std::erfc(alpha * r);
         result[0] = prefactor * scaling * term / r;
}
/*
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
*/
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
  double VDW_energy   = 0.0; double Coul_energy = 0.0;
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
                VDW_energy   += 0.5*result[0];
                if(std::abs(result[0]) > 10000) printf("Very High Energy (VDW), comps: %zu, %zu, MolID: %zu %zu, Atom: %zu %zu, E: %.5f\n", compi, compj, MoleculeID, MoleculeIDB, i, j, result[0]);
              }
              if (!FF.noCharges && rr_dot < FF.CutOffCoul)
              {
                const double r = sqrt(rr_dot);
                const double scalingCoul = scalingCoulombA * scalingCoulombB;
                double resultCoul[2] = {0.0, 0.0};
                CoulombReal_CPU(FF, chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
                Total_energy += 0.5*resultCoul[0]; //prefactor merged in the CoulombReal function
                Coul_energy  += 0.5*resultCoul[0];
                if(std::abs(result[0]) > 10000) printf("Very High Energy (Coul), comps: %zu, %zu, MolID: %zu %zu, Atom: %zu %zu, E: %.5f\n", compi, compj, MoleculeID, MoleculeIDB, i, j, resultCoul[0]);
              }
            }
          }
        }
      }
    }  
  }
  //printf("%zu interactions, within cutoff: %zu, energy: %.10f\n", count, Total_energy, cutoff_count);
  printf("CPU (one Thread) Total Energy: %.5f, VDW Energy: %.5f, Coulomb Energy: %.5f\n", Total_energy, VDW_energy, Coul_energy);
  return Total_energy;
}

__device__ void setScaleGPU(double lambda, double& scalingVDW, double& scalingCoulomb)
{
  int CFCType = 0; //0: Linear with pow(lambda,5); 1: Non-Linear (RASPA3 and Brick code)//
  switch(CFCType)
  {
    case 0:
    {
      scalingVDW     = lambda;
      scalingCoulomb = std::pow(lambda, 5);
      break;
    }
    case 1:
    {
      scalingVDW     = lambda < 0.5 ? 2.0 * lambda : 1.0;
      scalingCoulomb = lambda < 0.5 ? 0.0 : 2.0 * (lambda - 0.5);
      break;
    }
  }
}

////////////////////////////// GPU CODE //////////////////////////

__device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result) //Lennard-Jones 12-6
{
  // FFarg[0] = epsilon; FFarg[1] = sigma //
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

__device__ void CoulombReal(const ForceField FF, const double chargeA, const double chargeB, const double r, const double scaling, double* result, double prefactor, double alpha) //energy = -q1*q2/r
{
  double term      = chargeA * chargeB * std::erfc(alpha * r);
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

__global__ void one_thread_GPU_test(Boxsize Box, Atoms* System, ForceField FF, double* xxx)
{
  bool DEBUG=false;
  //Zhao's note: added temp_xxx values for checking individual energy for each molecule//
  double temp_energy = 0.0; double temp_firstbead = 0.0; double temp_chain = 0.0; int temp_count = -1;
  double Total_energy = 0.0; size_t count = 0; size_t cutoff_count=0;
  double VDW_energy = 0.0; double Coul_energy = 0.0;
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
                VDW_energy   += 0.5*result[0];
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
                CoulombReal(FF, chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
                Total_energy += 0.5*resultCoul[0]; //prefactor merged in the CoulombReal function
                Coul_energy  += 0.5*resultCoul[0];
              }
            }
          }
        }
      }
    }
  }
  if(DEBUG) printf("For Molecule 5, energy: %.10f, firstbead: %.10f, chain: %.10f\n", temp_energy, temp_firstbead, temp_chain);
  xxx[0] = Total_energy;
  printf("GPU (one Thread) Total Energy: %.5f, VDW Energy: %.5f, Coulomb Energy: %.5f\n", Total_energy, VDW_energy, Coul_energy);
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
      //printf("typeA: %lu, typeB: %lu, FF.size: %lu, row: %lu\n", typeA, typeB, FF.size, row);
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, scaling, result); 
      if(result[0] > FF.OverlapCriteria){ flag[trial]=true; Blockflag = true; }
      if(rr_dot < 0.01) {flag[trial]=true; Blockflag = true; } //DistanceCheck//
      tempy += result[0];
    }

    if (FF.VDWRealBias && !FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = NewMol.charge[j];
      const double scalingCoulombB = NewMol.scaleCoul[j];
      const double r = sqrt(rr_dot);
      const double scalingCoul = scalingCoulombA * scalingCoulombB;
      double resultCoul[2] = {0.0, 0.0};
      CoulombReal(FF, chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
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
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

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
          //std::complex<double> cksum_add = SystemComponents.storedEik[nvec] + cksum_new - cksum_old;
          //std::complex<double> cksum = cksum_new - cksum_old;
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
      SelfEOld      += Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
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
      SelfENew      += Box.Prefactor * factorA * factorB * std::erf(alpha * r) / r;
    }
  }
  ewaldE -= SelfENew - SelfEOld;

  //////////////////////////
  // Subtract self-energy //
  //////////////////////////
  double prefactor_self = Box.Prefactor * alpha / std::sqrt(M_PI);
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
 
__global__ void Energy_difference_LambdaChange(Boxsize Box, Atoms* System, Atoms Mol, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t chainsize, size_t Threadsize, bool* flag, double2 newScale)
{
  //Zhao's note: This function is for CBCF Lambda change move, it is supposed to be faster than using the functions for translation/rotations//
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
  if(!((MoleculeID == Mol.MolID[0]) &&(comp == ComponentID))) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    double posvec[3]; posvec[0] = Component.x[posi] - Mol.x[j]; posvec[1] = Component.y[posi] - Mol.y[j]; posvec[2] = Component.z[posi] - Mol.z[j];
    PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
    double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
    double result[2];
    if(rr_dot < FF.CutOffVDW)
    {
      const size_t typeB = Mol.Type[j];
      const double OldscaleB = Mol.scale[j];
      const double NewscaleB = newScale.x; //VDW part of the scale//
      const double Oldscaling = scaleA * OldscaleB;
      const double Newscaling = scaleA * NewscaleB;
      const size_t row = typeA*FF.size+typeB;
      const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
      VDW(FFarg, rr_dot, Oldscaling, result);
      tempy  -= result[0];
      tempdU -= result[1];
      VDW(FFarg, rr_dot, Newscaling, result);
      if(result[0] > FF.OverlapCriteria){ Blockflag = true; }
      tempy  += result[0];
      tempdU += result[1];
    }
    if (!FF.noCharges && rr_dot < FF.CutOffCoul)
    {
      const double chargeB = Mol.charge[j];
      const double OldscalingCoulombB = Mol.scaleCoul[j];
      const double NewscalingCoulombB = newScale.y; //Coulomb part of the scale//
      const double r = sqrt(rr_dot);
      const double OldscalingCoul = scalingCoulombA * OldscalingCoulombB;
      const double NewscalingCoul = scalingCoulombA * NewscalingCoulombB;
      CoulombReal(FF, chargeA, chargeB, r, OldscalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy -= result[0]; 
      CoulombReal(FF, chargeA, chargeB, r, NewscalingCoul, result, Box.Prefactor, Box.Alpha);
      tempy += result[0];
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

__device__ void VDWCoulEnergy_Total(Boxsize Box, Atoms ComponentA, Atoms ComponentB, size_t Aij, size_t Bij, ForceField FF, bool* flag, bool& Blockflag, double& tempy, size_t NA, size_t NB, bool UseOffset)
{
  for(size_t i = 0; i < NA; i++)
  {
          size_t OffsetA         = 0;
          size_t posi            = i + Aij;
          if(UseOffset) OffsetA  = ComponentA.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
    //Zhao's note: add protection here//
    if(posi >= ComponentA.size) continue;
    const double scaleA          = ComponentA.scale[posi];
    const double chargeA         = ComponentA.charge[posi];
    const double scalingCoulombA = ComponentA.scaleCoul[posi];
    const size_t typeA           = ComponentA.Type[posi];

    const double PosA[3] = {ComponentA.x[posi + OffsetA], ComponentA.y[posi + OffsetA], ComponentA.z[posi + OffsetA]};
    for(size_t j = 0; j < NB; j++)
    {
            size_t OffsetB         = 0;
            size_t posj            = j + Bij;
            if(UseOffset) OffsetB  = ComponentB.Allocate_size / 2; //Read the positions shifted to the later half of the storage//
      //Zhao's note: add protection here//
      //if(posj >= ComponentB.size) continue;
      const double scaleB          = ComponentB.scale[posj];
      const double chargeB         = ComponentB.charge[posj];
      const double scalingCoulombB = ComponentB.scaleCoul[posj];
      const size_t typeB           = ComponentB.Type[posj];
      //if(j == 6) printf("PAIR CHECK: i: %lu, j: %lu, MoleculeID: %lu, NewMol.MolID: %lu\n", i,j,MoleculeID, NewMol.MolID[0]);
      const double PosB[3] = {ComponentB.x[posj + OffsetB], ComponentB.y[posj + OffsetB], ComponentB.z[posj + OffsetB]};
      double posvec[3] = {PosA[0] - PosB[0], PosA[1] - PosB[1], PosA[2] - PosB[2]};
      //printf("thread: %lu, i:%lu, j:%lu, comp: %lu, posi: %lu\n", ij,i,j,comp, posi);
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      if(rr_dot < FF.CutOffVDW)
      {
        double result[2] = {0.0, 0.0};
        const double scaling = scaleA * scaleB;
        const size_t row = typeA*FF.size+typeB;
        //printf("typeA: %lu, typeB: %lu, FF.size: %lu, row: %lu\n", typeA, typeB, FF.size, row);
        const double FFarg[4] = {FF.epsilon[row], FF.sigma[row], FF.z[row], FF.shift[row]};
        VDW(FFarg, rr_dot, scaling, result);
        if(result[0] > FF.OverlapCriteria){ flag[0]=true; Blockflag = true;}
        if(rr_dot < 0.01) { flag[0]=true; Blockflag = true; } //DistanceCheck//
        if(result[0] > FF.OverlapCriteria || rr_dot < 0.01) printf("OVERLAP IN KERNEL!\n");
        tempy += result[0];
      }
      //Coulombic (REAL)//
      if (!FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double r = sqrt(rr_dot);
        const double scalingCoul = scalingCoulombA * scalingCoulombB;
        double resultCoul[2] = {0.0, 0.0};
        CoulombReal(FF, chargeA, chargeB, r, scalingCoul, resultCoul, Box.Prefactor, Box.Alpha);
        tempy += resultCoul[0]; //prefactor merged in the CoulombReal function
      }
    }
  }
}

__global__ void TotalVDWCoul(Boxsize Box, Atoms* System, ForceField FF, double* Blocksum, bool* flag, size_t totalthreads, size_t Host_threads, size_t NAds, size_t NFrameworkAtomsPerThread, bool HostHost, bool UseOffset)
{
  extern __shared__ double sdata[]; //shared memory for partial sum//
  int cache_id = threadIdx.x; 
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;
 
  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;
  sdata[ij_within_block] = 0.0;

  //Initialize Blocksum//
  if(cache_id == 0) Blocksum[blockIdx.x] = 0.0;
  __shared__ bool Blockflag = false;
 
  if(total_ij < totalthreads)
  {
    size_t totalsize = 0;
    const size_t NumberComp = 2; //Zhao's note: need to change here for multicomponent
    //Aij and Bij indicate the starting positions for the objects in the pairwise interaction//
    size_t Aij   = 0; size_t Bij   = 0;
    size_t MolA  = 0; size_t MolB  = 0;
    size_t compA = 0; size_t compB = 0;
    size_t NA    = 0; size_t NB    = 0;
    if(total_ij < Host_threads) //This thread belongs to the Host_threads//
    {
      MolA = 0;
      Aij  = total_ij / NAds * NFrameworkAtomsPerThread;
      MolB = total_ij % NAds;
      NA  = NFrameworkAtomsPerThread;
      if(total_ij == Host_threads - 1) 
        if(Host_threads % NFrameworkAtomsPerThread != 0)
          NA = Host_threads % NFrameworkAtomsPerThread; 
      if(!HostHost) compB = 1; //If we do not consider host-host, start with host-guest
      //Here we need to determine the Molecule ID and which component the molecule belongs to//
      
      for(size_t ijk = compB; ijk < NumberComp; ijk++)
      {
        size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
        totalsize     += Mol_ijk;
        if(MolB >= totalsize)
        {
          compB++;
          MolB -= Mol_ijk;
        }
      }
      NB = System[compB].Molsize;
      Bij = MolB * NB;
    }
    else //Adsorbate-Adsorbate//
    { 
      compA = 1; compB = 1;
      size_t Ads_i = total_ij - Host_threads;
      //https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
      MolA = NAds - 2 - std::floor(std::sqrt(-8*Ads_i + 4*NAds*(NAds-1)-7)/2.0 - 0.5);
      MolB = Ads_i + MolA + 1 - NAds*(NAds-1)/2 + (NAds-MolA)*((NAds- MolA)-1)/2;
      totalsize = 0;
      //Determine the Molecule IDs and the component of MolA and MolB//
      for(size_t ijk = 1; ijk < NumberComp; ijk++)
      {
        size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
        totalsize     += Mol_ijk;
        if(MolA >= totalsize)
        {
          compA++;
          MolA -= Mol_ijk;
        }
      }
      NA = System[compA].Molsize;
      totalsize = 0;
      for(size_t ijk = 1; ijk < NumberComp; ijk++)
      {
        size_t Mol_ijk = System[ijk].size / System[ijk].Molsize;
        totalsize     += Mol_ijk;
        if(MolB >= totalsize)
        {
          compB++;
          MolB -= Mol_ijk;
        }
      }
      NB = System[compB].Molsize;
      Aij = MolA * NA; Bij = MolB * NB;
    }
    //printf("Thread: %lu, compA: %lu, compB: %lu, MolA: %lu, MolB: %lu, Aij: %lu, Bij: %lu, Molsizes: %lu %lu\n", total_ij, compA, compB, MolA, MolB, Aij, Bij, System[0].Molsize, System[1].Molsize);

    sdata[ij_within_block] = 0.0;
    //Initialize Blocksum//
    if(cache_id == 0) Blocksum[blockIdx.x] = 0.0;

    // Manually fusing/collapsing the loop //

    const Atoms ComponentA=System[compA];
    const Atoms ComponentB=System[compB];
    double tempy = 0.0;
    VDWCoulEnergy_Total(Box, ComponentA, ComponentB, Aij, Bij, FF, flag, Blockflag, tempy, NA, NB, UseOffset);
    sdata[ij_within_block] = tempy;
    //printf("ThreadID: %lu, HostThread: %lu, compA: %lu, compB: %lu, Aij: %lu, Bij: %lu, NA: %lu, NB: %lu, tempy: %.5f\n", total_ij, Host_threads, compA, compB, Aij, Bij, NA, NB, tempy);
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
  else
    flag[0] = true;
}

static inline void Setup_threadblock_VDWCoul(size_t arraysize, size_t *Nblock, size_t *Nthread)
{
  size_t value = arraysize;
  if(value >= 128) value = 128;
  double ratio = (double)arraysize/value;
  size_t blockValue = ceil(ratio);
  if(blockValue == 0) blockValue++;
  //Zhao's note: Default thread should always be 64, 128, 256, 512, ...
  // This is because we are using partial sums, if arraysize is smaller than defaultthread, we need to make sure that
  //while Nthread is dividing by 2, it does not generate ODD NUMBER (for example, 5/2 = 2, then element 5 will be ignored)//
  *Nthread = 128;
  *Nblock = blockValue;
}

//Zhao's note: here the totMol does not consider framework atoms, ONLY Adsorbates//
double Total_VDW_Coulomb_Energy(Simulations& Sim, ForceField FF, size_t totMol, size_t Host_threads, size_t Guest_threads, size_t NFrameworkAtomsPerThread, bool ConsiderHostHost, bool UseOffset)
{
  if(Host_threads + Guest_threads == 0) return 0.0;
  double VDWRealE = 0.0;
  size_t Nblock = 0; size_t Nthread = 0;
  Setup_threadblock_VDWCoul(Host_threads + Guest_threads, &Nblock, &Nthread);
  if(Nblock > Sim.Nblocks)
  {
    printf("More blocks for block sum is needed\n");
    cudaMalloc(&Sim.Blocksum, Nblock * sizeof(double));
  }
  //Calculate the energy of the new systems//
  TotalVDWCoul<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Sim.Box, Sim.d_a, FF, Sim.Blocksum, Sim.device_flag, Host_threads + Guest_threads, Host_threads, totMol, NFrameworkAtomsPerThread, ConsiderHostHost, UseOffset);
  //printf("Total VDW + Real, Nblock = %zu, Nthread = %zu, Host: %zu, Guest: %zu, Allocated size: %zu\n", Nblock, Nthread, Host_threads, Guest_threads, Sim.Nblocks);
  //Zhao's note: consider using the flag to check for overlap here//
  //printf("Total Thread: %zu, Nblock: %zu, Nthread: %zu\n", Host_threads + Guest_threads, Nblock, Nthread);
  double BlockE[Nblock]; cudaMemcpy(BlockE, Sim.Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t id = 0; id < Nblock; id++) VDWRealE += BlockE[id];
  return VDWRealE;
}

__global__ void CoulombRealCorrection(Boxsize Box, Atoms* System, Atoms Old, ForceField FF, double* BlockEnergy, size_t ComponentID, size_t totalAtoms, size_t Oldsize, size_t Newsize, size_t chainsize) 
{
  ///////////////////////////////////////////////////////
  //All variables passed here should be device pointers//
  ///////////////////////////////////////////////////////
  extern __shared__ double sdata[]; //shared memory for partial sum, energy + dUdlambda//
  int cache_id = threadIdx.x;
  size_t ij = blockIdx.x * blockDim.x + threadIdx.x;

  size_t ij_within_block = ij - blockIdx.x * blockDim.x;

  sdata[ij_within_block] = 0.0; 
  //Initialize Blocky and BlockdUdlambda//
  BlockEnergy[blockIdx.x] = 0.0; 

  __shared__ bool Blockflag = false;

  if(ij < totalAtoms * chainsize)
  {
  BlockEnergy[blockIdx.x] = 0.0; //BlockdUdlambda[blockIdx.x] = 0.0;
  // Manually fusing/collapsing the loop //
  size_t i = ij/chainsize;
  size_t j = ij%chainsize; //+ ij/totalAtoms; // position in Old

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

  const Atoms Component=System[comp];
  const double scaleA = Component.scale[posi];
  const double chargeA = Component.charge[posi];
  const double scalingCoulombA = Component.scaleCoul[posi];
  const size_t typeA = Component.Type[posi];
  const size_t MoleculeID = Component.MolID[posi];
  double tempy = 0.0; double tempdU = 0.0;
  if(!((MoleculeID == Old.MolID[0]) &&(comp == ComponentID))) //ComponentID: Component ID for the molecule being translated
  {
    ///////////
    //  NEW  //
    ///////////
    if(Newsize > 0)
    {
      size_t new_j = j + Oldsize; //Oldsize usually equals to the molecule size, according to the Ewald setup, Old comes first, then New//
      double posvec[3] = {Component.x[posi] - Old.x[new_j], Component.y[posi] - Old.y[new_j], Component.z[posi] - Old.z[new_j]};

      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      double result[2] = {0.0, 0.0};
      if (!FF.VDWRealBias && !FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB = Old.charge[new_j];
        const double scalingCoulombB = Old.scaleCoul[new_j];
        const double r = sqrt(rr_dot);
        const double scalingCoul = scalingCoulombA * scalingCoulombB;
        CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
        tempy += result[0]; //prefactor merged in the CoulombReal function
      }
    }
    ///////////
    //  OLD  //
    ///////////
    if(Oldsize > 0)
    {
      double posvec[3] = {Component.x[posi] - Old.x[j], Component.y[posi] - Old.y[j], Component.z[posi] - Old.z[j]};
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      const double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      double result[2] = {0.0, 0.0};
      if (!FF.VDWRealBias && !FF.noCharges && rr_dot < FF.CutOffCoul)
      {
        const double chargeB = Old.charge[j];
        const double scalingCoulombB = Old.scaleCoul[j];
        const double r = sqrt(rr_dot);
        const double scalingCoul = scalingCoulombA * scalingCoulombB;
        CoulombReal(FF, chargeA, chargeB, r, scalingCoul, result, Box.Prefactor, Box.Alpha);
        tempy -= result[0]; //prefactor merged in the CoulombReal function
      }
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

static inline void Prepare_Old_New_sizes_CoulombCorrection(int MoveType, size_t chainsize, size_t& Oldsize, size_t& Newsize)
{
  switch(MoveType)
  {
    case TRANSLATION_ROTATION: case CBCF_LAMBDACHANGE: // Translation/Rotation/CBCF Lambda Change Move //
    {
      throw std::runtime_error("No Need for Ewald Correction for TRANSLATION/ROTATION/CBCF Lambda Change moves. They don't have CBMC!");
    }
    case INSERTION: case CBCF_INSERTION: // Insertion or CBCF Insertion //
    {
      Oldsize   = 0;
      Newsize   = chainsize;
      break;
    }
    case DELETION: case CBCF_DELETION: // Deletion or CBCF Deletion //
    {
      Oldsize   = chainsize;
      Newsize   = 0;
      break;
    }
    case REINSERTION: // Reinsertion //
    {
      throw std::runtime_error("Use the Special Function for Reinsertion");
    }
  }
}

///////////////////////////////////////////////
// Coulombic interaction Correction to CBMC  //
///////////////////////////////////////////////
double CoulombRealCorrection_General(Boxsize& Box, Atoms*& d_a, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType)
{
  if(FF.noCharges || FF.VDWRealBias) return 0.0;

  size_t Oldsize = 0; size_t Newsize = 0; size_t chainsize = SystemComponents.Moleculesize[SelectedComponent];
  size_t totalAtoms = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
    totalAtoms += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  if(totalAtoms == 0) return 0;
  Prepare_Old_New_sizes_CoulombCorrection(MoveType, chainsize, Oldsize, Newsize);
  size_t numberOfAtoms = Oldsize + Newsize;
  size_t Nblock = 0; size_t Nthread = 0;
  Setup_threadblock_VDWCoul(totalAtoms * chainsize, &Nblock, &Nthread);
  //Zhao's note: since the Ewald correction has already taken care of the new/old position data//
  //If we call this function right after the Ewald correction, we don't need to manipulate the new/old positions//
  CoulombRealCorrection<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, d_a, Old, FF, Blocksum, SelectedComponent, totalAtoms, Oldsize, Newsize, chainsize);
  double tot=0;
  double BlockResult[Nblock]; cudaMemcpy(BlockResult, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++) tot += BlockResult[i];
  //printf("Coulomb (Real) Energy is %.5f\n", tot);
  return tot;
}

double CoulombRealCorrection_Reinsertion(Boxsize& Box, Atoms*& d_a, Atoms& Old, double* tempx, double* tempy, double* tempz, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent)
{
  if(FF.noCharges || FF.VDWRealBias) return 0.0;
  size_t chainsize = SystemComponents.Moleculesize[SelectedComponent];
  size_t Oldsize = chainsize; size_t Newsize = chainsize;
  size_t totalAtoms = 0;
  for(size_t ijk = 0; ijk < SystemComponents.Total_Components; ijk++)
    totalAtoms += SystemComponents.Moleculesize[ijk] * SystemComponents.NumberOfMolecule_for_Component[ijk];
  if(totalAtoms == 0) return 0;
  size_t numberOfAtoms = Oldsize + Newsize;
  size_t Nblock = 0; size_t Nthread = 0;
  Setup_threadblock_VDWCoul(totalAtoms * chainsize, &Nblock, &Nthread);
  //Zhao's note: since the Ewald correction has already taken care of the new/old position data//
  //If we call this function right after the Ewald correction, we don't need to manipulate the new/old positions//
  CoulombRealCorrection<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, d_a, Old, FF, Blocksum, SelectedComponent, totalAtoms, Oldsize, Newsize, chainsize);
  double tot=0;
  double BlockResult[Nblock]; cudaMemcpy(BlockResult, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++) tot += BlockResult[i];
  //printf("Coulomb (Real) Energy for Reinsertion is %.5f\n", tot);
  return tot;
}
