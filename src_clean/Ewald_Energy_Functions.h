#include <omp.h>
#include "ewald_preparation.h"
inline void checkCUDAErrorEwald(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        printf("CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE);
    }
}
/*
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
*/

//////////////////////////////////////////////////////////
// General Functions for User-defined Complex Variables //
//////////////////////////////////////////////////////////
__device__ double ComplexNorm(Complex a)
{
  return a.real * a.real + a.imag * a.imag;
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
    double3 pos = Old.pos[posi];
    Box.eik_x[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_y[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_z[posi + 0 * numberOfAtoms] = tempcomplex;
    double3 s; matrix_multiply_by_vector(Box.InverseCell, pos, s); s*=2*M_PI; 
    tempcomplex.real = std::cos(s.x); tempcomplex.imag = std::sin(s.x); Box.eik_x[posi + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.y); tempcomplex.imag = std::sin(s.y); Box.eik_y[posi + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.z); tempcomplex.imag = std::sin(s.z); Box.eik_z[posi + 1 * numberOfAtoms] = tempcomplex;
  }
  //New//
  for(size_t posi=Oldsize; posi < Oldsize + Newsize; ++posi)
  {
    tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
    double3 pos = Old.pos[posi];
    Box.eik_x[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_y[posi + 0 * numberOfAtoms] = tempcomplex;
    Box.eik_z[posi + 0 * numberOfAtoms] = tempcomplex;
    double3 s ; matrix_multiply_by_vector(Box.InverseCell, pos, s); s*=2*M_PI; 
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
  if(MoveType == TRANSLATION || MoveType == ROTATION || MoveType == SPECIAL_ROTATION || MoveType == SINGLE_INSERTION || MoveType == SINGLE_DELETION) // Translation/Rotation/single_insertion/single_deletion //
  {
    //For Translation/Rotation, the Old positions are already in the Old struct, just need to put the New positions into Old, after the Old positions//
    for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
    {
      Old.pos[i]           = New.pos[i - Oldsize];
      Old.scale[i]         = New.scale[i - Oldsize];
      Old.charge[i]        = New.charge[i - Oldsize];
      Old.scaleCoul[i]     = New.scaleCoul[i - Oldsize];
    }
  }
  else if(MoveType == INSERTION || MoveType == CBCF_INSERTION) // Insertion & Fractional Insertion //
  {
    //Put the trial orientations in New to Old, right after the first bead position//
    for(size_t i = 0; i < chainsize; i++)
    {
      Old.pos[i + 1]       = New.pos[Location * chainsize + i];
      Old.scale[i + 1]     = New.scale[Location * chainsize + i];
      Old.charge[i + 1]    = New.charge[Location * chainsize + i];
      Old.scaleCoul[i + 1] = New.scaleCoul[Location * chainsize + i];
    }
  }
  else if(MoveType == DELETION || MoveType == CBCF_DELETION) // Deletion //
  {
    for(size_t i = 0; i < Oldsize; i++)
    {
      // For deletion, Location = UpdateLocation, see Deletion Move //
      Old.pos[i]           = d_a[SelectedComponent].pos[Location + i];
      Old.scale[i]         = d_a[SelectedComponent].scale[Location + i];
      Old.charge[i]        = d_a[SelectedComponent].charge[Location + i];
      Old.scaleCoul[i]     = d_a[SelectedComponent].scaleCoul[Location + i];
    }
  }
  Initialize_Vectors(Box, Oldsize, Newsize, Old, numberOfAtoms, kmax);
}

__global__ void Initialize_EwaldVector_LambdaChange(Boxsize Box, int3 kmax, Atoms* d_a, Atoms Old, size_t Oldsize, double2 newScale)
{
  Initialize_Vectors(Box, Oldsize, 0, Old, Oldsize, kmax);
}

__global__ void Initialize_EwaldVector_Reinsertion(Boxsize Box, int3 kmax, double3* temp, Atoms* d_a, Atoms Old, size_t Oldsize, size_t Newsize, size_t realpos, size_t numberOfAtoms, size_t SelectedComponent)
{
  for(size_t i = 0; i < Oldsize; i++)
  {
    Old.pos[i]       = d_a[SelectedComponent].pos[realpos + i];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i];
  }
  //Reinsertion New Positions stored in three arrays, other data are the same as the Old molecule information in d_a//
  for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
  {
    Old.pos[i]       = temp[i - Oldsize];
    Old.scale[i]     = d_a[SelectedComponent].scale[realpos + i - Oldsize];
    Old.charge[i]    = d_a[SelectedComponent].charge[realpos + i - Oldsize];
    Old.scaleCoul[i] = d_a[SelectedComponent].scaleCoul[realpos + i - Oldsize];
  }
  Initialize_Vectors(Box, Oldsize, Newsize, Old, numberOfAtoms, kmax);
}

__global__ void Initialize_EwaldVector_IdentitySwap(Boxsize Box, int3 kmax, double3* temp, Atoms* d_a, Atoms Old, size_t Oldsize, size_t Newsize, size_t realpos, size_t numberOfAtoms, size_t OLDComponent, size_t NEWComponent)
{
  for(size_t i = 0; i < Oldsize; i++)
  {
    Old.pos[i]       = d_a[OLDComponent].pos[realpos + i];
    Old.scale[i]     = d_a[OLDComponent].scale[realpos + i];
    Old.charge[i]    = d_a[OLDComponent].charge[realpos + i];
    Old.scaleCoul[i] = d_a[OLDComponent].scaleCoul[realpos + i];
  }
  //IdentitySwap New Positions stored in three arrays, other data are the same as the Old molecule information in d_a//
  //Zhao's note: assuming not performing identity swap on fractional molecules//
  for(size_t i = Oldsize; i < Oldsize + Newsize; i++) //chainsize here is the total size of the molecule for translation/rotation
  {
    Old.pos[i]       = temp[i - Oldsize];
    Old.scale[i]     = 1.0;
    Old.charge[i]    = d_a[NEWComponent].charge[i - Oldsize];
    Old.scaleCoul[i] = 1.0;
  }
  Initialize_Vectors(Box, Oldsize, Newsize, Old, numberOfAtoms, kmax);
}

__global__ void JustStore_Ewald(Boxsize Box, size_t nvec)
{
  size_t i         = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nvec) Box.tempEik[i] = Box.AdsorbateEik[i];
}

///////////////////////////////////////////////////////////////
// CALCULATE FOURIER PART OF THE COULOMBIC ENERGY FOR A MOVE //
///////////////////////////////////////////////////////////////

__global__ void Fourier_Ewald_Diff(Boxsize Box, Complex* SameTypeEik, Complex* CrossTypeEik, Atoms Old, double alpha_squared, double prefactor, int3 kmax, size_t Oldsize, size_t Newsize, double* Blocksum, bool UseTempVector, size_t Nblock)
{
  //Zhao's note: provide an additional Nblock to distinguish Host-Guest and Guest-Guest Ewald//
  //Guest-Guest is the first half, Host-Guest is the second//
  extern __shared__ double sdata[]; //shared memory for partial sum//
  size_t kxyz           = blockIdx.x * blockDim.x + threadIdx.x;
  int    cache_id       = threadIdx.x;
  size_t i_within_block = kxyz - blockIdx.x * blockDim.x; //for recording the position of the thread within a block
  double tempE = 0.0;
  size_t    kx_max  = kmax.x;
  size_t    ky_max  = kmax.y;
  size_t    kz_max  = kmax.z;
  size_t    nvec    = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);

  if(blockIdx.x >= Nblock) kxyz -= blockDim.x * Nblock; //If Host-Guest Interaction, adjust kxyz//
  if(kxyz < nvec)
  {
    //Box.tempEik[kxyz] = Box.AdsorbateEik[kxyz];
    sdata[i_within_block] = 0.0;
    int       kz      = kxyz%(2 * kz_max + 1) - kz_max;
    int       kxy     = kxyz/(2 * kz_max + 1);
    int       kx      = kxy /(2 * ky_max + 1);
    int       ky      = kxy %(2 * ky_max + 1) - ky_max;
    double    ksqr    = static_cast<double>(kx * kx + ky * ky + kz * kz);
    if(Box.UseLAMMPSEwald) //Overwrite ksqr if we use the LAMMPS Setup for Ewald//
    {
      const double lx = Box.Cell[0];
      const double ly = Box.Cell[4];
      const double lz = Box.Cell[8];
      const double xy = Box.Cell[3];
      const double xz = Box.Cell[6];
      const double yz = Box.Cell[7];
      const double ux = 2*M_PI/lx;
      const double uy = 2*M_PI*(-xy)/lx/ly;
      const double uz = 2*M_PI*(xy*yz - ly*xz)/lx/ly/lz;
      const double vy = 2*M_PI/ly;
      const double vz = 2*M_PI*(-yz)/ly/lz;
      const double wz = 2*M_PI/lz;
      const double kvecx = kx*ux;
      const double kvecy = kx*uy + ky*vy;
      const double kvecz = kx*uz + ky*vz + kz*wz;
      ksqr  = kvecx*kvecx + kvecy*kvecy + kvecz*kvecz;
    }
    if((ksqr > 1e-10) && (ksqr < Box.ReciprocalCutOff))
    {
      double3 ax; ax.x = Box.InverseCell[0]; ax.y = Box.InverseCell[3]; ax.z = Box.InverseCell[6];
      double3 ay; ay.x = Box.InverseCell[1]; ay.y = Box.InverseCell[4]; ay.z = Box.InverseCell[7];
      double3 az; az.x = Box.InverseCell[2]; az.y = Box.InverseCell[5]; az.z = Box.InverseCell[8];
      size_t numberOfAtoms = Oldsize + Newsize;
      Complex cksum_old; cksum_old.real = 0.0; cksum_old.imag = 0.0;
      Complex cksum_new; cksum_new.real = 0.0; cksum_new.imag = 0.0;
      double3 kvec_x = ax * 2.0 * M_PI * (double) kx;
      double3 kvec_y = ay * 2.0 * M_PI * (double) ky;
      double3 kvec_z = az * 2.0 * M_PI * (double) kz;
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
      double3 tempkvec   = kvec_x + kvec_y + kvec_z;
      double  rksq       = dot(tempkvec, tempkvec);
      double  temp       = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
      Complex newV; Complex OldV;
      //Zhao's note: this is for CBCF insertion, where insertion is the second step. The intermediate Eik is tempEik, is the one we should use//
      if(blockIdx.x < Nblock) //Same Type, do the normal Ewald//
      {
        if(UseTempVector)
        {
          OldV.real = Box.tempEik[kxyz].real;  OldV.imag = Box.tempEik[kxyz].imag;
        }
        else
        {
          OldV.real = SameTypeEik[kxyz].real; OldV.imag = SameTypeEik[kxyz].imag;
        }
        newV.real          = OldV.real + cksum_new.real - cksum_old.real;
        newV.imag          = OldV.imag + cksum_new.imag - cksum_old.imag;
        tempE             += temp * ComplexNorm(newV);
        tempE             -= temp * ComplexNorm(OldV);
        Box.tempEik[kxyz] = newV; //Guest-Guest, do the normal Ewald, update the wave vector//
      }
      else //Host-Guest//
      {
        OldV.real = CrossTypeEik[kxyz].real;  OldV.imag = CrossTypeEik[kxyz].imag;
        tempE += temp * (OldV.real * (cksum_new.real - cksum_old.real) + OldV.imag * (cksum_new.imag - cksum_old.imag));
      }
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

void Skip_Ewald(Boxsize& Box)
{
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(numberOfWaveVectors, &Nblock, &Nthread);
  JustStore_Ewald<<<Nblock, Nthread>>>(Box, numberOfWaveVectors);
}

void Copy_Ewald_Vector(Simulations& Sim)
{
  //Swap pointer
  Complex* temp = Sim.Box.tempEik;
  Sim.Box.tempEik = Sim.Box.AdsorbateEik;
  Sim.Box.AdsorbateEik = temp;

  Complex* tempFramework = Sim.Box.tempFrameworkEik;
  Sim.Box.tempFrameworkEik = Sim.Box.FrameworkEik;
  Sim.Box.FrameworkEik = tempFramework;
}

__global__ void Update_Ewald_Stored(Complex* Eik, Complex* Temp_Eik, size_t nvec)
{
  size_t i         = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nvec) Eik[i] = Temp_Eik[i];
}
void Update_Ewald_Vector(Boxsize& Box, bool CPU, Components& SystemComponents, size_t SelectedComponent)
{
  //else    //Update on the GPU//
  //{
    size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
    size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(numberOfWaveVectors, &Nblock, &Nthread);
    Complex* Stored_Eik;
    if(SelectedComponent < SystemComponents.NComponents.y)
    {
      Stored_Eik = Box.FrameworkEik;
    }
    else
    {
      Stored_Eik = Box.AdsorbateEik;
    }
    Update_Ewald_Stored<<<Nblock, Nthread>>>(Stored_Eik, Box.tempEik, numberOfWaveVectors);
  //}
}

////////////////////////////////////////////////
// Main Ewald Functions (Fourier + Exclusion) //
////////////////////////////////////////////////
double2 GPU_EwaldDifference_General(Boxsize& Box, Atoms*& d_a, Atoms& New, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location, double2 Scale)
{
  if(FF.noCharges && !SystemComponents.hasPartialCharge[SelectedComponent]) return {0.0, 0.0};
  //cudaDeviceSynchronize();
  double start = omp_get_wtime();
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t Oldsize = 0; size_t Newsize = 0; size_t chainsize = 0;
  bool UseTempVector = false; //Zhao's note: Whether or not to use the temporary Vectors (Only used for CBCF Insertion in this function)//

  //If framework molecules are moved, sameType = Framework-Framework, CrossType = Framework-Adsorbate//
  //If adsorbate Molecules are moved, sameType = Adsorbate-Adsorbate, CrossType = Framework-Adsorbate//
  Complex* SameType; Complex* CrossType;

  if(SelectedComponent < SystemComponents.NComponents.y)
  {
    SameType = Box.FrameworkEik; CrossType = Box.AdsorbateEik;
  }
  else
  {
    SameType = Box.AdsorbateEik; CrossType = Box.FrameworkEik;
  }
  switch(MoveType)
  {
    case TRANSLATION: case ROTATION: case SPECIAL_ROTATION: // Translation/Rotation Move //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent];
      break;
    }
    case INSERTION: case SINGLE_INSERTION: // Insertion //
    {
      Oldsize   = 0;
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case DELETION:  case SINGLE_DELETION: // Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case REINSERTION: // Reinsertion //
    {
      throw std::runtime_error("Use the Special Function for Reinsertion");
      //break;
    }
    case IDENTITY_SWAP:
    {
      throw std::runtime_error("Use the Special Function for IDENTITY SWAP!");
    }
    case CBCF_LAMBDACHANGE: // CBCF Lambda Change //
    {
      throw std::runtime_error("Use the Special Function for CBCF Lambda Change");
      //Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      //Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      //chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      //break;
    }
    case CBCF_INSERTION: // CBCF Lambda Insertion //
    {
      Oldsize      = 0;
      Newsize      = SystemComponents.Moleculesize[SelectedComponent];
      chainsize    = SystemComponents.Moleculesize[SelectedComponent] - 1;
      UseTempVector = true;
      break;
    }
    case CBCF_DELETION: // CBCF Lambda Deletion //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = 0;
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
  }
  size_t numberOfAtoms = Oldsize + Newsize;

  Initialize_EwaldVector_General<<<1,1>>>(Box, Box.kmax, d_a, New, Old, Oldsize, Newsize, SelectedComponent, Location, chainsize, numberOfAtoms, MoveType); checkCUDAErrorEwald("error Initializing Ewald Vectors");

  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(numberOfWaveVectors, &Nblock, &Nthread);

  //If we separate Host-Guest from Guest-Guest, we can double the Nblock, so the first half does Guest-Guest, and the second half does Host-Guest//
  Fourier_Ewald_Diff<<<Nblock * 2, Nthread, Nthread * sizeof(double)>>>(Box, SameType, CrossType, Old, alpha_squared, prefactor, Box.kmax, Oldsize, Newsize, Blocksum, UseTempVector, Nblock);
  
  double sum[Nblock * 2]; double SameSum = 0.0; double CrossSum = 0.0;
  cudaMemcpy(sum, Blocksum, 2 * Nblock * sizeof(double), cudaMemcpyDeviceToHost); //HG + GG Energies//
  for(size_t i = 0; i < Nblock; i++){SameSum += sum[i];}
  for(size_t i = Nblock; i < 2 * Nblock; i++){CrossSum += sum[i];}
  //Zhao's note: when adding fractional molecules, this might not be correct//
  double deltaExclusion = 0.0;
  
  if(SystemComponents.rigid[SelectedComponent])
  {
    if(MoveType == INSERTION || MoveType == SINGLE_INSERTION) // Insertion //
    {
      //Zhao's note: This is a bit messy, because when creating the molecules at the beginning of the simulation, we need to create a fractional molecule//
      //MoveType is 2, not 4. 4 is for the insertion after making the old fractional molecule full.//
      double delta_scale = std::pow(Scale.y,2) - 0.0;
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      //if(SystemComponents.CURRENTCYCLE == 16) printf("Exclusion energy: %.5f, delta_scale: %.5f, true_val: %.5f\n", deltaExclusion, delta_scale, (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]));
      SameSum -= deltaExclusion;
    }
    else if(MoveType == DELETION || MoveType == SINGLE_DELETION) // Deletion //
    {
      double delta_scale = 0.0 - 1.0;
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      SameSum -= deltaExclusion;
    }
    else if(MoveType == CBCF_INSERTION) // CBCF Lambda Insertion //
    {
      double delta_scale = std::pow(Scale.y,2) - 0.0;
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      SameSum -= deltaExclusion;
    }
    else if(MoveType == CBCF_DELETION) // CBCF Lambda Deletion //
    {
      double delta_scale = 0.0 - std::pow(Scale.y,2);
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      SameSum -= deltaExclusion;
    }
  }
  //cudaDeviceSynchronize();
  double end = omp_get_wtime();
  return {SameSum, 2.0 * CrossSum};
}

//Zhao's note: THIS IS A SPECIAL FUNCTION JUST FOR REINSERTION//
double2 GPU_EwaldDifference_Reinsertion(Boxsize& Box, Atoms*& d_a, Atoms& Old, double3* temp, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, size_t UpdateLocation)
{
  if(FF.noCharges && !SystemComponents.hasPartialCharge[SelectedComponent]) return {0.0, 0.0};
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  size_t Oldsize = 0; size_t Newsize = numberOfAtoms;
  //Zhao's note: translation/rotation/reinsertion involves new + old states. Insertion/Deletion only has the new state.
  Oldsize         = SystemComponents.Moleculesize[SelectedComponent];
  numberOfAtoms  += Oldsize;

  Complex* SameType = Box.AdsorbateEik; Complex* CrossType = Box.FrameworkEik;

  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  Initialize_EwaldVector_Reinsertion<<<1,1>>>(Box, Box.kmax, temp, d_a, Old, Oldsize, Newsize, UpdateLocation, numberOfAtoms, SelectedComponent);

  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(numberOfWaveVectors, &Nblock, &Nthread);
  Fourier_Ewald_Diff<<<Nblock * 2, Nthread, Nthread * sizeof(double)>>>(Box, SameType, CrossType, Old, alpha_squared, prefactor, Box.kmax, Oldsize, Newsize, Blocksum, false, Nblock);
  double sum[Nblock * 2]; double SameSum = 0.0;  double CrossSum = 0.0;
  cudaMemcpy(sum, Blocksum, 2 * Nblock * sizeof(double), cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < Nblock; i++){SameSum += sum[i];}
  for(size_t i = Nblock; i < 2 * Nblock; i++){CrossSum += sum[i];}

  return {SameSum, 2.0 * CrossSum};
}

double2 GPU_EwaldDifference_IdentitySwap(Boxsize& Box, Atoms*& d_a, Atoms& Old, double3* temp, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t OLDComponent, size_t NEWComponent, size_t UpdateLocation)
{
  if(FF.noCharges && !SystemComponents.hasPartialCharge[NEWComponent] && !SystemComponents.hasPartialCharge[OLDComponent]) return {0.0, 0.0};
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t numberOfAtoms = 0;
  size_t Oldsize = 0; size_t Newsize = 0;
  if(SystemComponents.hasPartialCharge[OLDComponent])
  {
    Oldsize = SystemComponents.Moleculesize[OLDComponent];
  }
  if(SystemComponents.hasPartialCharge[NEWComponent])
  {
    Newsize = SystemComponents.Moleculesize[NEWComponent];
  }
  numberOfAtoms = Oldsize + Newsize;

  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  Initialize_EwaldVector_IdentitySwap<<<1,1>>>(Box, Box.kmax, temp, d_a, Old, Oldsize, Newsize, UpdateLocation, numberOfAtoms, OLDComponent, NEWComponent);

  Complex* SameType = Box.AdsorbateEik; Complex* CrossType = Box.FrameworkEik;
  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(numberOfWaveVectors, &Nblock, &Nthread);
  Fourier_Ewald_Diff<<<Nblock * 2, Nthread, Nthread * sizeof(double)>>>(Box, SameType, CrossType, Old, alpha_squared, prefactor, Box.kmax, Oldsize, Newsize, Blocksum, false, Nblock);
  double sum[Nblock * 2]; double SameSum = 0.0;  double CrossSum = 0.0;
  cudaMemcpy(sum, Blocksum, 2 * Nblock * sizeof(double), cudaMemcpyDeviceToHost);

  for(size_t i = 0; i < Nblock; i++){SameSum += sum[i];}
  for(size_t i = Nblock; i < 2 * Nblock; i++){CrossSum += sum[i];}

  //Exclusion parts//
  if(SystemComponents.rigid[NEWComponent] && SystemComponents.hasPartialCharge[NEWComponent])
  {
    double delta_scale = 1.0;
    double deltaExclusion = (SystemComponents.ExclusionIntra[NEWComponent] + SystemComponents.ExclusionAtom[NEWComponent]) * delta_scale;
    SameSum -= deltaExclusion;
  }
  if(SystemComponents.rigid[OLDComponent] && SystemComponents.hasPartialCharge[OLDComponent])
  {
    double delta_scale = -1.0;
    double deltaExclusion = (SystemComponents.ExclusionIntra[OLDComponent] + SystemComponents.ExclusionAtom[OLDComponent]) * delta_scale;
    SameSum -= deltaExclusion;
  }

  return {SameSum, 2.0 * CrossSum};
}

///////////////////////////////////////////////////////////////////////////
// Zhao's note: Special function for the Ewald for Lambda change of CBCF //
///////////////////////////////////////////////////////////////////////////
__global__ void Fourier_Ewald_Diff_LambdaChange(Boxsize Box, Complex* SameTypeEik, Complex* CrossTypeEik, Atoms Old, double alpha_squared, double prefactor, int3 kmax, size_t Oldsize, size_t Newsize, double* Blocksum, bool UseTempVector, size_t Nblock, double newScale)
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

  if(blockIdx.x >= Nblock) kxyz -= blockDim.x * Nblock; //If Host-Guest Interaction, adjust kxyz//
  if(kxyz < nvec)
  {
    sdata[i_within_block] = 0.0;
    int       kz      = kxyz%(2 * kz_max + 1) - kz_max;
    int       kxy     = kxyz/(2 * kz_max + 1);
    int       kx      = kxy /(2 * ky_max + 1);
    int       ky      = kxy %(2 * ky_max + 1) - ky_max;
    double    ksqr    = static_cast<double>(kx * kx + ky * ky + kz * kz);

    if(Box.UseLAMMPSEwald) //Overwrite ksqr if we use the LAMMPS Setup for Ewald//
    {
      const double lx = Box.Cell[0];
      const double ly = Box.Cell[4];
      const double lz = Box.Cell[8];
      const double xy = Box.Cell[3];
      const double xz = Box.Cell[6];
      const double yz = Box.Cell[7];
      const double ux = 2*M_PI/lx;
      const double uy = 2*M_PI*(-xy)/lx/ly;
      const double uz = 2*M_PI*(xy*yz - ly*xz)/lx/ly/lz;
      const double vy = 2*M_PI/ly;
      const double vz = 2*M_PI*(-yz)/ly/lz;
      const double wz = 2*M_PI/lz;
      const double kvecx = kx*ux;
      const double kvecy = kx*uy + ky*vy;
      const double kvecz = kx*uz + ky*vz + kz*wz;
      ksqr  = kvecx*kvecx + kvecy*kvecy + kvecz*kvecz;
    }
    if((ksqr > 1e-10) && (ksqr < Box.ReciprocalCutOff))
    {
      double3 ax = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]};
      double3 ay = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]};
      double3 az = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]};
      size_t numberOfAtoms = Oldsize;
      Complex cksum_old; cksum_old.real = 0.0; cksum_old.imag = 0.0;
      Complex cksum_new; cksum_new.real = 0.0; cksum_new.imag = 0.0;
      double3 kvec_x = ax * 2.0 * M_PI * (double) kx;
      double3 kvec_y = ay * 2.0 * M_PI * (double) ky;
      double3 kvec_z = az * 2.0 * M_PI * (double) kz;
      double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);
      //OLD, For Lambda Change, there is no change in positions, so just loop over OLD//
      for(size_t i=0; i<Oldsize; ++i)
      {
        Complex eik_temp  = Box.eik_y[i + numberOfAtoms * static_cast<size_t>(std::abs(ky))];
        eik_temp.imag     = ky>=0 ? eik_temp.imag : -eik_temp.imag;
        Complex eik_xy    = multiply(Box.eik_x[i + numberOfAtoms * static_cast<size_t>(kx)], eik_temp);

        eik_temp          = Box.eik_z[i + numberOfAtoms * static_cast<size_t>(std::abs(kz))];
        eik_temp.imag     = kz>=0 ? eik_temp.imag : -eik_temp.imag;
        double charge     = Old.charge[i];
        double scaling    = Old.scaleCoul[i];
        Complex tempi     = multiply(eik_xy, eik_temp);

        cksum_old.real += scaling * charge * tempi.real;
        cksum_old.imag += scaling * charge * tempi.imag;
        cksum_new.real += newScale* charge * tempi.real;
        cksum_new.imag += newScale* charge * tempi.imag;
      }
      double3 tempkvec   = kvec_x + kvec_y + kvec_z;
      double  rksq       = dot(tempkvec, tempkvec);
      double  temp       = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
      Complex newV; Complex OldV;
      if(blockIdx.x < Nblock)
      {
        if(UseTempVector)
        {
          OldV.real = Box.tempEik[kxyz].real; OldV.imag = Box.tempEik[kxyz].imag;
        }
        else
        {
          OldV.real = SameTypeEik[kxyz].real; OldV.imag = SameTypeEik[kxyz].imag;
        }
        newV.real          = OldV.real + cksum_new.real - cksum_old.real;
        newV.imag          = OldV.imag + cksum_new.imag - cksum_old.imag;
        tempE             += temp * ComplexNorm(newV);
        tempE             -= temp * ComplexNorm(OldV);
        Box.tempEik[kxyz] = newV;
      }
      else //Host-Guest//
      {
        OldV.real = CrossTypeEik[kxyz].real;  OldV.imag = CrossTypeEik[kxyz].imag;
        tempE += temp * (OldV.real * (cksum_new.real - cksum_old.real) + OldV.imag * (cksum_new.imag - cksum_old.imag));
      }
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

//Zhao's note: THIS IS A SPECIAL FUNCTION JUST FOR LAMBDA MOVE FOR FRACTIONAL MOLECULES//
double2 GPU_EwaldDifference_LambdaChange(Boxsize& Box, Atoms*& d_a, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, double2 oldScale, double2 newScale, int MoveType)
{
  if(FF.noCharges && !SystemComponents.hasPartialCharge[SelectedComponent]) return {0.0, 0.0};
  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t numberOfAtoms = SystemComponents.Moleculesize[SelectedComponent];
  size_t Oldsize = numberOfAtoms;
  size_t Newsize = numberOfAtoms;
  size_t chainsize  = SystemComponents.Moleculesize[SelectedComponent];

  bool UseTempVector = false;
  if(MoveType == CBCF_DELETION) // CBCF Deletion, Lambda Change is its second step //
  {
    UseTempVector = true;
  }
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  Initialize_EwaldVector_LambdaChange<<<1,1>>>(Box, Box.kmax, d_a, Old, Oldsize, newScale);

  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock(numberOfWaveVectors, &Nblock, &Nthread);

  Complex* SameType = Box.AdsorbateEik;
  Complex* CrossType= Box.FrameworkEik;

  Fourier_Ewald_Diff_LambdaChange<<<Nblock * 2, Nthread, Nthread * sizeof(double)>>>(Box, SameType, CrossType, Old, alpha_squared, prefactor, Box.kmax, Oldsize, Newsize, Blocksum, UseTempVector, Nblock, newScale.y);
 
  double Host_sum[Nblock * 2]; double SameSum = 0.0; double CrossSum = 0.0;
  cudaMemcpy(Host_sum, Blocksum, 2 * Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++){SameSum += Host_sum[i];}
  for(size_t i = Nblock; i < 2 * Nblock; i++){CrossSum += Host_sum[i];}
  //printf("Fourier GPU lambda Change: %.5f\n", tot);
  double delta_scale = std::pow(newScale.y,2) - std::pow(oldScale.y,2);
  double deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
  SameSum -= deltaExclusion;
  //printf("Lambda Ewald, Same: %.5f, Cross: %.5f\n", SameSum, CrossSum);
  return {SameSum, 2.0 * CrossSum};
}

__global__ void Setup_Ewald_Vector(Boxsize Box, Complex* eik_x, Complex* eik_y, Complex* eik_z, Atoms* System, size_t numberOfAtoms, size_t NumberOfComponents, bool UseOffSet)
{
  // Construct exp(ik.r) for atoms and k-vectors kx, ky, kz = 0, 1 explicitly
  //determine the component for i
  size_t i = blockIdx.x * blockDim.x + threadIdx.x; //number of threads = number of atoms in the system
  if(i < numberOfAtoms)
  {
    size_t atom_i = i;
    Complex tempcomplex; tempcomplex.real = 1.0; tempcomplex.imag = 0.0;
    size_t comp = 0;
    for(size_t ijk = 0; ijk < NumberOfComponents; ijk++)
    {
      //totalsize += d_a[ijk].size;
      if(atom_i >= System[ijk].size)
      {
        comp++;
        atom_i -= System[ijk].size;
      }
      else
      {break;}
    }
    if(UseOffSet)
    {
      size_t HalfAllocateSize = System[comp].Allocate_size / 2;
      atom_i += HalfAllocateSize;
    }
    double3 pos;
    pos = System[comp].pos[atom_i];
    eik_x[i + 0 * numberOfAtoms] = tempcomplex;
    eik_y[i + 0 * numberOfAtoms] = tempcomplex;
    eik_z[i + 0 * numberOfAtoms] = tempcomplex;
    double3 s; matrix_multiply_by_vector(Box.InverseCell, pos, s); s*=2*M_PI;
    tempcomplex.real = std::cos(s.x); tempcomplex.imag = std::sin(s.x); eik_x[i + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.y); tempcomplex.imag = std::sin(s.y); eik_y[i + 1 * numberOfAtoms] = tempcomplex;
    tempcomplex.real = std::cos(s.z); tempcomplex.imag = std::sin(s.z); eik_z[i + 1 * numberOfAtoms] = tempcomplex;
    // Calculate remaining positive kx, ky and kz by recurrence
    for(size_t kx = 2; kx <= Box.kmax.x; ++kx)
    {
        eik_x[i + kx * numberOfAtoms] = multiply(eik_x[i + (kx - 1) * numberOfAtoms], eik_x[i + 1 * numberOfAtoms]);
    }

    for(size_t ky = 2; ky <= Box.kmax.y; ++ky)
    {
        eik_y[i + ky * numberOfAtoms] = multiply(eik_y[i + (ky - 1) * numberOfAtoms], eik_y[i + 1 * numberOfAtoms]);
    }

    for(size_t kz = 2; kz <= Box.kmax.z; ++kz)
    {
        eik_z[i + kz * numberOfAtoms] = multiply(eik_z[i + (kz - 1) * numberOfAtoms], eik_z[i + 1 * numberOfAtoms]);
    }
  }
}

//Zhao's note: Idea was that every block considers a grid point for Ewald (a set of kxyz)//
//So the number of atoms each thread needs to consider = TotalAtoms / Nthreads per block (usually 128)//
__global__ void TotalEwald(Atoms* d_a, Boxsize Box, double* BlockSum, Complex* eik_x, Complex* eik_y, Complex* eik_z, Complex* FrameworkEik, Complex* Eik, size_t totAtom, int2 NAtomPerThread, int2 residueAtoms, int2 HostGuestthreads, int3 NComponents, size_t Nblock)
{
  __shared__ double3 sdata[256]; //maybe just need Complex//
  //DEBUG
  //__shared__ int count_data[128]; count_data[threadIdx.x] = 0;
  int adsorbate_count = 0;

  int cache_id = threadIdx.x;
  size_t total_ij = blockIdx.x * blockDim.x + threadIdx.x;
  size_t ij_within_block = total_ij - blockIdx.x * blockDim.x;

  size_t kxyz = blockIdx.x; //Each block takes over one grid point//

  size_t HostThreads = HostGuestthreads.x;
  size_t GuestThreads= HostGuestthreads.y;

  int RealThreads = HostThreads;
 
  int RealNAtomPerThread = NAtomPerThread.x;
  int RealresidueAtoms   = residueAtoms.x;

  if(threadIdx.x >= HostThreads)
  {
    RealThreads        = GuestThreads;
    RealNAtomPerThread = NAtomPerThread.y;
    RealresidueAtoms   = residueAtoms.y;
  }

  sdata[threadIdx.x] = {0.0, 0.0, 0.0};
  sdata[threadIdx.x + blockDim.x] = {0.0, 0.0, 0.0};

  //size_t    kx_max  = Box.kmax.x;
  size_t    ky_max  = Box.kmax.y;
  size_t    kz_max  = Box.kmax.z;
  //size_t    nvec    = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);
  int       kz      = kxyz%(2 * kz_max + 1) - kz_max;
  int       kxy     = kxyz/(2 * kz_max + 1);
  int       kx      = kxy /(2 * ky_max + 1);
  int       ky      = kxy %(2 * ky_max + 1) - ky_max;
  double ksqr = static_cast<double>(kx * kx + ky * ky + kz * kz);

  double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

  double3 ax = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]};
  double3 ay = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]};
  double3 az = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]};
  double3 kvec_x = ax * 2.0 * M_PI * (double) kx;
  double3 kvec_y = ay * 2.0 * M_PI * (double) ky;
  double3 kvec_z = az * 2.0 * M_PI * (double) kz;
  double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);

  double3 tempkvec  = kvec_x + kvec_y + kvec_z;
  double  rksq      = dot(tempkvec, tempkvec);

  if(Box.UseLAMMPSEwald) //Overwrite ksqr if we use the LAMMPS Setup for Ewald//
  {
    const double lx = Box.Cell[0];
    const double ly = Box.Cell[4];
    const double lz = Box.Cell[8];
    const double xy = Box.Cell[3];
    const double xz = Box.Cell[6];
    const double yz = Box.Cell[7];
    const double ux = 2*M_PI/lx;
    const double uy = 2*M_PI*(-xy)/lx/ly;
    const double uz = 2*M_PI*(xy*yz - ly*xz)/lx/ly/lz;
    const double vy = 2*M_PI/ly;
    const double vz = 2*M_PI*(-yz)/ly/lz;
    const double wz = 2*M_PI/lz;
    const double kvecx = kx*ux;
    const double kvecy = kx*uy + ky*vy;
    const double kvecz = kx*uz + ky*vz + kz*wz;
    ksqr  = kvecx*kvecx + kvecy*kvecy + kvecz*kvecz;
  }
  size_t offset = 0;
  size_t atomoffset = 0;
  //If adsorbate atom, offset number of atoms by number of framework atoms//
  if(ij_within_block >= HostThreads)
  {
    offset = HostThreads;// * NAtomPerThread.x + residueAtoms.x; //offset in terms of # of thread
    atomoffset = offset * NAtomPerThread.x + residueAtoms.x;     //offset in terms of # of atoms
  }

  Complex cksum; cksum.real = 0.0; cksum.imag = 0.0;

  if((ksqr > 1e-10) && (ksqr < Box.ReciprocalCutOff))
  {
    //temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
    for(size_t a = 0; a < RealNAtomPerThread; a++)
    {
      size_t Atom = a + RealNAtomPerThread * (ij_within_block - offset) + atomoffset;
      size_t AbsoluteAtom = Atom; size_t comp = 0;
      for(size_t ijk = 0; ijk < NComponents.x; ijk++)
      {
        //totalsize += d_a[ijk].size;
        if(Atom >= d_a[ijk].size)
        {
          comp++;
          Atom -= d_a[ijk].size;
        }
        else
        {break;}
      }
      
      Complex eik_temp  = eik_y[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(ky))];
      eik_temp.imag     = ky>=0 ? eik_temp.imag : -eik_temp.imag;
      Complex eik_xy    = multiply(eik_x[AbsoluteAtom + totAtom * static_cast<size_t>(kx)], eik_temp);
      eik_temp          = eik_z[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(kz))];
      eik_temp.imag     = kz>=0 ? eik_temp.imag : -eik_temp.imag;

      double charge     = d_a[comp].charge[Atom];
      double scaling    = d_a[comp].scaleCoul[Atom];
      Complex tempi     = multiply(eik_xy, eik_temp);
      cksum.real       += scaling * charge * tempi.real;
      cksum.imag       += scaling * charge * tempi.imag;
      if(comp >= NComponents.y)
        adsorbate_count ++;
    }
    if(RealresidueAtoms > 0)
    {
      if((ij_within_block-offset) < RealresidueAtoms) //the thread will read one more atom in the RealresidueAtoms//
      //The thread from zero to RealresidueAtoms will each take another atom//
      {
        //Zhao's note: RealThreads = number of threads for framework/adsorbate in a block, RealThreads * RealNAtomPerThread = the last atom position if there is no residue//
        //for adsorbate, offset number of atoms by number of framework atoms//
        //size_t Atom = RealThreads * RealNAtomPerThread + (ij_within_block - offset) + offset * NAtomPerThread.x + residueAtoms.x;
        size_t Atom = RealThreads * RealNAtomPerThread + (ij_within_block - offset) + atomoffset;
        // - offset) + offset * NAtomPerThread.x + residueAtoms.x;
        size_t AbsoluteAtom = Atom; size_t comp = 0;
        for(size_t ijk = 0; ijk < NComponents.x; ijk++)
        {
          //totalsize += d_a[ijk].size;
          if(Atom >= d_a[ijk].size)
          {
            comp++;
            Atom -= d_a[ijk].size;
          }
          else
          {break;}
        }
        Complex eik_temp  = eik_y[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(ky))];
        eik_temp.imag     = ky>=0 ? eik_temp.imag : -eik_temp.imag;
        Complex eik_xy    = multiply(eik_x[AbsoluteAtom + totAtom * static_cast<size_t>(kx)], eik_temp);
        eik_temp          = eik_z[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(kz))];
        eik_temp.imag     = kz>=0 ? eik_temp.imag : -eik_temp.imag;

        double charge     = d_a[comp].charge[Atom];
        double scaling    = d_a[comp].scaleCoul[Atom];
        Complex tempi     = multiply(eik_xy, eik_temp);
        cksum.real       += scaling * charge * tempi.real;
        cksum.imag       += scaling * charge * tempi.imag; 
        //DEBUG
        /*
        if(blockIdx.x == 10)
        {
          printf("eikxyz locations: %lu, %lu, %lu\n", AbsoluteAtom + totAtom * static_cast<size_t>(kx), AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(ky)), AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(kz)));
          printf("RESIDUAL COMP: %lu, ATOM %lu, tempi: %.5f %.5f\n", comp, Atom, tempi.real, tempi.imag);
          printf("eik_x: %.5f %.5f\n", eik_x[AbsoluteAtom + totAtom * static_cast<size_t>(kx)].real, eik_x[AbsoluteAtom + totAtom * static_cast<size_t>(kx)].imag); 
          printf("eik_y: %.5f %.5f\n", eik_y[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(ky))].real, eik_y[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(ky))].imag); 
          printf("eik_xy: %.5f %.5f\n", eik_xy.real, eik_xy.imag);
          printf("eik_z: %.5f %.5f\n", eik_z[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(kz))].real, eik_z[AbsoluteAtom + totAtom * static_cast<size_t>(std::abs(kz))].imag);
        }
        */
      }
    }
  }
  size_t OFFSET = 0;
  if(threadIdx.x >= HostThreads)
  {
    OFFSET = blockDim.x;
  }
    sdata[ij_within_block+OFFSET].x = cksum.real;
    sdata[ij_within_block+OFFSET].y = cksum.imag;
  __syncthreads();
  //Partial block sum//
  int i=blockDim.x / 2;
  while(i != 0)
  {
    if(cache_id < i)
    {
      sdata[cache_id].x += sdata[cache_id + i].x;
      sdata[cache_id].y += sdata[cache_id + i].y;

      sdata[cache_id + blockDim.x].x += sdata[cache_id + i + blockDim.x].x;
      sdata[cache_id + blockDim.x].y += sdata[cache_id + i + blockDim.x].y;

      //DEBUG
      //count_data[cache_id] += count_data[cache_id + i];
    }
    __syncthreads();
    i /= 2;
  }
  if(cache_id == 0)
  {
    //Complex cksum; cksum.real = sdata[0].x; cksum.imag = sdata[0].y;
    //Complex cksum_ads = {sdata[blockDim.x].x,  sdata[blockDim.x].y};
    //Complex cksum_ads; cksum_ads.real = sdata[blockDim.x].x - cksum.real; cksum_ads.imag = sdata[blockDim.x].y - cksum.imag;
    //double NORM = ComplexNorm(cksum);
    //double NORM_ads = ComplexNorm(cksum_ads);
    //BlockSum[blockIdx.x] = temp * NORM; //framework-framework
    //BlockSum[blockIdx.x + Nblock] = temp * NORM_ads; //adsorbate-adsorbate
    //BlockSum[blockIdx.x + Nblock + Nblock] = temp * (cksum.real * cksum_ads.real + cksum.imag * cksum_ads.imag)*2.0; //framework-adsorbate, dont forget the 2.0!!!//
    
    FrameworkEik[blockIdx.x].real = sdata[0].x;
    FrameworkEik[blockIdx.x].imag = sdata[0].y;
    Eik[blockIdx.x].real = sdata[blockDim.x].x;
    Eik[blockIdx.x].imag = sdata[blockDim.x].y;
  }
}

__global__ void TotalEwald_CalculateEnergy(Boxsize Box, Complex* FrameworkEik, Complex* Eik, double* BlockSum, size_t kpoints, size_t kpoint_per_thread, size_t Nblocks)
{
  __shared__ double sdata[];
  size_t threadID = blockIdx.x * blockDim.x + threadIdx.x;
  double Framework_E = 0.0;
  double Adsorbate_E = 0.0;
  double F_A_E       = 0.0;

  sdata[threadIdx.x] = 0.0;
  sdata[threadIdx.x + blockDim.x] = 0.0;
  sdata[threadIdx.x + blockDim.x + blockDim.x] = 0.0;
  for(size_t i = 0; i < kpoint_per_thread; i++)
  {
    size_t ij = threadID * kpoint_per_thread + i;
    if(ij < kpoints)
    {
      //size_t    kx_max  = Box.kmax.x;
      size_t    ky_max  = Box.kmax.y;
      size_t    kz_max  = Box.kmax.z;
      size_t    kxyz    = ij;
      //size_t    nvec    = (kx_max + 1) * (2 * ky_max + 1) * (2 * kz_max + 1);
      int       kz      = kxyz%(2 * kz_max + 1) - kz_max;
      int       kxy     = kxyz/(2 * kz_max + 1);
      int       kx      = kxy /(2 * ky_max + 1);
      int       ky      = kxy %(2 * ky_max + 1) - ky_max;
      double ksqr = static_cast<double>(kx * kx + ky * ky + kz * kz);

      double alpha = Box.Alpha; double alpha_squared = alpha * alpha;
      double prefactor = Box.Prefactor * (2.0 * M_PI / Box.Volume);

      double3 ax = {Box.InverseCell[0], Box.InverseCell[3], Box.InverseCell[6]};
      double3 ay = {Box.InverseCell[1], Box.InverseCell[4], Box.InverseCell[7]};
      double3 az = {Box.InverseCell[2], Box.InverseCell[5], Box.InverseCell[8]};
      double3 kvec_x = ax * 2.0 * M_PI * (double) kx;
      double3 kvec_y = ay * 2.0 * M_PI * (double) ky;
      double3 kvec_z = az * 2.0 * M_PI * (double) kz;
      double factor = (kx == 0) ? (1.0 * prefactor) : (2.0 * prefactor);

      double3 tempkvec  = kvec_x + kvec_y + kvec_z;
      double  rksq      = dot(tempkvec, tempkvec);
      double  temp      = 0.0;

      if(Box.UseLAMMPSEwald) //Overwrite ksqr if we use the LAMMPS Setup for Ewald//
      {
        const double lx = Box.Cell[0];
        const double ly = Box.Cell[4];
        const double lz = Box.Cell[8];
        const double xy = Box.Cell[3];
        const double xz = Box.Cell[6];
        const double yz = Box.Cell[7];
        const double ux = 2*M_PI/lx;
        const double uy = 2*M_PI*(-xy)/lx/ly;
        const double uz = 2*M_PI*(xy*yz - ly*xz)/lx/ly/lz;
        const double vy = 2*M_PI/ly;
        const double vz = 2*M_PI*(-yz)/ly/lz;
        const double wz = 2*M_PI/lz;
        const double kvecx = kx*ux;
        const double kvecy = kx*uy + ky*vy;
        const double kvecz = kx*uz + ky*vz + kz*wz;
        ksqr  = kvecx*kvecx + kvecy*kvecy + kvecz*kvecz;
      }

      if((ksqr > 1e-10) && (ksqr < Box.ReciprocalCutOff))
      {
        temp = factor * std::exp((-0.25 / alpha_squared) * rksq) / rksq;
 
        Complex cksum_frm = {FrameworkEik[ij].real, FrameworkEik[ij].imag};
        Complex cksum_ads = {Eik[ij].real, Eik[ij].imag};

        double NORM_frm = ComplexNorm(cksum_frm);
        double NORM_ads = ComplexNorm(cksum_ads);
        Framework_E += temp * NORM_frm; //framework-framework
        Adsorbate_E += temp * NORM_ads; //adsorbate-adsorbate
        F_A_E       += temp * (cksum_frm.real * cksum_ads.real + cksum_frm.imag * cksum_ads.imag)*2.0;
      }
    }
  }
  sdata[threadIdx.x]                           = Framework_E;
  sdata[threadIdx.x + blockDim.x]              = Adsorbate_E;
  sdata[threadIdx.x + blockDim.x + blockDim.x] = F_A_E;
  __syncthreads();
  
  //Partial block sum//
  int i=blockDim.x / 2;
  size_t cache_id = threadIdx.x;
  while(i != 0)
  {
    if(cache_id < i)
    {
      sdata[cache_id] += sdata[cache_id + i];
      sdata[cache_id + blockDim.x] += sdata[cache_id + i + blockDim.x];
      sdata[cache_id + blockDim.x + blockDim.x] += sdata[cache_id + i + blockDim.x + blockDim.x];
    }
    __syncthreads();
    i /= 2;
  }
  __syncthreads();
  if(threadIdx.x == 0)
  {
    BlockSum[blockIdx.x]                     = sdata[0];
    BlockSum[blockIdx.x + Nblocks]           = sdata[blockDim.x];
    BlockSum[blockIdx.x + Nblocks + Nblocks] = sdata[blockDim.x + blockDim.x];
  }
}

__global__ void Framework_ComponentZero_Calculate_Intra_Self_Exclusion(Boxsize Box, Atoms* System, double* BlockSum, size_t IntraInteractions, size_t InteractionPerThread, size_t TotalThreadsNeeded)
{
  double E = 0.0; double prefactor_self = Box.Prefactor * Box.Alpha / std::sqrt(M_PI);
  extern __shared__ double sdata[];
  size_t SelectedComponent = 0;
  sdata[threadIdx.x] = 0.0; int cache_id = threadIdx.x;
  //put all the framework atoms threads in one block//
  size_t THREADIdx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t NAtoms = System[SelectedComponent].size;
  for(size_t i = 0; i != InteractionPerThread; i++)
  {
    size_t InteractionIdx = THREADIdx * InteractionPerThread + i;
    //Unwrap Atoms from interactions (upper triangle)//
    size_t AtomA = NAtoms - 2 - std::floor(std::sqrt(-8*InteractionIdx + 4*NAtoms*(NAtoms-1)-7)/2.0 - 0.5);
    size_t AtomB = InteractionIdx + AtomA + 1 - NAtoms*(NAtoms-1)/2 + (NAtoms-AtomA)*((NAtoms-AtomA)-1)/2;

    if(AtomA < NAtoms && AtomB < NAtoms)
    {
      double  factorA = System[SelectedComponent].scaleCoul[AtomA] * System[SelectedComponent].charge[AtomA];
      double3 posA = System[SelectedComponent].pos[AtomA];
 
      double  factorB = System[SelectedComponent].scaleCoul[AtomB] * System[SelectedComponent].charge[AtomB];
      double3 posB    = System[SelectedComponent].pos[AtomB];
      double3 posvec = posA - posB;
      PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
      double rr_dot = dot(posvec, posvec);
      double r = std::sqrt(rr_dot);
      E += Box.Prefactor * factorA * factorB * std::erf(Box.Alpha * r) / r;
      //For self interactions///////////////////////////////////////////////////////////
      //find those where AtomB > AtomA, AtomB - AtomA = 1 (along the diagonal)        //
      //use AtomB for these                                                           //
      //you will miss AtomB = 0, so if AtomB = 1 and AtomA = 0, use property of AtomA //
      //////////////////////////////////////////////////////////////////////////////////
      if(AtomB - AtomA == 1)
      {
        E += prefactor_self * factorB * factorB;
        if(AtomA == 0)
          E += prefactor_self * factorA * factorA;
      }
    }
  }
  //if(THREADIdx == 0) printf("ThreadID 0, ThreadE: %.5f", E);
  sdata[threadIdx.x] = -E;
  __syncthreads();
  //Partial block sum//
  int i=blockDim.x / 2;
  while(i != 0)
  {
    if(cache_id < i) {sdata[cache_id] += sdata[cache_id + i];}
    __syncthreads();
    i /= 2;
  }
  if(threadIdx.x == 0)
  {
    BlockSum[blockIdx.x] = sdata[0];
  }
}

__global__ void Calculate_Intra_Self_Exclusion(Boxsize Box, Atoms* System, double* BlockSum, size_t SelectedComponent)
{
  double E = 0.0; double prefactor_self = Box.Prefactor * Box.Alpha / std::sqrt(M_PI);
  extern __shared__ double sdata[]; 
  sdata[threadIdx.x] = 0.0; int cache_id = threadIdx.x;
  //Intra
  //Option 1: Unwrap upper triangle for pairwise interaction
  //https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
  //MolA = NAds - 2 - std::floor(std::sqrt(-8*Ads_i + 4*NAds*(NAds-1)-7)/2.0 - 0.5);
  //MolB = Ads_i + MolA + 1 - NAds*(NAds-1)/2 + (NAds-MolA)*((NAds- MolA)-1)/2;
  //Option 2: each thread takes care of a molecule, easier
  size_t THREADIdx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t Molsize   = System[SelectedComponent].Molsize;
  size_t AtomIdx   = THREADIdx * Molsize;

  //size_t totalmol  = System[SelectedComponent].size / Molsize;
  if(AtomIdx < System[SelectedComponent].size)
  { 
    //if(THREADIdx == 0) printf("Molsize: %lu, AtomIdx: %lu\n", Molsize, AtomIdx);
  for(size_t i = AtomIdx; i != Molsize + AtomIdx; i++)
  { 
    double  factorA = System[SelectedComponent].scaleCoul[i] * System[SelectedComponent].charge[i];
    double3 posA = System[SelectedComponent].pos[i];
    //for(size_t j = i; j != Molsize; j++)
    for(size_t j = i; j != Molsize + AtomIdx; j++)
    {
      if(i == j)
      //Self: atom and itself
      { 
        E += prefactor_self * factorA * factorA;
      }
      
      else//if(j < i)
      //Intra: within a molecule
      {
        double  factorB = System[SelectedComponent].scaleCoul[j] * System[SelectedComponent].charge[j];
        double3 posB    = System[SelectedComponent].pos[j];
        double3 posvec = posA - posB;
        PBC(posvec, Box.Cell, Box.InverseCell, Box.Cubic);
        double rr_dot = dot(posvec, posvec);
        double r = std::sqrt(rr_dot);
        E += Box.Prefactor * factorA * factorB * std::erf(Box.Alpha * r) / r;
      }
      
    }
  }
  }
  //if(THREADIdx == 0) printf("ThreadID 0, ThreadE: %.5f", E);
  sdata[threadIdx.x] = -E;
  __syncthreads();
  //Partial block sum//
  int i=blockDim.x / 2;
  while(i != 0)
  {
    if(cache_id < i) {sdata[cache_id] += sdata[cache_id + i];}
    __syncthreads();
    i /= 2;
  }
  if(threadIdx.x == 0) 
  {
    BlockSum[blockIdx.x] = sdata[0];
    
  }
}

MoveEnergy Ewald_TotalEnergy(Simulations& Sim, Components& SystemComponents, bool UseOffSet)
{
  SystemComponents.EnergyEvalTimes ++;
  //printf("Performed Ewald Total %zu times\n", SystemComponents.EnergyEvalTimes);
  size_t NTotalAtom = 0;
  size_t Nblock  = 0;
  size_t Nthread = 128;

  MoveEnergy E;

  Boxsize& Box= Sim.Box;
  Atoms* d_a = Sim.d_a;

  size_t NHostAtom = 0; size_t NGuestAtom = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    NHostAtom += SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
    NGuestAtom+= SystemComponents.Moleculesize[i] * SystemComponents.NumberOfMolecule_for_Component[i];

  NTotalAtom = NHostAtom + NGuestAtom;

  int2 NHostGuestthread; 
  //Number of Host threads in a block//
  NHostGuestthread.x = static_cast<int>(static_cast<double>(NHostAtom) / static_cast<double>(NTotalAtom) * Nthread);
  NHostGuestthread.y = Nthread - NHostGuestthread.x;

  if(NTotalAtom > 0)
  {
    int2 NAtomPerThread = {NHostAtom > 0 ? NHostAtom / NHostGuestthread.x : 0, NGuestAtom > 0 ? NGuestAtom / NHostGuestthread.y : 0};
    int2 residueAtoms   = {NHostAtom > 0 ? NHostAtom % NHostGuestthread.x : 0, NGuestAtom > 0 ? NGuestAtom % NHostGuestthread.y : 0};

    //Setup eikx, eiky, and eikz//
    Setup_threadblock(NTotalAtom, &Nblock, &Nthread);

    Complex* eikx; cudaMalloc(&eikx, NTotalAtom * (Box.kmax.x + 1) * sizeof(Complex));
    Complex* eiky; cudaMalloc(&eiky, NTotalAtom * (Box.kmax.y + 1) * sizeof(Complex));
    Complex* eikz; cudaMalloc(&eikz, NTotalAtom * (Box.kmax.z + 1) * sizeof(Complex));
    Setup_Ewald_Vector<<<Nblock, Nthread>>>(Box, eikx, eiky, eikz, d_a, NTotalAtom, SystemComponents.NComponents.x, UseOffSet);

    /*
    Complex* host_eikx; Complex* host_eiky; Complex* host_eikz;
    host_eikx = (Complex*) malloc(NTotalAtom * (Box.kmax.x + 1)*sizeof(Complex));
    host_eiky = (Complex*) malloc(NTotalAtom * (Box.kmax.y + 1)*sizeof(Complex));
    host_eikz = (Complex*) malloc(NTotalAtom * (Box.kmax.z + 1)*sizeof(Complex));

    cudaMemcpy(host_eikx, eikx, NTotalAtom * (Box.kmax.x + 1)*sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_eiky, eiky, NTotalAtom * (Box.kmax.y + 1)*sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_eikz, eikz, NTotalAtom * (Box.kmax.z + 1)*sizeof(Complex), cudaMemcpyDeviceToHost);
    */
    Nblock = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
    if(Nblock > SystemComponents.tempEikAllocateSize)
    {
      printf("Cycle: %zu, temp Allocated: %zu, Allocated: %zu, need: %zu, RE-ALLOCATE structure factors\n", SystemComponents.CURRENTCYCLE, SystemComponents.EikAllocateSize, SystemComponents.tempEikAllocateSize, Nblock);
      SystemComponents.tempEikAllocateSize = 2 * Nblock;
      //cudaFree(&Box.tempEik);
      //cudaFree(&Box.tempFrameworkEik);
      Complex* TEMP; Complex* TEMP_F;
       
      cudaMalloc(&TEMP,   SystemComponents.tempEikAllocateSize * sizeof(Complex));
      cudaMalloc(&TEMP_F, SystemComponents.tempEikAllocateSize * sizeof(Complex));
      //cudaMalloc(&Box.tempEik,          SystemComponents.tempEikAllocateSize * sizeof(Complex));
      //cudaMalloc(&Box.tempFrameworkEik, SystemComponents.tempEikAllocateSize * sizeof(Complex));
      std::swap(TEMP,   Sim.Box.tempEik);
      std::swap(TEMP_F, Sim.Box.tempFrameworkEik);
      cudaFree(TEMP);
      cudaFree(TEMP_F);
    }
    //Try to avoid realloc of Blocksum//
    /*
    if(3*Nblock > Sim.Nblocks)
    {
      printf("kmax: %d %d %d\n", Box.kmax.x, Box.kmax.y, Box.kmax.z);
      printf("Total Ewald Fourier, Need to Allocate more space for blocksum, allocated: %zu, need: %zu\n", Sim.Nblocks, 3*Nblock);
      Sim.Nblocks = 3*Nblock;
      cudaMalloc(&Sim.Blocksum,     Sim.Nblocks * sizeof(double));
    }
    cudaMemset(Sim.Blocksum, 0.0, Sim.Nblocks * sizeof(double));
    */
    Nthread= 128;

    //TotalEwald<<<Nblock, Nthread, Nthread * sizeof(double)>>>(d_a, Box, Sim.Blocksum, eikx, eiky, eikz, Box.tempFrameworkEik, Box.tempEik, NTotalAtom, NAtomPerThread, residueAtoms, NHostGuestthread, SystemComponents.NComponents, Nblock);
    TotalEwald<<<Nblock, Nthread>>>(d_a, Box, Sim.Blocksum, eikx, eiky, eikz, Box.tempFrameworkEik, Box.tempEik, NTotalAtom, NAtomPerThread, residueAtoms, NHostGuestthread, SystemComponents.NComponents, Nblock);
    checkCUDAErrorEwald("Error in Total Ewald Summation\n");
    cudaFree(eikx); cudaFree(eiky); cudaFree(eikz);
   
    //Sometimes kpoints exceed the size of blocksum, so need to reduce it//
    size_t kpoint_per_thread = 5;
    size_t NCudaBlock = Nblock / kpoint_per_thread;
    if(Nblock % kpoint_per_thread != 0) NCudaBlock++;

    size_t COUNT = 0;
    while(3 * NCudaBlock >= Sim.Nblocks)
    {
      kpoint_per_thread *= 2;
      NCudaBlock = Nblock / kpoint_per_thread;
      if(Nblock % kpoint_per_thread != 0) NCudaBlock++;
      COUNT++;
    }
    //printf("Sim.Nblocks: %zu, total kpoints: %zu, CUDAblock: %zu, each thread do %zu kpoints\n", Sim.Nblocks, Nblock, NCudaBlock, kpoint_per_thread);
    TotalEwald_CalculateEnergy<<<NCudaBlock, 128, 128*3*sizeof(double)>>>(Box, Box.tempFrameworkEik, Box.tempEik, Sim.Blocksum, Nblock, kpoint_per_thread, NCudaBlock);
    checkCUDAErrorEwald("Error in summing the energies of fourier part\n");

    double HostTotEwald[NCudaBlock*3]; //HH + HG + GG//
    double HHFourier = 0.0;
    double GGFourier = 0.0;
    double HGFourier = 0.0;

    cudaMemcpy(HostTotEwald, Sim.Blocksum, 3* NCudaBlock * sizeof(double), cudaMemcpyDeviceToHost);
    for(size_t i = 0; i < NCudaBlock; i++) HHFourier += HostTotEwald[i];
    for(size_t i = NCudaBlock;   i < NCudaBlock*2; i++) GGFourier += HostTotEwald[i];
    for(size_t i = NCudaBlock*2; i < NCudaBlock*3; i++) HGFourier += HostTotEwald[i];

    double TOTFourier = 0.0;
    for(size_t i = 0; i < NCudaBlock*3; i++) TOTFourier += HostTotEwald[i];

    //printf("Total Ewald, kmax: %d %d %d\n", Box.kmax.x, Box.kmax.y, Box.kmax.z);
    //printf("GPU fourier, HHFourier: %.5f, GGFourier: %.5f, HGFourier: %.5f, TOTFourier: %.5f\n", HHFourier, GGFourier, HGFourier, TOTFourier);
    E.HHEwaldE = HHFourier;
    E.HGEwaldE = HGFourier;
    E.GGEwaldE = GGFourier;

    //Recalculate exclusion part of Ewald summation//
    size_t Exclusion_Nblock = 0; size_t Exclusion_Nthread = 0;
    //Framework, component zero, assume the backbone of framework has only 1 molecule//
    //Parallelized over atoms//
    //Each block handles the Intra + self for one atom in framework, component zero//
    size_t FrameworkCompZeroAtoms = SystemComponents.Moleculesize[0] * SystemComponents.NumberOfMolecule_for_Component[0];
    if(FrameworkCompZeroAtoms > 0 && SystemComponents.NumberOfMolecule_for_Component[0] == 1)
    {
      size_t IntraInteractions = FrameworkCompZeroAtoms * (FrameworkCompZeroAtoms - 1) / 2;
      size_t InteractionPerThread = 100;
      size_t TotalThreadsNeeded = IntraInteractions / InteractionPerThread + (IntraInteractions % InteractionPerThread == 0 ? 0 : 1);
      Setup_threadblock(TotalThreadsNeeded, &Exclusion_Nblock, &Exclusion_Nthread);
      
      //DEBUG//printf("Component %zu, Atoms: %zu, IntraInteractions: %zu, totalThreads: %zu, Nblock: %zu, Nthread: %zu\n", 0, FrameworkCompZeroAtoms, IntraInteractions, TotalThreadsNeeded, Exclusion_Nblock, Exclusion_Nthread);

      Framework_ComponentZero_Calculate_Intra_Self_Exclusion<<<Exclusion_Nblock, Exclusion_Nthread, Exclusion_Nthread * sizeof(double)>>>(Box, d_a, Sim.Blocksum, IntraInteractions, InteractionPerThread, TotalThreadsNeeded); 
      checkCUDAErrorEwald("error Calculating Intra Self Exclusion (FRAMEWORK) for Ewald Summation for Total Ewald summation!!!");
      
      double FrameworkExclusion = 0.0;
      cudaMemcpy(HostTotEwald, Sim.Blocksum, Exclusion_Nblock * sizeof(double), cudaMemcpyDeviceToHost);
      for(size_t i = 0; i < Exclusion_Nblock; i++) FrameworkExclusion += HostTotEwald[i];
      //printf("Framework Component 0 Exclusion: %.5f\n", FrameworkExclusion);
      E.HHEwaldE += FrameworkExclusion;
    }
    //Framework component > 0 and Adsorbate//
    //Parallelized over molecules//
    for(size_t i = 1; i < SystemComponents.NComponents.x; i++)
    {
      if(SystemComponents.NumberOfMolecule_for_Component[i] == 0) continue;
      Setup_threadblock(SystemComponents.NumberOfMolecule_for_Component[i], &Exclusion_Nblock, &Exclusion_Nthread);
      //printf("Component %zu, Nblock: %zu, Nthread: %zu\n", i, Exclusion_Nblock, Exclusion_Nthread);
      Calculate_Intra_Self_Exclusion<<<Exclusion_Nblock, Exclusion_Nthread, Exclusion_Nthread * sizeof(double)>>>(Box, d_a, Sim.Blocksum, i); 
      checkCUDAErrorEwald("error Calculating Intra Self Exclusion (ADSORBATE) for Ewald Summation for Total Ewald summation!!!");
      //Curently we are assuming that the number of blocks for self+exclusion is smaller than total number of kspace points (99.99999% true)
      //Zhao's note: assume that you use less blocks for intra-self exclusion
      cudaMemcpy(HostTotEwald, Sim.Blocksum, Exclusion_Nblock * sizeof(double), cudaMemcpyDeviceToHost);
      double Component_Exclusion = 0.0;
      for(size_t ii = 0; ii < Exclusion_Nblock; ii++) Component_Exclusion += HostTotEwald[ii];
      //printf("Component %zu, Exclusion (self + intra) = %.5f\n", i, Component_Exclusion);
      if(i < SystemComponents.NComponents.y)
      {
        E.HHEwaldE += Component_Exclusion;
      }
      else E.GGEwaldE += Component_Exclusion;
    }
  }
  return E;
}
