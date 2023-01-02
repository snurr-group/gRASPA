#include <omp.h>
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

//////////////////////////////////////////////////////////
// General Functions for User-defined Complex Variables //
//////////////////////////////////////////////////////////
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
    }
  }
  else if(MoveType == INSERTION || MoveType == CBCF_INSERTION) // Insertion & Fractional Insertion //
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
    }
  }
  else if(MoveType == DELETION || MoveType == CBCF_DELETION) // Deletion //
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
    }
  }
  Initialize_Vectors(Box, Oldsize, Newsize, Old, numberOfAtoms, kmax);
}

__global__ void Initialize_EwaldVector_LambdaChange(Boxsize Box, int3 kmax, Atoms* d_a, Atoms Old, size_t Oldsize, double2 newScale)
{
  for(size_t i = Oldsize; i < Oldsize + Oldsize; i++) //chainsize here is the total size of the molecule for translation/rotation
  {
    Old.x[i]             = Old.x[i - Oldsize];
    Old.y[i]             = Old.y[i - Oldsize];
    Old.z[i]             = Old.z[i - Oldsize];
    Old.scale[i]         = newScale.x;
    Old.charge[i]        = Old.charge[i - Oldsize];
    Old.scaleCoul[i]     = newScale.y;

  }
  Initialize_Vectors(Box, Oldsize, Oldsize, Old, Oldsize + Oldsize, kmax);
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

__global__ void JustStore_Ewald(Boxsize Box, size_t nvec)
{
  size_t i         = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < nvec) Box.totalEik[i] = Box.storedEik[i];
}

///////////////////////////////////////////////////////////////
// CALCULATE FOURIER PART OF THE COULOMBIC ENERGY FOR A MOVE //
///////////////////////////////////////////////////////////////

__global__ void Fourier_Ewald_Diff(Boxsize Box, Atoms Old, double alpha_squared, double prefactor, int3 kmax, double recip_cutoff, size_t Oldsize, size_t Newsize, double* Blocksum, bool UseTempVector)
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
    //Box.totalEik[kxyz] = Box.storedEik[kxyz];
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
      Complex newV; Complex OldV;
      //Zhao's note: this is for CBCF insertion, where insertion is the second step. The intermediate Eik is totalEik, is the one we should use//
      if(UseTempVector)
      {
        OldV.real = Box.totalEik[kxyz].real;  OldV.imag = Box.totalEik[kxyz].imag;
      }
      else
      {
        OldV.real = Box.storedEik[kxyz].real; OldV.imag = Box.storedEik[kxyz].imag;
      }
      newV.real          = OldV.real + cksum_new.real - cksum_old.real;
      newV.imag          = OldV.imag + cksum_new.imag - cksum_old.imag;
      tempE             += temp * ComplexNorm(newV);
      tempE             -= temp * ComplexNorm(OldV);
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

void Skip_Ewald(Boxsize& Box)
{
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_EW(numberOfWaveVectors, &Nblock, &Nthread);
  JustStore_Ewald<<<Nblock, Nthread>>>(Box, numberOfWaveVectors);
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

////////////////////////////////////////////////
// Main Ewald Functions (Fourier + Exclusion) //
////////////////////////////////////////////////
//Zhao's note: Currently this only works for INSERTION, not even tested with Deletion//
double GPU_EwaldDifference_General(Boxsize& Box, Atoms*& d_a, Atoms& New, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, int MoveType, size_t Location, double2 newScale)
{
  if(FF.noCharges) return 0.0;
  //cudaDeviceSynchronize();
  double start = omp_get_wtime();
  double alpha = FF.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = FF.Prefactor * (2.0 * M_PI / Box.Volume);

  size_t Oldsize = 0; size_t Newsize = 0; size_t chainsize = 0;
  bool UseTempVector = false; //Zhao's note: Whether or not to use the temporary Vectors (Only used for CBCF Insertion in this function)//
  switch(MoveType)
  {
    case TRANSLATION_ROTATION: // Translation/Rotation Move //
    {
      Oldsize   = SystemComponents.Moleculesize[SelectedComponent];
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent];
      break;
    }
    case INSERTION: // Insertion //
    {
      Oldsize   = 0;
      Newsize   = SystemComponents.Moleculesize[SelectedComponent];
      chainsize = SystemComponents.Moleculesize[SelectedComponent] - 1;
      break;
    }
    case DELETION: // Deletion //
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

  Initialize_EwaldVector_General<<<1,1>>>(Box, Box.kmax, d_a, New, Old, Oldsize, Newsize, SelectedComponent, Location, chainsize, numberOfAtoms, MoveType); //checkCUDAError("error Initializing Ewald Vectors");

  //Fourier Loop//
  size_t numberOfWaveVectors = (Box.kmax.x + 1) * (2 * Box.kmax.y + 1) * (2 * Box.kmax.z + 1);
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_EW(numberOfWaveVectors, &Nblock, &Nthread);
  Fourier_Ewald_Diff<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Old, alpha_squared, prefactor, Box.kmax, Box.ReciprocalCutOff, Oldsize, Newsize, Blocksum, UseTempVector); //checkCUDAError("error Doing Fourier");
  double Host_sum[Nblock]; double tot = 0.0;
  cudaMemcpy(Host_sum, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++){tot += Host_sum[i];}
  //Zhao's note: when adding fractional molecules, this might not be correct//
  double deltaExclusion = 0.0;
  //printf("GPU Fourier Energy: %.5f\n", tot);
  if(SystemComponents.rigid[SelectedComponent])
  {
    if(MoveType == INSERTION) // Insertion //
    {
      //Zhao's note: This is a bit messy, because when creating the molecules at the beginning of the simulation, we need to create a fractional molecule//
      //MoveType is 2, not 4. 4 is for the insertion after making the old fractional molecule full.//
      double delta_scale = std::pow(newScale.y,2) - 0.0;
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      tot -= deltaExclusion;
    }
    else if(MoveType == DELETION) // Deletion //
    {
      double delta_scale = 0.0 - 1.0;
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      tot -= deltaExclusion;
    }
    else if(MoveType == CBCF_INSERTION) // CBCF Lambda Insertion //
    {
      double delta_scale = std::pow(newScale.y,2) - 0.0;
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      tot -= deltaExclusion;
    }
    else if(MoveType == CBCF_DELETION) // CBCF Lambda Deletion //
    {
      double delta_scale = 0.0 - std::pow(newScale.y,2);
      deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
      tot -= deltaExclusion;
    }
  }
  //cudaDeviceSynchronize();
  double end = omp_get_wtime();
  return tot;
}

//Zhao's note: THIS IS A SPECIAL FUNCTION JUST FOR REINSERTION//
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
  Fourier_Ewald_Diff<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Old, alpha_squared, prefactor, Box.kmax, Box.ReciprocalCutOff, Oldsize, Newsize, Blocksum, false);
  double Host_sum[Nblock]; double tot = 0.0;
  cudaMemcpy(Host_sum, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++){tot += Host_sum[i];}
  return tot;
}

//Zhao's note: THIS IS A SPECIAL FUNCTION JUST FOR LAMBDA MOVE FOR FRACTIONAL MOLECULES//
double GPU_EwaldDifference_LambdaChange(Boxsize& Box, Atoms*& d_a, Atoms& Old, ForceField& FF, double* Blocksum, Components& SystemComponents, size_t SelectedComponent, double2 oldScale, double2 newScale, int MoveType)
{
  if(FF.noCharges) return 0.0;
  double alpha = FF.Alpha; double alpha_squared = alpha * alpha;
  double prefactor = FF.Prefactor * (2.0 * M_PI / Box.Volume);

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
  size_t Nblock = 0; size_t Nthread = 0; Setup_threadblock_EW(numberOfWaveVectors, &Nblock, &Nthread);
  Fourier_Ewald_Diff<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Old, alpha_squared, prefactor, Box.kmax, Box.ReciprocalCutOff, Oldsize, Newsize, Blocksum, UseTempVector); //checkCUDAError("error Doing Fourier");
  //Fourier_Ewald_Diff_LambdaChange<<<Nblock, Nthread, Nthread * sizeof(double)>>>(Box, Old, alpha_squared, prefactor, Box.kmax, Box.ReciprocalCutOff, Oldsize, newscale, Blocksum, UseTempVector);
  double Host_sum[Nblock]; double tot = 0.0;
  cudaMemcpy(Host_sum, Blocksum, Nblock * sizeof(double), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < Nblock; i++){tot += Host_sum[i];}
  //printf("Fourier GPU lambda Change: %.5f\n", tot);
  double delta_scale = std::pow(newScale.y,2) - std::pow(oldScale.y,2);
  double deltaExclusion = (SystemComponents.ExclusionIntra[SelectedComponent] + SystemComponents.ExclusionAtom[SelectedComponent]) * delta_scale;
  tot -= deltaExclusion;
  return tot;
}
