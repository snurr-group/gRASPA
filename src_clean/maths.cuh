#include <iostream>
#define SWAP(x,y,z) {z=(x);x=(y);y=(z);} //RASPA2
__host__ __device__ void WrapInBox(double3 posvec, double* Cell, double* InverseCell, bool Cubic)
{
  if(Cubic)//cubic/cuboid
  {
    posvec.x -= static_cast<int>(posvec.x * InverseCell[0*3+0] + ((posvec.x >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0]; if(posvec.x < 0.0) posvec.x += Cell[0*3+0];
    posvec.y -= static_cast<int>(posvec.y * InverseCell[1*3+1] + ((posvec.y >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1]; if(posvec.y < 0.0) posvec.y += Cell[1*3+1];
    posvec.z -= static_cast<int>(posvec.z * InverseCell[2*3+2] + ((posvec.z >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2]; if(posvec.z < 0.0) posvec.z += Cell[2*3+2];
  }
  else
  {
    double3 s = {0.0, 0.0, 0.0};
    s.x=InverseCell[0*3+0]*posvec.x + InverseCell[1*3+0]*posvec.y + InverseCell[2*3+0]*posvec.z;
    s.y=InverseCell[0*3+1]*posvec.x + InverseCell[1*3+1]*posvec.y + InverseCell[2*3+1]*posvec.z;
    s.z=InverseCell[0*3+2]*posvec.x + InverseCell[1*3+2]*posvec.y + InverseCell[2*3+2]*posvec.z;

    s.x -= static_cast<int>(s.x + ((s.x >= 0.0) ? 0.5 : -0.5)); if(s.x < 0.0) s.x += 1.0;
    s.y -= static_cast<int>(s.y + ((s.y >= 0.0) ? 0.5 : -0.5)); if(s.y < 0.0) s.y += 1.0;
    s.z -= static_cast<int>(s.z + ((s.z >= 0.0) ? 0.5 : -0.5)); if(s.z < 0.0) s.z += 1.0;
    // convert from abc to xyz
    posvec.x=Cell[0*3+0]*s.x+Cell[1*3+0]*s.y+Cell[2*3+0]*s.z;
    posvec.y=Cell[0*3+1]*s.x+Cell[1*3+1]*s.y+Cell[2*3+1]*s.z;
    posvec.z=Cell[0*3+2]*s.x+Cell[1*3+2]*s.y+Cell[2*3+2]*s.z;
  }
}

double matrix_determinant(double* x) //9*1 array
{
  double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
  double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
  double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
  double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
  return determinant;
}

//3x3 matrix//
void inverse_matrix(double* x, double **inverse_x)
{
  double m11 = x[0*3+0]; double m21 = x[1*3+0]; double m31 = x[2*3+0];
  double m12 = x[0*3+1]; double m22 = x[1*3+1]; double m32 = x[2*3+1];
  double m13 = x[0*3+2]; double m23 = x[1*3+2]; double m33 = x[2*3+2];
  double determinant = +m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (m21 * m32 - m22 * m31);
  double* result = (double*) malloc(9 * sizeof(double));
  result[0] = +(m22 * m33 - m32 * m23) / determinant;
  result[3] = -(m21 * m33 - m31 * m23) / determinant;
  result[6] = +(m21 * m32 - m31 * m22) / determinant;
  result[1] = -(m12 * m33 - m32 * m13) / determinant;
  result[4] = +(m11 * m33 - m31 * m13) / determinant;
  result[7] = -(m11 * m32 - m31 * m12) / determinant;
  result[2] = +(m12 * m23 - m22 * m13) / determinant;
  result[5] = -(m11 * m23 - m21 * m13) / determinant;
  result[8] = +(m11 * m22 - m21 * m12) / determinant;
  *inverse_x = result;
}
//Converted from RASPA2
//inversed matrix is saved in the input matrix a, temp_matrix b is a temporary, not used
void GaussJordan(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& b)
{
  size_t n = a.size();
  size_t m = b.size();
  std::vector<size_t> indxc(n, 0.0), indxr(n, 0.0), ipiv(n, 0.0);
  size_t icol, irow;
  icol = irow = 0;
  double big, dum, pivinv, temp;

  //for (size_t j = 0; j < n; j++) ipiv[j] = 0;
  for (size_t i = 0; i < n; i++) 
  {
    big = 0.0;
    for (size_t j = 0; j < n; j++) 
    {
      if (ipiv[j] != 1) 
      {
        for (size_t k = 0; k < n; k++) 
        {
          if (ipiv[k] == 0) 
          {
            if (std::fabs(a[j][k]) >= big)
            {
              big = std::fabs(a[j][k]);
              irow = j;
              icol = k;
            }
          }
        }
      }
    }
    ++(ipiv[icol]);

    if (irow != icol) 
    {
      for (size_t l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l], temp);
      for (size_t l = 0; l < m; l++) SWAP(b[irow][l], b[icol][l], temp);
    }
    indxr[i] = irow;
    indxc[i] = icol;
    //if (a[icol][icol] < 1e-10)
    if (a[icol][icol] == 0.0)
    {
      //throw std::runtime_error("Matrix Inversion, Gauss-Jordan: Singular Matrix");
      printf("WARNING: Matrix Inversion, Gauss-Jordan: Singular Matrix! Qst for the current component may not get calculated correctly!\n");
      printf("TIP: Your molecule may have a hard time getting in/out of the system box. Consider making it better by using some moves or NVT moves to relax the system more!\n");
    }
    pivinv = 1.0 / a[icol][icol];
    a[icol][icol] = 1.0;
    for (size_t l = 0; l < n; l++) a[icol][l] *= pivinv;
    for (size_t l = 0; l < m; l++) b[icol][l] *= pivinv;
    for (size_t ll = 0; ll < n; ll++) 
    {
      if (ll != icol) 
      {
        dum = a[ll][icol];
        a[ll][icol] = 0.0;
        for (size_t l = 0; l < n; l++) a[ll][l] -= a[icol][l] * dum;
        for (size_t l = 0; l < m; l++) b[ll][l] -= b[icol][l] * dum;
      }
    }
  }
  for (int l = n - 1; l >= 0; l--)
  //for (size_t l = 0; l <= n-1; l++)
  {
    if (indxr[l] != indxc[l])
    {
      for (size_t k = 0; k < n; k++)
      {
        SWAP(a[k][indxr[l]], a[k][indxc[l]], temp);
      }
    }
  }
}

inline __host__ __device__ void matrix_multiply_by_vector(double* a, double3 b, double3 &c) //3x3(9*1) matrix (a) times 3x1(3*1) vector (b), a*b=c//
{
  c.x=a[0*3+0]*b.x + a[1*3+0]*b.y + a[2*3+0]*b.z;
  c.y=a[0*3+1]*b.x + a[1*3+1]*b.y + a[2*3+1]*b.z;
  c.z=a[0*3+2]*b.x + a[1*3+2]*b.y + a[2*3+2]*b.z;
}

// HIP's double3 (HIP_vector_type) already defines these componentwise
// operators, so redefining them would clash. Under nvcc the guard is true
// and the definitions are emitted verbatim. dot() (scalar return) and the
// MoveEnergy operators below are NOT provided by HIP and stay unguarded.
#if !defined(__HIP__)
__host__ __device__ void operator +=(double3 &a, double3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__host__ __device__ void operator +=(double3 &a, double b)
{
  a.x += b;
  a.y += b;
  a.z += b;
}


__host__ __device__ void operator -=(double3 &a, double3 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__host__ __device__ void operator -=(double3 &a, double b)
{
  a.x -= b;
  a.y -= b;
  a.z -= b;
}


__host__ __device__ void operator *=(double3 &a, double3 b)
{
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
}

__host__ __device__ void operator *=(double3 &a, double b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__host__ __device__ double3 operator +(double3 a, double3 b)
{
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ double3 operator +(double3 a, double b)
{
  return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ double3 operator -(double3 a, double3 b)
{
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ double3 operator -(double3 a, double b)
{
  return {a.x - b, a.y - b, a.z - b};
}

__host__ __device__ double3 operator *(double3 a, double3 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ double3 operator *(double3 a, double b)
{
    return {a.x * b, a.y * b, a.z * b};
}
#endif // !defined(__HIP__)

__host__ __device__ double dot(double3 a, double3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ MoveEnergy sqrt_MoveEnergy(MoveEnergy A)
{
  MoveEnergy X;
  X.storedHGVDW    = sqrt(A.storedHGVDW);
  X.storedHGReal   = sqrt(A.storedHGReal);
  X.storedHGEwaldE = sqrt(A.storedHGEwaldE);

  X.HHVDW    = sqrt(A.HHVDW);
  X.HGVDW    = sqrt(A.HGVDW);    
  X.GGVDW    = sqrt(A.GGVDW);  

  X.HHReal   = sqrt(A.HHReal);  
  X.HGReal   = sqrt(A.HGReal);   
  X.GGReal   = sqrt(A.GGReal);   

  X.HHEwaldE = sqrt(A.HHEwaldE); 
  X.HGEwaldE = sqrt(A.HGEwaldE); 
  X.GGEwaldE = sqrt(A.GGEwaldE);

  X.TailE    = sqrt(A.TailE);  
  X.DNN_E    = sqrt(A.DNN_E); 
  return X;
}

__host__ MoveEnergy operator *(MoveEnergy A, double B)
{
  MoveEnergy X;
  X.storedHGVDW     = A.storedHGVDW    * B;
  X.storedHGReal    = A.storedHGReal   * B;
  X.storedHGEwaldE  = A.storedHGEwaldE * B;

  X.HHVDW    = A.HHVDW * B;
  X.HGVDW    = A.HGVDW * B;
  X.GGVDW    = A.GGVDW * B;

  X.HHReal   = A.HHReal * B;
  X.HGReal   = A.HGReal * B;
  X.GGReal   = A.GGReal * B;

  X.HHEwaldE = A.HHEwaldE * B;
  X.HGEwaldE = A.HGEwaldE * B;
  X.GGEwaldE = A.GGEwaldE * B;

  X.TailE    = A.TailE * B;
  X.DNN_E    = A.DNN_E * B;
  return X;
}

__host__ MoveEnergy operator *(MoveEnergy A, MoveEnergy B)
{
  MoveEnergy X;
  X.storedHGVDW    = A.storedHGVDW * B.storedHGVDW;
  X.storedHGReal   = A.storedHGReal * B.storedHGReal;
  X.storedHGEwaldE = A.storedHGEwaldE * B.storedHGEwaldE;

  X.HHVDW    = A.HHVDW    * B.HHVDW;
  X.HGVDW    = A.HGVDW    * B.HGVDW;
  X.GGVDW    = A.GGVDW    * B.GGVDW;

  X.HHReal   = A.HHReal   * B.HHReal;
  X.HGReal   = A.HGReal   * B.HGReal;
  X.GGReal   = A.GGReal   * B.GGReal;

  X.HHEwaldE = A.HHEwaldE * B.HHEwaldE;
  X.HGEwaldE = A.HGEwaldE * B.HGEwaldE;
  X.GGEwaldE = A.GGEwaldE * B.GGEwaldE;

  X.TailE    = A.TailE    * B.TailE;
  X.DNN_E    = A.DNN_E    * B.DNN_E;
  return X;
}

__host__ MoveEnergy operator /(MoveEnergy A, double B)
{
  MoveEnergy X;
  double OneOverB = 1.0 / B;
  X.storedHGVDW     = A.storedHGVDW    * OneOverB;
  X.storedHGReal    = A.storedHGReal   * OneOverB;
  X.storedHGEwaldE  = A.storedHGEwaldE * OneOverB;

  X.HHVDW    = A.HHVDW * OneOverB;
  X.HGVDW    = A.HGVDW * OneOverB;
  X.GGVDW    = A.GGVDW * OneOverB;

  X.HHReal   = A.HHReal * OneOverB;
  X.HGReal   = A.HGReal * OneOverB;
  X.GGReal   = A.GGReal * OneOverB;

  X.HHEwaldE = A.HHEwaldE * OneOverB;
  X.HGEwaldE = A.HGEwaldE * OneOverB;
  X.GGEwaldE = A.GGEwaldE * OneOverB;

  X.TailE    = A.TailE * OneOverB;
  X.DNN_E    = A.DNN_E * OneOverB;
  return X;
}

__host__ MoveEnergy MoveEnergy_DIVIDE_DOUBLE(MoveEnergy A, double B)
{
  MoveEnergy X;
  double OneOverB = 1.0 / B;
  X.storedHGVDW     = A.storedHGVDW    * OneOverB;
  X.storedHGReal    = A.storedHGReal   * OneOverB;
  X.storedHGEwaldE  = A.storedHGEwaldE * OneOverB;

  X.HHVDW    = A.HHVDW * OneOverB;
  X.HGVDW    = A.HGVDW * OneOverB;
  X.GGVDW    = A.GGVDW * OneOverB;

  X.HHReal   = A.HHReal * OneOverB;
  X.HGReal   = A.HGReal * OneOverB;
  X.GGReal   = A.GGReal * OneOverB;

  X.HHEwaldE = A.HHEwaldE * OneOverB;
  X.HGEwaldE = A.HGEwaldE * OneOverB;
  X.GGEwaldE = A.GGEwaldE * OneOverB;

  X.TailE    = A.TailE * OneOverB;
  X.DNN_E    = A.DNN_E * OneOverB;
  return X;
}

__host__ MoveEnergy MoveEnergy_Multiply(MoveEnergy A, MoveEnergy B)
{
  MoveEnergy X;
  X.storedHGVDW    = A.storedHGVDW * B.storedHGVDW;
  X.storedHGReal   = A.storedHGReal * B.storedHGReal;
  X.storedHGEwaldE = A.storedHGEwaldE * B.storedHGEwaldE;

  X.HHVDW    = A.HHVDW    * B.HHVDW;
  X.HGVDW    = A.HGVDW    * B.HGVDW;
  X.GGVDW    = A.GGVDW    * B.GGVDW;

  X.HHReal   = A.HHReal   * B.HHReal;
  X.HGReal   = A.HGReal   * B.HGReal;
  X.GGReal   = A.GGReal   * B.GGReal;

  X.HHEwaldE = A.HHEwaldE * B.HHEwaldE;
  X.HGEwaldE = A.HGEwaldE * B.HGEwaldE;
  X.GGEwaldE = A.GGEwaldE * B.GGEwaldE;

  X.TailE    = A.TailE    * B.TailE;
  X.DNN_E    = A.DNN_E    * B.DNN_E;
  return X;
}

__host__ void operator +=(MoveEnergy& A, MoveEnergy B)
{
  A.storedHGVDW     += B.storedHGVDW;
  A.storedHGReal    += B.storedHGReal;
  A.storedHGEwaldE  += B.storedHGEwaldE;

  A.HHVDW    += B.HHVDW;
  A.HGVDW    += B.HGVDW;
  A.GGVDW    += B.GGVDW;

  A.HHReal   += B.HHReal;
  A.HGReal   += B.HGReal;
  A.GGReal   += B.GGReal;

  A.HHEwaldE += B.HHEwaldE;
  A.HGEwaldE += B.HGEwaldE;
  A.GGEwaldE += B.GGEwaldE;

  A.TailE    += B.TailE;
  A.DNN_E    += B.DNN_E;
}

__host__ void operator -=(MoveEnergy& A, MoveEnergy B)
{
  A.storedHGVDW     -= B.storedHGVDW;
  A.storedHGReal    -= B.storedHGReal;
  A.storedHGEwaldE  -= B.storedHGEwaldE;
  
  A.HHVDW    -= B.HHVDW;
  A.HGVDW    -= B.HGVDW;
  A.GGVDW    -= B.GGVDW;

  A.HHReal   -= B.HHReal;
  A.HGReal   -= B.HGReal;
  A.GGReal   -= B.GGReal;

  A.HHEwaldE -= B.HHEwaldE;
  A.HGEwaldE -= B.HGEwaldE;
  A.GGEwaldE -= B.GGEwaldE;

  A.TailE    -= B.TailE;
  A.DNN_E    -= B.DNN_E;
}

__host__ MoveEnergy operator +(MoveEnergy A, MoveEnergy B)
{
  MoveEnergy X; X = A;
  X += B; return X;
}

__host__ MoveEnergy operator -(MoveEnergy A, MoveEnergy B)
{
  MoveEnergy X; X = A;
  X -= B; return X;
}

__host__ __device__ void PBC(double3& posvec, double* Cell, double* InverseCell, bool Cubic)
{
  if(Cubic)//cubic/cuboid
  {
    posvec.x -= static_cast<int>(posvec.x * InverseCell[0*3+0] + ((posvec.x >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
    posvec.y -= static_cast<int>(posvec.y * InverseCell[1*3+1] + ((posvec.y >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
    posvec.z -= static_cast<int>(posvec.z * InverseCell[2*3+2] + ((posvec.z >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
  }
  else
  {
    double3 s = {0.0, 0.0, 0.0};
    s.x=InverseCell[0*3+0]*posvec.x + InverseCell[1*3+0]*posvec.y + InverseCell[2*3+0]*posvec.z;
    s.y=InverseCell[0*3+1]*posvec.x + InverseCell[1*3+1]*posvec.y + InverseCell[2*3+1]*posvec.z;
    s.z=InverseCell[0*3+2]*posvec.x + InverseCell[1*3+2]*posvec.y + InverseCell[2*3+2]*posvec.z;

    s.x -= static_cast<int>(s.x + ((s.x >= 0.0) ? 0.5 : -0.5));
    s.y -= static_cast<int>(s.y + ((s.y >= 0.0) ? 0.5 : -0.5));
    s.z -= static_cast<int>(s.z + ((s.z >= 0.0) ? 0.5 : -0.5));
    // convert from abc to xyz
    posvec.x=Cell[0*3+0]*s.x+Cell[1*3+0]*s.y+Cell[2*3+0]*s.z;
    posvec.y=Cell[0*3+1]*s.x+Cell[1*3+1]*s.y+Cell[2*3+1]*s.z;
    posvec.z=Cell[0*3+2]*s.x+Cell[1*3+2]*s.y+Cell[2*3+2]*s.z;
  }
}

static inline __host__ __device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result, bool use1264 = false) //Lennard-Jones (default) or 12-6-4 polynomial
{
  if(use1264)
  {
    // ---- Polynomial VDW mode (Use1264=true) ----
    // FFarg stores pre-computed coefficients: {C12, C6, C4, shift, C10}
    // U(r) = C12/r^12 - C6/r^6 + C10/r^10 + C4/r^4 - shift
    // Activated by "UseLJ1264 yes" in simulation.input
    // Supports: standard LJ (C4=C10=0), 12-6-4 LJ, GENERIC2_HC (Du et al. JCTC 2020)
    double C12   = FFarg[0];
    double C6    = FFarg[1];
    double C4    = FFarg[2];
    double shift = FFarg[3];
    double C10   = FFarg[4];

    double ri2  = 1.0 / rr_dot;
    double ri4  = ri2 * ri2;
    double ri6  = ri4 * ri2;
    double ri10 = ri4 * ri6;
    double ri12 = ri6 * ri6;

    double term = C12 * ri12 - C6 * ri6 + C10 * ri10 + C4 * ri4 - shift;
    result[0] = scaling * term;
    result[1] = 0.0;
  }
  else
  {
    // ---- Original LJ mode (Use1264=false, default) ----
    // FFarg stores: {epsilon, sigma, z(unused), shift}
    // Standard 12-6 LJ with soft-core lambda scaling
    double arg1 = 4.0 * FFarg[0];
    double arg2 = FFarg[1] * FFarg[1];
    double arg3 = FFarg[3];
    double temp = (rr_dot / arg2);
    double temp3 = temp * temp * temp;
    double rri3 = 1.0 / (temp3 + 0.5 * (1.0 - scaling) * (1.0 - scaling));
    double rri6 = rri3 * rri3;
    double term = arg1 * (rri3 * (rri3 - 1.0)) - arg3;
    double dlambda_term = scaling * arg1 * (rri6 * (2.0 * rri3 - 1.0));
    result[0] = scaling * term;
    result[1] = scaling < 1.0 ? term + (1.0 - scaling) * dlambda_term : 0.0;
  }
}

static inline __host__ __device__ void CoulombReal(const double chargeA, const double chargeB, const double r, const double scaling, double* result, double prefactor, double alpha) //energy = -q1*q2/r
{
  double term      = chargeA * chargeB * std::erfc(alpha * r);
         result[0] = prefactor * scaling * term / r;
}
