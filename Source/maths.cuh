#include <iostream>

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


inline __host__ __device__ void matrix_multiply_by_vector(double* a, double3 b, double3 &c) //3x3(9*1) matrix (a) times 3x1(3*1) vector (b), a*b=c//
{
  c.x=a[0*3+0]*b.x + a[1*3+0]*b.y + a[2*3+0]*b.z;
  c.y=a[0*3+1]*b.x + a[1*3+1]*b.y + a[2*3+1]*b.z;
  c.z=a[0*3+2]*b.x + a[1*3+2]*b.y + a[2*3+2]*b.z;
}

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


__host__ __device__ double dot(double3 a, double3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ void operator +=(MoveEnergy& A, MoveEnergy B)
{
  A.storedHGVDW     += B.storedHGVDW;
  A.storedHGReal    += B.storedHGReal;
  A.storedHGEwaldE  += B.storedHGEwaldE;

  A.HGVDW     += B.HGVDW;
  A.HGReal    += B.HGReal;
  A.GGVDW     += B.GGVDW;
  A.GGReal    += B.GGReal;
  A.EwaldE    += B.EwaldE;
  A.HGEwaldE  += B.HGEwaldE;
  A.TailE     += B.TailE;
  A.DNN_E     += B.DNN_E;
}

__host__ void operator -=(MoveEnergy& A, MoveEnergy B)
{
  A.storedHGVDW     -= B.storedHGVDW;
  A.storedHGReal    -= B.storedHGReal;
  A.storedHGEwaldE  -= B.storedHGEwaldE;

  A.HGVDW     -= B.HGVDW;
  A.HGReal    -= B.HGReal;
  A.GGVDW     -= B.GGVDW;
  A.GGReal    -= B.GGReal;
  A.EwaldE    -= B.EwaldE;
  A.HGEwaldE  -= B.HGEwaldE;
  A.TailE     -= B.TailE;
  A.DNN_E     -= B.DNN_E;
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

static inline __host__ __device__ void VDW(const double* FFarg, const double rr_dot, const double scaling, double* result) //Lennard-Jones 12-6
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

static inline __host__ __device__ void CoulombReal(const double chargeA, const double chargeB, const double r, const double scaling, double* result, double prefactor, double alpha) //energy = -q1*q2/r
{
  double term      = chargeA * chargeB * std::erfc(alpha * r);
         result[0] = prefactor * scaling * term / r;
}
