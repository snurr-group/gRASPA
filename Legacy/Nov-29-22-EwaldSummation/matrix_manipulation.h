#include <vector>

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
  std::vector<double>inverse(9);
  inverse[0] = +(m22 * m33 - m32 * m23) / determinant;
  inverse[3] = -(m21 * m33 - m31 * m23) / determinant;
  inverse[6] = +(m21 * m32 - m31 * m22) / determinant;
  inverse[1] = -(m12 * m33 - m32 * m13) / determinant;
  inverse[4] = +(m11 * m33 - m31 * m13) / determinant;
  inverse[7] = -(m11 * m32 - m31 * m12) / determinant;
  inverse[2] = +(m12 * m23 - m22 * m13) / determinant;
  inverse[5] = -(m11 * m23 - m21 * m13) / determinant;
  inverse[8] = +(m11 * m22 - m21 * m12) / determinant;
  double* result;
  result = Doubleconvert1DVectortoArray(inverse);
  *inverse_x = result;
}
