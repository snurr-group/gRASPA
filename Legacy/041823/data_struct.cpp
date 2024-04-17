#include "data_struct.h"
#include <cstdlib>
#include <random>

void VDW_CPU(const double* FFarg, const double rr_dot, const double scaling, double* result) //Lennard-Jones 12-6
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

void PBC_CPU(double* posvec, double* Cell, double* InverseCell, bool& Cubic)
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

double get_random_from_zero_to_one()
{
  //return (double) (rand()/RAND_MAX);
  //std::srand(3.0);
  return static_cast<double>(std::rand()) / RAND_MAX;
}
