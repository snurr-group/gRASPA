#include "framework_energy_difference.h"

#include <omp.h>

#include <array>

#include <algorithm>

std::array<double,3> computeFrameworkMoleculeEnergyDifferenceGPU(double* CellArray, double* InverseCellArray, double* FrameworkArray, size_t* FrameworkTypeArray, double* FFArray, int* FFTypeArray, double* MolArray, size_t* MolTypeArray, double* NewMolArray, double* FFParams, int* OtherParams, size_t Frameworksize, size_t Molsize, size_t FFsize, bool noCharges)
{
  std::array<double,3> energy_result{0.0, 0.0, 0.0};
  
  double energy_result_array[3] = {0.0, 0.0, 0.0};
/*
  const int length = 1000000;
  double NX[length];
  for (int i = 0; i < length; i++)
  {
    NX[i] = 1.234*static_cast<double> (i);
  }
  #pragma omp target teams distribute parallel for reduction(+:energy_result_array[:3]) map(to: NX[:length])
  for (int i = 0; i < length; i++)
  {
    double result = NX[i] - static_cast<int>(NX[i] * InverseCellArray[0*3+0] + ((NX[i] >= 0.0) ? 0.5 : -0.5)) * CellArray[0*3+0];
    energy_result_array[0] += result*FFParams[3]/static_cast<double> (FFsize);
  }
*/
    #pragma omp target teams distribute parallel for reduction(+:energy_result_array[:3]) firstprivate(noCharges, FFsize, Molsize) schedule(static,1) //GPU CLAUSE
  for (size_t i = 0; i < Frameworksize; i++)
  {
    const double scaleA = FrameworkArray[i*6+3];
    const double chargeA = FrameworkArray[i*6+4];
    const double scalingCoulombA = FrameworkArray[i*6+5];
    const size_t typeA = FrameworkTypeArray[i];
    for (size_t j = 0; j < Molsize; j++)
    {

      std::array<double, 3> posvec{FrameworkArray[i*6+0] - NewMolArray[j*6+0], FrameworkArray[i*6+1] - NewMolArray[j*6+1], FrameworkArray[i*6+2] - NewMolArray[j*6+2]};
      //posvec = PBC(CellArray, InverseCellArray, posvec);
      //MANUAL INLINE PBC//
      switch (OtherParams[0])
      {
      case 0:
      {
        posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCellArray[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * CellArray[0*3+0];
        posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCellArray[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * CellArray[1*3+1];
        posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCellArray[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * CellArray[2*3+2];
        break;
      }
      default:
      {
        std::array<double,3> s{ 0.0, 0.0, 0.0 };
        s[0]=InverseCellArray[0*3+0]*posvec[0]+InverseCellArray[1*3+0]*posvec[1]+InverseCellArray[2*3+0]*posvec[2];
        s[1]=InverseCellArray[0*3+1]*posvec[0]+InverseCellArray[1*3+1]*posvec[1]+InverseCellArray[2*3+1]*posvec[2];
        s[2]=InverseCellArray[0*3+2]*posvec[0]+InverseCellArray[1*3+2]*posvec[1]+InverseCellArray[2*3+2]*posvec[2];

        s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
        s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
        s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
        // convert from abc to xyz
        posvec[0]=CellArray[0*3+0]*s[0]+CellArray[1*3+0]*s[1]+CellArray[2*3+0]*s[2];
        posvec[1]=CellArray[0*3+1]*s[0]+CellArray[1*3+1]*s[1]+CellArray[2*3+1]*s[2];
        posvec[2]=CellArray[0*3+2]*s[0]+CellArray[1*3+2]*s[1]+CellArray[2*3+2]*s[2];
        break;
      }
      }
      
      double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      if(rr_dot < FFParams[1])
      {
        size_t typeB = MolTypeArray[j];
        double scaleB = NewMolArray[j*6+3];
        //calculate vdw energy
        double scaling = scaleA * scaleB;
  
        size_t row = typeA*FFsize*4+typeB*4; 
        std::array<double,4> FFarg = {FFArray[row+0], FFArray[row+1], FFArray[row+2], FFArray[row+3]};
        //std::array<double,2> result = GPUpotentialVDW(FFarg, scaling, rr_dot);
        //MANUAL INLINE VDW //
        std::array<double,2> result; 
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
        
        energy_result_array[0] += result[0];
        energy_result_array[2] += result[1];
      }
      if (!noCharges && rr_dot < FFParams[2])
      {
        double prefactor = FFParams[3];
        double alpha = FFParams[4];
        double chargeB = NewMolArray[j*6+4];
        double scalingCoulombB = NewMolArray[j*6+5];
        double r = std::sqrt(rr_dot);
        double scaling = scalingCoulombA * scalingCoulombB;
        std::array<double,2> result;
        double term = chargeA * chargeB * std::erfc(alpha * r);
        result[0] = scaling * term / r; result[1] = 0.0;
        energy_result_array[1] += prefactor * result[0];
        energy_result_array[2] += result[1];
      }
    }
    //old//
    for (size_t j = 0; j < Molsize; j++)
    {
      std::array<double, 3> posvec{FrameworkArray[i*6+0] - MolArray[j*6+0], FrameworkArray[i*6+1] - MolArray[j*6+1], FrameworkArray[i*6+2] - MolArray[j*6+2]};
      //posvec = PBC(CellArray, InverseCellArray, posvec);
      //MANUAL INLINE PBC//
      switch (OtherParams[0])
      {
      case 0:
      {
        posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCellArray[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * CellArray[0*3+0];
        posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCellArray[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * CellArray[1*3+1];
        posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCellArray[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * CellArray[2*3+2];
        break;
      }
      default:
      {
        std::array<double,3> s{ 0.0, 0.0, 0.0 };
        s[0]=InverseCellArray[0*3+0]*posvec[0]+InverseCellArray[1*3+0]*posvec[1]+InverseCellArray[2*3+0]*posvec[2];
        s[1]=InverseCellArray[0*3+1]*posvec[0]+InverseCellArray[1*3+1]*posvec[1]+InverseCellArray[2*3+1]*posvec[2];
        s[2]=InverseCellArray[0*3+2]*posvec[0]+InverseCellArray[1*3+2]*posvec[1]+InverseCellArray[2*3+2]*posvec[2];

        s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
        s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
        s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
        // convert from abc to xyz
        posvec[0]=CellArray[0*3+0]*s[0]+CellArray[1*3+0]*s[1]+CellArray[2*3+0]*s[2];
        posvec[1]=CellArray[0*3+1]*s[0]+CellArray[1*3+1]*s[1]+CellArray[2*3+1]*s[2];
        posvec[2]=CellArray[0*3+2]*s[0]+CellArray[1*3+2]*s[1]+CellArray[2*3+2]*s[2];
        break;
      }
      }
 
      double rr_dot = posvec[0]*posvec[0] + posvec[1]*posvec[1] + posvec[2]*posvec[2];
      if(rr_dot < FFParams[1])
      {
        size_t typeB = MolTypeArray[j];
        double scaleB = MolArray[j*6+3];
        //calculate vdw energy
        double scaling = scaleA * scaleB;

        size_t row = typeA*FFsize*4+typeB*4;
        std::array<double,4> FFarg = {FFArray[row+0], FFArray[row+1], FFArray[row+2], FFArray[row+3]};
        //std::array<double,2> result = GPUpotentialVDW(FFarg, scaling, rr_dot);
        //MANUAL INLINE VDW //
        std::array<double,2> result;
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
        energy_result_array[0] -= result[0];
        energy_result_array[2] -= result[1];
      }
      if (!noCharges && rr_dot < FFParams[2])
      {
        double prefactor = FFParams[3];
        double alpha = FFParams[4];
        double chargeB = MolArray[j*6+4];
        double scalingCoulombB = MolArray[j*6+5];
        double r = std::sqrt(rr_dot);
        double scaling = scalingCoulombA * scalingCoulombB;
        std::array<double,2> result;
        double term = chargeA * chargeB * std::erfc(alpha * r);
        result[0] = scaling * term / r; result[1] = 0.0;
        energy_result_array[1] -= prefactor * result[0];
        energy_result_array[2] -= result[1];
      }
    }
  }
  //printf("New Code Energy is %.5f\n", energy_result[0]);
  energy_result[0] = energy_result_array[0]; energy_result[1] = energy_result_array[1];
  energy_result[2] = energy_result_array[2];
  return energy_result;
}
