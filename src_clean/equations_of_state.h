/*
#include <vector>
#include <iostream>
#include <cmath>       // For std::abs, std::sqrt
#include <stdexcept>   // For std::runtime_error
#include "data_struct.h"
#include "equations_of_state.h"
*/

// Assuming the definition of Components and MOLAR_GAS_CONSTANT are provided elsewhere
const double MOLAR_GAS_CONSTANT = 8.314; // J/(mol*K), example value
#include <vector>
#include <array>
#include <algorithm>

using REAL = double;

void quadraticEquationSolver(const std::array<REAL, 4>& A, std::vector<REAL>& X, int* L) 
{
  REAL a = A[2];
  REAL b = A[1];
  REAL c = A[0];
  REAL discriminant = b*b - 4*a*c;

  if (a == 0) 
  {
    // Linear case: Ax + B = 0 -> x = -B/A
    if (b != 0) 
    {
      X.push_back(-c / b);
      *L = 1;
    } 
    else 
    {
      // No equation to solve
      *L = 0;
    }
  } 
  else if (discriminant > 0) 
  {
    // Two real solutions
    REAL sqrt_discriminant = std::sqrt(discriminant);
    X.push_back((-b + sqrt_discriminant) / (2*a));
    X.push_back((-b - sqrt_discriminant) / (2*a));
    *L = 2;
  }
  else if (discriminant == 0) 
  {
    // One real solution
    X.push_back(-b / (2*a));
    *L = 1;
  }
  else 
  {
    // Discriminant < 0, no real solutions
    *L = 0;
  }
}

int cubic(const std::array<REAL, 4>& A, std::vector<REAL>& X, int* L) 
{
  const REAL PI = 3.14159265358979323846;
  const REAL THIRD = 1.0 / 3.0;
  std::array<REAL, 3> U;
  REAL W, P, Q, DIS, PHI;

  if (A[3] != 0.0) 
  {
    // Cubic problem
    W = A[2] / A[3] * THIRD;
    P = std::pow(A[1] / A[3] * THIRD - std::pow(W, 2), 3);
    Q = -.5 * (2.0 * std::pow(W, 3) - (A[1] * W - A[0]) / A[3]);
    DIS = std::pow(Q, 2) + P;
    if (DIS < 0.0) 
    {
      // Three real solutions!
      PHI = std::acos(std::max(-1.0, std::min(1.0, Q / std::sqrt(-P))));
      P = 2.0 * std::pow((-P), 0.5 * THIRD);
      for (int i = 0; i < 3; i++) 
      {
        U[i] = P * std::cos((PHI + 2.0 * static_cast<REAL>(i) * PI) * THIRD) - W;
      }
      X = {U[0], U[1], U[2]};
      std::sort(X.begin(), X.end()); // Sort the roots
      *L = 3;
      } 
      else 
      {
        // Only one real solution!
        DIS = std::sqrt(DIS);
        X.push_back(cbrt(Q + DIS) + cbrt(Q - DIS) - W);
        *L = 1;
      }
    } 
    else if (A[2] != 0.0) 
    {
      // Quadratic problem
      quadraticEquationSolver(A, X, L); // Assume this function is defined elsewhere
    } 
    else if (A[1] != 0.0) 
    {
      // Linear equation
      X.push_back(A[0] / A[1]);
      *L = 1;
    } 
    else 
    {
      // No equation
      *L = 0;
    }

    // Perform one step of a Newton iteration to minimize round-off errors
    for (int i = 0; i < *L; i++) 
    {
      X[i] = X[i] - (A[0] + X[i] * (A[1] + X[i] * (A[2] + X[i] * A[3]))) / (A[1] + X[i] * (2.0 * A[2] + X[i] * 3.0 * A[3]));
    }
    return 0;
}

//Zhao's note: pressure and temperature should already be in Components variable
void ComputeFugacity(Components& TempComponents, double Pressure, double Temperature)
{
  printf("================FUGACITY COEFFICIENT CALCULATION================\n");
  double SumofFractions = 0.0;
  std::vector<double> MolFraction, PartialPressure, LocalPc, LocalTc, LocalAc, CompNum;
  std::vector<double> a, b, A, B; // Storing calculated properties

  // need to fix: now we did not consider the gas molecules, need to take in consideration of that. 
  // Idea: Need to find the size of the framework, should be provided somewhere like Components.sth.y
  int FrameworkComponents = TempComponents.NComponents.y;
  
  bool NeedEOSFugacityCoeff = false; //If false, then no need to calculate PR-EOS fugacity coefficient, return//
  for(size_t comp = FrameworkComponents; comp < TempComponents.Total_Components; comp++)
  {
    printf("Checking: Current Fugacity Coeff for %zu component: %.5f\n", comp, TempComponents.FugacityCoeff[comp]);
    if(TempComponents.FugacityCoeff[comp] < 0.0)
    {
      printf("Component %zu needs PR-EOS for fugacity coeff (negative value), going to ignore the fugacity coefficient for other components and calculate all using PR-EOS!", comp);
      NeedEOSFugacityCoeff = true;
    }
  }

  if(!NeedEOSFugacityCoeff){ printf("Every Adsorbate Component has fugacity coefficient assigned, skip EOS calculation!\n"); return;}
  printf("start calculating fugacity, Pressure: %.5f, Temperature: %.5f\n", Pressure, Temperature);

  for(size_t comp = FrameworkComponents; comp < TempComponents.Total_Components; comp++)
  {
    double ComponentMolFrac = TempComponents.MolFraction[comp];
    SumofFractions += ComponentMolFrac;
    MolFraction.push_back(ComponentMolFrac);

    double CurrPartialPressure = ComponentMolFrac * Pressure;
    PartialPressure.push_back(CurrPartialPressure);

    double ComponentPc = TempComponents.Pc[comp];
    double ComponentTc = TempComponents.Tc[comp];
    double ComponentAc = TempComponents.Accentric[comp];

    LocalPc.push_back(ComponentPc);
    LocalTc.push_back(ComponentTc);
    LocalAc.push_back(ComponentAc);

    // Assuming calculations need to be made per component:
    double Tr = Temperature / ComponentTc; // Reduced temperature
    double kappa = 0.37464 + 1.54226 * ComponentAc - 0.26992 * std::pow(ComponentAc, 2);
    double alpha = std::pow(1.0 + kappa * (1.0 - std::sqrt(Tr)), 2);
    a.push_back(0.45724 * alpha * std::pow(MOLAR_GAS_CONSTANT * ComponentTc, 2) / ComponentPc);
    b.push_back(0.07780 * MOLAR_GAS_CONSTANT * ComponentTc / ComponentPc);
    A.push_back(a.back() * Pressure / std::pow(MOLAR_GAS_CONSTANT * Temperature, 2));
    B.push_back(b.back() * Pressure / (MOLAR_GAS_CONSTANT * Temperature));

    CompNum.push_back(comp);

    printf("Component %zu, Pc: %.5f, Tc: %.5f, Ac: %.5f\n", comp, ComponentPc, ComponentTc, ComponentAc);
    printf("Component Mol Fraction: %.5f\n", SumofFractions);
    printf("kappa: %.5f, alpha: %.5f\n", kappa, alpha);
    //printf("a: %.5f, b: %.5f, A: %.5f, B: %.5f\n", a.back(), b.back(), A.back(), B.back());
  }


  if(std::abs(SumofFractions - 1.0) > 0.0001)
  {
    throw std::runtime_error("Sum of Mol Fractions does not equal 1.0");
  }

  // printf("now printing the component numbers");

  for(int num : CompNum) 
  {
    std::cout << num << std::endl;
  }

  size_t NumberOfComponents = CompNum.size();

  std::vector<std::vector<double>> BinaryInteractionParameter(NumberOfComponents, std::vector<double>(NumberOfComponents, 0.0)); // All zeros
  std::vector<std::vector<double>> aij(NumberOfComponents, std::vector<double>(NumberOfComponents));
  std::vector<std::vector<double>> Aij(NumberOfComponents, std::vector<double>(NumberOfComponents));

  for(size_t i = 0; i < NumberOfComponents; i++) 
  {
    for(size_t j = 0; j < NumberOfComponents; j++) 
    {
      aij[i][j] = (1.0 - BinaryInteractionParameter[i][j]) * std::sqrt(a[i] * a[j]);
      Aij[i][j] = (1.0 - BinaryInteractionParameter[i][j]) * std::sqrt(A[i] * A[j]);
      //printf("i: %zu, j: %zu, aij: %.5f, bij: %.5f\n", i,j,aij[i][j],Aij[i][j]);
    }
  }
  double Amix = 0.0;
  double Bmix = 0.0;

  for(size_t i = 0; i < NumberOfComponents; i++) 
  {
    Bmix += MolFraction[i] * b[i];
    for(size_t j = 0; j < NumberOfComponents; j++) 
    {
      Amix += MolFraction[i] * MolFraction[j] * aij[i][j];
    }
    //printf("MolFrac of %zu: %.5f\n", i, MolFraction[i]);
  }
  Amix *= Pressure / std::pow(MOLAR_GAS_CONSTANT * Temperature, 2);
  Bmix *= Pressure / (MOLAR_GAS_CONSTANT * Temperature);
  // Further calculations or usage of calculated vectors (a, b, A, B) can be done here
  double Coefficients[4];
  Coefficients[3] = 1.0;
  Coefficients[2] = Bmix - 1.0;
  Coefficients[1] = Amix - 3.0 * std::pow(Bmix, 2) - 2.0 * Bmix; // Using std::pow for square
  Coefficients[0] = -(Amix * Bmix - std::pow(Bmix, 2) - std::pow(Bmix, 3)); // Using std::pow for square and cube
  std::array<REAL, 4> CoefficientsArray = {Coefficients[0], Coefficients[1], Coefficients[2], Coefficients[3]};

  //printf("Amix: %.5f, Bmix: %.5f, Coeffs: %.5f, %.5f, %.5f, %.5f\n", Amix, Bmix, Coefficients[0], Coefficients[1], Coefficients[2], Coefficients[3]);

  std::vector<double> Compressibility;
  int NumberOfSolutions = 0;
  cubic(CoefficientsArray,Compressibility,&NumberOfSolutions);
  double temp = 0.0;

  printf("Number of Solutions: %d\n", NumberOfSolutions);

  //for(size_t i = 0; i < Compressibility.size(); i++)
  //  printf("Before Processing: Compressibility %zu = %.5f\n", i, Compressibility[i]);
  if(Compressibility.size() > 0)
  {
    if (Compressibility[0]<Compressibility[1])
    {
      temp=Compressibility[0];
      Compressibility[0]=Compressibility[1];
      Compressibility[1]=temp;
    }
    if(Compressibility.size() > 1)
    if (Compressibility[1]<Compressibility[2])
    {
      temp=Compressibility[1];
      Compressibility[1]=Compressibility[2];
      Compressibility[2]=temp;
    }
    if(Compressibility[0]<Compressibility[1])
    {
      temp=Compressibility[0];
      Compressibility[0]=Compressibility[1];
      Compressibility[1]=temp;
    }
  }
  // printf("Compressibility size = %zu", Compressibility.size());
  //for(size_t i = 0; i < Compressibility.size(); i++)
  //  printf("After Processing: Compressibility %zu = %.5f\n", i, Compressibility[i]);

  std::vector<std::vector<double>> FugacityCoefficients(NumberOfComponents, std::vector<double>(NumberOfSolutions, 0.0));
    
  for(size_t i = 0; i < NumberOfComponents; i++) 
  {
    for(size_t j = 0; j < NumberOfSolutions; j++) 
    {
      double sumAij = 0.0; // Correctly declared sum for Aij calculation
      for(size_t k = 0; k < NumberOfComponents; k++) 
      {
        sumAij += 2.0 * MolFraction[k] * Aij[i][k]; // Assuming MolFraction is the correct reference
      }

      // Ensure FugacityCoefficients is declared and sized appropriately before this block
      FugacityCoefficients[i][j] = exp((B[i] / Bmix) * (Compressibility[j] - 1.0) - log(Compressibility[j] - Bmix) 
						- (Amix / (2.0 * sqrt(2.0) * Bmix)) * (sumAij / Amix - B[i] / Bmix) * 
						log((Compressibility[j] + (1.0 + sqrt(2.0)) * Bmix) / (Compressibility[j] + (1.0 - sqrt(2.0)) * Bmix)));
    }
  }
  // for(size_t i = 0; i < FugacityCoefficients.size(); i++)
  //   printf("first row for calculated fugacity %zu = %.5f\n", i, Compressibility[i]);

  // ToDO: need to fix
  for(size_t i = 0; i < NumberOfComponents; i++)
  {
    int index = FrameworkComponents + i;
    if(NumberOfSolutions == 1)
    {
      // If there is only one solution, use it
      TempComponents.FugacityCoeff[index]= FugacityCoefficients[i][0];
    }
    else
    {
      // More than one solution exists
      // You can just use the length of Compressibility, instead of comparing values?
      //use the values in case there is some weird minus values
      if(Compressibility[2] > 0.0)
      {
        // Compare the first and third solutions
        if(FugacityCoefficients[i][0] < FugacityCoefficients[i][2])
        {
          TempComponents.FugacityCoeff[index] = FugacityCoefficients[i][0];  // Favor the first solution
        }
        else if(FugacityCoefficients[i][0] > FugacityCoefficients[i][2])
        {
          // Favor the third solution, interpreted as vapor (metastable) and liquid (stable)
          TempComponents.FugacityCoeff[index] = FugacityCoefficients[i][2];
        }
        else
        {
          // When they are equal, it indicates both vapor and liquid are stable
          TempComponents.FugacityCoeff[index] = FugacityCoefficients[i][0];
        }
      }
      else
      {
        // Default to the first solution if the third compressibility is not positive
        TempComponents.FugacityCoeff[index] = FugacityCoefficients[i][0];
      }
    }
    printf("Fugacity Coefficient for component %zu is %.10f\n", index, TempComponents.FugacityCoeff[index]);
    for(size_t ii = 0; ii < FugacityCoefficients.size(); ii++)
      for(size_t jj = 0; jj < FugacityCoefficients[ii].size(); jj++)
        printf("Check: Computed FugaCoeff for %zu, %zu is %.10f\n", ii, jj, FugacityCoefficients[ii][jj]);
  }
  printf("================END OF FUGACITY COEFFICIENT CALCULATION================\n");
}
