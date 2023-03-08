#include <stdio.h>
#include <cmath>
#include <array>
using namespace std;

static inline std::array<double,2> GPUpotentialCoulomb(const double alpha, const double scaling, const double r, const double chargeA, const double chargeB, const int ChargeMethod)
{
    //Ewald = 0,
    //Coulomb = 1,
    //Wolf = 2,
    //ModifiedWolf = 3
    std::array<double,2> result;
    switch(ChargeMethod)
    {
      case 0:
      {
        double term = chargeA * chargeB * std::erfc(alpha * r);
        result[0] = scaling * term / r; result[1] = 0.0;
        break;
      }
      case 1:
      {
        result[0] = scaling * chargeA * chargeB / r; result[1] = 0.0;
        break;
      }
      case 2:
      {
        result[0] = scaling * chargeA * chargeB * std::erfc(alpha * r) / r; result[1] = 0.0;
        break;
      }
      case 3:
      {
        result[0] = scaling * chargeA * chargeB * std::erfc(alpha * r) / r; result[1] = 0.0;
        break;
      }
      default:
      {
        printf("You are screwed");
      }
    }
    return result;
}
