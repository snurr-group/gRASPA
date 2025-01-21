#include "data_struct.h"

#include <cstdlib>
#include <random>

double Get_Uniform_Random()
{
  //return (double) (rand()/RAND_MAX);
  //std::srand(3.0);
  return static_cast<double>(std::rand()) / RAND_MAX;
}

// The polar form of the Box-Muller transformation, See Knuth v2, 3rd ed, p122, adapted from RASPA2
double Get_Gaussian_Random(void)
{
  double ran1,ran2,r2 = 0.0;

  while((r2>1.0)||(r2==0.0))
  {
    ran1=2.0*Get_Uniform_Random()-1.0;
    ran2=2.0*Get_Uniform_Random()-1.0;
    r2=pow(ran1,2)+pow(ran2,2);
  }
  return ran2*sqrt(-2.0*std::log(r2)/r2);
}
