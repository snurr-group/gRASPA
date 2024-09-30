#include "data_struct.h"

#include <cstdlib>
#include <random>

double Get_Uniform_Random()
{
  //return (double) (rand()/RAND_MAX);
  //std::srand(3.0);
  return static_cast<double>(std::rand()) / RAND_MAX;
}
