#include "RN.h"
#include <cstdlib>
#include <random>
double get_random_from_zero_to_one()
{
  //return (double) (rand()/RAND_MAX);
  //std::srand(1.0);
  return static_cast<double>(std::rand()) / RAND_MAX;
}
