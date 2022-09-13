#include <vector>
#include <array>
#include "convert_array.h"
#include <stdio.h>

double* Doubleconvert1DVectortoArray(std::vector<double>& Vector)
{
  size_t Vectorsize = Vector.size();
  double* result=new double[Vectorsize];
  double* walkarr=result;
  std::copy(Vector.begin(), Vector.end(), walkarr);
  //printf("done convert Mol Type, Origin: %zu, copied: %zu\n", MoleculeTypeArray[0], result[0]);
  return result;
}

size_t* Size_tconvert1DVectortoArray(std::vector<size_t>& Vector)
{
  size_t Vectorsize = Vector.size();
  size_t* result=new size_t[Vectorsize];
  size_t* walkarr=result;
  std::copy(Vector.begin(), Vector.end(), walkarr);
  //printf("done convert Mol Type, Origin: %zu, copied: %zu\n", MoleculeTypeArray[0], result[0]);
  return result;
}
int* Intconvert2DVectortoArray(std::vector<std::vector<int>>& Vector)
{
  std::size_t totalsize = 0;
  for (size_t i=0; i<Vector.size(); i++)
  {
    totalsize += Vector[i].size();
  }

  int* result=new int[totalsize];
  int* walkarr=result;

  for (size_t i=0; i<Vector.size(); i++) {
      std::copy(Vector[i].begin(), Vector[i].end(), walkarr);
      walkarr += Vector[i].size();
  }
  return result;
}


double* convert2DVectortoArray(std::vector<std::vector<double>>& Vector)
{
  std::size_t totalsize = 0;
  for (size_t i=0; i<Vector.size(); i++)
  {
    totalsize += Vector[i].size();
  }

  double* result=new double[totalsize];
  double* walkarr=result;

  for (size_t i=0; i<Vector.size(); i++) {
      std::copy(Vector[i].begin(), Vector[i].end(), walkarr);
      walkarr += Vector[i].size();
  }
  return result;
}

double* convert3DVectortoArray(std::vector<std::vector<std::vector<double>>>& Vector)
{
  std::size_t totalsize = 0;
  for (size_t i=0; i<Vector.size(); i++)
  {
    for (size_t j=0; j<Vector[i].size(); j++)
    {
      totalsize += Vector[i][j].size();
    }
  }
  double* result=new double[totalsize];
  double* walkarr=result;

  for (size_t i=0; i<Vector.size(); i++) {
    for (size_t j=0; j<Vector[i].size(); j++)
    {
      std::copy(Vector[i][j].begin(), Vector[i][j].end(), walkarr);
      walkarr += Vector[i][j].size();
    }
  }
  return result;
}
