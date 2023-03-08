#include <filesystem>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>

#include "convert_array.h"

void check_restart_file()
{
  const std::string simulationSettingsFileName = "Restart/System_0/restart_95_10000.data";
  std::filesystem::path pathfile = std::filesystem::path(simulationSettingsFileName);
  if (!std::filesystem::exists(pathfile)) 
  {
    throw std::runtime_error("restart file' not found\n");
  }else
  {
    printf("GOOD\n");
  }
}

inline std::vector<std::string> split(const std::string txt, char ch)
{
    size_t pos = txt.find(ch);
    size_t initialPos = 0;
    std::vector<std::string> strs{};

    // Decompose statement
    while (pos != std::string::npos) {

        std::string s = txt.substr(initialPos, pos - initialPos);
        if (!s.empty())
        {
            strs.push_back(s);
        }
        initialPos = pos + 1;

        pos = txt.find(ch, initialPos);
    }

    // Add the last one
    std::string s = txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1);
    if (!s.empty())
    {
        strs.push_back(s);
    }

    return strs;
}

double* read_framework_atoms_from_restart(size_t *value)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Framework Atoms";
  size_t NumberFrameworkAtom = 0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    { 
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[2].c_str(), "%zu", &NumberFrameworkAtom);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
      break;
    }
  }
  *value = NumberFrameworkAtom;
  file.clear();
  file.seekg(0);
  //read atom positions
  std::vector<std::vector<double>> FrameworkArray(NumberFrameworkAtom, std::vector<double>(6));
  size_t count=0;
  size_t lineNum = 0;
  while (std::getline(file, str))
  {
    lineNum++;
    //printf("%s\n", str.c_str());
    if(lineNum > (counter+2))
    {
    termsScannedLined = split(str, ' ');
    FrameworkArray[count][0] = std::stod(termsScannedLined[2]);
    FrameworkArray[count][1] = std::stod(termsScannedLined[3]);
    FrameworkArray[count][2] = std::stod(termsScannedLined[4]);
    FrameworkArray[count][3] = std::stod(termsScannedLined[5]);
    FrameworkArray[count][4] = std::stod(termsScannedLined[6]);
    FrameworkArray[count][5] = std::stod(termsScannedLined[7]);
    //printf("count is %zu, lineNum is %zu\n", count, lineNum);
    //printf("last line: %.5f, %.5f,%.5f,%.5f,%.5f,%.5f\n", FrameworkArray[count][0], FrameworkArray[count][1], FrameworkArray[count][2], FrameworkArray[count][3], FrameworkArray[count][4], FrameworkArray[count][5]);
    count++;
    }
    if(lineNum >= (counter+2+NumberFrameworkAtom)) break;
  }
  //printf("count is %zu\n", count);
  double* Array = convert2DVectortoArray(FrameworkArray);
  return Array;
}

size_t* read_framework_atoms_types_from_restart()
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Framework Atoms";
  size_t NumberFrameworkAtom = 0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    { 
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[2].c_str(), "%zu", &NumberFrameworkAtom);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
      break;
    }
  }
  file.clear();
  file.seekg(0);
  //read atom positions
  std::vector<size_t> FrameworkTypeArray(NumberFrameworkAtom);
  size_t count=0;
  size_t lineNum = 0;
  size_t temp = 0;
  while (std::getline(file, str))
  {
    lineNum++;
    //printf("%s\n", str.c_str());
    if(lineNum > (counter+2))
    {
    termsScannedLined = split(str, ' ');
    sscanf(termsScannedLined[1].c_str(), "%zu", &temp);
    FrameworkTypeArray[count] = temp;
    count++;
    }
    if(lineNum >= (counter+2+NumberFrameworkAtom)) break;
  }
  size_t* result=new size_t[NumberFrameworkAtom];
  size_t* walkarr=result;
  std::copy(FrameworkTypeArray.begin(), FrameworkTypeArray.end(), walkarr);
  return result;
}

// Read force field
double* read_force_field_from_restart(size_t *value)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Force field:";
  size_t FFsize = 0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[2].c_str(), "%zu", &FFsize);
      printf("line is %u, there are %zu Framework Atoms\n", counter, FFsize);
      break;
    }
  }
  *value = FFsize;
  file.clear();
  file.seekg(0);
  //read FF array
  std::vector<std::vector<std::vector<double>>> FFParameter(FFsize, std::vector<std::vector<double>>(FFsize, std::vector<double>(4)));
  size_t count=0;
  size_t lineNum = 0;
  size_t total_lines = FFsize*FFsize;
  size_t i = 0; size_t j = 0;
  while (std::getline(file, str))
  {
    lineNum++;
    //printf("%s\n", str.c_str());
    if(lineNum > (counter+2))
    {
    termsScannedLined = split(str, ' ');
    sscanf(termsScannedLined[0].c_str(), "%zu", &i);
    sscanf(termsScannedLined[1].c_str(), "%zu", &j);
    FFParameter[i][j][0] = std::stod(termsScannedLined[4]); //epsilon
    FFParameter[i][j][1] = std::stod(termsScannedLined[5]); //sigma
    FFParameter[i][j][2] = std::stod(termsScannedLined[6]); //third term
    FFParameter[i][j][3] = std::stod(termsScannedLined[7]); //shift in potential
    //printf("count is %zu, lineNum is %zu\n", count, lineNum);
    //printf("last line: %.5f, %.5f,%.5f,%.5f,%.5f,%.5f\n", FrameworkArray[count][0], FrameworkArray[count][1], FrameworkArray[count][2], FrameworkArray[count][3], FrameworkArray[count][4], FrameworkArray[count][5]);
    count++;
    }
    if(lineNum >= (counter+2+total_lines)) break;
  }
  //printf("count is %zu\n", count);
  double* Array = convert3DVectortoArray(FFParameter);
  return Array;
}

int* read_force_field_type_from_restart()
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Force field:";
  size_t FFsize = 0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[2].c_str(), "%zu", &FFsize);
      //printf("line is %u, there are %zu FF\n", counter, FFsize);
      break;
    }
  }
  file.clear();
  file.seekg(0);
  //read FF array
  std::vector<std::vector<int>> FFType(FFsize, std::vector<int>(FFsize));
  size_t count=0;
  size_t lineNum = 0;
  size_t total_lines = FFsize*FFsize;
  size_t i = 0; size_t j = 0; int temp = 0;
  while (std::getline(file, str))
  {
    lineNum++;
    //printf("%s\n", str.c_str());
    if(lineNum > (counter+2))
    {
    termsScannedLined = split(str, ' ');
    sscanf(termsScannedLined[0].c_str(), "%zu", &i);
    sscanf(termsScannedLined[1].c_str(), "%zu", &j);
    sscanf(termsScannedLined[3].c_str(), "%i", &temp);
    FFType[i][j] = temp;
    count++;
    }
    if(lineNum >= (counter+2+total_lines)) break;
  }
  //printf("count is %zu\n", count);
  int* Array = Intconvert2DVectortoArray(FFType);
  return Array;
}

// Read Cell
double* read_Cell_from_restart(size_t skip)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Cell and InverseCell";
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      //printf("Cell line is %u\n", counter);
      break;
    }
  }
  file.clear();
  file.seekg(0);
  //read FF array
  std::vector<double> Cell(9);
  size_t count=0;
  size_t lineNum = 0;
  size_t total_lines = 1;
  while (std::getline(file, str))
  {
    lineNum++;
    //printf("%s\n", str.c_str());
    if(lineNum > (counter+skip))
    {
    //printf("%s\n", str.c_str());
    termsScannedLined = split(str, ' ');
    Cell[0] = std::stod(termsScannedLined[0]);
    Cell[1] = std::stod(termsScannedLined[1]);
    Cell[2] = std::stod(termsScannedLined[2]);
    Cell[3] = std::stod(termsScannedLined[3]);
    Cell[4] = std::stod(termsScannedLined[4]); 
    Cell[5] = std::stod(termsScannedLined[5]); 
    Cell[6] = std::stod(termsScannedLined[6]); 
    Cell[7] = std::stod(termsScannedLined[7]);
    Cell[8] = std::stod(termsScannedLined[8]); 
    //printf("count is %zu, lineNum is %zu\n", count, lineNum);
    //printf("last line: %.5f, %.5f,%.5f,%.5f,%.5f,%.5f\n", FrameworkArray[count][0], FrameworkArray[count][1], FrameworkArray[count][2], FrameworkArray[count][3], FrameworkArray[count][4], FrameworkArray[count][5]);
    count++;
    }
    if(lineNum >= (counter+skip+total_lines)) break;
  }
  double* result=new double[9];
  double* walkarr=result;
  std::copy(Cell.begin(), Cell.end(), walkarr);
  return result;
}

// Read FF Parameters
double* read_FFParams_from_restart()
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Other Params: Cutoffs, alpha, prefactors";
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      //printf("Other Params is %u\n", counter);
      break;
    }
  }
  file.clear();
  file.seekg(0);
  //read FF array
  std::vector<double> Cell(5);
  size_t count=0;
  size_t lineNum = 0;
  size_t total_lines = 1;
  size_t skip = 2;
  while (std::getline(file, str))
  {
    lineNum++;
    //printf("%s\n", str.c_str());
    if(lineNum > (counter+skip))
    {
    //printf("%s\n", str.c_str());
    termsScannedLined = split(str, ' ');
    Cell[0] = std::stod(termsScannedLined[0]);
    Cell[1] = std::stod(termsScannedLined[1]);
    Cell[2] = std::stod(termsScannedLined[2]);
    Cell[3] = std::stod(termsScannedLined[3]);
    Cell[4] = std::stod(termsScannedLined[4]);
    //printf("count is %zu, lineNum is %zu\n", count, lineNum);
    //printf("last line: %.5f, %.5f,%.5f,%.5f,%.5f,%.5f\n", FrameworkArray[count][0], FrameworkArray[count][1], FrameworkArray[count][2], FrameworkArray[count][3], FrameworkArray[count][4], FrameworkArray[count][5]);
    count++;
    }
    if(lineNum >= (counter+skip+total_lines)) break;
  }
  double* result=new double[5];
  double* walkarr=result;
  std::copy(Cell.begin(), Cell.end(), walkarr);
  return result;
}
