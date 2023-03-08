#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
void create_movie_file(size_t Cycle, Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, char** AtomNames);

static inline void WriteBox(Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, std::ofstream& textrestartFile, char** AtomNames)
{
  size_t NumComponents = SystemComponents.Total_Components;
  size_t NumberOfAtoms = 0;
  for(size_t i = 0; i < NumComponents; i++)
  {
    NumberOfAtoms += SystemComponents.NumberOfMolecule_for_Component[i] * SystemComponents.Moleculesize[i];
  }
  textrestartFile << "# LAMMPS data file written by WriteLammpsdata function in RASPA(written by Zhao Li)" << '\n';
  textrestartFile << NumberOfAtoms << " atoms" << '\n';
  textrestartFile << 0 << " bonds" << '\n';
  textrestartFile << 0 << " angles" << '\n';
  textrestartFile << 0 << " dihedrals" << '\n';
  textrestartFile << 0 << " impropers" << '\n';
  textrestartFile << 0 << " " << Box.Cell[0] << " xlo xhi" << '\n';
  textrestartFile << 0 << " " << Box.Cell[4] << " ylo yhi" << '\n';
  textrestartFile << 0 << " " << Box.Cell[8] << " zlo zhi" << '\n';
  textrestartFile << 0.0 << " " << 0.0 << " " << 0.0 << " xy xz yz" << '\n' << '\n';
  textrestartFile << FF.size << " atom types" << '\n'; 
  textrestartFile << 0 << " bond types" << '\n';
  textrestartFile << 0 << " angle types" << '\n';
  textrestartFile << 0 << " dihedral types" << '\n';
  textrestartFile << 0 << " improper types" << '\n' << '\n';
  //textrestartFile << std::puts("%zu bonds\n", NumberOfAtoms);
  textrestartFile << "Masses" << '\n' << '\n';
  for(size_t i = 0; i < FF.size; i++)
  { 
    double mass = 0.0;
    textrestartFile << i+1 << " " << mass << " # " << AtomNames[i] << '\n';
  }
  textrestartFile << '\n' << "Pair Coeffs" << '\n' << '\n'; 
  for(size_t i = 0; i < FF.size; i++)
  { 
    const size_t row = i*FF.size+i;
    textrestartFile << i+1 << " " << FF.epsilon[row]/120.2 << " " << FF.sigma[row] << " # " << AtomNames[i] << '\n';
  }
}

static inline void WriteAtoms(Atoms* System, Components SystemComponents, std::ofstream& textrestartFile, char** AtomNames)
{
  size_t NumComponents = SystemComponents.Total_Components;
  textrestartFile << '\n' << "Atoms" << '\n' << '\n';
  size_t Atomcount=0; size_t Molcount=0;
  for(size_t i = 0; i < NumComponents; i++)
  {
    Atoms Data = System[i];
    for(size_t j = 0; j < Data.size; j++)
    {
      textrestartFile << Atomcount+1 << " " << Molcount+Data.MolID[j]+1 << " " << Data.Type[j]+1 << " " << Data.charge[j] << " " << Data.x[j] << " " << Data.y[j] << " " << Data.z[j] << " # " << AtomNames[Data.Type[j]] << '\n';
      Atomcount++;
    }
    Molcount+=SystemComponents.NumberOfMolecule_for_Component[i];
  }
}


static inline void create_movie_file(size_t Cycle, Atoms* System, Components SystemComponents, ForceField FF, Boxsize Box, char** AtomNames)
{
  std::ofstream textrestartFile{};
  std::filesystem::path cwd = std::filesystem::current_path();

  std::filesystem::path directoryName = cwd /"Movies/";
  std::filesystem::path fileName = cwd /"Movies/result.data";
  std::filesystem::create_directories(directoryName);
  textrestartFile = std::ofstream(fileName, std::ios::out);
  WriteBox(System, SystemComponents, FF, Box, textrestartFile, AtomNames);
  WriteAtoms(System, SystemComponents, textrestartFile, AtomNames);
}

