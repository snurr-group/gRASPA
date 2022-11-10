void check_restart_file()
{
  const std::string simulationSettingsFileName = "Restart/System_0/restart_95_10000.data";
  std::filesystem::path pathfile = std::filesystem::path(simulationSettingsFileName);
  if (!std::filesystem::exists(pathfile))
  {
    throw std::runtime_error("restart file' not found\n");
  }
}

void read_adsorbate_atoms_from_restart_SoA(size_t Component, size_t *value, size_t *Allocate_value, double **x, double **y, double **z, double **scale, double **charge, double **scaleCoul, size_t **Type, size_t **MolID)
{
  //for one component, read just the molecules in that one component
  //for multi-component, call this functions multiple times
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("Restart/System_0/restart_95_10000.data");
  unsigned int counter=0;
  std::string search = "Adsorbate Component "; //remember to add component value, need to cast (to_string)
  size_t NumberMolecules = 0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find(search, 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[4].c_str(), "%zu", &NumberMolecules);
      printf("line is %u, there are %zu Adsorbate Molecules for Adsorbate component %zu\n", counter, NumberMolecules, Component);
      break;
    }
  }
  size_t Moleculesize = 1; //Zhao's note: assume methane
  size_t NumberAtoms = NumberMolecules*Moleculesize;
  *value = NumberMolecules*Moleculesize;
  size_t Allocate_size = 1024;
  if(Allocate_size <= NumberAtoms){throw std::runtime_error("Not Enough Space for Molecules in Adsorbate Component");}
  *Allocate_value = Allocate_size;
  file.clear();
  file.seekg(0);
  //read atom positions
  std::vector<double> Ax(Allocate_size);
  std::vector<double> Ay(Allocate_size);
  std::vector<double> Az(Allocate_size);
  std::vector<double> Ascale(Allocate_size);
  std::vector<double> Acharge(Allocate_size);
  std::vector<double> AscaleCoul(Allocate_size);
  std::vector<size_t> AType(Allocate_size);
  std::vector<size_t> AMolID(Allocate_size);
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
      AType[count] = temp;
      Ax[count] = std::stod(termsScannedLined[2]);
      Ay[count] = std::stod(termsScannedLined[3]);
      Az[count] = std::stod(termsScannedLined[4]);
      Ascale[count] = std::stod(termsScannedLined[5]);
      Acharge[count] = std::stod(termsScannedLined[6]);
      AscaleCoul[count] = std::stod(termsScannedLined[7]);
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[8].c_str(), "%zu", &temp);
      AMolID[count] = temp;
    count++;
    }
    if(lineNum >= (counter+2+NumberAtoms)) break;
  }
  double* result;
  result = Doubleconvert1DVectortoArray(Ax);
  *x = result;
  result = Doubleconvert1DVectortoArray(Ay);
  *y = result;
  result = Doubleconvert1DVectortoArray(Az);
  *z = result;
  result = Doubleconvert1DVectortoArray(Ascale);
  *scale = result;
  result = Doubleconvert1DVectortoArray(Acharge);
  *charge = result;
  result = Doubleconvert1DVectortoArray(AscaleCoul);
  *scaleCoul = result;
  size_t* Size_tresult;
  Size_tresult = Size_tconvert1DVectortoArray(AType);
  *Type = Size_tresult;
  Size_tresult = Size_tconvert1DVectortoArray(AMolID);
  *MolID = Size_tresult;
}

//////////////////////////////////////
// Read force field SoA
void read_force_field_from_restart_SoA(size_t *value, double **Epsilon, double **Sigma, double **Z, double **Shifted, int **Type, bool shift)
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
      printf("line is %u, there are %zu Force Field Definitions\n", counter, FFsize);
      break;
    }
  }
  *value = FFsize;
  file.clear();
  file.seekg(0);
  //read FF array

  std::vector<double> AEpsilon(FFsize*FFsize);
  std::vector<double> ASigma(FFsize*FFsize);
  std::vector<double> AZ(FFsize*FFsize);
  std::vector<double> AShift(FFsize*FFsize);
  std::vector<int> AType(FFsize*FFsize);
  size_t count=0;
  size_t lineNum = 0;
  size_t total_lines = FFsize*FFsize;
  size_t i = 0; size_t j = 0; int temp = 0;
  while (std::getline(file, str))
  {
    lineNum++;
    if(lineNum > (counter+2))
    {
    termsScannedLined = split(str, ' ');
    sscanf(termsScannedLined[0].c_str(), "%zu", &i);
    sscanf(termsScannedLined[1].c_str(), "%zu", &j);
    AEpsilon[i*FFsize+j] = std::stod(termsScannedLined[4]); //epsilon
    ASigma[i*FFsize+j]   = std::stod(termsScannedLined[5]); //sigma
    AZ[i*FFsize+j]       = std::stod(termsScannedLined[6]); //third term
    if(shift)
    {
      AShift[i*FFsize+j] = std::stod(termsScannedLined[7]); //shift in potential
    }else
    { AShift[i*FFsize+j] = 0.0; }
    sscanf(termsScannedLined[3].c_str(), "%i", &temp);
    AType[i*FFsize+j]    = temp; //force field type
    count++;
    }
    if(lineNum >= (counter+2+total_lines)) break;
  }
  double* result;
  result = Doubleconvert1DVectortoArray(AEpsilon);
  *Epsilon = result;
  result = Doubleconvert1DVectortoArray(ASigma);
  *Sigma = result;
  result = Doubleconvert1DVectortoArray(AZ);
  *Z = result;
  result = Doubleconvert1DVectortoArray(AShift);
  *Shifted = result;
  int* Int_result;
  Int_result = Intconvert1DVectortoArray(AType);
  *Type = Int_result;
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

void Read_Cell_wrapper(Boxsize& Box)
{
  Box.Cell = read_Cell_from_restart(2);
  Box.InverseCell = read_Cell_from_restart(3);
  Box.Volume = matrix_determinant(Box.Cell);
}

void read_framework_atoms_from_restart_SoA(size_t *value, size_t *Allocate_value, double **Framework_x, double **Framework_y, double **Framework_z, double **Framework_scale, double **Framework_charge, double **Framework_scaleCoul, size_t **Framework_Type, size_t **MolID)
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
  *Allocate_value = NumberFrameworkAtom; //For framework atoms, allocate the same amount of memory as the number of framework atoms
  file.clear();
  file.seekg(0);
  //read atom positions
  std::vector<double> Framework_Ax(NumberFrameworkAtom);
  std::vector<double> Framework_Ay(NumberFrameworkAtom);
  std::vector<double> Framework_Az(NumberFrameworkAtom);
  std::vector<double> Framework_Ascale(NumberFrameworkAtom);
  std::vector<double> Framework_Acharge(NumberFrameworkAtom);
  std::vector<double> Framework_AscaleCoul(NumberFrameworkAtom);
  std::vector<size_t> Framework_AType(NumberFrameworkAtom);
  std::vector<size_t> Framework_AMolID(NumberFrameworkAtom);
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
      Framework_AType[count] = temp;
      Framework_Ax[count] = std::stod(termsScannedLined[2]);
      Framework_Ay[count] = std::stod(termsScannedLined[3]);
      Framework_Az[count] = std::stod(termsScannedLined[4]);
      Framework_Ascale[count] = std::stod(termsScannedLined[5]);
      Framework_Acharge[count] = std::stod(termsScannedLined[6]);
      Framework_AscaleCoul[count] = std::stod(termsScannedLined[7]);
      Framework_AMolID[count] =  0; //MoleculeID for framework atoms are ZERO
    count++;
    }
    if(lineNum >= (counter+2+NumberFrameworkAtom)) break;
  }
  double* result;
  result = Doubleconvert1DVectortoArray(Framework_Ax);
  *Framework_x = result;
  result = Doubleconvert1DVectortoArray(Framework_Ay);
  *Framework_y = result;
  result = Doubleconvert1DVectortoArray(Framework_Az);
  *Framework_z = result;
  result = Doubleconvert1DVectortoArray(Framework_Ascale);
  *Framework_scale = result;
  result = Doubleconvert1DVectortoArray(Framework_Acharge);
  *Framework_charge = result;
  result = Doubleconvert1DVectortoArray(Framework_AscaleCoul);
  *Framework_scaleCoul = result;
  size_t* Size_tresult;
  Size_tresult = Size_tconvert1DVectortoArray(Framework_AType);
  *Framework_Type = Size_tresult;
  Size_tresult = Size_tconvert1DVectortoArray(Framework_AMolID);
  *MolID = Size_tresult;
}
