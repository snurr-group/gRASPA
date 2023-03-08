#include <filesystem>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>

#include "convert_array.h"
#include "data_struct.h"
#include "matrix_manipulation.h"

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
bool caseInSensStringCompare(const std::string& str1, const std::string& str2)
{
    return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), [](auto a, auto b) {return std::tolower(a) == std::tolower(b); });
}

void Check_Component_size(Components& SystemComponents)
{
  size_t referenceVal = SystemComponents.MoleculeName.size();
  if(SystemComponents.Moleculesize.size() != referenceVal)                   throw std::runtime_error("Moleculesize does not match!");
  if(SystemComponents.NumberOfMolecule_for_Component.size() != referenceVal) throw std::runtime_error("NumberOfMolecule_for_Component does not match!");
  if(SystemComponents.MolFraction.size() != referenceVal)                    throw std::runtime_error("MolFraction does not match!");
  if(SystemComponents.IdealRosenbluthWeight.size() != referenceVal)          throw std::runtime_error("IdealRosenbluthWeight does not match!");
  if(SystemComponents.FugacityCoeff.size() != referenceVal)                  throw std::runtime_error("FugacityCoeff does not match!");
  if(SystemComponents.Tc.size() != referenceVal)                             throw std::runtime_error("Tc does not match!");
  if(SystemComponents.Pc.size() != referenceVal)                             throw std::runtime_error("Pc does not match!");
  if(SystemComponents.Accentric.size() != referenceVal)                      throw std::runtime_error("Accentric does not match!");
}

void read_number_of_sims_from_input(size_t *NumSims, bool *SingleSim)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  size_t tempnum = 0; bool tempsingle = false; size_t counter = 0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find("SingleSimulation", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempsingle = true;
        printf("running only one simulation\n");
      }
    }
    if (str.find("NumberOfSimulations", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &tempnum);
      printf("There are %zu simulations\n", tempnum);
    }
  }
  *NumSims = tempnum; *SingleSim = tempsingle;
}

void read_simulation_input(bool *UseGPUReduction, bool *Useflag, bool *noCharges, int *Cycles, size_t *Widom_Trial, size_t *NumberOfBlocks, double *Pressure, double *Temperature, bool *DualPrecision, size_t *AllocateSize)
{
  bool tempGPU = false; bool tempflag = false; bool nochargeflag = true;  //Ignore the changes if the chargemethod is not specified
  bool tempDualPrecision = false;
  int cycles=1; size_t widom;
  double temp = 0.0;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  int counter=0; size_t Nblock=0; size_t tempallocspace=0;
  double pres = 0.0;
  while (std::getline(file, str))
  {
    counter++;
    if (str.find("UseGPUReduction", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempGPU = true;
        printf("found GPU reduction\n");
      }
    }
    if (str.find("Useflag", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempflag = true;
        printf("found flag\n");
      }
    }
    if (str.find("AdsorbateAllocateSpace", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &tempallocspace);
      printf("line is %u, Allocate space for adsorbate is %zu\n", counter, tempallocspace);
    }
    if (str.find("Cycles", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%d", &cycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("Widom_Trial", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &widom);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfBlocks", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &Nblock);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("Pressure", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      pres=std::stod(termsScannedLined[1]);
    }
    if (str.find("Temperature", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      temp=std::stod(termsScannedLined[1]);
    }
    if (str.find("ChargeMethod", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "Ewald"))
      {
        nochargeflag = false;
        printf("found nochargeflag\n");
      }
    }
    if (str.find("DualPrecisionCBMC", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempDualPrecision = true;
        printf("found Dual-Precision CBMC flag\n");
      }
    }
    if(counter>200) break;
  }
  *UseGPUReduction=tempGPU; *Useflag=tempflag; *noCharges = nochargeflag;
  *Cycles=cycles; *Widom_Trial=widom; *NumberOfBlocks=Nblock;
  *Pressure = pres; *Temperature = temp;
  *DualPrecision = tempDualPrecision;
  *AllocateSize  = tempallocspace;
}

void read_FFParams_from_input(ForceField& FF)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;

  double tempOverlap = 1.0e6; double tempvdwcut = 12.0; double tempcoulcut = 12.0;
  double tempprefactor = 138935.48350;
  double tempprecision = 1.05023e-14;
  double tempalpha = 0.26506; //Zhao's note: here we used alpha equal to the preset value from raspa3. Need to revisit RASPA-2 for the exact calculation of alpha.
  std::vector<double> Cell(5);
  //Zhao's note: Using the heuresitic equation for converting Ewald Precision to alpha.

  std::ifstream file("simulation.input");
  while (std::getline(file, str))
  {
    if (str.find("OverlapCriteria", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      tempOverlap = std::stod(termsScannedLined[1]);
    }
    if (str.find("CutOffVDW", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      tempvdwcut = std::stod(termsScannedLined[1]);
    }
    if (str.find("CutOffCoulomb", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      tempcoulcut = std::stod(termsScannedLined[1]);
    }
    if (str.find("EwaldPrecision", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      tempprecision = std::stod(termsScannedLined[1]);
      tempalpha = (1.35 - 0.15 * log(tempprecision))/tempcoulcut; // Zhao's note: heurestic equation //
    }
    if (str.find("Component", 0) != std::string::npos) //When it reads component, skip//
      break;
  }
  //read FF array
  FF.OverlapCriteria = tempOverlap;
  FF.CutOffVDW       = tempvdwcut*tempvdwcut;
  FF.CutOffCoul      = tempcoulcut*tempcoulcut;
  FF.Prefactor       = tempprefactor;
  FF.Alpha           = tempalpha*tempalpha;
}

inline std::string& tolower(std::string& s)
{
    for (auto& c : s)
    {
        [[maybe_unused]] auto t = std::tolower(static_cast<unsigned char>(c));
    }

    return s;
}

inline std::string trim2(const std::string& s)
{
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        start++;
    }

    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}

double Get_Shifted_Value(double epsilon, double sigma, double CutOffSquared)
{
  double scaling = 1.0;
  double arg1 = epsilon;
  double arg2 = sigma * sigma;
  double rr = CutOffSquared;
  double temp = (rr / arg2);
  double rri3 = 1.0 / ((temp * temp * temp) + 0.5 * (1.0 - scaling) * (1.0 - scaling));
  double shift = scaling * (4.0 * arg1 * (rri3 * (rri3 - 1.0)));
  return shift;
}

double Mixing_Rule_Epsilon(double ep1, double ep2)
{
  return sqrt(ep1*ep2); //Assuming Lorentz Berthelot
}

double Mixing_rule_Sigma(double sig1, double sig2)
{
  return 0.5*(sig1+sig2); //Assuming Lorentz Berthelot
}
//Zhao's note: read force field definition first, then read pseudo-atoms
//The type numbering is determined in here, then used in pseudo_atoms.def
void ForceFieldParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom)
{
  double CutOffSquared = 144.0; //Zhao's note: need to fix this after testing this function, it should be read from simulation.input
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0;
  std::ifstream PseudoAtomfile("force_field_mixing_rules.def");
  size_t NumberOfDefinitions = 0;
  bool shifted = false; bool tail = false;
  //Temporary vectors for storing the data
  std::vector<double>Epsilon; std::vector<double>Sigma;
  // Some other temporary values
  double ep = 0.0; double sig = 0.0;
  // First read the pseudo atom file
  while (std::getline(PseudoAtomfile, str))
  {
    if(counter == 1) //read shifted/truncated
    {
      termsScannedLined = split(str, ' ');
      if(termsScannedLined[0] == "shifted")
        shifted = true;
    }
    else if(counter == 3) //read tail correction
    {
      termsScannedLined = split(str, ' ');
      if(termsScannedLined[0] == "yes")
      {
        tail = true; //Zhao's note: not implemented
        throw std::runtime_error("Tail correction not implemented YET...");
      }
    }
    else if(counter == 5) // read number of force field definitions
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[0].c_str(), "%zu", &NumberOfDefinitions);
      if (NumberOfDefinitions <= 0 || NumberOfDefinitions>200) throw std::runtime_error("Incorrect amount of force field definitions");
    }
    else if(counter >= 7) // read data for each force field definition
    {
      printf("%s\n", str.c_str());
      termsScannedLined = split(str, ' ');
      PseudoAtom.Name.push_back(termsScannedLined[0]);
      ep = std::stod(termsScannedLined[2]);
      sig= std::stod(termsScannedLined[3]);
      Epsilon.push_back(ep);
      Sigma.push_back(sig);
    }
    counter++;
    if(counter==7+NumberOfDefinitions) break; //in case there are extra empty rows, Zhao's note: I am skipping the mixing rule, assuming Lorentz-Berthelot
  }
  //Do mixing rule (assuming Lorentz-Berthelot)
  //Declare some temporary arrays
  std::vector<double>Mix_Epsilon; std::vector<double>Mix_Sigma; std::vector<double>Mix_Shift; 
  std::vector<double>Mix_Z; //temporary third array
  std::vector<int>Mix_Type; //Force Field Type (Zhao's note: assuming zero, which is Lennard-Jones)
  double temp_ep = 0.0; double temp_sig = 0.0;
  for(size_t i = 0; i < NumberOfDefinitions; i++)
  {
    for(size_t j = 0; j < NumberOfDefinitions; j++)
    {
      temp_ep = Mixing_Rule_Epsilon(Epsilon[i], Epsilon[j])/1.20272430057; //Zhao's note: caveat here: need to do full energy conversion
      temp_sig= Mixing_rule_Sigma(Sigma[i], Sigma[j]);
      Mix_Epsilon.push_back(temp_ep);
      Mix_Sigma.push_back(temp_sig);
      if(shifted){
        Mix_Shift.push_back(Get_Shifted_Value(temp_ep, temp_sig, CutOffSquared));}else{Mix_Shift.push_back(0.0);}
      if(tail){
      throw std::runtime_error("Tail correction not implemented YET...");} //Zhao's note: need to implement tail correction later //
      Mix_Z.push_back(0.0);
      Mix_Type.push_back(0);
    }
  }
  printf("NumberofDefinition: %zu, Mx_shift: %zu\n", NumberOfDefinitions, Mix_Shift.size());
  /*for(size_t i = 0; i < Mix_Shift.size(); i++)
  {
    size_t ii = i/NumberOfDefinitions; size_t jj = i%NumberOfDefinitions; printf("i: %zu, ii: %zu, jj: %zu", i,ii,jj);
    printf("i: %zu, ii: %zu, jj: %zu, Name_i: %s, Name_j: %s, ep: %.10f, sig: %.10f, shift: %.10f\n", i,ii,jj,PseudoAtom.Name[ii].c_str(), PseudoAtom.Name[jj].c_str(), Mix_Epsilon[i], Mix_Sigma[i], Mix_Shift[i]);
  }*/
  double* result; int* int_result;
  result = Doubleconvert1DVectortoArray(Mix_Epsilon); FF.epsilon = result;
  result = Doubleconvert1DVectortoArray(Mix_Sigma);   FF.sigma   = result;
  result = Doubleconvert1DVectortoArray(Mix_Z);       FF.z       = result;
  result = Doubleconvert1DVectortoArray(Mix_Shift);   FF.shift   = result;
  int_result = Intconvert1DVectortoArray(Mix_Type);   FF.FFType  = int_result;
  FF.size = NumberOfDefinitions;
}

void PseudoAtomParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom)
{
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0;
  std::ifstream PseudoAtomfile("pseudo_atoms.def");
  size_t NumberOfPseudoAtoms = 0; 
  // First read the pseudo atom file
  while (std::getline(PseudoAtomfile, str))
  {
    if(counter == 1) //read number definitions
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[0].c_str(), "%zu", &NumberOfPseudoAtoms);
      if (NumberOfPseudoAtoms <= 0 || NumberOfPseudoAtoms>200) throw std::runtime_error("Incorrect amount of pseudo-atoms");//DON'T DO TOO MANY
      if (NumberOfPseudoAtoms != FF.size) throw std::runtime_error("Number of VDW and pseudo-atom definitions don't match!"); 
    }
    else if(counter >= 3) // read data for each pseudo atom
    {
      termsScannedLined = split(str, ' ');
      if(termsScannedLined[0] != PseudoAtom.Name[counter-3]) throw std::runtime_error("Order of pseudo-atom and force field definition don't match!");
      PseudoAtom.oxidation.push_back(std::stod(termsScannedLined[4]));
      PseudoAtom.mass.push_back(std::stod(termsScannedLined[5]));
      PseudoAtom.charge.push_back(std::stod(termsScannedLined[6]));
      PseudoAtom.polar.push_back(std::stod(termsScannedLined[7]));
    }
    counter++;
    if(counter==3+NumberOfPseudoAtoms) break; //in case there are extra empty rows
  }
  //print out the values
  for (size_t i = 0; i < NumberOfPseudoAtoms; i++)
    printf("Name: %s, %.10f, %.10f, %.10f, %.10f\n", PseudoAtom.Name[i].c_str(), PseudoAtom.oxidation[i], PseudoAtom.mass[i], PseudoAtom.charge[i], PseudoAtom.polar[i]);
}

void POSCARParser(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom)
{
  //bool UseChargesFromCIFFile = false;  //Zhao's note: if not, use charge from pseudo atoms file, not implemented (if reading poscar, then self-defined charges probably need a separate file //
  std::vector<std::string> Names = PseudoAtom.Name;
  size_t temp = 0;
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0;
  //Determine framework name (file name)//
  std::ifstream simfile("simulation.input");
  std::string poscarfile;
  while (std::getline(simfile, str))
  {
    if (str.find("FrameworkName", 0) != std::string::npos) // get the molecule name
    {
      termsScannedLined = split(str, ' ');
      poscarfile = termsScannedLined[1];
      break;
    }
  }
  std::ifstream file(poscarfile);
  std::vector<double> Cell(9); 
  bool cartesian = false; std::vector<std::string> elementList; std::vector<std::string> amountList; std::vector<std::string> directOrCartesian; 
  // skip first line
  while (std::getline(file, str))
  {
    if (counter == 2)
    {
      termsScannedLined = split(str, ' ');
      Cell[0] = std::stod(termsScannedLined[0]); //ax
      Cell[1] = std::stod(termsScannedLined[1]); //ay
      Cell[2] = std::stod(termsScannedLined[2]); //az
      printf("Cell a: %.10f\n", Cell[0], Cell[1], Cell[2]);
    }
    else if(counter == 3)
    {
      termsScannedLined = split(str, ' ');
      Cell[3] = std::stod(termsScannedLined[0]); //bx
      Cell[4] = std::stod(termsScannedLined[1]); //by
      Cell[5] = std::stod(termsScannedLined[2]); //bz
      printf("Cell b: %.10f\n", Cell[3], Cell[4], Cell[5]);
    }
    else if(counter == 4)
    {  
      termsScannedLined = split(str, ' ');
      Cell[6] = std::stod(termsScannedLined[0]); //cx
      Cell[7] = std::stod(termsScannedLined[1]); //cy
      Cell[8] = std::stod(termsScannedLined[2]); //cz
      printf("Cell c: %.10f\n", Cell[6], Cell[7], Cell[8]);
    }
    else if(counter == 5) // This is the line that records all the elements in the structure
    {
      elementList = split(str, ' ');
      if (elementList.empty()) { throw std::runtime_error("List of types of atoms is empty"); }
    }
    else if(counter == 6) // This is the line that records the number of atoms for each element
    {
      amountList = split(str, ' ');
      printf("str: %s\n", str.c_str());
      if (amountList.empty()) { throw std::runtime_error("List of amount of atoms is empty"); }
    }
    else if(counter == 7) // Direct or catesian
    {
      directOrCartesian = split(str, ' ');
      if (tolower(directOrCartesian[0]) == "cartesian") cartesian = true;
    }
    counter++;
  }
  file.clear();
  file.seekg(0);
  // READ ATOMS FOR EACH ELEMENT
  std::vector<size_t>REAL_Amount; //CONVERT THE STRINGS TO VALUES
  std::vector<size_t>Types;       //READ THE TYPES
  std::vector<double>Charges;     //THE CHARGES ASSIGNED IN pseudo-atoms def file
  for (size_t k = 0; k < amountList.size(); k++)
  {
    sscanf(amountList[k].c_str(), "%zu", &temp); REAL_Amount.push_back(temp);
    std::string chemicalElementString = trim2(elementList[k]);
    // COMPARE NAMES
    for(size_t i = 0; i < Names.size(); i++)
    {
      if(chemicalElementString == Names[i]){
        Types.push_back(i); Charges.push_back(PseudoAtom.charge[i]);
        printf("%s is Type %zu, Charge: %.5f\n", chemicalElementString.c_str(), i, PseudoAtom.charge[i]); break;}
    }
  }
  //Loop over all the atoms
  size_t linenum = 0; counter=0; size_t TypeCounter = 0; size_t atomtype = 0; double atomcharge = 0.0;
  std::vector<double>X; std::vector<double>Y; std::vector<double>Z; std::vector<double>SCALE; std::vector<double>CHARGE; std::vector<double>SCALECOUL;
  std::vector<size_t>TYPE; std::vector<size_t>MOLID;
  while (std::getline(file, str))
  {
    if(linenum > 7) //Loop over all the atoms
    {
      // Determine the atom type //
      TypeCounter = 0;
      for(size_t i = 0; i < REAL_Amount.size(); i++)
      {
        TypeCounter += REAL_Amount[i];
        if(counter < TypeCounter)
        {
          atomtype = Types[i]; atomcharge = Charges[i]; break;
        }
      }
      termsScannedLined = split(str, ' '); 
      if(!cartesian)
      {
        double fx = std::stod(termsScannedLined[0]); double fy = std::stod(termsScannedLined[1]); double fz = std::stod(termsScannedLined[2]);
        // Matrix 3x3 multiple by xyz 3x1 //
        X.push_back(fx*Cell[0]+fy*Cell[3]+fz*Cell[6]); 
        Y.push_back(fx*Cell[1]+fy*Cell[4]+fz*Cell[7]);
        Z.push_back(fx*Cell[2]+fy*Cell[5]+fz*Cell[8]);
      }else
      {
        X.push_back(std::stod(termsScannedLined[0]));
        Y.push_back(std::stod(termsScannedLined[1]));
        Z.push_back(std::stod(termsScannedLined[2]));
      } 
      //Zhao's note: need to take care of charge later // 
      SCALE.push_back(1.0); CHARGE.push_back(atomcharge); SCALECOUL.push_back(1.0);
      TYPE.push_back(atomtype); MOLID.push_back(0);
      counter++;
    }
    linenum++;
  }
  printf("linenum: %zu, counter is %zu\n", linenum, counter);
  //CHECK CHARGE NEUTRALITY//
  double sum_of_charge = std::accumulate(CHARGE.begin(), CHARGE.end(), 0); if(sum_of_charge > 1e-50) throw std::runtime_error("Framework not neutral, not allowed for now...\n");
  //Get the numbers//
  //for(size_t i = 0; i < counter; i++)
  //  printf("i: %zu, xyz: %.10f, %.10f, %.10f, Type: %zu\n", i, X[i], Y[i], Z[i], TYPE[i]);
  double* result; result = Doubleconvert1DVectortoArray(Cell); Box.Cell = result;
  inverse_matrix(Box.Cell, &result); Box.InverseCell = result;
  Box.Volume = matrix_determinant(Box.Cell);
  //DETERMINE Whether Box is cubic/cuboid or not//
  Box.Cubic = true; // Start with cubic box shape, if any value (non-diagonal) is greater than 0, then set to false //
  if((Cell[1] + Cell[2] + Cell[3] + Cell[5] + Cell[6] + Cell[7]) > 1e-10) Box.Cubic = false;
  if(Box.Cubic)  printf("The Simulation Box is Cubic\n");
  if(!Box.Cubic) printf("The Simulation Box is NOT Cubic\n");
  //read the data into the framework struct//
  result = Doubleconvert1DVectortoArray(X); Framework.x = result;
  result = Doubleconvert1DVectortoArray(Y); Framework.y = result;
  result = Doubleconvert1DVectortoArray(Z);     Framework.z = result;
  result = Doubleconvert1DVectortoArray(SCALE); Framework.scale = result;
  result = Doubleconvert1DVectortoArray(CHARGE); Framework.charge = result;
  result = Doubleconvert1DVectortoArray(SCALECOUL); Framework.scaleCoul = result;
  size_t* size_t_result; 
  size_t_result = Size_tconvert1DVectortoArray(TYPE); Framework.Type = size_t_result;
  size_t_result = Size_tconvert1DVectortoArray(MOLID); Framework.MolID = size_t_result;
  Framework.Allocate_size = counter; Framework.size = counter;
}

size_t get_type_from_name(std::string Name, std::vector<std::string> PseudoAtomNames)
{
  size_t type = 0; bool match = false;
  for(size_t i = 0; i < PseudoAtomNames.size(); i++)
  {
    if(Name == PseudoAtomNames[i]){
      type = i; match = true; break;}
  }
  if(!match) throw std::runtime_error("Atom type not found in pseudo atoms definitions\n");
  return type;
}

void MoleculeDefinitionParser(Atoms& Mol, Components& SystemComponents, std::string MolName, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space)
{
  //check if molecule definition file exists
  const std::string MolFileName = MolName + ".def";
  std::filesystem::path pathfile = std::filesystem::path(MolFileName);
  if (!std::filesystem::exists(pathfile))
  {
    throw std::runtime_error("Definition file not found\n");
  }
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0; size_t temp_molsize = 0; size_t atomcount = 0;
  std::ifstream file(MolFileName);

  Mol.Allocate_size = Allocate_space;
  std::vector<double> Ax(Allocate_space);
  std::vector<double> Ay(Allocate_space);
  std::vector<double> Az(Allocate_space);
  std::vector<double> Ascale(Allocate_space);
  std::vector<double> Acharge(Allocate_space);
  std::vector<double> AscaleCoul(Allocate_space);
  std::vector<size_t> AType(Allocate_space);
  std::vector<size_t> AMolID(Allocate_space);
  
  double chargesum = 0.0; //a sum of charge for the atoms in the molecule, for checking charge neutrality
 
  // skip first line
  while (std::getline(file, str))
  {
    if(counter == 1) //read Tc
    {
      termsScannedLined = split(str, ' ');
      SystemComponents.Tc.push_back(std::stod(termsScannedLined[0]));
    }
    else if(counter == 2) //read Pc
    {
      termsScannedLined = split(str, ' ');
      SystemComponents.Pc.push_back(std::stod(termsScannedLined[0]));
    }
    else if(counter == 3) //read Accentric Factor
    {
      termsScannedLined = split(str, ' ');
      SystemComponents.Accentric.push_back(std::stod(termsScannedLined[0]));
    }
    else if(counter == 5) //read molecule size
    {
      temp_molsize = 0;
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[0].c_str(), "%zu", &temp_molsize);
      if(temp_molsize >= Allocate_space) throw std::runtime_error("Molecule size is greater than allocated size. Break\n");
      SystemComponents.Moleculesize.push_back(temp_molsize);
    }
    else if(counter >= 13 && atomcount < (temp_molsize)) //read atomic positions. Zhao's note: skipped reading groups for now
    {
      //for atomic positions, read them into Mol, at the positions for the first molecule.
      //Don't set the size of Mol, it is zero, since the data loaded is only a template.
      termsScannedLined = split(str, ' ');
      if(termsScannedLined.size() == 5) //for example: 0 CH4 0.0 0.0 0.0, position provided here.
      {
        Ax[atomcount] = std::stod(termsScannedLined[2]);
        Ay[atomcount] = std::stod(termsScannedLined[3]);
        Az[atomcount] = std::stod(termsScannedLined[4]);
      }
      else if(termsScannedLined.size() == 2 && temp_molsize == 1) //like methane, one can do: 0 CH4, with no positions
      {
        Ax[atomcount] = 0.0; Ay[atomcount] = 0.0; Az[atomcount] = 0.0;
      }
      else
      {
        throw std::runtime_error("Flexible molecules not implemented\n");
      }
      //Set other values for each atom
      Ascale[atomcount]  = 1.0;
      AscaleCoul[atomcount] = 1.0;
      std::string AtomName = termsScannedLined[1];
      AType[atomcount] = get_type_from_name(AtomName, PseudoAtom.Name); 
      Acharge[atomcount] = PseudoAtom.charge[AType[atomcount]]; //Determine the charge after determining the type
      chargesum += PseudoAtom.charge[AType[atomcount]]; 
      AMolID[atomcount] = 0;// Molecule ID = 0, since it is in the first position
      atomcount++;
    }
    else if(counter > (13 + temp_molsize +1)) //read bonding information
    {
      printf("Bonds not implemented. Break\n"); break;
    }
    counter++; 
  }
  if(chargesum > 1e-50) throw std::runtime_error("Molecule not neutral, bad\n");
  double* result;
  result = Doubleconvert1DVectortoArray(Ax);            Mol.x = result;
  result = Doubleconvert1DVectortoArray(Ay);            Mol.y = result;
  result = Doubleconvert1DVectortoArray(Az);            Mol.z = result;
  result = Doubleconvert1DVectortoArray(Ascale);        Mol.scale = result;
  result = Doubleconvert1DVectortoArray(Acharge);       Mol.charge = result;
  result = Doubleconvert1DVectortoArray(AscaleCoul);    Mol.scaleCoul = result;
  size_t* size_t_result;
  size_t_result = Size_tconvert1DVectortoArray(AType);  Mol.Type = size_t_result;
  size_t_result = Size_tconvert1DVectortoArray(AMolID); Mol.MolID = size_t_result;
}

void read_component_values_from_simulation_input(Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom, size_t* MoleculeNeedToCreate, size_t Allocate_space)
{
  //adsorbate component start from zero, but in the code, framework is zero-th component
  //This function also calls MoleculeDefinitionParser//
  size_t component = AdsorbateComponent+1;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  int counter=0; size_t start_counter = 0;
  double TranslationProb=0.0; double RotationProb=0.0; double WidomProb=0.0; double ReinsertionProb = 0.0; double SwapProb = 0.0; double TotalProb=0.0;
  size_t CreateMolecule = 0; double idealrosen = 0.0; double fugacoeff = 0.0; double Molfrac = 1.0; //Set Molfraction = 1.0
  std::string start_string = "Component " + std::to_string(AdsorbateComponent); //start when reading "Component 0" for example
  std::string terminate_string="Component " + std::to_string(component);     //terminate when reading "Component 1", if we are interested in Component 0
  //first get the line number of the destinated component
  while (std::getline(file, str))
  {
    if(str.find(start_string, 0) != std::string::npos){break;}
    start_counter++;
  }
  printf("%s starts at %zu\n", start_string.c_str(), start_counter); 
  file.clear();
  file.seekg(0); 
  while (std::getline(file, str))
  {
    if(str.find(terminate_string, 0) != std::string::npos){break;}
    if(counter >= start_counter) //start reading after touching the starting line number
    {
      if (str.find(start_string, 0) != std::string::npos) // get the molecule name
      {
        termsScannedLined = split(str, ' ');
        SystemComponents.MoleculeName.push_back(termsScannedLined[3]);
        MoleculeDefinitionParser(Mol, SystemComponents, termsScannedLined[3], PseudoAtom, Allocate_space);
      }
      if (str.find("IdealGasRosenbluthWeight", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        idealrosen = std::stod(termsScannedLined[1]); printf("idealrosen: %.10f\n", idealrosen);
      }
      if (str.find("TranslationProbability", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        TranslationProb=std::stod(termsScannedLined[1]);
        TotalProb+=TranslationProb;
      }
      if (str.find("RotationProbability", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        RotationProb=std::stod(termsScannedLined[1]);
        TotalProb+=RotationProb;
      }
      if (str.find("WidomProbability", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        WidomProb=std::stod(termsScannedLined[1]);
        TotalProb+=WidomProb;
      }
      if (str.find("ReinsertionProbability", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        ReinsertionProb=std::stod(termsScannedLined[1]);
        TotalProb+=ReinsertionProb;
      }
      if (str.find("SwapProbability", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        SwapProb=std::stod(termsScannedLined[1]);
        TotalProb+=SwapProb;
      }
      if (str.find("FugacityCoefficient", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        fugacoeff = std::stod(termsScannedLined[1]);
      }
      if (str.find("MolFraction", 0) != std::string::npos)
      {
         termsScannedLined = split(str, ' ');
         Molfrac = std::stod(termsScannedLined[1]);
      }
      if (str.find("CreateNumberOfMolecules", 0) != std::string::npos) // Number of Molecules to create
      { //Zhao's note: if create zero, nothing happens
        //If create 1, add 1 to Numberofmoleculeofcomponent, but no need to actually create molecules, the other parameters are set when reading mol definition//
        //If create more than 1, then we need to actually create the molecule by doing insertions, lets throw an error for now//
        termsScannedLined = split(str, ' ');
        sscanf(termsScannedLined[1].c_str(), "%zu", &CreateMolecule);
        //if(CreateMolecule > 1) throw std::runtime_error("Create more than 1 molecule is not allowed for now...");
      }
    }
    counter++;
  }
  TranslationProb/=TotalProb; RotationProb/=TotalProb; WidomProb/=TotalProb; SwapProb/=TotalProb; ReinsertionProb/=TotalProb;
  //Zhao's note: for the sake of the cleaness of the code, do accumulated probs//
  RotationProb    += TranslationProb;
  WidomProb       += RotationProb;
  ReinsertionProb += WidomProb;
  SwapProb        += ReinsertionProb;
  printf("Translation Prob: %.10f, Rotation Prob: %.10f, Widom Prob: %.10f, Swap Prob\n", TranslationProb, RotationProb, WidomProb, SwapProb);
  MoveStats.TranslationProb=TranslationProb; MoveStats.RotationProb=RotationProb; MoveStats.WidomProb=WidomProb; MoveStats.ReinsertionProb=ReinsertionProb; MoveStats.SwapProb=SwapProb;
  SystemComponents.NumberOfMolecule_for_Component.push_back(0); // Zhao's note: Molecules are created later in main.cpp //
  SystemComponents.Allocate_size.push_back(Allocate_space);
  if(idealrosen < 1e-150) throw std::runtime_error("Ideal-Rosenbluth weight not assigned (or not valid), bad. If rigid, assign 1.");
  SystemComponents.IdealRosenbluthWeight.push_back(idealrosen);
  //Zhao's note: for fugacity coefficient, if not assigned (0.0), do Peng-Robinson
  if(fugacoeff < 1e-150)
  {
    throw std::runtime_error("Need to do Peng-rob, but not implemented yet...");
  } 
  SystemComponents.FugacityCoeff.push_back(fugacoeff);
  //Zhao's note: for now, Molfraction = 1.0
  SystemComponents.MolFraction.push_back(Molfrac);
  *MoleculeNeedToCreate = CreateMolecule;
  //SystemComponents.TotalNumberOfMolecules+=CreateMolecule;
  //Finally, check if all values in SystemComponents are set properly//
  Check_Component_size(SystemComponents);
  //Initialize single values for Mol//
  Mol.size = 0;
}
