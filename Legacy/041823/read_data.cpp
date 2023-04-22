#include <filesystem>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>

#include <algorithm> //for remove_if

#include <iostream>

#include "convert_array.h"
#include "data_struct.h"
#include "matrix_manipulation.h"

#define MAX2(x,y) (((x)>(y))?(x):(y))                 // the maximum of two numbers
#define MAX3(x,y,z) MAX2((x),MAX2((y),(z)))        

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
  printf("reference size: %zu\n", referenceVal);
  if(SystemComponents.Moleculesize.size() != referenceVal)                   throw std::runtime_error("Moleculesize does not match!");
  if(SystemComponents.NumberOfMolecule_for_Component.size() != referenceVal) throw std::runtime_error("NumberOfMolecule_for_Component does not match!");
  if(SystemComponents.MolFraction.size() != referenceVal)                    throw std::runtime_error("MolFraction does not match!");
  if(SystemComponents.IdealRosenbluthWeight.size() != referenceVal)          throw std::runtime_error("IdealRosenbluthWeight does not match!");
  if(SystemComponents.FugacityCoeff.size() != referenceVal)                  throw std::runtime_error("FugacityCoeff does not match!");
  if(SystemComponents.Tc.size() != referenceVal)                             throw std::runtime_error("Tc does not match!");
  if(SystemComponents.Pc.size() != referenceVal)                             throw std::runtime_error("Pc does not match!");
  if(SystemComponents.Accentric.size() != referenceVal)                      throw std::runtime_error("Accentric does not match!");
  if(SystemComponents.rigid.size() != referenceVal)                          throw std::runtime_error("Rigidity (boolean vector) not match!");
  if(SystemComponents.hasfractionalMolecule.size() != referenceVal)          throw std::runtime_error("HasFractionalMolecule (boolean vector) not match!");
  if(SystemComponents.Lambda.size() != referenceVal)                         throw std::runtime_error("Lambda (fractional component vector) not match!");
  if(SystemComponents.Tmmc.size() != referenceVal)                         throw std::runtime_error("Tmmc (TMMC vector) not match!");
  printf("CreateMolecule size: %zu\n", SystemComponents.NumberOfCreateMolecules.size());
  if(SystemComponents.NumberOfCreateMolecules.size() != referenceVal)        throw std::runtime_error("Molecules need to create not match!");
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

void read_simulation_input(bool *UseGPUReduction, bool *Useflag, bool *noCharges, int *InitializationCycles, int *EquilibrationCycles, int *ProductionCycles, size_t *Widom_Trial, size_t *Widom_Orientation, size_t *NumberOfBlocks, double *Pressure, double *Temperature, size_t *AllocateSize, bool *ReadRestart, double *RANDOMSEED, bool *SameFrameworkEverySimulation)
{
  bool tempGPU = false; bool tempflag = false; bool nochargeflag = true;  //Ignore the changes if the chargemethod is not specified
  bool tempDualPrecision = false;

  bool tempRestart = false;  //Whether we read restart file or not

  int initializationcycles=1; int equilibrationcycles=0; int productioncycles=0;
  size_t widom = 8;
  double temp = 0.0;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  int counter=0; size_t Nblock=0; size_t tempallocspace=0; size_t widom_orientation = 8;
  double pres = 0.0;

  // Zhao's note: setup random seed in this function //
  double randomseed = 0.0;

  bool tempSameFrameworkEverySimulation = true;

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
  
    if (str.find("RandomSeed", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      randomseed = static_cast<double>(std::stoi(termsScannedLined[1]));
      printf("Random Seed is %.5f\n", randomseed);
    }

    if (str.find("AdsorbateAllocateSpace", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &tempallocspace);
      printf("line is %u, Allocate space for adsorbate is %zu\n", counter, tempallocspace);
    }
    if (str.find("NumberOfInitializationCycles", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%d", &initializationcycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfEquilibrationCycles", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%d", &equilibrationcycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfProductionCycles", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%d", &productioncycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("Widom_Trial", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &widom);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("Widom_Orientation", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &widom_orientation);
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
    if (str.find("RestartFile", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempRestart = true;
        printf("found Restart flag\n");
      }
    }
    if (str.find("DifferentFrameworks", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempSameFrameworkEverySimulation = false;
        printf("found SameFrameworkEverySimulation flag\n");
      }
    }
    if(counter>200) break;
  }
  *UseGPUReduction=tempGPU; *Useflag=tempflag; *noCharges = nochargeflag;
  *InitializationCycles=initializationcycles; *EquilibrationCycles=equilibrationcycles; *ProductionCycles=productioncycles;
  *Widom_Trial=widom; *NumberOfBlocks=Nblock; *Widom_Orientation=widom_orientation;
  *Pressure = pres; *Temperature = temp;
  *AllocateSize  = tempallocspace;
  *ReadRestart   = tempRestart;
  *RANDOMSEED    = randomseed;
  *SameFrameworkEverySimulation = tempSameFrameworkEverySimulation;
  //Setup RandomSeed here//
  //std::srand(randomseed);
}

void read_Gibbs_Stats(Gibbs& GibbsStatistics, bool& SetMaxStep, size_t& MaxStepPerCycle)
{
  size_t counter = 0;
  double temp = 0.0;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");

  while (std::getline(file, str))
  {
    counter++;
    if (str.find("GibbsVolumeChangeProbability", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      temp=std::stod(termsScannedLined[1]);
      if(temp > 0)
      {
        GibbsStatistics.DoGibbs = true;
        GibbsStatistics.GibbsBoxProb = temp;
      }
    }
    if (str.find("GibbsSwapProbability", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      temp=std::stod(termsScannedLined[1]);
      if(temp > 0)
      {
        GibbsStatistics.DoGibbs = true;
        GibbsStatistics.GibbsXferProb = temp;
      }
    }
    if (str.find("UseMaxStep", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        SetMaxStep = true;
        printf("Using Max Steps for a cycle!\n");
      }
    }
    if (str.find("MaxStepPerCycle", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[1].c_str(), "%zu", &MaxStepPerCycle);
      if(MaxStepPerCycle == 0) throw std::runtime_error("Max Steps per Cycle must be greater than ZERO!");
    }
    if(counter>200) break;
  }
  GibbsStatistics.GibbsBoxStats  = {0.0, 0.0};
  GibbsStatistics.GibbsXferStats = {0.0, 0.0};
}

void read_FFParams_from_input(ForceField& FF, double& precision)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;

  double tempOverlap = 1.0e6; double tempvdwcut = 12.0; double tempcoulcut = 12.0;
  double tempprecision = 1.0e-6;
  //double tempalpha = 0.26506; //Zhao's note: here we used alpha equal to the preset value from raspa3. Need to revisit RASPA-2 for the exact calculation of alpha.
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
      //tempalpha = (1.35 - 0.15 * log(tempprecision))/tempcoulcut; // Zhao's note: heurestic equation //
    }
    if (str.find("CBMCBiasingMethod", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[1], "LJ_And_Real_Biasing"))
      {
        FF.VDWRealBias = true; //By default, it is using LJ + Real Biasing
      }
      else if(caseInSensStringCompare(termsScannedLined[1], "LJ_Biasing"))
      {
        FF.VDWRealBias = false;
      }
    }
    if (str.find("Component", 0) != std::string::npos) //When it reads component, skip//
      break;
  }
  //read FF array
  FF.OverlapCriteria   = tempOverlap;
  FF.CutOffVDW         = tempvdwcut*tempvdwcut;
  FF.CutOffCoul        = tempcoulcut*tempcoulcut;
  precision            = tempprecision;
  //Box.Prefactor        = tempprefactor;
  //double tol = sqrt(fabs(log(tempprecision*tempcoulcut)));
  //tempalpha  = sqrt(fabs(log(tempprecision*tempcoulcut*tol)))/tempcoulcut;
  //double tol1= sqrt(-log(tempprecision*tempcoulcut*pow(2.0*tol*tempalpha, 2)));
  //Box.Alpha             = tempalpha;
  //Zhao's note: See InitializeEwald function in RASPA-2.0 //
  //Box.kmax.x           = std::round(0.25 + Box.Cell[0] * tempalpha * tol1/3.1415926);
  //Box.kmax.y           = std::round(0.25 + Box.Cell[4] * tempalpha * tol1/3.1415926);
  //Box.kmax.z           = std::round(0.25 + Box.Cell[8] * tempalpha * tol1/3.1415926);
  //Box.ReciprocalCutOff = pow(1.05*static_cast<double>(MAX3(Box.kmax.x, Box.kmax.y, Box.kmax.z)), 2);
  //printf("tol: %.5f, tol1: %.5f, ALpha is %.5f, Prefactor: %.5f, kmax: %d %d %d, ReciprocalCutOff: %.5f\n", tol, tol1, Box.Alpha, Box.Prefactor, Box.kmax.x, Box.kmax.y, Box.kmax.z, Box.ReciprocalCutOff);
}

void read_Ewald_Parameters_from_input(double CutOffCoul, Boxsize& Box, double precision)
{
  double tempprefactor = 138935.48350;
  double tempalpha = 0.26506; //Zhao's note: here we used alpha equal to the preset value from raspa3. Need to revisit RASPA-2 for the exact calculation of alpha.
  Box.Prefactor = tempprefactor;
  double tol = sqrt(fabs(log(precision*CutOffCoul)));
  tempalpha  = sqrt(fabs(log(precision*CutOffCoul*tol)))/CutOffCoul;
  double tol1= sqrt(-log(precision*CutOffCoul*pow(2.0*tol*tempalpha, 2)));
  Box.tol1   = tol1;
  Box.Alpha  = tempalpha;
  //Zhao's note: See InitializeEwald function in RASPA-2.0 //
  Box.kmax.x = std::round(0.25 + Box.Cell[0] * tempalpha * tol1/3.1415926);
  Box.kmax.y = std::round(0.25 + Box.Cell[4] * tempalpha * tol1/3.1415926);
  Box.kmax.z = std::round(0.25 + Box.Cell[8] * tempalpha * tol1/3.1415926);
  Box.ReciprocalCutOff = pow(1.05*static_cast<double>(MAX3(Box.kmax.x, Box.kmax.y, Box.kmax.z)), 2);
  printf("tol: %.5f, tol1: %.5f, ALpha is %.5f, Prefactor: %.5f, kmax: %d %d %d, ReciprocalCutOff: %.5f\n", tol, tol1, Box.Alpha, Box.Prefactor, Box.kmax.x, Box.kmax.y, Box.kmax.z, Box.ReciprocalCutOff);
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
  double CutOffSquared = 144.0; 
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

static inline size_t GetTypeForPseudoAtom(PseudoAtomDefinitions& PseudoAtom, std::string& AtomName)
{
  bool Found = false;
  size_t AtomTypeInt = 0;
  for(size_t j = 0; j < PseudoAtom.Name.size(); j++)
  {
    if(AtomName == PseudoAtom.Name[j])
    {
      AtomTypeInt = j;
      Found = true;
      break;
    }
  }
  if(!Found){throw std::runtime_error("Overwriting terms are not Found in Pseudo atoms!!!");}
  return AtomTypeInt;
}

static inline double GetTailCorrectionValue(size_t IJ, ForceField& FF)
{
  double scaling = 1.0;
  double arg1 = FF.epsilon[IJ];
  double sigma= FF.sigma[IJ];
  double arg2 = sigma * sigma * sigma;
  double rr = sqrt(FF.CutOffVDW);
  double term1= pow(arg2, 4) / (9.0 * pow(rr, 9)); //sigma^12/(9*r^9)
  double term2= pow(arg2, 2) / (3.0 * pow(rr, 3)); //sigma^6 /(3*r^3)
  double val = 16.0 * 3.14159265358979323846 / 2.0 * arg1 * (term1 - term2);
  return val;
}

static inline void PrepareTailCorrection(size_t i, size_t j, std::vector<Tail>& TempTail, PseudoAtomDefinitions& PseudoAtom, ForceField& FF)
{
  size_t IJ_Forward = i * FF.size + j;
  size_t IJ_Reverse = j * FF.size + i;
  TempTail[IJ_Forward].UseTail= true;
  TempTail[IJ_Forward].Energy = GetTailCorrectionValue(IJ_Forward, FF);
  if(i!=j) TempTail[IJ_Reverse] = TempTail[IJ_Forward];
  printf("TypeI: %zu, TypeJ: %zu, FF.size: %zu, Energy: %.5f\n", i, j, FF.size, TempTail[IJ_Forward].Energy);
}
//Function for Overwritten tail corrections
//For now, it only considers tail correction//
void OverWriteFFTerms(Components& SystemComponents, ForceField& FF, PseudoAtomDefinitions& PseudoAtom)
{
  size_t FFsize = FF.size;
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0;
  size_t OverWriteSize = 0;
  size_t typeI; size_t typeJ;
  std::vector<Tail>TempTail(FFsize * FFsize);
  std::ifstream OverWritefile("force_field.def");
  std::filesystem::path pathfile = std::filesystem::path("force_field.def");
  if (!std::filesystem::exists(pathfile))
  {
    //throw std::runtime_error("Force Field OverWrite file not found\n");
    printf("Force Field OverWrite file not found\n");
    SystemComponents.HasTailCorrection = false;
    SystemComponents.TailCorrection = TempTail;
    return;
  }
  else
  {
    SystemComponents.HasTailCorrection = true;
  } 
  while (std::getline(OverWritefile, str))
  {
    if(counter == 1) //read OverWriteSize
    {
      termsScannedLined = split(str, ' ');
      sscanf(termsScannedLined[0].c_str(), "%zu", &OverWriteSize);
    }
    else if(counter >= 3 && counter < (3 + OverWriteSize)) //read Terms to OverWrite
    {
      termsScannedLined = split(str, ' ');
      if(termsScannedLined[3] == "yes") //Use Tail Correction or not//
      {
        typeI = GetTypeForPseudoAtom(PseudoAtom, termsScannedLined[0]);
        typeJ = GetTypeForPseudoAtom(PseudoAtom, termsScannedLined[1]);
        PrepareTailCorrection(typeI, typeJ, TempTail, PseudoAtom, FF);
      }
    }
    counter ++;
    if(counter==3 + OverWriteSize) break; //in case there are extra empty rows, Zhao's note: I am skipping the mixing rule, assuming Lorentz-Berthelot
  }
  //Eliminate the terms that do not have tail corrections//
  for(size_t i = 0; i < TempTail.size(); i++)
    SystemComponents.TailCorrection.push_back(TempTail[i]);
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

void Split_Tab_Space(std::vector<std::string>& termsScannedLined, std::string& str)
{
  if (str.find("\t", 0) != std::string::npos) //if the delimiter is tab
  {
    termsScannedLined = split(str, '\t');
  }
  else
  {
    termsScannedLined = split(str, ' ');
  }
}

void remove_number(std::string& s)
{
  s.erase(std::remove_if(std::begin(s), std::end(s), [](auto ch) { return std::isdigit(ch); }), s.end());
}

void CheckFrameworkCIF(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom, std::string& Frameworkfile, bool UseChargeFromCIF, double3 NumberUnitCells, Components& SystemComponents)
{
  std::vector<std::string> termsScannedLined{};
  termsScannedLined = split(Frameworkfile, '.');
  std::string frameworkName = termsScannedLined[0];
  std::string CIFFile = frameworkName + ".cif";
  printf("CIF FILE IS: %s\n", CIFFile.c_str());
  std::ifstream simfile(CIFFile);
  std::filesystem::path pathfile = std::filesystem::path(CIFFile);
  if (!std::filesystem::exists(pathfile))
  {
    throw std::runtime_error("CIF file not found\n");
  }
  std::string str;

  printf("Number of Unit Cells: %.5f %.5f %.5f\n", NumberUnitCells.x, NumberUnitCells.y, NumberUnitCells.z);

  //Temp Vector For Counting Number of Pseudo Atoms in the framework//
  int2 tempint2 = {0, 0};
  std::vector<int2>TEMPINTTWO(PseudoAtom.Name.size(), tempint2);

  //double3 NumberUnitCells;
  //NumberUnitCells.x = 1; NumberUnitCells.y = 1; NumberUnitCells.z = 1;

  //Read angles, cell lengths, and volume//
  double3 angle; //x = alpha; y = beta; z = gamma;
  double3 abc; 
  double3 axbycz; //Diagonal of the matrix//
  while (std::getline(simfile, str))
  {
    if (str.find("_cell_length_a", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      abc.x = std::stod(termsScannedLined[1]);
    }
    if (str.find("_cell_length_b", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      abc.y = std::stod(termsScannedLined[1]);
    }
    if (str.find("_cell_length_c", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      abc.z = std::stod(termsScannedLined[1]);
    }
    if (str.find("_cell_angle_alpha", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      angle.x = std::stod(termsScannedLined[1]) / (180.0/3.14159265358979323846);
    }
    if (str.find("_cell_angle_beta", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      angle.y = std::stod(termsScannedLined[1]) / (180.0/3.14159265358979323846);
    }
    if (str.find("_cell_angle_gamma", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      angle.z = std::stod(termsScannedLined[1]) / (180.0/3.14159265358979323846);
    }
  }
  //Get xy(bx), xz(cx), and yz(cy)//
  axbycz.x = abc.x;
  axbycz.y = abc.y * std::sin(angle.z);
  double tempd = (std::cos(angle.x)-std::cos(angle.z)*std::cos(angle.y))/std::sin(angle.z);
  axbycz.z = abc.z * sqrt(1 - pow(std::cos(angle.y), 2) - pow(tempd, 2));
  double bx = abc.y * std::cos(angle.z);
  double cx = abc.z * std::cos(angle.y);
  double cy = abc.z * tempd;
  printf("Box size: \n%.10f %.5f %.5f\n%.10f %.10f %.5f\n%.10f %.10f %.10f\n", axbycz.x, 0.0, 0.0, bx, axbycz.y, 0.0, cx, cy, axbycz.z);
  Box.Cell = (double*) malloc(9 * sizeof(double));
  Box.Cell[0] = NumberUnitCells.x * axbycz.x; Box.Cell[1] = 0.0;                          Box.Cell[2] = 0.0;
  Box.Cell[3] = NumberUnitCells.y * bx;       Box.Cell[4] = NumberUnitCells.y * axbycz.y; Box.Cell[5] = 0.0;
  Box.Cell[6] = NumberUnitCells.z * cx;       Box.Cell[7] = NumberUnitCells.z * cy;       Box.Cell[8] = NumberUnitCells.z * axbycz.z;
  simfile.clear();
  simfile.seekg(0);

  // Check the atom_site keyword: get its first and last occurance//
  //.x is the start, .y is the end//
  int2 atom_site_occurance; atom_site_occurance.x = 0; atom_site_occurance.y = 0; int count=0;
  int atomsiteCount = 0;
  bool foundline = false; 
  //Initialize the order of the required columns in the CIF file with -1 (meaning non-existent)
  std::vector<int>Label_x_y_z_Charge_Order(5,-1);
  while (std::getline(simfile, str))
  {
    if (str.find("_atom_site", 0) != std::string::npos)
    {
      //Zhao's note: this relies on the fact that a cif file cannot start with _atom_site as the first line//
      if(atom_site_occurance.x == 0) atom_site_occurance.x = count;
      foundline = true;
      if(str.find("_atom_site_label", 0) != std::string::npos){  Label_x_y_z_Charge_Order[0] = atomsiteCount;}
      if(str.find("_atom_site_fract_x", 0) != std::string::npos){  Label_x_y_z_Charge_Order[1] = atomsiteCount;}
      if(str.find("_atom_site_fract_y", 0) != std::string::npos){  Label_x_y_z_Charge_Order[2] = atomsiteCount;}
      if(str.find("_atom_site_fract_z", 0) != std::string::npos){  Label_x_y_z_Charge_Order[3] = atomsiteCount;}
      if(str.find("_atom_site_charge", 0) != std::string::npos){ Label_x_y_z_Charge_Order[4] = atomsiteCount;}
      atomsiteCount++;
    }
    else //if cannot find _atom_site in this line, check if the "foundline" variable (meaning that if the keyword can be found in the previous line//
    {
      if(foundline)//Zhao's note: the assumes that "atom_site" only appears in one not multiple regions in the cif file
      {
        atom_site_occurance.y = count-1;
        break;
      }
    }
    count++;
  }
  //If label/x/y/z not found, abort the program!//
  if(Label_x_y_z_Charge_Order[0] < 0 || Label_x_y_z_Charge_Order[1] < 0 || Label_x_y_z_Charge_Order[2] < 0 || Label_x_y_z_Charge_Order[3] < 0) throw std::runtime_error("Couldn't find required columns in the CIF file! Abort.");
  printf("atom_site starts at line %d, and ends at %d\n", atom_site_occurance.x, atom_site_occurance.y);
  simfile.clear();
  simfile.seekg(0);

  //Loop Over the Atoms//
  double x; double y; double z; size_t Type; double Charge = 0.0;
  size_t AtomID = 0;
  int label_location  = Label_x_y_z_Charge_Order[0];
  int x_location      = Label_x_y_z_Charge_Order[1];
  int y_location      = Label_x_y_z_Charge_Order[2];
  int z_location      = Label_x_y_z_Charge_Order[3];
  int Charge_location = Label_x_y_z_Charge_Order[4];
  if(!UseChargeFromCIF) Charge_location = -1; //If we want to use Charge from the pseudo_atoms.def, sed charge_location to -1//
  printf("label location: %d, xyz location: %d %d %d, charge: %d\n", label_location, x_location, y_location, z_location, Charge_location);
  std::vector<double>super_x;
  std::vector<double>super_y;
  std::vector<double>super_z;
  std::vector<double>super_scale;
  std::vector<double>super_charge;
  std::vector<double>super_scaleCoul;
  std::vector<size_t>super_Type;
  std::vector<double>super_MolID;
  size_t i = 0;
  ////////////////////////////////////////////////////////////////
  //Zhao's note: For making supercells, the code can only do P1!//
  ////////////////////////////////////////////////////////////////
  double3 Shift;
  Shift.x = (double)1/NumberUnitCells.x; Shift.y = (double)1/NumberUnitCells.y; Shift.z = (double)1/NumberUnitCells.z;
  while (std::getline(simfile, str))
 
   {
    if(i <= atom_site_occurance.y){i++; continue;}
    AtomID = i - atom_site_occurance.y - 1;
    Split_Tab_Space(termsScannedLined, str);
    //Check when the elements in the line is less than 4, stop when it is less than 4 (it means the atom_site region is over//
    if(termsScannedLined.size() < 4) 
    {
      printf("Done reading Atoms in CIF file, there are %d Atoms\n", AtomID);
      break;
    }
    //printf("i: %zu, line: %s, splitted: %s\n", i, str.c_str(), termsScannedLined[0].c_str());
    double fx = std::stod(termsScannedLined[x_location]); //fx*=Shift.x; //Divide by the number of UnitCells
    double fy = std::stod(termsScannedLined[y_location]); //fy*=Shift.y; 
    double fz = std::stod(termsScannedLined[z_location]); //fz*=Shift.z;
    if(Charge_location >= 0) Charge = std::stod(termsScannedLined[Charge_location]);
    std::string AtomName = termsScannedLined[label_location];
    //Zhao's note: try to remove numbers from the Atom labels//
    remove_number(AtomName);
    size_t AtomTypeInt = 0;
    //Get the type (int) for this AtomName//
    bool AtomTypeFOUND = false;
    for(size_t j = 0; j < PseudoAtom.Name.size(); j++)
    {
      if(AtomName == PseudoAtom.Name[j])
      {
        AtomTypeInt = j;
        AtomTypeFOUND = true;
        if(!UseChargeFromCIF) Charge = PseudoAtom.charge[j];
        //Add to the number of pseudo atoms
        TEMPINTTWO[j].x =j;
        TEMPINTTWO[j].y ++;
        break;
      }
    }
    if(!AtomTypeFOUND){ printf("Couldn't find %s Type in Pseudo-Atoms\n", AtomName.c_str()); throw std::runtime_error("Error: Atom Label not defined!");}
    Type = AtomTypeInt;
    for(size_t ix = 0; ix < NumberUnitCells.x; ix++)
      for(size_t jy = 0; jy < NumberUnitCells.y; jy++)
        for(size_t kz = 0; kz < NumberUnitCells.z; kz++)
        {
          size_t UnitCellID = (ix * NumberUnitCells.y + jy) * NumberUnitCells.z + kz;
          //Get supercell fx, fy, fz, and get corresponding xyz//
          double super_fx = (fx + ix) * Shift.x;
          double super_fy = (fy + jy) * Shift.y;
          double super_fz = (fz + kz) * Shift.z;
          // Get real xyz from fractional xyz //
          x = super_fx*Box.Cell[0]+super_fy*Box.Cell[3]+super_fz*Box.Cell[6];
          y = super_fx*Box.Cell[1]+super_fy*Box.Cell[4]+super_fz*Box.Cell[7];
          z = super_fx*Box.Cell[2]+super_fy*Box.Cell[5]+super_fz*Box.Cell[8];
          super_x.push_back(x);
          super_y.push_back(y);
          super_z.push_back(z);
          super_scale.push_back(1.0); //For framework, use 1.0
          super_charge.push_back(Charge);
          super_scaleCoul.push_back(1.0);//For framework, use 1.0
          super_Type.push_back(Type);
          super_MolID.push_back(0); //Framework is usually component zero
        }
    i++;
  }
  Framework.x         = (double*) malloc(super_x.size() * sizeof(double));
  Framework.y         = (double*) malloc(super_x.size() * sizeof(double));
  Framework.z         = (double*) malloc(super_x.size() * sizeof(double));
  Framework.scale     = (double*) malloc(super_x.size() * sizeof(double));
  Framework.charge    = (double*) malloc(super_x.size() * sizeof(double));
  Framework.scaleCoul = (double*) malloc(super_x.size() * sizeof(double));
  Framework.Type      = (size_t*) malloc(super_x.size() * sizeof(size_t));
  Framework.MolID     = (size_t*) malloc(super_x.size() * sizeof(size_t));
  for(size_t i = 0; i < super_x.size(); i++)
  {
    Framework.x[i] = super_x[i]; Framework.y[i] = super_y[i]; Framework.z[i] = super_z[i];
    Framework.scale[i] = super_scale[i]; Framework.charge[i] = super_charge[i]; Framework.scaleCoul[i] = super_scaleCoul[i];
    Framework.Type[i] = super_Type[i]; Framework.MolID[i] = super_MolID[i];
  }
  Framework.size = super_x.size(); Framework.Molsize = super_x.size(); Framework.Allocate_size = Framework.size;
  //Record Number of PseudoAtoms for the Framework//
  std::vector<int2>NumberOfPseudoAtoms;
  for(size_t i = 0; i < TEMPINTTWO.size(); i++)
  {
    if(TEMPINTTWO[i].y == 0) continue; 
    TEMPINTTWO[i].y *= NumberUnitCells.x * NumberUnitCells.y * NumberUnitCells.z;
    NumberOfPseudoAtoms.push_back(TEMPINTTWO[i]);
    printf("Framework Pseudo Atom[%zu], Name: %s, #: %zu\n", TEMPINTTWO[i].x, PseudoAtom.Name[i].c_str(), TEMPINTTWO[i].y);
  }
  SystemComponents.NumberOfPseudoAtomsForSpecies.push_back(NumberOfPseudoAtoms);
}

void POSCARParser(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom, std::string& poscarfile)
{
  std::vector<std::string> Names = PseudoAtom.Name;
  size_t temp = 0;
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0;
  //Determine framework name (file name)//
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
        //printf("Type: %zu (%s), there are %zu atoms\n", i, Names[i].c_str(), REAL_Amount[i]);
        if(counter < TypeCounter)
        {
          atomtype = Types[i]; atomcharge = Charges[i]; break;
        }
      }
      //if(linenum == 1807) printf("Line 1807, type is %zu\n", atomtype);
      //if(linenum == 1808) printf("Line 1808, type is %zu\n", atomtype);
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
  sum_of_charge = std::accumulate(CHARGE.begin(), CHARGE.end(), 0); if(sum_of_charge > 1e-50) throw std::runtime_error("Framework not neutral, not allowed for now...\n");
}

void ReadFramework(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom, size_t FrameworkIndex, Components& SystemComponents)
{
  bool UseChargesFromCIFFile = true;  //Zhao's note: if not, use charge from pseudo atoms file, not implemented (if reading poscar, then self-defined charges probably need a separate file //
  std::vector<std::string> Names = PseudoAtom.Name;
  size_t temp = 0;
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t counter = 0;
  //Determine framework name (file name)//
  std::ifstream simfile("simulation.input");
  std::string Frameworkname;
  std::string InputType="cif";

  double3 NumberUnitCells;
  bool FrameworkFound = false;

  while (std::getline(simfile, str))
  {
    if (str.find("InputFileType", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      InputType=termsScannedLined[1];
    }
    //Zhao's note: When we are using Multiple frameworks, in simulation.input, list the frameworks one by one.//
    if (str.find("FrameworkName", 0) != std::string::npos) // get the molecule name
    {
      termsScannedLined = split(str, ' ');
      if((1+FrameworkIndex) >= termsScannedLined.size()) throw std::runtime_error("Not Enough Framework listed in input file...\n");
      Frameworkname = termsScannedLined[1 + FrameworkIndex];
      printf("Reading Framework %zu, FrameworkName: %s\n", FrameworkIndex, Frameworkname.c_str());
    }
    if (str.find("UnitCells " + std::to_string(FrameworkIndex), 0) != std::string::npos) // get the molecule name
    {
      termsScannedLined = split(str, ' ');
      NumberUnitCells.x = std::stod(termsScannedLined[2]);
      NumberUnitCells.y = std::stod(termsScannedLined[3]);
      NumberUnitCells.z = std::stod(termsScannedLined[4]);
      printf("Reading Framework %zu, UnitCells: %.2f %.2f %.2f\n", FrameworkIndex, NumberUnitCells.x, NumberUnitCells.y, NumberUnitCells.z);
      FrameworkFound = true;
    }
  }
  //If not cif or poscar, break the program!//
  if(InputType != "cif" && InputType != "poscar") throw std::runtime_error("Cannot identify framework input type. It can only be cif or poscar!");
  if(!FrameworkFound) throw std::runtime_error("Cannot find the framework with matching index!");
  std::string FrameworkFile = Frameworkname + "." + InputType;
  std::filesystem::path pathfile = std::filesystem::path(FrameworkFile);
  if (!std::filesystem::exists(pathfile)) throw std::runtime_error("Framework File not found!\n");
  /////////////////////////////////////////////////////////////////////////////////////
  //Zhao's note:                                                                     //
  //If reading poscar, then you cannot have user-defined charges for every atom      //
  //the charges used for poscar are defined in pseudo_atoms.def                      //
  //To use user-defined charge for every atom, use cif format                        //
  /////////////////////////////////////////////////////////////////////////////////////
  if(InputType == "poscar")
  {
    POSCARParser(Box, Framework, PseudoAtom, FrameworkFile);
  }
  else if(InputType == "cif")
  {
    CheckFrameworkCIF(Box, Framework, PseudoAtom, FrameworkFile, UseChargesFromCIFFile, NumberUnitCells, SystemComponents);
  }
  //Get Volume, cubic/non-cubic of the box//
  Box.InverseCell = (double*) malloc(9 * sizeof(double));
  inverse_matrix(Box.Cell, &Box.InverseCell);
  Box.Volume = matrix_determinant(Box.Cell);
  //DETERMINE Whether Box is cubic/cuboid or not//
  Box.Cubic = true; // Start with cubic box shape, if any value (non-diagonal) is greater than 0, then set to false //
  if((fabs(Box.Cell[3]) + fabs(Box.Cell[6]) + fabs(Box.Cell[7])) > 1e-10) Box.Cubic = false;
  if(Box.Cubic)  printf("The Simulation Box is Cubic\n");
  if(!Box.Cubic) printf("The Simulation Box is NOT Cubic\n");
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
 
  bool temprigid = false;

  size_t PseudoAtomSize = PseudoAtom.Name.size();
  size_t ComponentSize  = SystemComponents.Total_Components;
  //For Tail corrections//
  int2   TempINTTWO = {0, 0};
  std::vector<int2> ANumberOfPseudoAtomsForSpecies(PseudoAtomSize, TempINTTWO);

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
      Mol.Molsize = temp_molsize; //Set Molsize for the adsorbate molecule here//
    }
    else if(counter == 9) //read if the molecule is rigid
    {
      termsScannedLined = split(str, ' ');
      if(caseInSensStringCompare(termsScannedLined[0], "rigid"))
      {
        temprigid = true;
        printf("Adsorbate Component is rigid\n");
      }
      else
      {
        throw std::runtime_error("Currently Not allowing flexible molecule\n");
      }
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
      ANumberOfPseudoAtomsForSpecies[AType[atomcount]].x = AType[atomcount];
      ANumberOfPseudoAtomsForSpecies[AType[atomcount]].y ++;
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
  //Remove Elements from ANumberOfPseudoAtomsForSpecies if the ANumberOfPseudoAtomsForSpecies.y = 0
  std::vector<int2>TEMPINTTWO;
  for(size_t i = 0; i < ANumberOfPseudoAtomsForSpecies.size(); i++)
  {
    if(ANumberOfPseudoAtomsForSpecies[i].y == 0) continue;
    TEMPINTTWO.push_back(ANumberOfPseudoAtomsForSpecies[i]);
    printf("Adsorbate Type[%zu], Name: %s, #: %zu\n", ANumberOfPseudoAtomsForSpecies[i].x, PseudoAtom.Name[i].c_str(), ANumberOfPseudoAtomsForSpecies[i].y);
  }
  SystemComponents.NumberOfPseudoAtomsForSpecies.push_back(TEMPINTTWO);
  SystemComponents.rigid.push_back(temprigid);
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
  printf("AxAyAz: %.5f %.5f %.5f; Molxyz: %.5f %.5f %.5f\n", Ax[0], Ay[0], Az[0], Mol.x[0], Mol.y[0], Mol.z[0]);
}

void read_component_values_from_simulation_input(Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space)
{
  //adsorbate component start from zero, but in the code, framework is zero-th component
  //This function also calls MoleculeDefinitionParser//
  size_t component = AdsorbateComponent+1;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  int counter=0; size_t start_counter = 0;
  double TranslationProb=0.0; double RotationProb=0.0; double WidomProb=0.0; double ReinsertionProb = 0.0; double SwapProb = 0.0; double CBCFProb = 0.0; double TotalProb=0.0;
  size_t CreateMolecule = 0; double idealrosen = 0.0; double fugacoeff = 0.0; double Molfrac = 1.0; //Set Molfraction = 1.0
  bool temp_hasfracmol = false;
  int  LambdaType = SHI_MAGINN;

  TMMC temp_tmmc;

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
      if (str.find("CBCFProbability", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        CBCFProb=std::stod(termsScannedLined[1]);
        TotalProb+=CBCFProb;
        temp_hasfracmol=true;
      }
      //Zhao's note: If using CBCF Move, choose the lambda type//
      if (CBCFProb > 0.0)
      {
        if (str.find("LambdaType", 0) != std::string::npos)
        {
          termsScannedLined = split(str, ' ');
          if(caseInSensStringCompare(termsScannedLined[1], "ShiMaginn"))
          {
            LambdaType = SHI_MAGINN;
          }
          else if(caseInSensStringCompare(termsScannedLined[1], "BrickCFC"))
          {
            LambdaType = BRICK_CFC;
          }
        }
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
      if (str.find("RunTMMC", 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        if(caseInSensStringCompare(termsScannedLined[1], "yes"))
        {
          temp_tmmc.DoTMMC = true;
          printf("running TMMC simulation\n");
        }
      }
      if(temp_tmmc.DoTMMC)
      {
        if (str.find("TMMCMin", 0) != std::string::npos)
        {
          termsScannedLined = split(str, ' ');
          sscanf(termsScannedLined[1].c_str(), "%zu", &temp_tmmc.MinMacrostate);
        }
        else if (str.find("TMMCMax", 0) != std::string::npos)
        {
          termsScannedLined = split(str, ' ');
          sscanf(termsScannedLined[1].c_str(), "%zu", &temp_tmmc.MaxMacrostate);
        }
        if (str.find("UseBiasOnMacrostate", 0) != std::string::npos)
        {
          termsScannedLined = split(str, ' ');
          if(caseInSensStringCompare(termsScannedLined[1], "yes"))
          {
            temp_tmmc.DoUseBias = true;
            printf("Biasing Insertion/Deletions\n");
          }
        }
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
  TranslationProb/=TotalProb; RotationProb/=TotalProb; WidomProb/=TotalProb; SwapProb/=TotalProb; CBCFProb/=TotalProb; ReinsertionProb/=TotalProb;
  //Zhao's note: for the sake of the cleaness of the code, do accumulated probs//
  RotationProb    += TranslationProb;
  WidomProb       += RotationProb;
  ReinsertionProb += WidomProb;
  CBCFProb        += ReinsertionProb;
  SwapProb        += CBCFProb;
  printf("Translation Prob: %.10f, Rotation Prob: %.10f, Widom Prob: %.10f, CBCFProb: %.10f, Swap Prob\n", TranslationProb, RotationProb, WidomProb, CBCFProb, SwapProb);
  MoveStats.TranslationProb=TranslationProb; MoveStats.RotationProb=RotationProb; MoveStats.WidomProb=WidomProb; MoveStats.ReinsertionProb=ReinsertionProb; MoveStats.CBCFProb=CBCFProb; MoveStats.SwapProb=SwapProb; 
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
  SystemComponents.hasfractionalMolecule.push_back(temp_hasfracmol);
  
  LAMBDA lambda;
  if(temp_hasfracmol) //Prepare lambda struct if using CBCF//
  {
    lambda.newBin    = 0;
    lambda.delta     = 1.0/static_cast<double>(lambda.binsize); //Zhao's note: in raspa3, delta is 1/(nbin-1)
    lambda.WangLandauScalingFactor = 0.0; lambda.FractionalMoleculeID = 0;
    lambda.lambdatype = LambdaType;
    lambda.Histogram.resize(lambda.binsize); lambda.biasFactor.resize(lambda.binsize);
    std::fill(lambda.Histogram.begin(),  lambda.Histogram.end(),  0.0);
    std::fill(lambda.biasFactor.begin(), lambda.biasFactor.end(), 0.0);
  }
 
  if(temp_tmmc.DoTMMC) //Prepare tmmc struct if using TMMC//
  { 
    if(temp_tmmc.MaxMacrostate < temp_tmmc.MinMacrostate)
    {
      throw std::runtime_error("Bad Min/Max Macrostates for TMMC, Min has to be SMALLER THAN OR EQUAL TO Max.");
    }
    temp_tmmc.CMatrix.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    temp_tmmc.WLBias.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    temp_tmmc.TMBias.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    std::fill(temp_tmmc.TMBias.begin(), temp_tmmc.TMBias.end(), 1.0); //Initialize the TMBias//
    temp_tmmc.ln_g.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    temp_tmmc.lnpi.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    temp_tmmc.forward_lnpi.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    temp_tmmc.reverse_lnpi.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    temp_tmmc.Histogram.resize(temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);
    //Zhao's note: if we set the bounds for min/max macrostate, the number of createMolecule should fall in the range//
    if(temp_tmmc.RejectOutofBound && !SystemComponents.ReadRestart)
      if(CreateMolecule < temp_tmmc.MinMacrostate || CreateMolecule > temp_tmmc.MaxMacrostate)
        throw std::runtime_error("Number of created molecule fall out of the TMMC Macrostate range!");
  }

  SystemComponents.Lambda.push_back(lambda);
  SystemComponents.Tmmc.push_back(temp_tmmc);
  SystemComponents.NumberOfCreateMolecules.push_back(CreateMolecule);
  //Finally, check if all values in SystemComponents are set properly//
  Check_Component_size(SystemComponents);
  //Initialize single values for Mol//
  Mol.size = 0;
}

void RestartFileParser(Simulations& Sims, Atoms* Host_System, Components& SystemComponents)
{
  bool UseChargesFromCIFFile = true;  //Zhao's note: if not, use charge from pseudo atoms file, not implemented (if reading poscar, then self-defined charges probably need a separate file //
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  //Determine framework name (file name)//
  std::string Filename = "RestartInitial/System_0/restartfile";
  std::ifstream file(Filename);
  std::filesystem::path pathfile = std::filesystem::path(Filename);
  if (!std::filesystem::exists(pathfile))
  {
    throw std::runtime_error("RestartInitial file not found\n");
  }
  size_t counter = 0;
  for(size_t i = SystemComponents.NumberOfFrameworks; i < SystemComponents.Total_Components; i++)
  {
    size_t start = 0; size_t end = 0;
    while (std::getline(file, str))
    {
      //find range of the part we need to read// 
      if (str.find("Components " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos) 
        start = counter;
      if (str.find("Maximum-rotation-change component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
      {  end = counter; break; }
    }
    file.clear();
    file.seekg(0);
    //Start reading//
    while (std::getline(file, str))
    {
      if (str.find("Fractional-molecule-id component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        if(std::stoi(termsScannedLined[3]) == -1) SystemComponents.hasfractionalMolecule[i] = false;
        if(SystemComponents.hasfractionalMolecule[i])
        {
          sscanf(termsScannedLined[3].c_str(), "%zu", &SystemComponents.Lambda[i].FractionalMoleculeID);
          printf("Fractional Molecule ID: %zu\n", SystemComponents.Lambda[i].FractionalMoleculeID);
        }
      }
      if(SystemComponents.hasfractionalMolecule[i])
      {
        if (str.find("Lambda-factors component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
        {
          termsScannedLined = split(str, ' ');
          SystemComponents.Lambda[i].WangLandauScalingFactor = std::stod(termsScannedLined[3]);
          printf("WL Factor: %.5f\n", SystemComponents.Lambda[i].WangLandauScalingFactor);
        }
        if (str.find("Number-of-biasing-factors component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
        {
          termsScannedLined = split(str, ' ');
          sscanf(termsScannedLined[3].c_str(), "%zu", &SystemComponents.Lambda[i].binsize);
          printf("binsize: %zu\n", SystemComponents.Lambda[i].binsize);
          if(SystemComponents.Lambda[i].binsize != SystemComponents.Lambda[i].Histogram.size()) throw std::runtime_error("CFC Bin size don't match!");
        }
        if (str.find("Biasing-factors component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
        {
          for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
          {
            termsScannedLined = split(str, ' ');
            SystemComponents.Lambda[i].biasFactor[j] = std::stod(termsScannedLined[3 + j]); 
            printf("Biasing Factor %zu: %.5f\n", j, SystemComponents.Lambda[i].biasFactor[j]); 
          }
        }
      }
      //Read translation rotation maxes//
      if (str.find("Maximum-translation-change component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        Sims.MaxTranslation.x = std::stod(termsScannedLined[3]);
        Sims.MaxTranslation.y = std::stod(termsScannedLined[4]);
        Sims.MaxTranslation.z = std::stod(termsScannedLined[5]);
      }
      if (str.find("Maximum-rotation-change component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        Sims.MaxRotation.x = std::stod(termsScannedLined[3]);
        Sims.MaxRotation.y = std::stod(termsScannedLined[4]);
        Sims.MaxRotation.z = std::stod(termsScannedLined[5]);
        break;
      }
    }
    file.clear(); 
    file.seekg(0);
    //Start reading atom positions and other information//
    start = 0; end = 0; counter = 0;
    while (std::getline(file, str))
    {
      if (str.find("Component: " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
      {
        termsScannedLined = split(str, ' ');
        start = counter + 2;
        size_t temp; 
        sscanf(termsScannedLined[3].c_str(), "%zu", &temp);
        SystemComponents.NumberOfMolecule_for_Component[i] = temp;
        SystemComponents.TotalNumberOfMolecules += SystemComponents.NumberOfMolecule_for_Component[i];
        //printf("There are %zu Molecules, molsize = %zu, line is %zu\n", SystemComponents.NumberOfMolecule_for_Component[i], SystemComponents.Moleculesize[i], counter);
        break; 
      }
      counter++;
    }

    //Update The Number of pseudoAtoms//
    for(size_t Nmol = 0; Nmol < SystemComponents.NumberOfMolecule_for_Component[i]; Nmol++)
      SystemComponents.UpdatePseudoAtoms(INSERTION, i);

    if(SystemComponents.Tmmc[i].DoTMMC && SystemComponents.Tmmc[i].RejectOutofBound)
      if(SystemComponents.NumberOfMolecule_for_Component[i] < SystemComponents.Tmmc[i].MinMacrostate || SystemComponents.NumberOfMolecule_for_Component[i] > SystemComponents.Tmmc[i].MaxMacrostate)
        throw std::runtime_error("Number of molecule fall out of the TMMC Macrostate range!");

    counter  = 0;
    size_t interval = SystemComponents.NumberOfMolecule_for_Component[i]* SystemComponents.Moleculesize[i];
    double x[SystemComponents.NumberOfMolecule_for_Component[i]         * SystemComponents.Moleculesize[i]];
    double y[SystemComponents.NumberOfMolecule_for_Component[i]         * SystemComponents.Moleculesize[i]];
    double z[SystemComponents.NumberOfMolecule_for_Component[i]         * SystemComponents.Moleculesize[i]];
    double scale[SystemComponents.NumberOfMolecule_for_Component[i]     * SystemComponents.Moleculesize[i]];
    double charge[SystemComponents.NumberOfMolecule_for_Component[i]    * SystemComponents.Moleculesize[i]];
    double scaleCoul[SystemComponents.NumberOfMolecule_for_Component[i] * SystemComponents.Moleculesize[i]];
    size_t Type[SystemComponents.NumberOfMolecule_for_Component[i]      * SystemComponents.Moleculesize[i]];
    size_t MolID[SystemComponents.NumberOfMolecule_for_Component[i]     * SystemComponents.Moleculesize[i]];

    file.clear();
    file.seekg(0);
    size_t atom=0;
    while (std::getline(file, str))
    {
      //Read positions, Type and MolID//
      //Position: 0, velocity: 1, force: 2, charge: 3, scaling: 4
      if((counter >= start) && (counter < start + interval))
      {
        atom=counter - start;
        if (!(str.find("Adsorbate-atom-position", 0) != std::string::npos)) throw std::runtime_error("Cannot find matching strings in the range for reading positions!");
        termsScannedLined = split(str, ' ');
        x[atom] = std::stod(termsScannedLined[3]); y[atom] = std::stod(termsScannedLined[4]); z[atom] = std::stod(termsScannedLined[5]);
        sscanf(termsScannedLined[1].c_str(), "%zu", &MolID[atom]);
        size_t atomid; sscanf(termsScannedLined[2].c_str(), "%zu", &atomid);
        Type[atom] = Host_System[i].Type[atomid]; //for every component, the types of atoms for the first molecule is always there, just copy it//
        //printf("Reading Positions, atom: %zu, xyz: %.5f %.5f %.5f, Type: %zu, MolID: %zu\n", atom, x[atom], y[atom], z[atom], Type[atom], MolID[atom]);
      }
      //Read charge//
      if((counter >= start + interval * 3) && (counter < start + interval * 4))
      { 
        atom = counter - (start + interval * 3);
        termsScannedLined = split(str, ' ');
        charge[atom] = std::stod(termsScannedLined[3]);
        //printf("Reading charge, atom: %zu, charge: %.5f\n", atom, charge[atom]);
      }
      //Read scaling and scalingCoul//
      atom=0;
      if((counter >= start + interval * 4) && (counter < start + interval * 5))
      {
        atom = counter - (start + interval * 4);
        termsScannedLined  = split(str, ' ');
        double  lambda     = std::stod(termsScannedLined[3]);
        double2 val; val.x = 1.0; val.y = 1.0;
        if(lambda < 1.0) val = SystemComponents.Lambda[i].SET_SCALE(lambda);
        scale[atom]     = val.x;
        scaleCoul[atom] = val.y;
        //Determine the molecule ID// 
        size_t tempMOLID = 0;
        sscanf(termsScannedLined[1].c_str(), "%zu", &tempMOLID);
        //Determine the currentBin for the fractional molecule//
        if(tempMOLID == SystemComponents.Lambda[i].FractionalMoleculeID)
        { 
          printf("Lambda from RestartInitial is: %.100f\n", lambda);
          //floor/ceil functions
          double smallEpsilon = 1e-5; //
          size_t currentBIN = static_cast<size_t>(lambda/SystemComponents.Lambda[i].delta);
          //Zhao's note: do a casting test (0.7 is actually 0.69999, when using static cast, when delta is 0.1, the bin is 6 (should be 7)//
          double doubBin = static_cast<double>(currentBIN) * SystemComponents.Lambda[i].delta;
          if(abs(doubBin - lambda) > 1e-3)
          {
            printf("static cast round off too much!");
            if((doubBin - lambda) > 1e-3)  currentBIN--;
            if(-(doubBin - lambda) > 1e-3) currentBIN++;
          }
          printf("CURRENT BIN is %zu\n", currentBIN);
          SystemComponents.Lambda[i].currentBin = currentBIN;
        }
        //printf("Reading scaling, atom: %zu, scale: %.5f %.5f\n", atom, scale[atom], scaleCoul[atom]);
      }
      counter ++;
    }
    for(size_t j = 0; j < interval; j++)
    {
      Host_System[i].x[j] = x[j]; Host_System[i].y[j] = y[j]; Host_System[i].z[j] = z[j]; 
      Host_System[i].charge[j] = charge[j]; Host_System[i].scale[j] = scale[j]; Host_System[i].scaleCoul[j] = scaleCoul[j]; 
      Host_System[i].Type[j] = Type[j]; Host_System[i].MolID[j] = MolID[j];
      //printf("Data for %zu: %.5f %.5f %.5f %.5f %.5f %.5f %zu %zu\n", j, Host_System[i].x[j], Host_System[i].y[j], Host_System[i].z[j], Host_System[i].charge[j], Host_System[i].scale[j], Host_System[i].scaleCoul[j], Host_System[i].Type[j], Host_System[i].MolID[j]);
    }
    Host_System[i].size = interval;
    Host_System[i].Molsize = SystemComponents.Moleculesize[i];
    /*cudaMemcpy(device_System[i].x,         x,         interval * sizeof(double), cudaMemcpyHostToHost); 
    cudaMemcpy(device_System[i].y,         y,         interval * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_System[i].z,         z,         interval * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_System[i].scale,     scale,     interval * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_System[i].charge,    charge,    interval * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_System[i].scaleCoul, scaleCoul, interval * sizeof(double), cudaMemcpyHostToDevice); 
    cudaMemcpy(device_System[i].Type,      Type,      interval * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_System[i].MolID,     MolID,     interval * sizeof(size_t), cudaMemcpyHostToDevice);*/
  }
}
//Function related to Read DNN Model//
std::vector<double2> ReadMinMax()
{
  std::vector<double2> MinMax;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("minimax.txt");
  while (std::getline(file, str))
  {
    termsScannedLined = split(str, ' ');
    //printf("%s\n", str.c_str());
    double min = std::stod(termsScannedLined[0]);
    double max = std::stod(termsScannedLined[1]);
    MinMax.push_back({min, max});
  }
  //printf("MinMax size: %zu\n", MinMax.size());
  return MinMax;
}

void ReadDNNModelNames(Components& SystemComponents)
{
  std::vector<double2> MinMax;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  while (std::getline(file, str))
  {
    if (str.find("DNNModelName", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      SystemComponents.ModelName.push_back(termsScannedLined[1]);
    }
    if (str.find("DNNInputLayer", 0) != std::string::npos)
    {
      termsScannedLined = split(str, ' ');
      SystemComponents.InputLayer.push_back(termsScannedLined[1]);
    }
  }
  printf("DNN Model Name: %s, DNN Input Layer: %s\n", SystemComponents.ModelName[0].c_str(), SystemComponents.InputLayer[0].c_str());
}
