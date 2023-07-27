#include <filesystem>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>

#include <algorithm> //for remove_if

#include <iostream>

//#include <print>

//#include "data_struct.h"
#include "VDW_Coulomb.cuh"
//#include "convert_array.h"
#include "read_data.h"

//#include <torch/script.h> // One-stop header.

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

//Zhao's note: DO NOT MIX TAB AND SPACE!!!
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

void FindIfInputIsThere(std::string& InputCommand, std::string& exepath)
{
  bool exist = false;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file(exepath + "/" + "read_data.cpp");
  while (std::getline(file, str))
  {
    if(str.find(InputCommand) != std::string::npos)
      exist = true;
  }
  if(!exist)
  {
    printf("%s Input Command Not Found!!!!\n", InputCommand.c_str());
    throw std::runtime_error("Program Abort due to Unfound Input Command!");
  }
}

void Check_Inputs_In_read_data_cpp(std::string& exepath)
{
  printf("Checking if all inputs are defined\n");
  std::vector<std::string> termsScannedLined{};
  //Zhao's note: Hard-coded executable name//
  termsScannedLined = split(exepath, '/');
  exepath = "/";
  for(size_t i = 0 ; i < termsScannedLined.size() - 1; i++) exepath = exepath + termsScannedLined[i] + "/";
  printf("True path of exe is %s\n", exepath.c_str());
  std::string str;
  std::ifstream file("simulation.input");
  size_t tempnum = 0; bool tempsingle = false; size_t counter = 0;
  while (std::getline(file, str))
  {
    counter++;
    Split_Tab_Space(termsScannedLined, str);
    if(termsScannedLined.size() == 0) continue;
    std::string InputCommand = termsScannedLined[0];
    FindIfInputIsThere(InputCommand, exepath);
  }
}

bool caseInSensStringCompare(const std::string& str1, const std::string& str2)
{
    return str1.size() == str2.size() && std::equal(str1.begin(), str1.end(), str2.begin(), [](auto a, auto b) {return std::tolower(a) == std::tolower(b); });
}

void Check_Component_size(Components& SystemComponents)
{
  size_t referenceVal = SystemComponents.MoleculeName.size();
  //printf("reference size: %zu\n", referenceVal);
  if(SystemComponents.Moleculesize.size() != referenceVal)                   throw std::runtime_error("Moleculesize does not match! reference: " + std::to_string(referenceVal) + " Vector: " + std::to_string(SystemComponents.Moleculesize.size()));
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
  //printf("CreateMolecule size: %zu\n", SystemComponents.NumberOfCreateMolecules.size());
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
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempsingle = true;
        printf("running only one simulation\n");
      }
    }
    if (str.find("NumberOfSimulations", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%zu", &tempnum);
      printf("There are %zu simulations\n", tempnum);
    }
  }
  *NumSims = tempnum; *SingleSim = tempsingle;
}

void read_simulation_input(bool *UseGPUReduction, bool *Useflag, bool *noCharges, int *InitializationCycles, int *EquilibrationCycles, int *ProductionCycles, size_t *NumberOfTrialPositions, size_t *NumberOfTrialOrientations, double *Pressure, double *Temperature, size_t *AllocateSize, bool *ReadRestart, int *RANDOMSEED, bool *SameFrameworkEverySimulation, int3& NumberOfComponents)
{
  bool tempGPU = false; bool tempflag = false; bool nochargeflag = true;  //Ignore the changes if the chargemethod is not specified
  //bool tempDualPrecision = false;

  bool tempRestart = false;  //Whether we read restart file or not

  int initializationcycles=1; int equilibrationcycles=0; int productioncycles=0;
  size_t widom = 8;
  double temp = 0.0;
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  int counter=0; size_t tempallocspace=0; size_t widom_orientation = 8;
  double pres = 0.0;

  size_t tempNComp = 0; 

  // Zhao's note: setup random seed in this function //
  double randomseed = 0.0;

  bool tempSameFrameworkEverySimulation = true;
  bool tempSeparateFramework = false;

  while (std::getline(file, str))
  {
    counter++;
    if (str.find("UseGPUReduction", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempGPU = true;
      }
    }
    if (str.find("Useflag", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempflag = true;
      }
    }
  
    if (str.find("RandomSeed", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      randomseed = std::stoi(termsScannedLined[1]);
      printf("Random Seed is %d\n", randomseed);
    }

    if (str.find("AdsorbateAllocateSpace", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%zu", &tempallocspace);
      printf("Allocate space for adsorbate is %zu\n", tempallocspace);
    }
    if (str.find("NumberOfInitializationCycles", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%d", &initializationcycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfEquilibrationCycles", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%d", &equilibrationcycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfProductionCycles", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%d", &productioncycles);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfTrialPositions", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%zu", &widom);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("NumberOfTrialOrientations", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%zu", &widom_orientation);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    //Zhao's note: Move it somewhere else//
    if (str.find("NumberOfBlocks", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      //printf("line is %u, there are %zu Framework Atoms\n", counter, NumberFrameworkAtom);
    }
    if (str.find("Pressure", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      pres=std::stod(termsScannedLined[1]);
    }
    if (str.find("Temperature", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      temp=std::stod(termsScannedLined[1]);
    }
    if (str.find("ChargeMethod", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "Ewald"))
      {
        nochargeflag = false;
        printf("USE EWALD SUMMATION FOR CHARGE\n");
      }
    }
    if (str.find("RestartFile", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempRestart = true;
        printf("USE CONFIGURATION FROM RESTARTINITIAL FILE\n");
      }
    }
    if (str.find("DifferentFrameworks", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempSameFrameworkEverySimulation = false;
      }
    }
    //Check number of adsorbates to process, need to put them in serial order//
    //Component 0, Component 1, ...//
    if (str.find("Component " + std::to_string(tempNComp), 0) != std::string::npos)
    {
      tempNComp ++;
    }
    //Check if we need to separate framework species
    //species could be cations, linkers, attachments to the nodes, ...//
    if (str.find("SeparateFrameworkComponents", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        tempSeparateFramework = true;
        printf("NEED TO SEPARATE FRAMEWORK COMPONENTS\n");
      }
    }
    //If we need to separate framework species, then read the following//
    //This requires us to put the "FrameworkComponents XXX" command after "SeparateFrameworkComponents" command//
    //Zhao's note: make sure there are two spaces before the actual command//
    if(tempSeparateFramework)
    {
      if(str.find("NumberofFrameworkComponents", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        NumberOfComponents.y = std::stoi(termsScannedLined[1]);
        printf("THERE ARE %zu SEPARATE FRAMEWORK COMPONENTS\n", NumberOfComponents.y);
      }
    }
    if(counter>200) break;
  }
  *UseGPUReduction=tempGPU; *Useflag=tempflag; *noCharges = nochargeflag;
  *InitializationCycles=initializationcycles; *EquilibrationCycles=equilibrationcycles; *ProductionCycles=productioncycles;
  *NumberOfTrialPositions=widom; *NumberOfTrialOrientations=widom_orientation;
  *Pressure = pres; *Temperature = temp;
  *AllocateSize  = tempallocspace;
  *ReadRestart   = tempRestart;
  *RANDOMSEED    = randomseed;
  *SameFrameworkEverySimulation = tempSameFrameworkEverySimulation;
  NumberOfComponents.z = tempNComp; //z component is the adsorbate components//
  NumberOfComponents.x = tempNComp + NumberOfComponents.y;
  //printf("Finished Checking Number of Components, There are %zu framework, %zu Adsorbates, %zu total Components\n", NumberOfComponents.y, NumberOfComponents.z, NumberOfComponents.x);
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
      Split_Tab_Space(termsScannedLined, str);
      temp=std::stod(termsScannedLined[1]);
      if(temp > 0)
      {
        GibbsStatistics.DoGibbs = true;
        GibbsStatistics.GibbsBoxProb = temp;
      }
    }
    if (str.find("GibbsSwapProbability", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      temp=std::stod(termsScannedLined[1]);
      if(temp > 0)
      {
        GibbsStatistics.DoGibbs = true;
        GibbsStatistics.GibbsXferProb = temp;
      }
    }
    if (str.find("UseMaxStep", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        SetMaxStep = true;
      }
    }
    if (str.find("MaxStepPerCycle", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[1].c_str(), "%zu", &MaxStepPerCycle);
      if(MaxStepPerCycle == 0) throw std::runtime_error("Max Steps per Cycle must be greater than ZERO!");
    }  
    if(counter>200) break;
  }
  if(SetMaxStep) 
  {
    printf("Setting Maximum Number of Steps for a Cycle, Max Step = %zu\n", MaxStepPerCycle);
  }
  else
  {
    printf("Running Cycles in the Normal Way\n");
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
      Split_Tab_Space(termsScannedLined, str);
      tempOverlap = std::stod(termsScannedLined[1]);
    }
    if (str.find("CutOffVDW", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      tempvdwcut = std::stod(termsScannedLined[1]);
    }
    if (str.find("CutOffCoulomb", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      tempcoulcut = std::stod(termsScannedLined[1]);
    }
    if (str.find("EwaldPrecision", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      tempprecision = std::stod(termsScannedLined[1]);
      //tempalpha = (1.35 - 0.15 * log(tempprecision))/tempcoulcut; // Zhao's note: heurestic equation //
    }
    if (str.find("CBMCBiasingMethod", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
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
}

void ReadFrameworkComponentMoves(Move_Statistics& MoveStats, Components& SystemComponents, size_t comp)
{
  if(SystemComponents.NComponents.y <= 1)
  { printf("Only one Framework Component, No moves assigned\n"); return;}
  if(comp >= SystemComponents.NComponents.y) return;
  printf("Checking Framework Moves for Framework Component %zu\n", comp);
  std::string FrameworkComponentName = "Framework_Component_" + std::to_string(comp);

  std::vector<std::string> termsScannedLined{};
  std::string str;

  std::ifstream file("simulation.input");

  size_t start_counter = 0; bool FOUND = false;
  std::string start_string = FrameworkComponentName;
  std::string terminate_string="END_OF_" + FrameworkComponentName;
  //first get the line number of the destinated component
  while (std::getline(file, str))
  {
    if(str.find(start_string, 0) != std::string::npos){FOUND = true; break;}
    start_counter++;
  }

  if(!FOUND){printf("%s not found in simulation.input\n", FrameworkComponentName.c_str()); return;}
  
  printf("%s starts at line number %zu\n", start_string.c_str(), start_counter);

  file.clear();
  file.seekg(0);

  size_t counter = 0;
  while (std::getline(file, str))
  {
    if(str.find(terminate_string, 0) != std::string::npos){break;}
    if(counter >= start_counter) //start reading after touching the starting line number
    if (str.find(FrameworkComponentName, 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      printf("Found Framework_Component_%s in simulation.input file\n", std::to_string(comp).c_str());
    }
    if (str.find("TranslationProbability", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      MoveStats.TranslationProb=std::stod(termsScannedLined[1]);
    }
    if (str.find("RotationProbability", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      MoveStats.RotationProb=std::stod(termsScannedLined[1]);
    }
    if (str.find("RotationSpecialProbability", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      MoveStats.SpecialRotationProb=std::stod(termsScannedLined[1]);
      printf("WARNING: Special Rotations are rotations with pre-set Rotation Axes, Rotation Axes, Angles are needed to define in def files for %s !\n", FrameworkComponentName.c_str());
    }
    counter ++;
  }
  MoveStats.NormalizeProbabilities();
  MoveStats.PrintProbabilities();
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
  printf("----------------EWALD SUMMATION SETUP-----------------\n");
  printf("tol: %.5f, tol1: %.5f\n", tol, tol1);
  printf("ALpha is %.5f, Prefactor: %.5f\n", Box.Alpha, Box.Prefactor);
  printf("kmax: %d %d %d, ReciprocalCutOff: %.5f\n", Box.kmax.x, Box.kmax.y, Box.kmax.z, Box.ReciprocalCutOff);
  printf("------------------------------------------------------\n");
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
  printf("------------------PARSING FORCE FIELD MIXING RULES----------------\n");
  // First read the pseudo atom file
  while (std::getline(PseudoAtomfile, str))
  {
    if(counter == 1) //read shifted/truncated
    {
      Split_Tab_Space(termsScannedLined, str);
      if(termsScannedLined[0] == "shifted")
        shifted = true;
    }
    else if(counter == 3) //read tail correction
    {
      Split_Tab_Space(termsScannedLined, str);
      if(termsScannedLined[0] == "yes")
      {
        tail = true; //Zhao's note: not implemented
        throw std::runtime_error("Tail correction not implemented YET...");
      }
    }
    else if(counter == 5) // read number of force field definitions
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[0].c_str(), "%zu", &NumberOfDefinitions);
      if (NumberOfDefinitions <= 0 || NumberOfDefinitions>200) throw std::runtime_error("Incorrect amount of force field definitions");
    }
    else if(counter >= 7) // read data for each force field definition
    {
      printf("%s\n", str.c_str());
      Split_Tab_Space(termsScannedLined, str);
      PseudoAtom.Name.push_back(termsScannedLined[0]);
      ep = std::stod(termsScannedLined[2]);
      sig= std::stod(termsScannedLined[3]);
      Epsilon.push_back(ep);
      Sigma.push_back(sig);
    }
    counter++;
    if(counter==7+NumberOfDefinitions) break; //in case there are extra empty rows, Zhao's note: I am skipping the mixing rule, assuming Lorentz-Berthelot
  }
  printf("------------------------------------------------------------------\n");
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
      throw std::runtime_error("Tail correction not implemented YET in force_field_mixing_rules.def, use the overwritting file force_field.def instead...");} 
      Mix_Z.push_back(0.0);
      Mix_Type.push_back(0);
    }
  }
  //For checking if mixing rule terms are correct//
  /*
  for(size_t i = 0; i < Mix_Shift.size(); i++)
  {
    size_t ii = i/NumberOfDefinitions; size_t jj = i%NumberOfDefinitions; printf("i: %zu, ii: %zu, jj: %zu", i,ii,jj);
    printf("i: %zu, ii: %zu, jj: %zu, Name_i: %s, Name_j: %s, ep: %.10f, sig: %.10f, shift: %.10f\n", i,ii,jj,PseudoAtom.Name[ii].c_str(), PseudoAtom.Name[jj].c_str(), Mix_Epsilon[i], Mix_Sigma[i], Mix_Shift[i]);
  }
  */
  FF.epsilon = convert1DVectortoArray(Mix_Epsilon);
  FF.sigma   = convert1DVectortoArray(Mix_Sigma);
  FF.z       = convert1DVectortoArray(Mix_Z);
  FF.shift   = convert1DVectortoArray(Mix_Shift);
  FF.FFType  = convert1DVectortoArray(Mix_Type);
  FF.size    = NumberOfDefinitions;
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
  if(!Found){throw std::runtime_error("Overwriting terms are not Found in Pseudo atoms!!! [" + AtomName + "]");}
  return AtomTypeInt;
}

static inline double GetTailCorrectionValue(size_t IJ, ForceField& FF)
{
  //double scaling = 1.0; Zhao's note: Need more care about fractional molecules with tail corrections
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
    printf("Force Field OverWrite file not found\n");
    SystemComponents.HasTailCorrection = false;
    SystemComponents.TailCorrection = TempTail;
    return;
  }
  else
  {
    SystemComponents.HasTailCorrection = true;
  } 
  printf("----------------FORCE FIELD OVERWRITTEN (TAIL CORRECTION) PARAMETERS----------------\n");
  while (std::getline(OverWritefile, str))
  {
    if(counter == 1) //read OverWriteSize
    {
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[0].c_str(), "%zu", &OverWriteSize);
    }
    else if(counter >= 3 && counter < (3 + OverWriteSize)) //read Terms to OverWrite
    {
      Split_Tab_Space(termsScannedLined, str);
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
  printf("------------------------------------------------------------------------------------\n");
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
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[0].c_str(), "%zu", &NumberOfPseudoAtoms);
      if (NumberOfPseudoAtoms <= 0 || NumberOfPseudoAtoms>200) throw std::runtime_error("Incorrect amount of pseudo-atoms");//DON'T DO TOO MANY
      if (NumberOfPseudoAtoms != FF.size) throw std::runtime_error("Number of VDW and pseudo-atom definitions don't match!"); 
    }
    else if(counter >= 3) // read data for each pseudo atom
    {
      Split_Tab_Space(termsScannedLined, str);
      if(termsScannedLined[0] != PseudoAtom.Name[counter-3]) throw std::runtime_error("Order of pseudo-atom and force field definition don't match!");
      PseudoAtom.Symbol.push_back(termsScannedLined[2]);
      //Match 1-to-1 list of pseudo_atom type and symbol type//
      size_t SymbolIdx = PseudoAtom.MatchSymbolTypeFromSymbolName(termsScannedLined[2]);
      PseudoAtom.SymbolIndex.push_back(SymbolIdx);
      PseudoAtom.oxidation.push_back(std::stod(termsScannedLined[4]));
      PseudoAtom.mass.push_back(std::stod(termsScannedLined[5]));
      PseudoAtom.charge.push_back(std::stod(termsScannedLined[6]));
      PseudoAtom.polar.push_back(std::stod(termsScannedLined[7]));
    }
    counter++;
    if(counter==3+NumberOfPseudoAtoms) break; //in case there are extra empty rows
  }
  //print out the values
  printf("-------------PARSING PSEUDO ATOMS FILE-------------\n");
  for (size_t i = 0; i < NumberOfPseudoAtoms; i++)
    printf("Name: %s, %.5f, %.5f, %.5f, %.5f\n", PseudoAtom.Name[i].c_str(), PseudoAtom.oxidation[i], PseudoAtom.mass[i], PseudoAtom.charge[i], PseudoAtom.polar[i]);
  printf("---------------------------------------------------\n");
}

void remove_number(std::string& s)
{
  s.erase(std::remove_if(std::begin(s), std::end(s), [](auto ch) { return std::isdigit(ch); }), s.end());
}

void DetermineFrameworkComponent(Components& SystemComponents, size_t AtomCountPerUnitcell, size_t& Atom_Comp, size_t& MolID)
{ //Default Atom_Comp = 0; MolID = 0;
  if(SystemComponents.NComponents.y <= 1) return; //Then no need to separate
  for(size_t i = 1; i < SystemComponents.FrameworkComponentDef.size(); i++) //Component//
  {
    for(size_t j = 0; j < SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule.size(); j++) //Molecule//
      for(size_t k = 0; k < SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule[j].size(); k++) //Atom//
        if(AtomCountPerUnitcell == SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule[j][k])
        {
          Atom_Comp = i; MolID = j; return;
        }
  }
}

void CheckFrameworkComponentAtomOrder(Components& SystemComponents, std::vector<std::vector<double3>>& unit_fpos, std::vector<std::vector<double>>& unit_scale, std::vector<std::vector<double>>& unit_charge, std::vector<std::vector<double>>& unit_scaleCoul, std::vector<std::vector<size_t>>& unit_Type, std::vector<std::vector<size_t>>& unit_MolID, std::vector<std::vector<size_t>>& unit_AtomIndex)
{
  if(SystemComponents.NComponents.y <= 1) return; //Then no need to Check
  for(size_t i = 1; i < SystemComponents.FrameworkComponentDef.size(); i++) //Component//
  {
    size_t AtomCounter = 0; //Record the Atom indices
    for(size_t j = 0; j < SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule.size(); j++) //Molecule//
      for(size_t k = 0; k < SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule[j].size(); k++) //Atom//
      {
        for(size_t l = 0; l < unit_AtomIndex[i].size(); l++)
        {
          if(SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule[j][k] == unit_AtomIndex[i][l])
          { 
            size_t From = l; size_t To = AtomCounter;
            double3 temp_fpos      = unit_fpos[i][To];
            double  temp_scale     = unit_scale[i][To];
            double  temp_charge    = unit_charge[i][To];
            double  temp_scaleCoul = unit_scaleCoul[i][To];
            size_t  temp_Type      = unit_Type[i][To];
            size_t  temp_MolID     = unit_MolID[i][To];
            size_t  temp_Index     = unit_AtomIndex[i][To];
            //Copy the current one to the correct location// 
            unit_fpos[i][To]      = unit_fpos[i][From];
            unit_scale[i][To]     = unit_scale[i][From];
            unit_charge[i][To]    = unit_charge[i][From];
            unit_scaleCoul[i][To] = unit_scaleCoul[i][From];
            unit_Type[i][To]      = unit_Type[i][From];
            unit_MolID[i][To]     = unit_MolID[i][From];
            unit_AtomIndex[i][To] = unit_AtomIndex[i][From];
            //Copy the replaced data to the New location (a swap)
            unit_fpos[i][From]      = temp_fpos;
            unit_scale[i][From]     = temp_scale;
            unit_charge[i][From]    = temp_charge;
            unit_scaleCoul[i][From] = temp_scaleCoul;
            unit_Type[i][From]      = temp_Type;
            unit_MolID[i][From]     = temp_MolID;
            unit_AtomIndex[i][From] = temp_Index;
            if(j != unit_MolID[i][To])
              throw std::runtime_error("MolID from Framework Component and MolID from CIF doesn't match!!!!"); 
          }
        }
        AtomCounter ++;
      }
  }
  
} 

void CheckFrameworkCIF(Boxsize& Box, PseudoAtomDefinitions& PseudoAtom, std::string& Frameworkfile, bool UseChargeFromCIF, double3 NumberUnitCells, Components& SystemComponents)
{
  std::vector<std::string> termsScannedLined{};
  termsScannedLined = split(Frameworkfile, '.');
  std::string frameworkName = termsScannedLined[0];
  std::string CIFFile = frameworkName + ".cif";
  std::ifstream simfile(CIFFile);
  std::filesystem::path pathfile = std::filesystem::path(CIFFile);
  if (!std::filesystem::exists(pathfile))
  {
    throw std::runtime_error("CIF file [ " + CIFFile + "] not found\n");
  }
  std::string str;

  //Temp Vector For Counting Number of Pseudo Atoms in the framework//
  int2 tempint2 = {0, 0};
  std::vector<std::vector<int2>>TEMPINTTWO;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
  {
    std::vector<int2>INTTWO(PseudoAtom.Name.size(), tempint2);
    TEMPINTTWO.push_back(INTTWO);
  }

  size_t NMol_Framework = 0;

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
  //size_t AtomID = 0;
  int label_location  = Label_x_y_z_Charge_Order[0];
  int x_location      = Label_x_y_z_Charge_Order[1];
  int y_location      = Label_x_y_z_Charge_Order[2];
  int z_location      = Label_x_y_z_Charge_Order[3];
  int Charge_location = Label_x_y_z_Charge_Order[4];
  if(!UseChargeFromCIF) Charge_location = -1; //If we want to use Charge from the pseudo_atoms.def, sed charge_location to -1//
  printf("label location: %d, xyz location: %d %d %d, charge: %d\n", label_location, x_location, y_location, z_location, Charge_location);
  std::vector<std::vector<double3>>super_pos;
  std::vector<std::vector<double>>super_scale;
  std::vector<std::vector<double>>super_charge;
  std::vector<std::vector<double>>super_scaleCoul;
  std::vector<std::vector<size_t>>super_Type;
  std::vector<std::vector<size_t>>super_MolID;

  //Zhao's note: temporary values for storing unit cell//
  std::vector<std::vector<double3>>unit_fpos;
  std::vector<std::vector<double>>unit_scale;
  std::vector<std::vector<double>>unit_charge;
  std::vector<std::vector<double>>unit_scaleCoul;
  std::vector<std::vector<size_t>>unit_Type;
  std::vector<std::vector<size_t>>unit_MolID;
  std::vector<std::vector<size_t>>unit_AtomIndex; //For checking the order of the atoms, used for framework components//
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
  {
    super_pos.emplace_back();
    super_scale.emplace_back();
    super_charge.emplace_back();
    super_scaleCoul.emplace_back();
    super_Type.emplace_back();
    super_MolID.emplace_back();

    unit_fpos.emplace_back();
    unit_scale.emplace_back();
    unit_charge.emplace_back();
    unit_scaleCoul.emplace_back();
    unit_Type.emplace_back();
    unit_MolID.emplace_back();
    unit_AtomIndex.emplace_back();
    //printf("super_pos size: %zu\n", super_pos.size());
  }
  size_t i = 0;
  ////////////////////////////////////////////////////////////////
  //Zhao's note: For making supercells, the code can only do P1!//
  ////////////////////////////////////////////////////////////////
  double3 Shift;
  Shift.x = (double)1/NumberUnitCells.x; Shift.y = (double)1/NumberUnitCells.y; Shift.z = (double)1/NumberUnitCells.z;
  size_t AtomCountPerUnitcell = 0;
  while (std::getline(simfile, str))
  {
    if(i <= atom_site_occurance.y){i++; continue;}
    //AtomID = i - atom_site_occurance.y - 1;
    Split_Tab_Space(termsScannedLined, str);
    //Check when the elements in the line is less than 4, stop when it is less than 4 (it means the atom_site region is over//
    if(termsScannedLined.size() < 4) 
    {
      //printf("Done reading Atoms in CIF file, there are %d Atoms\n", AtomID);
      break;
    }
    //printf("i: %zu, line: %s, splitted: %s\n", i, str.c_str(), termsScannedLined[0].c_str());
    double3 fpos;
    fpos.x = std::stod(termsScannedLined[x_location]); //fx*=Shift.x; //Divide by the number of UnitCells
    fpos.y = std::stod(termsScannedLined[y_location]); //fy*=Shift.y; 
    fpos.z = std::stod(termsScannedLined[z_location]); //fz*=Shift.z;
    double Charge = 0.0;
    if(Charge_location >= 0) Charge = std::stod(termsScannedLined[Charge_location]);

    //DETERMINE WHAT Framework COMPONENT THIS ATOM BELONGS TO//
    size_t ATOM_COMP = 0;
    size_t MoleculeID = 0;
    DetermineFrameworkComponent(SystemComponents, AtomCountPerUnitcell, ATOM_COMP, MoleculeID);

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
        TEMPINTTWO[ATOM_COMP][j].x =j;
        TEMPINTTWO[ATOM_COMP][j].y ++;
        break;
      }
    }
    printf("Atom Count %zu, Component %zu, AtomType %zu (%s), MoleculeID %zu\n", AtomCountPerUnitcell, ATOM_COMP, AtomTypeInt, AtomName.c_str(), MoleculeID);
    if(!AtomTypeFOUND)throw std::runtime_error("Error: Atom Label [" + AtomName + "] not defined!");
    unit_fpos[ATOM_COMP].push_back(fpos);
    unit_scale[ATOM_COMP].push_back(1.0); //For framework, use 1.0
    unit_charge[ATOM_COMP].push_back(Charge);
    unit_scaleCoul[ATOM_COMP].push_back(1.0);//For framework, use 1.0
    unit_Type[ATOM_COMP].push_back(AtomTypeInt);
    unit_MolID[ATOM_COMP].push_back(MoleculeID);
    unit_AtomIndex[ATOM_COMP].push_back(AtomCountPerUnitcell);
    AtomCountPerUnitcell ++;
    i++;
  }
  //Zhao's note: Need to sort the atom positions for the framework component to match the order in Framework_Component definition files, see Hilal's example//
  CheckFrameworkComponentAtomOrder(SystemComponents, unit_fpos, unit_scale, unit_charge, unit_scaleCoul, unit_Type, unit_MolID, unit_AtomIndex);
  
  //Zhao's note: consider separating the duplication of supercell from reading unit cell values//
  //It may cause strange bugs if the MolID for atoms are not continuous//
  for(size_t comp = 0; comp < SystemComponents.NComponents.y; comp++)
    for(size_t ix = 0; ix < NumberUnitCells.x; ix++)
      for(size_t jy = 0; jy < NumberUnitCells.y; jy++)
        for(size_t kz = 0; kz < NumberUnitCells.z; kz++)
        {
          size_t UnitCellID = (ix * NumberUnitCells.y + jy) * NumberUnitCells.z + kz;
          double3 NCell = {(double) ix, (double) jy, (double) kz};
          for(size_t Atom = 0; Atom < unit_fpos[comp].size(); Atom++)
          {
            double3 fpos = unit_fpos[comp][Atom];
            //Get supercell fx, fy, fz, and get corresponding xyz//
            double3 super_fpos = (fpos + NCell) * Shift;
            // Get real xyz from fractional xyz //
            double3 pos;
            pos.x = super_fpos.x*Box.Cell[0]+super_fpos.y*Box.Cell[3]+super_fpos.z*Box.Cell[6];
            pos.y = super_fpos.x*Box.Cell[1]+super_fpos.y*Box.Cell[4]+super_fpos.z*Box.Cell[7];
            pos.z = super_fpos.x*Box.Cell[2]+super_fpos.y*Box.Cell[5]+super_fpos.z*Box.Cell[8];
            super_pos[comp].push_back(pos);
            double scale = unit_scale[comp][Atom];
            double charge= unit_charge[comp][Atom];
            double scaleCoul= unit_scaleCoul[comp][Atom];
            size_t Type = unit_Type[comp][Atom];
            
            super_scale[comp].push_back(scale);
            super_charge[comp].push_back(charge);
            super_scaleCoul[comp].push_back(scaleCoul);
            super_Type[comp].push_back(Type);
            size_t ActualMolID = unit_MolID[comp][Atom];
            //Only duplicate MolID if the component is separated from the CIF file//
            if(SystemComponents.FrameworkComponentDef[comp].SeparatedComponent)
            {
              ActualMolID = SystemComponents.FrameworkComponentDef[comp].Number_of_Molecules_for_Framework_component * UnitCellID + ActualMolID;
            }
            super_MolID[comp].push_back(ActualMolID);
          }
        }
  printf("Finished Reading Atoms\n");
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
  {
    SystemComponents.HostSystem[i].pos       = (double3*) malloc(super_pos[i].size() * sizeof(double3));
    SystemComponents.HostSystem[i].scale     = (double*)  malloc(super_pos[i].size() * sizeof(double));
    SystemComponents.HostSystem[i].charge    = (double*)  malloc(super_pos[i].size() * sizeof(double));
    SystemComponents.HostSystem[i].scaleCoul = (double*)  malloc(super_pos[i].size() * sizeof(double));
    SystemComponents.HostSystem[i].Type      = (size_t*)  malloc(super_pos[i].size() * sizeof(size_t));
    SystemComponents.HostSystem[i].MolID     = (size_t*)  malloc(super_pos[i].size() * sizeof(size_t));
    
    //std::vector<size_t> MolID_i = super_MolID[i];
    //size_t NMol = std::max(MolID_i);
    std::vector<size_t>::iterator maxValueIterator = std::max_element(super_MolID[i].begin(), super_MolID[i].end());
    size_t NMol = *maxValueIterator; NMol ++;
    //size_t NMol = (*std::max_element(begin(super_MolID), end(super_MolID), [](size_t& a, size_t& b){ return a[i] < b[i]; }))[i];
    size_t NMol_In_Def = SystemComponents.FrameworkComponentDef[i].Number_of_Molecules_for_Framework_component;
    if(SystemComponents.FrameworkComponentDef[i].SeparatedComponent)
      NMol_In_Def *= NumberUnitCells.x * NumberUnitCells.y * NumberUnitCells.z;
    printf("NMol = %zu, pos_size: %zu, NMol in FrameworkDef: %zu\n", NMol, super_pos[i].size(), NMol_In_Def);

    if(NMol != NMol_In_Def)
    {
      throw std::runtime_error("In CheckFrameworkCIF function, NMol and value in FrameworkComponentDef don't match!!!!\n");
    }
    SystemComponents.HostSystem[i].size          = super_pos[i].size();
    SystemComponents.HostSystem[i].Molsize       = super_pos[i].size() / NMol;
    SystemComponents.HostSystem[i].Allocate_size = super_pos[i].size();

    SystemComponents.Moleculesize.push_back(SystemComponents.HostSystem[i].Molsize);
    SystemComponents.Allocate_size.push_back(SystemComponents.HostSystem[i].size);
    SystemComponents.NumberOfMolecule_for_Component.push_back(NMol);

    NMol_Framework += NMol;

    printf("Framework Comp [%zu], size: %zu, Molsize: %zu, Allocate_size: %zu\n", i, super_pos[i].size(), SystemComponents.HostSystem[i].Molsize, SystemComponents.HostSystem[i].Allocate_size);
  }
  for(size_t i = 0; i < super_pos.size(); i++)
  {
    bool FrameworkHasPartialCharge = false; double ChargeSum = 0.0;
    for(size_t j = 0; j < super_pos[i].size(); j++)
    {
      SystemComponents.HostSystem[i].pos[j] = super_pos[i][j];
      SystemComponents.HostSystem[i].scale[j] = super_scale[i][j];
      SystemComponents.HostSystem[i].charge[j] = super_charge[i][j];
      SystemComponents.HostSystem[i].scaleCoul[j] = super_scaleCoul[i][j];
      SystemComponents.HostSystem[i].Type[j] = super_Type[i][j];
      SystemComponents.HostSystem[i].MolID[j] = super_MolID[i][j];
      ChargeSum += std::abs(SystemComponents.HostSystem[i].charge[j]);
    }
    if(ChargeSum > 1e-6) FrameworkHasPartialCharge = true;
    SystemComponents.hasPartialCharge.push_back(FrameworkHasPartialCharge);
  }

  printf("------------------CIF FILE SUMMARY------------------\n");
  printf("CIF FILE IS: %s\n", CIFFile.c_str());
  printf("Number of Unit Cells: %.2f %.2f %.2f\n", NumberUnitCells.x, NumberUnitCells.y, NumberUnitCells.z);
  printf("Box size: \n%.5f %.5f %.5f\n%.5f %.5f %.5f\n%.5f %.5f %.5f\n", Box.Cell[0], Box.Cell[1], Box.Cell[2], Box.Cell[3], Box.Cell[4], Box.Cell[5], Box.Cell[6], Box.Cell[7], Box.Cell[8]);

  //Record Number of PseudoAtoms for the Framework//
  for(size_t i = 0; i < TEMPINTTWO.size(); i++)
  {
    std::vector<int2>NumberOfPseudoAtoms;
    for(size_t j = 0; j < TEMPINTTWO[i].size(); j++)
    {
      if(TEMPINTTWO[i][j].y == 0) continue;
      TEMPINTTWO[i][j].y *= NumberUnitCells.x * NumberUnitCells.y * NumberUnitCells.z;
      NumberOfPseudoAtoms.push_back(TEMPINTTWO[i][j]);
      printf("Framework Pseudo Atom[%zu], Name: %s, #: %zu\n", TEMPINTTWO[i][j].x, PseudoAtom.Name[j].c_str(), TEMPINTTWO[i][j].y);
    }
    printf("NumberOfPseudoAtoms size: %zu\n", NumberOfPseudoAtoms.size());
    SystemComponents.NumberOfPseudoAtomsForSpecies.push_back(NumberOfPseudoAtoms);
  }
  for(size_t i = 0; i < SystemComponents.NumberOfPseudoAtomsForSpecies.size(); i++)
    for(size_t j = 0; j < SystemComponents.NumberOfPseudoAtomsForSpecies[i].size(); j++)
    {
      printf("Framework Component [%zu], Pseudo Atom [%zu], Name: %s, #: %zu\n", i, SystemComponents.NumberOfPseudoAtomsForSpecies[i][j].x, PseudoAtom.Name[SystemComponents.NumberOfPseudoAtomsForSpecies[i][j].x].c_str(), SystemComponents.NumberOfPseudoAtomsForSpecies[i][j].y);
    }
  //Add PseudoAtoms from the Framework to the total PseudoAtoms array//
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
    SystemComponents.UpdatePseudoAtoms(INSERTION, i);

  SystemComponents.TotalNumberOfMolecules = NMol_Framework; //If there is a framework, framework is counted as a molecule//
  SystemComponents.NumberOfFrameworks = NMol_Framework;
  printf("----------------------------------------------------\n");
  //throw std::runtime_error("BREAK!!!!\n");
}

void ReadFrameworkSpeciesDefinitions(Components& SystemComponents)
{
  SystemComponents.FrameworkComponentDef.resize(SystemComponents.NComponents.y);
  if(SystemComponents.NComponents.y <= 1)
  {
    SystemComponents.FrameworkComponentDef[0].Number_of_Molecules_for_Framework_component = 1;
    return; //Then no need to separate
  }
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  size_t tempval = 0;
  for(size_t i = 0; i < SystemComponents.NComponents.y; i++)
  { //Zhao's note:: for framework component value = 0, the default framework component, we may want to use a file_check to see if it is separatable//
    if(i == 0) 
    {
      SystemComponents.FrameworkComponentDef[0].SeparatedComponent = false;
      SystemComponents.FrameworkComponentDef[0].Number_of_Molecules_for_Framework_component = 1;
      continue; //No need for the default framework component
    }
    std::string FileName = "Framework_Component_" + std::to_string(i) + ".def";
    printf("Reading %s FILE\n", FileName.c_str());
    std::filesystem::path pathfile = std::filesystem::path(FileName);
    if (!std::filesystem::exists(pathfile))
    {
      throw std::runtime_error("Framework Component NOT FOUND!!!!\n");
    }
    SystemComponents.FrameworkComponentDef[i].SeparatedComponent = true;
    std::ifstream file(FileName);
    while (std::getline(file, str))
    {
      if (str.find("#", 0) != std::string::npos) continue;
      if (str.find("Framework_Component_Name", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        SystemComponents.MoleculeName.push_back(termsScannedLined[1]);
      }

      if (str.find("Number_of_Molecules_for_Framework_component", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        sscanf(termsScannedLined[1].c_str(), "%zu", &tempval);
        SystemComponents.FrameworkComponentDef[i].Number_of_Molecules_for_Framework_component = tempval;
        //SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule.resize(tempval);
      }
      if (str.find("Number_of_atoms_for_each_molecule", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        sscanf(termsScannedLined[1].c_str(), "%zu", &tempval);
        SystemComponents.FrameworkComponentDef[i].Number_of_atoms_for_each_molecule = tempval;
      }
      if (str.find("Atom_Indices_for_Molecule", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        printf("size of the indices: %zu\n", termsScannedLined.size() - 2);
        std::vector<size_t>Indices;
        for(size_t j = 2; j < termsScannedLined.size(); j++)
        {
          sscanf(termsScannedLined[j].c_str(), "%zu", &tempval);
          Indices.push_back(tempval);
        }
        SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule.push_back(Indices);
        if(Indices.size() != SystemComponents.FrameworkComponentDef[i].Number_of_atoms_for_each_molecule)
          throw std::runtime_error("Number of atoms for Framework Component != Number of Atom Indices!!!!\n");
      }
    }
    printf("================Framework Component [%zu] Summary================\n", i);
    printf("Name: %s\n", SystemComponents.MoleculeName[i].c_str());
    printf("Number of Molecules: %zu\n", SystemComponents.FrameworkComponentDef[i].Number_of_Molecules_for_Framework_component);
    printf("Number of Atoms per Molecule: %zu\n", SystemComponents.FrameworkComponentDef[i].Number_of_atoms_for_each_molecule);
    for(size_t j = 0; j < SystemComponents.FrameworkComponentDef[i].Number_of_Molecules_for_Framework_component; j++)
      for(size_t k = 0; k < SystemComponents.FrameworkComponentDef[i].Number_of_atoms_for_each_molecule; k++)
        printf("Molecule [%zu], Index: %zu\n", j, SystemComponents.FrameworkComponentDef[i].Atom_Indices_for_Molecule[j][k]);
  }
}

void ReadFramework(Boxsize& Box, PseudoAtomDefinitions& PseudoAtom, size_t FrameworkIndex, Components& SystemComponents)
{
  printf("------------------------PARSING FRAMEWORK DATA------------------------\n");
  bool UseChargesFromCIFFile = true;  //Zhao's note: if not, use charge from pseudo atoms file, not implemented (if reading poscar, then self-defined charges probably need a separate file //
  std::vector<std::string> Names = PseudoAtom.Name;
  //size_t temp = 0;
  std::string scannedLine; std::string str;
  std::vector<std::string> termsScannedLined{};
  //size_t counter = 0;
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
      Split_Tab_Space(termsScannedLined, str);
      InputType=termsScannedLined[1];
    }
    //Zhao's note: When we are using Multiple frameworks, in simulation.input, list the frameworks one by one.//
    if (str.find("FrameworkName", 0) != std::string::npos) // get the molecule name
    {
      Split_Tab_Space(termsScannedLined, str);
      if((1+FrameworkIndex) >= termsScannedLined.size()) throw std::runtime_error("Not Enough Framework listed in input file...\n");
      Frameworkname = termsScannedLined[1 + FrameworkIndex];
      printf("Reading Framework %zu, FrameworkName: %s\n", FrameworkIndex, Frameworkname.c_str());
    }
    if (str.find("UnitCells " + std::to_string(FrameworkIndex), 0) != std::string::npos) // get the molecule name
    {
      Split_Tab_Space(termsScannedLined, str);
      NumberUnitCells.x = std::stod(termsScannedLined[2]);
      NumberUnitCells.y = std::stod(termsScannedLined[3]);
      NumberUnitCells.z = std::stod(termsScannedLined[4]);
      printf("Reading Framework %zu, UnitCells: %.2f %.2f %.2f\n", FrameworkIndex, NumberUnitCells.x, NumberUnitCells.y, NumberUnitCells.z);
      FrameworkFound = true;
      SystemComponents.NumberofUnitCells = {(int) NumberUnitCells.x, (int) NumberUnitCells.y, (int) NumberUnitCells.z};
    }
  }
  //If not cif or poscar, break the program!//
  if(InputType != "cif" && InputType != "poscar") throw std::runtime_error("Cannot identify framework input type [" + InputType + "]. It can only be cif or poscar!");
  if(!FrameworkFound) throw std::runtime_error("Cannot find the framework with matching index!");
  std::string FrameworkFile = Frameworkname + "." + InputType;
  std::filesystem::path pathfile = std::filesystem::path(FrameworkFile);
  if (!std::filesystem::exists(pathfile)) throw std::runtime_error("Framework File ["+ FrameworkFile + "] not found!\n");
  /////////////////////////////////////////////////////////////////////////////////////
  //Zhao's note:                                                                     //
  //If reading poscar, then you cannot have user-defined charges for every atom      //
  //the charges used for poscar are defined in pseudo_atoms.def                      //
  //To use user-defined charge for every atom, use cif format                        //
  /////////////////////////////////////////////////////////////////////////////////////
  if(InputType == "cif")
  {
    SystemComponents.MoleculeName.push_back(FrameworkFile);

    ReadFrameworkSpeciesDefinitions(SystemComponents);
    CheckFrameworkCIF(Box, PseudoAtom, FrameworkFile, UseChargesFromCIFFile, NumberUnitCells, SystemComponents);
    printf("Reading CIF File\n");
  }
  else
  {
    throw std::runtime_error("Only supports reading CIF files\n");
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
  printf("----------------------END OF PARSING FRAMEWORK DATA----------------------\n");
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
  std::vector<double3> Apos(Allocate_space);
  std::vector<double>  Ascale(Allocate_space);
  std::vector<double>  Acharge(Allocate_space);
  std::vector<double>  AscaleCoul(Allocate_space);
  std::vector<size_t>  AType(Allocate_space);
  std::vector<size_t>  AMolID(Allocate_space);
 
  double chargesum = 0.0; //a sum of charge for the atoms in the molecule, for checking charge neutrality
 
  bool temprigid = false;

  size_t PseudoAtomSize = PseudoAtom.Name.size();
  //For Tail corrections//
  int2   TempINTTWO = {0, 0};
  std::vector<int2> ANumberOfPseudoAtomsForSpecies(PseudoAtomSize, TempINTTWO);

  // skip first line
  while (std::getline(file, str))
  {
    if(counter == 1) //read Tc
    {
      Split_Tab_Space(termsScannedLined, str);
      SystemComponents.Tc.push_back(std::stod(termsScannedLined[0]));
    }
    else if(counter == 2) //read Pc
    {
      Split_Tab_Space(termsScannedLined, str);
      SystemComponents.Pc.push_back(std::stod(termsScannedLined[0]));
    }
    else if(counter == 3) //read Accentric Factor
    {
      Split_Tab_Space(termsScannedLined, str);
      SystemComponents.Accentric.push_back(std::stod(termsScannedLined[0]));
    }
    else if(counter == 5) //read molecule size
    {
      temp_molsize = 0;
      Split_Tab_Space(termsScannedLined, str);
      sscanf(termsScannedLined[0].c_str(), "%zu", &temp_molsize);
      if(temp_molsize >= Allocate_space) throw std::runtime_error("Molecule size is greater than allocated size. Break\n");
      SystemComponents.Moleculesize.push_back(temp_molsize);
      Mol.Molsize = temp_molsize; //Set Molsize for the adsorbate molecule here//
    }
    else if(counter == 9) //read if the molecule is rigid
    {
      Split_Tab_Space(termsScannedLined, str);
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
      Split_Tab_Space(termsScannedLined, str);
      if(termsScannedLined.size() == 5) //for example: 0 CH4 0.0 0.0 0.0, position provided here.
      {
        Apos[atomcount] = {std::stod(termsScannedLined[2]), std::stod(termsScannedLined[3]), std::stod(termsScannedLined[4])};
      }
      else if(termsScannedLined.size() == 2 && temp_molsize == 1) //like methane, one can do: 0 CH4, with no positions
      {
        Apos[atomcount] = {0.0, 0.0, 0.0};
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

  bool MolHasPartialCharge = false; double ChargeSum = 0.0;
  for(size_t i = 0; i < Acharge.size(); i++) ChargeSum += std::abs(Acharge[i]);
  if(ChargeSum > 1e-6) MolHasPartialCharge = true;
  SystemComponents.hasPartialCharge.push_back(MolHasPartialCharge);

  SystemComponents.rigid.push_back(temprigid);
  if(chargesum > 1e-50) throw std::runtime_error("Molecule not neutral, bad\n");

  Mol.pos       = convert1DVectortoArray(Apos);
  Mol.scale     = convert1DVectortoArray(Ascale);
  Mol.charge    = convert1DVectortoArray(Acharge);
  Mol.scaleCoul = convert1DVectortoArray(AscaleCoul);

  Mol.Type      = convert1DVectortoArray(AType);
  Mol.MolID     = convert1DVectortoArray(AMolID);

  for(size_t i = 0; i < Mol.Molsize; i++)
    printf("Atom [%zu]: Type [%zu], Name: %s, %.5f %.5f %.5f\n", i, Mol.Type[i], PseudoAtom.Name[Mol.Type[i]].c_str(), Mol.pos[i].x, Mol.pos[i].y, Mol.pos[i].z);

  //Remove Elements from ANumberOfPseudoAtomsForSpecies if the ANumberOfPseudoAtomsForSpecies.y = 0
  std::vector<int2>TEMPINTTWO;
  for(size_t i = 0; i < ANumberOfPseudoAtomsForSpecies.size(); i++)
  {
    if(ANumberOfPseudoAtomsForSpecies[i].y == 0) continue;
    TEMPINTTWO.push_back(ANumberOfPseudoAtomsForSpecies[i]);
    printf("Adsorbate Type[%zu], Name: %s, #: %zu\n", ANumberOfPseudoAtomsForSpecies[i].x, PseudoAtom.Name[i].c_str(), ANumberOfPseudoAtomsForSpecies[i].y);
  }
  SystemComponents.NumberOfPseudoAtomsForSpecies.push_back(TEMPINTTWO);
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
  printf("%s starts at line number %zu\n", start_string.c_str(), start_counter); 
  file.clear();
  file.seekg(0); 
  std::string MolName;
  
  while (std::getline(file, str))
  {
    if(str.find(terminate_string, 0) != std::string::npos){break;}
    if(counter >= start_counter) //start reading after touching the starting line number
    {
      if (str.find(start_string, 0) != std::string::npos) // get the molecule name
      {
        Split_Tab_Space(termsScannedLined, str);
        MolName = termsScannedLined[3];
        SystemComponents.MoleculeName.push_back(MolName);
        std::cout << "-------------- READING " << start_string << " (" << MolName << ")" << " --------------\n";
        MoleculeDefinitionParser(Mol, SystemComponents, termsScannedLined[3], PseudoAtom, Allocate_space);
      }
      if (str.find("IdealGasRosenbluthWeight", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        idealrosen = std::stod(termsScannedLined[1]); printf("Ideal Chain Rosenbluth Weight: %.5f\n", idealrosen);
      }
      if (str.find("TranslationProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.TranslationProb=std::stod(termsScannedLined[1]);
        //printf("TranslationProb: %.5f, TotalProb: %.5f\n", TranslationProb, TotalProb);
      }
      if (str.find("RotationProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.RotationProb=std::stod(termsScannedLined[1]);
        //printf("RotationProb: %.5f, TotalProb: %.5f\n", RotationProb, TotalProb);
      }
      if (str.find("WidomProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.WidomProb=std::stod(termsScannedLined[1]);
        //printf("WidomProb: %.5f, TotalProb: %.5f\n", WidomProb, TotalProb);
      }
      if (str.find("ReinsertionProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.ReinsertionProb=std::stod(termsScannedLined[1]);
        //printf("ReinsertionProb: %.5f, TotalProb: %.5f\n", ReinsertionProb, TotalProb);
      }
      if (str.find("IdentityChangeProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.IdentitySwapProb=std::stod(termsScannedLined[1]);
        //printf("IdentityChangeProb: %.5f, TotalProb: %.5f\n", IdentitySwapProb, TotalProb);
      }
      if (str.find("SwapProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.SwapProb=std::stod(termsScannedLined[1]);
        //printf("SwapProb: %.5f, TotalProb: %.5f\n", SwapProb, TotalProb);
      }
      if (str.find("CBCFProbability", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        MoveStats.CBCFProb=std::stod(termsScannedLined[1]);
        temp_hasfracmol=true;
        //printf("CBCFProb: %.5f, TotalProb: %.5f\n", CBCFProb, TotalProb);
      }
      //Zhao's note: If using CBCF Move, choose the lambda type//
      if (MoveStats.CBCFProb > 0.0)
      {
        if (str.find("LambdaType", 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
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
        Split_Tab_Space(termsScannedLined, str);
        fugacoeff = std::stod(termsScannedLined[1]);
      }
      if (str.find("MolFraction", 0) != std::string::npos)
      {
         Split_Tab_Space(termsScannedLined, str);
         Molfrac = std::stod(termsScannedLined[1]);
      }
      if (str.find("RunTMMC", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        if(caseInSensStringCompare(termsScannedLined[1], "yes"))
        {
          temp_tmmc.DoTMMC = true;
          printf("TMMC: Running TMMC simulation\n");
        }
      }
      if (str.find("TURN_OFF_CBMC_SWAP", 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        if(caseInSensStringCompare(termsScannedLined[1], "yes"))
        {
          SystemComponents.SingleSwap = true;
          printf("SWAP WITH NO CBMC!\n");
        }
      }
      if(temp_tmmc.DoTMMC)
      {
        if (str.find("TMMCMin", 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
          sscanf(termsScannedLined[1].c_str(), "%zu", &temp_tmmc.MinMacrostate);
        }
        else if (str.find("TMMCMax", 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
          sscanf(termsScannedLined[1].c_str(), "%zu", &temp_tmmc.MaxMacrostate);
        }
        if (str.find("UseBiasOnMacrostate", 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
          if(caseInSensStringCompare(termsScannedLined[1], "yes"))
          {
            temp_tmmc.DoUseBias = true;
            printf("TMMC: Biasing Insertion/Deletions\n");
          }
        }
      }
      //This should be a general DNN setup//
      //Consider the positions of water TIP4P, you don't need to feed the position of the fictional charge to the model//
      if(SystemComponents.UseDNNforHostGuest)
      { 
        cudaMallocManaged(&SystemComponents.ConsiderThisAdsorbateAtom, sizeof(bool) * SystemComponents.Moleculesize[SystemComponents.NComponents.y]);

        //Zhao's note: Read the types of atoms that needs to be considered for the DNN Model//
        if (str.find("DNNPseudoAtoms", 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
          for(size_t a = 1; a < termsScannedLined.size(); a++)
          {
            std::string AtomName = termsScannedLined[a];
            size_t Type = get_type_from_name(AtomName, PseudoAtom.Name);
            //Find location of the pseudo atoms
            for(size_t b = 0; b < Mol.Molsize; b++)
            {
              if(Type == Mol.Type[b])
              {
                SystemComponents.ConsiderThisAdsorbateAtom[b] = true;
                printf("AtomName: %s, Type: %zu, b: %zu, Consider? %s\n", AtomName.c_str(), Type, b, SystemComponents.ConsiderThisAdsorbateAtom[b] ? "true" : "false");
              }
            }
          }
        }
      }
      if (str.find("CreateNumberOfMolecules", 0) != std::string::npos) // Number of Molecules to create
      {
        Split_Tab_Space(termsScannedLined, str); 
        sscanf(termsScannedLined[1].c_str(), "%zu", &CreateMolecule);
      }
    }
    counter++;
  }
  MoveStats.NormalizeProbabilities();
  MoveStats.PrintProbabilities();

  //Zhao's note: if monatomic molecule has rotation prob, break the program!//
  size_t currentCompsize = SystemComponents.Moleculesize.size();
  if(SystemComponents.Moleculesize[currentCompsize-1] == 1 && (MoveStats.RotationProb - MoveStats.TranslationProb) > 1e-10)
  {
    throw std::runtime_error("Molecule [" + SystemComponents.MoleculeName[currentCompsize-1] + "] is MONATOMIC, CANNOT DO ROTATION!\n");
  }

  SystemComponents.NumberOfMolecule_for_Component.push_back(0); // Zhao's note: Molecules are created later in main.cpp //
  SystemComponents.Allocate_size.push_back(Allocate_space);
  if(idealrosen < 1e-150) throw std::runtime_error("Ideal-Rosenbluth weight not assigned (or not valid), bad. If rigid, assign 1.");
  SystemComponents.IdealRosenbluthWeight.push_back(idealrosen);
  //Zhao's note: for fugacity coefficient, if not assigned (0.0), do Peng-Robinson
  if(fugacoeff < 1e-150)
  {
    throw std::runtime_error("Peng-rob EOS not implemented yet...");
  }
  
   
  SystemComponents.FugacityCoeff.push_back(fugacoeff);
  //Zhao's note: for now, Molfraction = 1.0
  SystemComponents.MolFraction.push_back(Molfrac);
  SystemComponents.hasfractionalMolecule.push_back(temp_hasfracmol);
  
  LAMBDA lambda;
  //Zhao's note: for bin number = 10, there are 11 bins, the first is when lambda = 0.0, last for lambda = 1.0//
  if(temp_hasfracmol) //Prepare lambda struct if using CBCF//
  {
    lambda.newBin    = 0;
    lambda.delta     = 1.0/static_cast<double>(lambda.binsize); 
    lambda.WangLandauScalingFactor = 0.0; lambda.FractionalMoleculeID = 0;
    lambda.lambdatype = LambdaType;
    lambda.Histogram.resize(lambda.binsize + 1); lambda.biasFactor.resize(lambda.binsize + 1);
    std::fill(lambda.Histogram.begin(),  lambda.Histogram.end(),  0.0);
    std::fill(lambda.biasFactor.begin(), lambda.biasFactor.end(), 0.0);
  }

  //Zhao's note: If we are using CBCFC + TMMC, turn off Normal Swap moves//
  if(temp_hasfracmol && temp_tmmc.DoTMMC)
  {
    if((MoveStats.SwapProb - MoveStats.CBCFProb) > 1e-10)
      throw std::runtime_error("CBCFC + TMMC: Cannot use normal (non-CFC) swap moves when you have CBCFC + TMMC!!!!");
  }
 
  if(temp_tmmc.DoTMMC) //Prepare tmmc struct if using TMMC//
  { 
    if(temp_tmmc.MaxMacrostate < temp_tmmc.MinMacrostate)
    {
      throw std::runtime_error("TMMC: Bad Min/Max Macrostates for TMMC, Min has to be SMALLER THAN OR EQUAL TO Max.");
    }
    temp_tmmc.nbinPerMacrostate = 1;
    if(temp_hasfracmol) temp_tmmc.nbinPerMacrostate = lambda.Histogram.size(); 
    size_t NBin = temp_tmmc.nbinPerMacrostate * (temp_tmmc.MaxMacrostate - temp_tmmc.MinMacrostate + 1);

    temp_tmmc.CMatrix.resize(NBin);
    temp_tmmc.WLBias.resize(NBin);
    temp_tmmc.TMBias.resize(NBin);
    std::fill(temp_tmmc.TMBias.begin(), temp_tmmc.TMBias.end(), 1.0); //Initialize the TMBias//
    temp_tmmc.ln_g.resize(NBin);
    temp_tmmc.lnpi.resize(NBin);
    temp_tmmc.forward_lnpi.resize(NBin);
    temp_tmmc.reverse_lnpi.resize(NBin);
    temp_tmmc.Histogram.resize(NBin);
    //Zhao's note: if we set the bounds for min/max macrostate, the number of createMolecule should fall in the range//
    if(temp_tmmc.RejectOutofBound && !SystemComponents.ReadRestart)
      if(CreateMolecule < temp_tmmc.MinMacrostate || CreateMolecule > temp_tmmc.MaxMacrostate)
        throw std::runtime_error("TMMC: Number of created molecule fall out of the TMMC Macrostate range!");
  }

  SystemComponents.Lambda.push_back(lambda);
  SystemComponents.Tmmc.push_back(temp_tmmc);
  SystemComponents.NumberOfCreateMolecules.push_back(CreateMolecule);
  //Finally, check if all values in SystemComponents are set properly//
  Check_Component_size(SystemComponents);
  //Initialize single values for Mol//
  Mol.size = 0;
  std::cout << "-------------- END OF READING " << start_string << " (" << MolName << ")" << " --------------\n";
}

void RestartFileParser(Simulations& Sims, Components& SystemComponents)
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
  
  //Zhao's note: MolID in our Atoms struct are relative IDs, for one component, they start with zero.
  //Yet, in Restart file, it start with an arbitrary number (equal to number of previous component molecules)
  //Need to substract it off//
  size_t PreviousCompNMol = 0;
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
  {
    size_t start = 0; size_t end = 0;
    while (std::getline(file, str))
    {
      //find range of the part we need to read// 
      if (str.find("Components " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos) 
        start = counter;
      if (str.find("Maximum-rotation-change component " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
      {  end = counter; break; }
    }
    file.clear();
    file.seekg(0);
    //Start reading//
    while (std::getline(file, str))
    {
      if (str.find("Fractional-molecule-id component " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        if(std::stoi(termsScannedLined[3]) == -1) SystemComponents.hasfractionalMolecule[i] = false;
        if(SystemComponents.hasfractionalMolecule[i])
        {
          sscanf(termsScannedLined[3].c_str(), "%zu", &SystemComponents.Lambda[i].FractionalMoleculeID);
          printf("Fractional Molecule ID: %zu\n", SystemComponents.Lambda[i].FractionalMoleculeID);
        }
      }
      if(SystemComponents.hasfractionalMolecule[i])
      {
        if (str.find("Lambda-factors component " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
          SystemComponents.Lambda[i].WangLandauScalingFactor = std::stod(termsScannedLined[3]);
          printf("WL Factor: %.5f\n", SystemComponents.Lambda[i].WangLandauScalingFactor);
        }
        if (str.find("Number-of-biasing-factors component " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
        {
          Split_Tab_Space(termsScannedLined, str);
          sscanf(termsScannedLined[3].c_str(), "%zu", &SystemComponents.Lambda[i].binsize);
          printf("binsize: %zu\n", SystemComponents.Lambda[i].binsize);
          if(SystemComponents.Lambda[i].binsize != SystemComponents.Lambda[i].Histogram.size()) throw std::runtime_error("CFC Bin size don't match!");
        }
        if (str.find("Biasing-factors component " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
        {
          for(size_t j = 0; j < SystemComponents.Lambda[i].binsize; j++)
          {
            Split_Tab_Space(termsScannedLined, str);
            SystemComponents.Lambda[i].biasFactor[j] = std::stod(termsScannedLined[3 + j]); 
            printf("Biasing Factor %zu: %.5f\n", j, SystemComponents.Lambda[i].biasFactor[j]);
          }
        }
      }
      //Read translation rotation maxes//
      if (str.find("Maximum-translation-change component " + std::to_string(i - SystemComponents.NumberOfFrameworks), 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        SystemComponents.MaxTranslation[i - SystemComponents.NComponents.y].x = std::stod(termsScannedLined[3]);
        SystemComponents.MaxTranslation[i - SystemComponents.NComponents.y].y = std::stod(termsScannedLined[4]);
        SystemComponents.MaxTranslation[i - SystemComponents.NComponents.y].z = std::stod(termsScannedLined[5]);
      }
      if (str.find("Maximum-rotation-change component " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        SystemComponents.MaxRotation[i - SystemComponents.NComponents.y].x = std::stod(termsScannedLined[3]);
        SystemComponents.MaxRotation[i - SystemComponents.NComponents.y].y = std::stod(termsScannedLined[4]);
        SystemComponents.MaxRotation[i - SystemComponents.NComponents.y].z = std::stod(termsScannedLined[5]);
        break;
      }
    }
    file.clear(); 
    file.seekg(0);
    //Start reading atom positions and other information//
    start = 0; end = 0; counter = 0;
    while (std::getline(file, str))
    {
      if (str.find("Component: " + std::to_string(i - SystemComponents.NComponents.y), 0) != std::string::npos)
      {
        Split_Tab_Space(termsScannedLined, str);
        start = counter + 2;
        size_t temp; 
        sscanf(termsScannedLined[3].c_str(), "%zu", &temp);
        printf("Adsorbate Component %zu, #: %zu\n", i, temp);
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
    file.clear();
    file.seekg(0);
    if(SystemComponents.NumberOfMolecule_for_Component[i] == 0) continue;
    size_t interval = SystemComponents.NumberOfMolecule_for_Component[i]* SystemComponents.Moleculesize[i];
    double3 pos[SystemComponents.NumberOfMolecule_for_Component[i]       * SystemComponents.Moleculesize[i]];
    double  scale[SystemComponents.NumberOfMolecule_for_Component[i]     * SystemComponents.Moleculesize[i]];
    double  charge[SystemComponents.NumberOfMolecule_for_Component[i]    * SystemComponents.Moleculesize[i]];
    double  scaleCoul[SystemComponents.NumberOfMolecule_for_Component[i] * SystemComponents.Moleculesize[i]];
    size_t  Type[SystemComponents.NumberOfMolecule_for_Component[i]      * SystemComponents.Moleculesize[i]];
    size_t  MolID[SystemComponents.NumberOfMolecule_for_Component[i]     * SystemComponents.Moleculesize[i]];

    size_t atom=0;
    while (std::getline(file, str))
    {
      //Read positions, Type and MolID//
      //Position: 0, velocity: 1, force: 2, charge: 3, scaling: 4
      if((counter >= start) && (counter < start + interval))
      {
        atom=counter - start;
        if (!(str.find("Adsorbate-atom-position", 0) != std::string::npos)) throw std::runtime_error("Cannot find matching strings in the range for reading positions!");
        Split_Tab_Space(termsScannedLined, str);
        pos[atom] = {std::stod(termsScannedLined[3]), std::stod(termsScannedLined[4]), std::stod(termsScannedLined[5])};
        sscanf(termsScannedLined[1].c_str(), "%zu", &MolID[atom]);
        size_t atomid; sscanf(termsScannedLined[2].c_str(), "%zu", &atomid);
        Type[atom] = SystemComponents.HostSystem[i].Type[atomid]; //for every component, the types of atoms for the first molecule is always there, just copy it//
        //printf("Reading Positions, atom: %zu, xyz: %.5f %.5f %.5f, Type: %zu, MolID: %zu\n", atom, pos[atom].x, pos[atom].y, pos[atom].z, Type[atom], MolID[atom]);
        //Zhao's note: adjust the MolID from absolute to relative to component//
        MolID[atom] -= PreviousCompNMol;
      }
      //Read charge//
      if((counter >= start + interval * 3) && (counter < start + interval * 4))
      { 
        atom = counter - (start + interval * 3);
        Split_Tab_Space(termsScannedLined, str);
        charge[atom] = std::stod(termsScannedLined[3]);
        //printf("Reading charge, atom: %zu, charge: %.5f\n", atom, charge[atom]);
      }
      //Read scaling and scalingCoul//
      atom=0;
      if((counter >= start + interval * 4) && (counter < start + interval * 5))
      {
        atom = counter - (start + interval * 4);
        Split_Tab_Space(termsScannedLined, str);
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
          //double smallEpsilon = 1e-5; //FOR DEBUGGING NUMERICAL ISSUES IN THE FUTURE//
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
      SystemComponents.HostSystem[i].pos[j] = pos[j]; 
      SystemComponents.HostSystem[i].charge[j] = charge[j]; 
      SystemComponents.HostSystem[i].scale[j] = scale[j]; SystemComponents.HostSystem[i].scaleCoul[j] = scaleCoul[j]; 
      SystemComponents.HostSystem[i].Type[j] = Type[j]; SystemComponents.HostSystem[i].MolID[j] = MolID[j];
      //printf("Data for %zu: %.5f %.5f %.5f %.5f %.5f %.5f %zu %zu\n", j, Host_System[i].pos[j].x, Host_System[i].pos[j].y, Host_System[i].pos[j].z, Host_System[i].charge[j], Host_System[i].scale[j], Host_System[i].scaleCoul[j], Host_System[i].Type[j], Host_System[i].MolID[j]);
    }
    SystemComponents.HostSystem[i].size = interval;
    SystemComponents.HostSystem[i].Molsize = SystemComponents.Moleculesize[i];

    PreviousCompNMol += SystemComponents.NumberOfMolecule_for_Component[i];
  }
}


void ReadDNNModelSetup(Components& SystemComponents)
{
  std::vector<std::string> termsScannedLined{};
  std::string str;
  std::ifstream file("simulation.input");
  while (std::getline(file, str))
  {
    if (str.find("UseDNNforHostGuest", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "yes"))
      {
        SystemComponents.UseDNNforHostGuest = true;
        printf("Using DNN Model\n");
      }
      break;
    }
  }
  if(!SystemComponents.UseDNNforHostGuest) return;

  file.clear();
  file.seekg(0);

  bool foundMethod = false; bool DNNUnitFound = false;
  while (std::getline(file, str))
  {
    if (str.find("DNNMethod", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      printf("Found DNNMethod\n");
      if(caseInSensStringCompare(termsScannedLined[1], "Allegro"))
      {
        printf("Found Allegro\n");
        SystemComponents.UseAllegro = true; foundMethod = true;
      }
      else if(caseInSensStringCompare(termsScannedLined[1], "LCLin"))
      {
        printf("Found LCLin\n");
        SystemComponents.UseLCLin = true; foundMethod = true;
      }
      else {throw std::runtime_error("CANNOT IDENTIFY DNN MODEL in simulation.input file");}
    }
    if (str.find("DNNEnergyUnit", 0) != std::string::npos)
    {
      Split_Tab_Space(termsScannedLined, str);
      if(caseInSensStringCompare(termsScannedLined[1], "kJ_mol"))
      {
        SystemComponents.DNNEnergyConversion = 100.0; //from kJ_mol to 10J_mol
        DNNUnitFound = true; printf("DNN Model is using kJ/mol as the Energy Unit\n");
      }
      else if(caseInSensStringCompare(termsScannedLined[1], "eV"))
      {
        SystemComponents.DNNEnergyConversion = 9648.53074992579265; //from eV to 10J_mol
        DNNUnitFound = true; printf("DNN Model is using eV as the Energy Unit\n");
      }
      else
      {throw std::runtime_error("Unknown Energy Unit for DNN Model");}
    }
  }
  if(SystemComponents.UseDNNforHostGuest && !DNNUnitFound)
    throw std::runtime_error("You are using DNN models but there is no ENERGY UNIT specified!!!!");

  if(SystemComponents.UseAllegro && SystemComponents.UseLCLin)
    throw std::runtime_error("CANNOT USE Li-Chiang Lin's and Allegro at the same time!!!!");
  if(!foundMethod)
    throw std::runtime_error("CANNOT FIND the DNNMethod INPUT COMMAND in simulation.input file");
}

//###PATCH_LCLIN_READDATA###//
//###PATCH_ALLEGRO_READDATA###//
