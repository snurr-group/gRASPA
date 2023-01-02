#include <stdio.h>
#include <vector>
#include <string>
#include <complex>

void VDW_CPU(const double* FFarg, const double rr_dot, const double scaling, double* result);

void PBC_CPU(double* posvec, double* Cell, double* InverseCell, bool& Cubic);

/*
void PBC_CPU(double* posvec, double* Cell, double* InverseCell, bool& Cubic)
{
  if(Cubic)//cubic/cuboid
  {
    posvec[0] = posvec[0] - static_cast<int>(posvec[0] * InverseCell[0*3+0] + ((posvec[0] >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
    posvec[1] = posvec[1] - static_cast<int>(posvec[1] * InverseCell[1*3+1] + ((posvec[1] >= 0.0) ? 0.5 : -0.5)) * Cell[1*3+1];
    posvec[2] = posvec[2] - static_cast<int>(posvec[2] * InverseCell[2*3+2] + ((posvec[2] >= 0.0) ? 0.5 : -0.5)) * Cell[2*3+2];
  }
  else
  {
    double s[3] = {0.0, 0.0, 0.0};
    s[0]=InverseCell[0*3+0]*posvec[0]+InverseCell[1*3+0]*posvec[1]+InverseCell[2*3+0]*posvec[2];
    s[1]=InverseCell[0*3+1]*posvec[0]+InverseCell[1*3+1]*posvec[1]+InverseCell[2*3+1]*posvec[2];
    s[2]=InverseCell[0*3+2]*posvec[0]+InverseCell[1*3+2]*posvec[1]+InverseCell[2*3+2]*posvec[2];

    s[0] -= static_cast<int>(s[0] + ((s[0] >= 0.0) ? 0.5 : -0.5));
    s[1] -= static_cast<int>(s[1] + ((s[1] >= 0.0) ? 0.5 : -0.5));
    s[2] -= static_cast<int>(s[2] + ((s[2] >= 0.0) ? 0.5 : -0.5));
    // convert from abc to xyz
    posvec[0]=Cell[0*3+0]*s[0]+Cell[1*3+0]*s[1]+Cell[2*3+0]*s[2];
    posvec[1]=Cell[0*3+1]*s[0]+Cell[1*3+1]*s[1]+Cell[2*3+1]*s[2];
    posvec[2]=Cell[0*3+2]*s[0]+Cell[1*3+2]*s[1]+Cell[2*3+2]*s[2];
  }
}
*/


enum MoveTypes {TRANSLATION_ROTATION = 0, INSERTION, DELETION, REINSERTION, CBCF_LAMBDACHANGE, CBCF_INSERTION, CBCF_DELETION};

enum CBMC_Types {CBMC_INSERTION = 0, CBMC_DELETION, REINSERTION_INSERTION, REINSERTION_RETRACE};

enum SIMULATION_MODE {CREATE_MOLECULE = 0, INITIALIZATION, EQUILIBRATION, PRODUCTION};

struct Complex
{
  double real;
  double imag;
};

struct Units
{
  double MassUnit;
  double TimeUnit;
  double LengthUnit;
  double energy_to_kelvin;
  double BoltzmannConstant;
};

struct LAMBDA
{
  int    newBin;     //Bins range from (-binsize/2, binsize/2), since lambda is from -1 to 1.
  int    currentBin;
  double delta;  //size of the bin in the lambda histogram//
  double WangLandauScalingFactor;
  size_t binsize = {10};
  size_t FractionalMoleculeID;
  std::vector<double>Histogram;
  std::vector<double>biasFactor;
};

struct Move_Statistics
{
  size_t NumberOfBlocks; //Number of blocks for statistical averaging
  //Translation Move//
  double TranslationProb;
  double RotationProb;
  double WidomProb;
  double SwapProb;
  double ReinsertionProb;
  double CBCFProb;
  //Translation Move//
  int TranslationAccepted;
  int TranslationTotal;
  double TranslationAccRatio;
  //Rotation Move//
  int RotationAccepted;
  int RotationTotal;
  double RotationAccRatio;
  //Insertion Move//
  size_t InsertionTotal;
  size_t InsertionAccepted;
  //Deletion Move//
  size_t DeletionTotal;
  size_t DeletionAccepted;
  //Reinsertion Move//
  size_t ReinsertionTotal;
  size_t ReinsertionAccepted;
  //CBCFSwap Move//
  size_t CBCFTotal;
  size_t CBCFAccepted;
  size_t CBCFInsertionTotal;
  size_t CBCFInsertionAccepted;
  size_t CBCFLambdaTotal;
  size_t CBCFLambdaAccepted;
  size_t CBCFDeletionTotal;
  size_t CBCFDeletionAccepted;
};

struct Components
{
  size_t  Total_Components;                           // Number of Components in the system (including framework)
  size_t  TotalNumberOfMolecules;                     // Total Number of Molecules (including framework)
  size_t  NumberOfFrameworks;                         // Total Number of framework species, usually 1.
  double  Beta;                                       // Inverse Temperature
  double  tempdeltaVDWReal;
  double  tempdeltaEwald;
  double  deltaVDWReal;
  double  deltaEwald;
  std::vector<bool>   hasfractionalMolecule;          // Whether this component has fractional molecules
  std::vector<LAMBDA> Lambda;                         // Vector of Lambda struct
  std::vector<bool>   rigid;                          // Determine if the component is rigid.
  std::vector<double> ExclusionIntra;                 // For Ewald Summation, Intra Molecular exclusion
  std::vector<double> ExclusionAtom;                  // For Ewald Summation, Atom-self exclusion
  std::vector<std::string> MoleculeName;              // Name of the molecule
  std::vector<size_t> Moleculesize;                   // Number of Pseudo-Atoms in the molecule
  std::vector<size_t> NumberOfMolecule_for_Component; // Number of Molecules for each component (including fractional molecule)
  std::vector<size_t> Allocate_size;                  // Allocate size 
  std::vector<double> MolFraction;                    // Mol fraction of the component(excluding the framework)
  std::vector<double> IdealRosenbluthWeight;          // Ideal Rosenbluth weight
  std::vector<double> FugacityCoeff;                  // Fugacity Coefficient
  std::vector<Move_Statistics>Moves;                  // Move statistics: total, acceptance, etc.
  std::vector<double>Tc;                              // Critical Temperature of the component
  std::vector<double>Pc;                              // Critical Pressure of the component
  std::vector<double>Accentric;                       // Accentric Factor of the component
  std::vector<std::complex<double>> eik_xy;           
  std::vector<std::complex<double>> eik_x;  
  std::vector<std::complex<double>> eik_y;
  std::vector<std::complex<double>> eik_z;
  std::vector<std::complex<double>> storedEik;        // Stored Ewald Vector
  std::vector<std::complex<double>> totalEik; 
};

struct Atoms
{
  double* x;
  double* y;
  double* z;
  double* scale;
  double* charge;
  double* scaleCoul;
  size_t* Type;
  size_t* MolID;
  size_t  Molsize;
  size_t  size;
  size_t  Allocate_size;
};

struct Simulations //For multiple simulations//
{
  Atoms*  d_a;                  // Pointer For Atom Data in the Simulation Box //
  Atoms   Old;                  // Temporary data storage for Old Configuration //
  Atoms   New;                  // Temporary data storage for New Configuration //
  double* Blocksum;             // Block sums for partial reduction //
  bool*   flag;                 // flags for checking overlaps //
  bool*   device_flag;          // flags for overlaps on the device //
  size_t  start_position;       // Start position for reading data in d_a when proposing a trial position for moves //
  size_t  Nblocks;              // Number of blocks for energy calculation //
  size_t  TotalAtoms;           // Number of Atoms in total for this simulation //
  double3 MaxTranslation;
  double3 MaxRotation;         
  bool    AcceptedFlag;           // Acceptance flag for every simulation // 
};

struct PseudoAtomDefinitions //Always a host struct, never on the device
{
  std::vector<std::string> Name;
  std::vector<double> oxidation;
  std::vector<double> mass;
  std::vector<double> charge;
  std::vector<double> polar; //polarizability
};

struct ForceField
{
  double* epsilon;
  double* sigma;
  double* z; // a third term
  double* shift;
  int*    FFType; //type of force field calculation  
  bool    noCharges;
  size_t  size;
  double  OverlapCriteria;
  double  CutOffVDW;
  double  CutOffCoul;
  double  Prefactor;
  double  Alpha;
};

struct Boxsize
{
  Complex* eik_x;
  Complex* eik_y;
  Complex* eik_z;
  Complex* eik_xy;
  Complex* storedEik;
  Complex* totalEik;
  double*  Cell;
  double*  InverseCell;
  double   Pressure;
  double   Volume;
  double   Temperature;
  bool     Cubic;
  double   ReciprocalCutOff;
  int3     kmax;
};

struct WidomStruct
{
  bool                UseGPUReduction;              // For calculating the energies for each bead
  bool                Useflag;                      // For using flags (for skipping reduction)
  size_t              NumberWidomTrials;            // Number of Trial Positions for the first bead //
  size_t              NumberWidomTrialsOrientations;// Number of Trial Orientations 
  size_t              WidomFirstBeadAllocatesize;   //space allocated for WidomFirstBeadResult
  std::vector<double> Rosenbluth;                   
  std::vector<double> RosenbluthSquared;
  std::vector<double> ExcessMu;
  std::vector<double> ExcessMuSquared;
  std::vector<int>    RosenbluthCount;
};

struct RandomNumber
{
  double* host_random;
  double* device_random;
  size_t  randomsize;
  size_t  offset;
};
