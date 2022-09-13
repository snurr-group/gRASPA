#include <stdio.h>

/*struct Components
{
  Atoms comp[N]; //Component
}*/

struct Units
{
  double MassUnit;
  double TimeUnit;
  double LengthUnit;
  double energy_to_kelvin;
  double BoltzmannConstant;
};

struct Components
{
  size_t  Total_Components; // Number of Components in the system (including framework)
  size_t* Moleculesize; // Number of Pseudo-Atoms in the molecule
  size_t* NumberOfMolecule_for_Component; // Number of Molecules for each component
  double* MolFraction;
  double* IdealRosenbluthWeight;
  double* FugacityCoeff; 
  size_t  TotalNumberOfMolecules; // Total Number of Molecules (including framework)
  size_t  NumberOfFrameworks; // Total Number of framework species, usually 1.
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
  //uint_32_t* CompID; //
  size_t* MolID;
  size_t  size;
  //uint_32_t  size;   // These values are small, the can use 32-bit, need to test
  size_t  Allocate_size;
};

struct ForceField
{
  double* epsilon;
  double* sigma;
  double* z; // a third term
  double* shift;
  int*    FFType; //type of force field calculation  
  double* FFParams; // cutoffVDW, cutoffCoulomb...
  int*    OtherParams;  
  bool    noCharges;
  size_t  size;
  double* MaxTranslation;
  double Beta;
};

struct Boxsize
{
  double* Cell;
  double* InverseCell;
  double  Pressure;
  double  Volume;
  double  Temperature;
};

struct WidomStruct
{
  size_t NumberWidomTrials;
  bool UseGPUReduction; //For calculating the energies for each bead
  bool Useflag;         //for using flags (for skipping reduction)
  double* WidomFirstBeadResult; //for device, temporary result array
  double* Rosenbluth;
  double* RosenbluthSquared;
  double* ExcessMu;
  double* ExcessMuSquared;
  int* RosenbluthCount;
  size_t  InsertionTotal;
  size_t  DeletionTotal;
  size_t  InsertionAccepted;
  size_t  DeletionAccepted;
  size_t  WidomFirstBeadAllocatesize; //space allocated for WidomFirstBeadResult
};

struct Move_Statistics
{
  double TranslationProb;
  int TranslationAccepted;
  int TranslationTotal;
  double TranslationAccRatio;
  double WidomProb;
  double SwapProb;
  size_t NumberOfBlocks; //Number of blocks for statistical averaging
};

struct RandomNumber
{
  double* host_random;
  double* device_random;
  size_t  randomsize;
  size_t  offset;
};
