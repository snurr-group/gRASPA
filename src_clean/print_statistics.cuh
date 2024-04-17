static inline void Print_Cycle_Statistics(size_t Cycle, Components& SystemComponents, std::string& Mode)
{
  /*std::string Mode;
  switch(SimulationMode)
  {
    case INITIALIZATION:{Mode = "INITIALIZATION"; break;}
    case EQUILIBRATION: {Mode = "EQUILIBRATION"; break;}
    case PRODUCTION:    {Mode = "PRODUCTION"; break;}
  }
  */
  printf("%s Cycle: %zu, %zu Adsorbate Molecules, Total Energy: %.5f  ||  ", Mode.c_str(), Cycle, SystemComponents.TotalNumberOfMolecules - SystemComponents.NumberOfFrameworks, SystemComponents.CreateMol_Energy.total() + SystemComponents.deltaE.total());
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
    printf("Component %zu [%s], %zu Molecules  ||  ", i, SystemComponents.MoleculeName[i].c_str(), SystemComponents.NumberOfMolecule_for_Component[i]);
  printf("\n");
}

static inline void Print_Translation_Statistics(Move_Statistics MoveStats, double3 MaxTranslation)
{
  if(MoveStats.TranslationTotal == 0) return;
  printf("=====================TRANSLATION MOVES=====================\n");
  printf("Translation Performed: %zu\n", MoveStats.TranslationTotal);
  printf("Translation Accepted: %zu\n", MoveStats.TranslationAccepted);
  printf("Max Translation: %.10f, %.10f, %.10f\n", MaxTranslation.x, MaxTranslation.y, MaxTranslation.z);
  printf("===========================================================\n");
}
 
static inline void Print_Rotation_Statistics(Move_Statistics MoveStats, double3 MaxRotation)
{
  if(MoveStats.RotationTotal == 0) return;
  printf("=====================ROTATION MOVES========================\n");
  printf("Rotation Performed: %zu\n", MoveStats.RotationTotal);
  printf("Rotation Accepted: %zu\n", MoveStats.RotationAccepted);
  printf("Max Rotation: %.10f, %.10f, %.10f\n", MaxRotation.x, MaxRotation.y, MaxRotation.z);
  printf("===========================================================\n");
}

static inline void Print_SpecialRotation_Statistics(Move_Statistics MoveStats, double3 Max)
{
  if(MoveStats.SpecialRotationTotal == 0) return;
  printf("=====================SPECIAL ROTATION MOVES========================\n");
  printf("Special Rotation Performed: %zu\n", MoveStats.SpecialRotationTotal);
  printf("Special Rotation Accepted:  %zu\n", MoveStats.SpecialRotationAccepted);
  printf("Max Special Rotation: %.10f, %.10f, %.10f\n", Max.x, Max.y, Max.z);
  printf("===========================================================\n");
}

static inline void ComputeFugacity(double& AverageWr, double& AverageMu, double& Fugacity, Boxsize& Box, Components& SystemComponents, double3 Stats, Units Constants)
{
  AverageWr = Stats.x/Stats.z;
  AverageMu = Constants.energy_to_kelvin*-(1.0/SystemComponents.Beta)*std::log(AverageWr);
  double WIG = 1.0; //Assume it is rigid molecule, so 1.0//
  size_t adsorbate = 1;
  Fugacity = WIG * Constants.BoltzmannConstant * SystemComponents.Temperature * (double) SystemComponents.NumberOfMolecule_for_Component[adsorbate] / (Box.Volume * 1.0e-30 * AverageWr);
}

static inline void Print_Widom_Statistics(Components& SystemComponents, Boxsize Box, Units& Constants, size_t comp)
{
  Move_Statistics MoveStats = SystemComponents.Moves[comp];
  double2 totRosen = {0.0, 0.0};
  double2 totMu    = {0.0, 0.0};
  double2 totFuga  = {0.0, 0.0};
  printf("=====================Rosenbluth Summary=====================\n");
  printf("There are %zu blocks\n", SystemComponents.Nblock);
  for(size_t i = 0; i < SystemComponents.Nblock; i++)
  {
    printf("=====BLOCK %zu=====\n", i);
    printf("Widom Performed: %.1f\n", MoveStats.Rosen[i].Total.z);
    if(MoveStats.Rosen[i].Total.z > 0)
    {
      double AverageWr = 0.0; double AverageMu = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Total, Constants);
      printf("(Total) Averaged Rosenbluth Weight: %.10f\n", AverageWr);
      printf("(Total) Averaged Excess Mu: %.10f\n", AverageMu);
      printf("(Total) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, SystemComponents.Temperature);
      totRosen.x += AverageWr; totRosen.y += AverageWr * AverageWr;
      totMu.x    += AverageMu; totMu.y    += AverageMu * AverageMu;
      totFuga.x  += Fugacity;  totFuga.y  += Fugacity * Fugacity;
    }
    if(MoveStats.Rosen[i].Insertion.z > 0)
    {
      double AverageWr = 0.0; double AverageMu = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Insertion, Constants);
      printf("(Insertion) Averaged Rosenbluth Weight: %.10f\n", AverageWr);
      printf("(Insertion) Averaged Excess Mu: %.10f\n", AverageMu);
      printf("(Insertion) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, SystemComponents.Temperature);
    } 
    if(MoveStats.Rosen[i].Deletion.z > 0)
    { 
      double AverageWr = 0.0; double AverageMu = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Deletion, Constants);
      printf("(Deletion) Averaged Rosenbluth Weight: %.10f\n", AverageWr);
      printf("(Deletion) Averaged Excess Mu: %.10f\n", AverageMu);
      printf("(Deletion) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, SystemComponents.Temperature);
    }
  }

  double2 AvgBlockRosen = {0.0, 0.0};
  AvgBlockRosen.x = totRosen.x / (double) SystemComponents.Nblock;
  AvgBlockRosen.y = totRosen.y / (double) SystemComponents.Nblock;
  double RosenError = 2.0 * pow((AvgBlockRosen.y - AvgBlockRosen.x * AvgBlockRosen.x), 0.5);

  double2 AvgBlockMu;
  AvgBlockMu.x = totMu.x / (double) SystemComponents.Nblock;
  AvgBlockMu.y = totMu.y / (double) SystemComponents.Nblock;
  double MuError = 2.0 * pow((AvgBlockMu.y - AvgBlockMu.x * AvgBlockMu.x), 0.5);

  double2 AvgBlockFuga;
  AvgBlockFuga.x = totFuga.x / (double) SystemComponents.Nblock;
  AvgBlockFuga.y = totFuga.y / (double) SystemComponents.Nblock;
  double FugaError = 2.0 * pow((AvgBlockFuga.y - AvgBlockFuga.x * AvgBlockFuga.x), 0.5);
  printf("=========================AVERAGE========================\n");
  printf("Averaged Rosenbluth Weight: %.5f +/- %.5f\n", AvgBlockRosen.x, RosenError);
  printf("Averaged Excess Chemical Potential: %.5f +/- %.5f\n", AvgBlockMu.x, MuError);
  printf("Averaged Fugacity: %.5f +/- %.5f\n", AvgBlockFuga.x, FugaError);
}

static inline void Print_Swap_Statistics(Move_Statistics MoveStats)
{
  printf("=====================SWAP MOVES=====================\n");
  printf("Insertion Performed:   %zu\n", MoveStats.InsertionTotal);
  printf("Insertion Accepted:    %zu\n", MoveStats.InsertionAccepted);
  printf("Deletion Performed:    %zu\n", MoveStats.DeletionTotal);
  printf("Deletion Accepted:     %zu\n", MoveStats.DeletionAccepted);
  printf("Reinsertion Performed: %zu\n", MoveStats.ReinsertionTotal);
  printf("Reinsertion Accepted:  %zu\n", MoveStats.ReinsertionAccepted);
  printf("====================================================\n");
}

static inline void Print_IdentitySwap_Statistics(Components& SystemComponents, size_t i)
{
  printf("=====================IDENTITY SWAP MOVES=====================\n");
  for(size_t j = 0; j < SystemComponents.Total_Components; j++)
  {
    if(SystemComponents.Moves[i].TotalProb < 1e-10) continue;
    if(SystemComponents.Moves[i].IdentitySwapProb - SystemComponents.Moves[i].ReinsertionProb < 1e-10) continue;
    if(j != i) printf("Identity Swap Performed, FROM [%s (%zu)] TO [%s (%zu)]: %zu (%zu Accepted)\n", SystemComponents.MoleculeName[i].c_str(), i, SystemComponents.MoleculeName[j].c_str(), j, SystemComponents.Moves[i].IdentitySwap_Total_TO[j], SystemComponents.Moves[i].IdentitySwap_Acc_TO[j]);
  }
  printf("=============================================================\n");
}

static inline void Print_CBCF_Statistics(Move_Statistics MoveStats)
{
  printf("=====================CBCF SWAP MOVES================\n");
  printf("CBCF Performed:           %zu\n", MoveStats.CBCFTotal);
  printf("CBCF Accepted:            %zu\n", MoveStats.CBCFAccepted);
  printf("CBCF Insertion Performed: %zu\n", MoveStats.CBCFInsertionTotal);
  printf("CBCF Insertion Accepted:  %zu\n", MoveStats.CBCFInsertionAccepted);
  printf("CBCF Lambda Performed:    %zu\n", MoveStats.CBCFLambdaTotal);
  printf("CBCF Lambda Accepted:     %zu\n", MoveStats.CBCFLambdaAccepted);
  printf("CBCF Deletion Performed:  %zu\n", MoveStats.CBCFDeletionTotal);
  printf("CBCF Deletion Accepted:   %zu\n", MoveStats.CBCFDeletionAccepted);
  printf("====================================================\n");
}

//Xiaoyi's code about mass unit data//
// This function converts the given molecule to its mass-mass representation.
// The function is marked "inline" for potential optimization by the compiler.
// It works on structures and vectors in order to provide the desired conversion.
static inline std::vector<double2> ConvertMoleculetoMassMass(Components& SystemComponents, size_t i, std::vector<double2>& input, size_t Nblock)
{  
  // Number of components in the framework
  int FrameworkComponents = SystemComponents.NComponents.y;

  // Initialize the mass of the cell
  double CellMass = 0.0;

  // Create an empty result vector to hold the mass-mass conversion
  // Reserve memory for efficiency based on the size of the input vector.
  std::vector<double2> result;
  result.reserve(input.size());

  // Loop through each framework component
  // The molecularweight here are per unit cell (for framework components)
  for(size_t j = 0; j < FrameworkComponents; j++)
  {
    //size_t NMol = SystemComponents.NumberOfMolecule_for_Component[j];
    // Incrementally sum up the molecular weight for each component
    size_t NCell = SystemComponents.NumberofUnitCells.x * SystemComponents.NumberofUnitCells.y * SystemComponents.NumberofUnitCells.z;
    CellMass += SystemComponents.MolecularWeight[j] * NCell;
    printf("Framework component %zu, molar mass: %.5f\n", j, SystemComponents.MolecularWeight[j]);
  }
  printf("Framework total mass: %.5f\n", CellMass);

  // Calculate the ratio of the molecular weight of the current molecule to the total cell mass, scaled to milligrams/gram
  double MiligramPerGram = 1000.0*SystemComponents.MolecularWeight[i]/CellMass;

  // Loop through each block of the input
  for(size_t i = 0; i < Nblock; i++)
  {
    // Extract the average and squared average values from the input
    double Average   = input[i].x;
    double SQAverage = input[i].y;

    // Declare a new variable to hold the mass-mass conversion
    double2 val; 

    // Convert the average and squared average to their mass-mass representation
    val.x = Average * MiligramPerGram; 
    val.y = SQAverage*MiligramPerGram*MiligramPerGram;

    // Append the converted values to the result vector
    result.push_back(val);
  }

  // Return the result vector
  return result;
}


// This function converts the given molecule data to its mole-mass representation.
// The function is marked "inline" for potential optimization by the compiler.
// It operates on structures and vectors to provide the desired conversion.
static inline std::vector<double2> ConvertMoleculetoMolMass(Components& SystemComponents, size_t i, std::vector<double2>& input, size_t Nblock)
{
  // Number of components in the framework
  int FrameworkComponents = SystemComponents.NComponents.y;

  // Initialize the mass of the unit cell
  double CellMass = 0.0;

  // Create an empty result vector to hold the mole-mass conversion.
  // Reserve memory based on the size of the input vector for efficiency.
  std::vector<double2> result;
  result.reserve(input.size());

  // Loop through each framework component
  for(size_t j = 0; j < FrameworkComponents; j++)
  {  

    // Calculate the total number of molecules in all unit cells
    size_t NCell = SystemComponents.NumberofUnitCells.x * SystemComponents.NumberofUnitCells.y * SystemComponents.NumberofUnitCells.z;
    
    // Incrementally sum up the molecular weight for each component
    // The molecular weight here for framework components are per unit cell
    CellMass += SystemComponents.MolecularWeight[j] * NCell;
    printf("Framework component %zu, molar mass: %.5f\n", j, SystemComponents.MolecularWeight[j]);
  }
  printf("Framework total mass: %.5f\n", CellMass);
  // Calculate the ratio of moles per kilogram based on the unit cell mass
  double MolPerKilogram = 1000.0/CellMass;

  // Loop through each block of the input
  for(size_t i = 0; i < Nblock; i++)
  {
    // Extract the average and squared average values from the input
    double Average   = input[i].x;
    double SQAverage = input[i].y;

    // Declare a new variable to hold the mole-mass conversion
    double2 val; 

    // Convert the average and squared average values to their mole-mass representations
    val.x = Average * MolPerKilogram; 
    val.y = SQAverage*MolPerKilogram*MolPerKilogram;

    // Append the converted values to the result vector
    result.push_back(val);
  }

  // Return the result vector
  return result;
}

// This function converts the given molecule data to its mass-volume representation.
// The function is marked "inline" for potential optimization by the compiler.
// It operates on structures and vectors to provide the desired conversion.
static inline std::vector<double2> ConvertMoleculetoMassVolume(Components& SystemComponents, size_t i, std::vector<double2>& input, size_t Nblock, Simulations& Sims)
{
  // Create an empty result vector to hold the mass-volume conversion.
  // Reserve memory based on the size of the input vector for efficiency.
  std::vector<double2> result;
  result.reserve(input.size());

  // Calculate the total number of molecules in all unit cells
  int NMol_In_Def = SystemComponents.NumberofUnitCells.x * SystemComponents.NumberofUnitCells.y * SystemComponents.NumberofUnitCells.z;

  // Print the total number of unit cells (for debugging or informational purposes)
  printf("Total Unit Cells %d \n", NMol_In_Def);

  // Retrieve the volume of the simulation box
  double Volume = Sims.Box.Volume;

  // Calculate the ratio of grams per liter, factoring in the number of unit cells, molecular weight, volume, and Avogadro's number
  double GramPerLiter = SystemComponents.MolecularWeight[i]*10000/Volume/6.0221408;

  // Loop through each block of the input
  for(size_t i = 0; i < Nblock; i++)
  {
    // Extract the average and squared average values from the input
    double Average   = input[i].x;
    double SQAverage = input[i].y;

    // Declare a new variable to hold the mass-volume conversion
    double2 val; 

    // Convert the average and squared average values to their mass-volume representations
    val.x = Average * GramPerLiter; 
    val.y = SQAverage*GramPerLiter*GramPerLiter;

    // Append the converted values to the result vector
    result.push_back(val);
  }

  // Return the result vector
  return result;
}

static inline void Gather_Averages(std::vector<double2>& Array, double init_energy, double running_energy, int Cycles, int Blocksize, size_t Nblock)
{
  //Determine the block id//
  size_t blockID = Cycles/Blocksize;
  if(blockID >= Nblock) blockID --;
  //Get total energy//
  double total_energy = init_energy + running_energy;
  Array[blockID].x += total_energy;
  Array[blockID].y += total_energy * total_energy;
}

static inline void Print_Values(std::vector<double2>& Array, int Cycles, int Blocksize, size_t Nblock)
{
  double OverallAverage    = 0.0;
  double OverallSQAverage  = 0.0;
  for(size_t i = 0; i < Nblock; i++)
  {
    double Average   = Array[i].x;
    double SQAverage = Array[i].y;
    if(i == Nblock-1)
    {
      if(Cycles % Blocksize != 0) Blocksize += Cycles % Blocksize;
    }
    Average   /= Blocksize;
    SQAverage /= Blocksize;
    double Errorbar  = 2.0 * pow((SQAverage - Average * Average), 0.5);
    printf("BLOCK [%zu], Blocksize: %i, Average: %.5f, ErrorBar: %.5f\n", i, Blocksize, Average, Errorbar);
    OverallAverage  += Average;
    OverallSQAverage += Average * Average;
  }
  printf("Overall: Average: %.5f, ErrorBar: %.5f\n", OverallAverage/Nblock, 2.0 * pow((OverallSQAverage/Nblock - OverallAverage/Nblock * OverallAverage/Nblock), 0.5));
}

static inline void Print_Averages(Components& SystemComponents, int Cycles, int Blocksize, Simulations& Sims)
{
  printf("=====================BLOCK AVERAGES (ENERGIES)================\n");
  std::vector<double2>Temp = SystemComponents.EnergyAverage;
  Print_Values(Temp, Cycles, Blocksize, SystemComponents.Nblock);
  printf("==============================================================\n");
  printf("=================== BLOCK AVERAGES (LOADING: # MOLECULES)=============\n");
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    Print_Values(Temp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("======================================================================\n");
  
  printf("=====================BLOCK AVERAGES (LOADING: mg/g)=============\n");
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    std::vector<double2>MMTemp = ConvertMoleculetoMassMass(SystemComponents, i, Temp, SystemComponents.Nblock);
    Print_Values(MMTemp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
  printf("=====================BLOCK AVERAGES (LOADING: mol/kg)=============\n");
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    std::vector<double2>MMTemp = ConvertMoleculetoMolMass(SystemComponents, i, Temp, SystemComponents.Nblock);
    Print_Values(MMTemp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
  printf("=====================BLOCK AVERAGES (LOADING: g/L)=============\n");
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    std::vector<double2>MMTemp = ConvertMoleculetoMassVolume(SystemComponents, i, Temp, SystemComponents.Nblock,Sims);
    Print_Values(MMTemp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
}

static inline void Gather_Averages_Types(std::vector<double2>& Array, double init_value, double running_value, int Cycles, int Blocksize, size_t Nblock)
{
  //Determine the block id//
  size_t blockID = Cycles/Blocksize;
  if(blockID >= Nblock) blockID --;
  //Get total energy//
  double total_value = init_value + running_value;
  Array[blockID].x += total_value;
  Array[blockID].y += total_value * total_value;
}

///////////////////////////////////////////////////////
// Wrapper for the functions for printing statistics //
///////////////////////////////////////////////////////
static inline void PrintAllStatistics(Components& SystemComponents, Simulations& Sims, size_t Cycles, int SimulationMode, size_t BlockAverageSize)
{
  for(size_t comp = 0; comp < SystemComponents.Total_Components; comp++)
  {
    if(SystemComponents.Moves[comp].TotalProb < 1e-10) continue;
    printf("======================== MOVE STATISTICS FOR COMPONENT [%zu] (%s) ========================\n", comp,SystemComponents.MoleculeName[comp].c_str());
    Print_Translation_Statistics(SystemComponents.Moves[comp], SystemComponents.MaxTranslation[comp]);
    Print_Rotation_Statistics(SystemComponents.Moves[comp], SystemComponents.MaxRotation[comp]);
    Print_SpecialRotation_Statistics(SystemComponents.Moves[comp], SystemComponents.MaxSpecialRotation[comp]);
    Print_Swap_Statistics(SystemComponents.Moves[comp]);
    Print_IdentitySwap_Statistics(SystemComponents, comp);
    if(SystemComponents.hasfractionalMolecule[comp]) Print_CBCF_Statistics(SystemComponents.Moves[comp]);
    printf("================================================================================================\n");
  }
  if(SimulationMode == PRODUCTION)
  {
    Print_Averages(SystemComponents, Cycles, BlockAverageSize, Sims);
  }
}

static inline void PrintGibbs(Gibbs& GibbsStatistics)
{
  printf("=====================GIBBS MONTE CARLO STATISTICS=====================\n");
  printf("GIBBS VOLUME MOVE ATTEMPTS: %zu\n", (size_t) GibbsStatistics.GibbsBoxStats.x);
  printf("GIBBS VOLUME MOVE ACCEPTED: %zu\n", (size_t) GibbsStatistics.GibbsBoxStats.y);
  printf("GIBBS VOLUME MOVE TOOK    : %.5f [seconds]\n", GibbsStatistics.GibbsTime);
  printf("GIBBS PARTICLE TRANSFER MOVE ATTEMPTS: %zu\n", (size_t) GibbsStatistics.GibbsXferStats.x);
  printf("GIBBS PARTICLE TRANSFER MOVE ACCEPTED: %zu\n", (size_t) GibbsStatistics.GibbsXferStats.y);
  printf("======================================================================\n");
}
