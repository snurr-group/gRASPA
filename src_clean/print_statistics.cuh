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
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
    printf("Component %zu [%s], %zu Molecules  ||  ", i, SystemComponents.MoleculeName[i].c_str(), SystemComponents.NumberOfMolecule_for_Component[i]);
  printf("\n");
}

static inline void Print_Translation_Statistics(Move_Statistics MoveStats, double3 MaxTranslation)
{
  if(MoveStats.CumTranslationTotal == 0 && MoveStats.TranslationTotal == 0) return;
  printf("=====================TRANSLATION MOVES=====================\n");
  printf("Translation Performed: %zu\n", MoveStats.CumTranslationTotal > MoveStats.TranslationTotal ? MoveStats.CumTranslationTotal : MoveStats.TranslationTotal);
  printf("Translation Accepted: %zu\n", MoveStats.CumTranslationAccepted > MoveStats.TranslationAccepted ? MoveStats.CumTranslationAccepted : MoveStats.TranslationAccepted);
  printf("Max Translation: %.10f, %.10f, %.10f\n", MaxTranslation.x, MaxTranslation.y, MaxTranslation.z);
  printf("===========================================================\n");
}
 
static inline void Print_Rotation_Statistics(Move_Statistics MoveStats, double3 MaxRotation)
{
  if(MoveStats.CumRotationTotal == 0) return;
  printf("=====================ROTATION MOVES========================\n");
  printf("Rotation Performed: %zu\n", MoveStats.CumRotationTotal > MoveStats.RotationTotal ? MoveStats.CumRotationTotal : MoveStats.RotationTotal);
  printf("Rotation Accepted: %zu\n", MoveStats.CumRotationAccepted > MoveStats.RotationAccepted ? MoveStats.CumRotationAccepted : MoveStats.RotationAccepted);
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

// Framework density
static inline double FrameworkDensity(Components& SystemComponents, Boxsize& Box, Units Constants)
{
  // Number of components in the framework
  int FrameworkComponents = SystemComponents.NComponents.y;

  // Initialize the mass of the unit cell
  double CellMass = 0.0;

  // Loop through each framework component
  for(size_t j = 0; j < FrameworkComponents; j++)
  {  

    // Calculate the total number of molecules in all unit cells
    size_t NCell = SystemComponents.NumberofUnitCells.x * SystemComponents.NumberofUnitCells.y * SystemComponents.NumberofUnitCells.z;

    // Incrementally sum up the molecular weight for each component
    // The molecular weight here for framework components are per unit cell
    CellMass += SystemComponents.MolecularWeight[j] * NCell;
    //printf("Framework component %zu, molar mass: %.5f\n", j, SystemComponents.MolecularWeight[j]);
  }
  //printf("Framework total mass: %.5f\n", CellMass);

  // Calculate framework density [kg/m^3]
  double rho = CellMass * 1.0e-3 / (Constants.Avogadro * Box.Volume * 1.0e-30);
  printf("Framework Density: %.5f [kg/m^3]\n", rho);

  // Return density
  return rho;
}

static inline void ComputeFugacity(double& AverageWr, double& AverageMu, double& AverageHenry, double& Fugacity, Boxsize& Box, Components& SystemComponents, double3 Stats, Units Constants)
{
  AverageWr = Stats.x/Stats.z;
  AverageMu = Constants.energy_to_kelvin*-(1.0/SystemComponents.Beta)*std::log(AverageWr);
  // Adsorption Henry's constant, assuming rigid molecule, WIG = 1.0
  double rho_framework = FrameworkDensity(SystemComponents, Box, Constants);
  AverageHenry = AverageWr/(Constants.gas_constant*SystemComponents.Temperature*rho_framework);  // units [mol/Pa/kg]
  double WIG = 1.0; //Assume it is rigid molecule, so 1.0//
  size_t adsorbate = 1;
  Fugacity = WIG * Constants.BoltzmannConstant * SystemComponents.Temperature * (double) SystemComponents.NumberOfMolecule_for_Component[adsorbate] / (Box.Volume * 1.0e-30 * AverageWr);
}

static inline void Print_Widom_Statistics(Components& SystemComponents, Boxsize Box, Units& Constants, size_t comp)
{
  Move_Statistics MoveStats = SystemComponents.Moves[comp];
  double2 totRosen = {0.0, 0.0};
  double2 totMu    = {0.0, 0.0};
  double2 totHenry = {0.0, 0.0};
  double2 totFuga  = {0.0, 0.0};
  printf("=====================Rosenbluth Summary=====================\n");
  printf("There are %zu blocks\n", SystemComponents.Nblock);
  for(size_t i = 0; i < SystemComponents.Nblock; i++)
  {
    printf("=====BLOCK %zu=====\n", i);
    printf("Widom Performed: %.1f\n", MoveStats.Rosen[i].Total.z);
    if(MoveStats.Rosen[i].Total.z > 0)
    {
      double AverageWr = 0.0; double AverageMu = 0.0; double AverageHenry = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, AverageHenry, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Total, Constants);
      printf("(Total) Averaged Rosenbluth Weight: %.10f\n", AverageWr);
      printf("(Total) Averaged Excess Mu: %.10f\n", AverageMu);
      printf("(Total) Averaged Henry Coefficient: %.10f\n", AverageHenry);
      printf("(Total) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, SystemComponents.Temperature);
      totRosen.x += AverageWr; totRosen.y += AverageWr * AverageWr;
      totMu.x    += AverageMu; totMu.y    += AverageMu * AverageMu;
      totHenry.x += AverageHenry; totHenry.y += AverageHenry * AverageHenry;
      totFuga.x  += Fugacity;  totFuga.y  += Fugacity * Fugacity;
    }
    if(MoveStats.Rosen[i].Insertion.z > 0)
    {
      double AverageWr = 0.0; double AverageMu = 0.0; double AverageHenry = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, AverageHenry, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Insertion, Constants);
      printf("(Insertion) Averaged Rosenbluth Weight: %.10f\n", AverageWr);
      printf("(Insertion) Averaged Excess Mu: %.10f\n", AverageMu);
      printf("(Insertion) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, SystemComponents.Temperature);
    } 
    if(MoveStats.Rosen[i].Deletion.z > 0)
    { 
      double AverageWr = 0.0; double AverageMu = 0.0; double AverageHenry = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, AverageHenry, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Deletion, Constants);
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

  double2 AvgBlockHenry;
  AvgBlockHenry.x = totHenry.x / (double) SystemComponents.Nblock;
  AvgBlockHenry.y = totHenry.y / (double) SystemComponents.Nblock;
  double HenryError = 2.0 * pow((AvgBlockHenry.y - AvgBlockHenry.x * AvgBlockHenry.x), 0.5);

  double2 AvgBlockFuga;
  AvgBlockFuga.x = totFuga.x / (double) SystemComponents.Nblock;
  AvgBlockFuga.y = totFuga.y / (double) SystemComponents.Nblock;
  double FugaError = 2.0 * pow((AvgBlockFuga.y - AvgBlockFuga.x * AvgBlockFuga.x), 0.5);
  printf("=========================AVERAGE========================\n");
  printf("Averaged Rosenbluth Weight: %.10f +/- %.10f\n", AvgBlockRosen.x, RosenError);
  printf("Averaged Excess Chemical Potential: %.10f +/- %.10f\n", AvgBlockMu.x, MuError);
  printf("Averaged Henry Coefficient [mol/kg/Pa]: %.10g +/- %.10g\n", AvgBlockHenry.x, HenryError);
  printf("Averaged Fugacity: %.10f +/- %.10f\n", AvgBlockFuga.x, FugaError);
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
  for(size_t j = 0; j < SystemComponents.NComponents.x; j++)
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

  // Calculate the ratio of grams per liter, factoring in the number of unit cells, molecular weight, volume, and Avogadro's number
  double GramPerLiter = SystemComponents.MolecularWeight[i]*10000/6.0221408;

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

//Zhao's note: do not use pass by ref for DeltaE
void Gather_Averages_MoveEnergy(Components& SystemComponents, int Cycles, int Blocksize, MoveEnergy DeltaE)
{
  size_t blockID = Cycles/Blocksize;
  if(blockID >= SystemComponents.Nblock) blockID --;
  //size_t blockID = SystemComponents.Moves[0].BlockID;
  //if(blockID == SystemComponents.Nblock-1)
  //{
  //  if(Cycles % Blocksize != 0) Blocksize += Cycles % Blocksize;
  //}
  //Get total energy//
  //MoveEnergy UpdateDeltaE = ;
  SystemComponents.BookKeepEnergy[blockID]    += DeltaE;
  //SystemComponents.BookKeepEnergy_SQ[blockID] += DeltaE / static_cast<double>(Blocksize) * DeltaE;
}

void Calculate_Overall_Averages_MoveEnergy(Components& SystemComponents, int Blocksize, int Cycles)
{
  //Calculate just the overall, now//
  MoveEnergy AverageEnergy_SQ;
  for(size_t i = 0; i < SystemComponents.Nblock; i++)
  {
    if(i == SystemComponents.Nblock-1)
    {
      if(Cycles % Blocksize != 0) Blocksize += Cycles % Blocksize;
    }
    MoveEnergy Average = SystemComponents.BookKeepEnergy[i] / static_cast<double>(Blocksize);
    SystemComponents.AverageEnergy          += Average / static_cast<double>(SystemComponents.Nblock);
    AverageEnergy_SQ                        += Average * Average / static_cast<double>(SystemComponents.Nblock);
  }
  SystemComponents.AverageEnergy_Errorbar = sqrt_MoveEnergy(AverageEnergy_SQ - SystemComponents.AverageEnergy * SystemComponents.AverageEnergy) * 2.0;
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

/////////////////////////////////////////////
//Qst, for single/multiple components      //
//Modified from RASPA-2                    //
//and Kaihang's 1 component implementation //
/////////////////////////////////////////////
static inline void Print_HeatOfAdsorption(Components& SystemComponents, int Cycles, int Blocksize, size_t Nblock, Units& Constants)
{
  double Temperature = SystemComponents.Temperature;
  size_t NumberOfAdsorbateComponents = SystemComponents.NComponents.x - SystemComponents.NComponents.y;

  std::vector<std::vector<double>>HeatOfAdsorption(NumberOfAdsorbateComponents, std::vector<double>(Nblock, 0.0));
  for(size_t i = 0; i < Nblock; i++)
  {
    if(i == Nblock-1)
    {
      if(Cycles % Blocksize != 0) Blocksize += Cycles % Blocksize;
    }

    // <E>
    double Average_E   = SystemComponents.BookKeepEnergy[i].total();
    Average_E -= SystemComponents.BookKeepEnergy[i].HHVDW;
    Average_E -= SystemComponents.BookKeepEnergy[i].HHReal;
    Average_E -= SystemComponents.BookKeepEnergy[i].HHEwaldE;
    Average_E /= Blocksize;

    std::vector<std::vector<double>>matrix(NumberOfAdsorbateComponents, std::vector<double>(NumberOfAdsorbateComponents, 0.0));
    std::vector<std::vector<double>>temp_matrix(NumberOfAdsorbateComponents, std::vector<double>(NumberOfAdsorbateComponents, 0.0));

    for(size_t compi = SystemComponents.NComponents.y; compi < SystemComponents.NComponents.x; compi++)
    {
      for(size_t compj = SystemComponents.NComponents.y; compj < SystemComponents.NComponents.x; compj++)
      {
        size_t adjust_compi = compi - SystemComponents.NComponents.y; //for matrix, shift by framework component
        size_t adjust_compj = compj - SystemComponents.NComponents.y; //for matrix, shift by framework component
        double Average_N    = SystemComponents.Moves[compi].MolAverage[i].x / Blocksize;
        double Average_Nj   = SystemComponents.Moves[compj].MolAverage[i].x / Blocksize;
        double Average_NxNj = SystemComponents.Moves[compi].MolSQPerComponent[compj][i] / Blocksize;

        matrix[adjust_compi][adjust_compj] = Average_NxNj - Average_N * Average_Nj;
      }
    }
    GaussJordan(matrix, temp_matrix); //inversed matrix is saved in the input matrix, temp_matrix is a temporary, not used

    for(size_t compi = SystemComponents.NComponents.y; compi < SystemComponents.NComponents.x; compi++)
    {
      size_t adjust_compi = compi - SystemComponents.NComponents.y; //for matrix, shift by framework component
      for(size_t compj = SystemComponents.NComponents.y; compj < SystemComponents.NComponents.x; compj++)
      {
        double Average_N   = SystemComponents.Moves[compj].MolAverage[i].x / Blocksize; // <N>
        double SQAverage_N = SystemComponents.Moves[compj].MolAverage[i].y / Blocksize; // <N^2>
        double Average_ExN = SystemComponents.EnergyTimesNumberOfMolecule[compj][i] / Blocksize; //<E*N>

        size_t adjust_compj = compj - SystemComponents.NComponents.y; //for matrix, shift by framework component
        double One_Over_Variance = 1.0 / (SQAverage_N - Average_N * Average_N);

        //if(SystemComponents.NComponents.x - SystemComponents.NComponents.y > 1) 
        One_Over_Variance = matrix[adjust_compj][adjust_compi]; //Inversed matrix, multiple components

        // Calculate heat of adsorption [kJ/mol]
        HeatOfAdsorption[adjust_compi][i] += ( Constants.energy_to_kelvin * (Average_ExN - Average_E * Average_N) * One_Over_Variance);
      }
      double kelvin_to_kjmol = 0.01 / Constants.energy_to_kelvin;
      HeatOfAdsorption[adjust_compi][i] -= Temperature; 
      HeatOfAdsorption[adjust_compi][i] *= kelvin_to_kjmol;
    }
  }
  //print values//
  for(size_t compi = SystemComponents.NComponents.y; compi < SystemComponents.NComponents.x; compi++)
  {
    double OverallHeatOfAdsorption    = 0.0;
    double OverallSQHeatOfAdsorption  = 0.0;
    size_t adjust_compi = compi - SystemComponents.NComponents.y; //for matrix, shift by framework component
    printf("COMPONENT [%zu] (%s)\n", compi, SystemComponents.MoleculeName[compi].c_str());
    for(size_t i = 0; i < Nblock; i++)
    {
      printf("BLOCK [%zu], Blocksize: %i, Average: %.5f\n", i, Blocksize, HeatOfAdsorption[adjust_compi][i]);
      OverallHeatOfAdsorption   += HeatOfAdsorption[adjust_compi][i];
      OverallSQHeatOfAdsorption += HeatOfAdsorption[adjust_compi][i] * HeatOfAdsorption[adjust_compi][i];
    }
    printf("Overall: Average: %.5f, ErrorBar: %.5f\n", OverallHeatOfAdsorption/Nblock, 2.0 * pow((OverallSQHeatOfAdsorption/Nblock - OverallHeatOfAdsorption/Nblock * OverallHeatOfAdsorption/Nblock), 0.5));
    printf("-----------------------------\n");
  }
}

static inline void Print_Averages(Components& SystemComponents, int Cycles, int Blocksize, Simulations& Sims, Units& Constants)
{
  printf("============= BLOCK AVERAGES (HEAT OF ADSORPTION: kJ/mol) =========\n");
  Print_HeatOfAdsorption(SystemComponents, Cycles, Blocksize, SystemComponents.Nblock, Constants);
  printf("==============================================================\n");

  printf("=================== BLOCK AVERAGES (LOADING: # MOLECULES)=============\n");
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    Print_Values(Temp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("======================================================================\n");
  
  printf("=====================BLOCK AVERAGES (LOADING: mg/g)=============\n");
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    std::vector<double2>MMTemp = ConvertMoleculetoMassMass(SystemComponents, i, Temp, SystemComponents.Nblock);
    Print_Values(MMTemp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
  printf("=====================BLOCK AVERAGES (LOADING: mol/kg)=============\n");
  for(size_t i = 0; i < SystemComponents.NComponents.x; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    std::vector<double2>MMTemp = ConvertMoleculetoMolMass(SystemComponents, i, Temp, SystemComponents.Nblock);
    Print_Values(MMTemp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
  printf("=====================BLOCK AVERAGES (LOADING: g/L)=============\n");
  for(size_t i = SystemComponents.NComponents.y; i < SystemComponents.NComponents.x; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>MMTemp = ConvertMoleculetoMassVolume(SystemComponents, i, SystemComponents.DensityPerComponent[i], SystemComponents.Nblock,Sims);
    Print_Values(MMTemp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
  printf("=====================BLOCK AVERAGES (VOLUME Å^3)================\n");
  std::vector<double2>Temp = SystemComponents.VolumeAverage;
  Print_Values(Temp, Cycles, Blocksize, SystemComponents.Nblock);
  printf("----------------------------------------------------------\n");
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

// Kaihang Shi: Added for the purpose of heat of adsorption
// Just to keep the average, not SQAverage
static inline void Gather_Averages_double(std::vector<double>& Array, double value, int Cycles, int Blocksize, size_t Nblock)
{
  //Determine the block id//
  size_t blockID = Cycles/Blocksize;
  if(blockID >= Nblock) blockID --;
  Array[blockID] += value;
}

///////////////////////////////////////////////////////
// Wrapper for the functions for printing statistics //
///////////////////////////////////////////////////////
static inline void PrintAllStatistics(Components& SystemComponents, Simulations& Sims, size_t Cycles, int SimulationMode, size_t BlockAverageSize, Units& Constants)
{
  for(size_t comp = 0; comp < SystemComponents.NComponents.x; comp++)
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
    Print_Averages(SystemComponents, Cycles, BlockAverageSize, Sims, Constants);
  }
}

static inline void PrintSystemMoves(Variables& Var)
{
  for(size_t i = 0; i < Var.SystemComponents.size(); i++)
  {
    Gibbs& GibbsStatistics = Var.GibbsStatistics;
    if(GibbsStatistics.DoGibbs)
    {
      printf("=====================GIBBS MONTE CARLO STATISTICS=====================\n");
      printf("GIBBS VOLUME MOVE ATTEMPTS: %zu\n", GibbsStatistics.TotalGibbsBoxStats.x > GibbsStatistics.GibbsBoxStats.x ? (size_t) GibbsStatistics.TotalGibbsBoxStats.x : (size_t) GibbsStatistics.GibbsBoxStats.x);
      printf("GIBBS VOLUME MOVE ACCEPTED: %zu\n", GibbsStatistics.TotalGibbsBoxStats.y > GibbsStatistics.GibbsBoxStats.y ? (size_t) GibbsStatistics.TotalGibbsBoxStats.y : (size_t) GibbsStatistics.GibbsBoxStats.y);
      printf("GIBBS VOLUME MOVE TOOK    : %.5f [seconds]\n", GibbsStatistics.GibbsTime);
      printf("GIBBS PARTICLE TRANSFER MOVE ATTEMPTS: %zu\n", (size_t) GibbsStatistics.GibbsXferStats.x);
      printf("GIBBS PARTICLE TRANSFER MOVE ACCEPTED: %zu\n", (size_t) GibbsStatistics.GibbsXferStats.y);
      printf("======================================================================\n");
    }
    if(Var.SystemComponents[i].PerformVolumeMove && (Var.SystemComponents[i].VolumeMoveTotalAttempts > 0 || Var.SystemComponents[i].VolumeMoveAttempts > 0))
    {
      printf("=====================VOLUME MOVE STATISTICS for BOX [%zu]=====================\n", i);
      size_t move_performed = Var.SystemComponents[i].VolumeMoveAttempts > Var.SystemComponents[i].VolumeMoveTotalAttempts ? Var.SystemComponents[i].VolumeMoveAttempts : Var.SystemComponents[i].VolumeMoveTotalAttempts;
      size_t move_accepted  = Var.SystemComponents[i].VolumeMoveAccepted > Var.SystemComponents[i].VolumeMoveTotalAccepted ? Var.SystemComponents[i].VolumeMoveAccepted : Var.SystemComponents[i].VolumeMoveTotalAccepted;
      printf("VOLUME MOVE ATTEMPTS:   %zu\n", move_performed);
      printf("VOLUME MOVE ACCEPTED:   %zu\n", move_accepted);
      printf("VOLUME MOVE MAX CHANGE: %.5f\n", Var.SystemComponents[i].VolumeMoveMaxChange);
      printf("VOLUME MOVE TOOK    : %.5f [seconds]\n", Var.SystemComponents[i].VolumeMoveTime);
      printf("======================================================================\n");
    }
  }
}
