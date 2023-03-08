#include "data_struct.h"

void Print_Translation_Statistics(Move_Statistics MoveStats, double3 MaxTranslation)
{
  printf("=====================TRANSLATION MOVES=====================\n");
  printf("Translation Performed: %zu\n", MoveStats.TranslationTotal);
  printf("Translation Accepted: %zu\n", MoveStats.TranslationAccepted);
  printf("Max Translation: %.10f, %.10f, %.10f\n", MaxTranslation.x, MaxTranslation.y, MaxTranslation.z);
  printf("===========================================================\n");
}
 
void Print_Rotation_Statistics(Move_Statistics MoveStats, double3 MaxRotation)
{
  printf("=====================ROTATION MOVES========================\n");
  printf("Rotation Performed: %zu\n", MoveStats.RotationTotal);
  printf("Rotation Accepted: %zu\n", MoveStats.RotationAccepted);
  printf("Max Rotation: %.10f, %.10f, %.10f\n", MaxRotation.x, MaxRotation.y, MaxRotation.z);
  printf("===========================================================\n");
}

static inline void ComputeFugacity(double& AverageWr, double& AverageMu, double& Fugacity, Boxsize& Box, Components& SystemComponents, double3 Stats, Units Constants)
{
  AverageWr = Stats.x/Stats.z;
  AverageMu = Constants.energy_to_kelvin*-(1.0/SystemComponents.Beta)*std::log(AverageWr);
  double WIG = 1.0; //Assume it is rigid molecule, so 1.0//
  size_t adsorbate = 1;
  Fugacity = WIG * Constants.BoltzmannConstant * Box.Temperature * (double) SystemComponents.NumberOfMolecule_for_Component[adsorbate] / (Box.Volume * 1.0e-30 * AverageWr);
}

void Print_Widom_Statistics(Components& SystemComponents, Boxsize Box, Units& Constants, size_t comp)
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
      printf("(Total) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, Box.Temperature);
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
      printf("(Insertion) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, Box.Temperature);
    } 
    if(MoveStats.Rosen[i].Deletion.z > 0)
    { 
      double AverageWr = 0.0; double AverageMu = 0.0; double Fugacity = 0.0;
      ComputeFugacity(AverageWr, AverageMu, Fugacity, Box, SystemComponents, MoveStats.Rosen[i].Deletion, Constants);
      printf("(Deletion) Averaged Rosenbluth Weight: %.10f\n", AverageWr);
      printf("(Deletion) Averaged Excess Mu: %.10f\n", AverageMu);
      printf("(Deletion) Converted to Fugacity: %.10f [Pascal], Temp: %.5f [K]\n", Fugacity, Box.Temperature);
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

void Print_Swap_Statistics(Move_Statistics MoveStats)
{
  printf("=====================SWAP MOVES=====================\n");
  printf("Insertion Performed:   %zu\n", MoveStats.InsertionTotal);
  printf("Insertion Accepted:    %zu\n", MoveStats.InsertionAccepted);
  printf("Deletion Performed:    %zu\n", MoveStats.DeletionTotal);
  printf("Deletion Accepted:     %zu\n", MoveStats.DeletionAccepted);
  printf("Reinsertion Performed: %zu\n", MoveStats.ReinsertionTotal);
  printf("Reinsertion Accepted:  %zu\n", MoveStats.ReinsertionAccepted);
  printf("==============================================================\n");
}

void Print_CBCF_Statistics(Move_Statistics MoveStats)
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

void Gather_Averages(std::vector<double2>& Array, double init_energy, double running_energy, int Cycles, int Blocksize, size_t Nblock)
{
  //Determine the block id//
  size_t blockID = Cycles/Blocksize;
  if(blockID >= Nblock) blockID --;
  //Get total energy//
  double total_energy = init_energy + running_energy;
  Array[blockID].x += total_energy;
  Array[blockID].y += total_energy * total_energy;
}

void Print_Values(std::vector<double2>& Array, int Cycles, int Blocksize, size_t Nblock)
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

void Print_Averages(Components& SystemComponents, int Cycles, int Blocksize)
{
  printf("=====================BLOCK AVERAGES (ENERGIES)================\n");
  std::vector<double2>Temp = SystemComponents.EnergyAverage;
  Print_Values(Temp, Cycles, Blocksize, SystemComponents.Nblock);
  printf("==============================================================\n");
  printf("=====================BLOCK AVERAGES (# MOLECULES)=============\n");
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    printf("COMPONENT [%zu] (%s)\n", i, SystemComponents.MoleculeName[i].c_str());
    std::vector<double2>Temp = SystemComponents.Moves[i].MolAverage;
    Print_Values(Temp, Cycles, Blocksize, SystemComponents.Nblock);
    printf("----------------------------------------------------------\n");
  }
  printf("==============================================================\n");
}

void Gather_Averages_Types(std::vector<double2>& Array, double init_value, double running_value, int Cycles, int Blocksize, size_t Nblock)
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
void PrintAllStatistics(Components& SystemComponents, Simulations& Sims, size_t Cycles, int SimulationMode, double running_energy, double init_energy, size_t BlockAverageSize)
{
  for(size_t comp = SystemComponents.NumberOfFrameworks; comp < SystemComponents.Total_Components; comp++)
  {
    Print_Translation_Statistics(SystemComponents.Moves[comp], Sims.MaxTranslation);
    Print_Rotation_Statistics(SystemComponents.Moves[comp], Sims.MaxRotation);
    Print_Swap_Statistics(SystemComponents.Moves[comp]);
    if(SystemComponents.hasfractionalMolecule[comp]) Print_CBCF_Statistics(SystemComponents.Moves[comp]);
    printf("running total: %.10f, DeltaVDW+Real: %.5f, DeltaEwald: %.5f\n", running_energy + init_energy, SystemComponents.deltaVDWReal, SystemComponents.deltaEwald);
  }
  if(SimulationMode == PRODUCTION)
  {
    Print_Averages(SystemComponents, Cycles, BlockAverageSize);
  }
}

void PrintGibbs(Gibbs& GibbsStatistics)
{
  printf("=====================GIBBS MONTE CARLO STATISTICS=====================\n");
  printf("GIBBS VOLUME MOVE ATTEMPTS: %zu\n", (size_t) GibbsStatistics.GibbsBoxStats.x);
  printf("GIBBS VOLUME MOVE ACCEPTED: %zu\n", (size_t) GibbsStatistics.GibbsBoxStats.y);
  printf("GIBBS PARTICLE TRANSFER MOVE ATTEMPTS: %zu\n", (size_t) GibbsStatistics.GibbsXferStats.x);
  printf("GIBBS PARTICLE TRANSFER MOVE ACCEPTED: %zu\n", (size_t) GibbsStatistics.GibbsXferStats.y);
  printf("======================================================================\n");
}
