void Print_Translation_Statistics(Move_Statistics MoveStats, double3 MaxTranslation);

void Print_Rotation_Statistics(Move_Statistics MoveStats, double3 MaxRotation);

void Print_Swap_Statistics(WidomStruct Widom, Move_Statistics MoveStats);

void Print_CBCF_Statistics(Move_Statistics MoveStats);

void Gather_Averages(std::vector<double2>& Array, double init_energy, double running_energy, int Cycles, int Blocksize, size_t Nblock);

void Print_Averages(Components& SystemComponents, int Cycles, int Blocksize);
/*
template <typename T>
void Gather_Averages_Types(std::vector<T>& Array, double init_value, double running_value, int Cycles, int Blocksize)
{
  //Determine the block id//
  size_t blockID = Cycles/Blocksize;
  if(blockID >= SystemComponents.Nblock) blockID --;
  //Get total energy//
  double total_value = init_value + running_value;
  Array[blockID].x += total_value;
  Array[blockID].y += total_value * total_value;
}
*/
void Gather_Averages_Types(std::vector<double2>& Array, double init_value, double running_value, int Cycles, int Blocksize, size_t Nblock);

void PrintAllStatistics(Components& SystemComponents, Simulations& Sims, size_t Cycles, int SimulationMode, double running_energy, double init_energy, size_t BlockAverageSize);

void PrintGibbs(Gibbs& GibbsStatistics);

void Print_Widom_Statistics(Components& SystemComponents, Boxsize Box, Units& Constants, size_t comp);
