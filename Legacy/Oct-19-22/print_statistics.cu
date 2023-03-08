#include "data_struct.h"

void Print_Translation_Statistics(Move_Statistics MoveStats, ForceField device_FF)
{
  printf("=====================TRANSLATION MOVES=====================\n");
  printf("Translation Performed: %zu\n", MoveStats.TranslationTotal);
  printf("Translation Accepted: %zu\n", MoveStats.TranslationAccepted);
  double Host_Max[3];
  cudaMemcpy(Host_Max, device_FF.MaxTranslation, 3*sizeof(double), cudaMemcpyDeviceToHost);
  printf("Max Translation: %.10f, %.10f, %.10f\n", Host_Max[0], Host_Max[1], Host_Max[2]);
  printf("===========================================================\n");
}

void Print_Widom_Statistics(WidomStruct Widom, Move_Statistics MoveStats, double Beta, double energy_to_kelvin)
{
  printf("=====================WIDOM MOVES=====================\n");
  for(size_t i = 0; i < MoveStats.NumberOfBlocks; i++)
  {
    printf("=====BLOCK %zu=====\n");
    printf("Widom Performed: %zu\n", Widom.RosenbluthCount[i]);
    if(Widom.RosenbluthCount[i])
    {
      printf("Averaged Rosenbluth Weight: %.10f\n", Widom.Rosenbluth[i]/static_cast<double>(Widom.RosenbluthCount[i]));
      printf("Averaged Excess Mu: %.10f\n", energy_to_kelvin*-(1.0/Beta)*std::log(Widom.Rosenbluth[i]/static_cast<double>(Widom.RosenbluthCount[i])));
    }
  }
  printf("===================================================\n");
}

void Print_Swap_Statistics(Move_Statistics MoveStats)
{
  printf("=====================SWAP MOVES For System 0=====================\n"); //Zhao's note: change after using multiple simulations//
  printf("Insertion Performed: %zu\n",   MoveStats.InsertionTotal);
  printf("Insertion Accepted: %zu\n",    MoveStats.InsertionAccepted);
  printf("Deletion Performed: %zu\n",    MoveStats.DeletionTotal);
  printf("Deletion Accepted: %zu\n",     MoveStats.DeletionAccepted);
  printf("Reinsertion Performed: %zu\n", MoveStats.ReinsertionTotal);
  printf("Reinsertion Accepted: %zu\n",  MoveStats.ReinsertionAccepted);
  printf("====================================================\n");
}
