#include "ewald_kernel.h"

double CPU_GPU_EwaldTotalEnergy(Boxsize& Box, Boxsize& device_Box, Atoms* System, Atoms* d_a, ForceField FF, ForceField device_FF, Components& SystemComponents)
{
  ///////////////////
  // Run CPU Ewald //
  ///////////////////
  double start = omp_get_wtime();
  double ewaldEnergy = Ewald_Total(Box, System, FF, SystemComponents);
  double end = omp_get_wtime(); double CPU_ewald_time = end-start;
  printf("HostEwald took %.5f sec\n", CPU_ewald_time);
  return ewaldEnergy;
}

void Calculate_Exclusion_Energy_Rigid(Boxsize& Box, Atoms* System, ForceField FF, Components& SystemComponents)
{
  for(size_t i = 0; i < SystemComponents.Total_Components; i++)
  {
    double IntraE = 0.0; double SelfE = 0.0;
    if(SystemComponents.rigid[i]) //Only Calculate this when the component is rigid//
    {
      IntraE = Calculate_Intra_Molecule_Exclusion(Box, System, Box.Alpha, Box.Prefactor, SystemComponents, i);
      SelfE  = Calculate_Self_Exclusion(Box, System, Box.Alpha, Box.Prefactor, SystemComponents, i);
    }
    SystemComponents.ExclusionIntra.push_back(IntraE);
    SystemComponents.ExclusionAtom.push_back(SelfE);
    printf("DEBUG: comp: %zu, IntraE: %.5f, SelfE: %.5f\n", i, SystemComponents.ExclusionIntra[i], SystemComponents.ExclusionAtom[i]);
  }
}
