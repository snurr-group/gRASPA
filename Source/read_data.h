//#include "VDW_Coulomb.cuh"

void read_number_of_sims_from_input(size_t *NumSims, bool *SingleSim);

void read_FFParams_from_input(ForceField& FF, double& precision);

void read_Gibbs_Stats(Gibbs& GibbsStatistics, bool& SetMaxStep, size_t& MaxStepPerCycle);

void read_simulation_input(bool *UseGPUReduction, bool *Useflag, bool *noCharges, int *InitializationCycles, int *EquilibrationCycles, int *ProductionCycles, size_t *Widom_Trial, size_t *Widom_Orientation, size_t *NumberOfBlocks, double *Pressure, double *Temperature, size_t *AllocateSize, bool *ReadRestart, int *RANDOMSEED, bool *SameFrameworkEverySimulation, size_t& NumberOfComponents);

void ReadFramework(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom, size_t FrameworkIndex, Components& SystemComponents);

//void POSCARParser(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom);

void ForceFieldParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

void PseudoAtomParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

void MoleculeDefinitionParser(Atoms& Mol, Components& SystemComponents, std::string MolName, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space);

void read_component_values_from_simulation_input(Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space);

void RestartFileParser(Simulations& Sims, Atoms* Host_System, Components& SystemComponents);

void read_Ewald_Parameters_from_input(double CutOffCoul, Boxsize& Box, double precision);

void OverWriteFFTerms(Components& SystemComponents, ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

std::vector<double2> ReadMinMax();
void ReadDNNModelSetup(Components& SystemComponents);

//Weird issues with using vector.data() for double and double3//
//So we keep this function, for now//
template<typename T>
T* convert1DVectortoArray(std::vector<T>& Vector)
{
  size_t Vectorsize = Vector.size();
  T* result=new T[Vectorsize];
  T* walkarr=result;
  std::copy(Vector.begin(), Vector.end(), walkarr);
  //printf("done convert Mol Type, Origin: %zu, copied: %zu\n", MoleculeTypeArray[0], result[0]);
  return result;
}

