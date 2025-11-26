#ifndef READ_DATA_H
#define READ_DATA_H

#include <cuda_runtime.h>
//#include "VDW_Coulomb.cuh"

void Check_Inputs_In_read_data_cpp(std::string& exepath);

void read_number_of_sims_from_input(size_t *NumSims, bool *SingleSim);

void read_FFParams_from_input(Input_Container& Input);

void read_Gibbs_and_Cycle_Stats(Variables& Vars, bool& SetMaxStep, size_t& MaxStepPerCycle);

void read_simulation_input(Variables& Vars, bool *ReadRestart, bool *SameFrameworkEverySimulation);

void ReadFramework(Boxsize& Box, PseudoAtomDefinitions& PseudoAtom, size_t FrameworkIndex, Components& SystemComponents);

void ReadFrameworkComponentMoves(Move_Statistics& MoveStats, Components& SystemComponents, size_t comp);

//void POSCARParser(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom);

void ForceFieldParser(Input_Container& Input, PseudoAtomDefinitions& PseudoAtom);
void ForceField_Processing(Input_Container& Input); //Processes LJ, shift, Tail//
void Copy_InputLoader_Data(Variables& Vars);     //Copy LJ, shift, Tail from Input loader to data//


void PseudoAtomParser(PseudoAtomDefinitions& PseudoAtom);
void PseudoAtomProcessing(Variables& Vars);


void MoleculeDefinitionParser(Atoms& Mol, Components& SystemComponents, std::string MolName, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space);

void read_component_values_from_simulation_input(Variables& Vars, Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space, size_t BoxIndex);

void ReadRestartInputFileType(Components& SystemComponents);

void LMPDataFileParser(Boxsize& Box, Components& SystemComponents);

void RestartFileParser(Boxsize& Box, Components& SystemComponents);

void read_Ewald_Parameters_from_input(double CutOffCoul, Boxsize& Box, double precision);

void OverWrite_Mixing_Rule(Input_Container& Input);

void OverWriteTailCorrection(Input_Container& Input);

void read_movies_stats_print(Components& SystemComponents, size_t sim);

std::vector<double2> ReadMinMax();

void ReadVoidFraction(Variables& Vars);

void ReadDNNModelSetup(Components& SystemComponents);

void ReadBlockPockets(Components& SystemComponents, size_t component, const std::string& filename);
void ReplicateBlockPockets(Components& SystemComponents, size_t component, Boxsize& Box);
// CheckBlockedPosition moved to mc_utilities.h as it's a runtime MC function
//###PATCH_LCLIN_READDATA_H###//
//###PATCH_ALLEGRO_READDATA_H###//

//Weird issues with using vector.data() for double and double3//
//So we keep this function, for now//
template<typename T>
inline T* convert1DVectortoArray(std::vector<T>& Vector)
{
  size_t Vectorsize = Vector.size();
  T* result=new T[Vectorsize];
  T* walkarr=result;
  std::copy(Vector.begin(), Vector.end(), walkarr);
  //printf("done convert Mol Type, Origin: %zu, copied: %zu\n", MoleculeTypeArray[0], result[0]);
  return result;
}

//void write_ReplicaPos(auto& pos, auto& ij2type, size_t ntotal, size_t nstep);

//void write_edges(auto& edges, auto& ij2type, size_t nedges, size_t nstep);

#endif // READ_DATA_H
