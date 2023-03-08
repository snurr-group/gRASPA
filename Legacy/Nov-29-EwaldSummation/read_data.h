void read_number_of_sims_from_input(size_t *NumSims, bool *SingleSim);

void Read_Cell_wrapper(Boxsize& Box);

void read_FFParams_from_input(ForceField& FF, Boxsize& Box);

void read_simulation_input(bool *UseGPUReduction, bool *Useflag, bool *noCharges, int *Cycles, size_t *Widom_Trial, size_t *Widom_Orientation, size_t *NumberOfBlocks, double *Pressure, double *Temperature, bool *DualPrecision, size_t *AllocateSize);

void POSCARParser(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom);

void ForceFieldParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

void PseudoAtomParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

void MoleculeDefinitionParser(Atoms& Mol, Components& SystemComponents, std::string MolName, PseudoAtomDefinitions PseudoAtom, size_t Allocate_space);

//void read_component_values_from_simulation_input(Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom);
void read_component_values_from_simulation_input(Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom, size_t* MoleculeNeedToCreate, size_t Allocate_space);
