void check_restart_file();

void read_framework_atoms_from_restart_SoA(size_t *value, size_t *Allocate_value, double **Framework_x, double **Framework_y, double **Framework_z, double **Framework_scale, double **Framework_charge, double **Framework_scaleCoul, size_t **Framework_Type, size_t **MolID);

void read_adsorbate_atoms_from_restart_SoA(size_t Component, size_t *value, size_t *Allocate_value, double **x, double **y, double **z, double **scale, double **charge, double **scaleCoul, size_t **Type, size_t **MolID);

void read_force_field_from_restart_SoA(size_t *value, double **Epsilon, double **Sigma, double **Z, double **Shifted, int **Type, bool shift);

double* read_Cell_from_restart(size_t skip);

void Read_Cell_wrapper(Boxsize& Box);

double* read_FFParams_from_restart();

void read_simulation_input(bool *UseGPUReduction, bool *Useflag, bool *noCharges, int *Cycles, size_t *Widom_Trial, size_t *NumberOfBlocks, double *Pressure, double *Temperature, bool *DualPrecision);

void POSCARParser(Boxsize& Box, Atoms& Framework, PseudoAtomDefinitions& PseudoAtom);

void ForceFieldParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

void PseudoAtomParser(ForceField& FF, PseudoAtomDefinitions& PseudoAtom);

void MoleculeDefinitionParser(Atoms& Mol, Components& SystemComponents, std::string MolName, PseudoAtomDefinitions PseudoAtom);

void read_component_values_from_simulation_input(Components& SystemComponents, Move_Statistics& MoveStats, size_t AdsorbateComponent, Atoms& Mol, PseudoAtomDefinitions PseudoAtom);
