void check_restart_file();

void read_framework_atoms_from_restart_SoA(size_t *value, size_t *Allocate_value, double **Framework_x, double **Framework_y, double **Framework_z, double **Framework_scale, double **Framework_charge, double **Framework_scaleCoul, size_t **Framework_Type, size_t **MolID);

void read_adsorbate_atoms_from_restart_SoA(size_t Component, size_t *value, size_t *Allocate_value, double **x, double **y, double **z, double **scale, double **charge, double **scaleCoul, size_t **Type, size_t **MolID);

void read_force_field_from_restart_SoA(size_t *value, double **Epsilon, double **Sigma, double **Z, double **Shifted, int **Type, bool shift);

double* read_Cell_from_restart(size_t skip);
