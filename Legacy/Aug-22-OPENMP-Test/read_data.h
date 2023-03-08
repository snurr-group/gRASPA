void check_restart_file();

double* read_framework_atoms_from_restart(size_t *value);

size_t* read_framework_atoms_types_from_restart();

double* read_force_field_from_restart(size_t *value);

int* read_force_field_type_from_restart();

double* read_Cell_from_restart(size_t skip);

double* read_FFParams_from_restart();
