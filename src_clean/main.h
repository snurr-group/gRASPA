Variables Initialize(void);
void RunSimulation(Variables& Vars);
void EndOfSimulationWrapUp(Variables& Vars);

MoveEnergy check_energy_wrapper(Variables& Var, size_t i);
void ENERGY_SUMMARY(Variables& Vars);
