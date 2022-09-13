#include <stdio.h>
#include <cmath>
#include <array>
using namespace std;

//std::optional<RunningEnergy> computeFrameworkMoleculeEnergyDifferenceGPU(System &system, std::span<const Atom> newatoms, std::span<const Atom> oldatoms);
//std::optional<RunningEnergy> computeFrameworkMoleculeEnergyDifferenceGPU(System &system, std::vector<std::vector<double>>& CellSize, std::vector<std::vector<double>>& InverseCellSize, int Boxtype, std::vector<std::vector<std::vector<double>>>& FFParameter, std::vector<std::vector<int>>& FFType, std::span<const Atom> newatoms, std::span<const Atom> oldatoms);

//std::optional<RunningEnergy> computeFrameworkMoleculeEnergyDifferenceGPU(System &system, std::vector<std::vector<double>>& CellSize, std::vector<std::vector<double>>& InverseCellSize, int Boxtype, std::vector<std::vector<double>>& FrameworkArray, std::vector<int>& FrameworkTypeArray, std::vector<std::vector<std::vector<double>>>& FFParameter, std::vector<std::vector<int>>& FFType, std::span<const Atom> newatoms, std::span<const Atom> oldatoms);

//std::optional<RunningEnergy> computeFrameworkMoleculeEnergyDifferenceGPU(System &system, std::vector<std::vector<double>>& CellSize, std::vector<std::vector<double>>& InverseCellSize, int Boxtype, std::vector<std::vector<double>>& FrameworkArray, std::vector<int>& FrameworkTypeArray, std::vector<std::vector<std::vector<double>>>& FFParameter, std::vector<std::vector<int>>& FFType, std::vector<std::vector<double>>& MoleculeArray, std::vector<size_t>& MoleculeTypeArray, std::vector<std::vector<double>>& NewMoleculeArray, std::vector<size_t>& NewMoleculeTypeArray);

//std::vector<double> computeFrameworkMoleculeEnergyDifferenceGPU(std::vector<std::vector<double>>& CellSize, std::vector<std::vector<double>>& InverseCellSize, int Boxtype, std::vector<std::vector<double>>& FrameworkArray, size_t FrameworkTypeArray[], bool noCharges, std::vector<double>& FFCutOffParameters, std::vector<std::vector<std::vector<double>>>& FFParameter, std::vector<std::vector<int>>& FFType, double alpha, int ChargeMethod, std::vector<std::vector<double>>& MoleculeArray, size_t MoleculeTypeArray[], std::vector<std::vector<double>>& NewMoleculeArray, size_t NewMoleculeTypeArray[]);

std::array<double,3> computeFrameworkMoleculeEnergyDifferenceGPU(double* CellArray, double* InverseCellArray, double* FrameworkArray, size_t* FrameworkTypeArray, double* FFArray, int* FFTypeArray, double* MolArray, size_t* MolTypeArray, double* NewMolArray, double* FFParams, int* OtherParams, size_t Frameworksize, size_t Molsize, size_t FFsize, bool noCharges);
