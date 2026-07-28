#ifndef FLEXIBLE_MOLECULE_TOPOLOGY_H
#define FLEXIBLE_MOLECULE_TOPOLOGY_H

#include <cstddef>
#include <string>
#include <vector>

// Host-side topology retained from a RASPA molecule definition. Potential
// parameters use the units and ordering stored in the corresponding .def
// entry.
struct MoleculeTopologyEntry
{
  std::vector<size_t> atoms;
  std::string potential;
  std::vector<double> parameters;
};

struct MoleculeTopologyDefinition
{
  std::string name;
  size_t number_of_atoms = 0;
  std::vector<MoleculeTopologyEntry> bonds;
  std::vector<MoleculeTopologyEntry> bends;
  std::vector<MoleculeTopologyEntry> torsions;
};

#endif // FLEXIBLE_MOLECULE_TOPOLOGY_H
