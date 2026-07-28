#include "flexible_incremental_bonded_growth.h"

#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void Require(bool condition, const std::string& message)
{
  if(!condition) throw std::runtime_error(message);
}

void RequireIndices(
    const std::vector<size_t>& actual,
    const std::vector<size_t>& expected,
    const std::string& message)
{
  if(actual != expected) throw std::runtime_error(message);
}

void RequireThrows(
    const std::function<void()>& action,
    const std::string& expected)
{
  try
  {
    action();
  }
  catch(const std::exception& error)
  {
    Require(
        std::string(error.what()).find(expected) != std::string::npos,
        "unexpected error: " + std::string(error.what()));
    return;
  }
  throw std::runtime_error("expected failure containing: " + expected);
}

MoleculeTopologyEntry Entry(
    std::initializer_list<size_t> atoms,
    const std::string& potential)
{
  return {std::vector<size_t>(atoms), potential, {}};
}

MoleculeTopologyDefinition RepresentativeChain()
{
  MoleculeTopologyDefinition topology;
  topology.name = "branched-head-linear-tail";
  topology.number_of_atoms = 11;
  topology.bonds = {
      Entry({0, 1}, "FIXED_BOND"),
      Entry({1, 2}, "FIXED_BOND"),
      Entry({2, 3}, "FIXED_BOND"),
      Entry({2, 4}, "FIXED_BOND"),
      Entry({4, 5}, "FIXED_BOND"),
      Entry({5, 6}, "FIXED_BOND"),
      Entry({6, 7}, "FIXED_BOND"),
      Entry({7, 8}, "FIXED_BOND"),
      Entry({8, 9}, "FIXED_BOND"),
      Entry({9, 10}, "FIXED_BOND")};
  topology.bends = {
      Entry({2, 4, 5}, "HARMONIC_BEND"),
      Entry({3, 2, 4}, "HARMONIC_BEND"),
      Entry({1, 2, 4}, "HARMONIC_BEND"),
      Entry({4, 5, 6}, "HARMONIC_BEND"),
      Entry({5, 6, 7}, "HARMONIC_BEND"),
      Entry({6, 7, 8}, "HARMONIC_BEND"),
      Entry({7, 8, 9}, "HARMONIC_BEND"),
      Entry({8, 9, 10}, "HARMONIC_BEND")};
  topology.torsions = {
      Entry({2, 4, 5, 6}, "FOURIER_SERIES_DIHEDRAL"),
      Entry({4, 5, 6, 7}, "FOURIER_SERIES_DIHEDRAL"),
      Entry({5, 6, 7, 8}, "FOURIER_SERIES_DIHEDRAL"),
      Entry({6, 7, 8, 9}, "FOURIER_SERIES_DIHEDRAL"),
      Entry({7, 8, 9, 10}, "FOURIER_SERIES_DIHEDRAL"),
      Entry({3, 2, 4, 5}, "TRAPPE_DIHEDRAL"),
      Entry({1, 2, 4, 5}, "TRAPPE_DIHEDRAL"),
      Entry({0, 1, 2, 4}, "TRAPPE_DIHEDRAL")};
  return topology;
}

void Mark(std::vector<bool>& positioned, size_t atom)
{
  Require(!positioned.at(atom), "test attempted to position an atom twice");
  positioned[atom] = true;
}

void TestInitialAndForwardStages()
{
  const MoleculeTopologyDefinition topology = RepresentativeChain();
  std::vector<bool> positioned(11, false);
  positioned[0] = true;
  const FlexibleIncrementalBondedStagePlan initial =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 0, {1, 2, 3});
  RequireIndices(initial.bond_indices, {0}, "initial rigid-head bonds");
  RequireIndices(initial.bend_indices, {}, "initial rigid-head bends");
  RequireIndices(initial.torsion_indices, {}, "initial rigid-head torsions");
  Require(initial.uses_simple_sphere(), "initial rigid-head policy");

  for(size_t atom = 1; atom < 4; atom++) positioned[atom] = true;
  const FlexibleIncrementalBondedStagePlan place4 =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 2, {4});
  RequireIndices(place4.bond_indices, {3}, "forward atom4 bond ownership");
  RequireIndices(place4.bend_indices, {1, 2}, "forward atom4 bend ownership");
  RequireIndices(place4.torsion_indices, {7}, "forward atom4 torsion ownership");
  RequireIndices(
      place4.positioned_bonded_neighbors,
      {1, 3},
      "forward atom4 previous neighbors");
  Require(
      place4.raspa2_previous_bead == 3,
      "forward atom4 RASPA2 PreviousBead");
  Require(
      !place4.runs_inner_rotation_trials(),
      "two-neighbor stage incorrectly enabled inner rotation");

  Mark(positioned, 4);
  const FlexibleIncrementalBondedStagePlan place5 =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 4, {5});
  RequireIndices(place5.bond_indices, {4}, "forward atom5 bond ownership");
  RequireIndices(place5.bend_indices, {0}, "forward atom5 bend ownership");
  RequireIndices(place5.torsion_indices, {5, 6}, "forward atom5 torsion ownership");
  Require(
      place5.runs_inner_rotation_trials(),
      "one-neighbor stage did not enable inner rotation");

  Mark(positioned, 5);
  for(size_t atom = 6; atom <= 10; atom++)
  {
    const size_t offset = atom - 6;
    const FlexibleIncrementalBondedStagePlan plan =
        BuildFlexibleIncrementalBondedStagePlan(
            topology, positioned, atom - 1, {atom});
    RequireIndices(plan.bond_indices, {atom - 1}, "forward tail bond");
    RequireIndices(plan.bend_indices, {3 + offset}, "forward tail bend");
    RequireIndices(plan.torsion_indices, {offset}, "forward tail torsion");
    Require(plan.runs_inner_rotation_trials(), "forward tail policy");
    Mark(positioned, atom);
  }
}

void TestReverseAndGroupedStages()
{
  const MoleculeTopologyDefinition topology = RepresentativeChain();
  std::vector<bool> positioned(11, false);
  positioned[10] = true;

  const FlexibleIncrementalBondedStagePlan place9 =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 10, {9});
  RequireIndices(place9.bond_indices, {9}, "reverse atom9 bond ownership");
  Require(place9.uses_simple_sphere(), "reverse atom9 policy");
  Mark(positioned, 9);

  const FlexibleIncrementalBondedStagePlan place8 =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 9, {8});
  RequireIndices(place8.bend_indices, {7}, "reverse atom8 bend ownership");
  RequireIndices(place8.torsion_indices, {}, "reverse atom8 torsion ownership");
  Require(
      place8.runs_inner_rotation_trials(),
      "zero-torsion stage did not retain K-way rotation");
  Mark(positioned, 8);

  const std::vector<size_t> atoms = {7, 6, 5, 4};
  for(size_t index = 0; index < atoms.size(); index++)
  {
    const size_t atom = atoms[index];
    const FlexibleIncrementalBondedStagePlan plan =
        BuildFlexibleIncrementalBondedStagePlan(
            topology, positioned, atom + 1, {atom});
    RequireIndices(plan.bend_indices, {6 - index}, "reverse tail bend");
    RequireIndices(plan.torsion_indices, {4 - index}, "reverse tail torsion");
    Mark(positioned, atom);
  }

  const FlexibleIncrementalBondedStagePlan place2 =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 4, {2});
  RequireIndices(place2.torsion_indices, {0}, "reverse atom2 torsion");
  Mark(positioned, 2);

  const FlexibleIncrementalBondedStagePlan grouped =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 2, {0, 1, 3});
  RequireIndices(grouped.bond_indices, {1, 2}, "grouped-head bonds");
  RequireIndices(grouped.bend_indices, {1, 2}, "grouped-head bends");
  RequireIndices(grouped.torsion_indices, {5, 6, 7}, "grouped-head torsions");
  Require(grouped.raspa2_previous_bead == 4, "grouped-head PreviousBead");
}

void TestOrderingAndValidation()
{
  MoleculeTopologyDefinition topology = RepresentativeChain();
  topology.torsions.push_back(topology.torsions[5]);
  std::vector<bool> positioned(11, false);
  for(size_t atom = 0; atom <= 4; atom++) positioned[atom] = true;
  const FlexibleIncrementalBondedStagePlan duplicate =
      BuildFlexibleIncrementalBondedStagePlan(
          topology, positioned, 4, {5});
  RequireIndices(
      duplicate.torsion_indices,
      {5, 6, 8},
      "duplicate topology entry was collapsed");

  MoleculeTopologyDefinition ordered;
  ordered.number_of_atoms = 3;
  ordered.bonds = {
      Entry({0, 2}, "FIXED_BOND"),
      Entry({0, 1}, "FIXED_BOND")};
  const FlexibleIncrementalBondedStagePlan bond_order =
      BuildFlexibleIncrementalBondedStagePlan(
          ordered, {true, false, false}, 0, {1, 2});
  RequireIndices(
      bond_order.bond_indices,
      {1, 0},
      "atoms-to-place outer-loop order was not preserved");

  std::vector<bool> validation_positioned(11, false);
  for(size_t atom = 0; atom < 4; atom++)
    validation_positioned[atom] = true;
  RequireThrows(
      [&] {
        BuildFlexibleIncrementalBondedStagePlan(
            topology, std::vector<bool>(10, false), 2, {4});
      },
      "wrong size");
  RequireThrows(
      [&] {
        BuildFlexibleIncrementalBondedStagePlan(
            topology, validation_positioned, 2, {4, 4});
      },
      "duplicate");
  RequireThrows(
      [&] {
        BuildFlexibleIncrementalBondedStagePlan(
            topology, validation_positioned, 2, {1});
      },
      "already positioned");

  MoleculeTopologyDefinition invalid = topology;
  invalid.torsions.push_back(
      Entry({0, 1, 2, 11}, "TRAPPE_DIHEDRAL"));
  std::vector<bool> invalid_positioned(11, false);
  for(size_t atom = 0; atom < 4; atom++)
    invalid_positioned[atom] = true;
  RequireThrows(
      [&] {
        BuildFlexibleIncrementalBondedStagePlan(
            invalid, invalid_positioned, 2, {4});
      },
      "out-of-range atom");
}
} // namespace

int main()
{
  try
  {
    TestInitialAndForwardStages();
    TestReverseAndGroupedStages();
    TestOrderingAndValidation();
    std::cout << "Flexible incremental bonded-growth tests passed\n";
    return 0;
  }
  catch(const std::exception& error)
  {
    std::cerr << "Flexible incremental bonded-growth tests failed: "
              << error.what() << '\n';
    return 1;
  }
}
