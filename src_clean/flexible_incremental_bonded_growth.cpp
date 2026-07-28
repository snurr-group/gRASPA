#include "flexible_incremental_bonded_growth.h"

#include <stdexcept>
#include <string>

namespace
{
void ValidateTopologyEntries(
    const std::vector<MoleculeTopologyEntry>& entries,
    size_t expected_arity,
    size_t atom_count,
    const std::string& section)
{
  for(size_t entry_index = 0; entry_index < entries.size(); entry_index++)
  {
    const MoleculeTopologyEntry& entry = entries[entry_index];
    if(entry.atoms.size() != expected_arity)
    {
      throw std::runtime_error(
          "Flexible bonded growth " + section + " entry " +
          std::to_string(entry_index) + " has invalid arity");
    }
    for(size_t atom : entry.atoms)
    {
      if(atom >= atom_count)
      {
        throw std::runtime_error(
            "Flexible bonded growth " + section + " entry " +
            std::to_string(entry_index) +
            " contains an out-of-range atom");
      }
    }
  }
}

bool BondConnects(
    const MoleculeTopologyEntry& bond,
    size_t left,
    size_t right)
{
  return
      (bond.atoms[0] == left && bond.atoms[1] == right) ||
      (bond.atoms[0] == right && bond.atoms[1] == left);
}

std::vector<size_t> OwnedEntryIndices(
    const std::vector<MoleculeTopologyEntry>& entries,
    const std::vector<bool>& positioned_after,
    const std::vector<bool>& to_place)
{
  std::vector<size_t> owned;
  for(size_t entry_index = 0; entry_index < entries.size(); entry_index++)
  {
    bool complete = true;
    bool touches_stage = false;
    for(size_t atom : entries[entry_index].atoms)
    {
      complete = complete && positioned_after[atom];
      touches_stage = touches_stage || to_place[atom];
    }
    if(complete && touches_stage) owned.push_back(entry_index);
  }
  return owned;
}
} // namespace

FlexibleIncrementalBondedStagePlan
BuildFlexibleIncrementalBondedStagePlan(
    const MoleculeTopologyDefinition& topology,
    const std::vector<bool>& positioned_before,
    size_t current_bead,
    const std::vector<size_t>& atoms_to_place)
{
  const size_t atom_count = topology.number_of_atoms;
  if(atom_count == 0)
    throw std::runtime_error(
        "Flexible bonded growth requires a non-empty topology");
  if(positioned_before.size() != atom_count)
    throw std::runtime_error(
        "Flexible bonded growth positioned mask has the wrong size");
  if(current_bead >= atom_count)
    throw std::runtime_error(
        "Flexible bonded growth current bead is out of range");
  if(!positioned_before[current_bead])
    throw std::runtime_error(
        "Flexible bonded growth current bead is not positioned");
  if(atoms_to_place.empty())
    throw std::runtime_error(
        "Flexible bonded growth stage has no atoms to place");

  ValidateTopologyEntries(topology.bonds, 2, atom_count, "bond");
  ValidateTopologyEntries(topology.bends, 3, atom_count, "bend");
  ValidateTopologyEntries(topology.torsions, 4, atom_count, "torsion");

  std::vector<bool> to_place(atom_count, false);
  for(size_t atom : atoms_to_place)
  {
    if(atom >= atom_count)
      throw std::runtime_error(
          "Flexible bonded growth atom to place is out of range");
    if(atom == current_bead)
      throw std::runtime_error(
          "Flexible bonded growth cannot place its current bead");
    if(to_place[atom])
      throw std::runtime_error(
          "Flexible bonded growth atom-to-place list contains a duplicate");
    if(positioned_before[atom])
      throw std::runtime_error(
          "Flexible bonded growth atom to place is already positioned");
    to_place[atom] = true;
  }

  std::vector<bool> current_neighbors(atom_count, false);
  bool stage_is_attached = false;
  for(const MoleculeTopologyEntry& bond : topology.bonds)
  {
    if(bond.atoms[0] == current_bead)
      current_neighbors[bond.atoms[1]] = true;
    if(bond.atoms[1] == current_bead)
      current_neighbors[bond.atoms[0]] = true;
  }
  for(size_t atom : atoms_to_place)
    stage_is_attached = stage_is_attached || current_neighbors[atom];
  if(!stage_is_attached)
    throw std::runtime_error(
        "Flexible bonded growth stage is not attached to its current bead");

  FlexibleIncrementalBondedStagePlan plan;
  plan.current_bead = current_bead;
  plan.atoms_to_place = atoms_to_place;

  // SetGrowingStatus scans atom indices in ascending order and leaves
  // PreviousBead equal to the final positioned bonded neighbor encountered.
  for(size_t atom = 0; atom < atom_count; atom++)
  {
    if(atom != current_bead && positioned_before[atom] &&
       current_neighbors[atom])
    {
      plan.positioned_bonded_neighbors.push_back(atom);
    }
  }
  if(!plan.positioned_bonded_neighbors.empty())
    plan.raspa2_previous_bead =
        plan.positioned_bonded_neighbors.back();

  if(plan.positioned_bonded_neighbors.empty())
  {
    plan.orientation_policy =
        FlexibleRASPA2StageOrientationPolicy::SimpleSphere;
  }
  else if(plan.positioned_bonded_neighbors.size() == 1)
  {
    plan.orientation_policy =
        FlexibleRASPA2StageOrientationPolicy::SmallMCWithInnerRotation;
  }
  else
  {
    plan.orientation_policy =
        FlexibleRASPA2StageOrientationPolicy::SmallMCWithoutInnerRotation;
  }

  // Match Interactions(): atoms-to-place order is outermost for bonds. Only
  // current-to-new bonds are listed; RIGID_BOND is excluded and FIXED_BOND is
  // retained for later small-MC handling.
  for(size_t atom : atoms_to_place)
  {
    for(size_t bond_index = 0; bond_index < topology.bonds.size();
        bond_index++)
    {
      const MoleculeTopologyEntry& bond = topology.bonds[bond_index];
      if(bond.potential != "RIGID_BOND" &&
         BondConnects(bond, current_bead, atom))
      {
        plan.bond_indices.push_back(bond_index);
      }
    }
  }

  std::vector<bool> positioned_after = positioned_before;
  for(size_t atom : atoms_to_place) positioned_after[atom] = true;
  plan.bend_indices =
      OwnedEntryIndices(topology.bends, positioned_after, to_place);
  plan.torsion_indices =
      OwnedEntryIndices(topology.torsions, positioned_after, to_place);
  return plan;
}
