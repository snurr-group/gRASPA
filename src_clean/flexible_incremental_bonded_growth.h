#ifndef FLEXIBLE_INCREMENTAL_BONDED_GROWTH_H
#define FLEXIBLE_INCREMENTAL_BONDED_GROWTH_H

#include "flexible_molecule_topology.h"

#include <cstddef>
#include <optional>
#include <vector>

enum class FlexibleRASPA2StageOrientationPolicy
{
  SimpleSphere,
  SmallMCWithInnerRotation,
  SmallMCWithoutInnerRotation
};

// Immutable plan for the bonded subset of a RASPA2 CBMC growth stage after
// the caller has selected its current bead and atoms to place. Entries are
// identified by topology-table index so duplicate interactions remain
// distinct energy terms.
struct FlexibleIncrementalBondedStagePlan
{
  size_t current_bead = 0;
  std::vector<size_t> atoms_to_place;
  std::vector<size_t> positioned_bonded_neighbors;
  std::optional<size_t> raspa2_previous_bead;
  std::vector<size_t> bond_indices;
  std::vector<size_t> bend_indices;
  std::vector<size_t> torsion_indices;
  FlexibleRASPA2StageOrientationPolicy orientation_policy =
      FlexibleRASPA2StageOrientationPolicy::SimpleSphere;

  bool uses_simple_sphere() const
  {
    return orientation_policy ==
           FlexibleRASPA2StageOrientationPolicy::SimpleSphere;
  }

  bool runs_small_mc() const
  {
    return orientation_policy !=
           FlexibleRASPA2StageOrientationPolicy::SimpleSphere;
  }

  // RASPA2 executes the K-way inner rotation whenever exactly one bonded
  // previous bead exists, even when the stage has no torsion term.
  bool runs_inner_rotation_trials() const
  {
    return orientation_policy ==
           FlexibleRASPA2StageOrientationPolicy::SmallMCWithInnerRotation;
  }
};

FlexibleIncrementalBondedStagePlan
BuildFlexibleIncrementalBondedStagePlan(
    const MoleculeTopologyDefinition& topology,
    const std::vector<bool>& positioned_before,
    size_t current_bead,
    const std::vector<size_t>& atoms_to_place);

#endif // FLEXIBLE_INCREMENTAL_BONDED_GROWTH_H
