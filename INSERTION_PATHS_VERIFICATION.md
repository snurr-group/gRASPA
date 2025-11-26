# Insertion Paths Verification

This document verifies that all insertion paths in gRASPA are properly checked for block pockets.

## All Insertion Paths in axpy.cu:

1. **SINGLE_INSERTION** (line 235-236)
   - Path: `SingleBodyMove(Vars, box_index)` → `SingleBody_Prepare()` → `SingleBody_Calculation()` → `SingleBody_Acceptance()`
   - **Blocking Check**: ✓ In `SingleBody_Prepare()` - checks ALL atoms
   - Location: `mc_single_particle.h:85-128`
   - Status: **COVERED**

2. **INSERTION (CBMC)** (line 229-230)
   - Path: `MOVES.INSERTION.Run()` → `Insertion_Body()` → `Widom_Move_FirstBead_PARTIAL()` → `Widom_Move_Chain_PARTIAL()` → `InsertionMove::Acceptance()`
   - **Blocking Check**: 
     - First bead: ✓ In `Widom_Move_FirstBead_PARTIAL()` - checks all trials
     - Final molecule: ✓ In `Insertion_Body()` - checks ALL atoms of selected molecule (line 34-86)
   - Location: `mc_swap_utilities.h:1-158`
   - Status: **COVERED**

3. **WIDOM** (line 172)
   - Path: `MOVES.INSERTION.WidomMove()` → `Insertion_Body()` (same as INSERTION)
   - **Blocking Check**: Same as INSERTION (uses `Insertion_Body`)
   - Location: `move_struct.h:75-80` → `mc_swap_utilities.h:1-158`
   - Status: **COVERED**

4. **REINSERTION** (line 197)
   - Path: `MOVES.REINSERTION.Run()` → `ReinsertionMove::Acceptance()`
   - **Blocking Check**: ✓ In `ReinsertionMove::Acceptance()` - checks ALL atoms
   - Location: `move_struct.h:ReinsertionMove::Acceptance()`
   - Status: **COVERED**

5. **CreateMolecule** (initialization, lines 338, 364)
   - Path: `MOVES.INSERTION.CreateMolecule()` → `Insertion_Body()` (same as INSERTION)
   - **Blocking Check**: Same as INSERTION (uses `Insertion_Body`)
   - Location: `move_struct.h:81-92` → `mc_swap_utilities.h:1-158`
   - Status: **COVERED** (though initialization may not need blocking)

## Blocking Check Details:

### 1. SINGLE_INSERTION, TRANSLATION, ROTATION
- **File**: `mc_single_particle.h`
- **Function**: `SingleBody_Prepare()`
- **Check**: ALL atoms of the molecule
- **Distance**: `dist_sq <= radius_sq * 1.01` (1% larger blocking zone)
- **Flag**: Sets `SystemComponents.TempVal.BlockedByPockets = true` and `Sims.device_flag[0] = true`
- **Rejection**: Explicit check in `SingleBody_Acceptance()` before Pacc

### 2. CBMC INSERTION (multi-atom molecules)
- **File**: `mc_swap_utilities.h`
- **Function**: `Insertion_Body()`
- **First Bead Check**: `Widom_Move_FirstBead_PARTIAL()` - checks all trials
- **Final Molecule Check**: After chain growth, checks ALL atoms of selected molecule
- **Distance**: `dist_sq <= radius_sq * 1.01` (1% larger blocking zone)
- **Rejection**: Sets `CBMC.SuccessConstruction = false` and `CBMC.Rosenbluth = 0.0`

### 3. CBMC INSERTION (single-atom molecules)
- **File**: `mc_swap_utilities.h`
- **Function**: `Insertion_Body()`
- **Check**: First bead position (already checked in `Widom_Move_FirstBead_PARTIAL()`)
- **Double-check**: Additional check in `Insertion_Body()` for consistency

### 4. REINSERTION
- **File**: `move_struct.h`
- **Function**: `ReinsertionMove::Acceptance()`
- **Check**: ALL atoms in `SystemComponents.tempMolStorage`
- **Distance**: `dist_sq <= radius_sq * 1.01` (1% larger blocking zone)
- **Rejection**: Sets `Pacc = -1.0` to force rejection

## Distance Check:
- **File**: `read_data.cpp`
- **Function**: `CheckBlockedPosition()`
- **Formula**: `dist_sq <= radius_sq * 1.01` (1% larger blocking zone, max)
- **PBC**: Applied correctly with `Apply_PBC()`

## Block Pocket Replication:
- **Lazy Replication**: Implemented in `CheckBlockedPosition()` to handle adsorbate components
- **Replication Function**: `ReplicateBlockPockets()` in `read_data.cpp`
- **Status**: **WORKING** - replicates on first use if not already replicated

## Summary:
✅ **ALL INSERTION PATHS ARE COVERED**

All insertion moves (SINGLE_INSERTION, CBMC INSERTION, WIDOM, REINSERTION, CreateMolecule) have blocking checks implemented:
- Single-body moves: Check all atoms in `SingleBody_Prepare()`
- CBMC moves: Check first bead trials + all atoms of final selected molecule
- Reinsertion: Check all atoms in `ReinsertionMove::Acceptance()`

The distance check uses `dist_sq <= radius_sq * 1.01` (1% larger blocking zone, max as requested).

