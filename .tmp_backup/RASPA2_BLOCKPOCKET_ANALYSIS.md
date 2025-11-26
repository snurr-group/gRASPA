# RASPA2 BlockPocket Implementation Analysis

## Overview
This document provides a detailed analysis of RASPA2's BlockPocket implementation to ensure gRASPA matches its behavior exactly.

## Key Functions

### 1. ReadBlockingPockets() - framework.c:6179

**Purpose**: Read blocking pocket centers from a `.block` file and convert to Cartesian coordinates.

**Key Implementation Details**:

```c
void ReadBlockingPockets(void)
{
  // For each component with BlockPockets enabled:
  // 1. Open file: ./{filename}.block or {RASPA_DIRECTORY}/share/raspa/structures/block/{filename}.block
  // 2. Read number of centers (first line)
  // 3. For each center:
  //    - Read x, y, z, radius (fractional coordinates 0-1)
  //    - Add ShiftUnitCell offset
  //    - Replicate across unit cells (j, k, l loops)
  //    - Convert fractional to fractional within unit cell: vec = (tempr + j,k,l) / NumberOfUnitCells
  //    - Convert from ABC (fractional) to XYZ (Cartesian) using ConvertFromABCtoXYZ(vec)
  //    - Store center and radius
}
```

**Important Points**:
- **ALWAYS treats coordinates as FRACTIONAL** (0-1 range)
- **Replicates across ALL unit cells** (j, k, l loops)
- Uses `ConvertFromABCtoXYZ()` to convert fractional to Cartesian
- Applies `ShiftUnitCell` offset before replication
- Final number of centers = `numCenters * NumberOfUnitCells.x * NumberOfUnitCells.y * NumberOfUnitCells.z`

**Code Flow**:
```c
for(i=0; i < NumberOfBlockCenters; i++) {
  fscanf(FilePtr, "%lf %lf %lf %lf\n", &tempr.x, &tempr.y, &tempr.z, &temp);
  tempr.x += Framework[CurrentSystem].ShiftUnitCell[0].x;
  tempr.y += Framework[CurrentSystem].ShiftUnitCell[0].y;
  tempr.z += Framework[CurrentSystem].ShiftUnitCell[0].z;
  
  for(j=0; j < NumberOfUnitCells.x; j++)
    for(k=0; k < NumberOfUnitCells.y; k++)
      for(l=0; l < NumberOfUnitCells.z; l++) {
        vec.x = (tempr.x + j) / NumberOfUnitCells.x;
        vec.y = (tempr.y + k) / NumberOfUnitCells.y;
        vec.z = (tempr.z + l) / NumberOfUnitCells.z;
        
        BlockCenters[index] = ConvertFromABCtoXYZ(vec);
        BlockDistance[index] = temp;
        index++;
      }
}
```

### 2. BlockedPocket() - framework.c:6252

**Purpose**: Check if a position is blocked by any blocking pocket.

**Key Implementation Details**:

```c
int BlockedPocket(VECTOR pos)
{
  if(!BlockPockets[CurrentSystem]) return FALSE;
  
  if(InvertBlockPockets) {
    // Invert mode: allow ONLY inside pockets
    for(i=0; i < NumberOfBlockCenters; i++) {
      dr = BlockCenters[i] - pos;
      dr = ApplyBoundaryConditionUnitCell(dr);  // Apply PBC
      r = sqrt(dr.x^2 + dr.y^2 + dr.z^2);
      if(r < BlockDistance[i]) return FALSE;  // Inside pocket, allowed
    }
    return TRUE;  // Not in any pocket, blocked
  }
  else {
    // Normal mode: block inside pockets
    for(i=0; i < NumberOfBlockCenters; i++) {
      dr = BlockCenters[i] - pos;
      dr = ApplyBoundaryConditionUnitCell(dr);  // Apply PBC
      r = sqrt(dr.x^2 + dr.y^2 + dr.z^2);
      if(r < BlockDistance[i]) return TRUE;  // Inside pocket, blocked
    }
    return FALSE;  // Not in any pocket, allowed
  }
}
```

**Important Points**:
- Uses `ApplyBoundaryConditionUnitCell()` for PBC (minimum image convention)
- Returns `TRUE` if blocked, `FALSE` if allowed
- Checks ALL block centers (stops early if blocked in normal mode)
- Invert mode: returns `FALSE` if inside any pocket, `TRUE` if outside all

### 3. ApplyBoundaryConditionUnitCell() - potentials.c:5715

**Purpose**: Apply periodic boundary conditions to a distance vector using minimum image convention.

**Key Implementation**:
- For CUBIC/RECTANGULAR: `dr.x -= UnitCellSize.x * NINT(dr.x/UnitCellSize.x)`
- For TRICLINIC: Convert to fractional, wrap to [-0.5, 0.5], convert back

### 4. Integration in MC Moves - sample.c:2686

**Key Integration Point**:
```c
if(OVERLAP || BlockedPocket(NewPosition[CurrentSystem][StartingBead]))
  value = 0.0;  // Set Rosenbluth weight to zero (reject move)
```

**Important**: When `BlockedPocket()` returns `TRUE`, the Rosenbluth weight is set to `0.0`, which completely rejects the move.

## Differences from gRASPA Implementation

### Current gRASPA Implementation Issues:

1. **Coordinate Format Detection**: 
   - gRASPA: Auto-detects Cartesian vs fractional
   - RASPA2: ALWAYS treats as fractional
   - **FIX NEEDED**: Remove auto-detection, always treat as fractional

2. **Unit Cell Replication**:
   - gRASPA: For Cartesian, doesn't replicate
   - RASPA2: ALWAYS replicates (even if coordinates look Cartesian)
   - **FIX NEEDED**: Always replicate for fractional coordinates

3. **ShiftUnitCell**:
   - gRASPA: Not applying ShiftUnitCell offset
   - RASPA2: Applies ShiftUnitCell before replication
   - **FIX NEEDED**: Apply ShiftUnitCell if available

4. **ConvertFromABCtoXYZ**:
   - gRASPA: Manual conversion using Box.Cell matrix
   - RASPA2: Uses ConvertFromABCtoXYZ function
   - **STATUS**: Should be equivalent if Box.Cell is correct

## Required Fixes

1. Remove automatic coordinate format detection
2. Always treat block file coordinates as fractional (0-1)
3. Always replicate across unit cells
4. Apply ShiftUnitCell offset (if available in gRASPA)
5. Ensure PBC calculation matches ApplyBoundaryConditionUnitCell

