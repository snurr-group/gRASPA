# BlockPocket PBC Calculation Difference

## Issue
gRASPA loading (83 molecules) is higher than RASPA2 (73.78 molecules), suggesting BlockPocket is not blocking enough.

## PBC Implementation Difference

### RASPA2 (ApplyBoundaryConditionUnitCell)
For CUBIC boxes:
```c
dr.x -= UnitCellSize[CurrentSystem].x * NINT(dr.x / UnitCellSize[CurrentSystem].x);
```
Where `NINT(x)` is `round(x)` (standard nearest integer rounding).

### gRASPA (PBC function)
For CUBIC boxes:
```cpp
posvec.x -= static_cast<int>(posvec.x * InverseCell[0*3+0] + ((posvec.x >= 0.0) ? 0.5 : -0.5)) * Cell[0*3+0];
```
Which is equivalent to:
```cpp
posvec.x -= Cell[0] * round(posvec.x / Cell[0] + 0.5);  // for positive
posvec.x -= Cell[0] * round(posvec.x / Cell[0] - 0.5);  // for negative
```

## The Difference

**RASPA2**: `round(dr / box_size)`  
**gRASPA**: `round(dr / box_size + 0.5)` for positive values

The `+0.5` offset changes rounding behavior at half-integer boundaries:
- At `dr = 12.25` (half of 24.5): 
  - RASPA2: `round(12.25/24.5) = round(0.5) = 0` or `1` (tie-break)
  - gRASPA: `round(12.25/24.5 + 0.5) = round(1.0) = 1`

This can cause different PBC wrapping, leading to different distances, which affects BlockPocket checks.

## Impact

Different PBC wrapping → Different distances to block centers → Different blocking decisions → Different molecule counts

## Solution Options

1. **Implement exact RASPA2 PBC for BlockPocket**:
   - Use `round()` without `+0.5` offset
   - Match RASPA2's `NINT` behavior exactly

2. **Verify if this is the actual issue**:
   - Add debug output to compare distances
   - Check if PBC differences are significant
   - Verify block centers are correct

3. **Check other potential issues**:
   - Block center positions
   - Distance calculation
   - Acceptance probability logic

## Current Status

- PBC difference identified
- Need to verify if this is the root cause
- May need to implement RASPA2-exact PBC for BlockPocket checks

