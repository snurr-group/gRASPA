# BlockPocket Issue: Loading Higher Than RASPA2

## Problem
- **gRASPA**: 83 molecules
- **RASPA2**: 73.78 molecules
- **Difference**: 9.22 molecules (12.5% higher)

## Root Cause Analysis

### 1. PBC Calculation
**Status**: ✅ Equivalent
- RASPA2's `NINT(x)` = `round(x + 0.5)` for positive (same as gRASPA)
- PBC implementations should produce same results

### 2. Block Center Positions
**Status**: ⚠️ POTENTIAL ISSUE
- **RASPA2**: Applies `ShiftUnitCell` offset before replication
  ```c
  tempr.x += Framework[CurrentSystem].ShiftUnitCell[0].x;
  tempr.y += Framework[CurrentSystem].ShiftUnitCell[0].y;
  tempr.z += Framework[CurrentSystem].ShiftUnitCell[0].z;
  ```
- **gRASPA**: Does NOT apply ShiftUnitCell offset
- **Impact**: Block centers may be in wrong positions, leading to insufficient blocking

### 3. BlockPocket Check Timing
**Status**: ✅ gRASPA is more aggressive
- **RASPA2**: Checks BlockedPocket AFTER trial selection (on selected trial)
- **gRASPA**: Checks BlockedPocket BEFORE trial selection (on all trials, excludes blocked)
- **Impact**: gRASPA should block more, not less (so this is NOT the issue)

### 4. Fractional to Cartesian Conversion
**Status**: ✅ Should be correct
- Both use same formula: `vec * Cell` matrix multiplication
- Conversion should match RASPA2

## Most Likely Issue: Missing ShiftUnitCell Offset

The missing `ShiftUnitCell` offset could cause block centers to be shifted, leading to:
- Block centers in wrong positions
- Incorrect distance calculations
- Insufficient blocking
- Higher molecule count

## Solution

1. **Check if gRASPA has ShiftUnitCell equivalent**:
   - Search for offset/shift mechanisms in framework reading
   - Check if there's a way to apply unit cell shifts

2. **If no ShiftUnitCell in gRASPA**:
   - Verify if this is needed for this specific framework
   - Check if block file coordinates already account for shift
   - Compare block center positions with RASPA2 output

3. **Alternative checks**:
   - Verify PBC calculation is exactly matching
   - Check if block centers match RASPA2 after conversion
   - Verify distance calculations

## Debug Steps

1. Add debug output to print block centers after conversion
2. Compare with RASPA2's block center positions
3. Check if ShiftUnitCell is non-zero for this framework
4. Verify distances to block centers match RASPA2

## Current Status

- BlockPocket implementation is functionally correct
- Code matches RASPA2's logic (except ShiftUnitCell)
- Missing ShiftUnitCell offset is the most likely cause of discrepancy
- Need to verify if gRASPA has equivalent or if offset is needed

