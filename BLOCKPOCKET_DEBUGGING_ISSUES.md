# BlockPocket Debugging: Loading Higher Than RASPA2

## Problem
- **gRASPA result**: 83 molecules
- **RASPA2 result**: 73.78 molecules  
- **Difference**: 9.22 molecules (12.5% higher)

This suggests BlockPocket is not blocking enough positions or not blocking effectively.

## Investigation Areas

### 1. PBC Calculation
**RASPA2**: Uses `ApplyBoundaryConditionUnitCell()` which:
- For CUBIC/RECTANGULAR: `dr -= UnitCellSize * NINT(dr / UnitCellSize)`
- For TRICLINIC: Converts to fractional, wraps, converts back

**gRASPA**: Uses `PBC()` function which:
- For CUBIC: `dr -= Cell[0] * NINT(dr * InverseCell[0] + 0.5)`
- For TRICLINIC: Converts to fractional, wraps, converts back

**Potential Issue**: The PBC calculation might differ slightly, leading to incorrect distances.

### 2. Block Center Positions
**Verification Needed**:
- Check if fractional to Cartesian conversion matches RASPA2
- Verify block centers are in correct positions after conversion
- Compare first few block centers with RASPA2 output

### 3. Rosenbluth Weight Calculation
**RASPA2**: When `BlockedPocket()` returns TRUE, sets `value = 0.0` (completely rejects move)

**gRASPA**: Excludes blocked trials from Rosenbluth calculation (should be equivalent)

**Potential Issue**: If blocked trials are excluded but remaining trials still give high Rosenbluth weight, moves might still be accepted.

### 4. Insertion Path Coverage
**RASPA2**: Checks BlockPocket in:
- CBMC first bead placement (sample.c:2686)
- Translation/rotation moves
- Initial molecule placement
- Movie output

**gRASPA**: Currently checks in:
- CBMC/Widom insertion (mc_widom.h)
- Single insertion (mc_single_particle.h)

**Potential Issue**: Might be missing some insertion paths.

### 5. Acceptance Probability
**Key Question**: When trials are blocked, does the acceptance probability decrease enough?

If 2 out of 10 trials are blocked:
- Remaining 8 trials contribute to Rosenbluth weight
- If these 8 trials have good energies, Rosenbluth weight might still be high
- Move might still be accepted

**RASPA2 behavior**: Sets weight to 0.0 if ANY trial is blocked (for first bead)

**gRASPA behavior**: Excludes blocked trials, but if remaining trials are good, move can still be accepted

## Debug Steps

1. **Verify PBC calculation**:
   - Add debug output showing distances before/after PBC
   - Compare with RASPA2's ApplyBoundaryConditionUnitCell

2. **Verify block centers**:
   - Print first few block centers after conversion
   - Compare with RASPA2 output

3. **Check blocking effectiveness**:
   - Print how many moves are rejected due to BlockPocket
   - Compare blocking rate with RASPA2

4. **Verify Rosenbluth calculation**:
   - Print Rosenbluth weights when trials are blocked
   - Check if weights are too high

5. **Check all insertion paths**:
   - Verify BlockPocket is checked in all insertion routines
   - Check if any paths are missing

## Potential Fixes

1. **Match RASPA2's PBC exactly**:
   - Use UnitCellSize for cubic boxes (if available)
   - Ensure triclinic PBC matches exactly

2. **Verify block center conversion**:
   - Compare converted centers with RASPA2
   - Check if ShiftUnitCell offset is needed

3. **Check acceptance logic**:
   - Verify that blocking trials reduces acceptance probability enough
   - Consider if we need to reject moves more aggressively

4. **Add more BlockPocket checks**:
   - Ensure all insertion paths are covered
   - Check translation/rotation moves

## Next Steps

1. Add detailed debug output for PBC calculation
2. Compare block centers with RASPA2
3. Check Rosenbluth weights when trials are blocked
4. Verify all insertion paths check BlockPocket

