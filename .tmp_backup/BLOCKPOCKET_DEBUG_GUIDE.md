# BlockPocket Debug Guide

## Debug Printouts in Log Files

This guide explains all debug printouts related to BlockPocket functionality in gRASPA.

### 1. Initialization Phase

#### ReadBlockingPockets() Output
```
ReadBlockingPockets called: component=X, systemId=Y
  BlockPockets enabled for component X, system Y
  Reading block file: treating coordinates as FRACTIONAL (RASPA2 behavior)
  Number of centers in file: N
  Unit cells: X x Y x Z
  Total block centers after replication: M
  Reading block centers (fractional coordinates, will be converted to Cartesian):
    Center 0 from file: frac=(x, y, z), radius=r
    BlockPocket[comp=X, sys=Y, idx=0]: frac=(x, y, z) -> cart=(X, Y, Z), radius=r, unit_cell=(j,k,l)
    ...
  Successfully read and converted M block centers
BlockPocket Summary: Component X, System Y: Read M block centers from file [./filename.block]
```

**What to check**:
- Number of centers matches file
- Fractional coordinates are in 0-1 range
- Cartesian coordinates are reasonable for box size
- Replication matches unit cell count

### 2. Runtime Phase

#### BlockedPocket() Statistics
```
BlockedPocket[comp=X, sys=Y] stats: N calls, M blocked (P%), invert=no
```

**What to check**:
- Percentage blocked should be reasonable (typically 10-30%)
- If 0%, BlockPocket may not be working
- If 100%, all positions blocked (check block centers)

#### Individual Blocking Events
```
BlockedPocket[comp=X, sys=Y]: Position (x, y, z) BLOCKED by center N (r=distance < radius=r)
```

**What to check**:
- Positions being blocked are actually inside block centers
- Distances are correct
- PBC is working (distances should be reasonable)

#### Trial Blocking in CBMC/Widom
```
BlockPocket[comp=X, sys=Y] trial check #N: Blocked M out of 10 trials (P%)
BlockPocket[comp=X, sys=Y] WARNING: ALL 10 trials blocked! Move will be rejected.
```

**What to check**:
- Average trials blocked per check
- If all trials blocked frequently, may need to adjust block centers
- If 0 trials blocked, BlockPocket may not be working

### 3. Common Issues and Debugging

#### Issue: BlockPocket not reducing molecule count

**Check**:
1. Verify BlockPocket is enabled: Look for "BlockPockets enabled" message
2. Check block centers are read: Look for "Successfully read and converted" message
3. Check blocking percentage: Look for "BlockedPocket stats" - should be > 0%
4. Check trial blocking: Look for "trial check" messages - should show trials being blocked

**Possible causes**:
- Block centers not in right locations
- Radii too small
- PBC calculation incorrect
- BlockPocket checks not applied to all insertion paths

#### Issue: All positions blocked

**Check**:
1. Block center coordinates (should be reasonable for box size)
2. Radii (should not be too large)
3. InvertBlockPockets flag (should be false for normal blocking)

#### Issue: No positions blocked

**Check**:
1. BlockPocket enabled flag
2. Block centers read correctly
3. BlockedPocket function being called
4. PBC calculation working

### 4. Comparing with RASPA2

To verify gRASPA matches RASPA2:

1. **Block Centers**: Compare Cartesian coordinates after conversion
2. **Blocking Percentage**: Should be similar
3. **Final Molecule Count**: Should match (within statistical error)

### 5. Key Differences from RASPA2

**gRASPA Implementation**:
- Uses automatic coordinate format detection (RASPA2 always uses fractional)
- Uses gRASPA's PBC function (should be equivalent to ApplyBoundaryConditionUnitCell)
- Stores block centers in vectors (RASPA2 uses arrays)

**Verification**:
- Check that fractional coordinates are correctly converted
- Verify PBC calculation matches RASPA2's minimum image convention
- Ensure all insertion paths check BlockPocket

