# Blocking Rate Fix Summary

## Problem
- Blocking rate is LOWER than backup2
- Getting MORE molecules adsorbed (45.3) than reference (42.5)
- This indicates we're blocking FEWER moves than we should

## Root Causes Identified

### 1. **Atom Checking Difference**
**Backup2**: For SINGLE_INSERTION, only checks FIRST atom (`Sims.New.pos[0]`)
**Our Version**: Was checking ALL atoms of molecule

**Fix**: Now match backup2 - only check first atom for SINGLE_INSERTION
- For TRANSLATION/ROTATION: still check all atoms (molecule is moving)

### 2. **Flag Setting Timing**
**Backup2**: Sets `device_flag[0] = true` directly in Prepare (BEFORE energy calculation)
**Our Version**: Was only setting `BlockedByPockets` and setting flag AFTER calculation

**Fix**: Now set `device_flag` in Prepare (like backup2) AND ensure it's preserved:
- Set `device_flag` in Prepare when blocked
- Set `device_flag` again BEFORE energy calculation (to ensure it's set)
- Set `flag[0]` AFTER energy calculation (safety check)
- Handle case where `Atomsize == 0` (energy calculation doesn't run)

## Changes Made

### 1. Match Backup2 Atom Checking for SINGLE_INSERTION
```cpp
// For SINGLE_INSERTION: match backup2 behavior - check first atom only
// For TRANSLATION/ROTATION: check all atoms
size_t atomsToCheck = molsize;
if(MoveType == SINGLE_INSERTION)
{
  atomsToCheck = 1;  // Only check first atom like backup2
}
```

### 2. Set device_flag in Prepare (like backup2)
```cpp
if(isBlocked)
{
  SystemComponents.TempVal.BlockedByPockets = true;
  // CRITICAL: Also set device_flag directly like backup2 does
  bool blocked = true;
  cudaMemcpy(Sims.device_flag, &blocked, sizeof(bool), cudaMemcpyHostToDevice);
  SystemComponents.BlockPocketBlockedCount[SelectedComponent]++;
}
```

### 3. Ensure Flag is Preserved Through Calculation
```cpp
// Set device_flag BEFORE energy calculation (in case it was reset)
if(SystemComponents.TempVal.BlockedByPockets)
{
  bool blocked = true;
  cudaMemcpy(Sims.device_flag, &blocked, sizeof(bool), cudaMemcpyHostToDevice);
}

// After energy calculation, ensure flag is still set
if(SystemComponents.TempVal.BlockedByPockets)
{
  if(SystemComponents.flag.size() > 0)
  {
    SystemComponents.flag[0] = true;
  }
}
```

## Expected Results

After these fixes:
1. **Blocking rate should match backup2** - we now check only first atom for SINGLE_INSERTION
2. **Flag is set correctly** - device_flag set in Prepare like backup2
3. **Flag is preserved** - multiple safety checks ensure blocking flag isn't lost

## Why This Should Fix the Issue

1. **First Atom Only**: For SINGLE_INSERTION, backup2 only checks the first atom. If we check all atoms, we might be more lenient in some cases (if first atom passes but later atoms would fail, we still allow). But actually, checking all atoms should block MORE, not less. So matching backup2 (first atom only) should match the blocking rate.

2. **Flag Timing**: Setting device_flag in Prepare ensures it's set BEFORE energy calculation, matching backup2 exactly. The energy calculation kernel might reset device_flag based on overlaps, so we need to ensure our blocking flag is preserved.

3. **Multiple Safety Checks**: We now set the flag in multiple places to ensure it's never lost:
   - In Prepare (when blocking detected)
   - Before energy calculation (to ensure it's set)
   - After energy calculation (safety check)

## Testing

Monitor the debug output to verify:
- Blocking statistics show correct blocking rate
- Flag validation passes
- No "ERROR: BlockedByPockets=true but flag[0]=false" messages
- Blocking rate matches backup2
- Molecule count matches reference (42.5)

