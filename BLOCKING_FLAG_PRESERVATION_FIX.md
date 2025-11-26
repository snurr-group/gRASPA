# Blocking Flag Preservation Fix

## Problem
The adsorbed molecule count was still at 44.98 (higher than reference 42.5), indicating that blocking was not working correctly. The issue was that the blocking flag set in `SingleBody_Prepare()` was being lost during the energy calculation kernel.

## Root Cause
1. **Flag Assignment Issue**: `SystemComponents.flag = Sims.device_flag;` was doing a pointer assignment instead of copying from device to host memory
2. **Flag Not Preserved**: The energy calculation kernel might reset or overwrite `device_flag[0]`, losing the blocking status set earlier
3. **No Persistence**: There was no mechanism to preserve the blocking status across kernel calls

## Solution

### 1. Added `BlockedByPockets` to `MoveTempStorage` struct
**File**: `/home/xiaoyi/gRASPA/src_clean/data_struct.h`

Added a new boolean field to store blocking status:
```cpp
bool   BlockedByPockets = false;
```

This allows the blocking status to persist across kernel calls.

### 2. Store Blocking Status in `SingleBody_Prepare()`
**File**: `/home/xiaoyi/gRASPA/src_clean/mc_single_particle.h`

When a position is blocked, we now store it:
```cpp
if(is_blocked)
{
  SystemComponents.TempVal.BlockedByPockets = true;
  bool blocked = true;
  cudaMemcpy(Sims.device_flag, &blocked, sizeof(bool), cudaMemcpyHostToDevice);
  SystemComponents.BlockPocketBlockedCount[SelectedComponent]++;
}
```

### 3. Preserve Flag After Energy Calculation
**File**: `/home/xiaoyi/gRASPA/src_clean/mc_single_particle.h`

After the energy calculation kernel completes, we restore the blocking flag if needed:
```cpp
cudaDeviceSynchronize();

// Preserve blocking flag after energy calculation kernel
// The kernel may have reset device_flag, so restore it if blocked by pockets
if(SystemComponents.TempVal.BlockedByPockets)
{
  bool blocked = true;
  cudaMemcpy(Sims.device_flag, &blocked, sizeof(bool), cudaMemcpyHostToDevice);
}

// Copy device_flag to host flag array (FIXED: was doing pointer assignment)
cudaMemcpy(SystemComponents.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);
```

### 4. Fixed Flag Copying
**Critical Fix**: Changed from pointer assignment to proper memory copy:
- **Before**: `SystemComponents.flag = Sims.device_flag;` (wrong - pointer assignment)
- **After**: `cudaMemcpy(SystemComponents.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);` (correct - memory copy)

## Expected Impact

1. **Blocking flag preserved**: The blocking status is now preserved through the energy calculation kernel
2. **Proper flag synchronization**: Device-to-host flag copying is now correct
3. **More moves rejected**: Blocked moves should now be properly rejected, leading to lower adsorbed molecule counts
4. **Closer to reference**: Should achieve adsorbed molecule count closer to 42.5

## Testing

The code compiles successfully. The blocking logic should now work correctly:
- Blocking status is stored in `BlockedByPockets`
- Flag is restored after energy calculation
- Flag is properly copied to host memory for acceptance check
- Blocked moves are rejected in `SingleBody_Acceptance()`

