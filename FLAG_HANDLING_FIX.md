# Flag Handling Fix - Matching backup2 Behavior

## Problem
Current version should theoretically block more than backup2 (checks all atoms vs first atom), but is getting higher adsorbed molecule count. This suggests flag handling is incorrect.

## Root Cause
In backup2, `SystemComponents.flag = Sims.device_flag;` is a **pointer assignment** (both are host memory - `device_flag` is allocated with `cudaMallocHost` which creates pinned host memory). This means they point to the same memory location.

In our version, we were using `cudaMemcpy` which creates a **copy** of the value at that moment. This means:
- If the energy kernel resets the flag, our copy doesn't reflect that
- If we restore the flag after copying, the copy doesn't get updated
- The acceptance check uses the old copied value, not the current value

## Solution

### 1. Use Pointer Assignment (Like backup2)
**File**: `/home/xiaoyi/gRASPA/src_clean/mc_single_particle.h`

Changed from:
```cpp
cudaMemcpy(SystemComponents.flag, Sims.device_flag, sizeof(bool), cudaMemcpyDeviceToHost);
```

To:
```cpp
SystemComponents.flag = Sims.device_flag; // Pointer assignment - both are host memory
```

### 2. Restore Flag AFTER Synchronize
**File**: `/home/xiaoyi/gRASPA/src_clean/mc_single_particle.h`

Changed order to:
```cpp
Calculate_Single_Body_Energy_VDWReal<<<...>>>(...);
SystemComponents.flag = Sims.device_flag; // Set pointer first
cudaDeviceSynchronize(); // Wait for kernel to complete

// Restore blocking flag AFTER synchronize
if(SystemComponents.TempVal.BlockedByPockets)
{
  Sims.device_flag[0] = true; // Direct assignment
  // Since flag points to device_flag, this automatically updates flag[0]
}
```

### 3. Use Direct Assignment for Flag Setting
**File**: `/home/xiaoyi/gRASPA/src_clean/mc_single_particle.h`

Changed from:
```cpp
cudaMemcpy(Sims.device_flag, &blocked, sizeof(bool), cudaMemcpyHostToDevice);
```

To:
```cpp
Sims.device_flag[0] = true; // Direct assignment since device_flag is pinned host memory
```

## Why This Matters

With pointer assignment:
- `SystemComponents.flag[0]` and `Sims.device_flag[0]` are the same memory location
- When we set `Sims.device_flag[0] = true`, `SystemComponents.flag[0]` automatically reflects that
- When the acceptance function checks `SystemComponents.flag[0]`, it's checking the current value
- This matches backup2's behavior exactly

## Expected Impact

This should fix the flag handling to match backup2, while still maintaining:
- All atoms checked for SINGLE_INSERTION, TRANSLATION, ROTATION
- All atoms checked for REINSERTION
- CBMC first bead trials checked
- Stricter distance check (`<=` instead of `<`)

The code should now block correctly and achieve lower adsorbed molecule count than backup2.

