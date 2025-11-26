# Critical Blocking Fix

## Problem
- Backup2 blocks more and gets lower adsorbed molecules
- Our version blocks less and gets higher adsorbed molecules
- Need to match or exceed backup2's blocking effectiveness

## Root Cause Identified

The `get_new_position` kernel sets `device_flag[i] = false` for EACH thread (line 600 in mc_utilities.h). This means:
1. `device_flag[0]` is initialized to `false` by the kernel
2. We set it to `true` in Prepare if blocked
3. Energy calculation kernel runs and might reset it or leave it as `false`
4. We need to set it AFTER the kernel completes

## Critical Fix

**Changed the order of operations in `SingleBody_Calculation()`:**

### Before (WRONG):
1. Set device_flag in Prepare
2. Run energy calculation kernel
3. Copy device_flag to SystemComponents.flag
4. Try to set flag[0] = true (but device_flag might have been reset)

### After (CORRECT):
1. Set device_flag in Prepare (initial blocking)
2. Run energy calculation kernel
3. **Synchronize** (wait for kernel to complete)
4. **Set device_flag[0] = true AFTER kernel** (preserve blocking)
5. Copy device_flag to SystemComponents.flag
6. Ensure flag[0] = true (final check)

## Key Changes

1. **Set device_flag AFTER energy calculation**: The kernel might reset device_flag[0] to false, so we set it to true AFTER the kernel completes
2. **Check ALL atoms for SINGLE_INSERTION**: More strict than backup2's first-atom-only
3. **Check TRANSLATION/ROTATION**: Added blocking for these move types (user requested)
4. **Multiple safety checks**: Ensure flag is never lost

## Expected Results

With these fixes:
- Blocking flag is preserved through energy calculation
- All atoms checked for SINGLE_INSERTION (more strict)
- Translation/rotation also checked (more comprehensive)
- Should block MORE moves than backup2
- Should get FEWER molecules adsorbed (closer to 42.5)

