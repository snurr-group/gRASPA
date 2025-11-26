# Critical Fix: Block Pocket Replication for Adsorbate Components

## Problem Identified
The adsorbed molecule count was much higher than expected (45.5 vs reference 42.5), indicating insufficient blocking. After comparing with backup2 and RASPA2, a critical bug was found:

**Block pockets for adsorbate components were NOT being replicated across unit cells!**

## Root Cause

### Block Pocket Reading Flow:
1. Framework components are read first → `ReadFramework()` is called
2. During framework reading, block pockets are replicated (line 1817-1823 in `read_data.cpp`)
3. **BUT** this replication only happens for components that exist at that time
4. Adsorbate components are read later → `read_component_values_from_simulation_input()` is called
5. Block pockets for adsorbates are read during adsorbate reading (line 2367 in `read_data.cpp`)
6. **BUT** replication never happens for adsorbate components!

### Impact:
- Only the original block pocket centers were checked (not replicated across unit cells)
- This means blocking only worked for positions very close to the original centers
- Positions near replicated centers in other unit cells were NOT blocked
- Result: Much less blocking than expected → Higher adsorbed molecule count

## Solution

### Fix in `/home/xiaoyi/gRASPA/src_clean/main.cpp`:

**Before:**
```cpp
// Replicate block pockets across unit cells now that Box is initialized
// This must be done after all components are read and Box is set up
for(size_t comp = 0; comp < Vars.SystemComponents[a].NComponents.x; comp++)
{
  if(comp < Vars.SystemComponents[a].UseBlockPockets.size() && Vars.SystemComponents[a].UseBlockPockets[comp])
  {
    // Box will be initialized later, so we'll replicate after Box setup
    // For now, mark that replication is needed
  }
}
```

**After:**
```cpp
// Replicate block pockets across unit cells now that Box is initialized
// This must be done after all components are read and Box is set up
// Box is already initialized from ReadFramework, so we can replicate now
for(size_t comp = 0; comp < Vars.SystemComponents[a].NComponents.x; comp++)
{
  if(comp < Vars.SystemComponents[a].UseBlockPockets.size() && Vars.SystemComponents[a].UseBlockPockets[comp])
  {
    ReplicateBlockPockets(Vars.SystemComponents[a], comp, Vars.Box[a]);
  }
}
```

## Why This Matters

RASPA2 replicates block pocket centers across all unit cells. For example, if you have a 2x2x2 unit cell system, each block pocket center is replicated 8 times (once per unit cell). This ensures that blocking works correctly regardless of which unit cell a molecule is in.

Without replication:
- Only 1 center per block pocket is checked
- Molecules in other unit cells are not blocked
- Result: Much less blocking

With replication:
- 8 centers per block pocket are checked (for 2x2x2 system)
- Molecules in any unit cell are properly blocked
- Result: Correct blocking behavior matching RASPA2

## Expected Impact

This fix should significantly increase blocking and reduce the adsorbed molecule count. Combined with:
- All atoms checked for SINGLE_INSERTION, TRANSLATION, ROTATION, REINSERTION
- Correct flag handling (pointer assignment like backup2)
- Stricter distance check (`<=` instead of `<`)

The code should now achieve blocking behavior matching or exceeding RASPA2.

