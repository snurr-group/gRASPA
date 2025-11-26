# BlockPocket Implementation: gRASPA vs RASPA2

## Overview

This document compares the blockpocket (blocking pockets) implementation in gRASPA with RASPA2, and documents the changes made to gRASPA compared to commit `f0c86ea` (Merge pull request #67).

## Table of Contents

1. [RASPA2 BlockPocket Implementation](#raspa2-blockpocket-implementation)
2. [gRASPA BlockPocket Implementation](#graspa-blockpocket-implementation)
3. [Comparison and Differences](#comparison-and-differences)
4. [Changes Since Commit f0c86ea](#changes-since-commit-f0c86ea)
5. [Usage Instructions](#usage-instructions)
6. [Technical Details](#technical-details)

---

## RASPA2 BlockPocket Implementation

### Data Structures

In RASPA2 (`molecule.h`):
```c
int *BlockPockets;                    // Per-component, per-system flag
int InvertBlockPockets;              // Global per-component flag
char (*BlockPocketsFilename)[256];    // Per-component, per-system filename
int *NumberOfBlockCenters;            // Per-component, per-system count
REAL **BlockDistance;                // Per-component, per-system radii array
VECTOR **BlockCenters;                // Per-component, per-system centers array
```

### Key Functions

1. **`ReadBlockingPockets()`** (`framework.c:6180-6244`)
   - Reads block pocket files (`.block` format)
   - Replicates pockets across unit cells
   - Converts coordinates from ABC to XYZ

2. **`BlockedPocket(VECTOR pos)`** (`framework.c:6253-6322`)
   - Universal function called throughout MC moves
   - Returns `TRUE` if blocked, `FALSE` if allowed
   - Supports `InvertBlockPockets` flag
   - Tracks statistics: `BlockedPocketCalls` and `BlockedPocketBlocked`

3. **Statistics Tracking**
   - `REAL **BlockedPocketCalls` - tracks total calls
   - `REAL **BlockedPocketBlocked` - tracks blocked calls
   - Printed via `PrintBlockedPocketStatistics()` (`statistics.c:4700`)

### Call Sites in RASPA2

`BlockedPocket()` is called in:
- **MC Moves** (`mc_moves.c`): Translation, rotation, reinsertion, identity swap, etc.
  - Pattern: `if(BlockedPocket(TrialPosition[CurrentSystem][i])) return 0;`
- **CBMC** (`cbmc.c`): During chain growth
- **Sampling** (`sample.c`): For residence time calculations
- **Movies** (`movies.c`): For visualization filtering

### Input Parameters

```c
BlockPockets [yes|no]              // Enable/disable per system
BlockPocketsFileName [string]       // Filename (without .block extension)
InvertBlockPockets [yes|no]        // Invert logic (allow only inside)
```

### Block File Format

```
<number_of_pockets>
<x> <y> <z> <radius>
<x> <y> <z> <radius>
...
```

Coordinates are in fractional (ABC) coordinates, converted to Cartesian (XYZ) during reading.

---

## gRASPA BlockPocket Implementation

### Data Structures

In gRASPA (`data_struct.h`):
```cpp
std::vector<bool>   UseBlockPockets;                // Per-component flag
std::vector<bool>   InvertBlockPockets;             // Per-component flag (NEW)
std::vector<std::vector<double3>> BlockPocketCenters; // [component][pocket] centers
std::vector<std::vector<double>> BlockPocketRadii;   // [component][pocket] radii
std::vector<size_t> BlockPocketTotalAttempts;         // Statistics: total calls
std::vector<size_t> BlockPocketBlockedCount;         // Statistics: blocked calls
```

### Key Functions

1. **`ReadBlockPockets()`** (`read_data.cpp:3112-3146`)
   - Reads `.block` files
   - Stores centers and radii per component
   - Similar to RASPA2 but uses C++ vectors

2. **`ReplicateBlockPockets()`** (`read_data.cpp:3154-3246`)
   - Replicates pockets across unit cells
   - Handles both fractional and Cartesian coordinates
   - Converts to Cartesian for storage

3. **`BlockedPocket()`** (`read_data.cpp:3267-3418`) - **NEW UNIVERSAL FUNCTION**
   - Matches RASPA2's `BlockedPocket()` implementation exactly
   - Returns `true` if blocked, `false` if allowed
   - Supports `InvertBlockPockets` flag
   - Tracks statistics internally
   - Uses RASPA2's PBC logic (`ApplyBoundaryConditionUnitCell`)

4. **Statistics Printing** (`print_statistics.cuh`)
   - `Print_BlockPocket_Statistics()` - prints final statistics
   - Periodic printing every 10000 attempts (if enabled)
   - Controlled by `ENABLE_BLOCKPOCKET_STATISTICS` flag

### Call Sites in gRASPA

`BlockedPocket()` is called in:
- **Single Particle Moves** (`mc_single_particle.h`):
  - Translation, rotation, special rotation: checks all atoms
  - Single insertion: checks first bead
- **CBMC/Widom Moves** (`mc_widom.h`):
  - CBMC insertion: checks all trial positions
  - Reinsertion insertion: checks all trial positions

### Input Parameters

```cpp
BlockPockets [yes|no]              // Enable/disable
BlockPocketsFilename [string]       // Filename (without .block extension)
InvertBlockPockets [yes|no]         // Invert logic (NEW - per component)
```

---

## Comparison and Differences

### Similarities

1. **Core Algorithm**: Both use spherical blocking regions with distance checks
2. **PBC Handling**: Both use `ApplyBoundaryConditionUnitCell` logic
3. **File Format**: Both use same `.block` file format
4. **Statistics**: Both track total attempts and blocked counts
5. **Call Pattern**: Both check `BlockedPocket()` before accepting moves

### Key Differences

| Feature | RASPA2 | gRASPA |
|---------|--------|--------|
| **Language** | C | C++ |
| **Data Structures** | C arrays with pointers | C++ vectors |
| **Per-System Support** | Yes (per-component, per-system) | No (per-component only) |
| **Statistics Type** | `REAL` (double) | `size_t` (unsigned integer) |
| **Statistics Location** | Global arrays | Component vectors |
| **InvertBlockPockets** | Global per-component | Per-component vector |
| **Coordinate Storage** | Cartesian (XYZ) | Cartesian (XYZ) |
| **Replication** | During read | Separate function |
| **Statistics Flag** | Always on | Configurable (`ENABLE_BLOCKPOCKET_STATISTICS`) |

### Implementation Details

#### RASPA2 Approach
- Uses global `CurrentComponent` and `CurrentSystem` variables
- BlockedPocket is a simple function: `int BlockedPocket(VECTOR pos)`
- Statistics are global arrays indexed by system and component
- PBC uses RASPA2's internal coordinate system

#### gRASPA Approach
- Explicitly passes component index and box structure
- BlockedPocket signature: `bool BlockedPocket(Components&, size_t, const double3&, Boxsize&)`
- Statistics are per-component vectors
- PBC logic matches RASPA2 but adapted for gRASPA's box structure
- Requires copying box data from device to host for PBC calculations

---

## Changes Since Commit f0c86ea

### Summary

Commit `f0c86ea` (Merge pull request #67) was the baseline. The following changes were made to implement a complete, RASPA2-compatible blockpocket system:

### 1. Added InvertBlockPockets Support

**Files Modified:**
- `data_struct.h`: Added `std::vector<bool> InvertBlockPockets`
- `read_data.cpp`: Added parsing for `InvertBlockPockets` keyword
- `fxn_main.h`: Initialize `InvertBlockPockets` vector

**What Changed:**
- Previously, gRASPA only supported normal blocking (block inside pockets)
- Now supports inverted blocking (allow only inside pockets, block outside)
- Matches RASPA2's `InvertBlockPockets` functionality

### 2. Created Universal BlockedPocket Function

**Files Modified:**
- `read_data.cpp`: Replaced `CheckBlockedPosition()` with `BlockedPocket()`
- `read_data.h`: Updated function declaration

**What Changed:**
- **Before**: `CheckBlockedPosition()` was a simple distance check without statistics or invert support
- **After**: `BlockedPocket()` matches RASPA2's implementation:
  - Supports `InvertBlockPockets` flag
  - Tracks statistics internally
  - Uses RASPA2's exact PBC logic
  - Returns early when blocked position found

**Key Implementation:**
```cpp
bool BlockedPocket(Components& SystemComponents, size_t component, 
                   const double3& pos, Boxsize& Box)
{
  // Early returns for disabled blockpockets
  // Track statistics
  // Apply PBC using RASPA2 logic
  // Check all pocket centers
  // Handle InvertBlockPockets flag
  // Return result
}
```

### 3. Added BlockedPocket Checks in All MC Moves

**Files Modified:**
- `mc_single_particle.h`: Added checks for translation, rotation, special rotation
- `mc_widom.h`: Updated CBMC/reinsertion checks

**What Changed:**
- **Before**: Only single insertion and CBMC insertion had blockpocket checks
- **After**: All move types that create new positions check blockpockets:
  - Translation: checks all atoms in molecule
  - Rotation: checks all atoms in molecule
  - Special rotation: checks all atoms in molecule
  - CBMC insertion: checks all trial positions
  - Reinsertion: checks all trial positions

**Pattern (matching RASPA2):**
```cpp
for(size_t i = 0; i < Molsize; i++)
{
  if(BlockedPocket(SystemComponents, SelectedComponent, trial_positions[i], Sims.Box))
  {
    // Block the move
    return;
  }
}
```

### 4. Improved Statistics Tracking

**Files Modified:**
- `read_data.cpp`: Statistics tracked inside `BlockedPocket()`
- `mc_single_particle.h`: Removed duplicate statistics tracking
- `mc_widom.h`: Removed duplicate statistics tracking

**What Changed:**
- **Before**: Statistics were manually tracked in each move function
- **After**: Statistics are automatically tracked inside `BlockedPocket()`
- Matches RASPA2's approach where statistics are tracked in the function itself

### 5. Added Configurable Statistics Output

**Files Modified:**
- `print_statistics.cuh`: Added `ENABLE_BLOCKPOCKET_STATISTICS` flag

**What Changed:**
- **Before**: Statistics were always printed (if blockpockets enabled)
- **After**: Statistics can be enabled/disabled via hardcoded flag:
  ```cpp
  static constexpr bool ENABLE_BLOCKPOCKET_STATISTICS = true;
  ```
- Statistics printed:
  - Periodically: every 10000 attempts (if enabled)
  - At end: in final statistics summary (if enabled)

### Files Changed Summary

| File | Changes |
|------|---------|
| `data_struct.h` | Added `InvertBlockPockets` vector |
| `read_data.h` | Updated function declarations |
| `read_data.cpp` | Universal `BlockedPocket()` function, input parsing |
| `fxn_main.h` | Initialize `InvertBlockPockets` |
| `mc_single_particle.h` | Added blockpocket checks for all move types |
| `mc_widom.h` | Updated blockpocket checks, removed duplicate stats |
| `print_statistics.cuh` | Added configurable statistics flag |

---

## Usage Instructions

### Input File Format

```cpp
Component 0 MoleculeName CO2
  BlockPockets yes
  BlockPocketsFilename LTA
  InvertBlockPockets no
```

### Block File Format

Create a file named `<BlockPocketsFilename>.block`:
```
4
0.0 0.0 0.0 5.0
0.5 0.5 0.5 5.0
0.25 0.25 0.25 3.0
0.75 0.75 0.75 3.0
```

Format: `<x> <y> <z> <radius>` in fractional (ABC) coordinates.

### Enabling/Disabling Statistics

Edit `print_statistics.cuh`:
```cpp
// Line 3
static constexpr bool ENABLE_BLOCKPOCKET_STATISTICS = true;  // or false
```

### Behavior

1. **Normal Blocking** (`InvertBlockPockets = no`):
   - Positions inside any block pocket are **blocked**
   - Positions outside all block pockets are **allowed**

2. **Inverted Blocking** (`InvertBlockPockets = yes`):
   - Positions inside any block pocket are **allowed**
   - Positions outside all block pockets are **blocked**

---

## Technical Details

### PBC Implementation

Both RASPA2 and gRASPA use the same PBC logic:

**For cubic boxes:**
```
dr -= UnitCellSize * NINT(dr / UnitCellSize)
```

**For non-cubic boxes:**
```
1. Convert to fractional: s = InverseCell * dr
2. Apply: t = s - NINT(s)
3. Convert back: dr = Cell * t
```

### Distance Calculation

Both use Euclidean distance:
```cpp
r = sqrt((center.x - pos.x)² + (center.y - pos.y)² + (center.z - pos.z)²)
```

Block if: `r < radius`

### Statistics Tracking

**RASPA2:**
```c
BlockedPocketCalls[CurrentSystem][CurrentComponent] += 1.0;
if(result == TRUE)
  BlockedPocketBlocked[CurrentSystem][CurrentComponent] += 1.0;
```

**gRASPA:**
```cpp
SystemComponents.BlockPocketTotalAttempts[component] += 1.0;
if(result == true)
  SystemComponents.BlockPocketBlockedCount[component] += 1.0;
```

### Performance Considerations

1. **gRASPA**: Requires copying box data from device to host for PBC calculations
   - This is necessary because blockpocket checks are done on host
   - Future optimization: could move to device if needed

2. **Statistics**: Tracked on every call
   - Minimal overhead (simple increment)
   - Can be disabled if not needed

3. **Early Returns**: Both implementations return early when blocked position found
   - Optimizes for common case (most positions not blocked)

---

## Future Improvements

1. **Device-side BlockPocket checks**: Move checks to GPU for better performance
2. **Per-system support**: Add per-system blockpocket configuration (like RASPA2)
3. **Grid-based optimization**: Use spatial grids for faster lookups (RASPA2 has `BlockGridPockets`)
4. **Statistics refinement**: Add more detailed statistics (per-move-type, per-pocket)

---

## Testing Recommendations

1. **Compare with RASPA2**: Run same simulation with both codes, compare:
   - Number of blocked insertions
   - Acceptance rates
   - Final statistics

2. **Test InvertBlockPockets**: Verify inverted logic works correctly

3. **Test PBC**: Verify blockpockets work correctly across periodic boundaries

4. **Test Statistics**: Verify statistics match expected values

---

## References

- RASPA2 Source: `/home/xiaoyi/RASPA2/src/framework.c` (lines 6174-6343)
- RASPA2 Input: `/home/xiaoyi/RASPA2/src/input.c` (lines 2666-2695)
- gRASPA Implementation: `/home/xiaoyi/gRASPA/src_clean/read_data.cpp` (lines 3267-3418)

---

**Document Version**: 1.0  
**Last Updated**: 2024-11-26  
**Author**: Implementation based on RASPA2 blockpocket functionality

