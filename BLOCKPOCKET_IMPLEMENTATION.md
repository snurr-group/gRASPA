# BlockPocket Functionality Implementation in gRASPA

## Analysis of RASPA2 BlockPocket Implementation

### Overview
BlockPocket is a feature in RASPA2 that allows blocking certain regions of the simulation box. Molecules are prevented from being placed in these "blocking pockets" (or, with inversion, only allowed in these pockets).

### Key Components in RASPA2

#### 1. Data Structures (molecule.h)
- `int *BlockPockets` - Array of flags (one per system) indicating if blockpockets are enabled
- `int InvertBlockPockets` - Flag to invert behavior (allow only inside pockets)
- `char (*BlockPocketsFilename)[256]` - Array of filenames (one per system)
- `int *NumberOfBlockCenters` - Number of blocking centers per system
- `REAL **BlockDistance` - Array of distances (radii) for each center
- `VECTOR **BlockCenters` - Array of center positions

#### 2. Input Parsing (input.c)
- `BlockPockets [yes|no]` - Enable/disable for each system
- `InvertBlockPockets [yes|no]` - Invert the blocking logic
- `BlockPocketsFileName [string]` - Filename for each system

#### 3. File Reading (framework.c - ReadBlockingPockets)
- Reads `.block` files with format:
  - First line: number of centers
  - Subsequent lines: x y z radius (in fractional coordinates)
- Replicates centers across unit cells
- Converts from fractional (ABC) to Cartesian (XYZ) coordinates

#### 4. Position Checking (framework.c - BlockedPocket)
- Takes a position (VECTOR) and checks if it's blocked
- Logic:
  - If `InvertBlockPockets` is TRUE: allow only positions INSIDE pockets (block if outside)
  - If `InvertBlockPockets` is FALSE: block positions INSIDE pockets (allow if outside)
- Uses periodic boundary conditions via `ApplyBoundaryConditionUnitCell`
- Returns TRUE if blocked, FALSE if allowed

#### 5. Integration Points
BlockedPocket is called in:
- MC moves: insertion, deletion, translation, rotation, reinsertion
- CBMC: first bead placement
- Movies: position validation
- Grid tests: for visualization

### Key Differences: RASPA2 vs gRASPA

1. **Coordinate System**: 
   - RASPA2: VECTOR (struct with x, y, z)
   - gRASPA: double3 (CUDA type)

2. **Data Structure**:
   - RASPA2: Components array with per-system arrays
   - gRASPA: Components struct with vectors (C++ style)

3. **Boundary Conditions**:
   - RASPA2: `ApplyBoundaryConditionUnitCell` function
   - gRASPA: Need to find equivalent or implement

4. **Coordinate Conversion**:
   - RASPA2: `ConvertFromABCtoXYZ` function
   - gRASPA: Need to check if similar function exists

5. **System Management**:
   - RASPA2: Uses CurrentSystem and CurrentComponent globals
   - gRASPA: Passes systemId and component explicitly

## Implementation Plan for gRASPA

### Step 1: Add Data Structures to Components
Add to `data_struct.h` in Components struct:
```cpp
std::vector<bool> BlockPockets;                    // One per system
bool InvertBlockPockets = false;                   // Global flag
std::vector<std::string> BlockPocketsFilename;    // One per system
std::vector<size_t> NumberOfBlockCenters;          // One per system
std::vector<std::vector<double>> BlockDistance;     // One per system, array of distances
std::vector<std::vector<double3>> BlockCenters;    // One per system, array of centers
```

### Step 2: Input Parsing
Add to `read_data.cpp` in `read_component_values_from_simulation_input`:
- Parse `BlockPockets` keyword
- Parse `InvertBlockPockets` keyword
- Parse `BlockPocketsFileName` keyword

### Step 3: File Reading Function
Create `ReadBlockingPockets` function:
- Read `.block` files
- Handle unit cell replication
- Convert fractional to Cartesian coordinates
- Store in Components structure

### Step 4: Position Check Function
Create `BlockedPocket` function:
- Take position (double3) and component/system IDs
- Check against all block centers
- Apply periodic boundary conditions
- Return bool (true if blocked)

### Step 5: Integration
Add checks in:
- `mc_single_particle.h`: SingleBody_Prepare, SingleBody_Calculation
- `mc_swap_moves.h`: Insertion/Deletion moves
- CBMC routines: First bead placement

### Step 6: Boundary Condition Handling
Need to implement or find equivalent of `ApplyBoundaryConditionUnitCell`:
- Calculate minimum image distance
- Use Box.Cell and Box.InverseCell for transformations

## Implementation Notes

1. **Coordinate System**: gRASPA uses Cartesian coordinates directly, so we may not need ABC->XYZ conversion if block files are already in Cartesian.

2. **Unit Cell Replication**: Need to check how gRASPA handles unit cells vs RASPA2.

3. **File Location**: Should check both current directory and a structures/block directory.

4. **GPU Considerations**: BlockPocket check might need to be on CPU or implemented as a GPU kernel if called frequently.

5. **Performance**: Consider caching or pre-computing if checks are expensive.

## Implementation Details

### Files Modified

1. **data_struct.h**: Added BlockPocket data structures to Components struct
   - `std::vector<std::vector<bool>> BlockPockets` - Enable/disable flag per component per system
   - `std::vector<bool> InvertBlockPockets` - Invert logic flag per component
   - `std::vector<std::vector<std::string>> BlockPocketsFilename` - Filenames per component per system
   - `std::vector<std::vector<size_t>> NumberOfBlockCenters` - Number of centers per component per system
   - `std::vector<std::vector<std::vector<double>>> BlockDistance` - Distances (radii) per component per system
   - `std::vector<std::vector<std::vector<double3>>> BlockCenters` - Center positions per component per system

2. **read_data.h**: Added function declarations
   - `ReadBlockingPockets()` - Read block files
   - `BlockedPocket()` - Check if position is blocked

3. **read_data.cpp**: 
   - Added input parsing for `BlockPockets`, `InvertBlockPockets`, `BlockPocketsFileName` keywords
   - Added initialization of BlockPocket vectors
   - Implemented `ReadBlockingPockets()` function:
     - Reads `.block` files (format: first line = number of centers, then x y z radius per line)
     - Handles unit cell replication
     - Converts fractional to Cartesian coordinates
     - Searches in current directory and RASPA_DIRECTORY/share/raspa/structures/block/
   - Implemented `BlockedPocket()` function:
     - Takes position (double3), component ID, system ID, and Box
     - Applies periodic boundary conditions (minimum image convention)
     - Checks distance to all block centers
     - Returns true if blocked, false if allowed
     - Handles InvertBlockPockets flag

4. **main.cpp**: Added call to `ReadBlockingPockets()` after reading components

5. **mc_single_particle.h**: 
   - Added BlockedPocket check in `SingleBody_Calculation()`
   - Checks first bead position for insertions
   - Checks new position for translations/rotations
   - Rejects move if position is blocked

### Key Implementation Differences from RASPA2

1. **Data Structure**: 
   - RASPA2 uses C-style arrays with per-system indexing
   - gRASPA uses C++ vectors with nested structure (component x system)

2. **Coordinate System**:
   - RASPA2: Uses VECTOR struct and ConvertFromABCtoXYZ function
   - gRASPA: Uses double3 (CUDA type) and direct matrix multiplication

3. **Boundary Conditions**:
   - RASPA2: `ApplyBoundaryConditionUnitCell()` function
   - gRASPA: Manual minimum image convention using fractional coordinates

4. **System Management**:
   - RASPA2: Uses global CurrentSystem and CurrentComponent
   - gRASPA: Passes systemId and component explicitly

### Usage

In `simulation.input`, for each component:

```
Component 0 MoleculeName
    BlockPockets              yes no
    InvertBlockPockets        no
    BlockPocketsFileName      ITQ-29 ITQ-29
```

- `BlockPockets`: yes/no for each system (space-separated)
- `InvertBlockPockets`: yes/no (global for component)
- `BlockPocketsFileName`: filename for each system (without .block extension)

Block file format (e.g., `ITQ-29.block`):
```
3
0.5 0.5 0.5 5.0
0.25 0.25 0.25 3.0
0.75 0.75 0.75 3.0
```

First line: number of centers
Subsequent lines: fractional_x fractional_y fractional_z radius

### Integration Points

BlockPocket checks are currently integrated in:
- `SingleBody_Calculation()`: Checks new positions for translation/rotation/insertion moves

Future integration points (not yet implemented):
- CBMC first bead placement
- Widom insertion moves
- Reinsertion moves
- Swap moves

## Testing Strategy

1. Test with simple block file (single center)
2. Test with multiple centers
3. Test InvertBlockPockets flag
4. Test with unit cell replication
5. Test integration with MC moves
6. Compare results with RASPA2 for same system

## Known Limitations

1. BlockPocket checks are currently only in `SingleBody_Calculation()` - need to add to:
   - CBMC routines
   - Widom moves
   - Reinsertion moves
   - Swap moves

2. GPU implementation: Currently BlockPocket checks are on CPU (copying position from GPU). For better performance, could implement as GPU kernel.

3. File path: Currently searches current directory and RASPA_DIRECTORY. Could add more search paths.

4. Coordinate conversion: Assumes block files are in fractional coordinates. May need to handle Cartesian coordinates as well.

