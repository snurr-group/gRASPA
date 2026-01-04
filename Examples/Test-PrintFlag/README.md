# Test Case: Print Flag Functionality

This test case verifies that the "print" flag in `pseudo_atoms.def` is correctly read and used when writing LAMMPS movie files.

## Test Setup

This test uses a modified `pseudo_atoms.def` file where some atoms have `print=no` and others have `print=yes`:

**Atoms with print=no (should NOT appear in movie files):**
- `C_co2`
- `N_n2`
- `N_com`
- `CH3_sp3`

**Atoms with print=yes (SHOULD appear in movie files):**
- `C`, `H`, `Zn`, `N`, `O` (framework atoms)
- `O_co2` (CO2 oxygen)
- `Lw`, `Hw`, `Ow` (water atoms)
- `CH4_sp3`

## Files

- `pseudo_atoms.def` - Modified to have some atoms with `print=no`
- `simulation.input` - Simple simulation with CO2 molecules in MFI framework
- `MFI-2x2x2-P1.cif` - MFI zeolite framework structure file
- `CO2.def` - CO2 molecule definition
- `force_field_mixing_rules.def` - Force field parameters
- `test_print_flag.sh` - Test script to verify functionality

## Running the Test

1. **Compile gRASPA** (if not already compiled):
   ```bash
   cd ../../src_clean
   make
   ```

2. **Run the test script**:
   ```bash
   cd Examples/Test-PrintFlag
   ./test_print_flag.sh
   ```

   Or run manually:
   ```bash
   cd Examples/Test-PrintFlag
   ../../src_clean/nvc_main.x
   ```

3. **Check the results**:
   - The test script will automatically check if atoms with `print=no` appear in the movie files
   - Movie files are created in `Movies/System_0/`
   - The movie file header should show both total atoms and printable atoms

## Expected Behavior

1. **Parsing**: The `PseudoAtomParser` should read the print flag from column 2 of `pseudo_atoms.def`

2. **Movie File Writing**: 
   - Only atoms with `print=yes` should appear in the LAMMPS movie files
   - The movie file header should show:
     - Total atoms: (all atoms in system)
     - Printable atoms: (only atoms with print=yes)
   - The "atoms" count in the LAMMPS file should match the printable atoms count

3. **Verification**:
   - Movie files should NOT contain any atoms with `print=no` (C_co2, N_n2, N_com, CH3_sp3)
   - Movie files SHOULD contain atoms with `print=yes` (O_co2, etc.)

## What This Tests

This test verifies the bug fix for:
- Reading the "print" flag from `pseudo_atoms.def`
- Storing the print flag in `PseudoAtomDefinitions` structure
- Filtering atoms based on print flag when writing LAMMPS movie files
- Correct atom counting in movie file headers

## Notes

- The CO2 molecule uses `C_co2` and `O_co2` atom types
- Since `C_co2` has `print=no`, only `O_co2` atoms should appear in the movie files
- This creates a visible difference that can be verified
- The test uses the MFI zeolite framework (MFI-2x2x2-P1) instead of an empty box
- Framework atoms (if any) will also be subject to the print flag filtering

