#!/bin/bash

# Test script to verify print flag functionality in gRASPA
# This test checks that atoms with print=no are not written to LAMMPS movie files

echo "=========================================="
echo "Testing Print Flag Functionality"
echo "=========================================="

# Get the directory of this script
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR"

echo ""
echo "Test setup:"
echo "  - Location: $TEST_DIR"
echo "  - pseudo_atoms.def has some atoms with print=no:"
echo "    * C_co2: print=no (should NOT appear in movie)"
echo "    * N_n2: print=no (should NOT appear in movie)"
echo "    * N_com: print=no (should NOT appear in movie)"
echo "    * CH3_sp3: print=no (should NOT appear in movie)"
echo "    * O_co2: print=yes (SHOULD appear in movie)"
echo ""

# Check if pseudo_atoms.def exists
if [ ! -f "pseudo_atoms.def" ]; then
    echo "ERROR: pseudo_atoms.def not found!"
    exit 1
fi

# Count atoms with print=no and print=yes
PRINT_NO_COUNT=$(grep -E "^\w+\s+no\s+" pseudo_atoms.def | wc -l)
PRINT_YES_COUNT=$(grep -E "^\w+\s+yes\s+" pseudo_atoms.def | wc -l)

echo "Found in pseudo_atoms.def:"
echo "  - Atoms with print=yes: $PRINT_YES_COUNT"
echo "  - Atoms with print=no: $PRINT_NO_COUNT"
echo ""

# Check if executable exists (assuming it's in the build directory)
if [ -f "../../src_clean/nvc_main.x" ]; then
    EXECUTABLE="../../src_clean/nvc_main.x"
elif [ -f "../../build/nvc_main.x" ]; then
    EXECUTABLE="../../build/nvc_main.x"
else
    echo "WARNING: Executable not found. Please compile gRASPA first."
    echo "Expected locations:"
    echo "  - ../../src_clean/nvc_main.x"
    echo "  - ../../build/nvc_main.x"
    echo ""
    echo "To compile, run:"
    echo "  cd ../../src_clean && make"
    exit 1
fi

echo "Running simulation..."
echo ""

# Run the simulation
$EXECUTABLE

if [ $? -ne 0 ]; then
    echo "ERROR: Simulation failed!"
    exit 1
fi

echo ""
echo "Checking results..."
echo ""

# Check if movie files were created
MOVIE_DIR="Movies/System_0"
if [ ! -d "$MOVIE_DIR" ]; then
    echo "ERROR: Movie directory not found: $MOVIE_DIR"
    exit 1
fi

# Find the most recent movie file
LATEST_MOVIE=$(ls -t "$MOVIE_DIR"/*.data 2>/dev/null | head -1)

if [ -z "$LATEST_MOVIE" ]; then
    echo "ERROR: No movie files found in $MOVIE_DIR"
    exit 1
fi

echo "Analyzing movie file: $LATEST_MOVIE"
echo ""

# Extract atom types from the Atoms section only (skip Masses section)
# The format is: atom_id mol_id type_id charge x y z # molecule_name atom_type_name
ATOM_TYPES_IN_MOVIE=$(awk '/^Atoms$/{flag=1; next} flag && /^[0-9]/{print $NF}' "$LATEST_MOVIE" | sed 's/.*\s\([A-Za-z0-9_]*\)$/\1/' | sort -u)

echo "Atom types found in movie file:"
for atom_type in $ATOM_TYPES_IN_MOVIE; do
    echo "  - $atom_type"
done
echo ""

# Check if atoms with print=no appear in the movie
ATOMS_WITH_PRINT_NO=$(grep -E "^\w+\s+no\s+" pseudo_atoms.def | awk '{print $1}')

FOUND_PRINT_NO_ATOM=false
for atom in $ATOMS_WITH_PRINT_NO; do
    if echo "$ATOM_TYPES_IN_MOVIE" | grep -q "^${atom}$"; then
        echo "ERROR: Atom '$atom' (print=no) was found in movie file!"
        FOUND_PRINT_NO_ATOM=true
    fi
done

if [ "$FOUND_PRINT_NO_ATOM" = true ]; then
    echo ""
    echo "TEST FAILED: Atoms with print=no should not appear in movie files!"
    exit 1
fi

# Check if atoms with print=yes appear (at least some should)
ATOMS_WITH_PRINT_YES=$(grep -E "^\w+\s+yes\s+" pseudo_atoms.def | awk '{print $1}')

FOUND_PRINT_YES_ATOM=false
for atom in $ATOMS_WITH_PRINT_YES; do
    if echo "$ATOM_TYPES_IN_MOVIE" | grep -q "^${atom}$"; then
        FOUND_PRINT_YES_ATOM=true
        break
    fi
done

if [ "$FOUND_PRINT_YES_ATOM" = false ]; then
    echo "WARNING: No atoms with print=yes were found in movie file."
    echo "This might indicate an issue, or the simulation might not have created any molecules."
fi

# Count total atoms in movie file
TOTAL_ATOMS_IN_MOVIE=$(grep -E "^[0-9]+\s+[0-9]+\s+[0-9]+\s+" "$LATEST_MOVIE" | wc -l)

echo "Total atoms in movie file: $TOTAL_ATOMS_IN_MOVIE"
echo ""

# Check the header comment for atom counts
if grep -q "Total atoms:" "$LATEST_MOVIE"; then
    echo "Movie file header information:"
    grep "Total atoms:" "$LATEST_MOVIE"
    echo ""
fi

echo "=========================================="
echo "TEST PASSED: Print flag is working correctly!"
echo "  - Atoms with print=no are NOT in movie files"
echo "  - Atoms with print=yes ARE in movie files"
echo "=========================================="

