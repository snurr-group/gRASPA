# gRASPA Installation on Quest (Northwestern University)

> **Note**: The installation scripts and documentation in this directory were AI-generated and have been tested on Quest cluster. They are provided as-is for convenience.

This directory contains scripts and instructions for compiling gRASPA (vanilla version, without ML potential support) on the Quest cluster at Northwestern University.

## Overview

gRASPA is a GPU-accelerated Monte Carlo simulation software for molecular adsorption in nanoporous materials. This installation guide covers the **vanilla version** without machine learning potential support.

## Prerequisites

- Access to Quest cluster with GPU nodes
- NVIDIA HPC SDK (nvhpc) module available on Quest
- CUDA-compatible GPU
- Quest account/project ID for job submission

## Quick Start

### Step 1: Prepare Source Code

1. Navigate to the gRASPA source directory:
   ```bash
   cd /path/to/gRASPA/src_clean
   ```

2. Ensure you're using `std::filesystem` (not `experimental::filesystem`). If needed, the source code should already use the standard filesystem library.

### Step 2: Copy Compilation Script

Copy the compilation script to your source directory:
```bash
cp Quest-Installation/NVC_COMPILE_QUEST_VANILLA .
chmod +x NVC_COMPILE_QUEST_VANILLA
```

### Step 3: Compile gRASPA

You have two options:

#### Option A: Submit as a Job (Recommended)

1. Copy and edit the job submission script:
   ```bash
   cp Quest-Installation/compile_graspa.job .
   ```

2. Edit `compile_graspa.job` and update:
   - `--account=YOUR_ACCOUNT_HERE`: Your Quest account/project ID
   - `GRASPA_SRC_DIR`: Path to your gRASPA source directory
   - `--gres`: GPU type if different (a100, v100, etc.)

3. Submit the job:
   ```bash
   sbatch compile_graspa.job
   ```

4. Monitor the job:
   ```bash
   squeue -u $USER
   ```

5. Check results:
   ```bash
   tail -f compile_output.<JOBID>
   ```

#### Option B: Compile on GPU Node Directly

1. Request an interactive GPU session:
   ```bash
   srun --account=YOUR_ACCOUNT --partition=gengpu --gres=gpu:a100:1 --time=02:00:00 --pty bash
   ```

2. Navigate to source directory and compile:
   ```bash
   cd /path/to/gRASPA/src_clean
   ./NVC_COMPILE_QUEST_VANILLA
   ```

## Verification

After compilation, verify the executable was created:
```bash
ls -lh nvc_main.x
file nvc_main.x
```

The executable should be approximately 4-5 MB in size.

## Running gRASPA

To run gRASPA simulations, you need to submit a GPU job. Create a run script similar to the compilation job:

```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=gRASPA_run
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --error=run_error.%J
#SBATCH --output=run_output.%J

cd /path/to/gRASPA/src_clean

# Load required modules
module purge
module use /hpc/software/spack_v17d2/spack/share/spack/modules/linux-rhel7-x86_64/
module load cuda/11.2.2-gcc
module load nvhpc/21.9-gcc

# Run gRASPA
./nvc_main.x <input_file>
```

## Compilation Details

### Compiler and Modules

The compilation script uses:
- **NVIDIA HPC SDK**: `nvhpc/21.9-gcc` (preferred) or `nvhpc/24.1-gcc-10.4.0` (fallback)
- **CUDA**: `cuda/11.2.2-gcc` (preferred) or `cuda/cuda-12.1.0-openmpi-4.1.4` (fallback)

### Compiler Flags

- `-O3`: Maximum optimization
- `-std=c++20`: C++20 standard
- `-Minline`: Inline function expansion
- `-fopenmp`: OpenMP support
- `-cuda`: CUDA support
- `-stdpar=multicore`: Standard parallelism for multicore

### Source Files Compiled

- `axpy.cu`: CUDA kernels for vector operations
- `main.cpp`: Main program entry point
- `read_data.cpp`: Input file reading
- `data_struct.cpp`: Data structures
- `VDW_Coulomb.cu`: Van der Waals and Coulomb interactions

## Troubleshooting

### Compilation Fails

1. **Check module availability**:
   ```bash
   module avail nvhpc
   module avail cuda
   ```

2. **Verify you're on a GPU node**: The compilation requires GPU access, so you must either:
   - Submit as a job (recommended)
   - Use an interactive GPU session

3. **Check error logs**: Look at `compile_error.<JOBID>` for detailed error messages

4. **Verify source code**: Ensure all source files are present in `src_clean/` directory

### Linker Errors

If you encounter linker errors with `libatomic.so`:
- The script uses `nvhpc/21.9-gcc` which avoids this issue
- If using a different nvhpc version, you may need to adjust library paths

### Module Not Found

If modules are not found:
- Check available modules: `module avail`
- Update module paths in the compilation script if Quest module structure has changed

## Files in This Directory

- `NVC_COMPILE_QUEST_VANILLA`: Main compilation script
- `compile_graspa.job`: SLURM job submission script template
- `README.md`: This file

## Additional Resources

- Quest IT Support: https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/
- gRASPA Documentation: https://zhaoli2042.github.io/gRASPA-mkdoc
- Check module versions: `module avail nvhpc` and `module avail cuda`

## Notes

- The compilation typically takes 1-2 minutes on a GPU node
- The executable `nvc_main.x` will be created in the source directory
- Always load the required modules before running the executable
- For production runs, adjust job resources (memory, time) based on your simulation needs

## Version Information

- **Tested on**: Quest cluster, Northwestern University
- **NVIDIA HPC SDK**: 21.9-gcc (primary), 24.1-gcc-10.4.0 (fallback)
- **CUDA**: 11.2.2-gcc (primary), 12.1.0-openmpi-4.1.4 (fallback)
- **gRASPA**: Vanilla version (no ML support)

