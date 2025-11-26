# gRASPA Compilation Guide

This guide provides instructions for compiling gRASPA on different systems and with different compilers.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Compilation Methods](#compilation-methods)
   - [Method 1: Using nvc++ (NVIDIA HPC SDK) - Recommended](#method-1-using-nvc-nvidia-hpc-sdk---recommended)
   - [Method 2: Using nvcc (CUDA Compiler)](#method-2-using-nvcc-cuda-compiler)
3. [Cluster-Specific Instructions](#cluster-specific-instructions)
4. [Troubleshooting](#troubleshooting)
5. [References](#references)

## Prerequisites

### Required Software
- **CUDA Toolkit** (version 11.0 or higher recommended)
- **C++ Compiler** (g++ 9.0+ or clang++ 10.0+)
- **NVIDIA GPU** with compute capability 7.0 or higher
- **OpenMP** library (usually included with compiler)

### Optional Dependencies
- **PyTorch/LibTorch** (for Allegro ML potential support)
- **TensorFlow/CppFlow** (for LCLIN ML potential support)

## Compilation Methods

### Method 1: Using nvc++ (NVIDIA HPC SDK) - Recommended

The **nvc++** compiler from NVIDIA HPC SDK is the recommended compiler for gRASPA as it provides better optimization and support for GPU-accelerated code.

#### Installation
1. Download NVIDIA HPC SDK from: https://developer.nvidia.com/hpc-sdk
2. Install following the instructions for your system
3. Ensure `nvc++` is in your PATH

#### Compilation Steps

```bash
cd /path/to/gRASPA/src_clean

# Make the compilation script executable
chmod +x ../NVC_COMPILE

# Run compilation
../NVC_COMPILE
```

The `NVC_COMPILE` script will:
- Compile all source files with GPU acceleration
- Link necessary libraries
- Create the executable `nvc_main.x`

#### Compiler Flags Used
- `-O3`: Maximum optimization
- `-std=c++20`: C++20 standard
- `-target=gpu`: GPU target
- `-Minline`: Inline function optimization
- `-fopenmp`: OpenMP support
- `-cuda`: CUDA support
- `-stdpar=multicore`: Standard parallelism for multicore

### Method 2: Using nvcc (CUDA Compiler) - Experimental

**Note**: gRASPA is primarily designed for **nvc++** (NVIDIA HPC SDK). Compilation with **nvcc** may require code modifications and is not officially supported. If **nvc++** is not available, you can attempt compilation with **nvcc**, but be aware that you may encounter compatibility issues.

#### Prerequisites
- CUDA Toolkit installed
- `nvcc` available in PATH

#### Compilation Steps

```bash
cd /path/to/gRASPA/src_clean

# Make the compilation script executable
chmod +x ../NVCC_COMPILE

# Edit GPU architecture if needed (see below)
# Then run compilation
../NVCC_COMPILE
```

#### GPU Architecture Selection

Before compiling, you need to specify your GPU's compute capability. Edit `NVCC_COMPILE` and change the `GPU_ARCH` variable:

```bash
# Common GPU architectures:
# sm_75  - Turing (RTX 20xx, GTX 16xx)
# sm_80  - Ampere (A100, RTX 30xx)
# sm_86  - Ampere (RTX 30xx mobile)
# sm_89  - Ada Lovelace (RTX 40xx)
# sm_90  - Hopper (H100)

GPU_ARCH="sm_75"  # Change to match your GPU
```

To find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

#### Compiler Flags Used
- `-O3`: Maximum optimization
- `-std=c++17`: C++17 standard
- `-arch=sm_XX`: GPU architecture
- `-Xcompiler -fopenmp`: OpenMP support
- `-lcudart -lcurand -lcublas`: CUDA libraries

## Cluster-Specific Instructions

### NERSC (Perlmutter)

See detailed instructions in: `Cluster-Setup/NERSC_Installation_Nov2025/`

Quick start:
```bash
# Load modules
module load PrgEnv-nvidia
module load cuda/12.4

# Compile
cd src_clean
cp ../Cluster-Setup/NERSC_Installation_Nov2025/NVC_COMPILE_NERSC .
chmod +x NVC_COMPILE_NERSC
./NVC_COMPILE_NERSC
```

### Quest (Northwestern)

See detailed instructions in: `Cluster-Setup/Quest/`

Quick start:
```bash
# Submit compilation job
cd src_clean
cp ../Cluster-Setup/Quest/compile_graspa.job .
# Edit compile_graspa.job to set your account
sbatch compile_graspa.job
```

## Troubleshooting

### Error: "nvc++: command not found"
**Solution**: Install NVIDIA HPC SDK or use `nvcc` compilation method instead.

### Error: "nvcc: command not found"
**Solution**: Install CUDA Toolkit and ensure it's in your PATH.

### Error: "undefined reference to 'double3'"
**Solution**: Ensure CUDA headers are properly included. The code uses CUDA vector types (`double3`, `int2`, `double2`) which require CUDA runtime headers.

### Error: "compute capability mismatch"
**Solution**: Update the `-arch` flag in the compilation script to match your GPU's compute capability.

### Error: "multiple definition of 'PBC'"
**Solution**: This can occur when mixing compilation methods. Ensure all source files are compiled with the same compiler and flags.

### Compilation takes too long
**Solution**: 
- Reduce optimization level (change `-O3` to `-O2`)
- Compile in parallel if using make
- Check available disk space

### Executable runs but gives incorrect results
**Solution**:
- Verify GPU architecture matches compilation settings
- Check CUDA driver version compatibility
- Ensure all required libraries are linked

## Verification

After successful compilation, verify the executable:

```bash
# Check executable exists
ls -lh nvc_main.x

# Check it's a valid binary
file nvc_main.x

# Run a simple test (if you have test input files)
./nvc_main.x
```

## References

- **gRASPA Manual**: https://zhaoli2042.github.io/gRASPA-mkdoc/
- **Installation Guide**: https://zhaoli2042.github.io/gRASPA-mkdoc/Installation.html
- **NVIDIA HPC SDK**: https://developer.nvidia.com/hpc-sdk
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **GitHub Repository**: https://github.com/snurr-group/gRASPA

## Notes

- The default compilation script (`NVC_COMPILE`) uses `nvc++` which is part of NVIDIA HPC SDK
- For systems without NVIDIA HPC SDK, use the `NVCC_COMPILE` script with `nvcc`
- ML potential support (Allegro/PyTorch or LCLIN/TensorFlow) requires additional dependencies
- Always verify your GPU's compute capability before compiling
- For production runs, use `-O3` optimization; for debugging, use `-O0 -g`

## Support

For issues or questions:
1. Check the [gRASPA manual](https://zhaoli2042.github.io/gRASPA-mkdoc/)
2. Review cluster-specific setup guides in `Cluster-Setup/`
3. Check GitHub issues: https://github.com/snurr-group/gRASPA/issues

