# Quest Installation Package Summary

> **Note**: This installation package was AI-generated and has been tested on Quest cluster. All scripts and documentation are provided as-is.

This directory contains a complete, ready-to-use installation package for compiling gRASPA (vanilla version) on Quest cluster at Northwestern University.

## Package Contents

### Core Scripts
- **`NVC_COMPILE_QUEST_VANILLA`** - Main compilation script (3.4 KB)
- **`compile_graspa.job`** - SLURM job submission template (2.5 KB)

### Documentation
- **`README.md`** - Comprehensive installation guide (5.4 KB)
- **`QUICKSTART.md`** - Quick start guide for experienced users (819 bytes)
- **`FILES.md`** - File structure and overview (2.0 KB)
- **`INSTALLATION_CHECKLIST.md`** - Step-by-step checklist
- **`SUMMARY.md`** - This file

### Configuration
- **`.gitignore`** - Git ignore rules for compilation artifacts

## Quick Overview

### What This Package Does
Compiles gRASPA vanilla version (without ML support) on Quest GPU nodes using NVIDIA HPC SDK compiler.

### What You Need
- Quest account with GPU access
- gRASPA source code in `src_clean/` directory
- 2-5 minutes for compilation

### What You Get
- Compiled executable: `nvc_main.x` (~4-5 MB)
- Ready to run gRASPA simulations

## Installation Time
- **Setup**: 2 minutes
- **Compilation**: 1-2 minutes (on GPU node)
- **Total**: ~5 minutes

## Tested Configuration
- **Cluster**: Quest, Northwestern University
- **NVIDIA HPC SDK**: 21.9-gcc (primary), 24.1-gcc-10.4.0 (fallback)
- **CUDA**: 11.2.2-gcc (primary), 12.1.0-openmpi-4.1.4 (fallback)
- **Status**: ✅ Successfully tested and verified

## File Sizes
- Total package size: ~20 KB
- All files are text-based (scripts and documentation)
- No binary files included

## Next Steps After Installation

1. **Test the executable**: Run a simple gRASPA simulation
2. **Create run scripts**: Set up job scripts for production runs
3. **Configure resources**: Adjust memory/time limits based on your needs

## Support

For issues or questions:
- Check `README.md` for detailed troubleshooting
- Review `INSTALLATION_CHECKLIST.md` for step-by-step verification
- Consult Quest IT support: https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/

## License

These installation scripts are provided as-is for use with gRASPA. Please refer to the main gRASPA repository for license information.

