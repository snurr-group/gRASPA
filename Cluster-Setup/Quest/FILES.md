# Files in Quest-Installation Directory

This directory contains all files needed to compile gRASPA (vanilla version) on Quest cluster.

## Core Files

### `NVC_COMPILE_QUEST_VANILLA`
- **Type**: Bash script
- **Purpose**: Main compilation script for gRASPA vanilla version
- **Usage**: Run directly or via job submission
- **Dependencies**: Requires nvhpc and CUDA modules

### `compile_graspa.job`
- **Type**: SLURM job script
- **Purpose**: Template for submitting compilation as a GPU job
- **Usage**: Edit account and paths, then `sbatch compile_graspa.job`
- **Note**: Update `YOUR_ACCOUNT_HERE` and `GRASPA_SRC_DIR` before use

## Documentation

### `README.md`
- **Type**: Markdown documentation
- **Purpose**: Comprehensive installation and usage guide
- **Contents**: 
  - Prerequisites
  - Step-by-step installation
  - Compilation details
  - Troubleshooting
  - Running instructions

### `QUICKSTART.md`
- **Type**: Markdown documentation
- **Purpose**: Minimal quick start guide
- **Contents**: Essential steps to get started quickly

### `FILES.md`
- **Type**: Markdown documentation (this file)
- **Purpose**: Overview of all files in this directory

## Ignored Files

The `.gitignore` file ensures that compilation artifacts and job output files are not tracked by git.

## File Structure

```
Quest-Installation/
├── NVC_COMPILE_QUEST_VANILLA    # Compilation script
├── compile_graspa.job            # Job submission template
├── README.md                      # Full documentation
├── QUICKSTART.md                  # Quick start guide
├── FILES.md                       # This file
└── .gitignore                     # Git ignore rules
```

## Usage Workflow

1. Copy `NVC_COMPILE_QUEST_VANILLA` to your `src_clean/` directory
2. Copy and edit `compile_graspa.job` (update account and paths)
3. Submit job: `sbatch compile_graspa.job`
4. Check results and use the generated `nvc_main.x` executable

For detailed instructions, see [README.md](README.md).

