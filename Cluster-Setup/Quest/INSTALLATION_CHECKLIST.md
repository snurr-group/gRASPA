# Installation Checklist

Use this checklist to ensure a successful gRASPA installation on Quest.

## Pre-Installation

- [ ] Have Quest account and project ID ready
- [ ] Have access to GPU nodes on Quest
- [ ] Located gRASPA source code directory (`src_clean/`)

## Installation Steps

- [ ] Copied `NVC_COMPILE_QUEST_VANILLA` to `src_clean/` directory
- [ ] Made compilation script executable: `chmod +x NVC_COMPILE_QUEST_VANILLA`
- [ ] Copied `compile_graspa.job` to `src_clean/` directory
- [ ] Edited `compile_graspa.job`:
  - [ ] Updated `--account=YOUR_ACCOUNT_HERE` with your Quest account
  - [ ] Updated `GRASPA_SRC_DIR` with correct path
- [ ] Submitted compilation job: `sbatch compile_graspa.job`
- [ ] Monitored job: `squeue -u $USER`
- [ ] Checked output: `tail -f compile_output.JOBID`

## Verification

- [ ] Job completed successfully
- [ ] Executable `nvc_main.x` exists in `src_clean/` directory
- [ ] Executable size is approximately 4-5 MB
- [ ] Verified executable: `file nvc_main.x` shows ELF 64-bit executable

## Post-Installation

- [ ] Created run script for gRASPA simulations (if needed)
- [ ] Tested running gRASPA with a sample input file
- [ ] Documented any custom paths or configurations

## Troubleshooting

If compilation fails:
- [ ] Checked error log: `cat compile_error.JOBID`
- [ ] Verified modules are available: `module avail nvhpc`
- [ ] Confirmed job ran on GPU node
- [ ] Verified all source files are present

## Notes

Add any custom notes or configurations here:
_____________________________________________
_____________________________________________
_____________________________________________

