# Quick Start Guide: gRASPA on Quest

> **Note**: This guide was AI-generated and tested on Quest cluster.

## Minimal Installation Steps

### 1. Copy Files
```bash
cd /path/to/gRASPA/src_clean
cp Cluster-Setup/Quest/NVC_COMPILE_QUEST_VANILLA .
cp Cluster-Setup/Quest/compile_graspa.job .
chmod +x NVC_COMPILE_QUEST_VANILLA
```

### 2. Edit Job Script
Edit `compile_graspa.job`:
- Change `YOUR_ACCOUNT_HERE` to your Quest account
- Update `GRASPA_SRC_DIR` to your source directory path

### 3. Submit Job
```bash
sbatch compile_graspa.job
```

### 4. Check Results
```bash
# Check job status
squeue -u $USER

# View output (replace JOBID with your job number)
tail -f compile_output.JOBID

# After completion, verify executable
ls -lh nvc_main.x
```

## That's It!

The executable `nvc_main.x` will be in your `src_clean/` directory.

For detailed instructions, see [README.md](README.md).

