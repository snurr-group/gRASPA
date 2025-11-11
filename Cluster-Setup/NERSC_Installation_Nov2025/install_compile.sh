#!/bin/bash
#SBATCH --job-name=compile_graspa
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gpus-per-node=1
#SBATCH --qos="regular"
#SBATCH --licenses="SCRATCH"
#SBATCH -A YOUR_ACCOUNT

# Email notifications (optional)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Compilation script for gRASPA on NERSC Perlmutter
# 
# IMPORTANT: 
# 1. Update YOUR_ACCOUNT and YOUR_EMAIL above before submitting
# 2. Update the SOURCE_DIR path below to point to your gRASPA src_clean directory

SOURCE_DIR="${HOME}/gRASPA/src_clean"  # Update this path to your gRASPA source directory

# Change to the source directory
cd ${SOURCE_DIR}

# Make sure the compilation script is executable
chmod +x NVC_COMPILE_NERSC

# Run the compilation script
./NVC_COMPILE_NERSC

