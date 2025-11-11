#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --qos="regular"
#SBATCH -A YOUR_ACCOUNT

# Email notifications (optional)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Install TensorFlow C++ API and CppFlow for gRASPA
# This script runs the installation in a SLURM job
# 
# IMPORTANT: Update YOUR_ACCOUNT and YOUR_EMAIL above before submitting

set -e  # Exit on error

echo "=========================================="
echo "Installing TensorFlow C++ API and CppFlow"
echo "Job started at: $(date)"
echo "=========================================="
echo ""

# Check if TensorFlow is already installed
if [ -d "${HOME}/ctensorflow/lib" ] && ([ -f "${HOME}/ctensorflow/lib/libtensorflow.so" ] || [ -f "${HOME}/ctensorflow/lib/libtensorflow.so.2" ]); then
    echo "TensorFlow appears to be already installed in ${HOME}/ctensorflow"
    echo "Checking installation..."
    ls -lh ${HOME}/ctensorflow/lib/libtensorflow* 2>/dev/null || echo "TensorFlow library files not found"
    echo ""
    echo "If you want to reinstall, please remove ${HOME}/ctensorflow first"
    echo "Skipping TensorFlow installation."
    SKIP_TF=true
else
    SKIP_TF=false
fi

# Install TensorFlow C++ API
if [ "$SKIP_TF" = false ]; then
    echo "Step 1: Installing TensorFlow C++ API..."
    mkdir -p ${HOME}/ctensorflow
    cd ${HOME}/ctensorflow
    
    if [ ! -f "libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz" ]; then
        echo "Downloading TensorFlow C++ API..."
        wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
    else
        echo "TensorFlow archive already exists, skipping download."
    fi
    
    echo "Extracting TensorFlow..."
    tar -xvf libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
    
    echo "TensorFlow C++ API installed successfully!"
    echo "Verifying installation..."
    ls -lh ${HOME}/ctensorflow/lib/libtensorflow* 2>/dev/null || echo "Warning: Library files not found after extraction"
    echo ""
fi

# Install CppFlow
echo "Step 2: Installing CppFlow..."
cd ${HOME}

if [ -d "cppflow" ]; then
    echo "CppFlow directory already exists."
    echo "If you want to reinstall, please remove ${HOME}/cppflow first"
    echo "Skipping CppFlow installation."
    SKIP_CPPFLOW=true
else
    SKIP_CPPFLOW=false
fi

if [ "$SKIP_CPPFLOW" = false ]; then
    echo "Cloning CppFlow repository..."
    git clone https://github.com/serizba/cppflow
    
    cd cppflow
    mkdir -p build
    cd build
    
    echo "Configuring CppFlow with CMake..."
    cmake -DCMAKE_PREFIX_PATH=${HOME}/ctensorflow/ ..
    
    echo "Building and installing CppFlow..."
    make install DESTDIR=${HOME}/ctensorflow/
    
    echo "CppFlow installed successfully!"
    echo "Verifying installation..."
    ls -lh ${HOME}/ctensorflow/usr/local/include/cppflow* 2>/dev/null || echo "Warning: CppFlow headers not found"
    echo ""
fi

# Update environment variables
echo "Step 3: Updating environment variables..."
BASHRC="${HOME}/.bashrc"

if ! grep -q "ctensorflow/lib" "$BASHRC" 2>/dev/null; then
    echo "" >> "$BASHRC"
    echo "# TensorFlow C++ API for gRASPA" >> "$BASHRC"
    echo "export LIBRARY_PATH=\$LIBRARY_PATH:~/ctensorflow/lib" >> "$BASHRC"
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/ctensorflow/lib" >> "$BASHRC"
    echo "Environment variables added to ~/.bashrc"
    echo "Please run: source ~/.bashrc (or log out and back in)"
else
    echo "Environment variables already configured in ~/.bashrc"
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "Job finished at: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
if [ "$SKIP_TF" = true ]; then
    echo "  - TensorFlow: Already installed (skipped)"
else
    echo "  - TensorFlow: Installed in ${HOME}/ctensorflow"
fi
if [ "$SKIP_CPPFLOW" = true ]; then
    echo "  - CppFlow: Already installed (skipped)"
else
    echo "  - CppFlow: Installed in ${HOME}/ctensorflow/usr/local"
fi
echo ""
echo "Next steps:"
echo "1. Source your .bashrc: source ~/.bashrc"
echo "2. Apply filesystem fix to source code (see README.md)"
echo "3. Compile gRASPA using the install script"
echo ""

