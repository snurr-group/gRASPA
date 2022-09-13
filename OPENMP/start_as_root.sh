#!/bin/bash

# the following must be performed with root privilege
export CUDA_VISIBLE_DEVICES="0,1"
#export CUDA_VISIBLE_DEVICES=
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
