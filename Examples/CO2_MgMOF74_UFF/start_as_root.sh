#!/bin/bash

# the following must be performed with root privilege
export CUDA_VISIBLE_DEVICES="1"
# Alvaro suggests that he does not need this following line to run on ALCF
#nvidia-smi -c EXCLUSIVE_PROCESS
#nvidia-cuda-mps-control -d
# no -d? Try it with/without.
# -d is probably for using all the visible devices
nvidia-cuda-mps-control -d
