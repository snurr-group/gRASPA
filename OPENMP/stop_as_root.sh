#!/bin/bash

echo quit | nvidia-cuda-mps-control
nvidia-smi -c DEFAULT
