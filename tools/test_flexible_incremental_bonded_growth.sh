#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin/nvc++}"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/graspa-flex-bonded-growth.XXXXXX")"
trap 'rm -rf -- "${RUN_DIR}"' EXIT

"${CXX}" -std=c++20 -cuda -O2 \
  -I "${ROOT_DIR}/src_clean" \
  "${ROOT_DIR}/tools/test_flexible_incremental_bonded_growth.cpp" \
  "${ROOT_DIR}/src_clean/flexible_incremental_bonded_growth.cpp" \
  -o "${RUN_DIR}/test_flexible_incremental_bonded_growth"

"${RUN_DIR}/test_flexible_incremental_bonded_growth"
