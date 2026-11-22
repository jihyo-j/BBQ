#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX="${CXX:-g++}"

"${CXX}" \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -march=native \
  -I"${SCRIPT_DIR}" \
  "${SCRIPT_DIR}/validate_qdrant_tq_official.cpp" \
  -L"${SCRIPT_DIR}/qdrant_tq_ffi/target/release" \
  -Wl,-rpath,"${SCRIPT_DIR}/qdrant_tq_ffi/target/release" \
  -lqdrant_tq_ffi \
  -o "${SCRIPT_DIR}/validate_qdrant_tq_official"

echo "Built ${SCRIPT_DIR}/validate_qdrant_tq_official"
