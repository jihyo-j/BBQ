#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RABITQ_BUILD="${REPO_ROOT}/experiment/rabitq_v051_adaptive_odirect_build"

g++ \
  -std=c++17 \
  -O3 \
  -DNDEBUG \
  -march=native \
  -fopenmp \
  -I"${RABITQ_BUILD}/include" \
  "${SCRIPT_DIR}/rabitq_paged_adaptive_benchmark.cpp" \
  "${RABITQ_BUILD}/lib/librabitq_core.a" \
  -luring \
  -o "${SCRIPT_DIR}/rabitq_paged_adaptive_benchmark"

echo "Built ${SCRIPT_DIR}/rabitq_paged_adaptive_benchmark"
