#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RABITQ_BUILD="${REPO_ROOT}/experiment/rabitq_v051_adaptive_odirect_build"

for page_kib in 4 8 16; do
  page_bytes=$((page_kib * 1024))

  PAGE_BYTES="${page_bytes}" \
  OUT="${SCRIPT_DIR}/adcbq_optimized/faiss_adcbq_hnsw_${page_kib}k_graph_async" \
    "${SCRIPT_DIR}/adcbq_optimized/build.sh"

  g++ \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -DRABITQ_PAGED_INDEX_PAGE_BYTES="${page_bytes}" \
    -march=native \
    -fopenmp \
    -I"${RABITQ_BUILD}/include" \
    "${SCRIPT_DIR}/rabitq_paged_adaptive_benchmark.cpp" \
    "${RABITQ_BUILD}/lib/librabitq_core.a" \
    -luring \
    -o "${SCRIPT_DIR}/rabitq_paged_adaptive_${page_kib}k_benchmark"
  echo "Built ${SCRIPT_DIR}/rabitq_paged_adaptive_${page_kib}k_benchmark"
done
