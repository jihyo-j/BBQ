#!/usr/bin/env bash
# Build the BBQ binaries.
#
#   ./bbq/build.sh               -> build/bbq_hnsw           (search, calibration pool dump, SSD rerank)
#   ./bbq/build.sh dynamic       -> build/bbq_dynamic_insert (online insertion; needs the TurboQuant FFI)
#   ./bbq/build.sh test          -> build/test_adcbq_sign_packing (AVX-512 bit-packing unit test)
#
# Requirements: g++ >= 11 (C++17), OpenMP, liburing, OpenBLAS, FAISS 1.14.x built with AVX2,
# and an AVX-512 CPU (-march=native; the paper used a Xeon Gold 6442Y).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/build}"
FAISS_ROOT="${FAISS_ROOT:?set FAISS_ROOT to the FAISS source tree}"
FAISS_BUILD="${FAISS_BUILD:-${FAISS_ROOT}/build/faiss}"
BLAS_LIB="${BLAS_LIB:-}"                 # directory holding libopenblas, if not on the default path
PAGE_BYTES="${PAGE_BYTES:-4096}"         # graph page size for the memory-constrained mode (4096/8192/16384)
CXX="${CXX:-g++}"

mkdir -p "${OUT_DIR}"
LIBDIRS=(-L"${FAISS_BUILD}")
RPATH="${FAISS_BUILD}"
if [[ -n "${BLAS_LIB}" ]]; then LIBDIRS+=(-L"${BLAS_LIB}"); RPATH="${RPATH}:${BLAS_LIB}"; fi
COMMON=(-std=c++17 -O3 -DNDEBUG -march=native -fopenmp -I"${FAISS_ROOT}" -I"${FAISS_BUILD}" -I"${HERE}/src")

case "${1:-search}" in
  search)
    out="${OUT:-${OUT_DIR}/bbq_hnsw}"
    "${CXX}" "${COMMON[@]}" -DADCBQ_PAGED_INDEX_PAGE_BYTES="${PAGE_BYTES}" \
      "${HERE}/src/bbq_hnsw.cpp" "${LIBDIRS[@]}" -Wl,-rpath,"${RPATH}" \
      -lfaiss_avx2 -lopenblas -luring -o "${out}"
    ;;
  dynamic)
    TQ_FFI="${ROOT}/exp/baselines/qdrant_tq_ffi"
    (cd "${TQ_FFI}" && cargo build --release)
    out="${OUT_DIR}/bbq_dynamic_insert"
    "${CXX}" "${COMMON[@]}" -I"${ROOT}/exp/baselines" \
      "${HERE}/src/bbq_dynamic_insert.cpp" "${LIBDIRS[@]}" -L"${TQ_FFI}/target/release" \
      -Wl,-rpath,"${RPATH}:${TQ_FFI}/target/release" \
      -lfaiss_avx2 -lopenblas -luring -lqdrant_tq_ffi -o "${out}"
    ;;
  test)
    out="${OUT_DIR}/test_adcbq_sign_packing"
    "${CXX}" -std=c++17 -O2 -march=native -I"${HERE}/src" "${HERE}/tests/test_adcbq_sign_packing.cpp" -o "${out}"
    ;;
  *) echo "usage: $0 [search|dynamic|test]" >&2; exit 2 ;;
esac
echo "Built ${out}"
