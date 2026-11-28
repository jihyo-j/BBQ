#!/usr/bin/env bash
# Build the BBQ binaries.
#
#   ./bbq/build.sh               -> build/bbq_hnsw           (search, calibration pool dump, SSD rerank)
#   ./bbq/build.sh dynamic       -> build/bbq_dynamic_insert (online insertion: fp / bq / adcbq graph add)
#   BBQ_WITH_TQ=1 ./bbq/build.sh dynamic   also links the TurboQuant-1bit baseline (--online-graph-add tq1, Fig 18);
#                                needs cargo and the patched Qdrant checkout (exp/tools/setup_workspace.sh)
#   ./bbq/build.sh test          -> build/test_adcbq_sign_packing (AVX-512 bit-packing unit test)
#
# Requirements: g++ >= 11 (C++17), OpenMP, liburing, OpenBLAS, upstream FAISS 46def469 built with AVX2,
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
    # SRC: an experiment variant of the same file (e.g. the query bit-width build in exp/scripts).
    # ADCBQ_FORCE_FP32_DIRECT_SCALAR=1: scalar FP32 ADC without LUT or SIMD, for the LUT ablation only.
    out="${OUT:-${OUT_DIR}/bbq_hnsw}"
    src="${SRC:-${HERE}/src/bbq_hnsw.cpp}"
    defs=(-DADCBQ_PAGED_INDEX_PAGE_BYTES="${PAGE_BYTES}")
    [[ "${ADCBQ_FORCE_FP32_DIRECT_SCALAR:-0}" == "1" ]] && defs+=(-DADCBQ_FORCE_FP32_DIRECT_SCALAR=1)
    "${CXX}" "${COMMON[@]}" "${defs[@]}" \
      "${src}" "${LIBDIRS[@]}" -Wl,-rpath,"${RPATH}" \
      -lfaiss_avx2 -lopenblas -luring -o "${out}"
    ;;
  dynamic)
    out="${OUT_DIR}/bbq_dynamic_insert"
    if [[ "${BBQ_WITH_TQ:-0}" == "1" ]]; then
      TQ_FFI="${ROOT}/exp/baselines/qdrant_tq_ffi"
      (cd "${TQ_FFI}" && cargo build --release)
      "${CXX}" "${COMMON[@]}" -DBBQ_WITH_TQ=1 -I"${ROOT}/exp/baselines" \
        "${HERE}/src/bbq_dynamic_insert.cpp" "${LIBDIRS[@]}" -L"${TQ_FFI}/target/release" \
        -Wl,-rpath,"${RPATH}:${TQ_FFI}/target/release" \
        -lfaiss_avx2 -lopenblas -luring -lqdrant_tq_ffi -o "${out}"
    else
      "${CXX}" "${COMMON[@]}" -DBBQ_WITH_TQ=0 \
        "${HERE}/src/bbq_dynamic_insert.cpp" "${LIBDIRS[@]}" -Wl,-rpath,"${RPATH}" \
        -lfaiss_avx2 -lopenblas -luring -o "${out}"
    fi
    ;;
  test)
    out="${OUT_DIR}/test_adcbq_sign_packing"
    "${CXX}" -std=c++17 -O2 -march=native -I"${HERE}/src" "${HERE}/tests/test_adcbq_sign_packing.cpp" -o "${out}"
    ;;
  *) echo "usage: $0 [search|dynamic|test]" >&2; exit 2 ;;
esac
echo "Built ${out}"
