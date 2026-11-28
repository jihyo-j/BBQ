#!/usr/bin/env bash
# Recreate the layout the runners expect: they resolve REPO = exp/ and address everything as
# REPO/"experiment/<name>" (scripts, C++ sources, binaries, result dirs). This builds exp/experiment/
# from symlinks and compiles the BBQ binaries under the names the runners call. Needs: source exp/env.sh
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; ROOT="$(cd "${EXP}/.." && pwd)"
W="${EXP}/experiment"; mkdir -p "${W}/adcbq_optimized"

for f in "${EXP}"/scripts/*; do ln -sfn "$f" "${W}/$(basename "$f")"; done
ln -sfn "${ROOT}/bbq/src/bbq_hnsw.cpp"            "${W}/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
ln -sfn "${ROOT}/bbq/src/bbq_hnsw.cpp"            "${W}/bbq_hnsw.cpp"
ln -sfn "${ROOT}/bbq/src/adcbq_sign_packing.hpp"  "${W}/adcbq_sign_packing.hpp"
ln -sfn "${ROOT}/bbq/src/bbq_dynamic_insert.cpp"  "${W}/adcbq_fpgraph_dynamic_5m.cpp"
ln -sfn "${EXP}/baselines/qdrant_tq_ffi"          "${W}/qdrant_tq_ffi"
for f in "${EXP}"/baselines/patches/qdrant/*.patch; do ln -sfn "$f" "${W}/$(basename "$f")"; done

# RaBitQ headers for the FAISS-based drivers: they compile with -I REPO/include (REPO = exp/).
if [[ ! -d "${EXP}/include/rabitqlib" ]]; then
  base=$(sed -n 's/^commit: //p' "${EXP}/baselines/patches/rabitq-library/BASE")
  dep="${W}/_deps/RaBitQ-Library"
  [[ -d "${dep}" ]] || git clone -q https://github.com/VectorDB-NTU/RaBitQ-Library.git "${dep}"
  git -C "${dep}" checkout -q "${base}"
  git -C "${dep}" apply "${EXP}/baselines/patches/rabitq-library/include.patch"
  ln -sfn "${dep}/include" "${EXP}/include"
fi

# Patched Qdrant (TurboQuant codec) for the Rust FFI in exp/baselines/qdrant_tq_ffi (Cargo path ../qdrant).
# Reuses $QDRANT_ROOT if it already is the patched checkout; otherwise fetches the pinned commit and patches it.
Q="${EXP}/baselines/qdrant"
if [[ ! -e "${Q}" ]]; then
  qbase=$(sed -n 's/^commit: //p' "${EXP}/baselines/patches/qdrant/BASE")
  if [[ -n "${QDRANT_ROOT:-}" && -d "${QDRANT_ROOT}/.git" && "$(git -C "${QDRANT_ROOT}" rev-parse HEAD)" == "${qbase}" ]]; then
    ln -sfn "${QDRANT_ROOT}" "${Q}"
  else
    git init -q "${W}/_deps/qdrant" && git -C "${W}/_deps/qdrant" fetch -q --depth 1 https://github.com/qdrant/qdrant.git "${qbase}"
    git -C "${W}/_deps/qdrant" checkout -q FETCH_HEAD
    git -C "${W}/_deps/qdrant" apply "${EXP}/baselines/patches/qdrant/qdrant.patch"
    ln -sfn "${W}/_deps/qdrant" "${Q}"
  fi
fi

# experiment/adcbq_optimized/build.sh (called by runners with OUT=<binary>)
cat > "${W}/adcbq_optimized/build.sh" <<SH
#!/usr/bin/env bash
exec "${ROOT}/bbq/build.sh" search
SH
chmod +x "${W}/adcbq_optimized/build.sh"
cat > "${W}/build_adcbq_fpgraph_dynamic_101m.sh" <<SH
#!/usr/bin/env bash
BBQ_WITH_TQ=1 OUT_DIR="${W}" "${ROOT}/bbq/build.sh" dynamic && mv -f "${W}/bbq_dynamic_insert" "${W}/adcbq_fpgraph_dynamic_101m"
SH
chmod +x "${W}/build_adcbq_fpgraph_dynamic_101m.sh"

# The paper runs used these binary names; all are built from bbq/src/bbq_hnsw.cpp.
for name in faiss_adcbq_hnsw faiss_adcbq_hnsw_calibrated faiss_adcbq_hnsw_same_pool_calibration faiss_adcbq_hnsw_equal_pool_qd; do
  OUT="${W}/adcbq_optimized/${name}" "${ROOT}/bbq/build.sh" search
done
for p in 4096:4k 8192:8k 16384:16k; do
  PAGE_BYTES="${p%%:*}" OUT="${W}/adcbq_optimized/faiss_adcbq_hnsw_${p##*:}_graph_batchcache_calibrated" "${ROOT}/bbq/build.sh" search
done
# Ablation builds (2026-11-23): LUT on/off (Sec. 5.1) and query bit-width (Sec. 3.1, all-DRAM comparison)
OUT="${W}/adcbq_optimized/faiss_adcbq_hnsw_lut_ablation" "${ROOT}/bbq/build.sh" search
ADCBQ_FORCE_FP32_DIRECT_SCALAR=1 OUT="${W}/adcbq_optimized/faiss_adcbq_hnsw_scalar_direct_ablation" "${ROOT}/bbq/build.sh" search
SRC="${EXP}/scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp" OUT="${W}/adcbq_optimized/faiss_adcbq_hnsw_intquery" \
  "${ROOT}/bbq/build.sh" search
cat > "${W}/adcbq_optimized/build_intquery.sh" <<SH
#!/usr/bin/env bash
SRC="${EXP}/scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp" OUT="\${OUT:-${W}/adcbq_optimized/faiss_adcbq_hnsw_intquery}" exec "${ROOT}/bbq/build.sh" search
SH
chmod +x "${W}/adcbq_optimized/build_intquery.sh"
cp -f "${W}/adcbq_optimized/faiss_adcbq_hnsw" "${W}/faiss_fpgraph_packed_asym_bq_hnsw"
echo "workspace ready: ${W}"
