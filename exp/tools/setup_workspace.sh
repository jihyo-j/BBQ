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

# experiment/adcbq_optimized/build.sh (called by runners with OUT=<binary>)
cat > "${W}/adcbq_optimized/build.sh" <<SH
#!/usr/bin/env bash
exec "${ROOT}/bbq/build.sh" search
SH
chmod +x "${W}/adcbq_optimized/build.sh"
cat > "${W}/build_adcbq_fpgraph_dynamic_101m.sh" <<SH
#!/usr/bin/env bash
OUT_DIR="${W}" "${ROOT}/bbq/build.sh" dynamic && mv -f "${W}/bbq_dynamic_insert" "${W}/adcbq_fpgraph_dynamic_101m"
SH
chmod +x "${W}/build_adcbq_fpgraph_dynamic_101m.sh"

# The paper runs used these binary names; all are built from bbq/src/bbq_hnsw.cpp.
for name in faiss_adcbq_hnsw faiss_adcbq_hnsw_calibrated faiss_adcbq_hnsw_same_pool_calibration faiss_adcbq_hnsw_equal_pool_qd; do
  OUT="${W}/adcbq_optimized/${name}" "${ROOT}/bbq/build.sh" search
done
for p in 4096:4k 8192:8k 16384:16k; do
  PAGE_BYTES="${p%%:*}" OUT="${W}/adcbq_optimized/faiss_adcbq_hnsw_${p##*:}_graph_batchcache_calibrated" "${ROOT}/bbq/build.sh" search
done
cp -f "${W}/adcbq_optimized/faiss_adcbq_hnsw" "${W}/faiss_fpgraph_packed_asym_bq_hnsw"
echo "workspace ready: ${W}"
