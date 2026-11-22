#!/usr/bin/env bash
# Kick-the-tires run (~1 minute, no datasets needed): build BBQ, then on a 20k-vector synthetic corpus
#   1. FP32-built HNSW + 1-bit codes, ADC search without and with a fixed FP32 rerank (R = 40)
#   2. dump the calibration pool, fit tau with bbq/calibration/calibrate.py (target 0.95)
#   3. serve with the calibrated per-query budget R(q) = |C_tau(q)|
# Requires: source exp/env.sh (FAISS_ROOT/FAISS_BUILD/BLAS_LIB), python3 + numpy, AVX-512 CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="${SMOKE_DIR:-${ROOT}/build/smoke}"
BIN="${ROOT}/build/bbq_hnsw"
mkdir -p "${W}"

[[ -x "${BIN}" ]] || "${ROOT}/bbq/build.sh" search
python3 "${ROOT}/exp/smoke/make_data.py" "${W}"

C=(--dataset smoke --train "${W}/base.fbin" --index "${W}/smoke.faiss" --m 16 --ef-construction 100 --k 10
   --rounds 1 --build-threads 8 --search-threads 1 --adc-mode int16 --lut-mode nibble --search-impl inline --graph-build fp)
T=(--query "${W}/test.query.fbin" --gt "${W}/test.gt.ibin")

echo "== 1. ADC-BQ traversal, no rerank vs fixed R=40"
"${BIN}" "${C[@]}" "${T[@]}" --out "${W}/no_rerank.csv" --efs 128 --rerank-n 128 --no-rerank | grep -o 'ef=.*recall@10=[0-9.]*'
"${BIN}" "${C[@]}" "${T[@]}" --out "${W}/fixed_r40.csv" --efs 128 --rerank-n 40 | grep -o 'ef=.*recall@10=[0-9.]*'

echo "== 2. calibration pool (1000 queries) -> tau"
"${BIN}" "${C[@]}" --query "${W}/cal.query.fbin" --gt "${W}/cal.gt.ibin" --out "${W}/pool.summary.csv" \
  --efs 128 --rerank-n 128 --candidate-only --no-rerank --candidates-out "${W}/pool.csv" > /dev/null
python3 "${ROOT}/bbq/calibration/calibrate.py" --pool "${W}/pool.csv" --queries "${W}/cal.query.fbin" \
  --base "${W}/base.fbin" --ef 128 --target 0.95 --out "${W}/tau.json" > /dev/null
TAU=$(python3 -c "import json; print(json.load(open('${W}/tau.json'))['tau'])")
echo "tau = ${TAU}"

echo "== 3. calibrated query-adaptive rerank"
"${BIN}" "${C[@]}" "${T[@]}" --out "${W}/calibrated.csv" --efs 128 --rerank-n 128 \
  --calibrated-threshold-rerank-tau "${TAU}" | grep -o 'recall@10=.*'

echo "== 4. same calibrated rerank, FP32 vectors read from disk with O_DIRECT (Fig. 16 modes)"
# The first run writes a 4 KiB-aligned copy of the base vectors next to it (base.fp32.header4096.fbin).
# SMOKE_DIR must be on a filesystem that supports O_DIRECT (not tmpfs).
for mode in "sequential:--fp-rerank-source odirect" \
            "async-batch:--fp-rerank-source odirect-uring --fp-rerank-qd 128" \
            "async-batch+overlap:--fp-rerank-source odirect-uring --fp-rerank-qd 128 --fp-rerank-overlap-queries"; do
  # shellcheck disable=SC2086
  "${BIN}" "${C[@]}" "${T[@]}" --out "${W}/ssd_${mode%%:*}.csv" --efs 128 --rerank-n 128 \
    --calibrated-threshold-rerank-tau "${TAU}" --fp-rerank-path "${W}/base.fp32.header4096.fbin" ${mode#*:} > /dev/null
  python3 -c "import csv,sys; r=next(csv.DictReader(open(sys.argv[1]))); \
print(f\"{sys.argv[2]:<20} Recall@10={float(r['recall_at_k']):.4f}  QPS={float(r['qps']):8.1f}  4KiB pages/query={float(r['rerank_read_bytes_mean'])/int(r['nq'])/4096:.1f}\")" \
    "${W}/ssd_${mode%%:*}.csv" "${mode%%:*}"
done

python3 - "${W}/calibrated.csv" <<'PY'
import csv, sys
r = next(csv.DictReader(open(sys.argv[1])))
rec, rbar = float(r["recall_at_k"]), float(r["adaptive_rerank_count_mean"])
print(f"calibrated: Recall@10 = {rec:.4f}, mean R(q) = {rbar:.1f} (min {r['adaptive_rerank_count_min']}, max {r['adaptive_rerank_count_max']})")
sys.exit(0 if rec >= 0.95 else 1)
PY
echo "SMOKE OK"
