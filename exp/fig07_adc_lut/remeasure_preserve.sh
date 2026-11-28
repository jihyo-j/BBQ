#!/usr/bin/env bash
# Re-measure Fig 7(a) (CandidateRecall@R, Sym-BQ vs ADC-BQ, MS MARCO BGE-M3) from the raw vectors.
#   needs: BGE-M3 MS MARCO vectors $BBQETA_SYN/msmarco-1m__vanilla_{docs,queries}.npy (1M x 1024 / queries, float32),
#          python3 with numpy + scipy + faiss (the Sec 2-3 harness), <= BBQETA_THREADS threads (default 24).
#   writes: $OUTDIR/work/msmarco_bgem3_perq.npz, $OUTDIR/out/msmarco_bgem3.json, then plot/preserve.dat and preserve.pdf.
# The harness scans all 1M documents exhaustively for 2,000 queries (seed 20261116) for every quantizer arm.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${OUTDIR:-$HERE/remeasure}"
export BBQETA_WORK="$OUTDIR/work" BBQETA_OUT="$OUTDIR/out"
mkdir -p "$BBQETA_WORK" "$BBQETA_OUT"
: "${BBQETA_SYN:?set BBQETA_SYN to the directory holding msmarco-1m__vanilla_docs.npy / _queries.npy}"
python3 "$HERE/../sec2_3_analysis/harness/src/measure.py" msmarco_bgem3
python3 "$HERE/plot/make_preserve_dat.py" "$BBQETA_WORK/msmarco_bgem3_perq.npz"
cd "$HERE/plot" && gnuplot preserve.gp
echo "Fig 7(a) re-measured -> $HERE/plot/preserve.pdf"
