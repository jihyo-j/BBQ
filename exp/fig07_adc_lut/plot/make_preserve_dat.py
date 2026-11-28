#!/usr/bin/env python3
"""Build preserve.dat (Fig 7(a): CandidateRecall@R under Sym-BQ and ADC-BQ, MS MARCO BGE-M3).

    make_preserve_dat.py [<key>_perq.npz]     (default: sec2_3_analysis/harness/work/msmarco_bgem3_perq.npz)

The per-query file is written by the Sec 2-3 harness (harness/src/measure.py msmarco_bgem3): exhaustive scan of the
1M corpus for 2,000 queries; ranks_<arm>[q, j] is the position, in the arm's code-score order, of the j-th exact FP32
top-10 neighbour of query q.  CandidateRecall@R = mean over queries of |{j : ranks <= R}| / 10.
Output blocks sym_a / adc_a: R, CandidateRecall@R on the printed grid.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "sec2_3_analysis/harness/work/msmarco_bgem3_perq.npz"
GRID = (10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640)
z = np.load(SRC)
out = [f"# built by make_preserve_dat.py from {SRC.name} ({z['ranks_adc_bq'].shape[0]} queries)", "# R  CandidateRecall@R"]
for arm, blk in (("sym_bq", "sym_a"), ("adc_bq", "adc_a")):
    rk = z[f"ranks_{arm}"]
    out += ["", "", f"# {blk}"] + [f"{r} {float((rk <= r).mean()):.6f}" for r in GRID]
(HERE / "preserve.dat").write_text("\n".join(out) + "\n")
print(f"wrote {HERE / 'preserve.dat'}")
