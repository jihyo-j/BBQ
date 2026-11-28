#!/usr/bin/env python3
"""Build colocation.dat (Fig 19) from the te3-small colocation run.

Input : ../results/te3small_ef256_t24/summary.csv  (run_msmarco_te3small1536_adcbq_colocation.py;
        median QPS of 3 fresh measured processes, warm-up process discarded)
Output: colocation.dat, gnuplot index blocks k{4,8,16}{co,base}_{a,b}_{curve,pts}
        (a) x = DRAM ratio, y = median QPS;  (b) x = DRAM ratio, y = graph page hit ratio (%)
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/te3small_ef256_t24/summary.csv"
OUT = HERE / "colocation.dat"

rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else SRC)))
series = {}
for r in rows:
    key = (int(r["page_bytes"]) // 1024, "co" if r["layout"] == "coloc" else "base")
    series.setdefault(key, []).append((float(r["bq_ram_ratio_target"]), float(r["qps_median"]),
                                       100.0 * float(r["graph_page_hit_ratio"])))

with open(OUT, "w") as f:
    f.write(f"# built by make_colocation_dat.py from {SRC.relative_to(HERE.parent)}\n")
    for panel, col in (("a", 1), ("b", 2)):
        for kib in (4, 8, 16):
            for lay in ("co", "base"):
                pts = sorted(series[(kib, lay)])
                assert len(pts) == 4, (kib, lay, pts)
                for kind in ("curve", "pts"):
                    f.write(f"\n\n# k{kib}{lay}_{panel}_{kind}\n")
                    for p in pts:
                        f.write(f"{p[0]:g} {p[col]:.6g}\n")
print(f"wrote {OUT}")
