#!/usr/bin/env python3
"""preserve_lut.dat (Fig. 7(b)) <- paired_ef_comparison_t1.csv of the Quora BGE-M3 LUT ablation (argv[1]).
Values copied, not recomputed: efSearch, median QPS and Recall@10 for the scalar (no LUT) and LUT builds."""
import csv, sys
rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: int(r['efSearch']))
print('# from quora_bge1024_adcbq_scalar_vs_lut_no_calibration_t1/paired_ef_comparison_t1.csv')
print('# efSearch qps recall@10')
for name, q, rc in (('nolut', 'no_lut_qps_median', 'no_lut_recall_at_10'), ('lut', 'lut_qps_median', 'lut_recall_at_10')):
    print('\n\n# ' + name)
    for r in rows:
        print(r['efSearch'], r[q], r[rc])
