#!/usr/bin/env python3
# target_sweep.csv -> target_sweep.dat (target, test coverage, final recall@10, mean FP reads)
import csv, sys
src = sys.argv[1] if len(sys.argv) > 1 else 'target_sweep.csv'
rows = sorted(csv.DictReader(open(src)), key=lambda r: float(r['target_pool_top10_coverage']))
with open('target_sweep.dat', 'w') as f:
    f.write('# from %s\n# target test_pool_top10_coverage final_mean_recall_at_10 mean_fp_reads\n' % src)
    for r in rows:
        f.write('%s %.6f %.6f %.3f\n' % (r['target_pool_top10_coverage'], float(r['test_empirical_pool_top10_coverage']),
                                       float(r['final_mean_recall_at_10']), float(r['mean_fp_reads'])))
