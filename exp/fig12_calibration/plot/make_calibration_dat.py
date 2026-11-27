#!/usr/bin/env python3
# calibration.dat for calibration.gp (Sec 6.5, MS MARCO Qwen-1024, ef=256)
#   index "decomp": recall@10 loss split at target 0.99 -> L_visit, L_select
#   index "sweep" : target, pool top-10 preserved, final recall@10, mean FP reads
# usage: make_calibration_dat.py <results/legacy_gt | results/exact_cosine_aligned> > calibration.dat
# The paper uses results/legacy_gt so that Recall@10 matches Fig 11 (0.9897) and the rest of the paper.
#   L_visit  = 1 - Recall@10 after exact FP32 rerank of the *entire* pool (loss refinement cannot recover)
#   L_select = loss added by calibrated selection within the pool
# Under legacy GT, L_visit also absorbs GT queries whose 10th neighbor is in the pool but not
# reproduced by exact rerank (147 queries); the exact-cosine GT has none.
import csv, sys, os

R = sys.argv[1] if len(sys.argv) > 1 else '../results/legacy_gt'

q = list(csv.DictReader(open(os.path.join(R, 'per_query_target099.csv'))))
n = len(q)
inpool = [int(r['global_gt10_in_candidate_pool']) for r in q]
full = [float(r['full_pool_exact_recall_at_10']) for r in q]
final = [float(r['final_recall_at_10']) for r in q]
mismatch = sum(a / 10 - f > 1e-9 for a, f in zip(inpool, full))
l_visit = sum(1 - f for f in full) / n
l_select = sum(f - g for f, g in zip(full, final)) / n
q_visit = sum(f < 1 - 1e-9 for f in full)
q_select = sum(f - g > 1e-9 for f, g in zip(full, final))
recall = sum(final) / n
assert abs((1 - recall) - (l_visit + l_select)) < 1e-9

print('# from %s (%d test queries, target 0.99)' % (R, n))
print('# queries with pool-absent GT neighbors = %d, in-pool GT not reproduced by exact rerank = %d'
      % (sum(a < 10 for a in inpool), mismatch))
print('# mean recall@10 = %.4f, 1-recall = %.5f' % (recall, 1 - recall))
print('# x name loss queries')
print('# decomp')
print('0 visit %.6f %d' % (l_visit, q_visit))
print('1 select %.6f %d' % (l_select, q_select))
print('\n')

rows = sorted(csv.DictReader(open(os.path.join(R, 'target_sweep.csv'))),
              key=lambda r: float(r['target_pool_top10_coverage']))
print('# sweep')
print('# target pool_top10_preserved final_recall_at_10 mean_fp_reads')
for r in rows:
    print('%s %.6f %.6f %.3f' % (r['target_pool_top10_coverage'], float(r['test_empirical_pool_top10_coverage']),
                                 float(r['final_mean_recall_at_10']), float(r['mean_fp_reads'])))
