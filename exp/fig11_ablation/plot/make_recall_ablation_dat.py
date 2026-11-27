# recall_ablation: Recall@10 vs QPS / 4 KB reads  (MS MARCO, Qwen3-0.6B, 1M)
import csv, sys
SRC = 'msmarco_qwen1024_recall_ablation_t1.csv'
order = ['Sym-BQ', 'ADC-BQ', 'ADC-BQ + Fixed R=75', 'BBQ + Calibrated R(q)']
rows = {m: [] for m in order}
for r in csv.DictReader(open(SRC)):
    rows[r['method']].append((int(r['efSearch']), float(r['recall_at_10']), float(r['qps']), float(r['fp_4k_pages_per_query_mean'])))
# optional argv[1]: equal_pool_qd_fixed75_vs_calibrated.csv -> replaces Fixed R=75 and Calibrated R(q)
# (equal candidate pool = efSearch and equal FP-rerank queue depth for both policies; qps_median)
EQ = sys.argv[1] if len(sys.argv) > 1 else None
if EQ:
    eq = list(csv.DictReader(open(EQ)))
    rows['ADC-BQ + Fixed R=75'] = [(int(r['efSearch']), float(r['fixed_recall_at_10']), float(r['fixed_qps_median']), float(r['fixed_reads_mean'])) for r in eq]
    rows['BBQ + Calibrated R(q)'] = [(int(r['efSearch']), float(r['calibrated_recall_at_10']), float(r['calibrated_qps_median']), float(r['calibrated_reads_mean'])) for r in eq]
with open('recall_ablation.dat', 'w') as f:
    f.write(f'# from {SRC}' + (f'; Fixed R=75 / Calibrated R(q) from {EQ.split("/")[-1]} (equal pool + QD)' if EQ else '') + '\n# columns: efSearch recall@10 qps fp_4k_pages_per_query_mean\n')
    for i, m in enumerate(order):
        f.write(f'\n\n# m{i+1} {m}\n')
        for t in sorted(rows[m]):
            f.write('%d %.6f %.3f %.4f\n' % t)
        print(m, len(rows[m]), min(t[1] for t in rows[m]), max(t[1] for t in rows[m]))
