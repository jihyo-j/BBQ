#!/usr/bin/env python3
# qwen1024_five_method_recall099_timing_t1.csv -> preprocessing_io.dat  (times in ms)
import csv, sys
src = sys.argv[1] if len(sys.argv) > 1 else 'qwen1024_five_method_recall099_timing_t1.csv'
ORDER = [('adcbq1', 'ADC-BQ'), ('qdranttq1', 'TurboQuant1'), ('rabitq1', 'HNSW-RaBitQ1'),
         ('diskann_pq128', 'DiskANN PQ128'), ('diskann_opq128', 'DiskANN OPQ128')]
rows = list(csv.DictReader(open(src)))
out = ['# from %s' % src, '# x prep_ms total_ms(prep+io_wait) label method prep_label']
for tag, ds in (('a', 'msmarco_qwen1024'), ('b', 'nq_qwen1024')):
    out += ['', '', '# bars_%s  (%s)' % (tag, ds)]
    for i, (key, name) in enumerate(ORDER, 1):
        r = next(r for r in rows if r['dataset'] == ds and r['method_key'] == key)
        p = float(r['query_preprocess_us']) / 1e3
        t = p + float(r['exact_fp_io_wait_us']) / 1e3
        lab = '%.2f' % t if t < 1 else '%.1f' % t
        us = p * 1e3
        plab = ('%.1fµs' % us) if us < 9.95 else ('%.0fµs' % us) if us < 1000 else ('%.0fms' % p)
        out.append('%d %.6f %.6f %s "%s" "%s"' % (i, p, t, lab, name, plab))
open('preprocessing_io.dat', 'w').write('\n'.join(out) + '\n')
