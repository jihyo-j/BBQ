# Appends TurboQuant1 (TQ1+, RAM graph + FP@R via io_uring) to page_io.dat
# from diskveclab_full_matrix/page_io_msmarco_4methods/normalized_page_io.csv
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['method_key'] == 'turboquant']
DS = {'a': 'msmarco_openai1536', 'b': 'msmarco_qwen1024'}
out = ['', '# TurboQuant1 from normalized_page_io.csv (total_4k_pages_per_query_mean)']
for t, ds in DS.items():
    pts = sorted((float(r['recall_at_10']), float(r['total_4k_pages_per_query_mean']), int(r['search_parameter_value']))
                 for r in rows if r['dataset'] == ds)
    out += ['', '', f'# tq_{t}_curve'] + [f'{x:.6f} {y:g}' for x, y, _ in pts]
    keep = [p for p in pts if p[2] in (128, 160, 224, 320, 512, 1024)] if t == 'b' else pts[::2] + ([pts[-1]] if len(pts) % 2 == 0 else [])
    out += ['', '', f'# tq_{t}_pts'] + [f'{x:.6f} {y:g}' for x, y, _ in keep]
open('page_io.dat', 'a').write('\n'.join(out) + '\n')
