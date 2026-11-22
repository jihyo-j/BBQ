# insertion_scale_data.csv: exactly the three plotted series of insertion_scale.pdf (online insert, efSearch = 256, T = 32),
# merged from the three source CSVs with full precision.  usage: <frozen_adcbq...csv> <rabitq_b1...csv> <tq1plus...csv> <out.csv>
import csv, sys, os
EF = 256
def load(p): return [r for r in csv.DictReader(open(p)) if int(r['efSearch']) == EF]
out = []
def add(method, src, mode, n, recall, sec, rcol, tcol):
    out.append(dict(method=method, inserted_vectors=n, inserted_M=n / 1e6, efSearch=EF, recall_at_10=recall,
                    cumulative_insert_seconds=sec, cumulative_insert_hours=sec / 3600,
                    source_csv=os.path.basename(src), source_mode=mode, recall_column=rcol, time_column=tcol))
fz, rb, tq = sys.argv[1], sys.argv[2], sys.argv[3]
R = load(fz); b = [r for r in R if r['mode'] == 'base_only'][0]
add('BBQ', fz, 'base_only', 0, float(b['adc_recall_at_10']), 0.0, 'adc_recall_at_10', '(base: 0)')
for r in sorted((r for r in R if r['mode'] == 'online_add_no_rebuild'), key=lambda r: int(r['insert_size'])):
    add('BBQ', fz, r['mode'], int(r['insert_size']), float(r['adc_recall_at_10']), float(r['adc_online_add_seconds']), 'adc_recall_at_10', 'adc_online_add_seconds')
for name, p in (('RaBitQ 1bit + adaptive rerank', rb), ('TurboQuant 1bit', tq)):
    R = load(p); b = [r for r in R if r['mode'] == 'base_only'][0]
    add(name, p, 'base_only', 0, float(b['recall_current_at_k']), 0.0, 'recall_current_at_k', '(base: 0)')
    for r in sorted((r for r in R if r['mode'] == 'online_add_no_rebuild'), key=lambda r: int(r['insert_size'])):
        add(name, p, r['mode'], int(r['insert_size']), float(r['recall_current_at_k']), float(r['add_seconds']), 'recall_current_at_k', 'add_seconds')
with open(sys.argv[4], 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), 'rows')
