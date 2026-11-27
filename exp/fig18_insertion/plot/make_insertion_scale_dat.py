# insertion_scale.dat <- frozen_adcbq_vs_rabitq_c16_comparison.csv
# columns: inserted[M]  recall@10  time[h]   (one block per series, at efSearch = EF)
#   BBQ online  : frozen 1M tau, online add (adc_online_add_seconds, cumulative from the 1M base)
#   BBQ rebuild : full rebuild + recalibrated tau (adc_build_or_rebuild_seconds)
#   RaBitQ      : rabitq_online_add_seconds / rabitq_build_or_rebuild_seconds
# x = 0 is the 1M base for every series (online time 0, rebuild time = base build).
import csv, sys
EF = int(sys.argv[2]) if len(sys.argv) > 2 else 256
rows = [r for r in csv.DictReader(open(sys.argv[1])) if int(r['efSearch']) == EF]
base = [r for r in rows if r['mode'] == 'base_only'][0]
S = [('bbq_online', 'online_add_no_rebuild', 'adc', 'adc_online_add_seconds'),
     ('bbq_rebuild', 'full_rebuild_oracle', 'adc', 'adc_build_or_rebuild_seconds'),
     ('rabitq_online', 'online_add_no_rebuild', 'rabitq', 'rabitq_online_add_seconds'),
     ('rabitq_rebuild', 'full_rebuild_oracle', 'rabitq', 'rabitq_build_or_rebuild_seconds')]
out = [f'# data: frozen_adcbq_vs_rabitq_c16_comparison.csv, efSearch = {EF}, T = 32', '# inserted[M] recall@10 time[h]']
for name, mode, p, tcol in S:
    pts = [(0.0, float(base[f'{p}_recall_at_10']), 0.0 if 'online' in name else float(base[tcol]) / 3600)]
    pts += sorted((int(r['insert_size']) / 1e6, float(r[f'{p}_recall_at_10']), float(r[tcol]) / 3600)
                  for r in rows if r['mode'] == mode)
    out += ['', '', f'# {name}'] + [f'{x:g} {y:.4f} {t:.4f}' for x, y, t in pts]
# RaBitQ: replaced by the 1-bit + adaptive-rerank run (rabitq_b1_adaptive_c16_random_independent_t32.csv), argv[3]
if len(sys.argv) > 3:
    rb = [r for r in csv.DictReader(open(sys.argv[3])) if int(r['efSearch']) == EF]
    blk = {}
    for name, mode, tkey in (('rabitq_online', 'online_add_no_rebuild', 'add_seconds'),
                             ('rabitq_rebuild', 'full_rebuild_oracle', 'build_or_rebuild_seconds')):
        b0 = [r for r in rb if r['mode'] == 'base_only'][0]
        pts = [(0.0, float(b0['recall_current_at_k']), 0.0 if 'online' in name else float(b0['build_or_rebuild_seconds']) / 3600)]
        pts += sorted((int(r['insert_size']) / 1e6, float(r['recall_current_at_k']), float(r[tkey]) / 3600) for r in rb if r['mode'] == mode)
        blk[name] = [f'{x:g} {y:.4f} {t:.4f}' for x, y, t in pts]
    o2, skip = [], False
    for line in out:
        if line.startswith('# rabitq_'):
            o2.append(line + '  (rabitq_b1_adaptive_c16)'); o2 += blk[line[2:].strip()]; skip = True; continue
        if skip and (line == '' or line.startswith('#')): skip = False
        if not skip: o2.append(line)
    out = o2
# TurboQuant 1bit (TQ1+, fixed R=60) online insert, argv[4]: tq1plus_r60_random_independent_t32.csv (partial: up to 30M so far)
if len(sys.argv) > 4:
    tq = [r for r in csv.DictReader(open(sys.argv[4])) if int(r['efSearch']) == EF]
    b0 = [r for r in tq if r['mode'] == 'base_only'][0]
    pts = [(0.0, float(b0['recall_current_at_k']), 0.0)]
    pts += sorted((int(r['insert_size']) / 1e6, float(r['recall_current_at_k']), float(r['add_seconds']) / 3600) for r in tq if r['mode'] == 'online_add_no_rebuild')
    out += ['', '', '# tq_online  (tq1plus_r60)'] + [f'{x:g} {y:.4f} {t:.4f}' for x, y, t in pts]
open('insertion_scale.dat', 'w').write('\n'.join(out) + '\n')
