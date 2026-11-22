#!/usr/bin/env python3
"""Replace TurboQuant1 / Faiss PQ / Faiss OPQ curves in recall-qps_main with measured data
(dataset-tuned R, io_uring) from recall_qps_selected_dataset_r_tq_pq_opq_..._t1.csv.
Curves = all measured points in efSearch order; markers = 5 evenly spaced points.
Input:  recall-qps_main_orig.dat / recall-qps_main_panels_orig.gp (extracted originals)
Output: recall-qps_main.dat / recall-qps_main_panels.gp"""
import csv, re, sys
from collections import OrderedDict
SRC = sys.argv[1] if len(sys.argv) > 1 else 'recall-qps_main_tq_pq_opq.csv'
PANEL = dict(zip('abcdefghijklmnop', [
    'nq_bge_m3_vanilla', 'nq_qwen06_1024', 'nq_nemotron4096', 'nq_e5mistral4096',
    'quora_bge_m3_vanilla', 'quora_qwen1024', 'quora_nemotron4096', 'quora_e5mistral4096',
    'msmarco_openai1536', 'msmarco_bge1024', 'msmarco_qwen1024', 'msmarco_e5mistral4096',
    'dbpedia_openai1536', 'dbpedia_bge1024', 'dbpedia_qwen1024', 'dbpedia_e5mistral4096']))
NEW = {'tq': 'turboquant_tq1_tuned_r', 'fpq': 'faiss_pq8m128_tuned_r', 'fopq': 'faiss_opq_pq8m128_tuned_r'}
ORDER = ['adcbq', 'esbbq', 'tq', 'mrq', 'rbq7', 'rbqssd', 'hnswrbq', 'dannrbq', 'spann', 'dannpq', 'fpq', 'fopq']
LS = {m: i + 1 for i, m in enumerate(ORDER)}
ORDER.remove('mrq')   # MRQ-SSD dropped from the figure (line style 4 kept unused)
CHAR = {'rbqssd': 'CH_RTRI', 'fopq': 'CH_HEX'}

# --- read original blocks
blocks, name, head = OrderedDict(), None, []
for l in open('recall-qps_main_orig.dat'):
    m = re.match(r'# (\S+_[a-p]_(pts|curve))\s*$', l)
    if m: name = m.group(1); blocks[name] = []; continue
    if name and l.strip() and not l.startswith('#'): blocks[name].append(l.strip())
    elif not name: head.append(l)

rows = list(csv.DictReader(open(SRC)))
for p, ds in PANEL.items():
    for m, key in NEW.items():
        pts = [(float(r['recall_at_10']), float(r['qps'])) for r in rows if r['dataset'] == ds and r['method_key'] == key]
        for suf in ('pts', 'curve'): blocks.pop(f'{m}_{p}_{suf}', None)
        if not pts: continue
        n = len(pts); idx = sorted({round(i * (n - 1) / 4) for i in range(5)}) if n > 5 else range(n)
        blocks[f'{m}_{p}_curve'] = ['%.6f %.4f' % xy for xy in pts]
        blocks[f'{m}_{p}_pts'] = ['%.6f %.4f' % pts[i] for i in idx]

with open('recall-qps_main.dat', 'w') as f:
    f.write('# data recovered from the original matplotlib PDF (vector coordinates);\n'
            '# tq / fpq / fopq replaced by measured points from %s\n' % SRC)
    for p in PANEL:
        for m in ORDER:
            for suf in ('pts', 'curve'):
                k = f'{m}_{p}_{suf}'
                if k in blocks: f.write('\n\n# %s\n%s\n' % (k, '\n'.join(blocks[k])))

# --- rewrite every panel's plot command from the method order
def plotcmd(p):
    have = [m for m in ORDER if f'{m}_{p}_curve' in blocks]
    L = [f'D index "{m}_{p}_curve" u 1:2 w l ls {LS[m]} notitle' for m in have]
    for m in have:
        if m in CHAR:
            L.append(f'D index "{m}_{p}_pts" u 1:2:({CHAR[m]}) w labels center font ",".sprintf("%g",CHFS) tc ls {LS[m]} notitle')
        else:
            L.append(f'D index "{m}_{p}_pts" u 1:2 w p ls {LS[m]} lw MLW notitle')
    return 'plot ' + ', \\\n     '.join(L) + '\n'
txt = open('recall-qps_main_panels_orig.gp').read()
lines, out, i = txt.split('\n'), [], 0
while i < len(lines):
    mo = re.match(r'plot D index "\w+_([a-p])_curve"', lines[i])
    if mo:
        while lines[i].rstrip().endswith('\\'): i += 1
        out.append(plotcmd(mo.group(1)).rstrip('\n'))
    else:
        out.append(lines[i])
    i += 1
open('recall-qps_main_panels.gp', 'w').write('\n'.join(out))
# y-range sanity
for p in PANEL:
    yr = re.search(r'# \(%s\)\n[^\n]*\n[^\n]*yrange \[([\d.]+):([\d.]+)\]' % p, txt)
    lo, hi = float(yr.group(1)), float(yr.group(2))
    for m in NEW:
        for l in blocks.get(f'{m}_{p}_curve', []):
            y = float(l.split()[1])
            if not lo <= y <= hi: print('out of y-range', p, m, l)
