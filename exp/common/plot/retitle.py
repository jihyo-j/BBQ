import re
RX = r'"Recall\\\\@10"'   # gnuplot literal: "Recall\\@10" inside eval string -> 4 backslashes in file
def edit(fn, pairs):
    s = open(fn).read()
    for a, b in pairs:
        if a not in s: raise SystemExit(f'{fn}: not found: {a!r}')
        s = s.replace(a, b)
    open(fn, 'w').write(s)

R = 'eval XLABEL("Recall\\\\\\\\@10")'
# comparision_hnsw
edit('comparision_hnsw.gp', [
 ('PBOT = PB(NROWS) + 1.0                 # panel bottom (one-line caption)', 'PBOT = PB(NROWS)                       # panel bottom (tick labels + axis name)'),
 ('H = PBOT + PH2 + TOPM; OUT', 'H = PBOT + PH2 + TOPM + TTL; OUT'),
 ('unset xlabel\neval CAP(91, "(a)  MS MARCO, Qwen3-0.6B", LM+PW/2)', R + '\neval TITLE("MS MARCO Qwen 1,024")'),
 ('unset xlabel\neval CAP(92, "(b)  DBpedia, Qwen3-0.6B", LM+PW+GM+PW/2)', R + '\neval TITLE("DBpedia Qwen 1,024")'),
])
# exact_re
edit('exact_re.gp', [
 ('eval TITLE("MS MARCO / Qwen1024")', 'eval TITLE("MS MARCO Qwen 1,024")'),
 ('eval TITLE("budget at target")', 'eval TITLE("MS MARCO Qwen 1,024")'),
])
# index_quality
edit('index_quality.gp', [
 ('H = PBOT + PH2 + TOPM; OUT', 'H = PBOT + PH2 + TOPM + TTL; OUT'),
 ('unset xlabel\neval CAP(91, "(a)  Recall\\\\\\\\@10 (MS MARCO, Qwen3-0.6B)", 64.05)', R + '\neval TITLE("MS MARCO Qwen 1,024")'),
 ('unset xlabel\neval CAP(92, "(b)  Recall\\\\\\\\@10 (Quora, Qwen3-0.6B)", 185.95)', R + '\neval TITLE("Quora Qwen 1,024")'),
])
# insertion_scale
edit('insertion_scale.gp', [
 ('PBOT = PB(NROWS) + 7.0                 # panel bottom (+ axis-name line above the caption)', 'PBOT = PB(NROWS)                       # panel bottom (tick labels + axis name)'),
 ('set label 90 "MS MARCO, Qwen3-0.6B (1M base, efSearch = 256)" at screen 0.5,sy(H-TTL/2-1.5) center font ",".sprintf("%g",FL) front\n', ''),
 ('eval CAP(91, "(a)  Recall\\\\\\\\@10", LM+PW/2)', 'eval TITLE("MS MARCO Qwen 1,024")'),
 ('eval CAP(92, "(b)  Online insert time", LM+PW+GM+PW/2)', 'eval TITLE("MS MARCO Qwen 1,024")'),
])
# preprocessing_io
edit('preprocessing_io.gp', [
 ('H = LEGTOP(NROWS) + XLABB + PH2 + TOPM; OUT', 'H = LEGTOP(NROWS) + XLABB + PH2 + TOPM + TTL; OUT'),
 ('eval XLABEL("(a)  MS MARCO, Qwen3-0.6B") }', 'unset xlabel; eval TITLE("MS MARCO Qwen 1,024") }'),
 ('eval XLABEL("(b)  NQ, Qwen3-0.6B") }', 'unset xlabel; eval TITLE("NQ Qwen 1,024") }'),
])
# preserve
edit('preserve.gp', [
 ('eval TITLE("MS MARCO / BGE-M3")', 'eval TITLE("MS MARCO BGE-M3 1,024")'),
 ('eval TITLE("Quora / BGE-M3")', 'eval TITLE("Quora BGE-M3 1,024")'),
])
# recall_ablation: per-panel titles instead of the common one
s = open('recall_ablation.gp').read()
s = re.sub(r'set label 90 "MS MARCO, Qwen3-0.6B".*\n', '', s)
s = s.replace('eval PANEL(LM, PBOT, PW, PH2)\n', 'eval PANEL(LM, PBOT, PW, PH2)\neval TITLE("MS MARCO Qwen 1,024")\n', 1)
s = s.replace('eval PANEL(LM+PW+GM, PBOT, PW, PH2)\n', 'eval PANEL(LM+PW+GM, PBOT, PW, PH2)\neval TITLE("MS MARCO Qwen 1,024")\n', 1)
assert s.count('eval TITLE(') == 2
open('recall_ablation.gp', 'w').write(s)
# recall_qps
edit('recall_qps.gp', [
 ('H = PB(NROWS) + PH2 + TOPM; OUT', 'H = PB(NROWS) + PH2 + TOPM + TTL; OUT'),
 ('eval XLABEL("(a)  Recall\\\\\\\\@10 (MS MARCO, Qwen3-0.6B)")', R + '\neval TITLE("MS MARCO Qwen 1,024")'),
 ('eval XLABEL("(b)  Recall\\\\\\\\@10 (NQ, Qwen3-0.6B)")', R + '\neval TITLE("NQ Qwen 1,024")'),
])
# recall-qps_main: one title row per panel row
edit('recall-qps_main.gp', [
 ('RP  = PH2 + XLAB + 4.0                      # row pitch', 'RP  = PH2 + XLAB + TTL + 2.0                # row pitch (panel + x labels + title)'),
 ('H = PB(NROWS) + 3*RP + PH2 + TOPM; OUT', 'H = PB(NROWS) + 3*RP + PH2 + TOPM + TTL; OUT'),
])
MODEL = {'Qwen3-0.6B': 'Qwen 1,024', 'BGE-M3': 'BGE 1,024', 'BGE': 'BGE 1,024', 'OpenAI': 'OpenAI 1,536',
         'E5-Mistral': 'E5 4,096', 'Nemotron': 'Nemotron 4,096'}
for fn in ('recall-qps_main_panels.gp',):
    s = open(fn).read()
    def rep(m):
        ds, mo = m.group(2), m.group(3)
        return R + f'\neval TITLE("{ds} {MODEL[mo]}")'
    s, n = re.subn(r'eval XLABEL\("\((\w)\)  Recall\\\\\\\\@10 \(([^,]+), ([^)]+)\)"\)', rep, s)
    print(fn, 'panels retitled:', n)
    open(fn, 'w').write(s)
