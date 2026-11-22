#!/usr/bin/env python3
"""Move BBQ (adcbq) curve + points to the end of every plot in recall-qps_md_panels.gp so BBQ is drawn on top.
Only the drawing order changes; data untouched.
Also removes the x title from the top row (shared x title)."""
import re
P = 'recall-qps_md_panels.gp'
src = open(P).read()
out = []
for blk in re.split(r'(?m)^(?=# \([a-p]\)\n)', src):
    lines = blk.split('\n')
    try:
        i0 = next(i for i, l in enumerate(lines) if l.startswith('plot '))
    except StopIteration:
        out.append(blk); continue
    i1 = i0
    while lines[i1].rstrip().endswith('\\'): i1 += 1
    items = [l.strip().rstrip('\\').rstrip().rstrip(',') for l in lines[i0:i1+1]]
    items[0] = items[0][len('plot '):]
    bbq = [s for s in items if '"adcbq_' in s]
    rest = [s for s in items if '"adcbq_' not in s]
    items = rest + bbq
    new = ['plot ' + items[0] + ', \\'] + ['     ' + s + ', \\' for s in items[1:-1]] + ['     ' + items[-1]]
    lines[i0:i1+1] = new
    out.append('\n'.join(lines))
# TOPROW: the x title is shared -- only the bottom row (m..p) carries "Recall@10"
src = ''.join(out)
blocks = re.split(r'(?m)^(?=# \([a-p]\)\n)', src)
for k, b in enumerate(blocks):
    if re.match(r'# \([i-l]\)\n', b):
        blocks[k] = re.sub(r'eval XLABEL\("Recall[^"]*"\)', 'unset xlabel', b)
open(P, 'w').write(''.join(blocks))
