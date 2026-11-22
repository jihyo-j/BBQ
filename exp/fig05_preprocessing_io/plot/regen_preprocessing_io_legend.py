# preprocessing_io_legend.gp = legend.py output with bigger swatches (14 x 7 pt) so the hatch reads
import re, subprocess
out = subprocess.run(['python3', '../tools/legend.py', 'preprocessing_io.legend', '241.15'], capture_output=True, text=True, check=True).stdout
def grow(m):
    x0, y0, x1, y1 = map(float, m.group(2, 3, 4, 5))
    xc, yc = x1, (y0 + y1) / 2          # keep the right edge (gap to the text), grow left / up / down
    return f'{m.group(1)}sx({x1 + 1 - 14:.2f}),sy({yc - 3.5:.2f}) to screen sx({x1 + 1:.2f}),sy({yc + 3.5:.2f})'
out = re.sub(r'(set object \d+ rect from screen )sx\(([\d.]+)\),sy\(([\d.]+)\) to screen sx\(([\d.]+)\),sy\(([\d.]+)\)', grow, out)
open('preprocessing_io_legend.gp', 'w').write(out)
print(out)

# pdfcairo's hatch spacing is fixed in device units, so a 14pt swatch shows only ~1 stroke.
# Replace the hatched swatch by an outline + hand-drawn "/" strokes every SP pt (clipped to the box).
m = re.search(r'set object (\d+) rect from screen sx\(([\d.]+)\),sy\(([\d.]+)\) to screen sx\(([\d.]+)\),sy\(([\d.]+)\) fc ls (\d+) fs pattern HATCH border lc ls \d+ lw MLW front', out)
tag, x0, y0, x1, y1, ls = int(m.group(1)), *map(float, m.group(2, 3, 4, 5)), int(m.group(6))
SP = 7.0            # (14 + 7) / 3 -> three strokes
segs = []
c = -(y1 - y0) + SP / 2
while c < (x1 - x0):
    # line through the box: (x0+c+t, y0+t), t in [0, y1-y0], clipped to x in [x0, x1]
    t0 = max(0.0, -c); t1 = min(y1 - y0, (x1 - x0) - c)
    if t1 > t0:
        segs.append((x0 + c + t0, y0 + t0, x0 + c + t1, y0 + t1))
    c += SP
new = [f'set object {tag} rect from screen sx({x0:.2f}),sy({y0:.2f}) to screen sx({x1:.2f}),sy({y1:.2f}) fc rgb MBLUE fs empty border lc rgb MBLUE lw MLW front']
for i, (a, b, cc, d) in enumerate(segs):
    new.append(f'set arrow {900+i} from screen sx({a:.2f}),sy({b:.2f}) to screen sx({cc:.2f}),sy({d:.2f}) nohead lc rgb MBLUE lw 0.6 front')
out = out.replace(m.group(0), '\n'.join(new))
open('preprocessing_io_legend.gp', 'w').write(out)
print(len(segs), 'hatch strokes')
