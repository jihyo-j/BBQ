# Regenerate recall-qps_main_legend.gp and re-attach the two character markers
# (RaBitQ-SSD ▷ = ls 6, Faiss OPQ ⬡ = ls 12), which legend.py cannot draw.
import re, subprocess, sys
out = subprocess.run([sys.executable, '../tools/legend.py', 'recall-qps_main.legend', '506.30'],
                     capture_output=True, text=True, check=True).stdout
def mid(ls):
    m = re.search(r'from screen sx\(([\d.]+)\),sy\(([\d.]+)\) to screen sx\(([\d.]+)\),sy\([\d.]+\) nohead ls %d front' % ls, out)
    return (float(m.group(1)) + float(m.group(3))) / 2, float(m.group(2))
x6, y6 = mid(6); x12, y12 = mid(12)
out += ('# character markers (no gnuplot point type): drawn as labels at the sample midpoint\n'
        f'set label 206 CH_RTRI at screen sx({x6:.2f}),sy({y6:.2f}) center font ",".sprintf("%g",CHFS) tc ls 6 front\n'
        f'set label 212 CH_HEX at screen sx({x12:.2f}),sy({y12:.2f}) center font ",".sprintf("%g",CHFS) tc ls 12 front\n')
open('recall-qps_main_legend.gp', 'w').write(out)
