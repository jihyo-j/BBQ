# restyle a two-panel common.gp figure to the f2 look: Carlito 6.5/7, f2 plot-box aspect, labelled
# ticks only (x/y axes, no mirror), legend rows on top, page cropped.
import re, sys
def restyle(name, rows, LM=24.0, GM=9.0, RM=8.0, legfont=6.5, SMP=14.0, GST=3.0, GCOL=9.0):
    s = open(name + '.gp').read()
    nrow = len(rows)
    a = s.index('NROWS'); b = s.index('eval SETTERM') + len('eval SETTERM')
    hdr = f'''# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, labelled ticks only on the
# x / y axes, legend on top (widths measured from Carlito {legfont}pt), page cropped to the content
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = {LM}; GM = {GM}; RM = {RM}
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
LEGB = {8.5*nrow + 1.5}
W = COLW; H = PBOT + PHX + 1.0 + TTL + LEGB; OUT = "{name}"
eval SETTERM'''
    s = s[:a] + hdr + s[b:]
    # legend
    leg = [f'SMP = {SMP}; GST = {GST}; GCOL = {GCOL}',
           "ENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d \"\" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d \"%s\" at screen sx(%g),sy(%g) left font \",%g\" front', n, x, y, x+SMP, y, ls, n+1, x+SMP/2, y, ls, n+2, txt, x+SMP+GST, y, LFS)",
           f'LFS = {legfont}']
    n = 100
    ncol = max(len(r) for r in rows)
    colw = [max(rows[r][c][2] for r in range(nrow) if c < len(rows[r])) for c in range(ncol)]
    tot = sum(SMP + GST + w for w in colw) + GCOL * (ncol - 1)
    for ri, r in enumerate(rows):
        leg.append(f'LY = H - {4.75 + 8.5*ri}; LX = (W - {tot:.2f})/2.0')
        for ci, (ls, txt, w) in enumerate(r):
            leg.append(f'eval ENT({n}, LX, LY, {ls}, "{txt}"); LX = LX + SMP + GST + {colw[ci]:.2f} + GCOL')
            n += 3
    leg += ['eval DRAWSCREEN', 'set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)',
            'unset mxtics; unset mytics',
            'XL(t) = sprintf(\'set xlabel "%s" font ",7" offset 0,1.0\', t)',
            'TT(t) = sprintf(\'set title "%s" font ",7" offset 0,-0.75\', t)']
    s = re.sub(r'load "%s_legend.gp"; *\n?eval DRAWSCREEN' % name, '\n'.join(leg), s)
    s = re.sub(r'set m[xy]tics [0-9]+', 'unset mxtics; unset mytics', s)
    s = re.sub(r'eval PANEL\(L2, PB\(NROWS\), PW2, PH2\)', 'eval PANEL(LM, PBOT, PW, PHX)', s)
    s = re.sub(r'eval PANEL\(L2\+PW2\+G2, PB\(NROWS\), PW2, PH2\)', 'eval PANEL(LM+PW+GM, PBOT, PW, PHX)', s)
    s = re.sub(r', "" 0\.99\d+\)', ')', s)                          # drop unlabelled x ticks at the end
    s = re.sub(r'(set xtics \([^)]*\))', r'\1 offset 0,0.45', s)
    s = re.sub(r'eval YLABEL\("([^"]*)"\); set ylabel offset [-\d.]+,0', r'set ylabel "\1" font ",7" offset 1.4,0', s)
    s = re.sub(r'eval XLABEL\("([^"]*)"\)', r'eval XL("\1")', s)
    s = re.sub(r'eval TITLE\("([^"]*)"\)', r'eval TT("\1")', s)
    open(name + '.gp', 'w').write(s)
if __name__ == '__main__':
    pass
