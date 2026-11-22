import lines_util as U, figlib as F
fig = F.Fig('../bbq_src/figures/bbq/index_quality.pdf')
blocks = []
for tag, ax in zip('ab', fig.axes):
    U.report(ax, tag)
    blocks += U.panel_blocks(ax, ['fp32', 'onebit'], tag)
F.write_blocks('index_quality.dat', blocks)
