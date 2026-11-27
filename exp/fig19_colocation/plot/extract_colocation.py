import lines_util as U, figlib as F
fig = F.Fig('../bbq_src/figures/bbq/colocation.pdf')
blocks = []
for tag, ax in zip('ab', fig.axes):
    U.report(ax, tag)
    blocks += U.panel_blocks(ax, ['k4co', 'k4base', 'k8co', 'k8base', 'k16co', 'k16base'], tag)
F.write_blocks('colocation.dat', blocks)
