# p_overpay_smooth.dat: PCHIP through the marker points of p_overpay_pts.dat (coverage -> log ratio),
# so the f4b curves are smooth; markers stay at the measured values.
import numpy as np
def pchip(x, y, xq):
    h = np.diff(x); d = np.diff(y) / h; n = len(x); m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k-1] * d[k] > 0:
            w1, w2 = 2*h[k] + h[k-1], h[k] + 2*h[k-1]
            m[k] = (w1 + w2) / (w1/d[k-1] + w2/d[k])
    m[0], m[-1] = d[0], d[-1]
    i = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    s = (xq - x[i]) / h[i]
    return ((2*s**3 - 3*s**2 + 1) * y[i] + (s**3 - 2*s**2 + s) * h[i] * m[i]
            + (-2*s**3 + 3*s**2) * y[i+1] + (s**3 - s**2) * h[i] * m[i+1])
P = np.loadtxt('p_overpay_pts.dat')
cols = []
for c in (1, 3, 5):
    x, y = P[:, c], P[:, c+1]
    xq = np.linspace(x[0], x[-1], 200)
    cols += [xq, np.exp(pchip(x, np.log(y), xq))]
out = np.column_stack([np.arange(200)] + cols)
np.savetxt('p_overpay_smooth.dat', out, fmt='%.6f',
           header='i  quora_cov quora_ratio  msmarco_qw_cov msmarco_qw_ratio  msmarco_bge_cov msmarco_bge_ratio  (PCHIP through p_overpay_pts.dat)')
