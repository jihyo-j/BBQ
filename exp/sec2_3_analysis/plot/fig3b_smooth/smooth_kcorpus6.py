#!/usr/bin/env python3
"""Smoothed copy of p_kcorpus6.dat for f3b (R*(k)/k): the per-k means are noisy step-to-step,
so each column is smoothed by a Gaussian-kernel local-linear fit in (log10 k, log R*) with
bandwidth BW decades.  k values are unchanged; the raw file is not modified.
Output: p_kcorpus6_smooth.dat (same columns as the input)."""
import numpy as np
BW = 0.06
raw = np.loadtxt('p_kcorpus6.dat')
x = np.log10(raw[:, 0]); out = raw.copy()
for c in range(1, raw.shape[1]):
    y = np.log(raw[:, c])
    for i, x0 in enumerate(x):
        w = np.exp(-0.5 * ((x - x0) / BW) ** 2)
        X = np.vstack([np.ones_like(x), x - x0]).T
        b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))
        out[i, c] = np.exp(b[0])
hdr = open('p_kcorpus6.dat').readline().rstrip() + '\n# smoothed by smooth_kcorpus6.py (local-linear, Gaussian kernel, %.2f decades in log k)\n' % BW
hdr += open('p_kcorpus6.dat').readlines()[1].rstrip()
np.savetxt('p_kcorpus6_smooth.dat', out, fmt='%.4f', header=hdr.replace('# ', '', 1) if False else hdr, comments='')
rel = np.abs(out[:, 1:] / raw[:, 1:] - 1)
print('max |smooth/raw-1| per column:', np.round(rel.max(0) * 100, 2), '%')
