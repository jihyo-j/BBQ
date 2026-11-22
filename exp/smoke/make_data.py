#!/usr/bin/env python3
"""Tiny synthetic corpus for the kick-the-tires run (no downloads).
Low-rank clustered vectors plus small noise, L2-normalised: a rough stand-in for Transformer embeddings
(signal concentrated in few directions), so 1-bit ADC behaves like it does on real text embeddings."""
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n, nq, d, k = 20000, 1200, 256, 100
rng = np.random.default_rng(7)
proj = rng.standard_normal((48, d)).astype(np.float32)
centers = rng.standard_normal((64, 48)).astype(np.float32)
def sample(m):
    z = centers[rng.integers(0, 64, m)] + 0.8 * rng.standard_normal((m, 48)).astype(np.float32)
    x = z @ proj + 0.15 * np.sqrt(48) * rng.standard_normal((m, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
base, query = sample(n), sample(nq)
def wbin(p, a, dt):
    with open(p, "wb") as f:
        np.array(a.shape, dtype=np.int32).tofile(f); a.astype(dt).tofile(f)
gt = np.argsort(-(query @ base.T), axis=1, kind="stable")[:, :k]
wbin(out / "base.fbin", base, np.float32)
for name, sl in (("cal", slice(0, 1000)), ("test", slice(1000, nq))):
    wbin(out / f"{name}.query.fbin", query[sl], np.float32)
    wbin(out / f"{name}.gt.ibin", gt[sl], np.int32)
print(f"wrote {n} x {d} base, 1000 calibration + {nq - 1000} test queries to {out}")
