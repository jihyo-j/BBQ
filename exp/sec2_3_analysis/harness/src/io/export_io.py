"""Export what refio needs: a page-aligned FP32 vector file, and the test-half
pool with the knobs each rule picked on the calibration half.

The knobs come from ONE calibration split (seed fixed); refio then replays only
the test half, so no arm is tuned on the queries it is timed on.
"""
import os, sys, json
sys.path.insert(0, "/home/sieun/projects/BBQETA/src")
os.environ.setdefault("BBQETA_THREADS", "24")
import numpy as np
from common import load, WORK, OUT, K
import refsel, rbq

IOD = "/home/smrc/samsung-nvme/sieun/bbqeta/io"


def vecfile(key, nq=4000):
    os.makedirs(IOD, exist_ok=True)
    X, Q, _ = load(key, nq=1)
    N, D = X.shape
    pagesz = int(np.ceil(4 * D / 4096.0)) * 4096
    p = f"{IOD}/vec_{key}_p{pagesz}.bin"
    if os.path.exists(p) and os.path.getsize(p) == N * pagesz:
        return p, pagesz, N, D
    pad = np.zeros(pagesz, dtype=np.uint8)
    with open(p, "wb") as f:
        buf = np.zeros((1024, pagesz), dtype=np.uint8)
        for lo in range(0, N, 1024):
            hi = min(lo + 1024, N)
            buf[:] = 0
            v = X[lo:hi].astype(np.float32).tobytes()
            b = np.frombuffer(v, dtype=np.uint8).reshape(hi - lo, 4 * D)
            buf[:hi - lo, :4 * D] = b
            f.write(buf[:hi - lo].tobytes())
    return p, pagesz, N, D


def export(key, ef, nq, target=0.99, seed=20261120):
    P = refsel.load_pool(key, ef, nq)
    T = refsel.prefix_tables(P)
    nqq, efc = P["dhat"].shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nqq)
    m = nqq // 2
    cal, tst = perm[:m], np.sort(perm[m:])
    ell = int(np.ceil((m + 1) * target))
    knobs = {}
    BP = {e: refsel.bound_pass(P, e) for e in refsel.EPS_GRID}
    feas = [e for e in refsel.EPS_GRID if BP[e]["pres"][cal].mean() >= target]
    knobs["seq_bound_tuned"] = float(min(feas) if feas else refsel.EPS_GRID[-1])
    knobs["seq_bound_default"] = 1.9
    for nm, fn in (("batch_boundM", refsel.bound_margin),
                   ("twophase", refsel.bound_2phase)):
        fe = [e for e in refsel.EPS_GRID if fn(P, T, e)["pres"][cal].mean() >= target]
        knobs[nm] = float(min(fe) if fe else refsel.EPS_GRID[-1])
    knobs["calib_tau"] = float(np.sort(T["Hmaxmarg"][cal])[min(ell, m) - 1])
    knobs["fixedR"] = int(min(np.sort(T["Rneed"][cal])[min(ell, m) - 1], efc))

    vp, pagesz, N, D = vecfile(key, nq)
    _, Q, _ = load(key, nq=nq)
    Qt = np.ascontiguousarray(Q[tst], dtype=np.float32)
    out = f"{IOD}/pool_{key}_ef{ef}.bin"
    with open(out, "wb") as f:
        np.array([len(tst), efc, D, K], dtype=np.int32).tofile(f)
        Qt.tofile(f)
        np.ascontiguousarray(P["ids"][tst], dtype=np.int32).tofile(f)
        np.ascontiguousarray(P["dhat"][tst], dtype=np.float32).tofile(f)
        np.ascontiguousarray(P["rad"][tst], dtype=np.float32).tofile(f)
    meta = dict(key=key, ef=int(ef), nq_test=int(len(tst)), D=int(D), K=K,
                pagesz=pagesz, N=int(N), vecfile=vp, poolfile=out, knobs=knobs,
                pages_per_vec=pagesz // 4096)
    json.dump(meta, open(f"{IOD}/meta_{key}_ef{ef}.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)
    return meta


if __name__ == "__main__":
    for a in sys.argv[1:]:
        key, ef, nq = a.split(":")
        export(key, int(ef), int(nq))
