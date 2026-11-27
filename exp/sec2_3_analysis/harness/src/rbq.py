"""RaBitQ-1bit codes, its error bound, and pools traversed with its estimator.

Everything the refinement-rule comparison needs is dumped per (config, ef) so
that every arm reads the SAME code, the SAME estimated distance and the SAME
candidate pool, and differs only in which pool members it sends to FP32.

RaBitQ (Gao & Long, SIGMOD'24), one global centroid c, as the HNSW variant uses:

    u = P^T (x-c)/||x-c||      (unit)      code b = sign(u) in {-1,+1}^D
    obar = b/sqrt(D)                       <obar,u> = ||u||_1/sqrt(D)
    w = P^T (q-c)/||q-c||      (unit)

    <u,w>  ~=  <obar,w> / <obar,u>                                  (estimator)
    |err|  <=  (eps0/<obar,u>) sqrt((1-<obar,u>^2)/(D-1))           (Thm 3.2)

    dhat(q,x) = ||x-c||^2 + ||q-c||^2 - 2||x-c||||q-c|| <obar,w>/<obar,u>
    B(q,x)    = 2||x-c||||q-c|| (eps0/<obar,u>) sqrt((1-<obar,u>^2)/(D-1))

eps0 = 1.9 is the constant RaBitQ's own implementation ships. B is linear in
eps0, so one stored radius at eps0=1 rescales to any confidence setting.
"""
import os, sys, json, heapq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BBQETA_THREADS", "24")
import numpy as np
from common import WORK, CONFIGS, K, load, SEED

M_DEFAULT, EFC_DEFAULT = 24, 200
EPS0_DEFAULT = 1.9
CH = 100_000


def rot(d, seed=SEED):
    rng = np.random.default_rng(seed)
    P, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return np.ascontiguousarray(P.astype(np.float32))


def _blas_threads(n):
    """Runtime BLAS thread control: the encode step wants the 24 the lab rule
    allows, the forked traversal workers want one thread each."""
    import ctypes, ctypes.util, glob
    cands = glob.glob("/home/sieun/.local/lib/python3.12/site-packages/"
                      "numpy.libs/libscipy_openblas*.so") + \
        glob.glob("/home/sieun/.local/lib/python3.12/site-packages/"
                  "scipy_openblas64/lib/libscipy_openblas*.so")
    for lib in cands + [None]:
        for fn in ("scipy_openblas_set_num_threads64_",
                   "openblas_set_num_threads64_", "openblas_set_num_threads"):
            try:
                h = ctypes.CDLL(lib) if lib else ctypes.CDLL(None)
                getattr(h, fn)(int(n))
                return f"{fn}@{os.path.basename(lib) if lib else 'self'}"
            except Exception:
                continue
    return None


def encode(X, P):
    """code (int8 +-1), ||x-c|| , <obar,u>, and the centroid."""
    N, D = X.shape
    c = np.zeros(D, dtype=np.float64)
    for lo in range(0, N, CH):
        c += X[lo:lo + CH].sum(0, dtype=np.float64)
    c = (c / N).astype(np.float32)
    code = np.empty((N, D), dtype=np.int8)
    nx = np.empty(N, dtype=np.float32)       # ||x-c||
    ou = np.empty(N, dtype=np.float32)       # <obar,u>
    for lo in range(0, N, CH):
        hi = min(lo + CH, N)
        Z = (X[lo:hi] - c) @ P
        n = np.linalg.norm(Z, axis=1)
        U = Z / np.maximum(n, 1e-30)[:, None]
        code[lo:hi] = np.sign(U).astype(np.int8)
        nx[lo:hi] = n
        ou[lo:hi] = np.abs(U).sum(1) / np.sqrt(D)
    return code, nx, ou, c


# --------------------------------------------------------------- traversal
_G = {}


def _nb(i, lev):
    o = _G["offsets"][i]
    a, b = o + _G["cum_nb"][lev], o + _G["cum_nb"][lev + 1]
    v = _G["neighbors"][a:b]
    return v[v >= 0]


def _dhat(ids, qi):
    """Estimated squared L2 distance of a set of ids, RaBitQ estimator."""
    w, qc2, qc = _G["W"][qi], _G["QC2"][qi], _G["QC"][qi]
    ip = np.asarray(_G["code"][ids], dtype=np.float32) @ w / _G["sqD"]
    nx, ou = _G["nx"][ids], _G["ou"][ids]
    est = ip / ou
    return nx * nx + qc2 - 2.0 * nx * qc * est


def _search(qi, ef, k=K):
    """HNSW beam search ranked by -dhat (RaBitQ estimator), as RaBitQ-HNSW does."""
    sc = lambda ids: -_dhat(np.asarray(ids, dtype=np.int64), qi)
    visited = set()
    ep = int(_G["entry"][0])
    cur = float(sc([ep])[0])
    visited.add(ep)
    for lev in range(int(_G["maxlev"][0]), 0, -1):
        moved = True
        while moved:
            moved = False
            nb = _nb(ep, lev)
            if len(nb) == 0:
                break
            s = sc(nb)
            j = int(np.argmax(s))
            visited.update(int(x) for x in nb)
            if s[j] > cur:
                cur, ep, moved = float(s[j]), int(nb[j]), True
    cand = [(-cur, ep)]
    W = [(cur, ep)]
    while cand:
        ns, c = heapq.heappop(cand)
        if len(W) >= ef and -ns < W[0][0]:
            break
        nb = _nb(c, 0)
        new = [int(x) for x in nb if int(x) not in visited]
        if not new:
            continue
        visited.update(new)
        s = sc(new)
        for x, sx in zip(new, s):
            sx = float(sx)
            if len(W) < ef or sx > W[0][0]:
                heapq.heappush(cand, (-sx, x))
                heapq.heappush(W, (sx, x))
                if len(W) > ef:
                    heapq.heappop(W)
    ids = np.array([i for _, i in W], dtype=np.int64)
    s = np.array([v for v, _ in W], dtype=np.float32)
    o = np.lexsort((ids, -s))                      # dhat ascending, id ascending
    return ids[o], -s[o], len(visited)


def _work(args):
    qi, ef = args
    ids, dh, nd = _search(qi, ef)
    out_i = np.full(ef, -1, dtype=np.int32)
    out_d = np.full(ef, np.inf, dtype=np.float32)
    out_i[:len(ids)] = ids
    out_d[:len(dh)] = dh
    return qi, out_i, out_d, nd


def path(key, ef, nq, M=M_DEFAULT):
    return f"{WORK}/rbq_{key}_M{M}_ef{ef}_nq{nq}.npz"


def run(key, efs=(128, 256, 512), nq=4000, M=M_DEFAULT, efc=EFC_DEFAULT, nproc=16):
    """Dump, per (key, ef): pool ids, dhat, bound radius at eps0=1, exact d, GT."""
    import multiprocessing as mp
    import pool as POOL
    todo = [ef for ef in efs if not os.path.exists(path(key, ef, nq, M))]
    if not todo:
        print(f"  {key}: all cached", flush=True)
        return
    X, Q, cfg = load(key, nq=nq)
    N, D = X.shape
    nq = len(Q)                       # may be fewer queries than asked for
    todo = [ef for ef in efs if not os.path.exists(path(key, ef, nq, M))]
    if not todo:
        print(f"  {key}: all cached (nq={nq})", flush=True)
        return
    gtp = [p for p in os.listdir(WORK) if p.startswith(f"gt_{key}_nq")]
    GT = None
    for p in sorted(gtp, key=lambda s: -int(s.split("nq")[-1].split(".")[0])):
        g = np.load(f"{WORK}/{p}")
        if len(g) >= nq:
            GT = g[:nq]
            break
    if GT is None:
        GT = POOL.gt_topk(X, Q)
        np.save(f"{WORK}/gt_{key}_nq{nq}.npy", GT)
    g = POOL.build_graph(key, X, M, efc)

    P = rot(D)
    print(f"  blas={_blas_threads(24)}", flush=True)
    code, nx, ou, c = encode(X, P)
    Qc = (Q - c) @ P
    QC = np.linalg.norm(Qc, axis=1).astype(np.float32)
    W = (Qc / np.maximum(QC, 1e-30)[:, None]).astype(np.float32)
    # bound radius at eps0 = 1: 2||x-c||||q-c|| sqrt((1-ou^2)/(D-1))/ou
    rad_x = (np.sqrt(np.maximum(1.0 - ou.astype(np.float64) ** 2, 0.0)
                     / (D - 1)) / ou).astype(np.float32)

    _G.update(g)
    _G.update(code=code, nx=nx, ou=ou, W=W, QC=QC, QC2=(QC ** 2).astype(np.float32),
              sqD=np.float32(np.sqrt(D)))

    _blas_threads(1)
    for ef in todo:
        ids = np.full((nq, ef), -1, dtype=np.int32)
        dh = np.full((nq, ef), np.inf, dtype=np.float32)
        ndis = np.zeros(nq, dtype=np.int32)
        with mp.Pool(nproc) as pr:
            for qi, i, d, nd in pr.imap_unordered(
                    _work, [(i, ef) for i in range(nq)], chunksize=8):
                ids[qi], dh[qi], ndis[qi] = i, d, nd
        # exact squared L2 and bound radius of every pool member
        dex = np.full((nq, ef), np.inf, dtype=np.float32)
        rad = np.zeros((nq, ef), dtype=np.float32)
        for r in range(nq):
            v = ids[r]
            m = v >= 0
            vv = v[m].astype(np.int64)
            dex[r, m] = 2.0 - 2.0 * (X[vv] @ Q[r])
            rad[r, m] = 2.0 * nx[vv] * QC[r] * rad_x[vv]
        # exact squared L2 of the GLOBAL top-k answers (for true recall@k)
        dgt = np.empty((nq, K), dtype=np.float32)
        for r in range(nq):
            dgt[r] = 2.0 - 2.0 * (X[GT[r].astype(np.int64)] @ Q[r])
        np.savez(path(key, ef, nq, M), ids=ids, dhat=dh, rad=rad,
                 dex=dex, ndis=ndis, gt=GT.astype(np.int32), dgt=dgt,
                 D=np.array([D]), N=np.array([N]))
        inpool = np.mean([np.isin(GT[r], ids[r]).mean() for r in range(nq)])
        print(f"  {key} ef={ef}: GT in pool {inpool:.4f} ndis {ndis.mean():.0f}",
              flush=True)


if __name__ == "__main__":
    for key in sys.argv[1:]:
        print(key, flush=True)
        run(key)
