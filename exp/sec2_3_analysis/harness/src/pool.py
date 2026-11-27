"""Pools from an FP32-built HNSW graph traversed with ADC-BQ scores.

This is the instrument the calibration theorem is stated over: a *fixed* search
procedure that returns a candidate list L(q), and the ADC score s(q,x) =
q^T sign(x) that both the traversal and the threshold rule read.  Everything
Section 6.4 and Appendix A need is derivable from three arrays per
(configuration, ef_search):

  pool_s   [nq, ef]  ADC scores of L(q), descending, ties broken by id
  gold_lv  [nq, k]   entry level T_j(q) = max(0, s_k(q) - s(q,x_(j))) of each
                     exact FP32 top-k answer, +inf when x_(j) notin L(q)
  ndis     [nq]      distance evaluations, the cost axis

Graph: faiss IndexHNSWFlat on the unit-normalised FP32 vectors.  On unit norms
L2 and inner product induce the same order, so the graph is the paper's FP32
graph; only the *search* reads sign codes.

Lab rule: <= 24 threads on this shared server.
"""
import os, sys, json, heapq, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BBQETA_THREADS", "24")
import numpy as np
from common import WORK, CONFIGS, K, load

M_DEFAULT, EFC_DEFAULT = 24, 200
EF_LADDER = (64, 128, 256, 512, 1024)


# --------------------------------------------------------------- ground truth
def gt_topk(X, Q, k=K, qchunk=256, xchunk=200_000):
    """Exact top-k by inner product, ties broken by ascending id."""
    nq = len(Q)
    out = np.empty((nq, k), dtype=np.int64)
    for a in range(0, nq, qchunk):
        Qb = Q[a:a + qchunk]
        best_s = np.full((len(Qb), 0), 0.0, dtype=np.float32)
        best_i = np.zeros((len(Qb), 0), dtype=np.int64)
        for b in range(0, len(X), xchunk):
            S = Qb @ X[b:b + xchunk].T
            m = min(k, S.shape[1])
            p = np.argpartition(-S, m - 1, axis=1)[:, :m]
            s = np.take_along_axis(S, p, 1)
            best_s = np.concatenate([best_s, s], 1)
            best_i = np.concatenate([best_i, p.astype(np.int64) + b], 1)
            if best_s.shape[1] > 4 * k:
                p = np.argpartition(-best_s, k - 1, axis=1)[:, :k]
                best_s = np.take_along_axis(best_s, p, 1)
                best_i = np.take_along_axis(best_i, p, 1)
        # final order: score descending, id ascending
        ordr = np.lexsort((best_i, -best_s), axis=1)
        out[a:a + qchunk] = np.take_along_axis(best_i, ordr, 1)[:, :k]
    return out


# --------------------------------------------------------------------- graph
def build_graph(key, X, M=M_DEFAULT, efc=EFC_DEFAULT):
    path = f"{WORK}/hnsw_{key}_M{M}_efc{efc}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {n: z[n] for n in z.files}
    import faiss
    faiss.omp_set_num_threads(24)
    idx = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    idx.hnsw.efConstruction = efc
    idx.add(X)
    h = idx.hnsw
    g = dict(levels=faiss.vector_to_array(h.levels).astype(np.int32),
             offsets=faiss.vector_to_array(h.offsets).astype(np.int64),
             neighbors=faiss.vector_to_array(h.neighbors).astype(np.int32),
             cum_nb=faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64),
             entry=np.array([h.entry_point], dtype=np.int64),
             maxlev=np.array([h.max_level], dtype=np.int32))
    np.savez(path, **g)
    del idx
    return g


# -------------------------------------------------------------- beam search
_G = {}          # worker-side globals, inherited through fork


def _nb(i, lev):
    o = _G["offsets"][i]
    a, b = o + _G["cum_nb"][lev], o + _G["cum_nb"][lev + 1]
    v = _G["neighbors"][a:b]
    return v[v >= 0]


def _search(qi, ef, k=K):
    q = _G["Q"][qi]
    Sc, visited = _G["Sc"], set()
    sc = lambda ids: np.asarray(Sc[ids], dtype=np.float32) @ q

    ep = int(_G["entry"][0])
    cur = float(sc([ep])[0])
    visited.add(ep)
    for lev in range(int(_G["maxlev"][0]), 0, -1):       # greedy descent
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

    cand = [(-cur, ep)]                                  # max-heap on score
    W = [(cur, ep)]                                      # min-heap on score
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
    o = np.lexsort((ids, -s))
    return ids[o], s[o], len(visited)


def _work(args):
    qi, ef = args
    ids, s, nd = _search(qi, ef)
    gold = _G["GT"][qi]
    pos = {int(v): j for j, v in enumerate(ids)}
    sk = float(s[K - 1]) if len(s) >= K else float(s[-1])
    lv = np.full(K, np.inf, dtype=np.float32)
    for j, gx in enumerate(gold):
        p = pos.get(int(gx))
        if p is not None:
            lv[j] = max(0.0, sk - float(s[p]))
    out_s = np.full(ef, -np.inf, dtype=np.float32)
    out_s[:len(s)] = s
    return qi, out_s, lv, nd


def run(key, efs=EF_LADDER, nq=6000, M=M_DEFAULT, efc=EFC_DEFAULT, nproc=16):
    import multiprocessing as mp
    X, Q, cfg = load(key, nq=nq)
    gtp = f"{WORK}/gt_{key}_nq{len(Q)}.npy"
    GT = np.load(gtp) if os.path.exists(gtp) else gt_topk(X, Q)
    if not os.path.exists(gtp):
        np.save(gtp, GT)
    g = build_graph(key, X, M, efc)
    Sc = np.sign(X).astype(np.int8)
    del X
    _G.update(g); _G.update(Sc=Sc, Q=Q, GT=GT)

    for ef in efs:
        out = f"{WORK}/pool_{key}_M{M}_ef{ef}.npz"
        if os.path.exists(out):
            print(f"  {key} ef={ef} cached", flush=True)
            continue
        pool_s = np.empty((len(Q), ef), dtype=np.float32)
        gold_lv = np.empty((len(Q), K), dtype=np.float32)
        ndis = np.empty(len(Q), dtype=np.int32)
        with mp.Pool(nproc) as p:
            for qi, s, lv, nd in p.imap_unordered(
                    _work, [(i, ef) for i in range(len(Q))], chunksize=16):
                pool_s[qi], gold_lv[qi], ndis[qi] = s, lv, nd
        np.savez(out, pool_s=pool_s, gold_lv=gold_lv, ndis=ndis)
        rec = (gold_lv[:, :] <= 0).mean()          # recall at R = k (prefix k)
        inpool = np.isfinite(gold_lv).mean()
        print(f"  {key} ef={ef}: in-pool {inpool:.4f} "
              f"R@10(no refine) {rec:.4f} ndis {ndis.mean():.0f}", flush=True)


if __name__ == "__main__":
    for key in sys.argv[1:]:
        print(key, flush=True)
        run(key)


# ------------------------------------------------- pool ids (one config only)
def run_ids(key, ef, nq=6000, M=M_DEFAULT, efc=EFC_DEFAULT, nproc=16):
    """Also keep the pool's document ids, so the *pool-relative* top-k -- the
    quantity Figure 7(b) reports -- can be separated from the global top-k the
    theorem is stated over."""
    import multiprocessing as mp
    out = f"{WORK}/poolid_{key}_M{M}_ef{ef}.npz"
    if os.path.exists(out):
        return out
    X, Q, cfg = load(key, nq=nq)
    GT = np.load(f"{WORK}/gt_{key}_nq{len(Q)}.npy")
    g = build_graph(key, X, M, efc)
    Sc = np.sign(X).astype(np.int8)
    _G.update(g); _G.update(Sc=Sc, Q=Q, GT=GT, X=X)
    ids = np.full((len(Q), ef), -1, dtype=np.int32)

    def one(qi):
        i, s, nd = _search(qi, ef)
        return qi, i
    with mp.Pool(nproc) as p:
        for qi, i in p.imap_unordered(_idwork, [(q, ef) for q in range(len(Q))],
                                      chunksize=16):
            ids[qi, :len(i)] = i
    # FP32 score of every pool member, for the pool-relative top-k
    fp = np.empty((len(Q), ef), dtype=np.float32)
    for a in range(0, len(Q), 256):
        b = min(a + 256, len(Q))
        for r in range(a, b):
            v = ids[r]
            m = v >= 0
            fp[r, m] = X[v[m]] @ Q[r]
            fp[r, ~m] = -np.inf
    np.savez(out, ids=ids, fp=fp, gt=GT)
    return out


def _idwork(args):
    qi, ef = args
    i, s, nd = _search(qi, ef)
    return qi, i


# ------------------------------------------------- the same pools under TurboQuant
def run_tq(key, efs=(256,), nq=6000, M=M_DEFAULT, efc=EFC_DEFAULT, nproc=16):
    """Pools from the SAME FP32 graph, traversed with TurboQuant-1bit scores.

    TurboQuant-1bit rotates and then applies the MSE-optimal 1-bit codebook.  On
    unit-norm data the codebook scale is a global constant, so its ranking -- and
    therefore every candidate set the threshold rule can select -- is that of a
    sign code on rotated vectors.  The rotation is a property of the scorer, not
    of the index, so the graph is left exactly as the sign-code arm built it and
    only the scoring changes.  Ground truth is unchanged: the exact FP32 top-k of
    the original vectors.
    """
    import multiprocessing as mp
    import quantizers as QZ
    X, Q, cfg = load(key, nq=nq)
    GT = np.load(f"{WORK}/gt_{key}_nq{len(Q)}.npy")
    g = build_graph(key, X, M, efc)
    P = QZ._rot(X.shape[1], QZ.SEED + 1)
    Sc = np.empty(X.shape, dtype=np.int8)
    for lo in range(0, len(X), 100_000):
        hi = min(lo + 100_000, len(X))
        Sc[lo:hi] = np.sign(X[lo:hi] @ P).astype(np.int8)
    Qr = np.ascontiguousarray(Q @ P, dtype=np.float32)
    del X
    _G.update(g); _G.update(Sc=Sc, Q=Qr, GT=GT)

    for ef in efs:
        out = f"{WORK}/pooltq_{key}_M{M}_ef{ef}.npz"
        if os.path.exists(out):
            print(f"  TQ {key} ef={ef} cached", flush=True); continue
        pool_s = np.empty((len(Qr), ef), dtype=np.float32)
        gold_lv = np.empty((len(Qr), K), dtype=np.float32)
        ndis = np.empty(len(Qr), dtype=np.int32)
        with mp.Pool(nproc) as p:
            for qi, s, lv, nd in p.imap_unordered(
                    _work, [(i, ef) for i in range(len(Qr))], chunksize=16):
                pool_s[qi], gold_lv[qi], ndis[qi] = s, lv, nd
        np.savez(out, pool_s=pool_s, gold_lv=gold_lv, ndis=ndis)
        print(f"  TQ {key} ef={ef}: in-pool {np.isfinite(gold_lv).mean():.4f} "
              f"ndis {ndis.mean():.0f}", flush=True)
