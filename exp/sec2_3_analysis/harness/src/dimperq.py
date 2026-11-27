"""Score error c_Q of EVERY quantizer across the MRL ladder (one checkpoint).

Figure 3(c) walks the MRL ladder with three arms; Figure 3(d) walks the
quantizer list at a fixed 1024d.  This fills in the grid: c_Q = nu/kappa for all
ten quantizers at d = 32 .. 4096 of the same Qwen3-Embedding-8B checkpoint on
MS MARCO, truncated to the first d coordinates and renormalised (the model
card's MRL recipe).  Nothing but d moves, so the question "does the blessing of
dimension belong to the sign code or to every quantizer?" is read off directly.

kappa is the Pearson correlation between the exact cosine t and the estimated
score a over the WHOLE corpus, per query, exactly as in measure.py -- but
computed from the score blocks themselves rather than from a d x d Gram, which
at d = 4096 with nq = 1000 is four times the arithmetic.  Sums are accumulated
in float64 over row blocks.  R* and CandidateRecall are deliberately not
measured here: they need a per-query rank scan and this grid is 80 cells.

The ADC-BQ column reproduces out/dimsweep_msmarco_qw8.json (same N, nq, dims),
which is the cross-check that the cheaper kappa route agrees with the Gram one.

  python3 src/dimperq.py [key] [nq]      ->  out/dimperq_<key>.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CONFIGS, OUT, label
import quantizers as QZ

DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
ARMS = QZ.ARMS                      # all ten, byte-matched ones and SQ alike
QCH_BYTES = 1.0e9                   # cap on one (N x nq_chunk) score block
RCH = 200_000                       # row block for the float64 accumulation


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def moments(B, out_s, out_ss):
    """Accumulate column sums and sums of squares of a score block in float64."""
    for lo in range(0, B.shape[0], RCH):
        C = B[lo:lo + RCH]
        out_s += C.sum(0, dtype=np.float64)
        out_ss += np.einsum("ij,ij->j", C, C, dtype=np.float64)


def cross(A, B, out):
    for lo in range(0, A.shape[0], RCH):
        out += np.einsum("ij,ij->j", A[lo:lo + RCH], B[lo:lo + RCH], dtype=np.float64)


def run(key="msmarco_qw8", nq=1000):
    nq = int(nq)
    c = CONFIGS[key]
    log(f"loading {key} ({label(key)})")
    X0 = c["base"]().astype(np.float32)
    Q0 = c["query"]().astype(np.float32)[:nq]
    N, D = X0.shape
    nq = Q0.shape[0]
    dims = [d for d in DIMS if d <= D]
    log(f"N={N} D={D} nq={nq} dims={dims} arms={len(ARMS)}")

    res = dict(key=key, corpus=c["corpus"], encoder=c["encoder"], N=int(N),
               nq=int(nq), D=int(D), dims=dims, arms=ARMS,
               nice={a: QZ.NICE[a] for a in ARMS}, rows=[])
    qch = max(1, int(QCH_BYTES / (N * 4)))

    for d in dims:
        t0 = time.time()
        if d == D:
            X = X0                                   # no copy at the full width
        else:
            X = np.ascontiguousarray(X0[:, :d])
        X /= np.linalg.norm(X, axis=1, keepdims=True).clip(1e-30)
        Q = np.ascontiguousarray(Q0[:, :d])
        Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-30)

        # exact side: the score block is kept, so kappa needs one pass per arm
        T = np.empty((N, nq), dtype=np.float32)
        s_t = np.zeros(nq); ss_t = np.zeros(nq)
        for lo in range(0, nq, qch):
            hi = min(lo + qch, nq)
            T[:, lo:hi] = X @ Q[lo:hi].T      # a column slice is not C-contiguous
            moments(T[:, lo:hi], s_t[lo:hi], ss_t[lo:hi])
        m_t = s_t / N
        sd_t = np.sqrt(np.maximum(ss_t / N - m_t ** 2, 1e-30))

        # f = <x, sign(x)> / (sqrt(d) ||x||): the sign code's own cosine.  The
        # isotropic reference sqrt(1-f^2)/f is what a sign code would cost if
        # the direction carried nothing beyond its coordinate magnitudes.
        sub = X[:200_000]
        f = float(np.mean(np.abs(sub).sum(1) / (np.sqrt(d) * np.linalg.norm(sub, axis=1))))
        row = dict(d=d, f_sign_cos=f, c_Q_iso=float(np.sqrt(1 - f * f) / f), arms={})

        for arm in ARMS:
            ta = time.time()
            Qt, Xh, nb, note = QZ.build(arm, X, Q)
            s_a = np.zeros(nq); ss_a = np.zeros(nq); s_ta = np.zeros(nq)
            for lo in range(0, nq, qch):
                hi = min(lo + qch, nq)
                A = Xh @ Qt[lo:hi].T
                moments(A, s_a[lo:hi], ss_a[lo:hi])
                cross(T[:, lo:hi], A, s_ta[lo:hi])
                del A
            del Qt, Xh
            m_a = s_a / N
            var_a = np.maximum(ss_a / N - m_a ** 2, 0.0)
            ms_a = np.maximum(ss_a / N, 1e-30)
            ok = var_a > 1e-12 * ms_a            # a degenerate code has no variance
            kap = np.full(nq, np.nan)
            kap[ok] = ((s_ta[ok] / N - m_t[ok] * m_a[ok])
                       / (sd_t[ok] * np.sqrt(var_a[ok])))
            kap = np.clip(kap, -1.0, 1.0)
            cq = np.sqrt(np.maximum(1 - kap ** 2, 0.0)) / kap
            row["arms"][arm] = dict(
                name=QZ.NICE[arm], bytes_per_vec=float(nb), note=note,
                kappa=float(np.nanmean(kap)), c_Q=float(np.nanmean(cq)),
                c_Q_med=float(np.nanmedian(cq)),
                c_Q_p95=float(np.nanquantile(cq, .95)),
                ndeg=int((~ok).sum()), secs=round(time.time() - ta, 1))
            a = row["arms"][arm]
            log(f"  d={d:<5d} {a['name']:22s} {nb:7.1f}B  kappa={a['kappa']:.4f}"
                f"  c_Q={a['c_Q']:.4f}  [{a['secs']}s]")
        del T
        if X is not X0:
            del X
        res["rows"].append(row)
        log(f"d={d} done in {time.time()-t0:.0f}s  (f={f:.4f}, "
            f"iso={row['c_Q_iso']:.4f})")
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(f"{OUT}/dimperq_{key}.json", "w"), indent=1)

    log(f"wrote {OUT}/dimperq_{key}.json")
    hdr = "d      f     iso   " + "".join(f"{QZ.NICE[a][:10]:>11s}" for a in ARMS)
    print("\n" + hdr)
    for r in res["rows"]:
        print(f"{r['d']:<6d} {r['f_sign_cos']:.3f} {r['c_Q_iso']:.3f} "
              + "".join(f"{r['arms'][a]['c_Q']:11.4f}" for a in ARMS))


if __name__ == "__main__":
    run(*(sys.argv[1:] or []))
