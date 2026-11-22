#!/usr/bin/env python3
"""Per-corpus calibration of BBQ's query-adaptive refinement threshold (Theorem 1, Appendix A).

Given, for m calibration queries, the ADC candidate pool L(q) returned by the frozen BBQ search
(``bbq_hnsw --candidate-only --no-rerank --candidates-out pool.csv``), this script

  1. finds H(q), the exact FP32 top-k *within the pool* (ties broken by document id),
  2. computes T(q) = max_{x in H(q)} [ s_k(q) - s(q,x) ]_+ in units of the query's int16 LUT scale,
  3. returns tau_hat = T_(l), the l-th order statistic with l = ceil((m+1)(1-rho)).

Serving then reranks C_tau(q) = { x in L(q) : s(q,x) >= s_k(q) - tau_hat }, i.e. the binary is run with
``--calibrated-threshold-rerank-tau <tau_hat>``. Under exchangeability of calibration and test queries,
Pr{ H(q) is fully retained } >= 1 - rho (marginal guarantee; see Appendix A.5 for scope).

The binary reports ADC *distances* d = -s, so "s >= s_k - tau" is "d <= d_k + tau * scale_q".
This is a standalone port of ``fit_pool_top10_conformal`` in
exp/scripts/run_adcbq_vs_fp_hnsw_fullvec_ram.py, which produced the paper's thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_fbin(path: Path, mmap: bool = False) -> np.ndarray:
    """Read an .fbin (int32 n, int32 d, float32 data). Supports an 8-byte or 4096-byte header."""
    with path.open("rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
    payload = int(n) * int(d) * 4
    size = path.stat().st_size
    offset = {8 + payload: 8, 4096 + payload: 4096}.get(size)
    if offset is None:
        raise ValueError(f"unexpected .fbin size for {path}")
    if mmap:
        return np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(int(n), int(d)))
    return np.fromfile(path, dtype=np.float32, offset=offset).reshape(int(n), int(d))


def int16_query_scales(queries: np.ndarray) -> np.ndarray:
    """Per-query LUT scale used by the int16 nibble ADC (32767 / max 4-dim block L1 norm)."""
    nq, dim = queries.shape
    if dim % 4:
        raise ValueError("int16 nibble LUT requires dim % 4 == 0")
    max_abs = np.abs(queries).reshape(nq, dim // 4, 4).sum(axis=2).max(axis=1)
    return np.divide(32767.0, max_abs, out=np.ones_like(max_abs, dtype=np.float32), where=max_abs > 0)


def load_pool(path: Path, nq: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the candidate pool CSV (columns: query_id, rank, id, distance)."""
    ids = np.full((nq, ef), -1, dtype=np.int64)
    dist = np.full((nq, ef), np.inf, dtype=np.float32)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            q, r = int(row["query_id"]), int(row["rank"])
            if 0 <= q < nq and 0 <= r < ef:
                ids[q, r] = int(row["id"])
                dist[q, r] = float(row["distance"])
    return ids, dist


def pool_thresholds(ids, dist, queries, base, scales, k: int):
    """T(q) for every calibration query, plus each query's exact pool top-k positions."""
    nq = ids.shape[0]
    T = np.zeros(nq, dtype=np.float64)
    top_pos = np.empty((nq, k), dtype=np.int32)
    for q in range(nq):
        valid = ids[q] >= 0
        row_ids, row_d = ids[q, valid], dist[q, valid]
        vecs = np.asarray(base[row_ids], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1)
        scores = np.divide(vecs @ queries[q], norms, out=np.full(len(row_ids), -np.inf, np.float32), where=norms > 0)
        order = np.lexsort((row_ids, -scores))[:k]          # FP32 score desc, then doc id asc
        top_pos[q] = order
        T[q] = max(0.0, float(np.max(row_d[order] - row_d[k - 1])) / float(scales[q]))
    return T, top_pos


def conformal_tau(T: np.ndarray, target: float) -> tuple[float, int]:
    m = len(T)
    order = math.ceil((m + 1) * target)
    if order > m:
        raise RuntimeError(f"calibration failure: target {target} needs m >= {math.ceil(1 / (1 - target)) - 1}, got m={m}")
    return float(np.partition(T, order - 1)[order - 1]), order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="candidates CSV from bbq_hnsw --candidates-out")
    ap.add_argument("--queries", type=Path, required=True, help="calibration queries (.fbin)")
    ap.add_argument("--base", type=Path, required=True, help="corpus FP32 vectors (.fbin), read by id")
    ap.add_argument("--ef", type=int, required=True, help="efSearch the pool was produced with (= pool size)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.99, help="1 - rho: pool top-k preservation probability")
    ap.add_argument("--out", type=Path, help="write the fitted threshold as JSON")
    a = ap.parse_args()

    queries = read_fbin(a.queries)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    scales = int16_query_scales(queries)
    ids, dist = load_pool(a.pool, len(queries), a.ef)
    base = read_fbin(a.base, mmap=True)

    T, top_pos = pool_thresholds(ids, dist, queries, base, scales, a.k)
    tau, order = conformal_tau(T, a.target)

    counts = np.minimum(np.maximum(a.k, np.sum(dist <= (dist[:, a.k - 1] + tau * scales)[:, None], axis=1)), a.ef)
    retained = np.all(top_pos < counts[:, None], axis=1)
    result = {
        "protocol": "per_ef_pool_topk_split_conformal",
        "efSearch": a.ef, "k": a.k, "tau": tau, "target": a.target,
        "conformal_order": order, "calibration_queries": len(T),
        "empirical_pool_topk_coverage": float(retained.mean()),
        "R_mean": float(counts.mean()), "R_p95": float(np.percentile(counts, 95)),
        "R_p99": float(np.percentile(counts, 99)), "R_max": int(counts.max()),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
