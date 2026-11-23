#!/usr/bin/env python3
"""NQ Qwen-1024 (Qwen3-Embedding-0.6B): RaBitQ 7-bit c16, everything in RAM, on the BBQ test split.

For a like-for-like comparison with BBQ with its FP32 rerank vectors in process RAM
(nq_qwen1024_bbq_vs_int16query_nolut_t1, mode bbq_int16_nibble_lut), this measures the
production RaBitQ 7-bit c16 HNSW index (M24, efC200, no FP rerank) on exactly the same
2,452 held-out test queries (seed-11 split) and the same ground truth, 1 search thread,
3 rounds (QPS = median), pinned with taskset by the caller.

  build : run_nq_quora_c16_rabitq.build_dataset(only_bits=(7,), build_threads=24)
          (same recipe as the October index, which is no longer on disk)
  search: --search (separate, pinned process)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))

import numpy as np  # noqa: E402

# The FAISS clustering and the RaBitQ native module pick incompatible MKL builds in one
# process, so clustering runs in its own process ("cluster" stage, no RaBitQ import) and the
# build/search stages import the RaBitQ runner lazily.
rq = None
calibration = None


def _load():
    global rq, calibration
    import run_nq_quora_c16_rabitq as _rq
    import run_adcbq_calibrated_threshold_overlap as _cal
    rq, calibration = _rq, _cal

OUT = EXPERIMENT / "nq_qwen1024_rabitq_b7_ram_testsplit_t1"
SPLIT = EXPERIMENT / "adcbq_calibrated_threshold_overlap/splits/nq_qwen06_1024"
TEST_Q = SPLIT / "seed11.cal1000.test2452.test.query.fbin"
TEST_GT = SPLIT / "seed11.cal1000.test2452.test.gt.ibin"
EFS = (64, 80, 128, 256, 512, 1024, 1536, 2048)
ROUNDS = 3


def dataset():
    (d,) = rq.selected_datasets(["nq_qwen06_1024"])
    return d


def cluster() -> None:
    """IVF16 centroids + assignment, identical to run_nq_quora_c16_rabitq.train_and_assign."""
    import faiss
    import run_v051_ssd_odirect_multidataset as base
    d = base.DATASETS["nq_qwen06_1024"]
    train = base.read_fbin(d.train).astype(np.float32, copy=False)
    for st in range(0, len(train), 65536):
        b = train[st : st + 65536]
        n = np.linalg.norm(b, axis=1, keepdims=True); np.maximum(n, 1e-12, out=n); b /= n
    faiss.omp_set_num_threads(24)
    q = faiss.index_factory(train.shape[1], "IVF16,Flat", faiss.METRIC_INNER_PRODUCT)
    q.verbose = True
    q.cp.seed = 100
    q.train(train)
    centroids = np.ascontiguousarray(q.quantizer.reconstruct_n(0, q.nlist), dtype=np.float32)
    _, ids = q.quantizer.search(train, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "centroids.npy", centroids)
    np.save(OUT / "cluster_ids.npy", np.ascontiguousarray(ids.reshape(-1), dtype=np.uint32))
    print("clustered", centroids.shape, flush=True)


def build() -> None:
    _load()
    d = dataset()
    path = rq.index_path(d, 7)
    if path.exists():
        print(f"index exists: {path}", flush=True)
        return
    train = rq.base.read_fbin(d.train)
    rq.normalize_rows(train)
    centroids = np.load(OUT / "centroids.npy")
    cluster_ids = np.load(OUT / "cluster_ids.npy")
    assert len(cluster_ids) == len(train) and centroids.shape == (rq.NUM_CLUSTERS, train.shape[1])
    index = rq.HnswIndex(dim=train.shape[1], max_elements=train.shape[0], M=rq.M,
                         ef_construction=rq.EF_CONSTRUCTION, nbits=7, metric="ip", random_seed=100)
    t = time.perf_counter()
    index.build(train, centroids, cluster_ids, num_threads=24, fast_quantization=False)
    print(f"built b7 in {time.perf_counter() - t:.1f}s", flush=True)
    index.save(str(path))
    print("saved", path, path.stat().st_size, flush=True)


def search() -> None:
    _load()
    d = dataset()
    path = rq.index_path(d, 7)
    q = rq.base.read_fbin(TEST_Q).astype(np.float32, copy=True)
    rq.normalize_rows(q)
    rows_gt, gt_dim = calibration.matrix_header(TEST_GT)
    gt = np.asarray(np.memmap(TEST_GT, dtype=np.int32, mode="r", offset=8, shape=(rows_gt, gt_dim))[:, :10])
    assert len(q) == len(gt) == 2452
    started = time.perf_counter()
    index = rq.HnswIndex.load(str(path))
    load_s = time.perf_counter() - started
    assert index.nbits == 7 and index.num_clusters == rq.NUM_CLUSTERS
    rows = []
    for ef in EFS:
        qps, rec = [], []
        for r in range(ROUNDS):
            t = time.perf_counter()
            ids, _ = index.search(q, k=10, ef=ef, num_threads=1)
            e = time.perf_counter() - t
            ids = np.asarray(ids)
            rec.append(float(np.mean([len(set(a[:10]) & set(b)) / 10 for a, b in zip(ids, gt)])))
            qps.append(len(q) / e)
            print(f"b7 ef={ef} round={r + 1} R@10={rec[-1]:.6f} QPS={qps[-1]:.1f}", flush=True)
        assert max(rec) - min(rec) < 1e-12
        rows.append(
            {
                "dataset": d.key,
                "method_key": "rabitq_b7_c16_ram",
                "method": "RaBitQ 7-bit c16 (RAM, no rerank)",
                "efSearch": ef,
                "test_queries": len(q),
                "recall_at_10": rec[0],
                "qps_median": statistics.median(qps),
                "qps_mean": statistics.fmean(qps),
                "rounds": ROUNDS,
                "search_threads": 1,
                "M": rq.M,
                "efConstruction": rq.EF_CONSTRUCTION,
                "num_clusters": rq.NUM_CLUSTERS,
                "index_path": str(path),
                "index_bytes": path.stat().st_size,
                "load_seconds": load_s,
                "queries": str(TEST_Q),
                "gt": str(TEST_GT),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary_t1.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("WROTE", OUT / "summary_t1.csv", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("cluster", "build", "search"))
    a = p.parse_args()
    {"cluster": cluster, "build": build, "search": search}[a.stage]()
