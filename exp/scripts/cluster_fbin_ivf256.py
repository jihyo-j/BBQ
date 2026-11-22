#!/usr/bin/env python3
"""Create normalized-IP IVF centroids and assignments for RaBitQ builds."""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import faiss  # type: ignore
import numpy as np


def read_fbin(path: Path) -> np.ndarray:
    with path.open("rb") as fp:
        n, dim = struct.unpack("<ii", fp.read(8))
        data = np.fromfile(fp, dtype=np.float32, count=n * dim)
    if data.size != n * dim:
        raise ValueError(f"short fbin: {path}")
    return np.ascontiguousarray(data.reshape(n, dim), dtype=np.float32)


def normalize_rows(x: np.ndarray, batch: int = 65536) -> None:
    for start in range(0, len(x), batch):
        block = x[start : start + batch]
        norms = np.sqrt(np.sum(block * block, axis=1, keepdims=True))
        np.maximum(norms, 1e-12, out=norms)
        block /= norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--centroids", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    print(f"CLUSTER read {args.train}", flush=True)
    train = read_fbin(args.train)
    normalize_rows(train)
    faiss.omp_set_num_threads(args.threads)
    coarse = faiss.index_factory(
        train.shape[1], f"IVF{args.clusters},Flat", faiss.METRIC_INNER_PRODUCT
    )
    coarse.cp.seed = args.seed
    start = time.perf_counter()
    coarse.train(train)
    centroids = np.ascontiguousarray(
        coarse.quantizer.reconstruct_n(0, coarse.nlist), dtype=np.float32
    )
    _, assigned = coarse.quantizer.search(train, 1)
    assignments = np.ascontiguousarray(assigned.reshape(-1), dtype=np.uint32)
    args.centroids.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.centroids, centroids)
    np.save(args.assignments, assignments)
    print(
        f"CLUSTER wrote centroids={centroids.shape}, assignments={assignments.shape}, "
        f"seconds={time.perf_counter()-start:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
