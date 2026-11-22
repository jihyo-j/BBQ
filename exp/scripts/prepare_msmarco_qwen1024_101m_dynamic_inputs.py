#!/usr/bin/env python3
"""Prepare the calibrated ADC-BQ 1M + {10M,30M,50M,100M} run.

The initial 1M corpus is the existing MS MARCO passage set.  Insertions are
the fixed-seed shuffled prefixes of the new MS MARCO v2.1 100M corpus.  Both
were encoded by Qwen3-Embedding-0.6B at 1024 dimensions and L2 normalized.

The v2.1 corpus already has exact top-100 results for all 4,552 queries at
10M/30M/50M/100M.  We compute exact top-100 only for the old 1M base, merge
the two exact lists, and write top-10 GT for the combined 11M/31M/51M/101M
corpora.  The combined aligned fbin is materialized in tmpfs for the dynamic
runner and exact-FP reranking.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024")
NEW_ROOT = Path("/home/smrc/samsung-nvme/sieun/qwen100m")
INSERT_SIZES = (10_000_000, 30_000_000, 50_000_000, 100_000_000)
DIM = 1024
BASE_N = 1_000_000
NEW_N = 100_000_000


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def matrix_header(path: Path, dtype: np.dtype) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        n, d = struct.unpack("<ii", handle.read(8))
    payload = int(n) * int(d) * np.dtype(dtype).itemsize
    size = path.stat().st_size
    if size == 8 + payload:
        offset = 8
    elif size == 4096 + payload:
        offset = 4096
    else:
        raise ValueError(
            f"unexpected matrix size for {path}: got {size}, expected {8 + payload} or {4096 + payload}"
        )
    return int(n), int(d), offset


def mmap_matrix(path: Path, dtype: np.dtype) -> np.memmap:
    n, d, offset = matrix_header(path, dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(n, d))


def write_matrix(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(values, dtype=dtype)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<ii", int(values.shape[0]), int(values.shape[1])))
        values.tofile(handle)
    os.replace(tmp, path)


def write_i64_range(path: Path, start: int, stop: int, batch: int = 2_000_000) -> None:
    expected = 8 + (stop - start) * 8
    if path.is_file() and path.stat().st_size == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<Q", stop - start))
        for first in range(start, stop, batch):
            np.arange(first, min(stop, first + batch), dtype=np.int64).tofile(handle)
    os.replace(tmp, path)


def split_queries(
    source: Path,
    out_dir: Path,
    calibration_count: int,
    test_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    query = mmap_matrix(source, np.float32)
    needed = calibration_count + test_count
    if needed > query.shape[0]:
        raise ValueError("calibration+test query count exceeds source query count")
    order = np.random.default_rng(seed).permutation(query.shape[0])[:needed]
    calibration_ids = np.sort(order[:calibration_count])
    test_ids = np.sort(order[calibration_count:])
    calibration = np.array(query[calibration_ids], dtype=np.float32, order="C", copy=True)
    test = np.array(query[test_ids], dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(calibration)
    faiss.normalize_L2(test)
    calibration_path = out_dir / "seed11.cal100.query.fbin"
    test_path = out_dir / "seed11.test1000.query.fbin"
    write_matrix(calibration_path, calibration, np.float32)
    write_matrix(test_path, test, np.float32)
    np.save(out_dir / "seed11.calibration_query_ids.npy", calibration_ids)
    np.save(out_dir / "seed11.test_query_ids.npy", test_ids)
    return calibration_ids, test_ids, calibration_path, test_path


def exact_old_base(
    base_path: Path,
    queries: np.ndarray,
    topk: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = mmap_matrix(base_path, np.float32)
    if base.shape != (BASE_N, DIM):
        raise ValueError(f"unexpected base shape: {base.shape}")
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(DIM)
    batch = 50_000
    started = time.perf_counter()
    for first in range(0, BASE_N, batch):
        block = np.array(base[first : first + batch], dtype=np.float32, order="C", copy=True)
        faiss.normalize_L2(block)
        index.add(block)
    q = np.array(queries, dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(q)
    scores, labels = index.search(q, topk)
    log(f"old 1M exact top-{topk}: {time.perf_counter() - started:.3f}s")
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def combined_top10(
    old_scores: np.ndarray,
    old_labels: np.ndarray,
    new_scores: np.ndarray,
    new_labels: np.ndarray,
) -> np.ndarray:
    scores = np.concatenate([old_scores, new_scores], axis=1)
    labels = np.concatenate(
        [old_labels.astype(np.int64), new_labels.astype(np.int64) + BASE_N], axis=1
    )
    keep = np.argpartition(scores, -10, axis=1)[:, -10:]
    kept_scores = np.take_along_axis(scores, keep, axis=1)
    kept_labels = np.take_along_axis(labels, keep, axis=1)
    result = np.empty((scores.shape[0], 10), dtype=np.int32)
    for row in range(scores.shape[0]):
        order = np.lexsort((kept_labels[row], -kept_scores[row]))
        result[row] = kept_labels[row, order]
    return result


def prepare_ground_truth(
    base_path: Path,
    query_path: Path,
    test_ids: np.ndarray,
    new_gt_dir: Path,
    out_dir: Path,
    threads: int,
) -> dict[str, str]:
    queries = np.asarray(mmap_matrix(query_path, np.float32), dtype=np.float32)
    old_scores, old_labels = exact_old_base(base_path, queries, 100, threads)
    base_gt_path = out_dir / "seed11.test1000.base1m.gt_k10.ibin"
    write_matrix(base_gt_path, old_labels[:, :10], np.int32)
    paths: dict[str, str] = {"base": str(base_gt_path)}
    for insert_size in INSERT_SIZES:
        new_label_path = new_gt_dir / f"gt_{insert_size}.ibin"
        new_score_path = new_gt_dir / f"gt_{insert_size}.dist.fbin"
        all_labels = mmap_matrix(new_label_path, np.int32)
        all_scores = mmap_matrix(new_score_path, np.float32)
        if all_labels.shape != all_scores.shape or all_labels.shape[1] < 100:
            raise ValueError(f"bad existing GT pair for insertion size {insert_size}")
        new_labels = np.asarray(all_labels[test_ids, :100], dtype=np.int32)
        new_scores = np.asarray(all_scores[test_ids, :100], dtype=np.float32)
        merged = combined_top10(old_scores, old_labels, new_scores, new_labels)
        output = out_dir / (
            f"qwen1024_1m_plus_{insert_size}_random_gt_k10_q{len(test_ids)}.ibin"
        )
        write_matrix(output, merged, np.int32)
        paths[str(insert_size)] = str(output)
        log(f"combined exact GT: 1M + {insert_size} -> {output}")
    return paths


def build_chunk_plan(emb_root: Path) -> list[tuple[Path, int, int]]:
    chunks = sorted(emb_root.glob("*/*_c[0-9][0-9][0-9].npy"))
    plan: list[tuple[Path, int, int]] = []
    total = 0
    for path in chunks:
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != DIM or array.dtype != np.float32:
            raise ValueError(f"unexpected embedding chunk: {path} {array.shape} {array.dtype}")
        take = min(int(array.shape[0]), NEW_N - total)
        if take <= 0:
            break
        plan.append((path, total, take))
        total += take
    if total != NEW_N:
        raise ValueError(f"embedding chunks provide {total}, need {NEW_N}")
    return plan


def combined_fbin_complete(path: Path) -> bool:
    expected = 4096 + (BASE_N + NEW_N) * DIM * 4
    if not path.is_file() or path.stat().st_size != expected:
        return False
    try:
        n, d, offset = matrix_header(path, np.float32)
        return (n, d, offset) == (BASE_N + NEW_N, DIM, 4096)
    except (OSError, ValueError):
        return False


def materialize_combined_fbin(
    base_path: Path,
    emb_root: Path,
    perm_path: Path,
    output: Path,
    threads: int,
    write_batch: int,
    materialize_mode: str,
) -> None:
    if combined_fbin_complete(output):
        log(f"reuse complete combined fbin: {output}")
        return
    plan = build_chunk_plan(emb_root)
    perm = np.load(perm_path, mmap_mode="r")
    if perm.shape != (NEW_N,) or perm.dtype != np.int64:
        raise ValueError(f"unexpected permutation: {perm.shape} {perm.dtype}")
    output.parent.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize
    required = 4096 + (BASE_N + NEW_N) * DIM * 4
    if free < required + 8 * 1024**3:
        raise RuntimeError(
            f"only {free / 1024**3:.1f} GiB free at {output.parent}; "
            f"need {(required + 8 * 1024**3) / 1024**3:.1f} GiB"
        )

    started = time.perf_counter()
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    base = mmap_matrix(base_path, np.float32)
    vectors: np.ndarray | None = None
    chunk_arrays: list[np.ndarray] = []
    chunk_starts = np.empty(0, dtype=np.int64)
    chunk_ends = np.empty(0, dtype=np.int64)
    if materialize_mode == "memory":
        log(f"loading {NEW_N} new vectors from {len(plan)} chunks into RAM")
        vectors = np.empty((NEW_N, DIM), dtype=np.float32)

        def load(job: tuple[Path, int, int]) -> str:
            path, offset, take = job
            assert vectors is not None
            vectors[offset : offset + take] = np.load(path, mmap_mode="r")[:take]
            return path.name

        with ThreadPoolExecutor(max_workers=threads) as load_pool:
            for done, _ in enumerate(load_pool.map(load, plan), start=1):
                if done == 1 or done % 20 == 0 or done == len(plan):
                    log(f"loaded chunks {done}/{len(plan)}")
        log(f"writing in-memory shuffled combined fbin -> {output}")
    elif materialize_mode == "streaming":
        chunk_arrays = [np.load(path, mmap_mode="r") for path, _, _ in plan]
        chunk_starts = np.asarray([offset for _, offset, _ in plan], dtype=np.int64)
        chunk_ends = np.asarray([offset + take for _, offset, take in plan], dtype=np.int64)
        log(
            f"streaming aligned combined fbin from {len(plan)} source chunks: "
            f"old 1M + shuffled new 100M -> {output}"
        )
    else:
        raise ValueError(f"unknown materialize mode: {materialize_mode}")
    with tmp.open("wb", buffering=16 * 1024 * 1024) as handle, ThreadPoolExecutor(
        max_workers=threads
    ) as pool:
        header = np.zeros(4096, dtype=np.uint8)
        header[:8] = np.frombuffer(struct.pack("<ii", BASE_N + NEW_N, DIM), dtype=np.uint8)
        header.tofile(handle)
        for first in range(0, BASE_N, write_batch):
            np.asarray(base[first : first + write_batch], dtype=np.float32).tofile(handle)
        for first in range(0, NEW_N, write_batch):
            stop = min(NEW_N, first + write_batch)
            if vectors is not None:
                block = vectors[perm[first:stop]]
            else:
                source_ids = np.asarray(perm[first:stop], dtype=np.int64)
                source_chunks = np.searchsorted(chunk_ends, source_ids, side="right")
                block = np.empty((stop - first, DIM), dtype=np.float32)

                def gather(chunk_id: int) -> None:
                    positions = np.flatnonzero(source_chunks == chunk_id)
                    local_ids = source_ids[positions] - chunk_starts[chunk_id]
                    block[positions] = chunk_arrays[chunk_id][local_ids]

                active_chunks = np.unique(source_chunks).tolist()
                list(pool.map(gather, active_chunks))
                del source_ids, source_chunks
            block.tofile(handle)
            del block
            if stop % 5_000_000 == 0 or stop == NEW_N:
                log(f"wrote shuffled new rows {stop}/{NEW_N}")
    os.replace(tmp, output)
    if not combined_fbin_complete(output):
        raise RuntimeError(f"combined fbin verification failed: {output}")
    log(f"combined fbin ready in {time.perf_counter() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-base", type=Path, default=OLD_ROOT / "msmarco-1m_base.fbin")
    parser.add_argument("--emb-root", type=Path, default=NEW_ROOT / "emb")
    parser.add_argument(
        "--source-query",
        type=Path,
        default=NEW_ROOT / "insert_exp/queries/queries.fbin",
    )
    parser.add_argument(
        "--new-gt-dir", type=Path, default=NEW_ROOT / "insert_exp/gt"
    )
    parser.add_argument(
        "--perm", type=Path, default=NEW_ROOT / "insert_exp/gt/perm.npy"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO / "experiment/dynamic_insert_101m/inputs"
    )
    parser.add_argument(
        "--combined-fbin",
        type=Path,
        default=Path("/dev/shm/msmarco_qwen1024_dynamic_101m/base101m.aligned4k.fbin"),
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--write-batch", type=int, default=250_000)
    parser.add_argument(
        "--materialize-mode",
        choices=("streaming", "memory"),
        default="streaming",
        help="memory is faster but needs roughly 772 GiB peak RAM+tmpfs for 101M",
    )
    parser.add_argument("--query-split-seed", type=int, default=11)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration_ids, test_ids, calibration_path, test_path = split_queries(
        args.source_query, args.out_dir, 100, 1000, args.query_split_seed
    )
    gt_paths = prepare_ground_truth(
        args.old_base,
        test_path,
        test_ids,
        args.new_gt_dir,
        args.out_dir,
        args.threads,
    )
    base_map = args.out_dir / "base_1m_to_combined_ids.i64bin"
    selected = args.out_dir / "new_100m_selected_ids.i64bin"
    write_i64_range(base_map, 0, BASE_N)
    write_i64_range(selected, BASE_N, BASE_N + NEW_N)

    materialize_combined_fbin(
        args.old_base,
        args.emb_root,
        args.perm,
        args.combined_fbin,
        args.threads,
        args.write_batch,
        args.materialize_mode,
    )
    manifest = {
        "base": str(args.old_base),
        "insert_source": str(args.emb_root),
        "combined_fbin": str(args.combined_fbin),
        "combined_shape": [BASE_N + NEW_N, DIM],
        "corpus_composition": "MSMARCO passage 1M base + shuffled MSMARCO v2.1 100M inserts",
        "embedding": "Qwen3-Embedding-0.6B, 1024D, L2 normalized",
        "insert_sizes": list(INSERT_SIZES),
        "corpus_sizes": [BASE_N + value for value in INSERT_SIZES],
        "centroid_schedule_not_used_by_adcbq": {
            "1000000": 1024,
            "11000000": 3072,
            "31000000": 6144,
            "51000000": 7168,
            "101000000": 10240,
        },
        "calibration_query_ids": calibration_ids.tolist(),
        "test_query_ids": test_ids.tolist(),
        "calibration_query": str(calibration_path),
        "test_query": str(test_path),
        "base_to_larger_ids": str(base_map),
        "selected_ids": str(selected),
        "gt_paths": gt_paths,
        "calibration_target": 0.99,
        "calibration_scope": "independent per corpus/mode/efSearch operating point",
        "query_preprocess_scope": "ADC-BQ int16 nibble LUT set_query plus calibrated tau scaling",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(f"manifest saved -> {manifest_path}")


if __name__ == "__main__":
    main()
