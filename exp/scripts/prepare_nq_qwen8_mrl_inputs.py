#!/usr/bin/env python3
"""Materialize normalized MRL prefixes of NQ Qwen3-8B and exact top-100 GT.

The 4096-D arm reuses the existing source files.  Lower-dimensional arms use
the first ``d`` components of the same 8B vectors, followed by L2
normalization.  Training files have a 4 KiB header so they can also serve as
the compact O_DIRECT FP store used by ADC-BQ and RaBitQ.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import faiss  # type: ignore
import numpy as np


SOURCE_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq")
SOURCE_TRAIN = SOURCE_ROOT / "nq_vanilla-qw8_train.fbin"
SOURCE_QUERY = SOURCE_ROOT / "nq_vanilla-qw8_query.fbin"
SOURCE_GT = SOURCE_ROOT / "nq_vanilla-qw8_neighbors_top100.ibin"
SOURCE_ALIGNED = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp/"
    "nq_qwen8_4096.fp32.header4096.fbin"
)
SOURCE_B1 = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/indexes/"
    "nq_qwen8_4096.rabitq_b1_c16.n1000000.d4096.M24.efC200.index"
)
SOURCE_B7 = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/indexes/"
    "nq_qwen8_4096.rabitq_b7_c16.n1000000.d4096.M24.efC200.index"
)
SOURCE_ADC = Path(
    "/home/smrc/samsung-nvme/jihyo/rabitq_v051_ssd_multidataset_indexes/"
    "nq_qwen8_4096/nq_qwen8_4096.fpgraph_adcbq.d4096.M24.efC200.faiss"
)
DEFAULT_OUTPUT = Path("/home/smrc/samsung-nvme/jihyo/nq_qwen8_mrl_four_methods_t1")
PAGE_BYTES = 4096
SOURCE_DIM = 4096
DEFAULT_DIMS = (256, 512, 1024, 2560, 4096)


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def matrix_info(path: Path, dtype: np.dtype) -> tuple[int, int, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"bad matrix header: {path}")
    rows, dim = struct.unpack("<ii", raw)
    payload = rows * dim * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    if actual == 8 + payload:
        offset = 8
    elif actual == PAGE_BYTES + payload:
        offset = PAGE_BYTES
    else:
        raise ValueError(f"unexpected matrix size: {path} ({actual})")
    return rows, dim, offset


def mmap_matrix(path: Path, dtype: np.dtype) -> np.memmap:
    rows, dim, offset = matrix_info(path, dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(rows, dim))


def normalized(block: np.ndarray) -> np.ndarray:
    result = np.array(block, dtype=np.float32, order="C", copy=True)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    np.maximum(norms, np.float32(1e-12), out=norms)
    result /= norms
    return result


def valid_fbin(path: Path, rows: int, dim: int, aligned: bool) -> bool:
    if not path.is_file():
        return False
    try:
        got_rows, got_dim, offset = matrix_info(path, np.float32)
        return (got_rows, got_dim, offset) == (
            rows,
            dim,
            PAGE_BYTES if aligned else 8,
        )
    except (OSError, ValueError):
        return False


def write_train_prefix(
    source: np.memmap,
    path: Path,
    dim: int,
    batch_rows: int,
) -> None:
    rows = source.shape[0]
    if valid_fbin(path, rows, dim, aligned=True):
        log(f"reuse MRL train d={dim}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb", buffering=16 * 1024 * 1024) as handle:
            header = bytearray(PAGE_BYTES)
            header[:8] = struct.pack("<ii", rows, dim)
            handle.write(header)
            for first in range(0, rows, batch_rows):
                stop = min(rows, first + batch_rows)
                normalized(source[first:stop, :dim]).tofile(handle)
                if stop == rows or stop % 100_000 == 0:
                    log(f"MRL train d={dim}: {stop:,}/{rows:,}")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if not valid_fbin(path, rows, dim, aligned=True):
        raise RuntimeError(f"invalid generated train file: {path}")


def write_query_prefix(source: np.memmap, path: Path, dim: int) -> None:
    rows = source.shape[0]
    if valid_fbin(path, rows, dim, aligned=False):
        return
    values = normalized(source[:, :dim])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<ii", rows, dim))
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_ibin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.int32)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<ii", *values.shape))
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exact_gt(
    train_path: Path,
    query_path: Path,
    output: Path,
    topk: int,
    threads: int,
    add_batch_rows: int,
) -> None:
    train = mmap_matrix(train_path, np.float32)
    queries = mmap_matrix(query_path, np.float32)
    if output.is_file():
        try:
            rows, width, offset = matrix_info(output, np.int32)
            if (rows, width, offset) == (queries.shape[0], topk, 8):
                log(f"reuse exact GT: {output}")
                return
        except (OSError, ValueError):
            pass
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(train.shape[1])
    started = time.perf_counter()
    for first in range(0, train.shape[0], add_batch_rows):
        stop = min(train.shape[0], first + add_batch_rows)
        index.add(np.ascontiguousarray(train[first:stop], dtype=np.float32))
        if stop == train.shape[0] or stop % 100_000 == 0:
            log(f"exact index d={train.shape[1]}: {stop:,}/{train.shape[0]:,}")
    _, labels = index.search(np.ascontiguousarray(queries), topk)
    write_ibin(output, labels)
    log(
        f"exact top-{topk} d={train.shape[1]} complete in "
        f"{time.perf_counter() - started:.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dims", nargs="+", type=int, default=list(DEFAULT_DIMS))
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()
    dims = list(dict.fromkeys(args.dims))
    if any(dim <= 0 or dim > SOURCE_DIM for dim in dims):
        parser.error(f"dimensions must be in [1, {SOURCE_DIM}]")
    train_source = mmap_matrix(SOURCE_TRAIN, np.float32)
    query_source = mmap_matrix(SOURCE_QUERY, np.float32)
    if train_source.shape[1] != SOURCE_DIM or query_source.shape[1] != SOURCE_DIM:
        raise ValueError("unexpected Qwen3-8B source dimension")

    data_root = args.output_root / "data"
    index_root = args.output_root / "indexes"
    entries: list[dict[str, object]] = []
    for dim in dims:
        key = f"nq_qwen8_mrl{dim}"
        if dim == SOURCE_DIM:
            train_path = SOURCE_TRAIN
            query_path = SOURCE_QUERY
            gt_path = SOURCE_GT
            fp_path = SOURCE_ALIGNED
            b1_path = SOURCE_B1
            b7_path = SOURCE_B7
            adc_path = SOURCE_ADC
            owned = False
            for path in (train_path, query_path, gt_path, fp_path, b1_path, b7_path, adc_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
        else:
            root = data_root / key
            train_path = root / f"{key}.train.header4096.fbin"
            query_path = root / f"{key}.query.fbin"
            gt_path = root / f"{key}.neighbors_top{args.topk}.ibin"
            write_train_prefix(train_source, train_path, dim, args.batch_rows)
            write_query_prefix(query_source, query_path, dim)
            exact_gt(
                train_path,
                query_path,
                gt_path,
                args.topk,
                args.threads,
                args.batch_rows,
            )
            fp_path = train_path
            method_root = index_root / key
            adc_path = method_root / f"{key}.adcbq.M24.efC200.faiss"
            b1_path = method_root / f"{key}.rabitq_b1_c16.M24.efC200.index"
            b7_path = method_root / f"{key}.rabitq_b7_c16.M24.efC200.index"
            owned = True
        entries.append(
            {
                "key": key,
                "corpus": "NQ",
                "model": f"Qwen3-Embedding-8B MRL-{dim}",
                "dim": dim,
                "train": str(train_path),
                "query": str(query_path),
                "gt": str(gt_path),
                "fp_sidecar": str(fp_path),
                "adcbq_index": str(adc_path),
                "b1_index": str(b1_path),
                "b7_index": str(b7_path),
                "owned_indexes": owned,
            }
        )
    manifest = args.output_root / "dataset_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "source_model": "Qwen3-Embedding-8B",
                "mrl_rule": "first-d-components-then-l2-normalize",
                "datasets": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    log(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
