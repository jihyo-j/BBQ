#!/usr/bin/env python3
"""Prepare shuffled-prefix Qwen-1024 inputs for the 100M scalability run.

All generated large files live under the caller-provided, user-owned NVMe
directory.  The source embedding tree and supplied exact ground truth are
read-only.
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
SOURCE_ROOT = Path("/home/smrc/samsung-nvme/sieun/qwen100m")
DIM = 1024
TOTAL_ROWS = 100_000_000
SCALES = (10_000_000, 30_000_000, 50_000_000, 100_000_000)
ROW_BYTES = DIM * np.dtype(np.float32).itemsize
ALIGNED_HEADER_BYTES = 4096


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"bad matrix header: {path}")
    return struct.unpack("<ii", raw)


def mmap_legacy(path: Path, dtype: np.dtype) -> np.memmap:
    n, d = matrix_shape(path)
    expected = 8 + n * d * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"unexpected legacy matrix size: {path}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def write_matrix(path: Path, rows: np.ndarray, dtype: np.dtype) -> None:
    rows = np.ascontiguousarray(rows, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<ii", *rows.shape))
        rows.tofile(handle)
    os.replace(tmp, path)


def build_chunk_plan(emb_root: Path) -> list[tuple[Path, int, int]]:
    chunks = sorted(emb_root.glob("*/*_c[0-9][0-9][0-9].npy"))
    plan: list[tuple[Path, int, int]] = []
    total = 0
    for path in chunks:
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != DIM or array.dtype != np.float32:
            raise ValueError(f"unexpected embedding chunk: {path} {array.shape} {array.dtype}")
        take = min(int(array.shape[0]), TOTAL_ROWS - total)
        if take <= 0:
            break
        plan.append((path, total, take))
        total += take
    if total != TOTAL_ROWS:
        raise ValueError(f"embedding chunks provide {total}, need {TOTAL_ROWS}")
    return plan


def prepare_splits(source_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    query_path = source_root / "insert_exp/queries/queries.fbin"
    query = mmap_legacy(query_path, np.float32)
    if query.shape != (4552, DIM):
        raise ValueError(f"unexpected query shape: {query.shape}")
    order = np.random.default_rng(11).permutation(query.shape[0])
    calibration_ids = order[:1000]
    test_ids = order[1000:]

    calibration_query = np.array(query[calibration_ids], dtype=np.float32, order="C")
    test_query = np.array(query[test_ids], dtype=np.float32, order="C")
    faiss.normalize_L2(calibration_query)
    faiss.normalize_L2(test_query)
    write_matrix(output / "seed11.cal1000.query.fbin", calibration_query, np.float32)
    write_matrix(output / "seed11.test3552.query.fbin", test_query, np.float32)
    np.save(output / "seed11.calibration_query_ids.npy", calibration_ids)
    np.save(output / "seed11.test_query_ids.npy", test_ids)

    gt_root = source_root / "insert_exp/gt"
    for scale in SCALES:
        source = mmap_legacy(gt_root / f"gt_{scale}.ibin", np.int32)
        if source.shape != (4552, 100):
            raise ValueError(f"unexpected GT shape for {scale}: {source.shape}")
        write_matrix(
            output / f"gt_{scale}.seed11.cal1000.k10.ibin",
            source[calibration_ids, :10],
            np.int32,
        )
        write_matrix(
            output / f"gt_{scale}.seed11.test3552.k10.ibin",
            source[test_ids, :10],
            np.int32,
        )
    metadata = {
        "source_query": str(query_path),
        "source_gt_root": str(gt_root),
        "split_seed": 11,
        "calibration_rows": 1000,
        "test_rows": 3552,
        "gt_width": 10,
        "scales": list(SCALES),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    log(f"query/GT splits ready -> {output}")


def current_rows(path: Path, repair_interrupted_append: bool = False) -> int:
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size < ALIGNED_HEADER_BYTES or (size - ALIGNED_HEADER_BYTES) % ROW_BYTES:
        raise ValueError(f"bad aligned prefix file size: {path} {size}")
    payload_rows = (size - ALIGNED_HEADER_BYTES) // ROW_BYTES
    with path.open("rb") as handle:
        n, d = struct.unpack("<ii", handle.read(8))
    if d == DIM and 0 <= n < payload_rows and repair_interrupted_append:
        repaired_size = ALIGNED_HEADER_BYTES + n * ROW_BYTES
        with path.open("r+b") as handle:
            handle.truncate(repaired_size)
            handle.flush()
            os.fsync(handle.fileno())
        log(
            f"recovered interrupted append: truncated payload rows "
            f"{payload_rows} -> committed header rows {n}"
        )
        payload_rows = n
    if d != DIM or n != payload_rows:
        raise ValueError(
            f"aligned prefix header/payload mismatch: header={(n, d)} payload_rows={payload_rows}"
        )
    return int(n)


def append_prefix(
    source_root: Path,
    perm_path: Path,
    output: Path,
    target_rows: int,
    threads: int,
    batch_rows: int,
    reserve_gib: int,
) -> None:
    if target_rows not in SCALES:
        raise ValueError(f"target rows must be one of {SCALES}")
    existing = current_rows(output, repair_interrupted_append=True)
    if existing == target_rows:
        log(f"reuse complete shuffled prefix {target_rows}: {output}")
        return
    if existing > target_rows:
        raise ValueError(f"cannot shrink prefix {existing} -> {target_rows}")

    growth = (target_rows - existing) * ROW_BYTES
    output.parent.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(output.parent)
    free = stat.f_bavail * stat.f_frsize
    reserve = reserve_gib * 1024**3
    if free < growth + reserve:
        raise RuntimeError(
            f"insufficient space under user-owned path {output.parent}: "
            f"free={free / 1024**3:.1f} GiB growth={growth / 1024**3:.1f} GiB "
            f"reserve={reserve_gib} GiB"
        )

    plan = build_chunk_plan(source_root / "emb")
    arrays = [np.load(path, mmap_mode="r") for path, _, _ in plan]
    starts = np.asarray([start for _, start, _ in plan], dtype=np.int64)
    ends = np.asarray([start + take for _, start, take in plan], dtype=np.int64)
    perm = np.load(perm_path, mmap_mode="r")
    if perm.shape != (TOTAL_ROWS,) or perm.dtype != np.int64:
        raise ValueError(f"unexpected permutation: {perm.shape} {perm.dtype}")

    first_creation = existing == 0
    write_path = output.with_name(output.name + ".tmp") if first_creation else output
    if first_creation and write_path.exists():
        write_path.unlink()
    mode = "wb" if first_creation else "r+b"
    started = time.perf_counter()
    with write_path.open(mode, buffering=16 * 1024 * 1024) as handle, ThreadPoolExecutor(
        max_workers=threads
    ) as pool:
        if first_creation:
            header = bytearray(ALIGNED_HEADER_BYTES)
            header[:8] = struct.pack("<ii", target_rows, DIM)
            handle.write(header)
        else:
            handle.seek(0, os.SEEK_END)
        for first in range(existing, target_rows, batch_rows):
            stop = min(target_rows, first + batch_rows)
            source_ids = np.asarray(perm[first:stop], dtype=np.int64)
            source_chunks = np.searchsorted(ends, source_ids, side="right")
            block = np.empty((stop - first, DIM), dtype=np.float32)

            def gather(chunk_id: int) -> None:
                positions = np.flatnonzero(source_chunks == chunk_id)
                local_ids = source_ids[positions] - starts[chunk_id]
                block[positions] = arrays[chunk_id][local_ids]

            list(pool.map(gather, np.unique(source_chunks).tolist()))
            block.tofile(handle)
            if stop % 5_000_000 == 0 or stop == target_rows:
                log(f"wrote shuffled prefix rows {stop}/{target_rows}")
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        handle.write(struct.pack("<ii", target_rows, DIM))
        handle.flush()
        os.fsync(handle.fileno())
    if first_creation:
        os.replace(write_path, output)
    if current_rows(output) != target_rows:
        raise RuntimeError(f"prefix verification failed: {output}")

    view = np.memmap(
        output,
        dtype=np.float32,
        mode="r",
        offset=ALIGNED_HEADER_BYTES,
        shape=(target_rows, DIM),
    )
    sample_ids = np.linspace(0, target_rows - 1, num=min(4096, target_rows), dtype=np.int64)
    error = float(np.max(np.abs(np.linalg.norm(view[sample_ids], axis=1) - 1.0)))
    if error > 2e-3:
        raise ValueError(f"source vectors are not normalized: max sampled norm error={error}")
    log(
        f"shuffled prefix ready rows={target_rows} seconds={time.perf_counter() - started:.1f} "
        f"sample_norm_error={error:.3g} -> {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-rows", type=int, required=True, choices=SCALES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO / "experiment/msmarco_qwen1024_100m_scalability_t24/inputs",
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--batch-rows", type=int, default=250_000)
    parser.add_argument("--reserve-gib", type=int, default=20)
    args = parser.parse_args()
    prepare_splits(args.source_root, args.split_dir)
    append_prefix(
        args.source_root,
        args.source_root / "insert_exp/gt/perm.npy",
        args.output,
        args.target_rows,
        args.threads,
        args.batch_rows,
        args.reserve_gib,
    )


if __name__ == "__main__":
    main()
