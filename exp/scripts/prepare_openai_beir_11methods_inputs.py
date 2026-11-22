#!/usr/bin/env python3
"""Prepare the OpenAI BEIR datasets and DBpedia BGE-large for T=1.

The source embeddings are already L2-normalized float32 arrays.  This script
converts them to the fbin layout consumed by the native benchmark binaries,
computes exact inner-product top-100 ground truth, and publishes one manifest
only after every requested dataset has durable inputs.  Every output is
restartable and is written atomically in its final external-SSD directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import time
from pathlib import Path

import faiss
import numpy as np


SOURCE_ROOT = Path("/home/smrc/samsung-nvme/sieun/beir_emb")
BGE_ROOT = Path("/home/smrc/samsung-nvme/sieun/SYN/emb")
OUTPUT_ROOT = Path("/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1")
PAGE_BYTES = 4096
CORPORA = ("dbpedia-entity", "msmarco", "nq", "quora")
MODELS = {
    "small": (1536, "OpenAI text-embedding-3-small"),
    "large": (3072, "OpenAI text-embedding-3-large"),
}
CALIBRATION_ROWS = {
    "dbpedia-entity": 200,
    "msmarco": 1000,
    "nq": 1000,
    "quora": 1000,
}
BGE_LARGE_KEY = "sieun_dbpedia_entity_bge_large1024_beir400"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def key_for(corpus: str, dim: int) -> str:
    return f"sieun_{corpus.replace('-', '_')}_openai{dim}"


def valid_matrix(path: Path, rows: int, dim: int, dtype: np.dtype) -> bool:
    if not path.is_file() or path.stat().st_size != 8 + rows * dim * dtype.itemsize:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == struct.pack("<ii", rows, dim)


def convert_npy(source: Path, output: Path, *, chunk_mib: int = 256) -> tuple[int, int]:
    values = np.load(source, mmap_mode="r")
    if values.ndim != 2 or values.dtype not in (np.float16, np.float32):
        raise ValueError(
            f"expected 2-D float16/float32 npy: {source}, "
            f"got {values.shape}/{values.dtype}"
        )
    rows, dim = map(int, values.shape)
    output_dtype = np.dtype(np.float32)
    if valid_matrix(output, rows, dim, output_dtype):
        print(f"REUSE {output}", flush=True)
        return rows, dim
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    chunk_rows = max(1, chunk_mib * 1024**2 // (dim * output_dtype.itemsize))
    print(
        f"CONVERT {source} -> {output} shape=({rows},{dim}) "
        f"{values.dtype}->float32",
        flush=True,
    )
    try:
        with temporary.open("wb", buffering=0) as handle:
            handle.write(struct.pack("<ii", rows, dim))
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                np.asarray(values[start:stop], dtype=output_dtype, order="C").tofile(handle)
                if stop == rows or stop % 100_000 < chunk_rows:
                    print(f"  rows={stop}/{rows}", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if not valid_matrix(output, rows, dim, output_dtype):
        raise RuntimeError(f"invalid converted matrix: {output}")
    return rows, dim


def valid_aligned(path: Path, rows: int, dim: int) -> bool:
    # The native ADC-BQ and RaBitQ readers use a 4 KiB-aligned payload start,
    # followed by contiguous FP32 rows.  Individual O_DIRECT reads are aligned
    # outward at fetch time; rows themselves must not be padded.
    if not path.is_file() or path.stat().st_size != PAGE_BYTES + rows * dim * 4:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == struct.pack("<ii", rows, dim)


def write_aligned_sidecar(
    source: Path, output: Path, *, chunk_mib: int = 256
) -> tuple[int, int]:
    values = np.load(source, mmap_mode="r")
    rows, dim = map(int, values.shape)
    if valid_aligned(output, rows, dim):
        print(f"REUSE {output}", flush=True)
        return rows, dim
    row_bytes = dim * 4
    chunk_rows = max(1, chunk_mib * 1024**2 // row_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    print(
        f"ALIGN {source} -> {output} shape=({rows},{dim}) "
        f"payload_offset={PAGE_BYTES} contiguous_row_bytes={row_bytes}",
        flush=True,
    )
    try:
        with temporary.open("wb", buffering=0) as handle:
            header = bytearray(PAGE_BYTES)
            header[:8] = struct.pack("<ii", rows, dim)
            handle.write(header)
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                block = np.asarray(values[start:stop], dtype=np.float32, order="C")
                block.tofile(handle)
                if stop == rows or stop % 100_000 < chunk_rows:
                    print(f"  aligned_rows={stop}/{rows}", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if not valid_aligned(output, rows, dim):
        raise RuntimeError(f"invalid aligned sidecar: {output}")
    return rows, dim


def valid_gt(path: Path, rows: int, width: int) -> bool:
    if not path.is_file() or path.stat().st_size != 8 + rows * width * 4:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == struct.pack("<ii", rows, width)


def exact_gt(
    corpus_npy: Path,
    query_npy: Path,
    output: Path,
    *,
    k: int,
    threads: int,
    add_batch: int,
    query_batch: int,
) -> None:
    corpus = np.load(corpus_npy, mmap_mode="r")
    queries = np.load(query_npy, mmap_mode="r")
    if corpus.ndim != 2 or queries.ndim != 2 or corpus.shape[1] != queries.shape[1]:
        raise ValueError(f"shape mismatch: corpus={corpus.shape}, queries={queries.shape}")
    nq, dim = map(int, queries.shape)
    if valid_gt(output, nq, k):
        print(f"REUSE {output}", flush=True)
        return
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(dim)
    print(
        f"EXACT_GT corpus={corpus.shape} queries={queries.shape} k={k} threads={threads}",
        flush=True,
    )
    for start in range(0, len(corpus), add_batch):
        stop = min(len(corpus), start + add_batch)
        index.add(np.asarray(corpus[start:stop], dtype=np.float32, order="C"))
        if stop == len(corpus) or stop % 100_000 < add_batch:
            print(f"  indexed={stop}/{len(corpus)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb", buffering=0) as handle:
            handle.write(struct.pack("<ii", nq, k))
            for start in range(0, nq, query_batch):
                stop = min(nq, start + query_batch)
                _, ids = index.search(
                    np.asarray(queries[start:stop], dtype=np.float32, order="C"), k
                )
                ids.astype(np.int32, copy=False).tofile(handle)
                print(f"  searched={stop}/{nq}", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if not valid_gt(output, nq, k):
        raise RuntimeError(f"invalid exact ground truth: {output}")
    del index
    gc.collect()


def spec_for(corpus: str, model_size: str, output_root: Path) -> dict[str, object]:
    dim, model_name = MODELS[model_size]
    key = key_for(corpus, dim)
    data_root = output_root / "data" / key
    index_root = output_root / "indexes" / key
    return {
        "key": key,
        "corpus": "DBpedia" if corpus == "dbpedia-entity" else corpus.upper(),
        "model": model_name,
        "dim": dim,
        "calibration_rows": CALIBRATION_ROWS[corpus],
        "source_train": str(SOURCE_ROOT / corpus / f"corpus_text-embedding-3-{model_size}.npy"),
        "source_query": str(SOURCE_ROOT / corpus / f"queries_text-embedding-3-{model_size}.npy"),
        "train": str(data_root / f"{key}.train.fbin"),
        "query": str(data_root / f"{key}.query.fbin"),
        "gt": str(data_root / f"{key}.gt100.ibin"),
        "fp_sidecar": str(data_root / f"{key}.train.row_aligned4k.fbin"),
        "adcbq_index": str(index_root / f"{key}.adcbq.M24.efC200.faiss"),
        "b1_index": str(index_root / f"{key}.rabitq_b1_c16.M24.efC200.index"),
        "b7_index": str(index_root / f"{key}.rabitq_b7_c16.M24.efC200.index"),
        "owned_indexes": True,
    }


def bge_large_beir400_spec(output_root: Path) -> dict[str, object]:
    key = BGE_LARGE_KEY
    data_root = output_root / "data" / key
    index_root = output_root / "indexes" / key
    return {
        "key": key,
        "corpus": "DBpedia",
        "model": "BGE-large-en-v1.5 (BEIR-400)",
        "dim": 1024,
        # Match the existing main-matrix DBpedia protocol: 200 calibration
        # queries and a disjoint 200-query held-out measurement split.
        "calibration_rows": 200,
        "source_train": str(BGE_ROOT / "dbpedia-entity__vanilla-bgel_docs.npy"),
        "source_query": str(BGE_ROOT / "dbpedia-entity__vanilla-bgel_queries.npy"),
        "train": str(data_root / f"{key}.train.fbin"),
        "query": str(data_root / f"{key}.query.fbin"),
        "gt": str(data_root / f"{key}.gt100.ibin"),
        "fp_sidecar": str(data_root / f"{key}.train.row_aligned4k.fbin"),
        "adcbq_index": str(index_root / f"{key}.adcbq.M24.efC200.faiss"),
        "b1_index": str(index_root / f"{key}.rabitq_b1_c16.M24.efC200.index"),
        "b7_index": str(index_root / f"{key}.rabitq_b7_c16.M24.efC200.index"),
        "owned_indexes": True,
    }


def publish_manifest(path: Path, specs: list[dict[str, object]]) -> None:
    payload = {
        "format": "diskveclab-extra-datasets-v1",
        "created_at": now(),
        "metric": "inner_product_on_l2_normalized_vectors",
        "ground_truth_k": 100,
        "datasets": specs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        choices=(1536, 3072),
        default=[1536],
        help="OpenAI dimensions to queue (3072 is supported but deferred by default)",
    )
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--add-batch", type=int, default=8192)
    parser.add_argument("--query-batch", type=int, default=64)
    parser.add_argument(
        "--skip-bge-large-beir400",
        action="store_true",
        help="prepare only the original eight OpenAI datasets",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.gt_k, args.add_batch, args.query_batch) < 1:
        parser.error("numeric arguments must be positive")
    dims = set(args.dims)
    specs = [
        spec_for(corpus, size, args.output_root)
        for corpus in args.corpora
        for size, (dim, _) in MODELS.items()
        if dim in dims
    ]
    if not args.skip_bge_large_beir400:
        specs.append(bge_large_beir400_spec(args.output_root))
    if args.dry_run:
        print(json.dumps({"datasets": specs}, indent=2))
        return 0
    status_path = args.output_root / "prepare_status.json"
    for position, spec in enumerate(specs, start=1):
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(
                {
                    "stage": "running",
                    "updated_at": now(),
                    "position": position,
                    "total": len(specs),
                    "dataset": spec["key"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        train_rows, train_dim = convert_npy(
            Path(str(spec["source_train"])), Path(str(spec["train"]))
        )
        query_rows, query_dim = convert_npy(
            Path(str(spec["source_query"])), Path(str(spec["query"]))
        )
        aligned_rows, aligned_dim = write_aligned_sidecar(
            Path(str(spec["source_train"])), Path(str(spec["fp_sidecar"]))
        )
        if train_dim != int(spec["dim"]) or query_dim != train_dim:
            raise RuntimeError(f"dimension mismatch for {spec['key']}")
        if (aligned_rows, aligned_dim) != (train_rows, train_dim):
            raise RuntimeError(f"aligned sidecar mismatch for {spec['key']}")
        if query_rows <= int(spec["calibration_rows"]):
            raise RuntimeError(f"not enough queries for split: {spec['key']}")
        exact_gt(
            Path(str(spec["source_train"])),
            Path(str(spec["source_query"])),
            Path(str(spec["gt"])),
            k=args.gt_k,
            threads=args.threads,
            add_batch=args.add_batch,
            query_batch=args.query_batch,
        )
        spec["nb"] = train_rows
        spec["nq"] = query_rows
    manifest = args.output_root / "dataset_manifest.json"
    publish_manifest(manifest, specs)
    status_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "updated_at": now(),
                "datasets": [spec["key"] for spec in specs],
                "manifest": str(manifest),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE manifest={manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
