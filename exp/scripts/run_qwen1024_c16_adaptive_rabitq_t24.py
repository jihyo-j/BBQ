#!/usr/bin/env python3
"""Run the official c16 RaBitQ adaptive-FP baseline on the matched query split."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BINDING = REPO / "experiment/rabitq_v051_adaptive_odirect_aligned4k_build/rabitqlib"
sys.path.insert(0, str(BINDING))
from _rabitqlib import HnswIndex  # type: ignore  # noqa: E402

import run_v051_ssd_odirect_multidataset as base  # noqa: E402


PAGE_BYTES = 4096
EXTERNAL = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq")
RESULT_ROOT = (
    REPO / "experiment/faiss_hnsw_bq_budget_paged_rerank80_results/adaptive_rabitq"
)


@dataclass(frozen=True)
class Dataset:
    key: str
    source_key: str
    index: Path
    fp: Path


DATASETS = {
    "msmarco_qwen1024": Dataset(
        key="msmarco_qwen1024",
        source_key="msmarco_qwen1024",
        index=EXTERNAL
        / "indexes/msmarco_qwen1024.rabitq_b1_c16.n1000000.d1024.M24.efC200.index",
        fp=EXTERNAL / "aligned_fp/msmarco_qwen1024.fp32.header4096.fbin",
    ),
    "nq_qwen1024": Dataset(
        key="nq_qwen1024",
        source_key="nq_qwen06_1024",
        index=EXTERNAL
        / "indexes/nq_qwen06_1024.rabitq_b1_c16.n1000000.d1024.M24.efC200.index",
        fp=EXTERNAL / "aligned_fp/nq_qwen06_1024.fp32.header4096.fbin",
    ),
}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run(dataset: Dataset, args: argparse.Namespace) -> Path:
    source = base.DATASETS[dataset.source_key]
    for path in (source.query, source.gt, dataset.index, dataset.fp):
        if not path.exists():
            raise FileNotFoundError(path)
    queries = base.read_fbin(source.query)[: args.max_queries].copy()
    base.normalize_rows(queries)
    gt = base.read_ibin(source.gt, len(queries))
    index = HnswIndex.load(str(dataset.index))
    if index.nbits != 1 or index.num_clusters != 16 or index.dim != 1024:
        raise RuntimeError("unexpected RaBitQ c16 index metadata")

    rows: list[dict[str, object]] = []
    for ef in args.efs:
        recalls: list[float] = []
        elapsed_values: list[float] = []
        fp_values: list[np.ndarray] = []
        bytes_values: list[int] = []
        request_values: list[int] = []
        for round_id in range(1, args.rounds + 1):
            started = time.perf_counter()
            ids, _, stats = index.search_1bit_adaptive_fp_odirect(
                queries,
                str(dataset.fp),
                k=10,
                ef=ef,
                num_threads=args.threads,
            )
            elapsed = time.perf_counter() - started
            if int(stats["fbin_data_offset"]) != PAGE_BYTES:
                raise RuntimeError("adaptive RaBitQ did not use aligned FP offset")
            fp_reads = np.asarray(stats["fp_read_count"], dtype=np.uint64)
            recall = base.recall_at_10(np.asarray(ids), gt)
            recalls.append(recall)
            elapsed_values.append(elapsed)
            fp_values.append(fp_reads)
            bytes_values.append(int(stats["odirect_read_bytes"]))
            request_values.append(int(stats["odirect_read_requests"]))
            print(
                f"{dataset.key} ef={ef} round={round_id} recall={recall:.6f} "
                f"qps={len(queries) / elapsed:.2f} fp/q={np.mean(fp_reads):.2f}",
                flush=True,
            )
        samples = np.concatenate(fp_values).astype(np.float64)
        elapsed = statistics.fmean(elapsed_values)
        mean_bytes = statistics.fmean(bytes_values)
        rows.append(
            {
                "dataset": dataset.key,
                "method": "RaBitQ-1b-c16-adaptive-FP",
                "metric": "ip",
                "k": 10,
                "nq": len(queries),
                "nb": index.max_elements,
                "dim": index.dim,
                "M": 24,
                "efConstruction": 200,
                "efSearch": ef,
                "rounds": args.rounds,
                "threads": args.threads,
                "recall_at_10": statistics.fmean(recalls),
                "qps": len(queries) / elapsed,
                "total_seconds": elapsed,
                "rerank_policy": "adaptive_lower_bound",
                "rerank_n": "adaptive",
                "fp_page_reads_per_query_mean": float(np.mean(samples)),
                "fp_page_reads_per_query_p50": percentile(samples, 50),
                "fp_page_reads_per_query_p95": percentile(samples, 95),
                "fp_page_reads_per_query_p99": percentile(samples, 99),
                "fp_page_reads_per_query_max": int(np.max(samples)),
                "fp_read_bytes_per_query": mean_bytes / len(queries),
                "fp_direct_requests_per_query": statistics.fmean(request_values)
                / len(queries),
                "io_engine": "sync pread O_DIRECT",
                "fp_store_layout": "header4096_rows_contiguous",
                "num_clusters": 16,
                "code_paging": "fully resident official RaBitQ index",
                "index_path": dataset.index,
                "fp_sidecar_path": dataset.fp,
            }
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / (
        f"{dataset.key}.rabitq1_c16_adaptive_t{args.threads}.csv"
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def combine(paths: list[Path], threads: int = 24) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    output = RESULT_ROOT / (
        f"qwen1024_rabitq1_c16_adaptive_t{threads}_combined.csv"
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--efs",
        nargs="+",
        type=int,
        default=[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
    )
    args = parser.parse_args()
    paths = [run(DATASETS[key], args) for key in args.datasets]
    combine(paths, args.threads)


if __name__ == "__main__":
    main()
