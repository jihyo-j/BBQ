#!/usr/bin/env python3
"""Measure pure synchronous O_DIRECT wait for RaBitQ c16 adaptive FP reads.

The timed binding accumulates only the wall time spent inside each ``pread``
syscall.  Query rotation, 1-bit traversal, FP normalization, and exact scoring
remain part of end-to-end latency but are excluded from ``io_wait_us``.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BINDING = (
    REPO
    / "experiment/rabitq_v051_adaptive_odirect_io_timing_build/rabitqlib"
)
sys.path.insert(0, str(BINDING))
from _rabitqlib import HnswIndex  # type: ignore  # noqa: E402

import run_qwen1024_c16_adaptive_rabitq_t24 as baseline  # noqa: E402
import run_v051_ssd_odirect_multidataset as data_io  # noqa: E402


RESULT_ROOT = REPO / "experiment/qwen1024_rabitq_adaptive_io_timing_t1"
TARGET_EF = {"msmarco_qwen1024": 768, "nq_qwen1024": 512}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(key: str, rounds: int, max_queries: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset = baseline.DATASETS[key]
    source = data_io.DATASETS[dataset.source_key]
    for path in (source.query, source.gt, dataset.index, dataset.fp):
        if not path.is_file():
            raise FileNotFoundError(path)

    queries = data_io.read_fbin(source.query)[:max_queries].copy()
    data_io.normalize_rows(queries)
    gt = data_io.read_ibin(source.gt, len(queries))
    index = HnswIndex.load(str(dataset.index))
    ef = TARGET_EF[key]

    rows: list[dict[str, object]] = []
    for round_id in range(1, rounds + 1):
        started = time.perf_counter()
        ids, _, stats = index.search_1bit_adaptive_fp_odirect(
            queries, str(dataset.fp), k=10, ef=ef, num_threads=1
        )
        elapsed = time.perf_counter() - started
        if int(stats["fbin_data_offset"]) != 4096:
            raise RuntimeError("expected aligned 4 KiB FP payload")
        fp_reads = np.asarray(stats["fp_read_count"], dtype=np.uint64)
        query_latency_ns = np.asarray(stats["query_latency_ns"], dtype=np.uint64)
        requests = int(stats["odirect_read_requests"])
        if requests != int(np.sum(fp_reads)):
            raise RuntimeError("O_DIRECT request count differs from adaptive FP reads")
        row = {
            "dataset": key,
            "method": "RaBitQ-1b-c16 + adaptive FP",
            "efSearch": ef,
            "round": round_id,
            "threads": 1,
            "nq": len(queries),
            "recall_at_10": data_io.recall_at_10(np.asarray(ids), gt),
            "qps": len(queries) / elapsed,
            "end_to_end_us_per_query": elapsed * 1e6 / len(queries),
            "mean_internal_query_latency_us": float(np.mean(query_latency_ns)) / 1000.0,
            "io_wait_us_per_query": int(stats["odirect_read_ns"]) / len(queries) / 1000.0,
            "adaptive_fp_reads_per_query": requests / len(queries),
            "odirect_bytes_per_query": int(stats["odirect_read_bytes"]) / len(queries),
            "io_engine": "synchronous O_DIRECT pread interleaved in HNSW search",
        }
        rows.append(row)
        print(
            f"{key} round={round_id} ef={ef} recall={row['recall_at_10']:.4f} "
            f"qps={row['qps']:.2f} reads/q={row['adaptive_fp_reads_per_query']:.2f} "
            f"pread_wait={row['io_wait_us_per_query']:.2f} us/q",
            flush=True,
        )

    numeric = (
        "recall_at_10",
        "qps",
        "end_to_end_us_per_query",
        "mean_internal_query_latency_us",
        "io_wait_us_per_query",
        "adaptive_fp_reads_per_query",
        "odirect_bytes_per_query",
    )
    summary = dict(rows[0])
    summary.pop("round")
    summary["rounds"] = rounds
    for field in numeric:
        summary[field] = statistics.fmean(float(row[field]) for row in rows)
    summary["qps_stddev"] = statistics.stdev(float(row["qps"]) for row in rows)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=TARGET_EF, default=list(TARGET_EF))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=1000)
    args = parser.parse_args()
    all_rounds: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for key in args.datasets:
        rows, summary = run_one(key, args.rounds, args.max_queries)
        all_rounds.extend(rows)
        summaries.append(summary)
    write_csv(RESULT_ROOT / "all_rounds.csv", all_rounds)
    write_csv(RESULT_ROOT / "summary.csv", summaries)


if __name__ == "__main__":
    main()
