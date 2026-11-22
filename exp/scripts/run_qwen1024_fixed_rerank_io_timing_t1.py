#!/usr/bin/env python3
"""Rerun the Recall@10>=0.99 ADC-BQ/TQ1+ points with I/O timing.

Compressed codes and HNSW stay resident.  Each query's fixed exact-FP set is
submitted as one io_uring batch.  ``rerank_io_wait_seconds`` accumulates only
the io_uring submit/wait syscall intervals and excludes exact dot products.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import run_faiss_hnsw_bq_budget_paged_codes as base
import run_qwen1024_all_quantizers_resident_rerank_pareto as resident


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/qwen1024_fixed_rerank_io_timing_t1"
CONFIGS = (
    ("msmarco_qwen1024", "adcbq1", 120, 384),
    ("msmarco_qwen1024", "qdranttq1", 80, 512),
    ("nq_qwen1024", "adcbq1", 80, 128),
    ("nq_qwen1024", "qdranttq1", 60, 128),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("adcbq1", "qdranttq1"),
        default=("adcbq1", "qdranttq1"),
        help="Rerun only the selected methods and preserve other existing summary rows.",
    )
    args = parser.parse_args()
    base.compile_binary()

    summaries: list[dict[str, object]] = []
    for dataset, method, rerank_n, ef in CONFIGS:
        if method not in args.methods:
            continue
        run_args = argparse.Namespace(
            result_root=RESULT_ROOT,
            fp_rerank_qd=0,
            fp_io_mode="uring-batch",
            force=True,
            efs=[ef],
            rounds=args.rounds,
            max_queries=args.max_queries,
            threads=1,
            query_e2e=True,
        )
        path = resident.run_fixed_one(dataset, method, rerank_n, run_args)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != args.rounds:
            raise RuntimeError(f"unexpected row count in {path}")
        numeric = (
            "recall_at_10",
            "qps",
            "search_seconds",
            "rerank_seconds",
            "rerank_io_wait_seconds",
            "total_seconds",
            "fp_page_reads_per_query",
        )
        summary: dict[str, object] = {
            "dataset": dataset,
            "method_key": method,
            "method": rows[0]["method"],
            "efSearch": ef,
            "rerank_n": rerank_n,
            "rounds": args.rounds,
            "nq": args.max_queries,
            "threads": 1,
            "io_engine": "per-query io_uring batch O_DIRECT",
            "source_csv": path,
        }
        for field in numeric:
            summary[field] = statistics.fmean(float(row[field]) for row in rows)
        summary["query_preprocess_us"] = 0.0
        summary["exact_fp_io_wait_us_per_query"] = (
            float(summary["rerank_io_wait_seconds"]) * 1e6 / args.max_queries
        )
        summary["qps_stddev"] = statistics.stdev(float(row["qps"]) for row in rows)
        summaries.append(summary)
        print(
            f"{dataset} {method} recall={summary['recall_at_10']:.4f} "
            f"qps={summary['qps']:.2f} io_wait={summary['exact_fp_io_wait_us_per_query']:.2f} us/q",
            flush=True,
        )
    summary_path = RESULT_ROOT / "summary.csv"
    if summary_path.is_file() and set(args.methods) != {"adcbq1", "qdranttq1"}:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        replaced = {
            (str(row["dataset"]), str(row["method_key"])) for row in summaries
        }
        summaries = [
            row for row in existing
            if (row["dataset"], row["method_key"]) not in replaced
        ] + summaries
        order = {
            (dataset, method): index
            for index, (dataset, method, _, _) in enumerate(CONFIGS)
        }
        summaries.sort(key=lambda row: order[(str(row["dataset"]), str(row["method_key"]))])
    write_csv(summary_path, summaries)


if __name__ == "__main__":
    main()
