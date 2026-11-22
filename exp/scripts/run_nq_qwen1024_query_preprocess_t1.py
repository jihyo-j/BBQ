#!/usr/bin/env python3
"""Measure method-specific query setup on the NQ Qwen-1024 queries."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import run_query_preprocess_overhead_qwen1024 as benchmark
import run_faiss_hnsw_bq_budget_paged_codes as datasets


REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "experiment/query_preprocess_overhead_qwen1024_t1/nq_cpu20"
RAW = OUTPUT_DIR / "nq_qwen1024.raw.csv"
SUMMARY = OUTPUT_DIR / "nq_qwen1024.method_totals.csv"


def main() -> None:
    benchmark.compile_binary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mrq = benchmark.MRQ_ROOT / "msmarco_qwen1024"
    command = [
        str(benchmark.BINARY),
        "--dataset", "nq_qwen1024",
        "--query", str(datasets.DATASETS["nq_qwen1024"].query),
        "--mrq-pca", str(mrq / "msmarco_qwen1024_pca.fvecs"),
        "--mrq-mean", str(mrq / "msmarco_qwen1024_mean.fvecs"),
        "--mrq-rotation", str(mrq / "RESP_C4096_B320.fvecs"),
        "--mrq-centroids", str(mrq / "RESCentroid_C4096_B320.fvecs"),
        "--opq-index", str(benchmark.OPQ_INDEX),
        "--tq-data", str(OUTPUT_DIR / "nq_qwen1024.tqplus1.tqdata"),
        "--tq-meta", str(OUTPUT_DIR / "nq_qwen1024.tqplus1.meta.json"),
        "--output", str(RAW),
        "--cpu", "20",
        "--rounds", "9",
        "--fixed-iterations", "6000",
        "--mrq-iterations", "1",
    ]
    subprocess.run(
        command,
        cwd=REPO,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        },
        check=True,
    )
    with RAW.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["stage"] == "method_total"
        ]
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
