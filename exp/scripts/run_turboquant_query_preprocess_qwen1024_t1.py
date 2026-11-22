#!/usr/bin/env python3
"""Remeasure only TurboQuant query setup and merge it into the T=1 inputs."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import run_faiss_hnsw_bq_budget_paged_codes as datasets
import run_query_preprocess_overhead_qwen1024 as benchmark


REPO = Path(__file__).resolve().parents[1]
MSM_DIR = (
    REPO
    / "experiment/query_preprocess_overhead_qwen1024_t1"
    / "clean_20261110_cpu20_sq_opq_tq"
)
NQ_DIR = REPO / "experiment/query_preprocess_overhead_qwen1024_t1/nq_cpu20"
TQ_METHOD = "TurboQuant TQ+ 1b"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_tq_only(
    dataset: str,
    query: Path,
    output_dir: Path,
    mrq_key: str,
) -> Path:
    mrq = benchmark.MRQ_ROOT / mrq_key
    output = output_dir / f"{dataset}.tq_only.raw.csv"
    command = [
        str(benchmark.BINARY),
        "--dataset", dataset,
        "--query", str(query),
        "--mrq-pca", str(mrq / f"{mrq_key}_pca.fvecs"),
        "--mrq-mean", str(mrq / f"{mrq_key}_mean.fvecs"),
        "--mrq-rotation", str(mrq / "RESP_C4096_B320.fvecs"),
        "--mrq-centroids", str(mrq / "RESCentroid_C4096_B320.fvecs"),
        "--opq-index", str(benchmark.OPQ_INDEX),
        "--tq-data", str(output_dir / f"{dataset}.tqplus1.tqdata"),
        "--tq-meta", str(output_dir / f"{dataset}.tqplus1.meta.json"),
        "--output", str(output),
        "--cpu", "20",
        "--rounds", "9",
        "--fixed-iterations", "6000",
        "--mrq-iterations", "1",
        "--tq-only",
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
    return output


def replace_tq_rows(path: Path, tq_path: Path) -> list[dict[str, str]]:
    existing = [row for row in read_csv(path) if row["method"] != TQ_METHOD]
    updated = existing + read_csv(tq_path)
    write_csv(path, updated)
    return updated


def update_msm_derived(tq_path: Path) -> None:
    derived_path = MSM_DIR / "query_preprocess_overhead_derived.csv"
    rows = read_csv(derived_path)
    total = next(
        row for row in read_csv(tq_path)
        if row["method"] == TQ_METHOD and row["stage"] == "method_total"
    )
    latency = float(total["median_us_per_query"])
    matched = False
    for row in rows:
        if (
            row["dataset"] == "msmarco_qwen1024"
            and row["method"] == TQ_METHOD
            and row["component"] == "method_total"
        ):
            row["median_us_per_query"] = str(latency)
            row["preprocess_only_qps_ceiling"] = str(1e6 / latency)
            matched = True
    if not matched:
        raise RuntimeError("missing MSMARCO TurboQuant derived row")
    write_csv(derived_path, rows)


def main() -> None:
    benchmark.compile_binary()

    msm_tq = run_tq_only(
        "msmarco_qwen1024",
        benchmark.WORKLOADS[0].query,
        MSM_DIR,
        "msmarco_qwen1024",
    )
    msm_raw = replace_tq_rows(MSM_DIR / "msmarco_qwen1024.raw.csv", msm_tq)
    dbpedia_raw = read_csv(MSM_DIR / "dbpedia_qwen1024.raw.csv")
    write_csv(MSM_DIR / "query_preprocess_overhead_raw.csv", msm_raw + dbpedia_raw)
    update_msm_derived(msm_tq)

    nq_tq = run_tq_only(
        "nq_qwen1024",
        datasets.DATASETS["nq_qwen1024"].query,
        NQ_DIR,
        "msmarco_qwen1024",
    )
    nq_raw = replace_tq_rows(NQ_DIR / "nq_qwen1024.raw.csv", nq_tq)
    nq_totals = [row for row in nq_raw if row["stage"] == "method_total"]
    write_csv(NQ_DIR / "nq_qwen1024.method_totals.csv", nq_totals)

    for dataset, path in (("msmarco_qwen1024", msm_tq), ("nq_qwen1024", nq_tq)):
        total = next(
            row for row in read_csv(path)
            if row["method"] == TQ_METHOD and row["stage"] == "method_total"
        )
        print(
            f"{dataset}: TurboQuant production set_query "
            f"{float(total['median_us_per_query']):.3f} us/query",
            flush=True,
        )


if __name__ == "__main__":
    main()
