#!/usr/bin/env python3
"""Run the remaining DBpedia/MS MARCO calibrated ADC-BQ matrix.

MS MARCO Jina variants are intentionally excluded.  A single tau is fitted at
efSearch=256 and reused at efSearch 64/128/256/512/1024.  Each search point is
measured at T=1 and T=24 for three rounds.  Existing per-dataset results are
reused, so this script is restartable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import run_adcbq_calibrated_threshold_overlap as calibration
import run_v051_ssd_odirect_multidataset as base


TARGETS = (
    ("dbpedia_qwen2560", 100),
    ("dbpedia_qwen4096", 100),
    ("dbpedia_bge768", 200),
    ("dbpedia_bge1024", 200),
    ("dbpedia_e5mistral4096", 100),
    ("dbpedia_jina768", 100),
    ("msmarco_qwen2560", 1000),
    ("msmarco_qwen4096", 1000),
    ("msmarco_bge768", 1000),
    ("msmarco_bge1024", 1000),
    ("msmarco_e5mistral4096", 1000),
    ("msmarco_openai1536", 1000),
)

EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
THREADS = (1, 24)
REFERENCE_EF = 256
FIXED_R = 80
PAGE_BYTES = 4096.0
RESULT_ROOT = calibration.RESULT_DIR


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def upsert_standard_summary(
    dataset_key: str, threads: int, new_rows: list[dict[str, object]]
) -> None:
    path = RESULT_ROOT / f"comparison_single_tau_ef{REFERENCE_EF}_t{threads}.csv"
    existing: list[dict[str, object]] = [
        dict(row) for row in read_csv(path) if row["dataset"] != dataset_key
    ]
    existing.extend(new_rows)
    existing.sort(key=lambda row: (str(row["dataset"]), int(row["efSearch"])))
    write_csv(path, existing)
    print(f"UPDATED {path} rows={len(existing)}", flush=True)


def wait_for_process(pid: int, poll_seconds: int, quiet_seconds: int) -> None:
    if pid <= 0:
        return
    proc = Path(f"/proc/{pid}")
    while proc.exists():
        print(
            f"WAIT pid={pid} is still running; next check in {poll_seconds}s",
            flush=True,
        )
        time.sleep(poll_seconds)
    if quiet_seconds:
        remaining = quiet_seconds
        while remaining > 0:
            interval = min(poll_seconds, remaining, 60)
            print(
                f"WAIT pid={pid} finished; quiet period remaining={remaining}s",
                flush=True,
            )
            time.sleep(interval)
            remaining -= interval


def fit_or_load_tau(
    dataset: base.Dataset,
    split: calibration.SplitFiles,
    force: bool,
    build_threads: int,
) -> float:
    path = RESULT_ROOT / dataset.key / (
        f"calibration.single_tau_ef{REFERENCE_EF}.json"
    )
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fitted = payload["calibration"]
        if int(payload["reference_ef"]) != REFERENCE_EF:
            raise RuntimeError(f"reference ef mismatch in {path}")
        if int(fitted["calibration_rows"]) != split.calibration_rows:
            raise RuntimeError(f"calibration split mismatch in {path}")
        print(f"REUSE tau={float(fitted['tau']):.9g} from {path}", flush=True)
        return float(fitted["tau"])

    fit_args = SimpleNamespace(
        force=force,
        search_threads=1,
        build_threads=build_threads,
    )
    candidates = calibration.calibration_candidates(
        dataset, split, REFERENCE_EF, fit_args
    )
    fitted = calibration.fit_tau(candidates, split, REFERENCE_EF, FIXED_R)
    payload = {"reference_ef": REFERENCE_EF, "calibration": fitted}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    candidates.unlink(missing_ok=True)
    print(
        f"FITTED {dataset.key} tau={float(fitted['tau']):.9g} "
        f"Rmean={float(fitted['calibrated_count_mean']):.2f}",
        flush=True,
    )
    return float(fitted["tau"])


def benchmark_dataset(
    dataset_key: str,
    calibration_queries: int,
    rounds: int,
    build_threads: int,
    force: bool,
) -> None:
    dataset = base.DATASETS[dataset_key]
    base.validate(dataset)
    split = calibration.make_split(dataset, calibration_queries, 11, 0)
    print(
        f"DATASET {dataset.key}: cal={split.calibration_rows} "
        f"test={split.test_rows}",
        flush=True,
    )
    tau = fit_or_load_tau(dataset, split, force, build_threads)
    sidecar = calibration.ALIGNED_FP_ROOT / f"{dataset.key}.fp32.aligned4k.fbin"
    sidecar_existed = sidecar.exists()
    try:
        for threads in THREADS:
            result_rows: list[dict[str, object]] = []
            bench_args = SimpleNamespace(
                force=force,
                fixed_r=FIXED_R,
                rounds=rounds,
                search_threads=threads,
                build_threads=build_threads,
            )
            tag = f"single_tau_ef{REFERENCE_EF}.t{threads}"
            for ef in EFS:
                result_path = calibration.benchmark_policy(
                    dataset,
                    split,
                    ef,
                    "calibrated",
                    tau,
                    sidecar,
                    bench_args,
                    f".{tag}",
                )
                result = read_one(result_path)
                submitted_bytes_per_query = (
                    float(result["direct_submitted_bytes_mean"]) / split.test_rows
                )
                row: dict[str, object] = {
                    "dataset": dataset.key,
                    "corpus": dataset.corpus,
                    "model": dataset.model,
                    "efSearch": ef,
                    "search_threads": threads,
                    "rounds": int(result.get("rounds", rounds)),
                    "calibration_queries": split.calibration_rows,
                    "test_queries": split.test_rows,
                    "calibration_reference_ef": REFERENCE_EF,
                    "tau": tau,
                    "recall_at_10": float(result["recall_at_k"]),
                    "qps": float(result["qps"]),
                    "pages_4k_mean": submitted_bytes_per_query / PAGE_BYTES,
                    "graph_4k_pages_per_query_mean": 0.0,
                    "fp_vectors_per_query_mean": float(result["adaptive_rerank_count_mean"]),
                    "fp_4k_pages_per_query_mean": submitted_bytes_per_query / PAGE_BYTES,
                    "total_4k_pages_per_query_mean": submitted_bytes_per_query / PAGE_BYTES,
                    # These legacy columns store selected-vector count
                    # percentiles, matching the existing summary schema.
                    "pages_4k_p95": float(result["adaptive_rerank_count_p95"]),
                    "pages_4k_p99": float(result["adaptive_rerank_count_p99"]),
                    "pages_4k_max": float(result["adaptive_rerank_count_max"]),
                    "io_engine": "io_uring O_DIRECT inter-query overlap",
                    "storage": "RAM ADC-BQ HNSW + SSD FP32 sidecar",
                    "result_path": str(result_path),
                }
                result_rows.append(row)
                print(
                    f"RESULT {dataset.key} T={threads} ef={ef} "
                    f"recall={row['recall_at_10']:.6f} "
                    f"qps={row['qps']:.2f} pages={row['pages_4k_mean']:.2f}",
                    flush=True,
                )
            upsert_standard_summary(dataset.key, threads, result_rows)
    finally:
        if not sidecar_existed and sidecar.exists():
            size = sidecar.stat().st_size
            sidecar.unlink()
            print(f"REMOVED transient sidecar bytes={size}: {sidecar}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--wait-for-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--quiet-seconds", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0 or args.build_threads <= 0:
        parser.error("rounds and build-threads must be positive")
    if args.poll_seconds <= 0 or args.quiet_seconds < 0:
        parser.error("invalid wait interval")

    wait_for_process(args.wait_for_pid, args.poll_seconds, args.quiet_seconds)
    calibration.ensure_binary()
    calibration.ALIGNED_FP_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        "START remaining DBpedia/MS MARCO calibration matrix; "
        "MS MARCO Jina excluded",
        flush=True,
    )
    for dataset_key, calibration_queries in TARGETS:
        benchmark_dataset(
            dataset_key,
            calibration_queries,
            args.rounds,
            args.build_threads,
            args.force,
        )
    print("ALL REMAINING CALIBRATION EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
