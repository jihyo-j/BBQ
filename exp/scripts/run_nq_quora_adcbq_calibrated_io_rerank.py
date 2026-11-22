#!/usr/bin/env python3
"""Measure calibrated ADC-BQ on the NQ/Quora BGE and Qwen panel.

The calibration protocol matches ``run_adcbq_calibrated_threshold_overlap``:
a single threshold tau is fitted at efSearch=256 to match fixed FP@80 recall
on 1,000 calibration queries, then reused at efSearch 64/128/256/512/1024.
Search is measured for three rounds at T=1 and T=24.

In addition to recall and QPS, the output explicitly records the selected
full-precision rerank-vector count and actual O_DIRECT submitted 4 KiB page
volume per query.  The 4 KiB-aligned-header FP store keeps rows contiguous;
for this panel each row submission occupies ceil(dim * 4 / 4096) pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import run_adcbq_calibrated_threshold_overlap as calibration
import run_v051_ssd_odirect_multidataset as base


RESULT_ROOT = Path(__file__).resolve().parent / "nq_quora_adcbq_calibrated_io_rerank"
TARGETS = (
    "nq_bge_m3_vanilla",
    "nq_qwen06_1024",
    "nq_qwen4_2560",
    "nq_qwen8_4096",
    "quora_bge_m3_vanilla",
    "quora_qwen1024",
    "quora_qwen2560",
    "quora_qwen4096",
)
EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
THREADS = (1, 24)
CALIBRATION_QUERIES = 1000
REFERENCE_EF = 256
FIXED_R = 80
PAGE_BYTES = 4096

FIELDS = (
    "dataset",
    "corpus",
    "model",
    "dim",
    "nb",
    "efSearch",
    "search_threads",
    "rounds",
    "calibration_queries",
    "test_queries",
    "calibration_reference_ef",
    "target_fixed_rerank",
    "tau",
    "recall_at_10",
    "qps",
    "rerank_vectors_per_query_mean",
    "rerank_vectors_per_query_min",
    "rerank_vectors_per_query_p95",
    "rerank_vectors_per_query_p99",
    "rerank_vectors_per_query_max",
    "fp_io_requests_per_query_mean",
    "fp_submitted_4k_pages_per_query_mean",
    "fp_submitted_4k_pages_per_query_p95",
    "fp_submitted_4k_pages_per_query_p99",
    "fp_submitted_4k_pages_per_query_max",
    "fp_submitted_bytes_per_query_mean",
    "fp_logical_4k_pages_per_query_mean",
    "fp_logical_bytes_per_query_mean",
    "vector_payload_bytes",
    "submitted_4k_pages_per_reranked_vector",
    "page_bytes",
    "storage",
    "io_engine",
    "query_scope",
    "raw_result_csv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    dataset_order = {key: index for index, key in enumerate(TARGETS)}
    return sorted(
        rows,
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            int(row["search_threads"]),
            int(row["efSearch"]),
        ),
    )


def save_progress(rows: list[dict[str, object]]) -> None:
    rows = sort_rows(rows)
    write_csv(RESULT_ROOT / "all_results.csv", rows)
    for threads in THREADS:
        selected = [row for row in rows if int(row["search_threads"]) == threads]
        write_csv(RESULT_ROOT / f"results_t{threads}.csv", selected)


def fit_or_load_tau(
    dataset: base.Dataset,
    split: calibration.SplitFiles,
    *,
    force: bool,
    build_threads: int,
) -> float:
    path = calibration.RESULT_DIR / dataset.key / (
        f"calibration.single_tau_ef{REFERENCE_EF}.json"
    )
    if path.exists() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fitted = payload["calibration"]
        if int(payload["reference_ef"]) != REFERENCE_EF:
            raise RuntimeError(f"reference ef mismatch in {path}")
        if int(fitted["calibration_rows"]) != split.calibration_rows:
            raise RuntimeError(f"calibration split mismatch in {path}")
        print(f"REUSE {dataset.key} tau={float(fitted['tau']):.9g}", flush=True)
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
        f"rerank_mean={float(fitted['calibrated_count_mean']):.2f}",
        flush=True,
    )
    return float(fitted["tau"])


def summarize(
    dataset: base.Dataset,
    split: calibration.SplitFiles,
    *,
    ef: int,
    threads: int,
    rounds: int,
    tau: float,
    result_path: Path,
) -> dict[str, object]:
    raw = read_one(result_path)
    if int(raw["search_threads"]) != threads or int(raw["rounds"]) != rounds:
        raise RuntimeError(
            f"{result_path}: expected T={threads}, rounds={rounds}; "
            f"found T={raw['search_threads']}, rounds={raw['rounds']}"
        )
    nq = split.test_rows
    dim = int(raw["dim"])
    rerank_mean = float(raw["adaptive_rerank_count_mean"])
    rerank_min = float(raw["adaptive_rerank_count_min"])
    rerank_p95 = float(raw["adaptive_rerank_count_p95"])
    rerank_p99 = float(raw["adaptive_rerank_count_p99"])
    rerank_max = float(raw["adaptive_rerank_count_max"])
    submitted_bytes = float(raw["direct_submitted_bytes_mean"]) / nq
    logical_bytes = float(raw["direct_logical_bytes_mean"]) / nq
    requests = float(raw["direct_requests_mean"]) / nq
    direct_rows = float(raw["direct_rows_mean"]) / nq
    pages_per_vector = math.ceil(dim * 4 / PAGE_BYTES)

    if not math.isclose(direct_rows, rerank_mean, rel_tol=2e-5, abs_tol=2e-4):
        raise RuntimeError(
            f"{dataset.key} T={threads} ef={ef}: direct rows {direct_rows} "
            f"!= rerank mean {rerank_mean}"
        )
    expected_pages = rerank_mean * pages_per_vector
    actual_pages = submitted_bytes / PAGE_BYTES
    if not math.isclose(actual_pages, expected_pages, rel_tol=2e-5, abs_tol=2e-4):
        raise RuntimeError(
            f"{dataset.key} T={threads} ef={ef}: submitted pages {actual_pages} "
            f"!= rerank pages {expected_pages}"
        )

    return {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "dim": dim,
        "nb": int(raw["nb"]),
        "efSearch": ef,
        "search_threads": threads,
        "rounds": rounds,
        "calibration_queries": split.calibration_rows,
        "test_queries": nq,
        "calibration_reference_ef": REFERENCE_EF,
        "target_fixed_rerank": FIXED_R,
        "tau": tau,
        "recall_at_10": float(raw["recall_at_k"]),
        "qps": float(raw["qps"]),
        "rerank_vectors_per_query_mean": rerank_mean,
        "rerank_vectors_per_query_min": rerank_min,
        "rerank_vectors_per_query_p95": rerank_p95,
        "rerank_vectors_per_query_p99": rerank_p99,
        "rerank_vectors_per_query_max": rerank_max,
        "fp_io_requests_per_query_mean": requests,
        "fp_submitted_4k_pages_per_query_mean": actual_pages,
        "fp_submitted_4k_pages_per_query_p95": rerank_p95 * pages_per_vector,
        "fp_submitted_4k_pages_per_query_p99": rerank_p99 * pages_per_vector,
        "fp_submitted_4k_pages_per_query_max": rerank_max * pages_per_vector,
        "fp_submitted_bytes_per_query_mean": submitted_bytes,
        "fp_logical_4k_pages_per_query_mean": logical_bytes / PAGE_BYTES,
        "fp_logical_bytes_per_query_mean": logical_bytes,
        "vector_payload_bytes": dim * 4,
        "submitted_4k_pages_per_reranked_vector": pages_per_vector,
        "page_bytes": PAGE_BYTES,
        "storage": "SSD O_DIRECT FP sidecar; HNSW graph in RAM",
        "io_engine": "io_uring inter-query overlap",
        "query_scope": "held-out test queries after 1000-query calibration split",
        "raw_result_csv": str(result_path),
    }


def run_dataset(
    dataset_key: str,
    *,
    rounds: int,
    build_threads: int,
    force: bool,
    rows_by_key: dict[tuple[str, int, int], dict[str, object]],
) -> None:
    dataset = base.DATASETS[dataset_key]
    base.validate(dataset)
    split = calibration.make_split(dataset, CALIBRATION_QUERIES, 11, 0)
    tau = fit_or_load_tau(
        dataset,
        split,
        force=force,
        build_threads=build_threads,
    )
    sidecar = calibration.ALIGNED_FP_ROOT / f"{dataset.key}.fp32.aligned4k.fbin"
    sidecar_existed = sidecar.exists()
    try:
        for threads in THREADS:
            args = SimpleNamespace(
                force=force,
                fixed_r=FIXED_R,
                rounds=rounds,
                search_threads=threads,
                build_threads=build_threads,
            )
            variant = f".single_tau_ef{REFERENCE_EF}.t{threads}"
            for ef in EFS:
                result_path = calibration.benchmark_policy(
                    dataset,
                    split,
                    ef,
                    "calibrated",
                    tau,
                    sidecar,
                    args,
                    variant,
                )
                row = summarize(
                    dataset,
                    split,
                    ef=ef,
                    threads=threads,
                    rounds=rounds,
                    tau=tau,
                    result_path=result_path,
                )
                rows_by_key[(dataset.key, threads, ef)] = row
                save_progress(list(rows_by_key.values()))
                print(
                    f"RESULT {dataset.key} T={threads} ef={ef} "
                    f"recall={float(row['recall_at_10']):.6f} "
                    f"qps={float(row['qps']):.2f} "
                    f"rerank={float(row['rerank_vectors_per_query_mean']):.2f} "
                    f"pages={float(row['fp_submitted_4k_pages_per_query_mean']):.2f}",
                    flush=True,
                )
    finally:
        if not sidecar_existed and sidecar.exists():
            size = sidecar.stat().st_size
            sidecar.unlink()
            print(f"REMOVED transient sidecar bytes={size}: {sidecar}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=TARGETS, default=list(TARGETS))
    args = parser.parse_args()
    if args.rounds <= 0 or args.build_threads <= 0:
        parser.error("rounds and build-threads must be positive")

    calibration.ensure_binary()
    calibration.ALIGNED_FP_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "complete.json").unlink(missing_ok=True)
    rows_by_key: dict[tuple[str, int, int], dict[str, object]] = {}
    if not args.force:
        for row in read_csv(RESULT_ROOT / "all_results.csv"):
            key = (row["dataset"], int(row["search_threads"]), int(row["efSearch"]))
            rows_by_key[key] = dict(row)

    print(
        "START NQ/Quora calibrated ADC-BQ BGE+Qwen matrix; "
        "metrics=recall,qps,rerank-vectors,submitted-4k-pages",
        flush=True,
    )
    for dataset_key in args.datasets:
        run_dataset(
            dataset_key,
            rounds=args.rounds,
            build_threads=args.build_threads,
            force=args.force,
            rows_by_key=rows_by_key,
        )

    rows = list(rows_by_key.values())
    expected = len(args.datasets) * len(THREADS) * len(EFS)
    selected = [row for row in rows if str(row["dataset"]) in args.datasets]
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} selected rows, found {len(selected)}")
    save_progress(rows)
    completion = RESULT_ROOT / "complete.json"
    completion.write_text(
        json.dumps(
            {
                "status": "complete",
                "datasets": list(args.datasets),
                "threads": list(THREADS),
                "efs": list(EFS),
                "rows": len(selected),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE rows={len(selected)} marker={completion}", flush=True)


if __name__ == "__main__":
    main()
