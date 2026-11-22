#!/usr/bin/env python3
"""Per-ef calibrated ADC-BQ at target mean Recall@10=0.99.

This intentionally does not reuse the legacy single-tau calibration.  For
every dataset and every efSearch value, it collects calibration candidates,
fits a fresh query-adaptive threshold with the randomized-index split-
conformal mean-recall rule, and evaluates that threshold on the common
held-out test split at T=1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import diskveclab_full_matrix as matrix
import run_adcbq_calibrated_threshold_overlap as calibration
import run_msmarco_qwen1024_adcbq_target99_compare as target99
import run_v051_ssd_odirect_multidataset as registry


RESULT_ROOT = matrix.RESULT_ROOT / "calibrated_adcbq_target99_per_ef"
SIDECAR_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/"
    "diskveclab_full_matrix_adcbq_target99_sidecars"
)
DEFAULT_EFS = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)
PAGE_BYTES = 4096.0


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def upsert_summary(row: dict[str, object]) -> None:
    path = RESULT_ROOT / "summary_t1.csv"
    existing: list[dict[str, object]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = [
                dict(item)
                for item in csv.DictReader(handle)
                if not (
                    item["dataset"] == str(row["dataset"])
                    and int(item["efSearch"]) == int(row["efSearch"])
                )
            ]
    existing.append(row)
    existing.sort(key=lambda item: (str(item["dataset"]), int(item["efSearch"])))
    write_csv(path, existing)


def calibration_path(dataset_key: str, ef: int) -> Path:
    return RESULT_ROOT / dataset_key / "calibration" / f"target99.ef{ef}.json"


def candidate_paths(dataset_key: str, ef: int) -> tuple[Path, Path]:
    root = RESULT_ROOT / dataset_key / "calibration"
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / f"candidates.ef{ef}.summary.csv",
        root / f"candidates.ef{ef}.csv",
    )


def load_calibration(
    path: Path,
    split: calibration.SplitFiles,
    ef: int,
    target_recall: float,
    randomized_index_seed: int,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        int(payload.get("ef", -1)) == ef
        and int(payload.get("calibration_rows", -1)) == split.calibration_rows
        and float(payload.get("target_mean_recall", -1.0)) == target_recall
        and int(payload.get("randomized_index_seed", -1)) == randomized_index_seed
        and payload.get("protocol") == "per-ef-randomized-index-conformal"
    )
    return payload if expected else None


def fit_per_ef(
    dataset: registry.Dataset,
    split: calibration.SplitFiles,
    ef: int,
    target_recall: float,
    randomized_index_seed: int,
    build_threads: int,
    force: bool,
    keep_candidates: bool,
) -> dict[str, object]:
    fitted_path = calibration_path(dataset.key, ef)
    if not force:
        cached = load_calibration(
            fitted_path, split, ef, target_recall, randomized_index_seed
        )
        if cached is not None:
            print(
                f"REUSE per-ef calibration {dataset.key} ef={ef} "
                f"status={cached['status']}",
                flush=True,
            )
            return cached

    summary, candidates = candidate_paths(dataset.key, ef)
    expected_lines = split.calibration_rows * ef + 1
    if not (
        not force
        and summary.exists()
        and candidates.exists()
        and sum(1 for _ in candidates.open(encoding="utf-8")) == expected_lines
    ):
        command = calibration.common_command(
            dataset,
            split.calibration_query,
            split.calibration_gt,
            summary,
            ef,
            1,
            1,
            build_threads,
        )
        command += [
            "--rerank-n", str(ef),
            "--candidate-only",
            "--no-rerank",
            "--candidates-out", str(candidates),
        ]
        calibration.run(command, 1)

    base_payload: dict[str, object] = {
        "protocol": "per-ef-randomized-index-conformal",
        "target_mean_recall": target_recall,
        "ef": ef,
        "calibration_rows": split.calibration_rows,
        "randomized_index_seed": randomized_index_seed,
    }
    try:
        fitted = target99.fit_randomized_mean_recall_target(
            candidates,
            split,
            ef,
            target_recall,
            randomized_index_seed,
        )
        payload = {**base_payload, **fitted, "status": "feasible"}
    except RuntimeError as error:
        payload = {**base_payload, "status": "infeasible", "reason": str(error)}
    write_json(fitted_path, payload)
    if not keep_candidates:
        candidates.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
    print(
        f"FITTED per-ef calibration {dataset.key} ef={ef} "
        f"status={payload['status']}",
        flush=True,
    )
    return payload


def result_complete(
    path: Path,
    split: calibration.SplitFiles,
    ef: int,
    rounds: int,
    tau: float,
) -> bool:
    if not path.exists():
        return False
    try:
        row = read_one(path)
        return (
            int(row["nq"]) == split.test_rows
            and int(row["efSearch"]) == ef
            and int(row["search_threads"]) == 1
            and int(row["rounds"]) == rounds
            and row["calibrated_threshold_rerank_enabled"] == "true"
            and abs(float(row["calibrated_threshold_rerank_tau"]) - tau)
            <= 1e-5 * max(1.0, abs(tau))
        )
    except (KeyError, RuntimeError, ValueError):
        return False


def benchmark(
    dataset: registry.Dataset,
    split: calibration.SplitFiles,
    ef: int,
    fitted: dict[str, object],
    sidecar: Path,
    rounds: int,
    build_threads: int,
    force: bool,
) -> Path:
    output = RESULT_ROOT / dataset.key / "test" / f"target99.ef{ef}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    tau = float(fitted["tau"])
    if not force and result_complete(output, split, ef, rounds, tau):
        print(f"REUSE test result {dataset.key} ef={ef}", flush=True)
        return output
    command = calibration.common_command(
        dataset,
        split.test_query,
        split.test_gt,
        output,
        ef,
        rounds,
        1,
        build_threads,
    )
    command += [
        "--rerank-n", str(ef),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", "odirect-uring",
        "--fp-rerank-norms", "onfetch",
        "--fp-rerank-path", str(sidecar),
        "--fp-rerank-overlap-queries",
        "--fp-rerank-qd", str(ef),
        "--calibrated-threshold-rerank-tau", repr(tau),
    ]
    calibration.run(command, 1)
    return output


def standard_row(
    dataset: registry.Dataset,
    split: calibration.SplitFiles,
    ef: int,
    fitted: dict[str, object],
    result_path: Path | None,
    rounds: int,
) -> dict[str, object]:
    nb, dim = registry.matrix_shape(dataset.train)
    base: dict[str, object] = {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method": "ADC-BQ per-ef calibrated target mean Recall@10=0.99",
        "search_parameter": "efSearch",
        "search_parameter_value": ef,
        "efSearch": ef,
        "calibration_status": fitted["status"],
        "calibration_protocol": fitted["protocol"],
        "target_mean_recall": fitted["target_mean_recall"],
        "calibration_recall_at_10": fitted.get("calibration_recall_at_10", ""),
        "max_candidate_recall_at_10": fitted.get("max_candidate_recall_at_10", ""),
        "tau": fitted.get("tau", ""),
        "recall_at_10": "",
        "qps": "",
        "rounds": rounds,
        "search_threads": 1,
        "build_threads": 24,
        "calibration_queries": split.calibration_rows,
        "test_queries": split.test_rows,
        "nb": nb,
        "dim": dim,
        "fp_vectors_per_query_mean": "",
        "fp_vectors_p95": "",
        "fp_vectors_p99": "",
        "fp_vectors_max": "",
        "fp_4k_pages_per_query_mean": "",
        "total_4k_pages_per_query_mean": "",
        "storage": "RAM ADC-BQ HNSW + SSD FP32 sidecar",
        "io_engine": "O_DIRECT io_uring inter-query overlap",
        "result_path": "" if result_path is None else str(result_path),
        "reason": fitted.get("reason", ""),
    }
    if result_path is None:
        return base
    result = read_one(result_path)
    submitted_per_query = (
        float(result["direct_submitted_bytes_mean"]) / split.test_rows
    )
    base.update(
        {
            "recall_at_10": float(result["recall_at_k"]),
            "qps": float(result["qps"]),
            "rounds": int(result["rounds"]),
            "fp_vectors_per_query_mean": float(result["adaptive_rerank_count_mean"]),
            "fp_vectors_p95": float(result["adaptive_rerank_count_p95"]),
            "fp_vectors_p99": float(result["adaptive_rerank_count_p99"]),
            "fp_vectors_max": float(result["adaptive_rerank_count_max"]),
            "fp_4k_pages_per_query_mean": submitted_per_query / PAGE_BYTES,
            "total_4k_pages_per_query_mean": submitted_per_query / PAGE_BYTES,
        }
    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--randomized-index-seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--keep-sidecars", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-benchmark",
        action="store_true",
        help="reuse fitted calibrations but rerun every test benchmark",
    )
    args = parser.parse_args()
    if not (0.0 < args.target_recall < 1.0):
        parser.error("--target-recall must be in (0, 1)")
    if args.rounds < 1 or args.build_threads != 24:
        parser.error("rounds must be positive and build threads must be 24")
    if min(args.efs) < 10:
        parser.error("every efSearch must be at least k=10")

    calibration.ensure_binary()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    SIDECAR_ROOT.mkdir(parents=True, exist_ok=True)
    for key in args.datasets:
        dataset = registry.DATASETS[key]
        registry.validate(dataset)
        split = matrix.split_for(key)
        sidecar = SIDECAR_ROOT / f"{key}.fp32.aligned4k.fbin"
        print(
            f"DATASET {key}: cal={split.calibration_rows} test={split.test_rows}",
            flush=True,
        )
        try:
            for ef in list(dict.fromkeys(args.efs)):
                fitted = fit_per_ef(
                    dataset,
                    split,
                    ef,
                    args.target_recall,
                    args.randomized_index_seed,
                    args.build_threads,
                    args.force,
                    args.keep_candidates,
                )
                if fitted["status"] != "feasible":
                    upsert_summary(
                        standard_row(dataset, split, ef, fitted, None, args.rounds)
                    )
                    continue
                result = benchmark(
                    dataset,
                    split,
                    ef,
                    fitted,
                    sidecar,
                    args.rounds,
                    args.build_threads,
                    args.force or args.force_benchmark,
                )
                row = standard_row(dataset, split, ef, fitted, result, args.rounds)
                upsert_summary(row)
                print(
                    f"RESULT {key} ef={ef} recall={float(row['recall_at_10']):.6f} "
                    f"qps={float(row['qps']):.2f} "
                    f"Rmean={float(row['fp_vectors_per_query_mean']):.2f}",
                    flush=True,
                )
        finally:
            if not args.keep_sidecars and sidecar.exists():
                size = sidecar.stat().st_size
                sidecar.unlink()
                print(f"REMOVED transient sidecar bytes={size}: {sidecar}", flush=True)
    print("ALL PER-EF TARGET-0.99 ADC-BQ MEASUREMENTS COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
