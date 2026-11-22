#!/usr/bin/env python3
"""Per-ef ADC-BQ calibrated to preserve candidate-pool exact top-10.

For each dataset and efSearch value, fit a split-conformal threshold whose
query-level target is to retain all ten exact-FP nearest candidates inside the
ADC candidate pool with probability 0.99.  Evaluate the fitted threshold on
the common DiskVecLab held-out split at one search thread.

This is intentionally separate from the randomized-index mean-Recall target
runner.  The two calibration objectives are not interchangeable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import run_adcbq_vs_fp_hnsw_fullvec_ram as pool_calibration
import run_diskveclab_full_matrix_adcbq_target99_per_ef as runner


RESULT_ROOT = (
    runner.matrix.RESULT_ROOT / "calibrated_adcbq_pool_top10_target99_per_ef"
)
SIDECAR_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/"
    "diskveclab_full_matrix_adcbq_pool_top10_target99_sidecars"
)
PROTOCOL = "per-ef-pool-top10-split-conformal"
PAGE_BYTES = 4096.0


def calibration_path(dataset_key: str, ef: int) -> Path:
    return (
        RESULT_ROOT
        / dataset_key
        / "calibration"
        / f"pool_top10_target99.ef{ef}.json"
    )


def load_calibration(
    path: Path,
    split: runner.calibration.SplitFiles,
    ef: int,
    target_recall: float,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        payload.get("protocol") == PROTOCOL
        and int(payload.get("ef", -1)) == ef
        and int(payload.get("calibration_rows", -1)) == split.calibration_rows
        and float(payload.get("target_pool_top10_coverage", -1.0))
        == target_recall
    )
    return payload if expected else None


def fit_per_ef(
    dataset: runner.registry.Dataset,
    split: runner.calibration.SplitFiles,
    ef: int,
    target_recall: float,
    _randomized_index_seed: int,
    build_threads: int,
    force: bool,
    keep_candidates: bool,
) -> dict[str, object]:
    fitted_path = calibration_path(dataset.key, ef)
    if not force:
        cached = load_calibration(fitted_path, split, ef, target_recall)
        if cached is not None:
            print(
                f"REUSE pool-top10 calibration {dataset.key} ef={ef} "
                f"status={cached['status']}",
                flush=True,
            )
            return cached

    summary, candidates = runner.candidate_paths(dataset.key, ef)
    expected_lines = split.calibration_rows * ef + 1
    if not (
        not force
        and summary.is_file()
        and candidates.is_file()
        and sum(1 for _ in candidates.open(encoding="utf-8")) == expected_lines
    ):
        command = runner.calibration.common_command(
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
            "--rerank-n",
            str(ef),
            "--candidate-only",
            "--no-rerank",
            "--candidates-out",
            str(candidates),
        ]
        runner.calibration.run(command, 1)

    base_payload: dict[str, object] = {
        "protocol": PROTOCOL,
        "target_pool_top10_coverage": target_recall,
        "ef": ef,
        "calibration_rows": split.calibration_rows,
    }
    try:
        pool_calibration.TARGET_COVERAGE = target_recall
        fitted = pool_calibration.fit_pool_top10_conformal(
            SimpleNamespace(dataset=dataset), split, candidates, ef
        )
        payload = {**base_payload, **fitted, "status": "feasible"}
    except RuntimeError as error:
        payload = {**base_payload, "status": "infeasible", "reason": str(error)}

    runner.write_json(fitted_path, payload)
    if not keep_candidates:
        candidates.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
    print(
        f"FITTED pool-top10 calibration {dataset.key} ef={ef} "
        f"status={payload['status']}",
        flush=True,
    )
    return payload


def benchmark(
    dataset: runner.registry.Dataset,
    split: runner.calibration.SplitFiles,
    ef: int,
    fitted: dict[str, object],
    sidecar: Path,
    rounds: int,
    build_threads: int,
    force: bool,
) -> Path:
    output = (
        RESULT_ROOT
        / dataset.key
        / "test"
        / f"pool_top10_target99.ef{ef}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tau = float(fitted["tau"])
    if not force and runner.result_complete(output, split, ef, rounds, tau):
        print(f"REUSE pool-top10 test result {dataset.key} ef={ef}", flush=True)
        return output

    command = runner.calibration.common_command(
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
        "--rerank-n",
        str(ef),
        "--rerank-dot",
        "fp32",
        "--fp-rerank-source",
        "odirect-uring",
        "--fp-rerank-norms",
        "onfetch",
        "--fp-rerank-path",
        str(sidecar),
        "--fp-rerank-overlap-queries",
        "--fp-rerank-qd",
        str(ef),
        "--calibrated-threshold-rerank-tau",
        repr(tau),
    ]
    runner.calibration.run(command, 1)
    return output


def standard_row(
    dataset: runner.registry.Dataset,
    split: runner.calibration.SplitFiles,
    ef: int,
    fitted: dict[str, object],
    result_path: Path | None,
    rounds: int,
) -> dict[str, object]:
    nb, dim = runner.registry.matrix_shape(dataset.train)
    base: dict[str, object] = {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method": "ADC-BQ per-ef candidate-pool top-10 preservation@0.99",
        "search_parameter": "efSearch",
        "search_parameter_value": ef,
        "efSearch": ef,
        "calibration_status": fitted["status"],
        "calibration_protocol": fitted["protocol"],
        "target_pool_top10_coverage": fitted["target_pool_top10_coverage"],
        "empirical_pool_top10_coverage": fitted.get(
            "empirical_all_top10_coverage", ""
        ),
        "conformal_order": fitted.get("conformal_order", ""),
        "tau": fitted.get("tau", ""),
        "calibration_rerank_count_mean": fitted.get("rerank_count_mean", ""),
        "calibration_rerank_count_p95": fitted.get("rerank_count_p95", ""),
        "calibration_rerank_count_p99": fitted.get("rerank_count_p99", ""),
        "calibration_rerank_count_max": fitted.get("rerank_count_max", ""),
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

    result = runner.read_one(result_path)
    submitted_per_query = (
        float(result["direct_submitted_bytes_mean"]) / split.test_rows
    )
    base.update(
        {
            "recall_at_10": float(result["recall_at_k"]),
            "qps": float(result["qps"]),
            "rounds": int(result["rounds"]),
            "fp_vectors_per_query_mean": float(
                result["adaptive_rerank_count_mean"]
            ),
            "fp_vectors_p95": float(result["adaptive_rerank_count_p95"]),
            "fp_vectors_p99": float(result["adaptive_rerank_count_p99"]),
            "fp_vectors_max": float(result["adaptive_rerank_count_max"]),
            "fp_4k_pages_per_query_mean": submitted_per_query / PAGE_BYTES,
            "total_4k_pages_per_query_mean": submitted_per_query / PAGE_BYTES,
        }
    )
    return base


def main() -> int:
    runner.__doc__ = __doc__
    runner.RESULT_ROOT = RESULT_ROOT
    runner.SIDECAR_ROOT = SIDECAR_ROOT
    runner.fit_per_ef = fit_per_ef
    runner.benchmark = benchmark
    runner.standard_row = standard_row
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
