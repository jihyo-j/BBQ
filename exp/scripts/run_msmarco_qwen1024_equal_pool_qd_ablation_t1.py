#!/usr/bin/env python3
"""Compare fixed and calibrated ADC-BQ with identical candidate pools and QD.

For every efSearch value both policies materialize ``candidate_pool=ef`` and
use ``fp_rerank_qd=ef``.  Fixed refinement reads exactly R=75 vectors, while
the calibrated policy reads its query-adaptive prefix.  This isolates exact-FP
read-count savings from candidate-pool and queue-depth differences.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import run_msmarco_qwen1024_recall_ablation_t1 as base


REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_equal_pool_qd"
RESULT_ROOT = REPO / "experiment/msmarco_qwen1024_equal_pool_qd_ablation_t1"
REFERENCE_ROOT = (
    REPO
    / "experiment/msmarco_qwen1024_recall_ablation_t1/"
    "clean_rerun_20261117_cpu39_v6/raw"
)
DEFAULT_EFS = (80, 100, 120, 140, 160, 180, 200, 220, 240, 256, 320, 512, 1024)


def read_one(path: Path) -> dict[str, str]:
    rows = base.read_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


def result_complete(
    path: Path,
    *,
    ef: int,
    fixed_r: int,
    rounds: int,
    nq: int,
    calibrated: bool,
    tau: float | None,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = read_one(path)
        if (
            int(row["efSearch"]) != ef
            or int(row["candidate_pool_size"]) != ef
            or int(row["fp_rerank_qd"]) != ef
            or int(row["rounds"]) != rounds
            or int(row["nq"]) != nq
            or int(row["search_threads"]) != 1
        ):
            return False
        enabled = row["calibrated_threshold_rerank_enabled"] == "true"
        if enabled != calibrated:
            return False
        if calibrated:
            assert tau is not None
            observed = float(row["calibrated_threshold_rerank_tau"])
            return abs(observed - tau) <= 1e-5 * max(1.0, abs(tau))
        return int(row["rerank_n"]) == fixed_r
    except (KeyError, ValueError, RuntimeError):
        return False


def rerank_args(*, rerank_n: int, candidate_pool: int, qd: int) -> list[str]:
    return [
        "--rerank-n", str(rerank_n),
        "--candidate-pool-size", str(candidate_pool),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", "odirect-uring",
        "--fp-rerank-norms", "onfetch",
        "--fp-rerank-path", str(base.FP_SIDECAR),
        "--fp-rerank-overlap-queries",
        "--fp-rerank-qd", str(qd),
    ]


def run_one(
    root: Path,
    *,
    ef: int,
    fixed_r: int,
    rounds: int,
    max_queries: int,
    nq: int,
    cpu: int,
    calibrated: bool,
    force: bool,
) -> Path:
    calibration = base.calibration_for_ef(ef)
    tau = float(calibration["tau"])
    policy = "calibrated" if calibrated else f"fixed_r{fixed_r}"
    output = root / "raw" / f"{policy}.pool{ef}.qd{ef}.ef{ef}.csv"
    if not force and result_complete(
        output,
        ef=ef,
        fixed_r=fixed_r,
        rounds=rounds,
        nq=nq,
        calibrated=calibrated,
        tau=tau if calibrated else None,
    ):
        print(f"REUSE {output}", flush=True)
        return output

    command = base.adcbq_base(
        base.TEST_QUERY,
        base.TEST_GT,
        output,
        [ef],
        rounds,
        max_queries,
    )
    command += rerank_args(
        rerank_n=ef if calibrated else fixed_r,
        candidate_pool=ef,
        qd=ef,
    )
    if calibrated:
        command += ["--calibrated-threshold-rerank-tau", repr(tau)]
    base.run(command, cpu)
    if not result_complete(
        output,
        ef=ef,
        fixed_r=fixed_r,
        rounds=rounds,
        nq=nq,
        calibrated=calibrated,
        tau=tau if calibrated else None,
    ):
        raise RuntimeError(f"incomplete or incompatible result: {output}")
    return output


def percent_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def validate_inputs(efs: list[int]) -> None:
    required = (
        BINARY,
        base.TRAIN,
        base.GRAPH,
        base.FP_SIDECAR,
        base.TEST_QUERY,
        base.TEST_GT,
        REFERENCE_ROOT / "adcbq_fixed_r75.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for ef in efs:
        calibration = base.CALIBRATION_ROOT / f"pool_top10_target99.ef{ef}.json"
        reference = REFERENCE_ROOT / f"adcbq_calibrated_ef{ef}.csv"
        if not calibration.is_file():
            raise FileNotFoundError(calibration)
        if not reference.is_file():
            raise FileNotFoundError(reference)


def write_summary(
    root: Path,
    efs: list[int],
    fixed_paths: dict[int, Path],
    calibrated_paths: dict[int, Path],
    fixed_r: int,
) -> Path:
    rows: list[dict[str, object]] = []
    for ef in efs:
        fixed = read_one(fixed_paths[ef])
        calibrated = read_one(calibrated_paths[ef])
        fixed_recall = float(fixed["recall_at_k"])
        calibrated_recall = float(calibrated["recall_at_k"])
        fixed_qps = float(fixed.get("qps_median") or fixed["qps"])
        calibrated_qps = float(calibrated.get("qps_median") or calibrated["qps"])
        fixed_reads = float(fixed["adaptive_rerank_count_mean"])
        calibrated_reads = float(calibrated["adaptive_rerank_count_mean"])

        old_fixed_path = REFERENCE_ROOT / "adcbq_fixed_r75.csv"
        old_fixed_rows = {
            int(row["efSearch"]): row for row in base.read_rows(old_fixed_path)
        }
        old_fixed_recall = float(old_fixed_rows[ef]["recall_at_k"])
        old_calibrated_path = REFERENCE_ROOT / f"adcbq_calibrated_ef{ef}.csv"
        old_calibrated_recall = float(read_one(old_calibrated_path)["recall_at_k"])

        rows.append(
            {
                "dataset": "msmarco_qwen1024",
                "efSearch": ef,
                "candidate_pool_size_both": ef,
                "fp_rerank_qd_both": ef,
                "fixed_r": fixed_r,
                "fixed_reads_mean": fixed_reads,
                "fixed_recall_at_10": fixed_recall,
                "fixed_qps_median": fixed_qps,
                "calibrated_tau": float(calibrated["calibrated_threshold_rerank_tau"]),
                "calibrated_reads_mean": calibrated_reads,
                "calibrated_reads_p95": float(calibrated["adaptive_rerank_count_p95"]),
                "calibrated_reads_p99": float(calibrated["adaptive_rerank_count_p99"]),
                "calibrated_reads_max": float(calibrated["adaptive_rerank_count_max"]),
                "calibrated_recall_at_10": calibrated_recall,
                "calibrated_qps_median": calibrated_qps,
                "read_reduction_pct": -percent_change(calibrated_reads, fixed_reads),
                "calibrated_qps_change_pct": percent_change(calibrated_qps, fixed_qps),
                "calibrated_minus_fixed_recall": calibrated_recall - fixed_recall,
                "fixed_recall_delta_vs_old_pool75": fixed_recall - old_fixed_recall,
                "calibrated_recall_delta_vs_old": calibrated_recall - old_calibrated_recall,
                "fixed_source_csv": str(fixed_paths[ef]),
                "calibrated_source_csv": str(calibrated_paths[ef]),
            }
        )

    output = root / "equal_pool_qd_fixed75_vs_calibrated.csv"
    base.write_rows(output, rows)
    manifest = {
        "protocol": "equal-candidate-pool-equal-qd-fixed75-vs-calibrated",
        "candidate_pool": "efSearch for both policies",
        "fp_rerank_qd": "efSearch for both policies",
        "fixed_reads": fixed_r,
        "calibrated_reads": "query-adaptive threshold prefix",
        "efs": efs,
        "binary": str(BINARY),
        "summary": str(output),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--fixed-r", type=int, default=75)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    efs = list(dict.fromkeys(args.efs))
    if min(efs) < args.fixed_r or args.fixed_r < 10:
        parser.error("require efSearch >= fixed R >= 10")
    if not BINARY.is_file():
        raise FileNotFoundError(f"build the fair-pool binary first: {BINARY}")
    base.ADCBQ_BINARY = BINARY
    validate_inputs(efs)

    nq = 5980 if args.max_queries == 0 else min(args.max_queries, 5980)
    binary_max_queries = 0 if nq == 5980 else nq
    root = args.result_root.resolve()
    fixed_paths: dict[int, Path] = {}
    calibrated_paths: dict[int, Path] = {}
    for ef in efs:
        fixed_paths[ef] = run_one(
            root,
            ef=ef,
            fixed_r=args.fixed_r,
            rounds=args.rounds,
            max_queries=binary_max_queries,
            nq=nq,
            cpu=args.cpu,
            calibrated=False,
            force=args.force,
        )
        calibrated_paths[ef] = run_one(
            root,
            ef=ef,
            fixed_r=args.fixed_r,
            rounds=args.rounds,
            max_queries=binary_max_queries,
            nq=nq,
            cpu=args.cpu,
            calibrated=True,
            force=args.force,
        )
    write_summary(root, efs, fixed_paths, calibrated_paths, args.fixed_r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
