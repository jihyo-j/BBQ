#!/usr/bin/env python3
"""Compare matched-fixed80 and direct conformal target ADC-BQ refinement.

The direct target follows ``adcbq_calibration/adc_budget.py``: one score-gap
statistic per calibration query, using the worst of the exact FP top-k.  Its
split-conformal quantile targets the probability that all FP top-k neighbors
are retained by the query-adaptive refinement set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import run_adcbq_calibrated_threshold_overlap as calibration
import run_v051_ssd_odirect_multidataset as datasets


REPO = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO / "experiment/msmarco_qwen1024_adcbq_target99_compare"
DEFAULT_FP_SIDECAR = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_nq_msmarco_qwen1024/"
    "msmarco_qwen1024/bbq/msmarco_qwen1024.fp32.aligned4k.fbin"
)
MATCHED_CALIBRATION = (
    REPO
    / "experiment/adcbq_calibrated_threshold_overlap/msmarco_qwen1024/"
    "calibration.single_tau_ef256.json"
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def collect_calibration_candidates(
    dataset: datasets.Dataset,
    split: calibration.SplitFiles,
    ef: int,
    build_threads: int,
    force: bool,
) -> Path:
    summary = RESULT_DIR / f"calibration_candidates.ef{ef}.summary.csv"
    candidates = RESULT_DIR / f"calibration_candidates.ef{ef}.csv"
    expected_lines = split.calibration_rows * ef + 1
    if (
        not force
        and summary.exists()
        and candidates.exists()
        and sum(1 for _ in candidates.open()) == expected_lines
    ):
        return candidates

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
        "--rerank-n",
        str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out",
        str(candidates),
    ]
    calibration.run(command, 1)
    return candidates


def fit_allk_target(
    candidate_path: Path,
    split: calibration.SplitFiles,
    ef: int,
    target_coverage: float,
) -> dict[str, float | int | bool]:
    ids, distances = calibration.load_candidate_csv(
        candidate_path, split.calibration_rows, ef
    )
    gt = np.memmap(
        split.calibration_gt,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(split.calibration_rows, split.gt_dim),
    )[:, :10]
    scales = calibration.int16_query_scales(split.calibration_query)

    ranks = np.full((split.calibration_rows, 10), ef + 1, dtype=np.int32)
    requirements = np.full((split.calibration_rows, 10), np.inf, dtype=np.float64)
    for qi in range(split.calibration_rows):
        positions = {int(doc): rank for rank, doc in enumerate(ids[qi]) if doc >= 0}
        d_k = float(distances[qi, 9])
        for gj, gold in enumerate(gt[qi]):
            rank = positions.get(int(gold))
            if rank is None:
                continue
            ranks[qi, gj] = rank + 1
            requirements[qi, gj] = (
                float(distances[qi, rank]) - d_k
            ) / float(scales[qi])

    per_query_need = requirements.max(axis=1)
    order = int(math.ceil((split.calibration_rows + 1) * target_coverage))
    if order > split.calibration_rows:
        raise RuntimeError(
            f"target {target_coverage} is not finite-sample feasible with "
            f"m={split.calibration_rows}: conformal order {order} exceeds m"
        )
    tau = float(np.sort(per_query_need)[order - 1])
    max_candidate_recall = float((ranks <= ef).mean())
    max_allk_coverage = float((ranks <= ef).all(axis=1).mean())
    if not np.isfinite(tau):
        raise RuntimeError(
            f"all-k target {target_coverage:.6f} is unreachable at ef={ef}: "
            f"calibration max all-k coverage={max_allk_coverage:.6f}, "
            f"max candidate Recall@10={max_candidate_recall:.6f}"
        )

    cutoffs = distances[:, 9] + tau * scales
    counts = np.maximum(10, (distances <= cutoffs[:, None]).sum(axis=1))
    counts = np.minimum(counts, ef).astype(np.int32)
    hit = ranks <= counts[:, None]
    return {
        "target_allk_coverage": target_coverage,
        "rho": 1.0 - target_coverage,
        "conformal_order": order,
        "calibration_rows": split.calibration_rows,
        "ef": ef,
        "tau": tau,
        "calibration_allk_coverage": float(hit.all(axis=1).mean()),
        "calibration_recall_at_10": float(hit.mean()),
        "max_candidate_allk_coverage": max_allk_coverage,
        "max_candidate_recall_at_10": max_candidate_recall,
        "rerank_mean": float(counts.mean()),
        "rerank_p50": float(np.percentile(counts, 50)),
        "rerank_p95": float(np.percentile(counts, 95)),
        "rerank_p99": float(np.percentile(counts, 99)),
        "rerank_max": int(counts.max()),
    }


def fit_randomized_mean_recall_target(
    candidate_path: Path,
    split: calibration.SplitFiles,
    ef: int,
    target_recall: float,
    randomized_index_seed: int,
) -> dict[str, float | int | bool]:
    """Conformalize one uniformly sampled GT position per calibration query.

    For a fresh uniform J in {1, ..., k}, marginal coverage of the sampled
    neighbor is exactly expected Recall@k.  This targets mean Recall@k rather
    than the stronger event that every one of the k neighbors is retained.
    """
    ids, distances = calibration.load_candidate_csv(
        candidate_path, split.calibration_rows, ef
    )
    gt = np.memmap(
        split.calibration_gt,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(split.calibration_rows, split.gt_dim),
    )[:, :10]
    scales = calibration.int16_query_scales(split.calibration_query)
    ranks = np.full((split.calibration_rows, 10), ef + 1, dtype=np.int32)
    requirements = np.full((split.calibration_rows, 10), np.inf, dtype=np.float64)
    for qi in range(split.calibration_rows):
        positions = {int(doc): rank for rank, doc in enumerate(ids[qi]) if doc >= 0}
        d_k = float(distances[qi, 9])
        for gj, gold in enumerate(gt[qi]):
            rank = positions.get(int(gold))
            if rank is None:
                continue
            ranks[qi, gj] = rank + 1
            requirements[qi, gj] = (
                float(distances[qi, rank]) - d_k
            ) / float(scales[qi])

    order = int(math.ceil((split.calibration_rows + 1) * target_recall))
    if order > split.calibration_rows:
        raise RuntimeError(
            f"target {target_recall} is not finite-sample feasible with "
            f"m={split.calibration_rows}: conformal order {order} exceeds m"
        )
    rng = np.random.default_rng(randomized_index_seed)
    sampled_positions = rng.integers(0, 10, size=split.calibration_rows)
    sampled_requirements = requirements[
        np.arange(split.calibration_rows), sampled_positions
    ]
    tau = float(np.sort(sampled_requirements)[order - 1])
    if not np.isfinite(tau):
        raise RuntimeError(
            f"randomized mean-recall target {target_recall:.6f} is unreachable "
            f"at ef={ef}: sampled traversal misses="
            f"{int(np.isinf(sampled_requirements).sum())}"
        )

    cutoffs = distances[:, 9] + tau * scales
    counts = np.maximum(10, (distances <= cutoffs[:, None]).sum(axis=1))
    counts = np.minimum(counts, ef).astype(np.int32)
    hit = ranks <= counts[:, None]
    return {
        "target_mean_recall": target_recall,
        "rho": 1.0 - target_recall,
        "conformal_order": order,
        "randomized_index_seed": randomized_index_seed,
        "sampled_traversal_misses": int(np.isinf(sampled_requirements).sum()),
        "calibration_rows": split.calibration_rows,
        "ef": ef,
        "tau": tau,
        "calibration_allk_coverage": float(hit.all(axis=1).mean()),
        "calibration_recall_at_10": float(hit.mean()),
        "max_candidate_allk_coverage": float((ranks <= ef).all(axis=1).mean()),
        "max_candidate_recall_at_10": float((ranks <= ef).mean()),
        "rerank_mean": float(counts.mean()),
        "rerank_p50": float(np.percentile(counts, 50)),
        "rerank_p95": float(np.percentile(counts, 95)),
        "rerank_p99": float(np.percentile(counts, 99)),
        "rerank_max": int(counts.max()),
    }


def benchmark(
    dataset: datasets.Dataset,
    split: calibration.SplitFiles,
    ef: int,
    tau: float,
    label: str,
    sidecar: Path,
    rounds: int,
    build_threads: int,
    force: bool,
) -> Path:
    output = RESULT_DIR / f"test.{label}.ef{ef}.csv"
    if output.exists() and not force:
        try:
            row = read_row(output)
        except (RuntimeError, KeyError, ValueError):
            row = None
        if row is not None and int(row["nq"]) == split.test_rows:
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
    calibration.run(command, 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-coverage", type=float, default=0.99)
    parser.add_argument(
        "--target-mode",
        choices=("allk-coverage", "mean-recall-randomized"),
        default="allk-coverage",
    )
    parser.add_argument("--randomized-index-seed", type=int, default=11)
    parser.add_argument("--ef", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--fp-sidecar", type=Path, default=DEFAULT_FP_SIDECAR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not (0.0 < args.target_coverage < 1.0):
        parser.error("--target-coverage must be in (0, 1)")
    if not args.fp_sidecar.exists():
        parser.error(f"missing aligned FP sidecar: {args.fp_sidecar}")
    if not MATCHED_CALIBRATION.exists():
        parser.error(f"missing matched-R80 calibration: {MATCHED_CALIBRATION}")

    calibration.ensure_binary()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = datasets.DATASETS["msmarco_qwen1024"]
    datasets.validate(dataset)
    split = calibration.make_split(dataset, 1000, 11, 0)
    candidates = collect_calibration_candidates(
        dataset, split, args.ef, args.build_threads, args.force
    )
    if args.target_mode == "allk-coverage":
        target = fit_allk_target(candidates, split, args.ef, args.target_coverage)
        target_label = "target_allk_0p99"
        target_policy = "conformal_target_allk_0.99"
        target_metric = "all-top10 coverage"
        target_value = float(target["target_allk_coverage"])
        calibration_filename = "calibration.target_allk.json"
    else:
        target = fit_randomized_mean_recall_target(
            candidates,
            split,
            args.ef,
            args.target_coverage,
            args.randomized_index_seed,
        )
        target_label = "target_mean_recall_0p99"
        target_policy = "conformal_target_mean_recall_0.99"
        target_metric = "expected Recall@10 (randomized-index conformal)"
        target_value = float(target["target_mean_recall"])
        calibration_filename = "calibration.target_mean_recall.json"
    write_json(RESULT_DIR / calibration_filename, target)

    matched_payload = json.loads(MATCHED_CALIBRATION.read_text())
    matched = matched_payload["calibration"]
    matched_path = benchmark(
        dataset,
        split,
        args.ef,
        float(matched["tau"]),
        "matched_fixed80",
        args.fp_sidecar,
        args.rounds,
        args.build_threads,
        args.force,
    )
    target_path = benchmark(
        dataset,
        split,
        args.ef,
        float(target["tau"]),
        target_label,
        args.fp_sidecar,
        args.rounds,
        args.build_threads,
        args.force,
    )

    rows: list[dict[str, object]] = []
    for policy, calibration_info, result_path in (
        ("matched_fixed80_recall", matched, matched_path),
        (target_policy, target, target_path),
    ):
        result = read_row(result_path)
        rows.append(
            {
                "dataset": dataset.key,
                "policy": policy,
                "efSearch": args.ef,
                "calibration_queries": split.calibration_rows,
                "test_queries": split.test_rows,
                "tau": float(calibration_info["tau"]),
                "calibration_target": (
                    float(calibration_info["fixed_calibration_recall"])
                    if policy == "matched_fixed80_recall"
                    else target_value
                ),
                "calibration_target_metric": (
                    "Recall@10 of fixed rerank-80"
                    if policy == "matched_fixed80_recall"
                    else target_metric
                ),
                "calibration_recall_at_10": (
                    float(calibration_info["calibrated_calibration_recall"])
                    if policy == "matched_fixed80_recall"
                    else float(calibration_info["calibration_recall_at_10"])
                ),
                "calibration_allk_coverage": (
                    ""
                    if policy == "matched_fixed80_recall"
                    else float(calibration_info["calibration_allk_coverage"])
                ),
                "test_recall_at_10": float(result["recall_at_k"]),
                "test_qps": float(result["qps"]),
                "rerank_mean": float(result["adaptive_rerank_count_mean"]),
                "rerank_p95": float(result["adaptive_rerank_count_p95"]),
                "rerank_p99": float(result["adaptive_rerank_count_p99"]),
                "rerank_max": float(result["adaptive_rerank_count_max"]),
                "fp_4kb_reads_per_query": float(result["direct_requests_mean"])
                / split.test_rows,
                "fp_submitted_bytes_per_query": float(
                    result["direct_submitted_bytes_mean"]
                )
                / split.test_rows,
                "rounds": args.rounds,
                "search_threads": 1,
                "fp_rerank_source": "O_DIRECT io_uring",
            }
        )

    summary = RESULT_DIR / f"comparison.{args.target_mode}.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        RESULT_DIR / f"run_manifest.{args.target_mode}.json",
        {
            "dataset": dataset.key,
            "target": target,
            "matched_fixed80_calibration": matched,
            "efSearch": args.ef,
            "rounds": args.rounds,
            "search_threads": 1,
            "fp_sidecar": str(args.fp_sidecar),
            "summary": str(summary),
        },
    )
    print(f"WROTE {summary}")
    for row in rows:
        print(
            f"{row['policy']}: recall={row['test_recall_at_10']:.6f}, "
            f"QPS={row['test_qps']:.1f}, rerank mean/p95/p99/max="
            f"{row['rerank_mean']:.1f}/{row['rerank_p95']:.0f}/"
            f"{row['rerank_p99']:.0f}/{row['rerank_max']:.0f}"
        )


if __name__ == "__main__":
    main()
