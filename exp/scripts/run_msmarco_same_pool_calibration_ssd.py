#!/usr/bin/env python3
"""Link the ADC-BQ mean-Recall calibration guarantee to measured SSD savings.

The experiment keeps the encoder, graph, scorer, efSearch, returned candidate
pool, and query split identical between a calibration-tuned fixed-R policy and
the randomized-index conformal threshold policy.  Candidate generation is
performed once for all queries; repeated split calibration is then offline.
Selected finite-threshold splits are replayed end to end with strict T=1
O_DIRECT io_uring exact-FP reranking and query-level latency metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

import run_adcbq_calibrated_threshold_overlap as calibration
import run_v051_ssd_odirect_multidataset as datasets


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
BUILD_SCRIPT = REPO / "experiment/adcbq_optimized/build.sh"
BINARY = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_same_pool_calibration"
TRACE_ROOT = REPO / "experiment/msmarco_same_pool_calibration_ssd"
RESULT_ROOT = REPO / "experiment/msmarco_same_pool_mean_recall_calibration_ssd"
DEFAULT_EXACT_GT = (
    REPO
    / "experiment/msmarco_qwen1024_calibration_analysis/exact_cosine_gt/"
    "msmarco_qwen1024.exact_cosine.k10.ibin"
)
DEFAULT_FP_SIDECAR = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp/"
    "msmarco_qwen1024.fp32.header4096.fbin"
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def ensure_binary() -> None:
    if BINARY.exists() and BINARY.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
        return
    env = {**os.environ, "OUT": str(BINARY)}
    subprocess.run([str(BUILD_SCRIPT)], cwd=REPO, env=env, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def common_command(
    dataset: datasets.Dataset,
    query: Path,
    gt: Path,
    output: Path,
    ef: int,
    rounds: int,
    max_queries: int = 0,
) -> list[str]:
    return [
        str(BINARY),
        "--dataset", dataset.key,
        "--train", str(dataset.train),
        "--query", str(query),
        "--gt", str(gt),
        "--index", str(dataset.adcbq_index),
        "--out", str(output),
        "--m", "24",
        "--ef-construction", "200",
        "--k", "10",
        "--max-queries", str(max_queries),
        "--rounds", str(rounds),
        "--build-threads", "1",
        "--search-threads", "1",
        "--adc-mode", "int16",
        "--lut-mode", "nibble",
        "--graph-build", "fp",
        "--distance-mode", "adcbq",
        "--search-impl", "inline",
        "--code-layout", "block32",
        "--efs", str(ef),
    ]


def run(command: list[str], cpu: int) -> None:
    full = ["taskset", "-c", str(cpu), *command]
    print("RUN " + " ".join(full), flush=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    subprocess.run(full, cwd=REPO, env=env, check=True)


def candidate_paths(trace_root: Path, ef: int, pool: int) -> tuple[Path, Path]:
    root = trace_root / f"ef{ef}_pool{pool}"
    return root / "all_queries.candidates.summary.csv", root / "all_queries.candidates.csv"


def pool_label_paths(trace_root: Path, ef: int, pool: int) -> tuple[Path, Path]:
    root = trace_root / f"ef{ef}_pool{pool}"
    return root / "full_pool_exact.summary.csv", root / "full_pool_exact.labels.csv"


def collect_candidates(
    dataset: datasets.Dataset,
    trace_root: Path,
    ef: int,
    pool: int,
    cpu: int,
    force: bool,
) -> Path:
    summary, candidates = candidate_paths(trace_root, ef, pool)
    nq, _ = calibration.matrix_header(dataset.query)
    expected_lines = nq * pool + 1
    if not force and summary.exists() and candidates.exists():
        with candidates.open(encoding="utf-8") as handle:
            if sum(1 for _ in handle) == expected_lines:
                print(f"REUSE {candidates}", flush=True)
                return candidates
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = common_command(dataset, dataset.query, dataset.gt, summary, ef, 1)
    command += [
        "--candidate-pool-size", str(pool),
        "--rerank-n", str(pool),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out", str(candidates),
    ]
    run(command, cpu)
    return candidates


def collect_full_pool_exact_labels(
    dataset: datasets.Dataset,
    trace_root: Path,
    ef: int,
    pool: int,
    cpu: int,
    force: bool,
) -> Path:
    summary, labels = pool_label_paths(trace_root, ef, pool)
    nq, _ = calibration.matrix_header(dataset.query)
    expected_lines = nq * 10 + 1
    if not force and summary.exists() and labels.exists():
        with labels.open(encoding="utf-8") as handle:
            if sum(1 for _ in handle) == expected_lines:
                print(f"REUSE {labels}", flush=True)
                return labels
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = common_command(dataset, dataset.query, dataset.gt, summary, ef, 1)
    command += [
        "--candidate-pool-size", str(pool),
        "--rerank-n", str(pool),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", "ram",
        "--fp-rerank-norms", "onfetch",
        "--final-labels-out", str(labels),
    ]
    run(command, cpu)
    return labels


def load_candidates(path: Path, nq: int, pool: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((nq, pool), -1, dtype=np.int32)
    distances = np.full((nq, pool), np.inf, dtype=np.float32)
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            if not (0 <= qi < nq and 0 <= rank < pool):
                raise ValueError(f"bad candidate coordinate q={qi}, rank={rank}")
            ids[qi, rank] = int(row["id"])
            distances[qi, rank] = float(row["distance"])
            seen += 1
    if seen != nq * pool:
        raise ValueError(f"expected {nq * pool} candidate rows, got {seen}")
    return ids, distances


def load_final_labels(path: Path, nq: int) -> np.ndarray:
    labels = np.full((nq, 10), -1, dtype=np.int32)
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            if not (0 <= qi < nq and 0 <= rank < 10):
                raise ValueError(f"bad final-label coordinate q={qi}, rank={rank}")
            labels[qi, rank] = int(row["id"])
            seen += 1
    if seen != nq * 10:
        raise ValueError(f"expected {nq * 10} final labels, got {seen}")
    return labels


def query_scales(query_path: Path) -> np.ndarray:
    nq, dim = calibration.matrix_header(query_path)
    queries = np.memmap(
        query_path, dtype=np.float32, mode="r", offset=8, shape=(nq, dim)
    )
    block_l1 = np.abs(np.asarray(queries)).reshape(nq, dim // 4, 4).sum(axis=2)
    max_abs = block_l1.max(axis=1)
    return np.divide(
        32767.0,
        max_abs,
        out=np.ones_like(max_abs, dtype=np.float32),
        where=max_abs > 0,
    )


def ground_truth_ranks(
    ids: np.ndarray,
    full_pool_exact: np.ndarray,
    gt10: np.ndarray,
    pool: int,
) -> tuple[np.ndarray, np.ndarray]:
    nq = ids.shape[0]
    ranks = np.full((nq, 10), pool + 1, dtype=np.int32)
    membership_ranks = np.full((nq, 10), pool + 1, dtype=np.int32)
    # Work in query chunks to avoid a pool x 10 comparison tensor for all rows.
    for begin in range(0, nq, 256):
        end = min(nq, begin + 256)
        matches = ids[begin:end, :, None] == gt10[begin:end, None, :]
        exact_pool_hit = (
            full_pool_exact[begin:end, :, None] == gt10[begin:end, None, :]
        ).any(axis=1)
        found = matches.any(axis=1) & exact_pool_hit
        first = matches.argmax(axis=1) + 1
        membership_ranks[begin:end] = np.where(matches.any(axis=1), first, pool + 1)
        ranks[begin:end] = np.where(found, first, pool + 1)
    return ranks, membership_ranks


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def split_indices(nq: int, calibration_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(seed).permutation(nq)
    return permutation[:calibration_rows], permutation[calibration_rows:]


def make_physical_split(
    dataset: datasets.Dataset,
    result_root: Path,
    calibration_rows: int,
    seed: int,
) -> calibration.SplitFiles:
    """Materialize a split under this experiment's exact-GT result root."""
    nq, dim = calibration.matrix_header(dataset.query)
    ng, gt_dim = calibration.matrix_header(dataset.gt)
    if nq != ng:
        raise ValueError(f"query/GT row mismatch: {nq} != {ng}")
    cal_ids, test_ids = split_indices(nq, calibration_rows, seed)
    tag = f"seed{seed}.cal{calibration_rows}.test{len(test_ids)}"
    root = result_root / "splits"
    paths = calibration.SplitFiles(
        root / f"{tag}.cal.query.fbin",
        root / f"{tag}.cal.gt.ibin",
        root / f"{tag}.test.query.fbin",
        root / f"{tag}.test.gt.ibin",
        len(cal_ids),
        len(test_ids),
        dim,
        gt_dim,
    )
    expected = {
        paths.calibration_query: (len(cal_ids), dim),
        paths.calibration_gt: (len(cal_ids), gt_dim),
        paths.test_query: (len(test_ids), dim),
        paths.test_gt: (len(test_ids), gt_dim),
    }
    if all(
        path.exists() and calibration.matrix_header(path) == shape
        for path, shape in expected.items()
    ):
        return paths
    queries = np.memmap(
        dataset.query, dtype=np.float32, mode="r", offset=8, shape=(nq, dim)
    )
    gt = np.memmap(
        dataset.gt, dtype=np.int32, mode="r", offset=8, shape=(nq, gt_dim)
    )
    calibration.write_matrix(paths.calibration_query, np.asarray(queries[cal_ids]))
    calibration.write_matrix(paths.calibration_gt, np.asarray(gt[cal_ids]))
    calibration.write_matrix(paths.test_query, np.asarray(queries[test_ids]))
    calibration.write_matrix(paths.test_gt, np.asarray(gt[test_ids]))
    return paths


def evaluate_splits(
    ranks: np.ndarray,
    membership_ranks: np.ndarray,
    distances: np.ndarray,
    scales: np.ndarray,
    ef: int,
    pool: int,
    target: float,
    calibration_rows: int,
    split_seeds: list[int],
    randomized_seed_offset: int,
    fixed_r_override: int,
) -> list[dict[str, object]]:
    nq = ranks.shape[0]
    requirements = np.full((nq, 10), np.inf, dtype=np.float64)
    present = ranks <= pool
    rows = np.arange(nq)[:, None]
    gold = np.arange(10)[None, :]
    present_rows, present_gold = np.nonzero(present)
    candidate_rank = ranks[present_rows, present_gold] - 1
    requirements[present_rows, present_gold] = (
        distances[present_rows, candidate_rank] - distances[present_rows, 9]
    ) / scales[present_rows]
    pool_recall = present.mean(axis=1)
    pool_all10 = present.all(axis=1)
    membership_present = membership_ranks <= pool
    membership_recall = membership_present.mean(axis=1)
    membership_all10 = membership_present.all(axis=1)
    order = int(math.ceil((calibration_rows + 1) * target))
    if order > calibration_rows:
        raise ValueError(
            f"target {target} is finite-sample infeasible with m={calibration_rows}"
        )

    output: list[dict[str, object]] = []
    for seed in split_seeds:
        cal, test = split_indices(nq, calibration_rows, seed)
        cal_curve = (ranks[cal, :, None] <= np.arange(1, pool + 1)).mean(axis=(0, 1))
        if fixed_r_override > 0:
            feasible_fixed = fixed_r_override <= pool
            fixed_r = fixed_r_override
            fixed_policy = "fixed_r"
            fixed_r_source = "predeclared_override"
        else:
            feasible_fixed = bool(cal_curve[-1] >= target)
            fixed_r = (
                int(np.searchsorted(cal_curve, target, side="left") + 1)
                if feasible_fixed
                else 0
            )
            fixed_policy = "tuned_fixed_r"
            fixed_r_source = "calibration_minimum_for_target"

        randomized_seed = randomized_seed_offset + seed
        rng = np.random.default_rng(randomized_seed)
        sampled_gold = rng.integers(0, 10, size=calibration_rows)
        sampled_requirements = requirements[cal, sampled_gold]
        sampled_misses = int(np.isinf(sampled_requirements).sum())
        tau = float(np.sort(sampled_requirements)[order - 1])
        tau_infinite = not np.isfinite(tau)

        common = {
            "dataset": "msmarco_qwen1024",
            "efSearch": ef,
            "candidate_pool_size": pool,
            "target_mean_recall_at_10": target,
            "calibration_rows": calibration_rows,
            "test_rows": len(test),
            "split_seed": seed,
            "randomized_index_seed": randomized_seed,
            "conformal_order": order,
            "calibration_pool_recall_ceiling": float(pool_recall[cal].mean()),
            "test_pool_recall_ceiling": float(pool_recall[test].mean()),
            "calibration_candidate_membership_ceiling": float(
                membership_recall[cal].mean()
            ),
            "test_candidate_membership_ceiling": float(
                membership_recall[test].mean()
            ),
            "calibration_pool_all10_coverage": float(pool_all10[cal].mean()),
            "test_pool_all10_coverage": float(pool_all10[test].mean()),
            "calibration_candidate_all10_coverage": float(
                membership_all10[cal].mean()
            ),
            "test_candidate_all10_coverage": float(
                membership_all10[test].mean()
            ),
            "test_gt_item_miss_fraction": float(1.0 - pool_recall[test].mean()),
            "test_query_any_gt_miss_fraction": float(1.0 - pool_all10[test].mean()),
            "test_candidate_gt_item_miss_fraction": float(
                1.0 - membership_recall[test].mean()
            ),
            "test_candidate_query_any_gt_miss_fraction": float(
                1.0 - membership_all10[test].mean()
            ),
            "sampled_pool_misses": sampled_misses,
            "tau_infinite": tau_infinite,
            "fallback_required": tau_infinite,
            "fixed_r_source": fixed_r_source,
        }

        if feasible_fixed:
            fixed_cal_recall = float((ranks[cal] <= fixed_r).mean())
            fixed_test_per_query = (ranks[test] <= fixed_r).mean(axis=1)
            output.append(
                {
                    **common,
                    "policy": fixed_policy,
                    "fixed_r": fixed_r,
                    "tau": "",
                    "calibration_recall_at_10": fixed_cal_recall,
                    "test_recall_at_10": float(fixed_test_per_query.mean()),
                    "target_met_on_test": bool(fixed_test_per_query.mean() >= target),
                    "mean_fp_reads": float(fixed_r),
                    "p99_fp_reads": float(fixed_r),
                }
            )
        else:
            output.append(
                {
                    **common,
                    "policy": fixed_policy,
                    "fixed_r": "",
                    "tau": "",
                    "calibration_recall_at_10": float(cal_curve[-1]),
                    "test_recall_at_10": "",
                    "target_met_on_test": False,
                    "mean_fp_reads": "",
                    "p99_fp_reads": "",
                }
            )

        if tau_infinite:
            output.append(
                {
                    **common,
                    "policy": "calibrated_threshold",
                    "fixed_r": "",
                    "tau": "inf",
                    "calibration_recall_at_10": "",
                    "test_recall_at_10": "",
                    "target_met_on_test": "",
                    "mean_fp_reads": "",
                    "p99_fp_reads": "",
                }
            )
            continue

        cutoffs = distances[:, 9] + tau * scales
        counts = np.maximum(10, (distances <= cutoffs[:, None]).sum(axis=1))
        counts = np.minimum(counts, pool).astype(np.int32)
        calibrated_hit = ranks <= counts[:, None]
        test_per_query = calibrated_hit[test].mean(axis=1)
        output.append(
            {
                **common,
                "policy": "calibrated_threshold",
                "fixed_r": "",
                "tau": tau,
                "calibration_recall_at_10": float(calibrated_hit[cal].mean()),
                "test_recall_at_10": float(test_per_query.mean()),
                "target_met_on_test": bool(test_per_query.mean() >= target),
                "mean_fp_reads": float(counts[test].mean()),
                "p99_fp_reads": percentile(counts[test], 99),
            }
        )
    return output


def aggregate_split_rows(
    rows: list[dict[str, object]], database_rows: int
) -> list[dict[str, object]]:
    tau_inf_fraction = float(
        np.mean(
            [
                bool(row["tau_infinite"])
                for row in rows
                if row["policy"] == "calibrated_threshold"
            ]
        )
    )
    output: list[dict[str, object]] = []
    policies = [
        policy
        for policy in ("fixed_r", "tuned_fixed_r", "calibrated_threshold")
        if any(row["policy"] == policy for row in rows)
    ]
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        finite = [row for row in selected if row["test_recall_at_10"] != ""]
        recalls = np.asarray([float(row["test_recall_at_10"]) for row in finite])
        mean_reads = np.asarray([float(row["mean_fp_reads"]) for row in finite])
        p99_reads = np.asarray([float(row["p99_fp_reads"]) for row in finite])
        fallback_splits = len(selected) - len(finite)
        fallback_fraction = fallback_splits / len(selected)
        fallback_assisted_recall = (
            float(recalls.sum()) + float(fallback_splits)
        ) / len(selected)
        fallback_assisted_reads = (
            float(mean_reads.sum()) + float(fallback_splits * database_rows)
        ) / len(selected)
        output.append(
            {
                "dataset": selected[0]["dataset"],
                "efSearch": selected[0]["efSearch"],
                "candidate_pool_size": selected[0]["candidate_pool_size"],
                "policy": policy,
                "splits": len(selected),
                "finite_policy_splits": len(finite),
                "exact_fallback_splits": fallback_splits,
                "exact_fallback_split_fraction": fallback_fraction,
                "target_mean_recall_at_10": selected[0]["target_mean_recall_at_10"],
                "pool_recall_ceiling_mean": float(
                    np.mean([float(row["test_pool_recall_ceiling"]) for row in selected])
                ),
                "candidate_membership_ceiling_mean": float(
                    np.mean(
                        [
                            float(row["test_candidate_membership_ceiling"])
                            for row in selected
                        ]
                    )
                ),
                "candidate_gt_item_miss_fraction_mean": float(
                    np.mean(
                        [
                            float(row["test_candidate_gt_item_miss_fraction"])
                            for row in selected
                        ]
                    )
                ),
                "candidate_all10_coverage_mean": float(
                    np.mean(
                        [
                            float(row["test_candidate_all10_coverage"])
                            for row in selected
                        ]
                    )
                ),
                "candidate_query_any_gt_miss_fraction_mean": float(
                    np.mean(
                        [
                            float(row["test_candidate_query_any_gt_miss_fraction"])
                            for row in selected
                        ]
                    )
                ),
                "pool_gt_item_miss_fraction_mean": float(
                    np.mean([float(row["test_gt_item_miss_fraction"]) for row in selected])
                ),
                "pool_all10_coverage_mean": float(
                    np.mean([float(row["test_pool_all10_coverage"]) for row in selected])
                ),
                "pool_query_any_gt_miss_fraction_mean": float(
                    np.mean(
                        [float(row["test_query_any_gt_miss_fraction"]) for row in selected]
                    )
                ),
                "tau_infinite_split_fraction": tau_inf_fraction,
                "test_recall_mean": float(recalls.mean()) if len(recalls) else "",
                "test_recall_std": float(recalls.std()) if len(recalls) else "",
                "test_recall_worst_split": float(recalls.min()) if len(recalls) else "",
                "target_success_fraction_finite": float(
                    np.mean([bool(row["target_met_on_test"]) for row in finite])
                ) if finite else "",
                "mean_fp_reads_across_splits": float(mean_reads.mean()) if len(mean_reads) else "",
                "mean_p99_fp_reads_across_splits": float(p99_reads.mean()) if len(p99_reads) else "",
                "fallback_assisted_recall_mean": fallback_assisted_recall,
                "fallback_assisted_mean_fp_reads": fallback_assisted_reads,
                "fallback_exact_fp_reads_per_query": database_rows,
            }
        )
    return output


def physical_benchmark(
    dataset: datasets.Dataset,
    fp_sidecar: Path,
    result_root: Path,
    split_rows: list[dict[str, object]],
    ef: int,
    pool: int,
    rounds: int,
    calibration_rows: int,
    physical_seeds: int,
    cpu: int,
    force: bool,
) -> list[dict[str, object]]:
    calibrated_by_seed = {
        int(row["split_seed"]): row
        for row in split_rows
        if row["policy"] == "calibrated_threshold" and row["tau"] not in ("", "inf")
    }
    fixed_by_seed = {
        int(row["split_seed"]): row
        for row in split_rows
        if row["policy"] in ("fixed_r", "tuned_fixed_r")
        and row["fixed_r"] != ""
    }
    seeds = sorted(set(calibrated_by_seed) & set(fixed_by_seed))[:physical_seeds]
    outputs: list[dict[str, object]] = []
    for seed in seeds:
        split = make_physical_split(dataset, result_root, calibration_rows, seed)
        fixed_policy = str(fixed_by_seed[seed]["policy"])
        policies = (
            (fixed_policy, int(fixed_by_seed[seed]["fixed_r"]), None),
            ("calibrated_threshold", pool, float(calibrated_by_seed[seed]["tau"])),
        )
        for policy, rerank_n, tau in policies:
            root = result_root / f"ef{ef}_pool{pool}" / "physical" / f"seed{seed}"
            summary_path = root / f"{policy}.summary.csv"
            metrics_path = root / f"{policy}.per_query.csv"
            expected_metric_rows = split.test_rows * rounds
            complete = summary_path.exists() and metrics_path.exists()
            if complete:
                with metrics_path.open(newline="", encoding="utf-8") as handle:
                    existing_metrics = list(csv.DictReader(handle))
                complete = (
                    len(existing_metrics) == expected_metric_rows
                    and {row.get("latency_mode", "") for row in existing_metrics}
                    == {"strict_query_wall"}
                )
            if force or not complete:
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                command = common_command(
                    dataset, split.test_query, split.test_gt, summary_path, ef, rounds
                )
                command += [
                    "--candidate-pool-size", str(pool),
                    "--rerank-n", str(rerank_n),
                    "--rerank-dot", "fp32",
                    "--fp-rerank-source", "odirect-uring",
                    "--fp-rerank-norms", "onfetch",
                    "--fp-rerank-path", str(fp_sidecar),
                    "--fp-rerank-qd", str(pool),
                    "--strict-query-e2e",
                    "--per-query-metrics-out", str(metrics_path),
                ]
                if tau is not None:
                    command += ["--calibrated-threshold-rerank-tau", repr(tau)]
                run(command, cpu)

            summary = read_one(summary_path)
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            latency_modes = {row.get("latency_mode", "") for row in metrics}
            if latency_modes != {"strict_query_wall"}:
                raise RuntimeError(
                    f"expected strict query-wall metrics in {metrics_path}, "
                    f"found {sorted(latency_modes)}; rerun with --force-physical"
                )
            selected_reads = np.asarray(
                [float(row["selected_fp_reads"]) for row in metrics]
            )
            io_wait = np.asarray([float(row["io_wait_us"]) for row in metrics])
            e2e = np.asarray([float(row["end_to_end_us"]) for row in metrics])
            outputs.append(
                {
                    "dataset": dataset.key,
                    "efSearch": ef,
                    "candidate_pool_size": pool,
                    "split_seed": seed,
                    "policy": policy,
                    "fixed_r": rerank_n if policy != "calibrated_threshold" else "",
                    "tau": tau if tau is not None else "",
                    "rounds": rounds,
                    "test_queries": split.test_rows,
                    "recall_at_10": float(summary["recall_at_k"]),
                    "qps": float(summary["qps"]),
                    "mean_fp_reads": float(selected_reads.mean()),
                    "p99_fp_reads": percentile(selected_reads, 99),
                    "mean_io_wait_us": float(io_wait.mean()),
                    "p99_io_wait_us": percentile(io_wait, 99),
                    "mean_end_to_end_us": float(e2e.mean()),
                    "p99_end_to_end_us": percentile(e2e, 99),
                    "summary_csv": str(summary_path),
                    "per_query_csv": str(metrics_path),
                }
            )
    return outputs


def aggregate_physical_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    policy_mean_reads: dict[str, float] = {}
    fixed_policies = sorted(
        {
            str(row["policy"])
            for row in rows
            if row["policy"] in ("fixed_r", "tuned_fixed_r")
        }
    )
    for policy in (*fixed_policies, "calibrated_threshold"):
        selected = [row for row in rows if row["policy"] == policy]
        if not selected:
            continue
        def values(field: str) -> np.ndarray:
            return np.asarray([float(row[field]) for row in selected], dtype=np.float64)

        recalls = values("recall_at_10")
        reads = values("mean_fp_reads")
        p99_reads = values("p99_fp_reads")
        p99_io = values("p99_io_wait_us")
        p99_e2e = values("p99_end_to_end_us")
        policy_mean_reads[policy] = float(reads.mean())
        output.append(
            {
                "dataset": selected[0]["dataset"],
                "efSearch": selected[0]["efSearch"],
                "candidate_pool_size": selected[0]["candidate_pool_size"],
                "policy": policy,
                "physical_splits": len(selected),
                "rounds_per_split": selected[0]["rounds"],
                "recall_mean": float(recalls.mean()),
                "recall_std": float(recalls.std()),
                "recall_worst_split": float(recalls.min()),
                "target_0p99_success_fraction": float((recalls >= 0.99).mean()),
                "qps_mean": float(values("qps").mean()),
                "fp_reads_mean": float(reads.mean()),
                "p99_fp_reads_mean": float(p99_reads.mean()),
                "p99_fp_reads_worst_split": float(p99_reads.max()),
                "io_wait_us_mean": float(values("mean_io_wait_us").mean()),
                "p99_io_wait_us_mean": float(p99_io.mean()),
                "p99_io_wait_us_worst_split": float(p99_io.max()),
                "end_to_end_us_mean": float(values("mean_end_to_end_us").mean()),
                "p99_end_to_end_us_mean": float(p99_e2e.mean()),
                "p99_end_to_end_us_worst_split": float(p99_e2e.max()),
            }
        )
    fixed_key = fixed_policies[0] if fixed_policies else None
    if fixed_key is not None and "calibrated_threshold" in policy_mean_reads:
        reduction = 1.0 - (
            policy_mean_reads["calibrated_threshold"] /
            policy_mean_reads[fixed_key]
        )
        for row in output:
            row["calibrated_read_reduction_vs_fixed"] = (
                reduction if row["policy"] == "calibrated_threshold" else 0.0
            )
    return output


def reviewer_table_rows(
    split_aggregate: list[dict[str, object]],
    physical_aggregate: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Join the repeated-split guarantee and bounded SSD measurements.

    Physical rows condition on a finite calibrated threshold; exact fallback
    latency is intentionally not imputed.  The fallback-assisted read column
    in the repeated-split aggregate charges a full database scan instead.
    """
    physical_by_policy = {
        str(row["policy"]): row for row in physical_aggregate
    }
    output: list[dict[str, object]] = []
    for split in split_aggregate:
        physical = physical_by_policy.get(str(split["policy"]), {})
        output.append(
            {
                "dataset": split["dataset"],
                "efSearch": split["efSearch"],
                "candidate_pool_size": split["candidate_pool_size"],
                "policy": split["policy"],
                "target_mean_recall_at_10": split["target_mean_recall_at_10"],
                "repeated_splits": split["splits"],
                "pool_recall_ceiling": split["pool_recall_ceiling_mean"],
                "candidate_membership_ceiling": split[
                    "candidate_membership_ceiling_mean"
                ],
                "candidate_gt_item_miss_fraction": split[
                    "candidate_gt_item_miss_fraction_mean"
                ],
                "candidate_query_any_gt_miss_fraction": split[
                    "candidate_query_any_gt_miss_fraction_mean"
                ],
                "pool_gt_item_miss_fraction": split[
                    "pool_gt_item_miss_fraction_mean"
                ],
                "pool_query_any_gt_miss_fraction": split[
                    "pool_query_any_gt_miss_fraction_mean"
                ],
                "tau_infinite_split_fraction": split[
                    "tau_infinite_split_fraction"
                ],
                "policy_exact_fallback_split_fraction": split[
                    "exact_fallback_split_fraction"
                ],
                "finite_split_recall_mean": split["test_recall_mean"],
                "finite_split_recall_std": split["test_recall_std"],
                "finite_split_recall_worst": split["test_recall_worst_split"],
                "finite_split_target_success_fraction": split[
                    "target_success_fraction_finite"
                ],
                "finite_split_mean_fp_reads": split[
                    "mean_fp_reads_across_splits"
                ],
                "finite_split_mean_p99_fp_reads": split[
                    "mean_p99_fp_reads_across_splits"
                ],
                "fallback_assisted_recall_mean": split[
                    "fallback_assisted_recall_mean"
                ],
                "fallback_assisted_mean_fp_reads": split[
                    "fallback_assisted_mean_fp_reads"
                ],
                "physical_finite_splits": physical.get("physical_splits", ""),
                "physical_recall_mean": physical.get("recall_mean", ""),
                "physical_recall_worst": physical.get("recall_worst_split", ""),
                "physical_qps_mean": physical.get("qps_mean", ""),
                "physical_mean_fp_reads": physical.get("fp_reads_mean", ""),
                "physical_p99_fp_reads_mean": physical.get(
                    "p99_fp_reads_mean", ""
                ),
                "physical_p99_fp_reads_worst": physical.get(
                    "p99_fp_reads_worst_split", ""
                ),
                "physical_mean_io_wait_us": physical.get("io_wait_us_mean", ""),
                "physical_p99_io_wait_us_mean": physical.get(
                    "p99_io_wait_us_mean", ""
                ),
                "physical_p99_end_to_end_us_mean": physical.get(
                    "p99_end_to_end_us_mean", ""
                ),
                "physical_p99_end_to_end_us_worst": physical.get(
                    "p99_end_to_end_us_worst_split", ""
                ),
                "physical_calibrated_read_reduction_vs_fixed": physical.get(
                    "calibrated_read_reduction_vs_fixed", ""
                ),
                "physical_scope": (
                    "finite-threshold strict-query bounded runs; exact fallback latency excluded"
                    if physical else ""
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ef", type=int, default=1024)
    parser.add_argument("--pool", type=int, default=1024)
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument(
        "--fixed-r",
        type=int,
        default=0,
        help="Predeclared fixed refinement budget; 0 tunes the minimum R on calibration data.",
    )
    parser.add_argument("--calibration-rows", type=int, default=1000)
    parser.add_argument("--split-count", type=int, default=100)
    parser.add_argument("--split-seed-start", type=int, default=0)
    parser.add_argument("--randomized-seed-offset", type=int, default=100000)
    parser.add_argument("--physical-seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=20)
    parser.add_argument("--fp-sidecar", type=Path, default=DEFAULT_FP_SIDECAR)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_EXACT_GT)
    parser.add_argument("--trace-root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--force-candidates", action="store_true")
    parser.add_argument("--force-pool-labels", action="store_true")
    parser.add_argument("--force-physical", action="store_true")
    args = parser.parse_args()
    if args.pool < 10 or args.pool > args.ef:
        parser.error("--pool must be in [10, --ef]")
    if args.fixed_r < 0 or args.fixed_r > args.pool:
        parser.error("--fixed-r must be 0 or in [1, --pool]")
    if not (0.0 < args.target < 1.0):
        parser.error("--target must be in (0, 1)")
    if args.split_count <= 0 or args.calibration_rows <= 0:
        parser.error("split count and calibration rows must be positive")

    ensure_binary()
    dataset = replace(
        datasets.DATASETS["msmarco_qwen1024"], gt=args.ground_truth.resolve()
    )
    datasets.validate(dataset)
    if not args.fp_sidecar.is_file():
        raise FileNotFoundError(args.fp_sidecar)
    candidates_path = collect_candidates(
        dataset,
        args.trace_root,
        args.ef,
        args.pool,
        args.cpu,
        args.force_candidates,
    )
    pool_labels_path = collect_full_pool_exact_labels(
        dataset,
        args.trace_root,
        args.ef,
        args.pool,
        args.cpu,
        args.force_pool_labels,
    )
    nq, _ = calibration.matrix_header(dataset.query)
    _, gt_dim = calibration.matrix_header(dataset.gt)
    ids, distances = load_candidates(candidates_path, nq, args.pool)
    full_pool_exact = load_final_labels(pool_labels_path, nq)
    gt = np.memmap(
        dataset.gt, dtype=np.int32, mode="r", offset=8, shape=(nq, gt_dim)
    )[:, :10]
    scales = query_scales(dataset.query)
    ranks, membership_ranks = ground_truth_ranks(
        ids, full_pool_exact, np.asarray(gt), args.pool
    )
    seeds = list(
        range(args.split_seed_start, args.split_seed_start + args.split_count)
    )
    split_rows = evaluate_splits(
        ranks,
        membership_ranks,
        distances,
        scales,
        args.ef,
        args.pool,
        args.target,
        args.calibration_rows,
        seeds,
        args.randomized_seed_offset,
        args.fixed_r,
    )
    root = args.result_root / f"ef{args.ef}_pool{args.pool}"
    split_csv = root / "repeated_split_results.csv"
    aggregate_csv = root / "repeated_split_aggregate.csv"
    write_csv(split_csv, split_rows)
    train_rows, _ = calibration.matrix_header(dataset.train)
    aggregate_rows = aggregate_split_rows(split_rows, train_rows)
    write_csv(aggregate_csv, aggregate_rows)

    physical_rows: list[dict[str, object]] = []
    physical_aggregate_rows: list[dict[str, object]] = []
    if not args.offline_only and args.physical_seeds > 0:
        physical_rows = physical_benchmark(
            dataset,
            args.fp_sidecar,
            args.result_root,
            split_rows,
            args.ef,
            args.pool,
            args.rounds,
            args.calibration_rows,
            args.physical_seeds,
            args.cpu,
            args.force_physical,
        )
        if physical_rows:
            write_csv(root / "physical_summary.csv", physical_rows)
            physical_aggregate_rows = aggregate_physical_rows(physical_rows)
            write_csv(root / "physical_aggregate.csv", physical_aggregate_rows)

    reviewer_csv = root / "reviewer_table.csv"
    write_csv(
        reviewer_csv,
        reviewer_table_rows(aggregate_rows, physical_aggregate_rows),
    )

    manifest = {
        "dataset": dataset.key,
        "encoder": dataset.model,
        "train": str(dataset.train),
        "query": str(dataset.query),
        "ground_truth": str(dataset.gt),
        "ground_truth_sha256": file_sha256(dataset.gt),
        "graph": str(dataset.adcbq_index),
        "graph_sha256": file_sha256(dataset.adcbq_index),
        "fp_sidecar": str(args.fp_sidecar),
        "scorer": "ADC-BQ int16 nibble, optimized set_query",
        "graph_M": 24,
        "graph_efConstruction": 200,
        "efSearch": args.ef,
        "candidate_pool_size": args.pool,
        "target_mean_recall_at_10": args.target,
        "fixed_r": args.fixed_r if args.fixed_r > 0 else None,
        "fixed_r_source": (
            "predeclared_override"
            if args.fixed_r > 0
            else "calibration_minimum_for_target"
        ),
        "calibration_rows": args.calibration_rows,
        "split_seeds": seeds,
        "randomized_index_seed_offset": args.randomized_seed_offset,
        "search_threads": 1,
        "io": (
            "O_DIRECT io_uring; T=1 strict query-local search -> selection -> "
            "read -> exact-score wall latency; no inter-query overlap"
        ),
        "candidate_csv": str(candidates_path),
        "full_pool_exact_labels_csv": str(pool_labels_path),
        "split_results_csv": str(split_csv),
        "aggregate_csv": str(aggregate_csv),
        "physical_summary_csv": str(root / "physical_summary.csv") if physical_rows else None,
        "physical_aggregate_csv": str(root / "physical_aggregate.csv") if physical_rows else None,
        "reviewer_table_csv": str(reviewer_csv),
        "infinite_tau_policy": (
            "bounded configuration is marked infeasible; an exhaustive exact-search "
            "fallback is required and is not included in the bounded SSD timings"
        ),
    }
    write_json(root / "manifest.json", manifest)
    print(f"WROTE {split_csv}", flush=True)
    print(f"WROTE {aggregate_csv}", flush=True)
    for row in aggregate_rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
