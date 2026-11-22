#!/usr/bin/env python3
"""Run the original calibrated ADC-BQ protocol in the DiskVecLab harness.

The original experiment fits one threshold at reference efSearch=256 so that
its calibration CandidateRecall matches fixed FP rerank@80.  That same tau is
then reused at every search ef, while the threshold produces a query-dependent
FP refinement count R(q).  This is intentionally separate from calibrating tau
to an externally supplied 0.95/0.99 target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BBQ_DIR = REPO / "src/BBQ"
sys.path.insert(0, str(BBQ_DIR))
import run_calibrated_adcbq as core  # noqa: E402


DEFAULT_MANIFEST = BBQ_DIR / "manifest.nq_msmarco_qwen1024.json"
DEFAULT_EFS = (80, 128, 256, 512, 1024)


def fit_tau_to_fixed_r(
    ids: np.ndarray,
    distances: np.ndarray,
    gt: np.ndarray,
    scales: np.ndarray,
    k: int,
    fixed_r: int,
) -> dict[str, float | int]:
    """Reproduce the original fixed-R-matched single-tau calibration."""
    nq, ef = ids.shape
    if fixed_r > ef or gt.shape[0] != nq or gt.shape[1] < k:
        raise core.CalibrationError("incompatible fixed-R calibration inputs")
    requirements = np.full((nq, k), np.inf, dtype=np.float64)
    ranks = np.full((nq, k), ef + 1, dtype=np.int32)
    for query_id in range(nq):
        positions = {
            int(doc_id): rank
            for rank, doc_id in enumerate(ids[query_id])
            if doc_id >= 0
        }
        anchor = float(distances[query_id, k - 1])
        scale = float(scales[query_id])
        for gold_index, gold_id in enumerate(gt[query_id, :k]):
            rank = positions.get(int(gold_id))
            if rank is None:
                continue
            ranks[query_id, gold_index] = rank + 1
            requirements[query_id, gold_index] = (
                float(distances[query_id, rank]) - anchor
            ) / scale

    fixed_hits = int((ranks <= fixed_r).sum())
    finite = np.sort(requirements[np.isfinite(requirements)])
    if fixed_hits <= 0 or fixed_hits > len(finite):
        raise core.CalibrationError(
            f"cannot fit tau: fixed_hits={fixed_hits}, finite={len(finite)}"
        )
    tau = float(finite[fixed_hits - 1])
    cutoffs = distances[:, k - 1] + tau * scales
    counts = (distances <= cutoffs[:, None]).sum(axis=1).clip(min=k, max=ef)
    calibrated_hits = ranks <= counts[:, None]
    return {
        "tau": tau,
        "fixed_r": fixed_r,
        "fixed_calibration_recall": fixed_hits / ranks.size,
        "calibrated_calibration_recall": float(calibrated_hits.mean()),
        "candidate_count_mean": float(counts.mean()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_p95": float(np.percentile(counts, 95)),
        "candidate_count_p99": float(np.percentile(counts, 99)),
        "candidate_count_max": int(counts.max()),
        "calibration_queries": nq,
        "reference_ef": ef,
    }


def benchmark(
    binary: Path,
    bbq_root: Path,
    dataset: core.DatasetSpec,
    split: core.SplitFiles,
    ef: int,
    tau: float,
    rounds: int,
    args: argparse.Namespace,
) -> Path:
    output = dataset.output_dir / f"test.fixedr{args.fixed_r}.refef{args.reference_ef}.ef{ef}.csv"
    if output.is_file() and not args.force:
        return output
    command = core.common_command(
        binary,
        dataset,
        split.test_query,
        split.test_gt,
        output,
        ef,
        rounds,
        args,
    )
    command += [
        "--rerank-n", str(ef),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", "odirect-uring",
        "--fp-rerank-norms", "onfetch",
        "--calibrated-threshold-rerank-tau", repr(tau),
        "--fp-rerank-path", str(dataset.fp_sidecar),
        "--fp-rerank-qd", str(ef),
        "--fp-rerank-overlap-queries",
    ]
    core.run_command(command, args.search_threads, bbq_root)
    return output


def result_row(
    dataset: core.DatasetSpec,
    calibration: dict[str, float | int],
    measured: dict[str, str],
) -> dict[str, Any]:
    qps = float(measured["qps"])
    return {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method": "BBQ original fixedR80-matched calibration",
        "calibration_protocol": "single tau at reference ef matching fixed rerank R",
        "reference_ef": calibration["reference_ef"],
        "fixed_r": calibration["fixed_r"],
        "calibration_tau": calibration["tau"],
        "fixed_calibration_recall": calibration["fixed_calibration_recall"],
        "calibrated_calibration_recall": calibration["calibrated_calibration_recall"],
        "calibration_r_mean": calibration["candidate_count_mean"],
        "calibration_r_p95": calibration["candidate_count_p95"],
        "calibration_r_p99": calibration["candidate_count_p99"],
        "calibration_r_max": calibration["candidate_count_max"],
        "efSearch": measured["efSearch"],
        "recall_at_10": measured["recall_at_k"],
        "qps": measured["qps"],
        "latency_mean_ms": 1000.0 / qps if qps > 0 else math.nan,
        "fp_vectors_mean": measured["adaptive_rerank_count_mean"],
        "fp_vectors_p95": measured["adaptive_rerank_count_p95"],
        "fp_vectors_p99": measured["adaptive_rerank_count_p99"],
        "fp_vectors_max": measured["adaptive_rerank_count_max"],
        "fp_read_bytes_per_query": measured["rerank_read_bytes_per_query"],
        "fp_rerank_source": measured["fp_rerank_source"],
        "fp_rerank_overlap_queries": measured["fp_rerank_overlap_queries"],
        "threads": measured["search_threads"],
        "rounds": measured["rounds"],
        "nq": measured["nq"],
        "index_path": measured["index_path"],
        "fp_rerank_path": measured["fp_rerank_path"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--reference-ef", type=int, default=256)
    parser.add_argument("--fixed-r", type=int, default=80)
    parser.add_argument("--calibration-queries", type=int, default=1000)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.reference_ef < args.fixed_r or args.search_threads != 1:
        parser.error("reference ef must cover fixed R and this benchmark requires T=1")

    args.manifest = args.manifest.expanduser().resolve()
    args.bbq_root = (REPO.parent / "RaBitQ-Library").resolve()
    args.binary = args.bbq_root / "experiment/adcbq_optimized/faiss_adcbq_hnsw_calibrated"
    args.rebuild_binary = False
    args.rebuild_index = False
    args.fp_source = "odirect-uring"
    args.overlap_queries = True

    output_root, datasets = core.load_manifest(args.manifest)
    if args.datasets:
        requested = set(args.datasets)
        datasets = [dataset for dataset in datasets if dataset.key in requested]
        missing = requested - {dataset.key for dataset in datasets}
        if missing:
            raise core.ConfigurationError(f"unknown datasets: {sorted(missing)}")
    core.ensure_binary(args.bbq_root, args.binary, False)

    rows: list[dict[str, Any]] = []
    for original in datasets:
        core.validate_dataset(original)
        work = original.output_dir / f"original_fixedr{args.fixed_r}_refef{args.reference_ef}"
        dataset = replace(original, output_dir=work)
        split = core.make_split(
            dataset, args.calibration_queries, args.max_test_queries, args.seed
        )
        calibration_gt = np.memmap(
            split.calibration_gt,
            dtype=np.int32,
            mode="r",
            offset=8,
            shape=(split.calibration_rows, split.gt_dim),
        )
        scales = core.int16_query_scales(split.calibration_query)
        candidates_path = core.generate_candidates(
            args.binary,
            args.bbq_root,
            dataset,
            split,
            args.reference_ef,
            args,
            False,
        )
        ids, distances = core.load_candidates(
            candidates_path, split.calibration_rows, args.reference_ef
        )
        calibration = fit_tau_to_fixed_r(
            ids,
            distances,
            np.asarray(calibration_gt),
            scales,
            args.k,
            args.fixed_r,
        )
        calibration_path = work / "calibration.json"
        calibration_path.write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"CALIBRATED {dataset.key}: tau={calibration['tau']:.12g}, "
            f"fixedR{args.fixed_r} recall={calibration['fixed_calibration_recall']:.6f}, "
            f"R mean/p95/p99/max={calibration['candidate_count_mean']:.1f}/"
            f"{calibration['candidate_count_p95']:.0f}/"
            f"{calibration['candidate_count_p99']:.0f}/"
            f"{calibration['candidate_count_max']}",
            flush=True,
        )
        candidates_path.unlink(missing_ok=True)
        for ef in args.efs:
            measured_path = benchmark(
                args.binary,
                args.bbq_root,
                dataset,
                split,
                ef,
                float(calibration["tau"]),
                args.rounds,
                args,
            )
            rows.append(
                result_row(dataset, calibration, core.read_single_row(measured_path))
            )
        core.write_csv(work / "results.csv", rows[-len(args.efs):])

    destination = output_root / "bbq_original_fixedr80_results.csv"
    core.write_csv(destination, rows)
    print(f"WROTE {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
