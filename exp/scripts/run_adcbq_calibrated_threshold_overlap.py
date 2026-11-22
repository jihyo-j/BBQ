#!/usr/bin/env python3
"""Compare calibrated per-query ADC-BQ reranking with fixed rerank@80.

The calibration rule is the threshold rule from ``adcbq_calibration``:

    R(q) = |{x in the HNSW result list: s(q, x) >= s_k(q) - tau}|

``tau`` is fitted once at a reference efSearch (256 by default), then reused
unchanged at every tested efSearch.  Fixed and calibrated policies are measured
on the same held-out queries.  Both use FP32 rows from a 4 KiB aligned sidecar
through O_DIRECT + io_uring, with query A's I/O overlapped by query B's ADC-HNSW
traversal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v051_ssd_odirect_multidataset as base  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
BUILD_SCRIPT = REPO / "experiment/adcbq_optimized/build.sh"
BIN = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_calibrated"
RESULT_DIR = REPO / "experiment/adcbq_calibrated_threshold_overlap"
ALIGNED_FP_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/adcbq_calibrated_threshold_aligned4k"
)
DEFAULT_DATASETS = (
    "msmarco_qwen1024",
    "nq_qwen06_1024",
    "quora_qwen1024",
)


@dataclass(frozen=True)
class SplitFiles:
    calibration_query: Path
    calibration_gt: Path
    test_query: Path
    test_gt: Path
    calibration_rows: int
    test_rows: int
    dim: int
    gt_dim: int


def matrix_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"bad matrix header: {path}")
    return struct.unpack("ii", raw)


def write_matrix(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("ii", *rows.shape))
        rows.tofile(handle)


def make_split(
    dataset: base.Dataset,
    calibration_rows: int,
    seed: int,
    max_test_queries: int,
) -> SplitFiles:
    nq, dim = matrix_header(dataset.query)
    ng, gt_dim = matrix_header(dataset.gt)
    if nq != ng:
        raise ValueError(f"query/GT row mismatch for {dataset.key}: {nq} != {ng}")
    if calibration_rows <= 0 or calibration_rows >= nq:
        raise ValueError(
            f"calibration rows must be in [1, {nq - 1}] for {dataset.key}"
        )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(nq)
    cal_ids = permutation[:calibration_rows]
    test_ids = permutation[calibration_rows:]
    if max_test_queries:
        test_ids = test_ids[:max_test_queries]

    tag = f"seed{seed}.cal{calibration_rows}.test{len(test_ids)}"
    root = RESULT_DIR / "splits" / dataset.key
    paths = SplitFiles(
        root / f"{tag}.cal.query.fbin",
        root / f"{tag}.cal.gt.ibin",
        root / f"{tag}.test.query.fbin",
        root / f"{tag}.test.gt.ibin",
        calibration_rows,
        len(test_ids),
        dim,
        gt_dim,
    )
    expected = {
        paths.calibration_query: (calibration_rows, dim),
        paths.calibration_gt: (calibration_rows, gt_dim),
        paths.test_query: (len(test_ids), dim),
        paths.test_gt: (len(test_ids), gt_dim),
    }
    if all(path.exists() and matrix_header(path) == shape for path, shape in expected.items()):
        return paths

    queries = np.memmap(
        dataset.query,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(nq, dim),
    )
    gt = np.memmap(
        dataset.gt,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(nq, gt_dim),
    )
    write_matrix(paths.calibration_query, np.asarray(queries[cal_ids]))
    write_matrix(paths.calibration_gt, np.asarray(gt[cal_ids]))
    write_matrix(paths.test_query, np.asarray(queries[test_ids]))
    write_matrix(paths.test_gt, np.asarray(gt[test_ids]))
    return paths


def ensure_binary() -> None:
    if BIN.exists() and BIN.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
        return
    env = os.environ.copy()
    env["OUT"] = str(BIN)
    subprocess.run([str(BUILD_SCRIPT)], cwd=REPO, env=env, check=True)


def common_command(
    dataset: base.Dataset,
    query: Path,
    gt: Path,
    out: Path,
    ef: int,
    rounds: int,
    search_threads: int,
    build_threads: int,
) -> list[str]:
    return [
        str(BIN),
        "--dataset",
        dataset.key,
        "--train",
        str(dataset.train),
        "--query",
        str(query),
        "--gt",
        str(gt),
        "--index",
        str(dataset.adcbq_index),
        "--out",
        str(out),
        "--m",
        "24",
        "--ef-construction",
        "200",
        "--k",
        "10",
        "--max-queries",
        "0",
        "--rounds",
        str(rounds),
        "--build-threads",
        str(build_threads),
        "--search-threads",
        str(search_threads),
        "--adc-mode",
        "int16",
        "--lut-mode",
        "nibble",
        "--graph-build",
        "fp",
        "--distance-mode",
        "adcbq",
        "--search-impl",
        "inline",
        "--code-layout",
        "block32",
        "--efs",
        str(ef),
    ]


def run(command: list[str], search_threads: int) -> None:
    print("RUN " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(search_threads),
            "OPENBLAS_NUM_THREADS": str(search_threads),
            "MKL_NUM_THREADS": str(search_threads),
        }
    )
    subprocess.run(command, cwd=REPO, env=env, check=True)


def calibration_candidates(
    dataset: base.Dataset,
    split: SplitFiles,
    ef: int,
    args: argparse.Namespace,
) -> Path:
    root = RESULT_DIR / dataset.key
    out = root / f"calibration_candidates.ef{ef}.summary.csv"
    candidates = root / f"calibration_candidates.ef{ef}.csv"
    expected_lines = split.calibration_rows * ef + 1
    if (
        not args.force
        and out.exists()
        and candidates.exists()
        and sum(1 for _ in candidates.open()) == expected_lines
    ):
        return candidates
    command = common_command(
        dataset,
        split.calibration_query,
        split.calibration_gt,
        out,
        ef,
        1,
        args.search_threads,
        args.build_threads,
    )
    command += [
        "--rerank-n",
        str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out",
        str(candidates),
    ]
    run(command, args.search_threads)
    return candidates


def int16_query_scales(query_path: Path) -> np.ndarray:
    nq, dim = matrix_header(query_path)
    if dim % 4:
        raise ValueError("int16 nibble scale requires a dimension divisible by 4")
    queries = np.memmap(
        query_path,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(nq, dim),
    )
    block_l1 = np.abs(np.asarray(queries)).reshape(nq, dim // 4, 4).sum(axis=2)
    max_abs = block_l1.max(axis=1)
    return np.divide(
        32767.0,
        max_abs,
        out=np.ones_like(max_abs, dtype=np.float32),
        where=max_abs > 0,
    )


def load_candidate_csv(path: Path, nq: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((nq, ef), -1, dtype=np.int64)
    distances = np.full((nq, ef), np.inf, dtype=np.float32)
    seen = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            if not (0 <= qi < nq and 0 <= rank < ef):
                raise ValueError(f"candidate coordinate out of range: q={qi}, rank={rank}")
            ids[qi, rank] = int(row["id"])
            distances[qi, rank] = float(row["distance"])
            seen += 1
    if seen != nq * ef:
        raise ValueError(f"expected {nq * ef} candidates, read {seen} from {path}")
    return ids, distances


def fit_tau(
    candidate_path: Path,
    split: SplitFiles,
    ef: int,
    fixed_r: int,
) -> dict[str, float | int]:
    ids, distances = load_candidate_csv(candidate_path, split.calibration_rows, ef)
    gt = np.memmap(
        split.calibration_gt,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(split.calibration_rows, split.gt_dim),
    )[:, :10]
    scales = int16_query_scales(split.calibration_query)
    requirements = np.full((split.calibration_rows, 10), np.inf, dtype=np.float64)
    ranks = np.full((split.calibration_rows, 10), ef + 1, dtype=np.int32)
    for qi in range(split.calibration_rows):
        positions = {int(doc): rank for rank, doc in enumerate(ids[qi]) if doc >= 0}
        dk = float(distances[qi, 9])
        for gj, gold in enumerate(gt[qi]):
            rank = positions.get(int(gold))
            if rank is None:
                continue
            ranks[qi, gj] = rank + 1
            requirements[qi, gj] = (float(distances[qi, rank]) - dk) / float(scales[qi])

    fixed_hits = int((ranks <= fixed_r).sum())
    target = fixed_hits / ranks.size
    finite = np.sort(requirements[np.isfinite(requirements)])
    if fixed_hits <= 0 or fixed_hits > len(finite):
        raise RuntimeError(
            f"cannot fit tau: fixed_hits={fixed_hits}, finite requirements={len(finite)}"
        )
    tau = float(finite[fixed_hits - 1])
    cutoffs = distances[:, 9] + tau * scales
    counts = np.maximum(10, (distances <= cutoffs[:, None]).sum(axis=1)).astype(np.int32)
    counts = np.minimum(counts, ef)
    calibrated_recall = float((ranks <= counts[:, None]).mean())
    return {
        "tau": tau,
        "fixed_r": fixed_r,
        "fixed_calibration_recall": target,
        "calibrated_calibration_recall": calibrated_recall,
        "calibrated_count_mean": float(counts.mean()),
        "calibrated_count_p95": float(np.percentile(counts, 95)),
        "calibrated_count_p99": float(np.percentile(counts, 99)),
        "calibrated_count_max": int(counts.max()),
        "calibration_rows": split.calibration_rows,
        "ef": ef,
    }


def benchmark_policy(
    dataset: base.Dataset,
    split: SplitFiles,
    ef: int,
    policy: str,
    tau: float,
    sidecar: Path,
    args: argparse.Namespace,
    output_variant: str = "",
) -> Path:
    out = RESULT_DIR / dataset.key / f"test.{policy}{output_variant}.ef{ef}.csv"
    if out.exists() and not args.force:
        with out.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) == 1 and int(rows[0]["nq"]) == split.test_rows:
            return out
    rerank_n = args.fixed_r if policy == "fixed80" else ef
    qd = rerank_n
    command = common_command(
        dataset,
        split.test_query,
        split.test_gt,
        out,
        ef,
        args.rounds,
        args.search_threads,
        args.build_threads,
    )
    command += [
        "--rerank-n",
        str(rerank_n),
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
        str(qd),
    ]
    if policy == "calibrated":
        command += ["--calibrated-threshold-rerank-tau", repr(tau)]
    run(command, args.search_threads)
    return out


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


def write_summary(
    rows: list[dict[str, object]], filename: str = "comparison.csv"
) -> Path:
    out = RESULT_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(base.DATASETS),
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--calibration-queries", type=int, default=1000)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--fixed-r", type=int, default=80)
    parser.add_argument(
        "--calibration-reference-ef",
        type=int,
        default=256,
        help="fit tau once at this efSearch and reuse it for every --efs value",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--aligned-fp-root", type=Path, default=ALIGNED_FP_ROOT)
    parser.add_argument("--keep-aligned-fp", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument(
        "--optimized-calibrated-only",
        action="store_true",
        help=(
            "reuse calibration.json and fixed@80 results, rerun only the "
            "optimized calibrated policy, and write comparison_optimized.csv"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.search_threads != 1:
        parser.error("CPU/SSD inter-query comparison requires --search-threads 1")
    if any(ef < args.fixed_r for ef in args.efs):
        parser.error("every efSearch must be >= fixed R")
    if (
        not args.optimized_calibrated_only
        and args.calibration_reference_ef not in args.efs
    ):
        parser.error("--calibration-reference-ef must be included in --efs")
    ensure_binary()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    args.aligned_fp_root.mkdir(parents=True, exist_ok=True)

    comparison: list[dict[str, object]] = []
    for dataset_key in args.datasets:
        dataset = base.DATASETS[dataset_key]
        base.validate(dataset)
        split = make_split(
            dataset,
            args.calibration_queries,
            args.seed,
            args.max_test_queries,
        )
        sidecar = args.aligned_fp_root / f"{dataset.key}.fp32.aligned4k.fbin"
        sidecar_existed = sidecar.exists()
        if not args.optimized_calibrated_only:
            calibration_path = RESULT_DIR / dataset.key / (
                f"calibration.single_tau_ef{args.calibration_reference_ef}.json"
            )
        else:
            calibration_path = RESULT_DIR / dataset.key / "calibration.json"
        if args.optimized_calibrated_only:
            if not calibration_path.exists():
                raise FileNotFoundError(
                    f"optimized-only rerun requires existing {calibration_path}"
                )
            calibrations = json.loads(calibration_path.read_text())
        else:
            calibrations = {}
        try:
            single_calibration: dict[str, float | int] | None = None
            if not args.optimized_calibrated_only:
                reference_ef = args.calibration_reference_ef
                candidate_path = calibration_candidates(dataset, split, reference_ef, args)
                single_calibration = fit_tau(
                    candidate_path,
                    split,
                    reference_ef,
                    args.fixed_r,
                )
                calibrations = {
                    "reference_ef": reference_ef,
                    "calibration": single_calibration,
                }
                print(
                    f"SINGLE-TAU {dataset.key} reference ef={reference_ef}: "
                    f"tau={single_calibration['tau']:.9g}, "
                    f"R mean/p95/p99/max={single_calibration['calibrated_count_mean']:.1f}/"
                    f"{single_calibration['calibrated_count_p95']:.0f}/"
                    f"{single_calibration['calibrated_count_p99']:.0f}/"
                    f"{single_calibration['calibrated_count_max']}, "
                    f"recall={single_calibration['calibrated_calibration_recall']:.6f} "
                    f"vs fixed{args.fixed_r}="
                    f"{single_calibration['fixed_calibration_recall']:.6f}",
                    flush=True,
                )
                if not args.keep_candidates:
                    candidate_path.unlink(missing_ok=True)
            for ef in args.efs:
                if not args.optimized_calibrated_only:
                    assert single_calibration is not None
                    calibration = single_calibration
                else:
                    if str(ef) not in calibrations:
                        raise KeyError(f"no saved calibration for ef={ef}: {calibration_path}")
                    calibration = calibrations[str(ef)]
                if args.optimized_calibrated_only:
                    print(
                        f"CALIBRATED {dataset.key} ef={ef}: tau={calibration['tau']:.9g}, "
                        f"R mean/p95/p99/max={calibration['calibrated_count_mean']:.1f}/"
                        f"{calibration['calibrated_count_p95']:.0f}/"
                        f"{calibration['calibrated_count_p99']:.0f}/"
                        f"{calibration['calibrated_count_max']}, "
                        f"recall={calibration['calibrated_calibration_recall']:.6f} "
                        f"vs fixed80={calibration['fixed_calibration_recall']:.6f}",
                        flush=True,
                    )

                fixed_path = RESULT_DIR / dataset.key / f"test.fixed80.ef{ef}.csv"
                if not args.optimized_calibrated_only:
                    if not fixed_path.exists():
                        fixed_path = benchmark_policy(
                            dataset, split, ef, "fixed80", 0.0, sidecar, args
                        )
                elif not fixed_path.exists():
                    raise FileNotFoundError(
                        f"optimized-only rerun requires existing {fixed_path}"
                    )
                calibrated_path = benchmark_policy(
                    dataset,
                    split,
                    ef,
                    "calibrated",
                    float(calibration["tau"]),
                    sidecar,
                    args,
                    (
                        f".single_tau_ef{args.calibration_reference_ef}"
                        if not args.optimized_calibrated_only
                        else ".optimized"
                    ),
                )
                fixed = read_single_row(fixed_path)
                adaptive = read_single_row(calibrated_path)
                preoptimization_adaptive = None
                if args.optimized_calibrated_only:
                    preoptimization_path = (
                        RESULT_DIR / dataset.key / f"test.calibrated.ef{ef}.csv"
                    )
                    if not preoptimization_path.exists():
                        raise FileNotFoundError(
                            "optimized-only comparison requires existing "
                            f"{preoptimization_path}"
                        )
                    preoptimization_adaptive = read_single_row(preoptimization_path)
                fixed_qps = float(fixed["qps"])
                adaptive_qps = float(adaptive["qps"])
                fixed_recall = float(fixed["recall_at_k"])
                adaptive_recall = float(adaptive["recall_at_k"])
                mean_r = float(adaptive["adaptive_rerank_count_mean"])
                result_row: dict[str, object] = {
                        "dataset": dataset.key,
                        "corpus": dataset.corpus,
                        "model": dataset.model,
                        "efSearch": ef,
                        "calibration_queries": split.calibration_rows,
                        "test_queries": split.test_rows,
                        "tau": calibration["tau"],
                        "calibration_reference_ef": (
                            args.calibration_reference_ef
                            if not args.optimized_calibrated_only
                            else ef
                        ),
                        "fixed_rerank_mean": args.fixed_r,
                        "calibrated_rerank_mean": mean_r,
                        "calibrated_rerank_p95": float(
                            adaptive["adaptive_rerank_count_p95"]
                        ),
                        "calibrated_rerank_p99": float(
                            adaptive["adaptive_rerank_count_p99"]
                        ),
                        "calibrated_rerank_max": float(
                            adaptive["adaptive_rerank_count_max"]
                        ),
                        "read_count_reduction_pct": 100.0 * (1.0 - mean_r / args.fixed_r),
                        "fixed_recall_at_10": fixed_recall,
                        "calibrated_recall_at_10": adaptive_recall,
                        "recall_delta": adaptive_recall - fixed_recall,
                        "fixed_qps": fixed_qps,
                        "calibrated_qps": adaptive_qps,
                        "qps_gain_pct": 100.0 * (adaptive_qps / fixed_qps - 1.0),
                        "fixed_submitted_bytes_per_query": float(
                            fixed["direct_submitted_bytes_mean"]
                        )
                        / split.test_rows,
                        "calibrated_submitted_bytes_per_query": float(
                            adaptive["direct_submitted_bytes_mean"]
                        )
                        / split.test_rows,
                        "fixed_pipeline_ready_ratio": float(fixed["pipeline_ready_ratio"]),
                        "calibrated_pipeline_ready_ratio": float(
                            adaptive["pipeline_ready_ratio"]
                        ),
                    }
                if preoptimization_adaptive is not None:
                    preoptimization_qps = float(preoptimization_adaptive["qps"])
                    result_row["preoptimization_calibrated_qps"] = (
                        preoptimization_qps
                    )
                    result_row["optimization_qps_gain_pct"] = 100.0 * (
                        adaptive_qps / preoptimization_qps - 1.0
                    )
                comparison.append(result_row)
        finally:
            if not args.optimized_calibrated_only:
                calibration_path.parent.mkdir(parents=True, exist_ok=True)
                calibration_path.write_text(json.dumps(calibrations, indent=2) + "\n")
            if not args.keep_aligned_fp and not sidecar_existed and sidecar.exists():
                size = sidecar.stat().st_size
                sidecar.unlink()
                print(f"REMOVED transient sidecar ({size} bytes): {sidecar}", flush=True)

    if comparison:
        path = write_summary(
            comparison,
            (
                "comparison_optimized.csv"
                if args.optimized_calibrated_only
                else f"comparison_single_tau_ef{args.calibration_reference_ef}.csv"
            ),
        )
        print(f"WROTE {path}", flush=True)
        for row in comparison:
            print(
                f"{row['dataset']} ef={row['efSearch']}: "
                f"R 80 -> {row['calibrated_rerank_mean']:.1f}, "
                f"recall {row['fixed_recall_at_10']:.6f} -> "
                f"{row['calibrated_recall_at_10']:.6f}, "
                f"QPS {row['fixed_qps']:.1f} -> {row['calibrated_qps']:.1f} "
                f"({row['qps_gain_pct']:+.1f}%)"
            )


if __name__ == "__main__":
    main()
