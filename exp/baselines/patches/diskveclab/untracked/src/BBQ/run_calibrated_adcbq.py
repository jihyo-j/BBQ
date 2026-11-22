#!/usr/bin/env python3
"""Run calibrated ADC-BQ (BBQ) as an external DiskVecLab method.

This adapter deliberately keeps BBQ separate from DiskANN's quantizer factory.
BBQ searches an FP-built in-memory HNSW with one-bit document codes, chooses a
query-dependent refinement count from a held-out calibration split, and then
refines the selected FP32 vectors in one batch.  Treating it as a DiskANN
quantizer would instead measure DiskANN+ADC-BQ and would change the query path.

The executable is built from RaBitQ-Library's experimental implementation.  A
manifest supplies the common DiskVecLab dataset paths and output locations.
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
from typing import Any, Iterable

import numpy as np


DISKVECLAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BBQ_ROOT = DISKVECLAB_ROOT.parent / "RaBitQ-Library"


class ConfigurationError(RuntimeError):
    pass


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    corpus: str
    model: str
    metric: str
    base: Path
    query: Path
    groundtruth: Path
    index: Path
    output_dir: Path
    fp_sidecar: Path


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


def resolve_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_manifest(path: Path) -> tuple[Path, list[DatasetSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), list):
        raise ConfigurationError("manifest must contain a datasets list")
    root = path.resolve().parent
    output_root = resolve_path(raw.get("output_root", "results/bbq"), root)
    datasets: list[DatasetSpec] = []
    seen: set[str] = set()
    for item in raw["datasets"]:
        if not isinstance(item, dict):
            raise ConfigurationError("each dataset entry must be an object")
        key = str(item.get("key", "")).strip()
        if not key or key in seen:
            raise ConfigurationError(f"invalid or duplicate dataset key: {key!r}")
        seen.add(key)
        dataset_output = resolve_path(item.get("output_dir", output_root / key), root)
        base = resolve_path(item["base"], root)
        datasets.append(
            DatasetSpec(
                key=key,
                corpus=str(item.get("corpus", key)),
                model=str(item.get("model", "unknown")),
                metric=str(item.get("metric", "cosine")).lower(),
                base=base,
                query=resolve_path(item["query"], root),
                groundtruth=resolve_path(item["groundtruth"], root),
                index=resolve_path(
                    item.get("index", dataset_output / f"{key}.bbq_hnsw.faiss"), root
                ),
                output_dir=dataset_output,
                fp_sidecar=resolve_path(
                    item.get("fp_sidecar", dataset_output / f"{key}.fp32.aligned4k.fbin"),
                    root,
                ),
            )
        )
    return output_root, datasets


def matrix_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ConfigurationError(f"matrix header is truncated: {path}")
    rows, dim = struct.unpack("ii", header)
    if rows <= 0 or dim <= 0:
        raise ConfigurationError(f"invalid matrix shape {(rows, dim)}: {path}")
    return rows, dim


def validate_dataset(dataset: DatasetSpec) -> tuple[int, int, int, int]:
    for path in (dataset.base, dataset.query, dataset.groundtruth):
        if not path.is_file():
            raise ConfigurationError(f"missing input file: {path}")
    nb, dim = matrix_header(dataset.base)
    nq, query_dim = matrix_header(dataset.query)
    ng, gt_dim = matrix_header(dataset.groundtruth)
    if query_dim != dim:
        raise ConfigurationError(
            f"base/query dimension mismatch for {dataset.key}: {dim} != {query_dim}"
        )
    if nq != ng:
        raise ConfigurationError(
            f"query/GT row mismatch for {dataset.key}: {nq} != {ng}"
        )
    if dim % 8:
        raise ConfigurationError(
            f"ADC-BQ requires a dimension divisible by 8, got {dim} for {dataset.key}"
        )
    if dataset.metric not in {"cosine", "ip"}:
        raise ConfigurationError(
            f"BBQ adapter supports cosine or ip, got {dataset.metric!r}"
        )
    return nb, nq, dim, gt_dim


def write_matrix(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    contiguous = np.ascontiguousarray(rows)
    with tmp.open("wb") as handle:
        handle.write(struct.pack("ii", *contiguous.shape))
        contiguous.tofile(handle)
    tmp.replace(path)


def make_split(
    dataset: DatasetSpec,
    calibration_rows: int,
    max_test_queries: int,
    seed: int,
) -> SplitFiles:
    nq, dim = matrix_header(dataset.query)
    ng, gt_dim = matrix_header(dataset.groundtruth)
    if nq != ng:
        raise ConfigurationError(f"query/GT row mismatch for {dataset.key}")
    if not 0 < calibration_rows < nq:
        raise ConfigurationError(
            f"calibration rows must be in [1, {nq - 1}], got {calibration_rows}"
        )
    permutation = np.random.default_rng(seed).permutation(nq)
    calibration_ids = permutation[:calibration_rows]
    test_ids = permutation[calibration_rows:]
    if max_test_queries > 0:
        test_ids = test_ids[:max_test_queries]
    tag = f"seed{seed}.cal{calibration_rows}.test{len(test_ids)}"
    split_root = dataset.output_dir / "splits"
    split = SplitFiles(
        calibration_query=split_root / f"{tag}.cal.query.fbin",
        calibration_gt=split_root / f"{tag}.cal.gt.ibin",
        test_query=split_root / f"{tag}.test.query.fbin",
        test_gt=split_root / f"{tag}.test.gt.ibin",
        calibration_rows=calibration_rows,
        test_rows=len(test_ids),
        dim=dim,
        gt_dim=gt_dim,
    )
    expected = {
        split.calibration_query: (calibration_rows, dim),
        split.calibration_gt: (calibration_rows, gt_dim),
        split.test_query: (len(test_ids), dim),
        split.test_gt: (len(test_ids), gt_dim),
    }
    if all(path.is_file() and matrix_header(path) == shape for path, shape in expected.items()):
        return split

    queries = np.memmap(
        dataset.query, dtype=np.float32, mode="r", offset=8, shape=(nq, dim)
    )
    gt = np.memmap(
        dataset.groundtruth,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(nq, gt_dim),
    )
    write_matrix(split.calibration_query, np.asarray(queries[calibration_ids]))
    write_matrix(split.calibration_gt, np.asarray(gt[calibration_ids]))
    write_matrix(split.test_query, np.asarray(queries[test_ids]))
    write_matrix(split.test_gt, np.asarray(gt[test_ids]))
    return split


def int16_query_scales(query_path: Path) -> np.ndarray:
    nq, dim = matrix_header(query_path)
    if dim % 4:
        raise ConfigurationError("int16 nibble LUT requires dimension divisible by 4")
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


def load_candidates(path: Path, nq: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((nq, ef), -1, dtype=np.int64)
    distances = np.full((nq, ef), np.inf, dtype=np.float32)
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query_id = int(row["query_id"])
            rank = int(row["rank"])
            if not (0 <= query_id < nq and 0 <= rank < ef):
                raise CalibrationError(
                    f"candidate coordinate out of range: query={query_id}, rank={rank}"
                )
            ids[query_id, rank] = int(row["id"])
            distances[query_id, rank] = float(row["distance"])
            seen += 1
    if seen != nq * ef:
        raise CalibrationError(
            f"expected {nq * ef} candidate rows, found {seen} in {path}"
        )
    return ids, distances


def fit_target_mean_recall(
    ids: np.ndarray,
    distances: np.ndarray,
    gt: np.ndarray,
    scales: np.ndarray,
    k: int,
    target: float,
) -> dict[str, float | int]:
    """Fit the cheapest threshold attaining target mean CandidateRecall.

    Runtime uses d(x) <= d_k + tau * scale(q).  Missing gold documents are
    retained as traversal misses, so an unattainable target fails explicitly.
    """
    if not 0.0 < target <= 1.0:
        raise CalibrationError(f"target recall must be in (0, 1], got {target}")
    nq, ef = ids.shape
    if gt.shape[0] != nq or gt.shape[1] < k or ef < k or scales.shape != (nq,):
        raise CalibrationError("incompatible calibration array shapes")

    requirements: list[float] = []
    gold_distances = np.full((nq, k), np.inf, dtype=np.float32)
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
            distance = float(distances[query_id, rank])
            gold_distances[query_id, gold_index] = distance
            requirements.append(max(0.0, (distance - anchor) / scale))

    total = nq * k
    needed = int(math.ceil(target * total - 1e-12))
    achievable = len(requirements) / total
    if needed > len(requirements):
        raise CalibrationError(
            f"target {target:.6f} exceeds traversal ceiling {achievable:.6f}"
        )
    tau = 0.0 if needed <= 0 else float(
        np.partition(np.asarray(requirements), needed - 1)[needed - 1]
    )
    cutoffs = distances[:, k - 1] + tau * scales
    counts = (distances <= cutoffs[:, None]).sum(axis=1).clip(min=k, max=ef)
    hits = gold_distances <= cutoffs[:, None]
    return {
        "target_mean_recall": target,
        "tau": tau,
        "calibration_mean_recall": float(hits.mean()),
        "traversal_recall_ceiling": achievable,
        "candidate_count_mean": float(counts.mean()),
        "candidate_count_min": int(counts.min()),
        "candidate_count_p95": float(np.percentile(counts, 95)),
        "candidate_count_p99": float(np.percentile(counts, 99)),
        "candidate_count_max": int(counts.max()),
        "calibration_queries": nq,
        "ef_search": ef,
    }


def run_command(command: list[str], threads: int, cwd: Path) -> None:
    print("RUN " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
        }
    )
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def ensure_binary(bbq_root: Path, binary: Path, force: bool) -> None:
    source = bbq_root / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
    build_script = bbq_root / "experiment/adcbq_optimized/build.sh"
    if not source.is_file() or not build_script.is_file():
        raise ConfigurationError(f"not a compatible RaBitQ-Library checkout: {bbq_root}")
    if binary.is_file() and not force and binary.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return
    binary.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OUT"] = str(binary)
    subprocess.run([str(build_script)], cwd=bbq_root, env=environment, check=True)


def common_command(
    binary: Path,
    dataset: DatasetSpec,
    query: Path,
    gt: Path,
    output: Path,
    ef: int,
    rounds: int,
    args: argparse.Namespace,
    rebuild: bool = False,
) -> list[str]:
    command = [
        str(binary),
        "--dataset", dataset.key,
        "--train", str(dataset.base),
        "--query", str(query),
        "--gt", str(gt),
        "--index", str(dataset.index),
        "--out", str(output),
        "--m", str(args.m),
        "--ef-construction", str(args.ef_construction),
        "--k", str(args.k),
        "--max-queries", "0",
        "--rounds", str(rounds),
        "--build-threads", str(args.build_threads),
        "--search-threads", str(args.search_threads),
        "--adc-mode", "int16",
        "--lut-mode", "nibble",
        "--graph-build", "fp",
        "--distance-mode", "adcbq",
        "--search-impl", "inline",
        "--code-layout", "block32",
        "--efs", str(ef),
    ]
    if rebuild:
        command.append("--rebuild")
    return command


def generate_candidates(
    binary: Path,
    bbq_root: Path,
    dataset: DatasetSpec,
    split: SplitFiles,
    ef: int,
    args: argparse.Namespace,
    rebuild: bool,
) -> Path:
    output = dataset.output_dir / f"calibration.ef{ef}.summary.csv"
    candidates = dataset.output_dir / f"calibration.ef{ef}.candidates.csv"
    if candidates.is_file() and output.is_file() and not args.force:
        return candidates
    output.parent.mkdir(parents=True, exist_ok=True)
    command = common_command(
        binary,
        dataset,
        split.calibration_query,
        split.calibration_gt,
        output,
        ef,
        1,
        args,
        rebuild=rebuild,
    )
    command += [
        "--rerank-n", str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out", str(candidates),
    ]
    run_command(command, args.search_threads, bbq_root)
    return candidates


def benchmark(
    binary: Path,
    bbq_root: Path,
    dataset: DatasetSpec,
    split: SplitFiles,
    ef: int,
    calibration: dict[str, float | int],
    args: argparse.Namespace,
) -> Path:
    target_tag = f"{float(calibration['target_mean_recall']):.6f}".rstrip("0").rstrip(".")
    output = dataset.output_dir / f"test.target{target_tag}.ef{ef}.csv"
    if output.is_file() and not args.force:
        return output
    command = common_command(
        binary,
        dataset,
        split.test_query,
        split.test_gt,
        output,
        ef,
        args.rounds,
        args,
    )
    command += [
        "--rerank-n", str(ef),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", args.fp_source,
        "--fp-rerank-norms", "onfetch",
        "--calibrated-threshold-rerank-tau", repr(float(calibration["tau"])),
    ]
    if args.fp_source in {"odirect", "odirect-uring"}:
        command += [
            "--fp-rerank-path", str(dataset.fp_sidecar),
            "--fp-rerank-qd", str(ef),
        ]
    if args.overlap_queries:
        command.append("--fp-rerank-overlap-queries")
    run_command(command, args.search_threads, bbq_root)
    return output


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


def pick(row: dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return value if value is not None else default


def result_row(
    dataset: DatasetSpec,
    calibration: dict[str, float | int],
    measured: dict[str, str],
) -> dict[str, Any]:
    qps = float(pick(measured, "qps", "nan"))
    return {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method": "BBQ-calibrated-ADC-BQ",
        "status": "ok",
        "target_mean_recall": calibration["target_mean_recall"],
        "calibration_tau": calibration["tau"],
        "calibration_mean_recall": calibration["calibration_mean_recall"],
        "traversal_recall_ceiling": calibration["traversal_recall_ceiling"],
        "calibration_r_mean": calibration["candidate_count_mean"],
        "calibration_r_p95": calibration["candidate_count_p95"],
        "calibration_r_p99": calibration["candidate_count_p99"],
        "calibration_r_max": calibration["candidate_count_max"],
        "k": pick(measured, "k"),
        "efSearch": pick(measured, "efSearch"),
        "M": pick(measured, "M"),
        "efConstruction": pick(measured, "efConstruction"),
        "nq": pick(measured, "nq"),
        "nb": pick(measured, "nb"),
        "dim": pick(measured, "dim"),
        "recall_at_k": pick(measured, "recall_at_k"),
        "qps": pick(measured, "qps"),
        "latency_mean_ms": 1000.0 / qps if qps > 0 else math.nan,
        "search_seconds_mean": pick(measured, "search_seconds_mean"),
        "rerank_seconds_mean": pick(measured, "rerank_seconds_mean"),
        "total_seconds_mean": pick(measured, "total_seconds_mean"),
        "fp_vectors_mean": pick(measured, "adaptive_rerank_count_mean"),
        "fp_vectors_p95": pick(measured, "adaptive_rerank_count_p95"),
        "fp_vectors_p99": pick(measured, "adaptive_rerank_count_p99"),
        "fp_vectors_max": pick(measured, "adaptive_rerank_count_max"),
        "fp_read_bytes_per_query": pick(measured, "rerank_read_bytes_per_query"),
        "fp_rerank_source": pick(measured, "fp_rerank_source"),
        "index_path": pick(measured, "index_path"),
        "fp_rerank_path": pick(measured, "fp_rerank_path"),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbq-root", type=Path, default=Path(os.environ.get("BBQ_ROOT", DEFAULT_BBQ_ROOT)))
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--efs", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--targets", nargs="+", type=float, default=[0.95, 0.99])
    parser.add_argument("--calibration-queries", type=int, default=1000)
    parser.add_argument("--max-test-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument(
        "--fp-source",
        choices=("mmap", "pread", "odirect", "odirect-uring"),
        default="odirect-uring",
    )
    parser.add_argument("--overlap-queries", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--rebuild-binary", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.bbq_root = args.bbq_root.expanduser().resolve()
    args.binary = (
        args.binary.expanduser().resolve()
        if args.binary
        else args.bbq_root / "experiment/adcbq_optimized/faiss_adcbq_hnsw_calibrated"
    )
    if any(ef < args.k for ef in args.efs):
        parser.error("every efSearch must be >= k")
    if any(not 0.0 < target <= 1.0 for target in args.targets):
        parser.error("targets must be in (0, 1]")
    if args.overlap_queries and args.search_threads != 1:
        parser.error("inter-query overlap currently requires --search-threads 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root, datasets = load_manifest(args.manifest)
    if args.datasets:
        requested = set(args.datasets)
        datasets = [dataset for dataset in datasets if dataset.key in requested]
        missing = requested - {dataset.key for dataset in datasets}
        if missing:
            raise ConfigurationError(f"unknown datasets: {sorted(missing)}")
    if not datasets:
        raise ConfigurationError("no datasets selected")

    for dataset in datasets:
        nb, nq, dim, gt_dim = validate_dataset(dataset)
        print(
            f"PLAN {dataset.key}: nb={nb} nq={nq} d={dim} gt={gt_dim} "
            f"efs={args.efs} targets={args.targets} output={dataset.output_dir}",
            flush=True,
        )
    if args.plan:
        return 0

    ensure_binary(args.bbq_root, args.binary, args.rebuild_binary)
    all_results: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset.output_dir.mkdir(parents=True, exist_ok=True)
        dataset.index.parent.mkdir(parents=True, exist_ok=True)
        split = make_split(
            dataset,
            args.calibration_queries,
            args.max_test_queries,
            args.seed,
        )
        calibration_gt = np.memmap(
            split.calibration_gt,
            dtype=np.int32,
            mode="r",
            offset=8,
            shape=(split.calibration_rows, split.gt_dim),
        )
        scales = int16_query_scales(split.calibration_query)
        rebuild_pending = args.rebuild_index
        for ef in args.efs:
            candidate_path = generate_candidates(
                args.binary,
                args.bbq_root,
                dataset,
                split,
                ef,
                args,
                rebuild_pending,
            )
            rebuild_pending = False
            candidate_ids, candidate_distances = load_candidates(
                candidate_path, split.calibration_rows, ef
            )
            calibrations: list[dict[str, float | int]] = []
            for target in args.targets:
                try:
                    calibration = fit_target_mean_recall(
                        candidate_ids,
                        candidate_distances,
                        np.asarray(calibration_gt),
                        scales,
                        args.k,
                        target,
                    )
                except CalibrationError as error:
                    print(f"SKIP {dataset.key} ef={ef} target={target}: {error}", file=sys.stderr)
                    all_results.append(
                        {
                            "dataset": dataset.key,
                            "corpus": dataset.corpus,
                            "model": dataset.model,
                            "method": "BBQ-calibrated-ADC-BQ",
                            "status": "unachievable",
                            "target_mean_recall": target,
                            "efSearch": ef,
                            "reason": str(error),
                        }
                    )
                    continue
                calibrations.append(calibration)
                calibration_path = dataset.output_dir / (
                    f"calibration.target{target:.6f}.ef{ef}.json"
                )
                calibration_path.write_text(
                    json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8"
                )
                print(
                    f"CALIBRATED {dataset.key} ef={ef} target={target:.4f}: "
                    f"tau={calibration['tau']:.9g}, "
                    f"recall={calibration['calibration_mean_recall']:.6f}, "
                    f"R mean/p95/p99/max={calibration['candidate_count_mean']:.1f}/"
                    f"{calibration['candidate_count_p95']:.0f}/"
                    f"{calibration['candidate_count_p99']:.0f}/"
                    f"{calibration['candidate_count_max']}",
                    flush=True,
                )
            for calibration in calibrations:
                measured_path = benchmark(
                    args.binary,
                    args.bbq_root,
                    dataset,
                    split,
                    ef,
                    calibration,
                    args,
                )
                all_results.append(
                    result_row(dataset, calibration, read_single_row(measured_path))
                )
            if not args.keep_candidates:
                candidate_path.unlink(missing_ok=True)
        write_csv(dataset.output_dir / "bbq_results.csv", [
            row for row in all_results if row["dataset"] == dataset.key
        ])
    write_csv(output_root / "bbq_results.csv", all_results)
    print(f"WROTE {output_root / 'bbq_results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfigurationError, CalibrationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
