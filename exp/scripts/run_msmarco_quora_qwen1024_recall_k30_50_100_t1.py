#!/usr/bin/env python3
"""Held-out Recall@30/50/100 sweeps for three Qwen-1024 SSD methods.

The runner compares calibrated ADC-BQ, the official RaBitQ 1-bit c16
adaptive-FP path, and resident TurboQuant TQ1+ with io_uring FP refinement on
NQ.  A seed-11 split reserves 1,000 queries for calibration and uses every
remaining query for measurement.  For each k:

* ADC-BQ independently fits one distance threshold at every efSearch using
  split conformal calibration.  The threshold retains all exact-FP top-k
  candidates inside each ADC candidate pool with target probability 0.99.
* TurboQuant sweeps (efSearch, R) on the calibration queries, builds the joint
  Recall/QPS Pareto frontier, and selects the maximum-QPS point reaching
  Recall@k >= 0.99.  Only the selected R is then fixed for the held-out curve.
* RaBitQ uses its native adaptive lower-bound FP policy.

The final curves use 20 geometrically spaced efSearch values, with extra
density near the smallest valid ef.  Every method is evaluated with one search
thread and three rounds by default.  The implementation is deliberately
restartable: complete raw files are reused unless --force is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiment"))
sys.path.insert(
    0,
    str(REPO / "experiment/rabitq_v051_adaptive_odirect_aligned4k_build/rabitqlib"),
)

import diskveclab_full_matrix as matrix  # noqa: E402
import run_adcbq_calibrated_threshold_overlap as adc  # noqa: E402
import run_diskveclab_tq_pq_opq_uring as tq  # noqa: E402
import run_v051_ssd_odirect_multidataset as registry  # noqa: E402
from _rabitqlib import HnswIndex  # type: ignore  # noqa: E402


RESULT_ROOT = REPO / "experiment/nq_msmarco_qwen1024_recall_k30_50_100_t1"
FIGURE_ROOT = REPO / "experiment/figures/nq_msmarco_qwen1024_recall_k30_50_100_t1"
RAW_ROOT = RESULT_ROOT / "raw"
SPLIT_ROOT = RESULT_ROOT / "splits"
TQ_INDEX_ROOT = RESULT_ROOT / "tq_indexes"
SUMMARY_CSV = RESULT_ROOT / "nq_msmarco_qwen1024_recall_k30_50_100_t1.csv"
ADC_PROTOCOL = "per-ef-pool-topk-split-conformal"
ADC_RAW_NAME = "adc_per_ef_pool_topk_target99"

DATASET_KEYS = ("nq_qwen06_1024",)
KS = (30, 50, 100)
CALIBRATION_ROWS = 1000
SEED = 11
TQ_TARGET_RECALL = 0.99
TQ_PARETO_EFS = (32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024)
ADC_TARGET_COVERAGE = 0.99
PAGE_BYTES = 4096
ALIGNED_FP_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp")
MSMARCO_TOP100_GT = (
    REPO
    / "experiment/msmarco_quora_qwen1024_recall_k30_50_100_t1/"
    "msmarco_qwen1024_exact_top100.ibin"
)
ADC_BIN = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_same_pool_calibration"

RABITQ_INDEXES = {
    "nq_qwen06_1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/indexes/"
        "nq_qwen06_1024.rabitq_b1_c16.n1000000.d1024.M24.efC200.index"
    ),
    "msmarco_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/indexes/"
        "msmarco_qwen1024.rabitq_b1_c16.n1000000.d1024.M24.efC200.index"
    ),
    "quora_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/indexes/"
        "quora_qwen1024.rabitq_b1_c16.n522931.d1024.M24.efC200.index"
    ),
}


@dataclass(frozen=True)
class Split:
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(struct.pack("ii", *rows.shape))
        rows.tofile(handle)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run(command: list[str], *, threads: int = 1) -> None:
    print("RUN " + " ".join(command), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=REPO, env=environment, check=True)


def ensure_exact_msmarco_top100(build_threads: int, force: bool) -> Path:
    dataset = registry.DATASETS["msmarco_qwen1024"]
    nq, dim = matrix_header(dataset.query)
    nb, train_dim = matrix_header(dataset.train)
    if dim != train_dim:
        raise RuntimeError("MS MARCO train/query dimension mismatch")
    if not force and MSMARCO_TOP100_GT.is_file():
        if matrix_header(MSMARCO_TOP100_GT) == (nq, 100):
            print(f"REUSE {MSMARCO_TOP100_GT}", flush=True)
            return MSMARCO_TOP100_GT

    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "exact top-100 generation requires the hnsw conda Python with faiss"
        ) from exc

    partial_path = MSMARCO_TOP100_GT.with_suffix(".partial.i32")
    progress_path = MSMARCO_TOP100_GT.with_suffix(".progress.json")
    MSMARCO_TOP100_GT.parent.mkdir(parents=True, exist_ok=True)
    expected_partial_bytes = nq * 100 * np.dtype(np.int32).itemsize
    if force:
        partial_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    print(
        f"BUILD exact MS MARCO top-100: nq={nq} nb={nb} dim={dim} "
        f"threads={build_threads}",
        flush=True,
    )
    faiss.omp_set_num_threads(build_threads)
    train = np.memmap(
        dataset.train, dtype=np.float32, mode="r", offset=8, shape=(nb, dim)
    )
    queries = np.memmap(
        dataset.query, dtype=np.float32, mode="r", offset=8, shape=(nq, dim)
    )
    index = faiss.IndexFlatIP(dim)
    add_batch = 100_000
    for start in range(0, nb, add_batch):
        stop = min(nb, start + add_batch)
        train_values = np.array(train[start:stop], dtype=np.float32, copy=True, order="C")
        normalize_rows_inplace(train_values)
        index.add(train_values)
        print(f"  exact-GT add {stop}/{nb}", flush=True)
    resume_row = 0
    if partial_path.is_file() and partial_path.stat().st_size == expected_partial_bytes:
        try:
            resume_row = int(json.loads(progress_path.read_text(encoding="utf-8"))["rows"])
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            resume_row = 0
    mode = "r+" if partial_path.is_file() and partial_path.stat().st_size == expected_partial_bytes else "w+"
    ids = np.memmap(partial_path, dtype=np.int32, mode=mode, shape=(nq, 100))
    if mode == "w+":
        ids[:] = -1
        ids.flush()
    if resume_row:
        print(f"  exact-GT resume at {resume_row}/{nq}", flush=True)
    query_batch = 64
    for start in range(resume_row, nq, query_batch):
        stop = min(nq, start + query_batch)
        query_values = np.array(queries[start:stop], dtype=np.float32, copy=True, order="C")
        normalize_rows_inplace(query_values)
        # Avoid the MKL tiny-tail path: the final MS MARCO batch has only four
        # queries and has triggered an AVX-512 loader failure on this host.
        if stop - start < query_batch:
            padded = np.zeros((query_batch, dim), dtype=np.float32)
            padded[: stop - start] = query_values
            query_values = padded
        _, batch_ids = index.search(query_values, 100)
        ids[start:stop] = batch_ids[: stop - start].astype(np.int32, copy=False)
        if stop == nq or stop % 512 == 0:
            ids.flush()
            progress_path.write_text(
                json.dumps({"rows": stop, "nq": nq, "k": 100}) + "\n",
                encoding="utf-8",
            )
        print(f"  exact-GT search {stop}/{nq}", flush=True)
    old_nq, old_k = matrix_header(dataset.gt)
    old_gt = np.memmap(
        dataset.gt,
        dtype=np.int32,
        mode="r",
        offset=8,
        shape=(old_nq, old_k),
    )
    overlap = np.mean(
        [
            len(set(map(int, ids[qi, :10])).intersection(map(int, old_gt[qi, :10])))
            / 10.0
            for qi in range(nq)
        ]
    )
    # The historical k=10 file differs slightly from a fresh exhaustive
    # IndexFlatIP run (about 0.26% of neighbours on this host), likely from
    # numeric/tie-handling differences in its original generator.  The new
    # k=100 file is itself exhaustive over the current FP32 corpus, so use the
    # old file as a high-overlap provenance check rather than as ground truth.
    if overlap < 0.995:
        raise RuntimeError(f"new MS MARCO GT disagrees with existing top-10: {overlap}")
    write_matrix(MSMARCO_TOP100_GT, ids)
    del ids
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    print(f"WROTE {MSMARCO_TOP100_GT}; top10 overlap={overlap:.6f}", flush=True)
    return MSMARCO_TOP100_GT


def dataset_with_top100(key: str, build_threads: int, force_gt: bool):
    dataset = registry.DATASETS[key]
    gt = (
        ensure_exact_msmarco_top100(build_threads, force_gt)
        if key == "msmarco_qwen1024"
        else dataset.gt
    )
    nq, gt_dim = matrix_header(gt)
    if gt_dim < max(KS):
        raise RuntimeError(f"{key} GT has only {gt_dim} neighbours")
    if nq != matrix_header(dataset.query)[0]:
        raise RuntimeError(f"{key} query/GT row mismatch")
    return replace(dataset, gt=gt)


def make_split(dataset) -> Split:
    nq, dim = matrix_header(dataset.query)
    ng, gt_dim = matrix_header(dataset.gt)
    test_rows = nq - CALIBRATION_ROWS
    if nq != ng or test_rows <= 0:
        raise RuntimeError(f"invalid split inputs for {dataset.key}")
    permutation = np.random.default_rng(SEED).permutation(nq)
    calibration_ids = permutation[:CALIBRATION_ROWS]
    test_ids = permutation[CALIBRATION_ROWS:]
    tag = f"seed{SEED}.cal{CALIBRATION_ROWS}.test{test_rows}.top100"
    root = SPLIT_ROOT / dataset.key
    split = Split(
        root / f"{tag}.cal.query.fbin",
        root / f"{tag}.cal.gt.ibin",
        root / f"{tag}.test.query.fbin",
        root / f"{tag}.test.gt.ibin",
        CALIBRATION_ROWS,
        test_rows,
        dim,
        gt_dim,
    )
    expected = {
        split.calibration_query: (CALIBRATION_ROWS, dim),
        split.calibration_gt: (CALIBRATION_ROWS, gt_dim),
        split.test_query: (test_rows, dim),
        split.test_gt: (test_rows, gt_dim),
    }
    if all(path.is_file() and matrix_header(path) == shape for path, shape in expected.items()):
        return split
    queries = np.memmap(
        dataset.query, dtype=np.float32, mode="r", offset=8, shape=(nq, dim)
    )
    gt = np.memmap(
        dataset.gt, dtype=np.int32, mode="r", offset=8, shape=(nq, gt_dim)
    )
    write_matrix(split.calibration_query, np.asarray(queries[calibration_ids]))
    write_matrix(split.calibration_gt, np.asarray(gt[calibration_ids]))
    write_matrix(split.test_query, np.asarray(queries[test_ids]))
    write_matrix(split.test_gt, np.asarray(gt[test_ids]))
    print(f"WROTE held-out split for {dataset.key}", flush=True)
    return split


def aligned_fp(key: str) -> Path:
    path = ALIGNED_FP_ROOT / f"{key}.fp32.header4096.fbin"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def recall_at_k(ids: np.ndarray, gt: np.ndarray, k: int) -> float:
    predicted = np.asarray(ids)[:, :k]
    truth = np.asarray(gt)[:, :k]
    hits = 0
    for left, right in zip(predicted, truth):
        hits += len(set(map(int, left)).intersection(map(int, right)))
    return hits / float(len(predicted) * k)


def normalize_rows_inplace(values: np.ndarray) -> None:
    squared = np.einsum("ij,ij->i", values, values, optimize=False)
    norms = np.sqrt(squared, dtype=np.float32)
    np.divide(values, norms[:, None], out=values, where=norms[:, None] > 0)


def geometric_efs(start: int, count: int = 20) -> tuple[int, ...]:
    if start > 1024:
        raise ValueError(f"ef start exceeds 1024: {start}")
    if start == 1024:
        return (1024,)
    # A power curve is denser than a geometric series at the front while still
    # covering the complete tail.  Round to the conventional multiples of 8.
    ordered: list[int] = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, count)):
        raw = start + (1024 - start) * float(fraction) ** 1.7
        value = int(round(raw / 8.0) * 8)
        if index == 0:
            value = start
        elif index == count - 1:
            value = 1024
        value = min(1024, max(start, value))
        if value not in ordered:
            ordered.append(value)
    if len(ordered) < count:
        for value in range(start, 1025):
            if value not in ordered:
                ordered.append(value)
            if len(ordered) == count:
                break
        ordered.sort()
    if len(ordered) != count:
        raise RuntimeError(f"failed to construct {count} unique ef values: {ordered}")
    return tuple(ordered)


def adc_common_command(
    dataset,
    query: Path,
    gt: Path,
    out: Path,
    *,
    k: int,
    ef: int,
    rounds: int,
    build_threads: int,
    cpu: int,
) -> list[str]:
    return [
        "taskset", "-c", str(cpu),
        str(ADC_BIN),
        "--dataset", dataset.key,
        "--train", str(dataset.train),
        "--query", str(query),
        "--gt", str(gt),
        "--index", str(dataset.adcbq_index),
        "--out", str(out),
        "--m", "24",
        "--ef-construction", "200",
        "--k", str(k),
        "--max-queries", "0",
        "--rounds", str(rounds),
        "--build-threads", str(build_threads),
        "--search-threads", "1",
        "--adc-mode", "int16",
        "--lut-mode", "nibble",
        "--graph-build", "fp",
        "--distance-mode", "adcbq",
        "--search-impl", "inline",
        "--code-layout", "block32",
        "--efs", str(ef),
    ]


def adc_root(dataset_key: str, k: int) -> Path:
    return RAW_ROOT / dataset_key / f"k{k}" / ADC_RAW_NAME


def adc_calibration_path(dataset_key: str, k: int, ef: int) -> Path:
    return adc_root(dataset_key, k) / f"calibration.ef{ef}.json"


def adc_candidates(
    dataset,
    split: Split,
    k: int,
    ef: int,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    root = adc_root(dataset.key, k) / "candidates"
    summary = root / f"ef{ef}.summary.csv"
    candidates = root / f"ef{ef}.csv"
    expected_lines = split.calibration_rows * ef + 1
    if (
        not args.force
        and summary.is_file()
        and candidates.is_file()
        and sum(1 for _ in candidates.open(encoding="utf-8")) == expected_lines
    ):
        return summary, candidates
    root.mkdir(parents=True, exist_ok=True)
    command = adc_common_command(
        dataset,
        split.calibration_query,
        split.calibration_gt,
        summary,
        k=k,
        ef=ef,
        rounds=1,
        build_threads=args.build_threads,
        cpu=args.cpu,
    )
    command += [
        "--rerank-n", str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out", str(candidates),
    ]
    run(command)
    return summary, candidates


def load_adc_candidates(
    path: Path, nq: int, ef: int
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((nq, ef), -1, dtype=np.int64)
    distances = np.full((nq, ef), np.inf, dtype=np.float32)
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            ids[qi, rank] = int(row["id"])
            distances[qi, rank] = float(row["distance"])
            seen += 1
    if seen != nq * ef:
        raise RuntimeError(f"incomplete ADC candidate trace: {path}")
    return ids, distances


def fit_adc_tau_per_ef(
    dataset,
    split: Split,
    k: int,
    ef: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    path = adc_calibration_path(dataset.key, k, ef)
    if path.is_file() and not args.force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") == ADC_PROTOCOL
            and payload.get("status") == "feasible"
            and int(payload.get("k", -1)) == k
            and int(payload.get("efSearch", -1)) == ef
            and int(payload.get("calibration_queries", -1))
            == split.calibration_rows
            and float(payload.get("target_pool_topk_coverage", -1.0))
            == ADC_TARGET_COVERAGE
        ):
            print(
                f"REUSE ADC_CAL {dataset.key} k={k} ef={ef} "
                f"tau={float(payload['tau']):.9g}",
                flush=True,
            )
            return payload

    summary_path, candidate_path = adc_candidates(dataset, split, k, ef, args)
    ids, distances = load_adc_candidates(
        candidate_path, split.calibration_rows, ef
    )
    nq, dim = matrix_header(split.calibration_query)
    queries = np.array(
        np.memmap(
            split.calibration_query,
            dtype=np.float32,
            mode="r",
            offset=8,
            shape=(nq, dim),
        ),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    normalize_rows_inplace(queries)
    ntrain, train_dim = matrix_header(dataset.train)
    if train_dim != dim:
        raise RuntimeError(f"train/query dimension mismatch for {dataset.key}")
    train = np.memmap(
        dataset.train,
        dtype=np.float32,
        mode="r",
        offset=8,
        shape=(ntrain, train_dim),
    )
    scales = adc.int16_query_scales(split.calibration_query)
    requirements = np.zeros(split.calibration_rows, dtype=np.float64)
    exact_ranks = np.empty((split.calibration_rows, k), dtype=np.int32)
    for qi in range(split.calibration_rows):
        valid = ids[qi] >= 0
        row_ids = ids[qi, valid]
        row_distances = distances[qi, valid]
        if len(row_ids) < k:
            raise RuntimeError(
                f"ADC candidate pool has {len(row_ids)} < k={k}: "
                f"{dataset.key} ef={ef} query={qi}"
            )
        vectors = np.asarray(train[row_ids], dtype=np.float32)
        vector_norms = np.linalg.norm(vectors, axis=1)
        scores = vectors @ queries[qi]
        scores = np.divide(
            scores,
            vector_norms,
            out=np.full_like(scores, -np.inf),
            where=vector_norms > 0,
        )
        exact_order = np.lexsort((row_ids, -scores))
        exact_positions = exact_order[:k]
        exact_ranks[qi] = exact_positions.astype(np.int32) + 1
        anchor = float(row_distances[k - 1])
        needed = row_distances[exact_positions] - anchor
        requirements[qi] = max(
            0.0, float(np.max(needed)) / float(scales[qi])
        )

    conformal_order = math.ceil(
        (split.calibration_rows + 1) * ADC_TARGET_COVERAGE
    )
    if conformal_order > split.calibration_rows:
        raise RuntimeError(
            f"target {ADC_TARGET_COVERAGE} is infeasible with "
            f"{split.calibration_rows} calibration queries"
        )
    tau = float(
        np.partition(requirements, conformal_order - 1)[conformal_order - 1]
    )
    cutoffs = distances[:, k - 1] + tau * scales
    counts = np.maximum(k, (distances <= cutoffs[:, None]).sum(axis=1))
    counts = np.minimum(counts, ef).astype(np.int32)
    retained = np.all(exact_ranks <= counts[:, None], axis=1)
    result: dict[str, object] = {
        "protocol": ADC_PROTOCOL,
        "status": "feasible",
        "dataset": dataset.key,
        "k": k,
        "efSearch": ef,
        "target_pool_topk_coverage": ADC_TARGET_COVERAGE,
        "empirical_pool_topk_coverage": float(np.mean(retained)),
        "conformal_order": conformal_order,
        "calibration_queries": split.calibration_rows,
        "reference": f"exact FP top-{k} within the ADC-HNSW candidate pool",
        "tau": tau,
        "rerank_count_mean": float(np.mean(counts)),
        "rerank_count_min": int(np.min(counts)),
        "rerank_count_p95": float(np.percentile(counts, 95)),
        "rerank_count_p99": float(np.percentile(counts, 99)),
        "rerank_count_max": int(np.max(counts)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not args.keep_candidates:
        candidate_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    print(
        f"ADC_CAL {dataset.key} k={k} ef={ef} tau={tau:.9g} "
        f"coverage={result['empirical_pool_topk_coverage']:.6f} "
        f"meanR={result['rerank_count_mean']:.2f}",
        flush=True,
    )
    return result


def adc_raw_complete(
    path: Path,
    dataset_key: str,
    k: int,
    ef: int,
    rounds: int,
    tau: float,
) -> bool:
    rows = read_csv(path)
    try:
        return len(rows) == 1 and all(
            row["dataset"] == dataset_key
            and int(row["k"]) == k
            and int(row["efSearch"]) == ef
            and int(row["rounds"]) == rounds
            and int(row["search_threads"]) == 1
            and row["calibrated_threshold_rerank_enabled"] == "true"
            and math.isclose(
                float(row["calibrated_threshold_rerank_tau"]),
                tau,
                rel_tol=5e-5,
                abs_tol=5e-7,
            )
            for row in rows
        )
    except (KeyError, ValueError):
        return False


def run_adc_curve(
    dataset,
    split: Split,
    k: int,
    calibrations: dict[int, dict[str, object]],
    efs: tuple[int, ...],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for ef in efs:
        calibration = calibrations[ef]
        tau = float(calibration["tau"])
        path = adc_root(dataset.key, k) / f"heldout.ef{ef}.csv"
        if args.force or not adc_raw_complete(
            path, dataset.key, k, ef, args.rounds, tau
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            command = adc_common_command(
                dataset,
                split.test_query,
                split.test_gt,
                path,
                k=k,
                ef=ef,
                rounds=args.rounds,
                build_threads=args.build_threads,
                cpu=args.cpu,
            )
            command += [
                "--rerank-n", str(ef),
                "--rerank-dot", "fp32",
                "--fp-rerank-source", "odirect-uring",
                "--fp-rerank-norms", "onfetch",
                "--fp-rerank-path", str(aligned_fp(dataset.key)),
                "--fp-rerank-overlap-queries",
                "--fp-rerank-qd", str(ef),
                "--calibrated-threshold-rerank-tau", repr(tau),
            ]
            run(command)
        raw = read_csv(path)[0]
        output_rows.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method_key": "adc_bq_calibrated",
                "method": f"ADC-BQ per-ef pool-top{k}@0.99",
                "k": k,
                "efSearch": ef,
                "recall_at_k": float(raw["recall_at_k"]),
                "qps": float(raw["qps"]),
                "rounds": args.rounds,
                "search_threads": 1,
                "rerank_policy": "per_ef_pool_topk_split_conformal",
                "rerank_n_or_mean": float(raw["adaptive_rerank_count_mean"]),
                "fp_reads_per_query": float(raw["direct_rows_mean"]) / float(raw["nq"]),
                "io_engine": "io_uring O_DIRECT inter-query overlap",
                "calibration_queries": CALIBRATION_ROWS,
                "test_queries": split.test_rows,
                "calibration_reference_ef": ef,
                "calibration_retention_target": ADC_TARGET_COVERAGE,
                "calibration_protocol": ADC_PROTOCOL,
                "target_pool_topk_coverage": ADC_TARGET_COVERAGE,
                "empirical_pool_topk_coverage": float(
                    calibration["empirical_pool_topk_coverage"]
                ),
                "calibration_rerank_count_mean": float(
                    calibration["rerank_count_mean"]
                ),
                "tau": tau,
                "calibration_json": str(
                    adc_calibration_path(dataset.key, k, ef).relative_to(REPO)
                ),
                "source_csv": str(path.relative_to(REPO)),
            }
        )
        write_summary(output_rows, append_method=False)
    return output_rows


def configure_tq() -> None:
    tq.RESULT_ROOT = RESULT_ROOT / "tq_work"
    tq.INDEX_ROOT = TQ_INDEX_ROOT
    tq.SIDECAR_ROOT = RESULT_ROOT / "unused_tq_sidecars"
    tq.EXTERNAL_ROOT = RESULT_ROOT
    tq.EXTERNAL_FLOOR_GIB = 1.0
    tq.ROOT_FLOOR_GIB = 1.0
    tq.SUMMARY = tq.RESULT_ROOT / "unused_summary.csv"
    tq.STATUS = tq.RESULT_ROOT / "status.json"
    tq.LOCK = tq.RESULT_ROOT / "queue.lock"


def ensure_tq_codec(dataset_key: str, build_threads: int) -> tuple[Path, Path]:
    configure_tq()
    tq.codec_builder.compile_binary()
    tq.search_runner.compile_binary()
    method = tq.Method("qdranttq1", "TurboQuant TQ1+", 100)
    build_args = SimpleNamespace(
        build_threads=build_threads,
        train_size=50_000,
        opq_niter=25,
        opq_blas_threads=1,
    )
    tq.build_codec(dataset_key, method, build_args)
    index, meta = tq.codec_paths(dataset_key, method.key)
    if not index.is_file() or not meta.is_file():
        raise FileNotFoundError((index, meta))
    return index, meta


def tq_reranks(k: int) -> tuple[int, ...]:
    grids = {
        30: (30, 40, 60, 80, 120, 160, 200, 256),
        50: (50, 60, 80, 100, 120, 160, 200, 256, 320),
        100: (100, 120, 140, 160, 200, 240, 320, 400, 512),
    }
    return grids[k]


def tq_pareto_efs(rerank_n: int) -> tuple[int, ...]:
    return tuple(ef for ef in TQ_PARETO_EFS if ef >= rerank_n)


def tq_pareto_raw_complete(
    path: Path, dataset_key: str, k: int, rerank_n: int
) -> bool:
    rows = read_csv(path)
    efs = tq_pareto_efs(rerank_n)
    try:
        return (
            {(int(row["efSearch"]), int(row["round"])) for row in rows}
            == {(ef, 1) for ef in efs}
            and all(
                row["dataset"] == dataset_key
                and int(row["k"]) == k
                and int(row["rerank_n"]) == rerank_n
                and int(row["threads"]) == 1
                and row["code_storage"] == "resident"
                and row["fp_io_mode"] == "uring-batch"
                and row["measurement_mode"] == "inter_query_overlap"
                and row["first_stage_search_impl"] == "inline_batched_prefetch"
                for row in rows
            )
        )
    except (KeyError, ValueError):
        return False


def pareto_recall_qps(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    best_at_recall: dict[float, dict[str, object]] = {}
    for row in rows:
        recall = float(row["recall_at_k"])
        old = best_at_recall.get(recall)
        if old is None or float(row["qps"]) > float(old["qps"]):
            best_at_recall[recall] = row
    frontier_desc: list[dict[str, object]] = []
    best_qps = -math.inf
    for recall in sorted(best_at_recall, reverse=True):
        row = best_at_recall[recall]
        qps = float(row["qps"])
        if qps > best_qps:
            frontier_desc.append(row)
            best_qps = qps
    return list(reversed(frontier_desc))


def tq_select_r_from_pareto(
    dataset,
    split: Split,
    k: int,
    codec: Path,
    meta: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    root = RAW_ROOT / dataset.key / f"k{k}" / "tq"
    root.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, object]] = []
    for rerank_n in tq_reranks(k):
        efs = tq_pareto_efs(rerank_n)
        path = root / f"pareto.r{rerank_n}.csv"
        if args.force or not tq_pareto_raw_complete(
            path, dataset.key, k, rerank_n
        ):
            path.unlink(missing_ok=True)
            command = [
                "taskset", "-c", str(args.cpu),
                str(tq.search_runner.BINARY),
                "--dataset", dataset.key,
                "--method", "qdranttq1",
                "--metric", "ip",
                "--query", str(split.calibration_query),
                "--gt", str(split.calibration_gt),
                "--graph", str(dataset.adcbq_index),
                "--codec-index", str(codec),
                "--meta", str(meta),
                "--sidecar", str(RESULT_ROOT / "unused_resident_codes"),
                "--fp-sidecar", str(aligned_fp(dataset.key)),
                "--out", str(path),
                "--k", str(k),
                "--max-queries", str(split.calibration_rows),
                "--rounds", "1",
                "--threads", "1",
                "--rerank-n", str(rerank_n),
                "--resident-codes",
                "--inline-hnsw",
                "--fp-io-mode", "uring-batch",
                "--fp-rerank-qd", str(rerank_n),
                "--fp-overlap-queries",
                "--efs", *[str(ef) for ef in efs],
            ]
            run(command)
        if not tq_pareto_raw_complete(path, dataset.key, k, rerank_n):
            raise RuntimeError(f"incomplete TQ Pareto arm: {path}")
        for row in read_csv(path):
            sweep_rows.append(
                {
                    "dataset": dataset.key,
                    "k": k,
                    "efSearch": int(row["efSearch"]),
                    "rerank_n": rerank_n,
                    "recall_at_k": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "measurement_mode": row["measurement_mode"],
                    "source_csv": str(path.relative_to(REPO)),
                }
            )

    sweep_rows.sort(key=lambda row: (int(row["rerank_n"]), int(row["efSearch"])))
    sweep_path = root / "pareto_sweep.csv"
    write_csv(sweep_path, sweep_rows)
    frontier = pareto_recall_qps(sweep_rows)
    frontier_path = root / "pareto_frontier.csv"
    write_csv(frontier_path, frontier)
    feasible = [
        row for row in frontier if float(row["recall_at_k"]) >= TQ_TARGET_RECALL
    ]
    if feasible:
        chosen = max(feasible, key=lambda row: float(row["qps"]))
        target_reached = True
    else:
        chosen = max(
            frontier,
            key=lambda row: (float(row["recall_at_k"]), float(row["qps"])),
        )
        target_reached = False
    result: dict[str, object] = {
        "dataset": dataset.key,
        "k": k,
        "protocol": "max_qps_on_joint_ef_rerank_pareto",
        "target_recall": TQ_TARGET_RECALL,
        "target_reached": target_reached,
        "selected_r": int(chosen["rerank_n"]),
        "selected_ef": int(chosen["efSearch"]),
        "selected_recall_at_k": float(chosen["recall_at_k"]),
        "selected_qps": float(chosen["qps"]),
        "calibration_queries": split.calibration_rows,
        "rerank_sweep": list(tq_reranks(k)),
        "ef_sweep": list(TQ_PARETO_EFS),
        "sweep_csv": str(sweep_path.relative_to(REPO)),
        "frontier_csv": str(frontier_path.relative_to(REPO)),
    }
    selection_path = root / "pareto_selected_r.json"
    selection_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"TQ_PARETO {dataset.key} k={k} R={result['selected_r']} "
        f"ef={result['selected_ef']} recall={result['selected_recall_at_k']:.6f} "
        f"qps={result['selected_qps']:.2f} reached99={target_reached}",
        flush=True,
    )
    return result


def tq_raw_complete(
    path: Path,
    dataset_key: str,
    k: int,
    rerank_n: int,
    efs: tuple[int, ...],
    rounds: int,
) -> bool:
    rows = read_csv(path)
    expected = {(ef, round_id) for ef in efs for round_id in range(1, rounds + 1)}
    try:
        actual = {(int(row["efSearch"]), int(row["round"])) for row in rows}
        return actual == expected and all(
            row["dataset"] == dataset_key
            and int(row["k"]) == k
            and int(row["rerank_n"]) == rerank_n
            and int(row["threads"]) == 1
            and row["code_storage"] == "resident"
            and row["fp_io_mode"] == "uring-batch"
            and row["measurement_mode"] == "inter_query_overlap"
            and row["first_stage_search_impl"] == "inline_batched_prefetch"
            for row in rows
        )
    except (KeyError, ValueError):
        return False


def run_tq_curve(
    dataset,
    split: Split,
    k: int,
    rerank_n: int,
    efs: tuple[int, ...],
    codec: Path,
    meta: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    path = RAW_ROOT / dataset.key / f"k{k}" / "tq" / f"heldout.r{rerank_n}.csv"
    if args.force or not tq_raw_complete(
        path, dataset.key, k, rerank_n, efs, args.rounds
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "taskset", "-c", str(args.cpu),
            str(tq.search_runner.BINARY),
            "--dataset", dataset.key,
            "--method", "qdranttq1",
            "--metric", "ip",
            "--query", str(split.test_query),
            "--gt", str(split.test_gt),
            "--graph", str(dataset.adcbq_index),
            "--codec-index", str(codec),
            "--meta", str(meta),
            "--sidecar", str(RESULT_ROOT / "unused_resident_codes"),
            "--fp-sidecar", str(aligned_fp(dataset.key)),
            "--out", str(path),
            "--k", str(k),
            "--max-queries", str(split.test_rows),
            "--rounds", str(args.rounds),
            "--threads", "1",
            "--rerank-n", str(rerank_n),
            "--resident-codes",
            "--inline-hnsw",
            "--fp-io-mode", "uring-batch",
            "--fp-rerank-qd", str(rerank_n),
            "--fp-overlap-queries",
            "--efs", *[str(ef) for ef in efs],
        ]
        run(command)
    raw_rows = read_csv(path)
    output: list[dict[str, object]] = []
    for ef in efs:
        selected = [row for row in raw_rows if int(row["efSearch"]) == ef]
        qps_values = [float(row["qps"]) for row in selected]
        recall_values = [float(row["recall_at_10"]) for row in selected]
        output.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method_key": "turboquant_tq1_fp",
                "method": f"TurboQuant TQ1+ + FP@{rerank_n}",
                "k": k,
                "efSearch": ef,
                "recall_at_k": statistics.fmean(recall_values),
                "qps": statistics.median(qps_values),
                "qps_min": min(qps_values),
                "qps_max": max(qps_values),
                "rounds": args.rounds,
                "search_threads": 1,
                "rerank_policy": "pareto_selected_fixed_r",
                "rerank_n_or_mean": rerank_n,
                "fp_reads_per_query": rerank_n,
                "measurement_mode": "inter_query_overlap",
                "io_engine": "io_uring O_DIRECT inter-query overlap",
                "calibration_queries": CALIBRATION_ROWS,
                "test_queries": split.test_rows,
                "selection_target_recall": TQ_TARGET_RECALL,
                "selection_protocol": "max_qps_on_joint_ef_rerank_pareto",
                "tau": "",
                "source_csv": str(path.relative_to(REPO)),
            }
        )
    return output


def rabitq_raw_complete(
    path: Path, dataset_key: str, k: int, efs: tuple[int, ...], rounds: int
) -> bool:
    rows = read_csv(path)
    expected = {(ef, round_id) for ef in efs for round_id in range(1, rounds + 1)}
    try:
        actual = {(int(row["efSearch"]), int(row["round"])) for row in rows}
        return actual == expected and all(
            row["dataset"] == dataset_key
            and int(row["k"]) == k
            and int(row["search_threads"]) == 1
            for row in rows
        )
    except (KeyError, ValueError):
        return False


def run_rabitq_curve(
    dataset,
    split: Split,
    k: int,
    efs: tuple[int, ...],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    path = RAW_ROOT / dataset.key / f"k{k}" / "rabitq" / "heldout.csv"
    if not args.force and rabitq_raw_complete(path, dataset.key, k, efs, args.rounds):
        raw_rows = read_csv(path)
    else:
        index_path = RABITQ_INDEXES[dataset.key]
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        queries = registry.read_fbin(split.test_query).copy()
        registry.normalize_rows(queries)
        gt = registry.read_ibin(split.test_gt, split.test_rows)
        index = HnswIndex.load(str(index_path))
        if index.nbits != 1 or index.num_clusters != 16 or index.dim != 1024:
            raise RuntimeError(f"unexpected RaBitQ index metadata: {index_path}")
        raw_rows: list[dict[str, object]] = []
        if path.is_file() and not args.force:
            prior_rows = read_csv(path)
            try:
                if all(
                    row["dataset"] == dataset.key
                    and int(row["k"]) == k
                    and int(row["efSearch"]) in efs
                    and 1 <= int(row["round"]) <= args.rounds
                    and int(row["search_threads"]) == 1
                    for row in prior_rows
                ):
                    raw_rows = [dict(row) for row in prior_rows]
            except (KeyError, ValueError):
                raw_rows = []
        completed = {
            (int(row["efSearch"]), int(row["round"])) for row in raw_rows
        }
        for ef in efs:
            for round_id in range(1, args.rounds + 1):
                if (ef, round_id) in completed:
                    print(
                        f"REUSE RABITQ {dataset.key} k={k} ef={ef} "
                        f"round={round_id}",
                        flush=True,
                    )
                    continue
                started = time.perf_counter()
                ids, _, stats = index.search_1bit_adaptive_fp_odirect(
                    queries,
                    str(aligned_fp(dataset.key)),
                    k=k,
                    ef=ef,
                    num_threads=1,
                )
                elapsed = time.perf_counter() - started
                reads = np.asarray(stats["fp_read_count"], dtype=np.float64)
                recall = recall_at_k(np.asarray(ids), gt, k)
                row: dict[str, object] = {
                    "dataset": dataset.key,
                    "k": k,
                    "efSearch": ef,
                    "round": round_id,
                    "search_threads": 1,
                    "nq": split.test_rows,
                    "recall_at_k": recall,
                    "qps": split.test_rows / elapsed,
                    "total_seconds": elapsed,
                    "fp_reads_per_query_mean": float(np.mean(reads)),
                    "fp_reads_per_query_p95": float(np.percentile(reads, 95)),
                    "fp_reads_per_query_p99": float(np.percentile(reads, 99)),
                    "fp_reads_per_query_max": int(np.max(reads)),
                    "odirect_read_bytes": int(stats["odirect_read_bytes"]),
                    "odirect_read_requests": int(stats["odirect_read_requests"]),
                    "index_path": str(index_path),
                }
                raw_rows.append(row)
                raw_rows.sort(
                    key=lambda item: (
                        int(item["efSearch"]), int(item["round"])
                    )
                )
                write_csv(path, raw_rows)
                print(
                    f"RABITQ {dataset.key} k={k} ef={ef} round={round_id} "
                    f"recall={recall:.6f} qps={row['qps']:.2f} "
                    f"reads={row['fp_reads_per_query_mean']:.2f}",
                    flush=True,
                )
    output: list[dict[str, object]] = []
    for ef in efs:
        selected = [row for row in raw_rows if int(row["efSearch"]) == ef]
        qps_values = [float(row["qps"]) for row in selected]
        output.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method_key": "rabitq_1bit_adaptive_fp",
                "method": "RaBitQ 1-bit c16 + adaptive FP",
                "k": k,
                "efSearch": ef,
                "recall_at_k": statistics.fmean(
                    float(row["recall_at_k"]) for row in selected
                ),
                "qps": statistics.median(qps_values),
                "qps_min": min(qps_values),
                "qps_max": max(qps_values),
                "rounds": args.rounds,
                "search_threads": 1,
                "rerank_policy": "native_adaptive_lower_bound",
                "rerank_n_or_mean": statistics.fmean(
                    float(row["fp_reads_per_query_mean"]) for row in selected
                ),
                "fp_reads_per_query": statistics.fmean(
                    float(row["fp_reads_per_query_mean"]) for row in selected
                ),
                "io_engine": "sync pread O_DIRECT",
                "calibration_queries": CALIBRATION_ROWS,
                "test_queries": split.test_rows,
                "calibration_reference_ef": "",
                "calibration_retention_target": "",
                "tau": "",
                "source_csv": str(path.relative_to(REPO)),
            }
        )
    return output


_SUMMARY_ROWS: list[dict[str, object]] = []


def write_summary(rows: list[dict[str, object]], *, append_method: bool = True) -> None:
    global _SUMMARY_ROWS
    if rows:
        # Replace this dataset/k/method slice so resumes never duplicate rows.
        key = (
            str(rows[0]["dataset"]),
            int(rows[0]["k"]),
            str(rows[0]["method_key"]),
        )
        _SUMMARY_ROWS = [
            row
            for row in _SUMMARY_ROWS
            if (
                str(row["dataset"]),
                int(row["k"]),
                str(row["method_key"]),
            )
            != key
        ] + rows
    if _SUMMARY_ROWS:
        _SUMMARY_ROWS.sort(
            key=lambda row: (
                DATASET_KEYS.index(str(row["dataset"])),
                KS.index(int(row["k"])),
                str(row["method_key"]),
                int(row["efSearch"]),
            )
        )
        write_csv(SUMMARY_CSV, _SUMMARY_ROWS)


def plot_summary() -> None:
    rows = read_csv(SUMMARY_CSV)
    if not rows:
        raise FileNotFoundError(SUMMARY_CSV)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "adc_bq_calibrated": ("ADC-BQ per-ef pool-top-k@0.99", "#1f77b4", "o"),
        "rabitq_1bit_adaptive_fp": ("RaBitQ-1 + adaptive FP", "#9467bd", "X"),
        "turboquant_tq1_fp": ("TurboQuant TQ1+", "#8c564b", "D"),
    }
    fig, axes = plt.subplots(
        len(KS), len(DATASET_KEYS), figsize=(7.0, 12.0), squeeze=False
    )
    for row_index, k in enumerate(KS):
        for column_index, dataset_key in enumerate(DATASET_KEYS):
            axis = axes[row_index, column_index]
            for method_key, (label, color, marker) in styles.items():
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["dataset"] == dataset_key
                        and int(row["k"]) == k
                        and row["method_key"] == method_key
                    ),
                    key=lambda row: int(row["efSearch"]),
                )
                axis.plot(
                    [float(row["recall_at_k"]) for row in selected],
                    [float(row["qps"]) for row in selected],
                    label=label,
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.8,
                )
            corpus = registry.DATASETS[dataset_key].corpus
            axis.set_title(f"{corpus} Qwen-1024 — Recall@{k}")
            axis.set_xlabel(f"Recall@{k}")
            axis.set_ylabel("QPS (T=1, log scale)")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "NQ Qwen-1024: per-ef calibrated ADC-BQ vs RaBitQ vs TurboQuant\n"
        "seed-11 held-out remainder; Pareto-selected fixed-R TQ; T=1",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_ROOT / "nq_msmarco_qwen1024_recall_k30_50_100_t1"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {stem}.png/.pdf", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-gt", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    global _SUMMARY_ROWS
    args = parse_args()
    if args.plot_only:
        plot_summary()
        return
    if not ADC_BIN.is_file():
        raise FileNotFoundError(ADC_BIN)
    datasets = {
        key: dataset_with_top100(key, args.build_threads, args.force_gt)
        for key in DATASET_KEYS
    }
    splits = {key: make_split(dataset) for key, dataset in datasets.items()}
    codecs = {
        key: ensure_tq_codec(key, args.build_threads) for key in DATASET_KEYS
    }
    if args.prepare_only:
        print("Preparation complete", flush=True)
        return
    if SUMMARY_CSV.is_file() and not args.force:
        _SUMMARY_ROWS = [dict(row) for row in read_csv(SUMMARY_CSV)]

    for dataset_key in DATASET_KEYS:
        dataset = datasets[dataset_key]
        split = splits[dataset_key]
        codec, meta = codecs[dataset_key]
        for k in KS:
            tq_selection = tq_select_r_from_pareto(
                dataset, split, k, codec, meta, args
            )
            selected_r = int(tq_selection["selected_r"])
            efs = geometric_efs(max(k, selected_r), 20)
            adc_calibrations = {
                ef: fit_adc_tau_per_ef(dataset, split, k, ef, args)
                for ef in efs
            }
            manifest = {
                "dataset": dataset_key,
                "k": k,
                "efs": list(efs),
                "adc_calibration_protocol": ADC_PROTOCOL,
                "adc_target_pool_topk_coverage": ADC_TARGET_COVERAGE,
                "adc_tau_by_ef": {
                    str(ef): float(adc_calibrations[ef]["tau"]) for ef in efs
                },
                "tq_selected_r": selected_r,
                "tq_selection_protocol": tq_selection["protocol"],
                "tq_selection_target_recall": TQ_TARGET_RECALL,
                "tq_selection_target_reached": tq_selection["target_reached"],
                "tq_selection_ef": tq_selection["selected_ef"],
                "tq_selection_recall_at_k": tq_selection["selected_recall_at_k"],
                "calibration_rows": CALIBRATION_ROWS,
                "test_rows": split.test_rows,
                "seed": SEED,
                "rounds": args.rounds,
                "threads": 1,
            }
            manifest_path = RAW_ROOT / dataset_key / f"k{k}" / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            adc_rows = run_adc_curve(
                dataset,
                split,
                k,
                adc_calibrations,
                efs,
                args,
            )
            write_summary(adc_rows, append_method=False)
            tq_rows = run_tq_curve(
                dataset,
                split,
                k,
                selected_r,
                efs,
                codec,
                meta,
                args,
            )
            write_summary(tq_rows)
            rabitq_rows = run_rabitq_curve(dataset, split, k, efs, args)
            write_summary(rabitq_rows)
    plot_summary()
    print(f"WROTE {SUMMARY_CSV} ({len(read_csv(SUMMARY_CSV))} rows)", flush=True)


if __name__ == "__main__":
    main()
