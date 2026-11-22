#!/usr/bin/env python3
"""Calibrate TurboQuant's adaptive FP rerank prefix at target 0.99.

For every dataset and efSearch independently, the calibration statistic is the
largest TQ distance margin (from the approximate rank-10 anchor) needed to keep
all ten exact-FP top candidates inside the TQ candidate pool.  The finite-sample
split-conformal order is ceil((n_cal + 1) * 0.99).  Test search then reads and
reranks only the query-specific prefix selected by the fitted margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

# Pin BLAS/OpenMP before NumPy or the helper modules are imported.  The search
# subprocess is single-threaded too, but without this the Python coordinator can
# eagerly create a large idle BLAS worker pool and make process monitors look as
# though the benchmark itself is multi-threaded.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = "1"

import numpy as np

import diskveclab_full_matrix as matrix
import run_adcbq_calibrated_threshold_overlap as adc
import run_adcbq_vs_fp_hnsw_fullvec_ram as adc_pool
import run_diskveclab_tq_pq_opq_uring as tq
import run_faiss_hnsw_bq_budget_paged_codes as search
import run_nq_quora_adcbq_calibrated_io_rerank as adc_panel


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/turboquant_calibrated_target099_main4_t1"
CALIBRATION_ROOT = RESULT_ROOT / "calibration"
RAW_ROOT = RESULT_ROOT / "raw"
SUMMARY = RESULT_ROOT / "summary_t1.csv"
STATUS = RESULT_ROOT / "status.json"
ADC_RESULT_ROOT = REPO / "experiment/nq_qwen1024_calibrated_adcbq_current_load_t1"
ADC_CALIBRATION_ROOT = ADC_RESULT_ROOT / "calibration"
ADC_RAW_ROOT = ADC_RESULT_ROOT / "raw"
ADC_SUMMARY = ADC_RESULT_ROOT / "summary_t1.csv"
TARGET = 0.99
EFS = (
    80, 84, 88, 92, 96, 100, 104, 108, 112, 116,
    120, 128, 144, 160, 192, 224, 256, 320, 512, 1024,
)
DATASETS = (
    "nq_qwen06_1024",
    "dbpedia_qwen1024",
    "msmarco_qwen1024",
    "quora_qwen1024",
)
PROTOCOL = "per-ef-pool-top10-split-conformal"
PAGE_BYTES = 4096
TQ_INLINE_HNSW = False

FP_SIDECARS = {
    "nq_qwen06_1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp/"
        "nq_qwen06_1024.fp32.header4096.fbin"
    ),
    "dbpedia_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/adcbq_aligned4k_fp_transient/"
        "dbpedia_qwen1024.fp32.aligned4k.fbin"
    ),
    "msmarco_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/adcbq_aligned4k_fp_transient/"
        "msmarco_qwen1024.fp32.aligned4k.fbin"
    ),
    "quora_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp/"
        "quora_qwen1024.fp32.header4096.fbin"
    ),
}

QUORA_CODEC = (
    REPO
    / "experiment/msmarco_quora_qwen1024_recall_k30_50_100_t1/tq_indexes/"
    "quora_qwen1024/quora_qwen1024.qdranttq1.tqdata",
    REPO
    / "experiment/msmarco_quora_qwen1024_recall_k30_50_100_t1/tq_indexes/"
    "quora_qwen1024/quora_qwen1024.qdranttq1.meta",
)


def update_status(**updates: object) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if STATUS.is_file():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    if updates.get("phase") in {"starting", "complete"}:
        payload.pop("error", None)
    payload.update(updates)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], cpu: int) -> None:
    pinned = ["taskset", "-c", str(cpu), *command]
    print("RUN " + " ".join(pinned), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    subprocess.run(pinned, cwd=REPO, env=environment, check=True)


def codec_paths(dataset_key: str) -> tuple[Path, Path]:
    if dataset_key == "quora_qwen1024":
        return QUORA_CODEC
    return tq.codec_paths(dataset_key, "qdranttq1")


def ensure_codec(dataset_key: str, build_threads: int) -> tuple[Path, Path]:
    codec, meta = codec_paths(dataset_key)
    if codec.is_file() and codec.stat().st_size > 4096 and meta.is_file():
        print(f"REUSE_CODEC {dataset_key} {codec}", flush=True)
        return codec, meta
    if dataset_key == "quora_qwen1024":
        raise FileNotFoundError(codec)
    build_args = SimpleNamespace(
        build_threads=build_threads,
        train_size=100_000,
        opq_niter=1,
        opq_blas_threads=1,
    )
    tq.build_codec(
        dataset_key,
        tq.Method("qdranttq1", "TurboQuant TQ1+", 80),
        build_args,
    )
    if not codec.is_file() or not meta.is_file():
        raise RuntimeError(f"failed to build TQ codec for {dataset_key}")
    return codec, meta


def fbin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        rows, dim = np.fromfile(handle, dtype="<i4", count=2)
    return np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=8,
        shape=(int(rows), int(dim)),
    )


def normalized_queries(path: Path) -> np.ndarray:
    values = np.asarray(fbin(path), dtype=np.float32).copy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    np.maximum(norms, np.float32(1e-30), out=norms)
    values /= norms
    return values


def aligned_fp(path: Path, expected_dim: int) -> np.ndarray:
    with path.open("rb") as handle:
        rows, dim = np.fromfile(handle, dtype="<i4", count=2)
    if int(dim) != expected_dim:
        raise RuntimeError(f"FP sidecar dimension mismatch: {path}")
    stride_bytes = ((int(dim) * 4 + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    expected_bytes = PAGE_BYTES + int(rows) * stride_bytes
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"FP sidecar size mismatch: {path}")
    mapped = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=PAGE_BYTES,
        shape=(int(rows), stride_bytes // 4),
    )
    return mapped[:, : int(dim)]


def calibration_paths(dataset_key: str, ef: int) -> tuple[Path, Path, Path]:
    root = CALIBRATION_ROOT / dataset_key
    return (
        root / f"ef{ef}.fit.json",
        root / f"ef{ef}.candidates.csv",
        root / f"ef{ef}.candidate_summary.csv",
    )


def candidate_trace_complete(path: Path, nq: int, ef: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    rows = 0
    query_ids: set[int] = set()
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                query_ids.add(int(row["query_id"]))
                if int(row["candidate_k"]) != ef:
                    return False
                float(row["approx_distance"])
        return rows == nq * ef and query_ids == set(range(nq))
    except (OSError, KeyError, ValueError):
        return False


def generate_candidate_trace(
    dataset_key: str,
    ef: int,
    codec: Path,
    meta: Path,
    cpu: int,
    force: bool,
) -> Path:
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    _, trace, candidate_summary = calibration_paths(dataset_key, ef)
    if not force and candidate_trace_complete(trace, split.calibration_rows, ef):
        print(f"REUSE_TRACE {dataset_key} ef={ef}", flush=True)
        return trace
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.unlink(missing_ok=True)
    candidate_summary.unlink(missing_ok=True)
    command = [
        str(search.BINARY),
        "--dataset", dataset_key,
        "--method", "qdranttq1",
        "--metric", "ip",
        "--query", str(split.calibration_query),
        "--gt", str(split.calibration_gt),
        "--graph", str(dataset.adcbq_index),
        "--codec-index", str(codec),
        "--meta", str(meta),
        "--sidecar", str(RESULT_ROOT / "unused_resident_sidecar"),
        "--out", str(candidate_summary),
        "--max-queries", str(split.calibration_rows),
        "--rounds", "1",
        "--threads", "1",
        "--rerank-n", str(ef),
        "--resident-codes",
        "--candidate-trace", str(trace),
        "--candidate-trace-only",
        "--efs", str(ef),
    ]
    if TQ_INLINE_HNSW:
        command.append("--inline-hnsw")
    run(command, cpu)
    if not candidate_trace_complete(trace, split.calibration_rows, ef):
        raise RuntimeError(f"incomplete candidate trace: {trace}")
    return trace


def fit_tau(
    dataset_key: str,
    ef: int,
    trace: Path,
    force: bool,
) -> dict[str, object]:
    fit_path, _, _ = calibration_paths(dataset_key, ef)
    split = matrix.split_for(dataset_key)
    if fit_path.is_file() and not force:
        payload = json.loads(fit_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") == PROTOCOL
            and int(payload.get("efSearch", -1)) == ef
            and int(payload.get("calibration_queries", -1))
            == split.calibration_rows
            and float(payload.get("target_pool_top10_coverage", -1.0)) == TARGET
        ):
            print(
                f"REUSE_FIT {dataset_key} ef={ef} tau={float(payload['tau']):.9g}",
                flush=True,
            )
            return payload

    nq = split.calibration_rows
    ids = np.full((nq, ef), -1, dtype=np.int64)
    approx = np.full((nq, ef), np.inf, dtype=np.float32)
    with trace.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            ids[qi, rank] = int(row["vector_id"])
            approx[qi, rank] = float(row["approx_distance"])
    if np.any(ids < 0) or not np.all(np.isfinite(approx)):
        raise RuntimeError(f"invalid candidate trace: {trace}")

    queries = normalized_queries(split.calibration_query)
    vectors = aligned_fp(FP_SIDECARS[dataset_key], split.dim)
    requirements = np.zeros(nq, dtype=np.float64)
    exact_positions = np.empty((nq, 10), dtype=np.int32)
    for qi in range(nq):
        labels = ids[qi]
        exact_distances = -(np.asarray(vectors[labels]) @ queries[qi])
        order = np.lexsort((labels, exact_distances))[:10]
        exact_positions[qi] = order
        anchor = float(approx[qi, 9])
        requirements[qi] = max(
            0.0,
            max(float(approx[qi, int(position)]) - anchor for position in order),
        )
        if (qi + 1) % 100 == 0 or qi + 1 == nq:
            print(f"  EXACT {dataset_key} ef={ef} {qi + 1}/{nq}", flush=True)

    conformal_order = min(nq, math.ceil((nq + 1) * TARGET))
    tau = float(np.partition(requirements, conformal_order - 1)[conformal_order - 1])
    counts = np.empty(nq, dtype=np.int32)
    covered = np.empty(nq, dtype=np.bool_)
    for qi in range(nq):
        cutoff = float(approx[qi, 9]) + tau
        count = int(np.searchsorted(approx[qi], cutoff, side="right"))
        count = max(10, min(ef, count))
        counts[qi] = count
        covered[qi] = bool(np.all(exact_positions[qi] < count))

    payload: dict[str, object] = {
        "dataset": dataset_key,
        "protocol": PROTOCOL,
        "calibration_reference": "all exact-FP top10 within TQ candidate pool",
        "target_pool_top10_coverage": TARGET,
        "efSearch": ef,
        "candidate_cap": ef,
        "calibration_queries": nq,
        "conformal_order": conformal_order,
        "tau": tau,
        "empirical_all_top10_coverage": float(np.mean(covered)),
        "rerank_count_mean": float(np.mean(counts)),
        "rerank_count_min": int(np.min(counts)),
        "rerank_count_p50": float(np.percentile(counts, 50)),
        "rerank_count_p95": float(np.percentile(counts, 95)),
        "rerank_count_p99": float(np.percentile(counts, 99)),
        "rerank_count_max": int(np.max(counts)),
        "requirements_min": float(np.min(requirements)),
        "requirements_max": float(np.max(requirements)),
    }
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = fit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, fit_path)
    print(
        f"FIT {dataset_key} ef={ef} tau={tau:.9g} "
        f"coverage={float(np.mean(covered)):.4f} Rmean={float(np.mean(counts)):.2f}",
        flush=True,
    )
    return payload


def adc_calibration_paths(
    dataset_key: str, ef: int
) -> tuple[Path, Path, Path]:
    root = ADC_CALIBRATION_ROOT / dataset_key
    return (
        root / f"ef{ef}.fit.json",
        root / f"ef{ef}.candidates.csv",
        root / f"ef{ef}.candidate_summary.csv",
    )


def fit_adcbq(
    dataset_key: str,
    ef: int,
    *,
    cpu: int,
    build_threads: int,
    force: bool,
    keep_candidates: bool,
) -> dict[str, object]:
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    fit_path, candidate_path, summary_path = adc_calibration_paths(dataset_key, ef)
    if fit_path.is_file() and not force:
        payload = json.loads(fit_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") == PROTOCOL
            and int(payload.get("efSearch", -1)) == ef
            and int(payload.get("calibration_queries", -1))
            == split.calibration_rows
            and float(payload.get("target_pool_top10_coverage", -1.0)) == TARGET
        ):
            print(
                f"REUSE ADC_FIT {dataset_key} ef={ef} "
                f"tau={float(payload['tau']):.9g}",
                flush=True,
            )
            return payload

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)
    command = adc.common_command(
        dataset,
        split.calibration_query,
        split.calibration_gt,
        summary_path,
        ef,
        1,
        1,
        build_threads,
    )
    command += [
        "--rerank-n", str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out", str(candidate_path),
    ]
    run(command, cpu)
    adc_pool.TARGET_COVERAGE = TARGET
    fitted = adc_pool.fit_pool_top10_conformal(
        SimpleNamespace(dataset=dataset), split, candidate_path, ef
    )
    payload: dict[str, object] = {
        "dataset": dataset_key,
        "protocol": PROTOCOL,
        "calibration_reference": "all exact-FP top10 within ADC-BQ candidate pool",
        "target_pool_top10_coverage": TARGET,
        "candidate_cap": ef,
        **fitted,
    }
    write_json(fit_path, payload)
    if not keep_candidates:
        candidate_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    print(
        f"ADC_FIT {dataset_key} ef={ef} tau={float(payload['tau']):.9g} "
        f"coverage={float(payload['empirical_all_top10_coverage']):.4f} "
        f"Rmean={float(payload['rerank_count_mean']):.2f}",
        flush=True,
    )
    return payload


def benchmark_adcbq(
    dataset_key: str,
    ef: int,
    fit: dict[str, object],
    *,
    rounds: int,
    cpu: int,
    build_threads: int,
    force: bool,
) -> dict[str, object]:
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    output = ADC_RAW_ROOT / dataset_key / f"ef{ef}.csv"
    stats_output = ADC_RAW_ROOT / dataset_key / f"ef{ef}.search_stats.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if force:
        output.unlink(missing_ok=True)
        stats_output.unlink(missing_ok=True)
    command = adc.common_command(
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
        "--fp-rerank-path", str(FP_SIDECARS[dataset_key]),
        "--fp-rerank-overlap-queries",
        "--fp-rerank-qd", str(ef),
        "--calibrated-threshold-rerank-tau", repr(float(fit["tau"])),
        "--stats-out", str(stats_output),
    ]
    run(command, cpu)
    measured = adc_panel.summarize(
        dataset,
        split,
        ef=ef,
        threads=1,
        rounds=rounds,
        tau=float(fit["tau"]),
        result_path=output,
    )
    stats_rows = read_csv(stats_output)
    if len(stats_rows) != 1:
        raise RuntimeError(f"missing ADC-BQ search stats: {stats_output}")
    search_stats = stats_rows[0]
    row: dict[str, object] = {
        "dataset": dataset_key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method_key": "adc_bq_calibrated",
        "method": "ADC-BQ calibrated adaptive rerank",
        "efSearch": ef,
        "recall_at_10": measured["recall_at_10"],
        "qps": measured["qps"],
        "rounds": rounds,
        "search_threads": 1,
        "first_stage_search_impl": "inline_batched_prefetch",
        "first_stage_distance_evaluations_per_query": float(
            search_stats["ndis_mean"]
        ),
        "first_stage_upper_distance_evaluations_per_query": float(
            search_stats["upper_distance_eval_count_mean"]
        ),
        "first_stage_base_distance_evaluations_per_query": float(
            search_stats["base_distance_eval_count_mean"]
        ),
        "calibration_protocol": PROTOCOL,
        "target_pool_top10_coverage": TARGET,
        "calibration_queries": split.calibration_rows,
        "test_queries": split.test_rows,
        "conformal_order": fit["conformal_order"],
        "tau": fit["tau"],
        "calibration_empirical_all_top10_coverage": fit[
            "empirical_all_top10_coverage"
        ],
        "calibration_rerank_count_mean": fit["rerank_count_mean"],
        "rerank_count_mean": measured["rerank_vectors_per_query_mean"],
        "rerank_count_min": measured["rerank_vectors_per_query_min"],
        "rerank_count_p50": "",
        "rerank_count_p95": measured["rerank_vectors_per_query_p95"],
        "rerank_count_p99": measured["rerank_vectors_per_query_p99"],
        "rerank_count_max": measured["rerank_vectors_per_query_max"],
        "fp_reads_per_query": measured["rerank_vectors_per_query_mean"],
        "fp_4k_pages_per_query": measured[
            "fp_submitted_4k_pages_per_query_mean"
        ],
        "candidate_cap": ef,
        "M": 24,
        "efConstruction": 200,
        "code_storage": "resident ADC-BQ codes",
        "fp_storage": "SSD O_DIRECT aligned FP32",
        "io_engine": measured["io_engine"],
        "source_csv": str(output.relative_to(REPO)),
    }
    print(
        f"ADC_RESULT {dataset_key} ef={ef} "
        f"recall={float(row['recall_at_10']):.6f} "
        f"qps={float(row['qps']):.2f} "
        f"Rmean={float(row['rerank_count_mean']):.2f} "
        f"Rp95={float(row['rerank_count_p95']):.2f}",
        flush=True,
    )
    return row


def raw_path(dataset_key: str, ef: int) -> Path:
    return RAW_ROOT / dataset_key / f"ef{ef}.csv"


def raw_complete(path: Path, dataset_key: str, ef: int, rounds: int, tau: float) -> bool:
    rows = read_csv(path)
    try:
        return len(rows) == rounds and all(
            row["dataset"] == dataset_key
            and int(row["efSearch"]) == ef
            and int(row["round"]) in range(1, rounds + 1)
            and int(row["threads"]) == 1
            and row["adaptive_rerank"] == "true"
            and row["measurement_mode"] == "inter_query_overlap"
            and abs(float(row["adaptive_tau"]) - tau) <= 1e-7 * max(1.0, abs(tau))
            for row in rows
        )
    except (KeyError, ValueError):
        return False


def benchmark(
    dataset_key: str,
    ef: int,
    fit: dict[str, object],
    codec: Path,
    meta: Path,
    rounds: int,
    cpu: int,
    force: bool,
) -> Path:
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    output = raw_path(dataset_key, ef)
    tau = float(fit["tau"])
    if not force and raw_complete(output, dataset_key, ef, rounds, tau):
        print(f"REUSE_TEST {dataset_key} ef={ef}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    qd = ef
    command = [
        str(search.BINARY),
        "--dataset", dataset_key,
        "--method", "qdranttq1",
        "--metric", "ip",
        "--query", str(split.test_query),
        "--gt", str(split.test_gt),
        "--graph", str(dataset.adcbq_index),
        "--codec-index", str(codec),
        "--meta", str(meta),
        "--sidecar", str(RESULT_ROOT / "unused_resident_sidecar"),
        "--fp-sidecar", str(FP_SIDECARS[dataset_key]),
        "--out", str(output),
        "--max-queries", str(split.test_rows),
        "--rounds", str(rounds),
        "--threads", "1",
        "--rerank-n", str(ef),
        "--adaptive-rerank-tau", format(tau, ".10g"),
        "--resident-codes",
        "--fp-io-mode", "uring-batch",
        "--fp-rerank-qd", str(qd),
        "--fp-overlap-queries",
        "--efs", str(ef),
    ]
    if TQ_INLINE_HNSW:
        command.append("--inline-hnsw")
    run(command, cpu)
    if not raw_complete(output, dataset_key, ef, rounds, tau):
        raise RuntimeError(f"incomplete adaptive test result: {output}")
    return output


def summary_row(
    dataset_key: str,
    ef: int,
    fit: dict[str, object],
    raw: Path,
) -> dict[str, object]:
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    rows = read_csv(raw)
    qps = [float(row["qps"]) for row in rows]
    recalls = [float(row["recall_at_10"]) for row in rows]
    r_mean = [float(row["adaptive_rerank_count_mean"]) for row in rows]
    physical_r = [float(row["fp_page_reads_per_query"]) for row in rows]
    if max(abs(left - right) for left, right in zip(r_mean, physical_r)) > 1e-6:
        raise RuntimeError(f"adaptive R/read accounting mismatch: {raw}")
    return {
        "dataset": dataset_key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method_key": "turboquant_tq1_calibrated",
        "method": "TurboQuant TQ1+ calibrated adaptive rerank",
        "efSearch": ef,
        "recall_at_10": statistics.median(recalls),
        "qps": statistics.median(qps),
        "qps_stddev": statistics.pstdev(qps),
        "rounds": len(rows),
        "search_threads": 1,
        "first_stage_search_impl": rows[0].get(
            "first_stage_search_impl", "faiss"
        ),
        "first_stage_distance_evaluations_per_query": statistics.median(
            [
                float(row.get("first_stage_distance_evaluations_per_query", 0.0))
                for row in rows
            ]
        ),
        "first_stage_upper_distance_evaluations_per_query": statistics.median(
            [
                float(
                    row.get(
                        "first_stage_upper_distance_evaluations_per_query", 0.0
                    )
                )
                for row in rows
            ]
        ),
        "first_stage_base_distance_evaluations_per_query": statistics.median(
            [
                float(
                    row.get(
                        "first_stage_base_distance_evaluations_per_query", 0.0
                    )
                )
                for row in rows
            ]
        ),
        "first_stage_scalar_distance_calls": statistics.median(
            [float(row.get("first_stage_scalar_distance_calls", 0.0)) for row in rows]
        ),
        "first_stage_batch4_calls": statistics.median(
            [float(row.get("first_stage_batch4_calls", 0.0)) for row in rows]
        ),
        "calibration_protocol": PROTOCOL,
        "target_pool_top10_coverage": TARGET,
        "calibration_queries": split.calibration_rows,
        "test_queries": split.test_rows,
        "conformal_order": int(fit["conformal_order"]),
        "tau": float(fit["tau"]),
        "calibration_empirical_all_top10_coverage": float(
            fit["empirical_all_top10_coverage"]
        ),
        "calibration_rerank_count_mean": float(fit["rerank_count_mean"]),
        "rerank_count_mean": statistics.median(r_mean),
        "rerank_count_min": statistics.median(
            [float(row["adaptive_rerank_count_min"]) for row in rows]
        ),
        "rerank_count_p50": statistics.median(
            [float(row["adaptive_rerank_count_p50"]) for row in rows]
        ),
        "rerank_count_p95": statistics.median(
            [float(row["adaptive_rerank_count_p95"]) for row in rows]
        ),
        "rerank_count_p99": statistics.median(
            [float(row["adaptive_rerank_count_p99"]) for row in rows]
        ),
        "rerank_count_max": statistics.median(
            [float(row["adaptive_rerank_count_max"]) for row in rows]
        ),
        "fp_reads_per_query": statistics.median(physical_r),
        "fp_4k_pages_per_query": statistics.median(
            [float(row["fp_read_bytes_per_query"]) / PAGE_BYTES for row in rows]
        ),
        "candidate_cap": ef,
        "M": 24,
        "efConstruction": 200,
        "code_storage": "resident",
        "fp_storage": "SSD O_DIRECT aligned FP32",
        "io_engine": "io_uring inter-query overlap with per-query adaptive batch",
        "source_csv": str(raw.relative_to(REPO)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--efs", nargs="+", type=int, default=EFS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-traces", action="store_true")
    parser.add_argument(
        "--paired-nq-adcbq",
        "--paired-adcbq",
        dest="paired_nq_adcbq",
        action="store_true",
        help="interleave ADC-BQ calibration/QPS with TurboQuant at every ef",
    )
    parser.add_argument(
        "--wait-pid",
        type=int,
        help="wait for this competing process to exit before benchmarking",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=0,
        help="extra idle time after --wait-pid exits",
    )
    args = parser.parse_args()
    if args.rounds <= 0 or args.build_threads <= 0:
        parser.error("rounds and build-threads must be positive")
    if any(ef < 10 for ef in args.efs):
        parser.error("all efSearch values must be at least 10")
    if args.wait_pid is not None and args.wait_pid <= 0:
        parser.error("--wait-pid must be positive")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    if args.wait_pid is not None:
        process_path = Path(f"/proc/{args.wait_pid}")
        while process_path.exists():
            print(
                f"WAIT competing pid={args.wait_pid} before clean rerun",
                flush=True,
            )
            time.sleep(30)
        if args.cooldown_seconds:
            print(
                f"COOLDOWN {args.cooldown_seconds}s after pid={args.wait_pid} exited",
                flush=True,
            )
            time.sleep(args.cooldown_seconds)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    update_status(
        phase="starting",
        datasets=list(args.datasets),
        efs=list(args.efs),
        rounds=args.rounds,
        target=TARGET,
    )
    try:
        search.compile_binary()
        tq.codec_builder.compile_binary()
        if args.paired_nq_adcbq:
            adc.BIN = (
                REPO
                / "experiment/adcbq_optimized/"
                "faiss_adcbq_hnsw_same_pool_calibration"
            )
            adc.RESULT_DIR = ADC_RAW_ROOT
            adc.ensure_binary()
        all_rows: list[dict[str, object]] = []
        adc_rows: list[dict[str, object]] = []
        for dataset_key in args.datasets:
            dataset = matrix.registry.DATASETS[dataset_key]
            matrix.registry.validate(dataset)
            split = matrix.split_for(dataset_key)
            for required in (
                split.calibration_query,
                split.calibration_gt,
                split.test_query,
                split.test_gt,
                dataset.adcbq_index,
                FP_SIDECARS[dataset_key],
            ):
                if not required.is_file():
                    raise FileNotFoundError(required)
            update_status(phase="codec", current_dataset=dataset_key)
            codec, meta = ensure_codec(dataset_key, args.build_threads)
            for ef in args.efs:
                update_status(
                    phase="calibration",
                    current_dataset=dataset_key,
                    current_ef=ef,
                )
                fit_path, trace_path, _ = calibration_paths(dataset_key, ef)
                fit: dict[str, object] | None = None
                if fit_path.is_file() and not args.force:
                    candidate = json.loads(fit_path.read_text(encoding="utf-8"))
                    if (
                        candidate.get("protocol") == PROTOCOL
                        and int(candidate.get("efSearch", -1)) == ef
                        and int(candidate.get("calibration_queries", -1))
                        == split.calibration_rows
                        and float(candidate.get("target_pool_top10_coverage", -1.0))
                        == TARGET
                    ):
                        fit = candidate
                        print(
                            f"REUSE_FIT {dataset_key} ef={ef} "
                            f"tau={float(fit['tau']):.9g}",
                            flush=True,
                        )
                if fit is None:
                    trace = generate_candidate_trace(
                        dataset_key, ef, codec, meta, args.cpu, args.force
                    )
                    fit = fit_tau(dataset_key, ef, trace, args.force)
                    if not args.keep_traces:
                        trace.unlink(missing_ok=True)
                elif not args.keep_traces:
                    trace_path.unlink(missing_ok=True)
                adc_fit: dict[str, object] | None = None
                if args.paired_nq_adcbq:
                    update_status(
                        phase="adc_calibration",
                        current_dataset=dataset_key,
                        current_ef=ef,
                    )
                    adc_fit = fit_adcbq(
                        dataset_key,
                        ef,
                        cpu=args.cpu,
                        build_threads=args.build_threads,
                        force=args.force,
                        keep_candidates=args.keep_traces,
                    )
                update_status(
                    phase="test",
                    current_dataset=dataset_key,
                    current_ef=ef,
                    tau=float(fit["tau"]),
                )
                raw = benchmark(
                    dataset_key,
                    ef,
                    fit,
                    codec,
                    meta,
                    args.rounds,
                    args.cpu,
                    args.force,
                )
                row = summary_row(dataset_key, ef, fit, raw)
                all_rows.append(row)
                write_csv(SUMMARY, all_rows)
                print(
                    f"RESULT {dataset_key} ef={ef} "
                    f"recall={float(row['recall_at_10']):.6f} "
                    f"qps={float(row['qps']):.2f} "
                    f"Rmean={float(row['rerank_count_mean']):.2f} "
                    f"Rp95={float(row['rerank_count_p95']):.2f}",
                    flush=True,
                )
                if adc_fit is not None:
                    update_status(
                        phase="adc_test",
                        current_dataset=dataset_key,
                        current_ef=ef,
                        adc_tau=float(adc_fit["tau"]),
                    )
                    adc_row = benchmark_adcbq(
                        dataset_key,
                        ef,
                        adc_fit,
                        rounds=args.rounds,
                        cpu=args.cpu,
                        build_threads=args.build_threads,
                        force=args.force,
                    )
                    adc_rows.append(adc_row)
                    write_csv(ADC_SUMMARY, adc_rows)
        update_status(
            phase="complete",
            current_dataset=None,
            current_ef=None,
            summary=str(SUMMARY),
            rows=len(all_rows),
        )
        return 0
    except BaseException as error:
        update_status(phase="failed", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
