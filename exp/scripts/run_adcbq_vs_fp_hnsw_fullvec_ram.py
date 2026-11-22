#!/usr/bin/env python3
"""Compare calibrated ADC-BQ with FP-HNSW when both own full FP32 vectors in RAM.

The two methods use the same FP-built M=24/efConstruction=200 HNSW topology.
ADC-BQ uses its packed 1-bit traversal codes and a per-ef split-conformal
threshold that retains the exact-FP top ten inside the ADC candidate pool with
target probability 0.99.  Exact refinement reads from a process-owned copy of
the complete FP32 base matrix (``--fp-rerank-source ram``).  FP-HNSW attaches a
process-owned normalized FP32 store to a copy of that same topology.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiment"
sys.path.insert(0, str(EXPERIMENT))

import run_adcbq_calibrated_threshold_overlap as adc_cal  # noqa: E402
import run_v051_ssd_odirect_multidataset as registry  # noqa: E402


SOURCE = EXPERIMENT / "faiss_fp_hnsw_resident_from_binary_graph.cpp"
FP_BINARY = EXPERIMENT / "faiss_fp_hnsw_resident_from_binary_graph"
ADC_BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_calibrated"
ADC_BUILD = EXPERIMENT / "adcbq_optimized/build.sh"
RESULT_ROOT = EXPERIMENT / "adcbq_vs_fp_hnsw_fullvec_ram"
RESULT_DIR = RESULT_ROOT
DEFAULT_EFS = (
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    512,
    640,
    768,
    1024,
)
TARGET_COVERAGE = 0.99
M = 24
EF_CONSTRUCTION = 200
K = 10


@dataclass(frozen=True)
class Workload:
    key: str
    title: str
    calibration_queries: int
    max_test_queries: int

    @property
    def dataset(self) -> registry.Dataset:
        return registry.DATASETS[self.key]


WORKLOADS = {
    "msmarco_qwen1024": Workload(
        "msmarco_qwen1024", "MSMARCO Qwen-1024 (1M)", 1000, 0
    ),
    "dbpedia_qwen1024": Workload(
        "dbpedia_qwen1024", "DBpedia Qwen-1024 (1M)", 100, 300
    ),
}


def compile_binaries() -> None:
    env = os.environ.copy()
    env["OUT"] = str(ADC_BINARY)
    subprocess.run([str(ADC_BUILD)], cwd=REPO, env=env, check=True)

    faiss_root = Path("/home/jihyo/workspace/faiss")
    faiss_build = faiss_root / "build_hnsw_conda/faiss"
    conda_lib = Path("/home/jihyo/anaconda3/envs/hnsw/lib")
    command = [
        "g++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-march=native",
        "-fopenmp",
        f"-I{faiss_root}",
        f"-I{faiss_build}",
        str(SOURCE),
        f"-L{faiss_build}",
        f"-L{conda_lib}",
        f"-Wl,-rpath,{faiss_build}:{conda_lib}",
        "-lfaiss_avx2",
        "-lopenblas",
        "-o",
        str(FP_BINARY),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def run_command(command: list[str], threads: int) -> None:
    print("RUN " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }
    )
    subprocess.run(command, cwd=REPO, env=env, check=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def split_for(workload: Workload) -> adc_cal.SplitFiles:
    return adc_cal.make_split(
        workload.dataset,
        workload.calibration_queries,
        11,
        workload.max_test_queries,
    )


def adc_common(
    workload: Workload,
    query: Path,
    gt: Path,
    output: Path,
    ef: int,
    rounds: int,
    threads: int,
) -> list[str]:
    dataset = workload.dataset
    return [
        str(ADC_BINARY),
        "--dataset",
        workload.key,
        "--train",
        str(dataset.train),
        "--query",
        str(query),
        "--gt",
        str(gt),
        "--index",
        str(dataset.adcbq_index),
        "--out",
        str(output),
        "--m",
        str(M),
        "--ef-construction",
        str(EF_CONSTRUCTION),
        "--k",
        str(K),
        "--max-queries",
        "0",
        "--rounds",
        str(rounds),
        "--build-threads",
        str(threads),
        "--search-threads",
        str(threads),
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


def candidate_paths(workload: Workload, ef: int) -> tuple[Path, Path]:
    root = RESULT_DIR / workload.key / "calibration"
    return root / f"ef{ef}.summary.csv", root / f"ef{ef}.candidates.csv"


def dump_candidates(
    workload: Workload,
    split: adc_cal.SplitFiles,
    ef: int,
    threads: int,
    force: bool,
) -> Path:
    summary, candidates = candidate_paths(workload, ef)
    expected_lines = split.calibration_rows * ef + 1
    if not force and summary.is_file() and candidates.is_file():
        with candidates.open(encoding="utf-8") as handle:
            if sum(1 for _ in handle) == expected_lines:
                return candidates
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = adc_common(
        workload, split.calibration_query, split.calibration_gt, summary, ef, 1, threads
    )
    command += [
        "--rerank-n",
        str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out",
        str(candidates),
    ]
    run_command(command, threads)
    return candidates


def read_fbin(path: Path) -> np.ndarray:
    rows, dim = adc_cal.matrix_header(path)
    payload_bytes = rows * dim * np.dtype(np.float32).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes == 8 + payload_bytes:
        data_offset = 8
    elif actual_bytes == 4096 + payload_bytes:
        data_offset = 4096
    else:
        raise ValueError(f"unexpected fbin size: {path}")
    mapped = np.memmap(
        path, dtype=np.float32, mode="r", offset=data_offset, shape=(rows, dim)
    )
    return np.array(mapped, dtype=np.float32, order="C", copy=True)


def fit_pool_top10_conformal(
    workload: Workload,
    split: adc_cal.SplitFiles,
    candidate_path: Path,
    ef: int,
) -> dict[str, object]:
    ids, distances = adc_cal.load_candidate_csv(
        candidate_path, split.calibration_rows, ef
    )
    queries = read_fbin(split.calibration_query)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= np.maximum(norms, 1e-30)

    dataset = workload.dataset
    ntrain, dim = adc_cal.matrix_header(dataset.train)
    payload_bytes = ntrain * dim * np.dtype(np.float32).itemsize
    actual_bytes = dataset.train.stat().st_size
    if actual_bytes == 8 + payload_bytes:
        train_offset = 8
    elif actual_bytes == 4096 + payload_bytes:
        train_offset = 4096
    else:
        raise ValueError(f"unexpected fbin size: {dataset.train}")
    train = np.memmap(
        dataset.train,
        dtype=np.float32,
        mode="r",
        offset=train_offset,
        shape=(ntrain, dim),
    )
    scales = adc_cal.int16_query_scales(split.calibration_query)
    requirements = np.zeros(split.calibration_rows, dtype=np.float64)
    exact_ranks = np.empty((split.calibration_rows, K), dtype=np.int32)
    for query_id in range(split.calibration_rows):
        valid = ids[query_id] >= 0
        row_ids = ids[query_id, valid]
        row_distances = distances[query_id, valid]
        vectors = np.asarray(train[row_ids], dtype=np.float32)
        vector_norms = np.linalg.norm(vectors, axis=1)
        scores = vectors @ queries[query_id]
        scores = np.divide(
            scores,
            vector_norms,
            out=np.full_like(scores, -np.inf),
            where=vector_norms > 0,
        )
        exact_order = np.lexsort((row_ids, -scores))
        exact_positions = exact_order[:K]
        exact_ranks[query_id] = exact_positions.astype(np.int32) + 1
        anchor = float(row_distances[K - 1])
        needed = row_distances[exact_positions] - anchor
        requirements[query_id] = max(
            0.0, float(np.max(needed)) / float(scales[query_id])
        )

    order = math.ceil((split.calibration_rows + 1) * TARGET_COVERAGE)
    if order > split.calibration_rows:
        raise RuntimeError(
            f"target {TARGET_COVERAGE} is infeasible with "
            f"{split.calibration_rows} calibration queries"
        )
    tau = float(np.partition(requirements, order - 1)[order - 1])
    cutoffs = distances[:, K - 1] + tau * scales
    counts = np.maximum(K, np.sum(distances <= cutoffs[:, None], axis=1))
    counts = np.minimum(counts, ef).astype(np.int32)
    retained = np.all(exact_ranks <= counts[:, None], axis=1)
    return {
        "efSearch": ef,
        "tau": tau,
        "target_all_top10_coverage": TARGET_COVERAGE,
        "empirical_all_top10_coverage": float(np.mean(retained)),
        "conformal_order": order,
        "calibration_queries": split.calibration_rows,
        "reference": "exact FP top-10 within the ADC-HNSW candidate pool",
        "rerank_count_mean": float(np.mean(counts)),
        "rerank_count_min": int(np.min(counts)),
        "rerank_count_p95": float(np.percentile(counts, 95)),
        "rerank_count_p99": float(np.percentile(counts, 99)),
        "rerank_count_max": int(np.max(counts)),
    }


def calibrate(
    workload: Workload,
    split: adc_cal.SplitFiles,
    efs: list[int],
    threads: int,
    force: bool,
    keep_candidates: bool,
) -> dict[int, dict[str, object]]:
    path = RESULT_DIR / workload.key / "calibration_target099.json"
    saved: dict[str, dict[str, object]] = {}
    if path.is_file() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") == "per_ef_pool_top10_split_conformal":
            saved = payload.get("operating_points", {})
    for ef in efs:
        if str(ef) in saved and not force:
            continue
        candidates = dump_candidates(workload, split, ef, threads, force)
        fit = fit_pool_top10_conformal(workload, split, candidates, ef)
        saved[str(ef)] = fit
        write_json(
            path,
            {
                "dataset": workload.key,
                "protocol": "per_ef_pool_top10_split_conformal",
                "target": TARGET_COVERAGE,
                "split_seed": 11,
                "calibration_queries": split.calibration_rows,
                "test_queries": split.test_rows,
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "operating_points": saved,
            },
        )
        print(
            f"CALIBRATED {workload.key} ef={ef}: tau={fit['tau']:.9g} "
            f"Rmean={fit['rerank_count_mean']:.2f} "
            f"coverage={fit['empirical_all_top10_coverage']:.4f}",
            flush=True,
        )
        if not keep_candidates:
            candidates.unlink(missing_ok=True)
    return {int(key): value for key, value in saved.items() if int(key) in efs}


def adc_result_valid(
    path: Path,
    split: adc_cal.SplitFiles,
    rounds: int,
    threads: int,
    require_median: bool,
) -> bool:
    try:
        row = read_single_csv(path)
        return (
            int(row["nq"]) == split.test_rows
            and int(row["rounds"]) == rounds
            and int(row["search_threads"]) == threads
            and row["fp_rerank_source"] == "ram"
            and (not require_median or float(row["qps_median"]) > 0.0)
        )
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def benchmark_adcbq(
    workload: Workload,
    split: adc_cal.SplitFiles,
    calibration: dict[int, dict[str, object]],
    efs: list[int],
    rounds: int,
    threads: int,
    force: bool,
) -> list[Path]:
    outputs: list[Path] = []
    require_median = workload.key == "dbpedia_qwen1024"
    for ef in efs:
        output = RESULT_DIR / workload.key / f"adcbq_t{threads}" / f"ef{ef}.csv"
        outputs.append(output)
        if not force and adc_result_valid(
            output, split, rounds, threads, require_median
        ):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = adc_common(
            workload, split.test_query, split.test_gt, output, ef, rounds, threads
        )
        command += [
            "--rerank-n",
            str(ef),
            "--rerank-dot",
            "fp32",
            "--fp-rerank-source",
            "ram",
            "--fp-rerank-norms",
            "precompute",
            "--calibrated-threshold-rerank-tau",
            repr(float(calibration[ef]["tau"])),
        ]
        run_command(command, threads)
    return outputs


def fp_result_valid(
    path: Path,
    split: adc_cal.SplitFiles,
    efs: list[int],
    rounds: int,
    threads: int,
    require_median: bool,
) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return (
            [int(row["efSearch"]) for row in rows] == efs
            and all(int(row["nq"]) == split.test_rows for row in rows)
            and all(int(row["rounds"]) == rounds for row in rows)
            and all(int(row["threads"]) == threads for row in rows)
            and all(row["full_vector_storage"] == "process_owned_ram" for row in rows)
            and (
                not require_median
                or all(float(row["qps_median"]) > 0.0 for row in rows)
            )
        )
    except (OSError, KeyError, ValueError):
        return False


def benchmark_fp_hnsw(
    workload: Workload,
    split: adc_cal.SplitFiles,
    efs: list[int],
    rounds: int,
    threads: int,
    force: bool,
) -> Path:
    output = RESULT_DIR / workload.key / f"fp_hnsw_t{threads}.csv"
    require_median = workload.key == "dbpedia_qwen1024"
    if not force and fp_result_valid(
        output, split, efs, rounds, threads, require_median
    ):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = workload.dataset
    command = [
        str(FP_BINARY),
        "--dataset",
        workload.key,
        "--train",
        str(dataset.train),
        "--query",
        str(split.test_query),
        "--gt",
        str(split.test_gt),
        "--graph",
        str(dataset.adcbq_index),
        "--out",
        str(output),
        "--m",
        str(M),
        "--ef-construction",
        str(EF_CONSTRUCTION),
        "--k",
        str(K),
        "--rounds",
        str(rounds),
        "--threads",
        str(threads),
        "--efs",
        *[str(ef) for ef in efs],
    ]
    run_command(command, threads)
    return output


def combine(
    workloads: list[Workload],
    splits: dict[str, adc_cal.SplitFiles],
    calibrations: dict[str, dict[int, dict[str, object]]],
    adc_outputs: dict[str, list[Path]],
    fp_outputs: dict[str, Path],
    threads: int,
) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for workload in workloads:
        dataset = workload.dataset
        split = splits[workload.key]
        qps_column = "qps_median" if workload.key == "dbpedia_qwen1024" else "qps"
        qps_statistic = "median_20" if workload.key == "dbpedia_qwen1024" else "mean_5"
        vector_bytes = dataset.train.stat().st_size - 8
        index_bytes = dataset.adcbq_index.stat().st_size
        for path in adc_outputs[workload.key]:
            raw = read_single_csv(path)
            ef = int(raw["efSearch"])
            fit = calibrations[workload.key][ef]
            qps = float(raw[qps_column])
            rows.append(
                {
                    "dataset": workload.key,
                    "dataset_title": workload.title,
                    "method": "Calibrated ADC-BQ",
                    "efSearch": ef,
                    "recall_at_10": float(raw["recall_at_k"]),
                    "qps": qps,
                    "qps_statistic": qps_statistic,
                    "latency_ms_per_query": 1000.0 / qps,
                    "rounds": int(raw["rounds"]),
                    "threads": int(raw["search_threads"]),
                    "test_queries": split.test_rows,
                    "calibration_queries": split.calibration_rows,
                    "calibration_tau": float(fit["tau"]),
                    "rerank_vectors_mean": float(raw["adaptive_rerank_count_mean"]),
                    "rerank_vectors_p95": float(raw["adaptive_rerank_count_p95"]),
                    "full_vector_storage": "process_owned_ram",
                    "resident_full_vector_bytes": vector_bytes,
                    "compressed_index_or_graph_bytes": index_bytes,
                    "modeled_resident_bytes": vector_bytes + index_bytes + 4_000_000,
                    "process_rss_bytes": "",
                    "same_topology": True,
                    "source_csv": str(path),
                }
            )
        with fp_outputs[workload.key].open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                qps = float(raw[qps_column])
                rows.append(
                    {
                        "dataset": workload.key,
                        "dataset_title": workload.title,
                        "method": "FP-HNSW",
                        "efSearch": int(raw["efSearch"]),
                        "recall_at_10": float(raw["recall_at_k"]),
                        "qps": qps,
                        "qps_statistic": qps_statistic,
                        "latency_ms_per_query": 1000.0 / qps,
                        "rounds": int(raw["rounds"]),
                        "threads": int(raw["threads"]),
                        "test_queries": split.test_rows,
                        "calibration_queries": 0,
                        "calibration_tau": "",
                        "rerank_vectors_mean": 0,
                        "rerank_vectors_p95": 0,
                        "full_vector_storage": raw["full_vector_storage"],
                        "resident_full_vector_bytes": int(raw["resident_vector_bytes"]),
                        "compressed_index_or_graph_bytes": int(raw["graph_template_bytes"]),
                        "modeled_resident_bytes": "",
                        "process_rss_bytes": int(raw["process_rss_bytes"]),
                        "same_topology": True,
                        "source_csv": str(fp_outputs[workload.key]),
                    }
                )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["method"]), int(row["efSearch"])))
    combined = RESULT_DIR / f"recall_qps_fullvec_ram_t{threads}.csv"
    write_csv(combined, rows)

    target_rows: list[dict[str, object]] = []
    for workload in workloads:
        for target in (0.95, 0.98, 0.99, 0.995):
            selected: dict[str, dict[str, object] | None] = {}
            for method in ("Calibrated ADC-BQ", "FP-HNSW"):
                candidates = [
                    row
                    for row in rows
                    if row["dataset"] == workload.key
                    and row["method"] == method
                    and float(row["recall_at_10"]) >= target
                ]
                selected[method] = max(candidates, key=lambda row: float(row["qps"])) if candidates else None
            adc = selected["Calibrated ADC-BQ"]
            fp = selected["FP-HNSW"]
            target_rows.append(
                {
                    "dataset": workload.key,
                    "target_recall": target,
                    "adcbq_reached": adc is not None,
                    "adcbq_efSearch": "" if adc is None else adc["efSearch"],
                    "adcbq_recall": "" if adc is None else adc["recall_at_10"],
                    "adcbq_qps": "" if adc is None else adc["qps"],
                    "adcbq_rerank_mean": "" if adc is None else adc["rerank_vectors_mean"],
                    "fp_hnsw_reached": fp is not None,
                    "fp_hnsw_efSearch": "" if fp is None else fp["efSearch"],
                    "fp_hnsw_recall": "" if fp is None else fp["recall_at_10"],
                    "fp_hnsw_qps": "" if fp is None else fp["qps"],
                    "adcbq_over_fp_qps": (
                        ""
                        if adc is None or fp is None
                        else float(adc["qps"]) / float(fp["qps"])
                    ),
                }
            )
    targets = RESULT_DIR / f"target_recall_summary_t{threads}.csv"
    write_csv(targets, target_rows)
    return combined, targets


def plot(combined: Path, threads: int) -> tuple[Path, Path] | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plot", flush=True)
        return None
    with combined.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    for axis, workload in zip(axes, WORKLOADS.values()):
        for method, marker, color in (
            ("Calibrated ADC-BQ", "o", "#1f77b4"),
            ("FP-HNSW", "s", "#d62728"),
        ):
            points = [
                row
                for row in rows
                if row["dataset"] == workload.key and row["method"] == method
            ]
            points.sort(key=lambda row: int(row["efSearch"]))
            axis.plot(
                [float(row["recall_at_10"]) for row in points],
                [float(row["qps"]) for row in points],
                marker=marker,
                markersize=4,
                linewidth=1.6,
                label=method,
                color=color,
            )
        axis.set_title(workload.title)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel(f"QPS ({threads} thread{'s' if threads != 1 else ''})")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.suptitle("Full FP32 vectors resident in process-owned RAM; same HNSW topology")
    png = RESULT_DIR / f"recall_qps_fullvec_ram_t{threads}.png"
    pdf = RESULT_DIR / f"recall_qps_fullvec_ram_t{threads}.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(WORKLOADS), default=list(WORKLOADS)
    )
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--result-tag", default="clean_t1_cpu20_20261109")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    global RESULT_DIR
    args = parse_args()
    if args.threads != 1:
        raise SystemExit("this comparison is fixed to one search thread")
    if any(ef < K for ef in args.efs):
        raise SystemExit(f"every efSearch must be >= {K}")
    if not args.result_tag or Path(args.result_tag).name != args.result_tag:
        raise SystemExit("--result-tag must be one directory name")
    RESULT_DIR = RESULT_ROOT / args.result_tag
    compile_binaries()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    workloads = [WORKLOADS[key] for key in args.datasets]
    splits: dict[str, adc_cal.SplitFiles] = {}
    calibrations: dict[str, dict[int, dict[str, object]]] = {}
    adc_outputs: dict[str, list[Path]] = {}
    fp_outputs: dict[str, Path] = {}
    rounds_by_dataset: dict[str, int] = {}
    for workload in workloads:
        workload_rounds = max(args.rounds, 20) if workload.key == "dbpedia_qwen1024" else args.rounds
        rounds_by_dataset[workload.key] = workload_rounds
        registry.validate(workload.dataset)
        split = split_for(workload)
        splits[workload.key] = split
        calibration = calibrate(
            workload,
            split,
            args.efs,
            args.threads,
            args.force,
            args.keep_candidates,
        )
        calibrations[workload.key] = calibration
        adc_outputs[workload.key] = benchmark_adcbq(
            workload,
            split,
            calibration,
            args.efs,
            workload_rounds,
            args.threads,
            args.force,
        )
        fp_outputs[workload.key] = benchmark_fp_hnsw(
            workload,
            split,
            args.efs,
            workload_rounds,
            args.threads,
            args.force,
        )

    combined, targets = combine(
        workloads, splits, calibrations, adc_outputs, fp_outputs, args.threads
    )
    figures = plot(combined, args.threads)
    manifest = {
        "protocol": "calibrated_adcbq_vs_fp_hnsw_fullvec_process_owned_ram",
        "datasets": args.datasets,
        "efs": args.efs,
        "rounds_by_dataset": rounds_by_dataset,
        "qps_statistic_by_dataset": {
            key: ("median_20" if key == "dbpedia_qwen1024" else "mean_5")
            for key in args.datasets
        },
        "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average_at_manifest": list(os.getloadavg()),
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "k": K,
        "calibration_target": TARGET_COVERAGE,
        "calibration_reference": "exact FP top-10 within each ADC candidate pool",
        "split_seed": 11,
        "full_vector_condition": {
            "adcbq": "complete FP32 base copied into process-owned RAM before timing",
            "fp_hnsw": "complete normalized FP32 base stored in IndexHNSWFlat RAM storage",
        },
        "same_topology": True,
        "combined_csv": str(combined),
        "target_summary_csv": str(targets),
        "figures": [] if figures is None else [str(path) for path in figures],
        "adc_binary": str(ADC_BINARY),
        "fp_binary": str(FP_BINARY),
    }
    write_json(RESULT_DIR / "run_manifest.json", manifest)
    print(f"WROTE {combined}")
    print(f"WROTE {targets}")


if __name__ == "__main__":
    main()
