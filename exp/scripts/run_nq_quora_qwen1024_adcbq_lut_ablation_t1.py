#!/usr/bin/env python3
"""NQ/Quora Qwen-1024 calibrated ADC-BQ LUT ablation.

Compare the source tree's no-LUT FP32-direct ADC scorer with the current
optimized int16 4-bit/nibble LUT scorer.  Both modes use the same FP-built
M=24, efConstruction=200 HNSW topology and a seed-11 1000-query calibration
split.  A separate per-ef split-conformal threshold is fitted for each mode
to preserve the exact-FP top ten inside that mode's ADC candidate pool with
probability 0.99.  Exact reranking uses a process-owned FP32 RAM copy so the
measurement isolates the traversal scorer rather than SSD timing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiment"
sys.path.insert(0, str(EXPERIMENT))

import run_adcbq_calibrated_threshold_overlap as calibration  # noqa: E402
import run_v051_ssd_odirect_multidataset as registry  # noqa: E402


SOURCE = EXPERIMENT / "faiss_fpgraph_packed_asym_bq_hnsw.cpp"
BUILD_SCRIPT = EXPERIMENT / "adcbq_optimized/build.sh"
BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_lut_ablation"
RESULT_ROOT = EXPERIMENT / "nq_quora_qwen1024_adcbq_lut_ablation_t1"
TARGET_COVERAGE = 0.99
CALIBRATION_QUERIES = 1000
SPLIT_SEED = 11
K = 10
M = 24
EF_CONSTRUCTION = 200
DEFAULT_CPU = 44
DEFAULT_EFS = (
    80,
    84,
    88,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    128,
    144,
    160,
    192,
    224,
    256,
    320,
    512,
    1024,
)


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    adc_mode: str
    lut_mode: str

    @property
    def uses_query_scale(self) -> bool:
        return self.adc_mode == "int16"


MODES = (
    Mode("no_lut_fp32_direct", "ADC-BQ no LUT (FP32 direct)", "fp32-direct", "nibble"),
    Mode(
        "optimized_int16_nibble_lut",
        "ADC-BQ optimized int16 nibble LUT",
        "int16",
        "nibble",
    ),
)


NQ_GRAPH = (
    EXPERIMENT
    / "nq_inmemory_fp_hnsw_recall_k30_50_100_t1"
    / "nq_qwen06_1024.fp_built_graph.M24.efC200.faiss"
)
QUORA_GRAPH = (
    REPO
    / "indexes/msmarco_quora_qwen1024_graph_construction_precision"
    / "fresh_build_timing"
    / "quora.fp_graph_bq_storage.n522931.d1024.M24.efC200.faiss"
)


def datasets() -> dict[str, registry.Dataset]:
    return {
        "nq_qwen06_1024": replace(
            registry.DATASETS["nq_qwen06_1024"], adcbq_index=NQ_GRAPH
        ),
        "quora_qwen1024": replace(
            registry.DATASETS["quora_qwen1024"], adcbq_index=QUORA_GRAPH
        ),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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
    temporary.replace(path)


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ensure_binary() -> None:
    if BINARY.exists() and BINARY.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
        return
    env = {**os.environ, "OUT": str(BINARY)}
    subprocess.run([str(BUILD_SCRIPT)], cwd=REPO, env=env, check=True)


def run(command: list[str], cpu: int) -> None:
    full = ["taskset", "-c", str(cpu), *command]
    print("RUN " + " ".join(full), flush=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    subprocess.run(full, cwd=REPO, env=env, check=True)


def common_command(
    dataset: registry.Dataset,
    mode: Mode,
    query: Path,
    gt: Path,
    output: Path,
    ef: int,
    rounds: int,
) -> list[str]:
    return [
        str(BINARY),
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
        "1",
        "--search-threads",
        "1",
        "--adc-mode",
        mode.adc_mode,
        "--lut-mode",
        mode.lut_mode,
        "--graph-build",
        "fp",
        "--distance-mode",
        "adcbq",
        "--search-impl",
        "inline",
        "--code-layout",
        "vector",
        "--efs",
        str(ef),
    ]


def result_complete(
    path: Path,
    split: calibration.SplitFiles,
    mode: Mode,
    ef: int,
    rounds: int,
    tau: float,
) -> bool:
    try:
        row = read_one(path)
        return (
            int(row["nq"]) == split.test_rows
            and int(row["efSearch"]) == ef
            and int(row["rounds"]) == rounds
            and int(row["search_threads"]) == 1
            and row["adc_mode"] == mode.adc_mode
            and row["fp_rerank_source"] == "ram"
            and math.isclose(
                float(row["calibrated_threshold_rerank_tau"]),
                tau,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
        )
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def candidate_paths(dataset: registry.Dataset, mode: Mode, ef: int) -> tuple[Path, Path]:
    root = RESULT_ROOT / dataset.key / mode.key / "calibration"
    return root / f"ef{ef}.summary.csv", root / f"ef{ef}.candidates.csv"


def collect_candidates(
    dataset: registry.Dataset,
    mode: Mode,
    split: calibration.SplitFiles,
    ef: int,
    cpu: int,
    force: bool,
) -> Path:
    summary, candidates = candidate_paths(dataset, mode, ef)
    expected_lines = split.calibration_rows * ef + 1
    if not force and summary.is_file() and candidates.is_file():
        with candidates.open(encoding="utf-8") as handle:
            if sum(1 for _ in handle) == expected_lines:
                return candidates
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = common_command(
        dataset,
        mode,
        split.calibration_query,
        split.calibration_gt,
        summary,
        ef,
        1,
    )
    command += [
        "--rerank-n",
        str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out",
        str(candidates),
    ]
    run(command, cpu)
    return candidates


def read_fbin(path: Path) -> np.ndarray:
    rows, dim = calibration.matrix_header(path)
    payload_bytes = rows * dim * np.dtype(np.float32).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes == 8 + payload_bytes:
        offset = 8
    elif actual_bytes == 4096 + payload_bytes:
        offset = 4096
    else:
        raise ValueError(f"unexpected fbin size: {path}")
    mapped = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(rows, dim))
    return np.array(mapped, dtype=np.float32, order="C", copy=True)


def train_memmap(path: Path) -> np.memmap:
    rows, dim = calibration.matrix_header(path)
    payload_bytes = rows * dim * np.dtype(np.float32).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes == 8 + payload_bytes:
        offset = 8
    elif actual_bytes == 4096 + payload_bytes:
        offset = 4096
    else:
        raise ValueError(f"unexpected fbin size: {path}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(rows, dim))


def query_scales(mode: Mode, query_path: Path) -> np.ndarray:
    rows, _ = calibration.matrix_header(query_path)
    if not mode.uses_query_scale:
        return np.ones(rows, dtype=np.float32)
    return calibration.int16_query_scales(query_path)


def fit_pool_top10_conformal(
    dataset: registry.Dataset,
    mode: Mode,
    split: calibration.SplitFiles,
    candidate_path: Path,
    ef: int,
) -> dict[str, object]:
    ids, distances = calibration.load_candidate_csv(
        candidate_path, split.calibration_rows, ef
    )
    queries = read_fbin(split.calibration_query)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    train = train_memmap(dataset.train)
    scales = query_scales(mode, split.calibration_query)
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
        "protocol": "per-ef-pool-top10-split-conformal",
        "mode": mode.key,
        "adc_mode": mode.adc_mode,
        "lut_mode": "none" if mode.adc_mode == "fp32-direct" else mode.lut_mode,
        "efSearch": ef,
        "tau": tau,
        "target_all_top10_coverage": TARGET_COVERAGE,
        "empirical_all_top10_coverage": float(np.mean(retained)),
        "conformal_order": order,
        "calibration_queries": split.calibration_rows,
        "reference": "exact FP top-10 within this scorer's ADC-HNSW candidate pool",
        "rerank_count_mean": float(np.mean(counts)),
        "rerank_count_min": int(np.min(counts)),
        "rerank_count_p50": float(np.percentile(counts, 50)),
        "rerank_count_p95": float(np.percentile(counts, 95)),
        "rerank_count_p99": float(np.percentile(counts, 99)),
        "rerank_count_max": int(np.max(counts)),
    }


def calibration_path(dataset: registry.Dataset, mode: Mode, ef: int) -> Path:
    return RESULT_ROOT / dataset.key / mode.key / "calibration" / f"ef{ef}.fit.json"


def calibrate(
    dataset: registry.Dataset,
    mode: Mode,
    split: calibration.SplitFiles,
    ef: int,
    cpu: int,
    force: bool,
    keep_candidates: bool,
) -> dict[str, object]:
    fitted_path = calibration_path(dataset, mode, ef)
    if fitted_path.is_file() and not force:
        payload = json.loads(fitted_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") == "per-ef-pool-top10-split-conformal"
            and payload.get("mode") == mode.key
            and int(payload.get("efSearch", -1)) == ef
            and int(payload.get("calibration_queries", -1)) == split.calibration_rows
        ):
            print(
                f"REUSE calibration {dataset.key} {mode.key} ef={ef} "
                f"tau={float(payload['tau']):.9g}",
                flush=True,
            )
            return payload
    candidates = collect_candidates(dataset, mode, split, ef, cpu, force)
    payload = fit_pool_top10_conformal(dataset, mode, split, candidates, ef)
    write_json(fitted_path, payload)
    print(
        f"CALIBRATED {dataset.key} {mode.key} ef={ef}: "
        f"tau={float(payload['tau']):.9g} "
        f"Rmean={float(payload['rerank_count_mean']):.2f} "
        f"coverage={float(payload['empirical_all_top10_coverage']):.4f}",
        flush=True,
    )
    if not keep_candidates:
        candidates.unlink(missing_ok=True)
        candidates.with_name(candidates.name.replace(".candidates.csv", ".summary.csv")).unlink(
            missing_ok=True
        )
    return payload


def benchmark(
    dataset: registry.Dataset,
    mode: Mode,
    split: calibration.SplitFiles,
    ef: int,
    fitted: dict[str, object],
    rounds: int,
    cpu: int,
    force: bool,
) -> Path:
    output = RESULT_ROOT / dataset.key / mode.key / "test" / f"ef{ef}.csv"
    tau = float(fitted["tau"])
    if not force and result_complete(output, split, mode, ef, rounds, tau):
        print(f"REUSE test {dataset.key} {mode.key} ef={ef}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = common_command(
        dataset,
        mode,
        split.test_query,
        split.test_gt,
        output,
        ef,
        rounds,
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
        repr(tau),
    ]
    run(command, cpu)
    return output


def summarize(
    dataset: registry.Dataset,
    mode: Mode,
    split: calibration.SplitFiles,
    ef: int,
    fitted: dict[str, object],
    result_path: Path,
    rounds: int,
) -> dict[str, object]:
    raw = read_one(result_path)
    return {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "dim": 1024,
        "method_key": mode.key,
        "method": mode.label,
        "adc_mode": mode.adc_mode,
        "lut_mode": "none" if mode.adc_mode == "fp32-direct" else mode.lut_mode,
        "efSearch": ef,
        "recall_at_10": float(raw["recall_at_k"]),
        "qps": float(raw["qps"]),
        "qps_median": float(raw.get("qps_median", raw["qps"])),
        "rounds": rounds,
        "search_threads": 1,
        "calibration_protocol": fitted["protocol"],
        "target_pool_top10_coverage": TARGET_COVERAGE,
        "calibration_queries": split.calibration_rows,
        "test_queries": split.test_rows,
        "conformal_order": fitted["conformal_order"],
        "tau": fitted["tau"],
        "calibration_empirical_all_top10_coverage": fitted[
            "empirical_all_top10_coverage"
        ],
        "calibration_rerank_count_mean": fitted["rerank_count_mean"],
        "rerank_count_mean": float(raw["adaptive_rerank_count_mean"]),
        "rerank_count_min": float(raw["adaptive_rerank_count_min"]),
        "rerank_count_p95": float(raw["adaptive_rerank_count_p95"]),
        "rerank_count_p99": float(raw["adaptive_rerank_count_p99"]),
        "rerank_count_max": float(raw["adaptive_rerank_count_max"]),
        "first_stage_distance_evaluations_per_query": float(
            raw.get("search_stats_ndis_mean", 0.0)
        ),
        "graph": str(dataset.adcbq_index),
        "graph_bytes": dataset.adcbq_index.stat().st_size,
        "graph_build": "FP-built topology, one-bit ADC-BQ payload",
        "code_storage": "resident one-bit codes",
        "fp_storage": "process-owned FP32 RAM",
        "source_csv": str(result_path),
    }


def save_summary(rows: list[dict[str, object]]) -> Path:
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["method_key"]),
            int(row["efSearch"]),
        )
    )
    path = RESULT_ROOT / "summary_t1.csv"
    write_csv(path, rows)
    return path


def save_comparisons(
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    by_key = {
        (str(row["dataset"]), str(row["method_key"]), int(row["efSearch"])): row
        for row in rows
    }
    paired: list[dict[str, object]] = []
    for dataset_key in sorted({str(row["dataset"]) for row in rows}):
        efs = sorted(
            {int(row["efSearch"]) for row in rows if row["dataset"] == dataset_key}
        )
        for ef in efs:
            direct = by_key[(dataset_key, "no_lut_fp32_direct", ef)]
            lut = by_key[(dataset_key, "optimized_int16_nibble_lut", ef)]
            paired.append(
                {
                    "dataset": dataset_key,
                    "efSearch": ef,
                    "no_lut_recall_at_10": direct["recall_at_10"],
                    "lut_recall_at_10": lut["recall_at_10"],
                    "lut_minus_no_lut_recall": float(lut["recall_at_10"])
                    - float(direct["recall_at_10"]),
                    "no_lut_qps_median": direct["qps_median"],
                    "lut_qps_median": lut["qps_median"],
                    "lut_over_no_lut_qps": float(lut["qps_median"])
                    / float(direct["qps_median"]),
                    "no_lut_rerank_mean": direct["rerank_count_mean"],
                    "lut_rerank_mean": lut["rerank_count_mean"],
                    "lut_minus_no_lut_rerank_mean": float(lut["rerank_count_mean"])
                    - float(direct["rerank_count_mean"]),
                }
            )
    paired_path = RESULT_ROOT / "paired_ef_comparison_t1.csv"
    write_csv(paired_path, paired)

    targets: list[dict[str, object]] = []
    for dataset_key in sorted({str(row["dataset"]) for row in rows}):
        for target in (0.99, 0.995, 0.997):
            selected: dict[str, dict[str, object] | None] = {}
            for mode_key in ("no_lut_fp32_direct", "optimized_int16_nibble_lut"):
                candidates = [
                    row
                    for row in rows
                    if row["dataset"] == dataset_key
                    and row["method_key"] == mode_key
                    and float(row["recall_at_10"]) >= target
                ]
                selected[mode_key] = (
                    max(candidates, key=lambda row: float(row["qps_median"]))
                    if candidates
                    else None
                )
            direct = selected["no_lut_fp32_direct"]
            lut = selected["optimized_int16_nibble_lut"]
            targets.append(
                {
                    "dataset": dataset_key,
                    "target_recall_at_10": target,
                    "no_lut_reached": direct is not None,
                    "no_lut_efSearch": "" if direct is None else direct["efSearch"],
                    "no_lut_recall_at_10": ""
                    if direct is None
                    else direct["recall_at_10"],
                    "no_lut_qps_median": "" if direct is None else direct["qps_median"],
                    "no_lut_rerank_mean": ""
                    if direct is None
                    else direct["rerank_count_mean"],
                    "lut_reached": lut is not None,
                    "lut_efSearch": "" if lut is None else lut["efSearch"],
                    "lut_recall_at_10": "" if lut is None else lut["recall_at_10"],
                    "lut_qps_median": "" if lut is None else lut["qps_median"],
                    "lut_rerank_mean": "" if lut is None else lut["rerank_count_mean"],
                    "lut_over_no_lut_qps": ""
                    if direct is None or lut is None
                    else float(lut["qps_median"]) / float(direct["qps_median"]),
                }
            )
    targets_path = RESULT_ROOT / "target_recall_summary_t1.csv"
    write_csv(targets_path, targets)
    return paired_path, targets_path


def plot(rows: list[dict[str, object]]) -> tuple[Path, Path] | None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rabitq-matplotlib-cache")
    try:
        import matplotlib
    except ImportError:
        print("matplotlib unavailable; skipping plots", flush=True)
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset_order = ("nq_qwen06_1024", "quora_qwen1024")
    titles = {"nq_qwen06_1024": "NQ", "quora_qwen1024": "Quora"}
    styles = {
        "no_lut_fp32_direct": ("#D55E00", "s", "No LUT (FP32 direct)"),
        "optimized_int16_nibble_lut": (
            "#0072B2",
            "o",
            "Optimized int16 nibble LUT",
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    for axis, dataset_key in zip(axes, dataset_order):
        for mode_key, (color, marker, label) in styles.items():
            points = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset_key and row["method_key"] == mode_key
                ),
                key=lambda row: int(row["efSearch"]),
            )
            axis.plot(
                [float(row["recall_at_10"]) for row in points],
                [float(row["qps_median"]) for row in points],
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.8,
                label=label,
            )
        axis.set_title(f"{titles[dataset_key]} · Qwen-1024")
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("QPS (T=1; median of 3 rounds)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Calibrated ADC-BQ LUT ablation · per-ef pool-top10 target 0.99 · FP rerank in RAM"
    )
    png = RESULT_ROOT / "recall_qps_lut_ablation_t1.png"
    pdf = RESULT_ROOT / "recall_qps_lut_ablation_t1.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nq_qwen06_1024", "quora_qwen1024"),
        default=("nq_qwen06_1024", "quora_qwen1024"),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=DEFAULT_EFS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if any(ef < K or ef > 1024 for ef in args.efs):
        raise SystemExit("every efSearch must be between 10 and 1024")
    if args.cpu not in os.sched_getaffinity(0):
        raise SystemExit(f"CPU {args.cpu} is outside this process affinity")

    ensure_binary()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_datasets = datasets()
    rows: list[dict[str, object]] = []
    split_manifest: dict[str, object] = {}
    graph_manifest: dict[str, object] = {}
    for dataset_key in args.datasets:
        dataset = selected_datasets[dataset_key]
        registry.validate(dataset)
        split = calibration.make_split(dataset, CALIBRATION_QUERIES, SPLIT_SEED, 0)
        split_manifest[dataset_key] = {
            "calibration_queries": split.calibration_rows,
            "test_queries": split.test_rows,
            "calibration_query": str(split.calibration_query),
            "test_query": str(split.test_query),
        }
        graph_manifest[dataset_key] = {
            "path": str(dataset.adcbq_index),
            "bytes": dataset.adcbq_index.stat().st_size,
            "sha256": file_sha256(dataset.adcbq_index),
        }
        # Pair the two scorer modes at each ef so both QPS measurements see
        # nearly identical host load and CPU-frequency conditions.
        for ef in args.efs:
            for mode in MODES:
                fitted = calibrate(
                    dataset,
                    mode,
                    split,
                    ef,
                    args.cpu,
                    args.force,
                    args.keep_candidates,
                )
                output = benchmark(
                    dataset,
                    mode,
                    split,
                    ef,
                    fitted,
                    args.rounds,
                    args.cpu,
                    args.force,
                )
                current = summarize(
                    dataset, mode, split, ef, fitted, output, args.rounds
                )
                rows.append(current)
                save_summary(rows)
                print(
                    f"DONE {dataset.key} {mode.key} ef={ef}: "
                    f"recall={float(current['recall_at_10']):.6f} "
                    f"qps_median={float(current['qps_median']):.2f} "
                    f"Rmean={float(current['rerank_count_mean']):.2f}",
                    flush=True,
                )

    summary = save_summary(rows)
    paired, targets = save_comparisons(rows)
    figures = plot(rows)
    manifest = {
        "experiment": "NQ/Quora Qwen-1024 calibrated ADC-BQ LUT ablation",
        "comparison": {
            "no_lut": "fp32-direct AVX-512 sign-bit accumulation; no query LUT",
            "optimized_lut": "current int16 4-dim/16-entry nibble LUT SIMD path",
        },
        "datasets": list(args.datasets),
        "efs": list(args.efs),
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "k": K,
        "threads": 1,
        "rounds": args.rounds,
        "cpu_affinity": args.cpu,
        "split_seed": SPLIT_SEED,
        "calibration_protocol": "per-ef-pool-top10-split-conformal",
        "calibration_target": TARGET_COVERAGE,
        "conformal_order": 991,
        "fp_rerank_storage": "process-owned FP32 RAM",
        "same_graph_within_dataset": True,
        "splits": split_manifest,
        "graphs": graph_manifest,
        "binary": str(BINARY),
        "source": str(SOURCE),
        "summary_csv": str(summary),
        "paired_ef_comparison_csv": str(paired),
        "target_recall_summary_csv": str(targets),
        "plot_png": "" if figures is None else str(figures[0]),
        "plot_pdf": "" if figures is None else str(figures[1]),
    }
    write_json(RESULT_ROOT / "manifest.json", manifest)
    print(f"WROTE {summary}", flush=True)
    print(f"WROTE {paired}", flush=True)
    print(f"WROTE {targets}", flush=True)
    if figures is not None:
        print(f"WROTE {figures[0]}", flush=True)
        print(f"WROTE {figures[1]}", flush=True)


if __name__ == "__main__":
    main()
