#!/usr/bin/env python3
"""Query-side precision sweep for calibrated ADC-BQ and RaBitQ-1b adaptive FP.

Input queries are quantized independently with a symmetric uniform quantizer,
dequantized to float32, and L2-normalized.  The database, graph, exact ground
truth, calibration/test split, and all non-query parameters stay fixed.  This
measures query representation error against the original FP32-query ground
truth.  RaBitQ retains its production internal four-bit query estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

PYTHON311 = Path("/home/jihyo/anaconda3/envs/hnsw/bin/python3.11")
if sys.version_info[:2] != (3, 11) and PYTHON311.is_file():
    os.execv(str(PYTHON311), [str(PYTHON311), *sys.argv])

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiment"
sys.path.insert(0, str(EXPERIMENT))

import run_adcbq_calibrated_threshold_overlap as calibration  # noqa: E402
import run_nq_quora_c16_rabitq as rabitq_runner  # noqa: E402
import run_nq_quora_qwen1024_adcbq_lut_ablation_t1 as adcbq_runner  # noqa: E402
import run_v051_ssd_odirect_multidataset as registry  # noqa: E402


RESULT_ROOT = EXPERIMENT / "nq_quora_qwen1024_query_bitwidth_sweep_t1"
ADCBQ_ROOT = RESULT_ROOT / "adcbq"
# The experiment sweep is 2/4/8/16 bits.  FP32 is measured only as the
# reference needed to quantify saturation; it is not a quantized sweep point.
BITS = (2, 4, 8, 16, 32)
EF = 256
K = 10
CALIBRATION_QUERIES = 1000
SPLIT_SEED = 11
TARGET_COVERAGE = 0.99
NEAR_FP32_EPSILON = 0.001  # 0.1 percentage point in Recall@10.
DEFAULT_CPU = 44
DEFAULT_BUILD_THREADS = 24

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


@dataclass(frozen=True)
class PreparedSplit:
    files: calibration.SplitFiles
    calibration_ids: np.ndarray
    test_ids: np.ndarray
    query_cosine_mean: float
    query_cosine_p01: float


def datasets() -> dict[str, registry.Dataset]:
    return {
        "nq_qwen06_1024": replace(
            registry.DATASETS["nq_qwen06_1024"], adcbq_index=NQ_GRAPH
        ),
        "quora_qwen1024": replace(
            registry.DATASETS["quora_qwen1024"], adcbq_index=QUORA_GRAPH
        ),
    }


def bit_label(bits: int) -> str:
    return "fp32" if bits == 32 else f"{bits}bit"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_adcbq(command: list[str], cpu: int) -> None:
    """Run the C++ benchmark without the base-Conda MKL/libstdc++ overlay."""
    full = ["taskset", "-c", str(cpu), *command]
    print("RUN " + " ".join(full), flush=True)
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": "/home/jihyo/anaconda3/envs/hnsw/lib",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    subprocess.run(full, cwd=REPO, env=env, check=True)


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    out = np.asarray(rows, dtype=np.float32).copy(order="C")
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(norms, 1e-30)
    return out


def quantize_queries(rows: np.ndarray, bits: int) -> np.ndarray:
    normalized = normalize_rows(rows)
    if bits == 32:
        return normalized
    levels = (1 << bits) - 1
    max_abs = np.max(np.abs(normalized), axis=1, keepdims=True)
    step = np.divide(
        2.0 * max_abs,
        float(levels),
        out=np.ones_like(max_abs, dtype=np.float32),
        where=max_abs > 0,
    )
    quantized = np.rint((normalized + max_abs) / step)
    np.clip(quantized, 0, levels, out=quantized)
    reconstructed = quantized * step - max_abs
    return normalize_rows(reconstructed)


def split_ids(nq: int) -> tuple[np.ndarray, np.ndarray]:
    permutation = np.random.default_rng(SPLIT_SEED).permutation(nq)
    return permutation[:CALIBRATION_QUERIES], permutation[CALIBRATION_QUERIES:]


def prepare_split(dataset: registry.Dataset, bits: int) -> PreparedSplit:
    nq, dim = calibration.matrix_header(dataset.query)
    ng, gt_dim = calibration.matrix_header(dataset.gt)
    if nq != ng:
        raise ValueError(f"query/GT mismatch for {dataset.key}")
    cal_ids, test_ids = split_ids(nq)
    root = RESULT_ROOT / "queries" / dataset.key / bit_label(bits)
    tag = f"seed{SPLIT_SEED}.cal{len(cal_ids)}.test{len(test_ids)}"
    files = calibration.SplitFiles(
        root / f"{tag}.cal.query.fbin",
        root / f"{tag}.cal.gt.ibin",
        root / f"{tag}.test.query.fbin",
        root / f"{tag}.test.gt.ibin",
        len(cal_ids),
        len(test_ids),
        dim,
        gt_dim,
    )
    metadata = root / "quantization.json"
    expected = {
        files.calibration_query: (len(cal_ids), dim),
        files.calibration_gt: (len(cal_ids), gt_dim),
        files.test_query: (len(test_ids), dim),
        files.test_gt: (len(test_ids), gt_dim),
    }
    if all(path.is_file() and calibration.matrix_header(path) == shape for path, shape in expected.items()) and metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        return PreparedSplit(
            files,
            cal_ids,
            test_ids,
            float(payload["query_cosine_mean"]),
            float(payload["query_cosine_p01"]),
        )

    queries = np.memmap(
        dataset.query, dtype=np.float32, mode="r", offset=8, shape=(nq, dim)
    )
    gt = np.memmap(
        dataset.gt, dtype=np.int32, mode="r", offset=8, shape=(nq, gt_dim)
    )
    original = normalize_rows(np.asarray(queries))
    reconstructed = quantize_queries(original, bits)
    cosine = np.sum(original * reconstructed, axis=1)
    calibration.write_matrix(files.calibration_query, reconstructed[cal_ids])
    calibration.write_matrix(files.calibration_gt, np.asarray(gt[cal_ids]))
    calibration.write_matrix(files.test_query, reconstructed[test_ids])
    calibration.write_matrix(files.test_gt, np.asarray(gt[test_ids]))
    write_csv(
        root / "query_ids.csv",
        [
            {"split": "calibration", "split_query_id": i, "original_query_id": int(qid)}
            for i, qid in enumerate(cal_ids)
        ]
        + [
            {"split": "test", "split_query_id": i, "original_query_id": int(qid)}
            for i, qid in enumerate(test_ids)
        ],
    )
    payload = {
        "dataset": dataset.key,
        "query_bits": bits,
        "query_precision": bit_label(bits),
        "quantizer": "per-query symmetric uniform endpoints, dequantize FP32, L2 normalize",
        "ground_truth": "original FP32-query exact top-100",
        "query_cosine_mean": float(np.mean(cosine)),
        "query_cosine_p01": float(np.percentile(cosine, 1)),
    }
    write_json(metadata, payload)
    return PreparedSplit(
        files,
        cal_ids,
        test_ids,
        float(payload["query_cosine_mean"]),
        float(payload["query_cosine_p01"]),
    )


def load_gt(path: Path) -> np.ndarray:
    rows, dim = calibration.matrix_header(path)
    mapped = np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(rows, dim))
    return np.asarray(mapped[:, :K], dtype=np.int32)


def per_query_recall(ids: np.ndarray, gt: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)[:, :K]
    return np.asarray(
        [len(set(map(int, got)) & set(map(int, truth))) / K for got, truth in zip(ids, gt)],
        dtype=np.float64,
    )


def load_final_labels(path: Path, nq: int) -> np.ndarray:
    labels = np.full((nq, K), -1, dtype=np.int64)
    seen = 0
    for row in read_csv(path):
        qi, rank = int(row["query_id"]), int(row["rank"])
        if not (0 <= qi < nq and 0 <= rank < K):
            raise ValueError(f"bad final label coordinate: q={qi}, rank={rank}")
        labels[qi, rank] = int(row["id"])
        seen += 1
    if seen != nq * K:
        raise ValueError(f"expected {nq * K} labels in {path}, got {seen}")
    return labels


def adcbq_measure(
    dataset: registry.Dataset,
    bits: int,
    split: PreparedSplit,
    cpu: int,
    force: bool,
) -> tuple[dict[str, object], np.ndarray]:
    mode = adcbq_runner.MODES[1]
    tagged = replace(dataset, key=f"{dataset.key}__query_{bit_label(bits)}")
    adcbq_runner.RESULT_ROOT = ADCBQ_ROOT
    fitted = adcbq_runner.calibrate(
        tagged, mode, split.files, EF, cpu, force, keep_candidates=False
    )
    root = ADCBQ_ROOT / tagged.key / mode.key / "test"
    output = root / f"ef{EF}.csv"
    labels = root / f"ef{EF}.final_labels.csv"
    valid = False
    if output.is_file() and labels.is_file() and not force:
        try:
            row = read_one(output)
            valid = (
                int(row["nq"]) == split.files.test_rows
                and int(row["efSearch"]) == EF
                and row["calibrated_threshold_rerank_enabled"] == "true"
                and len(read_csv(labels)) == split.files.test_rows * K
            )
        except (OSError, KeyError, ValueError, RuntimeError):
            valid = False
    if not valid:
        root.mkdir(parents=True, exist_ok=True)
        command = adcbq_runner.common_command(
            tagged,
            mode,
            split.files.test_query,
            split.files.test_gt,
            output,
            EF,
            1,
        )
        command += [
            "--rerank-n",
            str(EF),
            "--rerank-dot",
            "fp32",
            "--fp-rerank-source",
            "ram",
            "--fp-rerank-norms",
            "precompute",
            "--calibrated-threshold-rerank-tau",
            repr(float(fitted["tau"])),
            "--final-labels-out",
            str(labels),
        ]
        adcbq_runner.run(command, cpu)
    raw = read_one(output)
    ids = load_final_labels(labels, split.files.test_rows)
    gt = load_gt(split.files.test_gt)
    recalls = per_query_recall(ids, gt)
    summary = {
        "dataset": dataset.key,
        "method": "calibrated_adcbq",
        "method_label": "Calibrated ADC-BQ",
        "query_bits": bits,
        "query_precision": bit_label(bits),
        "efSearch": EF,
        "queries": split.files.test_rows,
        "recall_at_10": float(np.mean(recalls)),
        "qps": float(raw.get("qps_median", raw["qps"])),
        "refinement_count_mean": float(raw["adaptive_rerank_count_mean"]),
        "refinement_count_p95": float(raw["adaptive_rerank_count_p95"]),
        "tau": float(fitted["tau"]),
        "calibration_queries": split.files.calibration_rows,
        "calibration_target": TARGET_COVERAGE,
        "calibration_coverage": float(fitted["empirical_all_top10_coverage"]),
        "query_cosine_to_fp32_mean": split.query_cosine_mean,
        "query_cosine_to_fp32_p01": split.query_cosine_p01,
        "source_csv": str(output),
    }
    return summary, recalls


def ensure_rabitq_index(dataset: registry.Dataset, build_threads: int) -> Path:
    path = rabitq_runner.index_path(dataset, 1)
    if path.is_file():
        return path
    args = SimpleNamespace(
        only_bits=(1,),
        force=False,
        build_threads=build_threads,
    )
    print(f"BUILD missing RaBitQ c16 b1 index for {dataset.key}", flush=True)
    rabitq_runner.build_dataset(dataset, args)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def rabitq_measure(
    dataset: registry.Dataset,
    bits: int,
    split: PreparedSplit,
    index: object,
    sidecar: Path,
    force: bool,
) -> tuple[dict[str, object], np.ndarray]:
    root = RESULT_ROOT / "rabitq" / dataset.key / bit_label(bits)
    output = root / f"ef{EF}.summary.csv"
    per_query_path = root / f"ef{EF}.per_query.csv"
    if output.is_file() and per_query_path.is_file() and not force:
        rows = read_csv(per_query_path)
        if len(rows) == split.files.test_rows:
            return dict(read_one(output)), np.asarray(
                [float(row["recall_at_10"]) for row in rows], dtype=np.float64
            )

    queries = adcbq_runner.read_fbin(split.files.test_query)
    gt = load_gt(split.files.test_gt)
    started = time.perf_counter()
    ids, _, stats = index.search_1bit_adaptive_fp_odirect(
        np.ascontiguousarray(queries, dtype=np.float32),
        str(sidecar),
        k=K,
        ef=EF,
        num_threads=1,
    )
    elapsed = time.perf_counter() - started
    recalls = per_query_recall(np.asarray(ids), gt)
    fp_reads = np.asarray(stats["fp_read_count"], dtype=np.float64)
    if len(fp_reads) != len(recalls):
        raise RuntimeError("RaBitQ per-query stats length mismatch")
    summary: dict[str, object] = {
        "dataset": dataset.key,
        "method": "rabitq_1bit_adaptive_fp",
        "method_label": "RaBitQ 1-bit + adaptive FP",
        "query_bits": bits,
        "query_precision": bit_label(bits),
        "efSearch": EF,
        "queries": len(recalls),
        "recall_at_10": float(np.mean(recalls)),
        "qps": len(recalls) / elapsed,
        "refinement_count_mean": float(np.mean(fp_reads)),
        "refinement_count_p95": float(np.percentile(fp_reads, 95)),
        "tau": "",
        "calibration_queries": 0,
        "calibration_target": "",
        "calibration_coverage": "",
        "query_cosine_to_fp32_mean": split.query_cosine_mean,
        "query_cosine_to_fp32_p01": split.query_cosine_p01,
        "source_csv": str(per_query_path),
    }
    write_csv(output, [summary])
    write_csv(
        per_query_path,
        [
            {
                "dataset": dataset.key,
                "method": "rabitq_1bit_adaptive_fp",
                "query_bits": bits,
                "query_precision": bit_label(bits),
                "split_query_id": qi,
                "original_query_id": int(split.test_ids[qi]),
                "recall_at_10": float(recalls[qi]),
                "fp_vectors_read": int(fp_reads[qi]),
            }
            for qi in range(len(recalls))
        ],
    )
    return summary, recalls


def analyze(
    summaries: list[dict[str, object]],
    recall_rows: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    by_summary = {
        (str(row["dataset"]), str(row["method"]), int(row["query_bits"])): row
        for row in summaries
    }
    enhanced: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in summaries}):
        for method in sorted({str(row["method"]) for row in summaries if row["dataset"] == dataset}):
            fp32 = float(by_summary[(dataset, method, 32)]["recall_at_10"])
            previous: float | None = None
            for bits in BITS:
                row = dict(by_summary[(dataset, method, bits)])
                recall = float(row["recall_at_10"])
                row["recall_gain_from_previous"] = "" if previous is None else recall - previous
                row["recall_gain_from_2bit"] = recall - float(
                    by_summary[(dataset, method, 2)]["recall_at_10"]
                )
                row["recall_gap_to_fp32"] = fp32 - recall
                enhanced.append(row)
                previous = recall
    summary_path = RESULT_ROOT / "summary_t1.csv"
    write_csv(summary_path, enhanced)

    grouped: dict[tuple[str, str, int], dict[int, float]] = {}
    for row in recall_rows:
        grouped.setdefault(
            (str(row["dataset"]), str(row["method"]), int(row["query_bits"])), {}
        )[int(row["original_query_id"])] = float(row["recall_at_10"])
    transitions: list[dict[str, object]] = []
    for dataset, method in sorted({(key[0], key[1]) for key in grouped}):
        for lo, hi in zip(BITS[:-1], BITS[1:]):
            before = grouped[(dataset, method, lo)]
            after = grouped[(dataset, method, hi)]
            if before.keys() != after.keys():
                raise RuntimeError(f"query IDs differ for {dataset} {method} {lo}->{hi}")
            delta = np.asarray([after[qid] - before[qid] for qid in sorted(before)])
            transitions.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "from_bits": lo,
                    "to_bits": hi,
                    "transition": f"{bit_label(lo)}->{bit_label(hi)}",
                    "queries": len(delta),
                    "mean_recall_change": float(np.mean(delta)),
                    "improved_query_fraction": float(np.mean(delta > 1e-12)),
                    "unchanged_query_fraction": float(np.mean(np.abs(delta) <= 1e-12)),
                    "worsened_query_fraction": float(np.mean(delta < -1e-12)),
                    "mean_gain_among_improved": float(np.mean(delta[delta > 1e-12]))
                    if np.any(delta > 1e-12)
                    else 0.0,
                }
            )
    transition_path = RESULT_ROOT / "query_improvement_transitions_t1.csv"
    write_csv(transition_path, transitions)

    saturation: list[dict[str, object]] = []
    for dataset, method in sorted({(key[0], key[1]) for key in by_summary}):
        fp32 = float(by_summary[(dataset, method, 32)]["recall_at_10"])
        qualifying = [
            bits
            for bits in BITS
            if abs(float(by_summary[(dataset, method, bits)]["recall_at_10"]) - fp32)
            <= NEAR_FP32_EPSILON
        ]
        saturation.append(
            {
                "dataset": dataset,
                "method": method,
                "near_fp32_epsilon": NEAR_FP32_EPSILON,
                "fp32_recall_at_10": fp32,
                "minimum_bits_within_epsilon": min(qualifying),
            }
        )
    saturation_path = RESULT_ROOT / "fp32_saturation_t1.csv"
    write_csv(saturation_path, saturation)
    return summary_path, transition_path, saturation_path


def plot(summary_path: Path, transition_path: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    summary = read_csv(summary_path)
    transitions = read_csv(transition_path)
    labels = [bit_label(bits) for bits in BITS]
    method_style = {
        "calibrated_adcbq": ("#2166ac", "s", "Calibrated ADC-BQ"),
        "rabitq_1bit_adaptive_fp": ("#b2182b", "o", "RaBitQ-1b + adaptive FP"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3))
    for col, dataset in enumerate(("nq_qwen06_1024", "quora_qwen1024")):
        title = "NQ · Qwen-1024" if dataset.startswith("nq_") else "Quora · Qwen-1024"
        for method, (color, marker, label) in method_style.items():
            rows = sorted(
                (row for row in summary if row["dataset"] == dataset and row["method"] == method),
                key=lambda row: BITS.index(int(row["query_bits"])),
            )
            axes[0, col].plot(
                range(len(BITS)),
                [float(row["recall_at_10"]) for row in rows],
                color=color,
                marker=marker,
                linewidth=1.8,
                label=label,
            )
            trans = sorted(
                (row for row in transitions if row["dataset"] == dataset and row["method"] == method),
                key=lambda row: BITS.index(int(row["from_bits"])),
            )
            axes[1, col].plot(
                range(len(BITS) - 1),
                [100.0 * float(row["improved_query_fraction"]) for row in trans],
                color=color,
                marker=marker,
                linewidth=1.8,
                label=label,
            )
        axes[0, col].set_title(title)
        axes[0, col].set_xticks(range(len(BITS)), labels)
        axes[0, col].set_ylabel("Mean Recall@10")
        axes[0, col].grid(True, alpha=0.25)
        axes[1, col].set_xticks(
            range(len(BITS) - 1),
            [f"{labels[i]}→{labels[i + 1]}" for i in range(len(BITS) - 1)],
            rotation=25,
            ha="right",
        )
        axes[1, col].set_ylabel("Queries improved (%)")
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].grid(True, alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Query-side precision sweep (fixed efSearch=256)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    png = RESULT_ROOT / "query_bitwidth_recall_and_improvement_t1.png"
    pdf = RESULT_ROOT / "query_bitwidth_recall_and_improvement_t1.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    parser.add_argument("--build-threads", type=int, default=DEFAULT_BUILD_THREADS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nq_qwen06_1024", "quora_qwen1024"),
        default=("nq_qwen06_1024", "quora_qwen1024"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = tuple(args.datasets)
    dataset_map = datasets()

    adcbq_runner.ensure_binary()
    adcbq_runner.RESULT_ROOT = ADCBQ_ROOT
    adcbq_runner.run = run_adcbq
    summaries: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    for dataset_key in selected:
        dataset = dataset_map[dataset_key]
        for path in (dataset.train, dataset.query, dataset.gt, dataset.adcbq_index):
            if not path.is_file():
                raise FileNotFoundError(path)
        index_path = ensure_rabitq_index(dataset, args.build_threads)
        sidecar = rabitq_runner.fp_path(dataset)
        rabitq_runner.validate_aligned_fp(sidecar, dataset)
        print(f"LOAD RaBitQ {dataset.key}: {index_path}", flush=True)
        index = rabitq_runner.HnswIndex.load(str(index_path))
        if index.nbits != 1 or index.num_clusters != rabitq_runner.NUM_CLUSTERS:
            raise RuntimeError(f"unexpected RaBitQ metadata: {index_path}")
        for bits in BITS:
            print(f"QUERY SWEEP {dataset.key} {bit_label(bits)}", flush=True)
            split = prepare_split(dataset, bits)
            adc_summary, adc_recalls = adcbq_measure(
                dataset, bits, split, args.cpu, args.force
            )
            rq_summary, rq_recalls = rabitq_measure(
                dataset, bits, split, index, sidecar, args.force
            )
            summaries.extend((adc_summary, rq_summary))
            for method, values in (
                ("calibrated_adcbq", adc_recalls),
                ("rabitq_1bit_adaptive_fp", rq_recalls),
            ):
                recall_rows.extend(
                    {
                        "dataset": dataset.key,
                        "method": method,
                        "query_bits": bits,
                        "query_precision": bit_label(bits),
                        "split_query_id": qi,
                        "original_query_id": int(split.test_ids[qi]),
                        "recall_at_10": float(value),
                    }
                    for qi, value in enumerate(values)
                )
        del index

    per_query_path = RESULT_ROOT / "per_query_recall_t1.csv"
    write_csv(per_query_path, recall_rows)
    summary_path, transition_path, saturation_path = analyze(summaries, recall_rows)
    png, pdf = plot(summary_path, transition_path)
    manifest = RESULT_ROOT / "manifest.json"
    write_json(
        manifest,
        {
            "datasets": list(selected),
            "query_bits": list(BITS),
            "query_quantizer": "per-query symmetric uniform endpoints; FP32 dequantization; L2 normalization",
            "ground_truth": "original FP32-query exact neighbors",
            "split_seed": SPLIT_SEED,
            "calibration_queries": CALIBRATION_QUERIES,
            "calibration_target": TARGET_COVERAGE,
            "efSearch": EF,
            "threads": 1,
            "near_fp32_epsilon": NEAR_FP32_EPSILON,
            "rabitq_note": "production RaBitQ retains its internal 4-bit query estimator after input-query quantization",
            "summary": str(summary_path),
            "per_query": str(per_query_path),
            "transitions": str(transition_path),
            "saturation": str(saturation_path),
            "figure_png": str(png),
            "figure_pdf": str(pdf),
        },
    )
    for path in (
        summary_path,
        per_query_path,
        transition_path,
        saturation_path,
        png,
        pdf,
        manifest,
    ):
        print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
