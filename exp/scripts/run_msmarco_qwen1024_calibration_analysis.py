#!/usr/bin/env python3
"""Paper-ready calibration analysis for MS MARCO / Qwen-1024, T=1.

The experiment uses the same seed-11 calibration/test split and FP-built HNSW
graph as the recall ablation.  It measures the split-conformal guarantee that
the selected ADC prefix contains the exact-FP top-10 *inside the HNSW candidate
pool*, then separately reports final global Recall@10 after real O_DIRECT
io_uring FP32 refinement.
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
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-msmarco-calibration-analysis")

REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/msmarco_qwen1024_calibration_analysis"
FP_RERANK_METRIC = "cosine"
BINARY = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_same_pool_calibration"
TRAIN = Path(
    "/home/smrc/samsung-nvme/jihyo/qwen/"
    "msmarco-1m__vanilla-qwen06b_train.fbin"
)
GRAPH = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_qwen1024_scale_v014/"
    "indexes/faiss_adcbq/"
    "msmarco-qwen1024-1m.fpgraph_pgbit_binary_hnsw."
    "n1000000.d1024.M24.efC200.faiss"
)
FP_SIDECAR = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/"
    "aligned_fp/msmarco_qwen1024.fp32.header4096.fbin"
)
SPLIT_ROOT = (
    REPO
    / "experiment/calibrated_adcbq_ivf_vs_rabitq_ssd/splits/msmarco_qwen1024"
)
CAL_QUERY = SPLIT_ROOT / "seed11.cal1000.test5980.cal.query.fbin"
CAL_GT = SPLIT_ROOT / "seed11.cal1000.test5980.cal.gt.ibin"
TEST_QUERY = SPLIT_ROOT / "seed11.cal1000.test5980.test.query.fbin"
TEST_GT = SPLIT_ROOT / "seed11.cal1000.test5980.test.gt.ibin"

K = 10
PAGE_BYTES = 4096.0
DEFAULT_TARGETS = (0.90, 0.95, 0.975, 0.99, 0.995)
DEFAULT_FIXED_RS = (10, 20, 30, 40, 50, 60, 75, 80, 100, 120, 140, 160, 192, 224, 256)


def matrix_header(path: Path) -> tuple[int, int]:
    import struct

    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"bad matrix header: {path}")
    return struct.unpack("ii", raw)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def run(command: list[str], cpu: int) -> None:
    full = ["taskset", "-c", str(cpu), *command]
    print("RUN " + " ".join(full), flush=True)
    subprocess.run(
        full,
        cwd=REPO,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=True,
    )


def base_command(query: Path, gt: Path, out: Path, ef: int) -> list[str]:
    return [
        str(BINARY),
        "--dataset", "msmarco_qwen1024",
        "--train", str(TRAIN),
        "--query", str(query),
        "--gt", str(gt),
        "--index", str(GRAPH),
        "--out", str(out),
        "--m", "24",
        "--ef-construction", "200",
        "--k", str(K),
        "--max-queries", "0",
        "--rounds", "1",
        "--build-threads", "1",
        "--search-threads", "1",
        "--adc-mode", "int16",
        "--lut-mode", "nibble",
        "--graph-build", "fp",
        "--distance-mode", "adcbq",
        "--search-impl", "inline",
        "--code-layout", "block32",
        "--efs", str(ef),
        "--candidate-pool-size", str(ef),
    ]


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def collect_candidates(
    name: str,
    query: Path,
    gt: Path,
    nq: int,
    ef: int,
    cpu: int,
    force: bool,
) -> Path:
    root = RESULT_ROOT / "trace"
    summary = root / f"{name}.candidates.summary.csv"
    candidates = root / f"{name}.candidates.csv"
    if not force and candidates.is_file() and count_lines(candidates) == nq * ef + 1:
        print(f"REUSE {candidates}", flush=True)
        return candidates
    root.mkdir(parents=True, exist_ok=True)
    command = base_command(query, gt, summary, ef)
    command += [
        "--rerank-n", str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out", str(candidates),
    ]
    run(command, cpu)
    return candidates


def collect_pool_labels(
    name: str,
    query: Path,
    gt: Path,
    nq: int,
    ef: int,
    cpu: int,
    force: bool,
) -> Path:
    root = RESULT_ROOT / "trace"
    summary = root / f"{name}.full_pool_exact.summary.csv"
    labels = root / f"{name}.full_pool_exact.labels.csv"
    if not force and labels.is_file() and count_lines(labels) == nq * K + 1:
        print(f"REUSE {labels}", flush=True)
        return labels
    root.mkdir(parents=True, exist_ok=True)
    command = base_command(query, gt, summary, ef)
    command += [
        "--rerank-n", str(ef),
        "--rerank-dot", "fp32",
        "--fp-rerank-metric", FP_RERANK_METRIC,
        "--fp-rerank-source", "ram",
        "--fp-rerank-norms", "onfetch",
        "--final-labels-out", str(labels),
    ]
    run(command, cpu)
    return labels


def load_candidates(path: Path, nq: int, ef: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((nq, ef), -1, dtype=np.int32)
    distances = np.full((nq, ef), np.inf, dtype=np.float32)
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            if not (0 <= qi < nq and 0 <= rank < ef):
                raise ValueError(f"bad candidate coordinate q={qi}, rank={rank}")
            ids[qi, rank] = int(row["id"])
            distances[qi, rank] = float(row["distance"])
            seen += 1
    if seen != nq * ef:
        raise ValueError(f"expected {nq * ef} candidates, got {seen}")
    return ids, distances


def load_labels(path: Path, nq: int) -> np.ndarray:
    labels = np.full((nq, K), -1, dtype=np.int32)
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            qi = int(row["query_id"])
            rank = int(row["rank"])
            if not (0 <= qi < nq and 0 <= rank < K):
                raise ValueError(f"bad label coordinate q={qi}, rank={rank}")
            labels[qi, rank] = int(row["id"])
            seen += 1
    if seen != nq * K:
        raise ValueError(f"expected {nq * K} labels, got {seen}")
    return labels


def load_gt(path: Path) -> np.ndarray:
    nq, dim = matrix_header(path)
    mapped = np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(nq, dim))
    return np.array(mapped[:, :K], dtype=np.int32, copy=True)


def int16_query_scales(path: Path) -> np.ndarray:
    nq, dim = matrix_header(path)
    queries = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(nq, dim))
    block_l1 = np.abs(np.asarray(queries)).reshape(nq, dim // 4, 4).sum(axis=2)
    max_abs = block_l1.max(axis=1)
    return np.divide(
        32767.0,
        max_abs,
        out=np.ones_like(max_abs, dtype=np.float32),
        where=max_abs > 0,
    )


def ranks_in_candidates(ids: np.ndarray, labels: np.ndarray, ef: int) -> np.ndarray:
    nq = ids.shape[0]
    ranks = np.full((nq, K), ef + 1, dtype=np.int32)
    for begin in range(0, nq, 256):
        end = min(nq, begin + 256)
        matches = ids[begin:end, :, None] == labels[begin:end, None, :]
        found = matches.any(axis=1)
        first = matches.argmax(axis=1) + 1
        ranks[begin:end] = np.where(found, first, ef + 1)
    if np.any(ranks > ef):
        raise RuntimeError("an exact pool label is absent from its source candidate pool")
    return ranks


def membership_ranks(ids: np.ndarray, labels: np.ndarray, ef: int) -> np.ndarray:
    nq = ids.shape[0]
    ranks = np.full((nq, K), ef + 1, dtype=np.int32)
    for begin in range(0, nq, 256):
        end = min(nq, begin + 256)
        matches = ids[begin:end, :, None] == labels[begin:end, None, :]
        found = matches.any(axis=1)
        first = matches.argmax(axis=1) + 1
        ranks[begin:end] = np.where(found, first, ef + 1)
    return ranks


def query_requirements(
    distances: np.ndarray,
    exact_pool_ranks: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    rows = np.arange(distances.shape[0])[:, None]
    positions = exact_pool_ranks - 1
    needed = distances[rows, positions] - distances[:, K - 1, None]
    return np.maximum(0.0, needed.max(axis=1) / scales.astype(np.float64))


def selected_counts(distances: np.ndarray, scales: np.ndarray, tau: float, ef: int) -> np.ndarray:
    cutoffs = distances[:, K - 1] + tau * scales
    counts = (distances <= cutoffs[:, None]).sum(axis=1)
    return np.minimum(ef, np.maximum(K, counts)).astype(np.int32)


def fit_target(requirements: np.ndarray, target: float) -> tuple[int, float]:
    order = int(math.ceil((len(requirements) + 1) * target))
    if order > len(requirements):
        raise ValueError(f"target {target} is infeasible for n={len(requirements)}")
    tau = float(np.partition(requirements, order - 1)[order - 1])
    return order, tau


def policy_tag(prefix: str, value: float | int) -> str:
    if isinstance(value, float):
        token = f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    else:
        token = str(value)
    return f"{prefix}{token}"


def complete_physical(summary: Path, metrics: Path, nq: int) -> bool:
    if not summary.is_file() or not metrics.is_file():
        return False
    try:
        return len(read_rows(summary)) == 1 and len(read_rows(metrics)) == nq
    except (OSError, csv.Error):
        return False


def run_physical_policy(
    policy: str,
    value: float | int,
    ef: int,
    nq: int,
    cpu: int,
    force: bool,
) -> tuple[Path, Path]:
    tag = policy_tag("target" if policy == "adaptive" else "r", value)
    root = RESULT_ROOT / "physical"
    summary = root / f"{policy}.{tag}.summary.csv"
    metrics = root / f"{policy}.{tag}.per_query.csv"
    if not force and complete_physical(summary, metrics, nq):
        print(f"REUSE {summary}", flush=True)
        return summary, metrics
    root.mkdir(parents=True, exist_ok=True)
    command = base_command(TEST_QUERY, TEST_GT, summary, ef)
    rerank_n = ef if policy == "adaptive" else int(value)
    command += [
        "--rerank-n", str(rerank_n),
        "--rerank-dot", "fp32",
        "--fp-rerank-metric", FP_RERANK_METRIC,
        "--fp-rerank-source", "odirect-uring",
        "--fp-rerank-norms", "onfetch",
        "--fp-rerank-path", str(FP_SIDECAR),
        "--fp-rerank-qd", str(ef),
        "--strict-query-e2e",
        "--per-query-metrics-out", str(metrics),
    ]
    if policy == "adaptive":
        command += ["--calibrated-threshold-rerank-tau", repr(float(value))]
    run(command, cpu)
    return summary, metrics


def physical_stats(summary_path: Path, metrics_path: Path) -> tuple[dict[str, object], np.ndarray, list[dict[str, str]]]:
    summary = read_rows(summary_path)[0]
    metrics = read_rows(metrics_path)
    reads = np.asarray([int(row["selected_fp_reads"]) for row in metrics], dtype=np.int32)
    recalls = np.asarray([float(row["recall_at_k"]) for row in metrics], dtype=np.float64)
    nq = len(metrics)
    submitted = float(summary["direct_submitted_bytes_mean"])
    requests = float(summary["direct_requests_mean"])
    stats: dict[str, object] = {
        "final_mean_recall_at_10": float(recalls.mean()),
        "final_perfect_queries": int((recalls >= 1.0 - 1e-12).sum()),
        "final_perfect_query_rate": float((recalls >= 1.0 - 1e-12).mean()),
        "mean_fp_reads": float(reads.mean()),
        "p50_fp_reads": percentile(reads, 50),
        "p95_fp_reads": percentile(reads, 95),
        "p99_fp_reads": percentile(reads, 99),
        "max_fp_reads": int(reads.max()),
        "mean_submitted_4k_pages": submitted / nq / PAGE_BYTES,
        "mean_direct_requests": requests / nq,
        "qps_strict_query_wall": float(summary["qps"]),
        "mean_end_to_end_us": float(
            np.mean([float(row["end_to_end_us"]) for row in metrics])
        ),
        "p99_end_to_end_us": percentile(
            np.asarray([float(row["end_to_end_us"]) for row in metrics]), 99
        ),
    }
    return stats, reads, metrics


def pool_recall(labels: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return (labels[:, :, None] == gt[:, None, :]).any(axis=2).mean(axis=1)


def original_test_ids() -> np.ndarray:
    nq, _ = matrix_header(CAL_QUERY)
    nt, _ = matrix_header(TEST_QUERY)
    permutation = np.random.default_rng(11).permutation(nq + nt)
    return permutation[nq:]


def calibration_size_sensitivity(
    requirements: np.ndarray,
    test_distances: np.ndarray,
    test_scales: np.ndarray,
    test_exact_ranks: np.ndarray,
    ef: int,
    target: float,
    seeds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for size in (100, 250, 500, 1000):
        for seed in range(seeds):
            rng = np.random.default_rng(10000 + seed)
            subset = rng.choice(len(requirements), size=size, replace=False)
            order, tau = fit_target(requirements[subset], target)
            counts = selected_counts(test_distances, test_scales, tau, ef)
            covered = np.all(test_exact_ranks <= counts[:, None], axis=1)
            rows.append(
                {
                    "dataset": "msmarco_qwen1024",
                    "target_pool_top10_coverage": target,
                    "calibration_queries": size,
                    "subsample_seed": seed,
                    "conformal_order": order,
                    "tau": tau,
                    "test_covered_queries": int(covered.sum()),
                    "test_queries": len(covered),
                    "test_empirical_coverage": float(covered.mean()),
                    "mean_fp_reads": float(counts.mean()),
                    "p95_fp_reads": percentile(counts, 95),
                    "p99_fp_reads": percentile(counts, 99),
                    "max_fp_reads": int(counts.max()),
                }
            )
    return rows


def plot_results(
    target_rows: list[dict[str, object]],
    fixed_rows: list[dict[str, object]],
    target99_reads: np.ndarray,
    error_rows: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))

    ax = axes[0, 0]
    targets = np.asarray([float(row["target_pool_top10_coverage"]) for row in target_rows])
    coverages = np.asarray([float(row["test_empirical_pool_top10_coverage"]) for row in target_rows])
    mean_reads = np.asarray([float(row["mean_fp_reads"]) for row in target_rows])
    ax.plot(targets, coverages, "o-", color="#0072b2", label="Held-out coverage")
    bounds = [min(targets.min(), coverages.min()) - 0.005, 1.001]
    ax.plot(bounds, bounds, "--", color="0.5", label="Target = empirical")
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("Calibration target (pool top-10 coverage)")
    ax.set_ylabel("Held-out empirical coverage")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(targets, mean_reads, "s:", color="#d55e00", label="Mean FP reads")
    ax2.set_ylabel("Mean 4-KiB FP pages/query", color="#d55e00")
    lines = ax.get_lines()[:2] + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="lower right", fontsize=8)
    ax.set_title("(a) Target reliability and read cost")

    ax = axes[0, 1]
    ax.plot(
        [float(row["mean_fp_reads"]) for row in fixed_rows],
        [float(row["final_mean_recall_at_10"]) for row in fixed_rows],
        "o-", color="#666666", label="Fixed R",
    )
    ax.scatter(
        [float(row["mean_fp_reads"]) for row in target_rows],
        [float(row["final_mean_recall_at_10"]) for row in target_rows],
        marker="D", s=55, color="#009e73", label="Calibrated R(q)", zorder=3,
    )
    annotation_offsets = {
        0.9: (5, 3),
        0.95: (-3, 11),
        0.975: (-20, 10),
        0.99: (5, -14),
        0.995: (7, 8),
    }
    for row in target_rows:
        target = float(row["target_pool_top10_coverage"])
        ax.annotate(
            f"{target:g}",
            (float(row["mean_fp_reads"]), float(row["final_mean_recall_at_10"])),
            xytext=annotation_offsets.get(target, (4, 4)),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean exact-FP reads/query (4-KiB pages)")
    ax.set_ylabel("Final Recall@10")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("(b) Fixed versus calibrated refinement")

    ax = axes[1, 0]
    values = np.sort(target99_reads)
    cdf = np.arange(1, len(values) + 1) / len(values)
    ax.step(values, cdf, where="post", color="#009e73", label="Target 0.99")
    ax.axvline(75, color="#d55e00", linestyle="--", label="Fixed R=75")
    ax.set_xlabel("Selected R(q) / FP pages")
    ax.set_ylabel("CDF of test queries")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("(c) Query-adaptive R distribution")

    ax = axes[1, 1]
    names = [str(row["category"]) for row in error_rows]
    counts = [int(row["queries"]) for row in error_rows]
    colors = ["#009e73", "#e69f00", "#56b4e9", "#cc79a7"][: len(names)]
    bars = ax.bar(names, counts, color=colors)
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3, fontsize=9)
    ax.set_ylabel("Queries (N=5,980)")
    ax.tick_params(axis="x", rotation=12)
    ax.set_title("(d) Final perfect-Recall error decomposition, target 0.99")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "MS MARCO · Qwen3-Embedding-0.6B (1024D) — ADC-BQ calibration analysis "
        f"({FP_RERANK_METRIC}), efSearch=256, T=1",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for suffix in ("png", "pdf"):
        output = RESULT_ROOT / f"msmarco_qwen1024_calibration_analysis.{suffix}"
        fig.savefig(output, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        print(f"WROTE {output}", flush=True)
    plt.close(fig)


def main() -> None:
    global RESULT_ROOT, FP_RERANK_METRIC, CAL_GT, TEST_GT
    # The system Python on the benchmark host intentionally has no plotting
    # stack.  Re-enter through the already-installed analysis environment so
    # the default command is fully reproducible, including the paper figure.
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        plotting_python = Path("/home/jihyo/anaconda3/bin/python")
        if plotting_python.is_file() and Path(sys.executable) != plotting_python:
            os.execv(str(plotting_python), [str(plotting_python), *sys.argv])
        raise

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ef", type=int, default=256)
    parser.add_argument("--cpu", type=int, default=46)
    parser.add_argument("--targets", type=float, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--fixed-rs", type=int, nargs="+", default=DEFAULT_FIXED_RS)
    parser.add_argument("--sensitivity-seeds", type=int, default=20)
    parser.add_argument("--force-trace", action="store_true")
    parser.add_argument("--force-physical", action="store_true")
    parser.add_argument(
        "--fp-rerank-metric", choices=("cosine", "ip"), default="cosine"
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--cal-gt", type=Path, default=CAL_GT)
    parser.add_argument("--test-gt", type=Path, default=TEST_GT)
    args = parser.parse_args()
    RESULT_ROOT = args.result_root.resolve()
    FP_RERANK_METRIC = args.fp_rerank_metric
    CAL_GT = args.cal_gt.resolve()
    TEST_GT = args.test_gt.resolve()
    ef = args.ef
    if ef < K:
        parser.error("--ef must be at least 10")
    if any(not 0.0 < target < 1.0 for target in args.targets):
        parser.error("targets must be in (0, 1)")
    if any(r < K or r > ef for r in args.fixed_rs):
        parser.error("every fixed R must be in [10, ef]")

    for path in (BINARY, TRAIN, GRAPH, FP_SIDECAR, CAL_QUERY, CAL_GT, TEST_QUERY, TEST_GT):
        if not path.is_file():
            raise FileNotFoundError(path)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ncal, dim = matrix_header(CAL_QUERY)
    ntest, test_dim = matrix_header(TEST_QUERY)
    if (ncal, ntest, dim, test_dim) != (1000, 5980, 1024, 1024):
        raise RuntimeError(f"unexpected split shapes: {(ncal, ntest, dim, test_dim)}")

    cal_candidates_path = collect_candidates(
        "calibration", CAL_QUERY, CAL_GT, ncal, ef, args.cpu, args.force_trace
    )
    test_candidates_path = collect_candidates(
        "test", TEST_QUERY, TEST_GT, ntest, ef, args.cpu, args.force_trace
    )
    cal_labels_path = collect_pool_labels(
        "calibration", CAL_QUERY, CAL_GT, ncal, ef, args.cpu, args.force_trace
    )
    test_labels_path = collect_pool_labels(
        "test", TEST_QUERY, TEST_GT, ntest, ef, args.cpu, args.force_trace
    )

    cal_ids, cal_distances = load_candidates(cal_candidates_path, ncal, ef)
    test_ids, test_distances = load_candidates(test_candidates_path, ntest, ef)
    cal_labels = load_labels(cal_labels_path, ncal)
    test_labels = load_labels(test_labels_path, ntest)
    cal_exact_ranks = ranks_in_candidates(cal_ids, cal_labels, ef)
    test_exact_ranks = ranks_in_candidates(test_ids, test_labels, ef)
    cal_scales = int16_query_scales(CAL_QUERY)
    test_scales = int16_query_scales(TEST_QUERY)
    cal_requirements = query_requirements(cal_distances, cal_exact_ranks, cal_scales)
    test_requirements = query_requirements(test_distances, test_exact_ranks, test_scales)
    test_gt = load_gt(TEST_GT)
    test_gt_membership_ranks = membership_ranks(test_ids, test_gt, ef)
    test_pool_recall = pool_recall(test_labels, test_gt)

    fitted: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    target_reads: dict[float, np.ndarray] = {}
    target_metrics: dict[float, list[dict[str, str]]] = {}
    for target in sorted(set(args.targets)):
        order, tau = fit_target(cal_requirements, target)
        cal_counts = selected_counts(cal_distances, cal_scales, tau, ef)
        test_counts = selected_counts(test_distances, test_scales, tau, ef)
        cal_covered = np.all(cal_exact_ranks <= cal_counts[:, None], axis=1)
        test_covered = np.all(test_exact_ranks <= test_counts[:, None], axis=1)
        fitted.append(
            {
                "target": target,
                "conformal_order": order,
                "tau": tau,
                "calibration_covered_queries": int(cal_covered.sum()),
                "calibration_queries": ncal,
                "calibration_empirical_coverage": float(cal_covered.mean()),
            }
        )
        summary_path, metrics_path = run_physical_policy(
            "adaptive", tau, ef, ntest, args.cpu, args.force_physical
        )
        stats, physical_reads, metrics = physical_stats(summary_path, metrics_path)
        if not np.array_equal(physical_reads, test_counts):
            delta = np.abs(physical_reads.astype(np.int64) - test_counts.astype(np.int64))
            raise RuntimeError(
                f"offline/C++ selected-count mismatch at target {target}: "
                f"max={delta.max()}, nonzero={(delta != 0).sum()}"
            )
        target_reads[target] = physical_reads
        target_metrics[target] = metrics
        target_rows.append(
            {
                "dataset": "msmarco_qwen1024",
                "fp_rerank_metric": FP_RERANK_METRIC,
                "efSearch": ef,
                "candidate_pool_size": ef,
                "target_pool_top10_coverage": target,
                "conformal_order": order,
                "tau": tau,
                "calibration_covered_queries": int(cal_covered.sum()),
                "calibration_queries": ncal,
                "calibration_empirical_coverage": float(cal_covered.mean()),
                "test_pool_top10_covered_queries": int(test_covered.sum()),
                "test_queries": ntest,
                "test_empirical_pool_top10_coverage": float(test_covered.mean()),
                **stats,
                "summary_csv": str(summary_path),
                "per_query_csv": str(metrics_path),
            }
        )
    write_rows(RESULT_ROOT / "target_sweep.csv", target_rows)
    write_json(
        RESULT_ROOT / "fitted_thresholds.json",
        {
            "protocol": "per-ef-pool-top10-split-conformal",
            "reference": "exact FP32 top-10 within each ADC-HNSW candidate pool",
            "dataset": "msmarco_qwen1024",
            "efSearch": ef,
            "calibration_queries": ncal,
            "test_queries": ntest,
            "fits": fitted,
        },
    )

    fixed_rows: list[dict[str, object]] = []
    for fixed_r in sorted(set(args.fixed_rs)):
        summary_path, metrics_path = run_physical_policy(
            "fixed", fixed_r, ef, ntest, args.cpu, args.force_physical
        )
        stats, _, _ = physical_stats(summary_path, metrics_path)
        pool_covered = np.all(test_exact_ranks <= fixed_r, axis=1)
        fixed_rows.append(
            {
                "dataset": "msmarco_qwen1024",
                "fp_rerank_metric": FP_RERANK_METRIC,
                "efSearch": ef,
                "candidate_pool_size": ef,
                "fixed_r": fixed_r,
                "test_pool_top10_covered_queries": int(pool_covered.sum()),
                "test_queries": ntest,
                "test_empirical_pool_top10_coverage": float(pool_covered.mean()),
                **stats,
                "summary_csv": str(summary_path),
                "per_query_csv": str(metrics_path),
            }
        )
    write_rows(RESULT_ROOT / "fixed_r_sweep.csv", fixed_rows)

    target99 = min(target_rows, key=lambda row: abs(float(row["target_pool_top10_coverage"]) - 0.99))
    target99_value = float(target99["target_pool_top10_coverage"])
    target99_counts = target_reads[target99_value]
    metrics99 = target_metrics[target99_value]
    recall99 = np.asarray([float(row["recall_at_k"]) for row in metrics99])
    covered99 = np.all(test_exact_ranks <= target99_counts[:, None], axis=1)
    pool_perfect = test_pool_recall >= 1.0 - 1e-12
    final_perfect = recall99 >= 1.0 - 1e-12
    membership_hits = (test_gt_membership_ranks <= ef).sum(axis=1)
    original_ids = original_test_ids()
    per_query_rows: list[dict[str, object]] = []
    for qi, metric in enumerate(metrics99):
        per_query_rows.append(
            {
                "split_query_id": qi,
                "original_query_id": int(original_ids[qi]),
                "selected_r": int(target99_counts[qi]),
                "hit_candidate_pool_cap": bool(target99_counts[qi] == ef),
                "pool_top10_preserved": bool(covered99[qi]),
                "required_tau": float(test_requirements[qi]),
                "global_gt10_in_candidate_pool": int(membership_hits[qi]),
                "global_gt10_all_in_candidate_pool": bool(membership_hits[qi] == K),
                "full_pool_exact_recall_at_10": float(test_pool_recall[qi]),
                "full_pool_perfect_recall": bool(pool_perfect[qi]),
                "final_recall_at_10": float(recall99[qi]),
                "final_perfect_recall": bool(final_perfect[qi]),
                "search_us": float(metric["search_us"]),
                "selection_us": float(metric["selection_us"]),
                "io_wait_us": float(metric["io_wait_us"]),
                "rerank_us": float(metric["rerank_us"]),
                "end_to_end_us": float(metric["end_to_end_us"]),
            }
        )
    write_rows(RESULT_ROOT / "per_query_target099.csv", per_query_rows)

    group_specs = (
        ("1.0", np.isclose(recall99, 1.0)),
        ("0.9", np.isclose(recall99, 0.9)),
        ("<=0.8", recall99 <= 0.8 + 1e-12),
    )
    grouped_rows: list[dict[str, object]] = []
    end_to_end_ms = np.asarray(
        [float(row["end_to_end_us"]) / 1000.0 for row in metrics99], dtype=np.float64
    )
    for label, mask in group_specs:
        grouped_rows.append(
            {
                "per_query_recall_at_10": label,
                "queries": int(mask.sum()),
                "query_fraction": float(mask.mean()),
                "query_percent": float(mask.mean() * 100.0),
                "mean_fp_reads": float(target99_counts[mask].mean()),
                "mean_latency_ms": float(end_to_end_ms[mask].mean()),
                "fp_rerank_metric": FP_RERANK_METRIC,
            }
        )
    write_rows(RESULT_ROOT / "per_query_recall_groups_target099.csv", grouped_rows)

    reference_mismatch = (
        membership_hits == K
    ) & ~pool_perfect
    error_rows = [
        {
            "category": "Final perfect",
            "queries": int(final_perfect.sum()),
            "fraction": float(final_perfect.mean()),
            "definition": "final Recall@10 = 1.0",
        },
        {
            "category": "Selection-limited",
            "queries": int(((membership_hits == K) & pool_perfect & ~final_perfect).sum()),
            "fraction": float(((membership_hits == K) & pool_perfect & ~final_perfect).mean()),
            "definition": "full-pool perfect but calibrated prefix is not",
        },
        {
            "category": "Metric/reference mismatch",
            "queries": int(reference_mismatch.sum()),
            "fraction": float(reference_mismatch.mean()),
            "definition": "all global GT10 are present but full-pool exact rerank is not perfect",
        },
        {
            "category": "Traversal-limited",
            "queries": int((membership_hits < K).sum()),
            "fraction": float((membership_hits < K).mean()),
            "definition": "at least one global GT10 is absent from the 256-candidate pool",
        },
    ]
    write_rows(RESULT_ROOT / "error_decomposition_target099.csv", error_rows)

    sensitivity = calibration_size_sensitivity(
        cal_requirements,
        test_distances,
        test_scales,
        test_exact_ranks,
        ef,
        0.99,
        args.sensitivity_seeds,
    )
    write_rows(RESULT_ROOT / "calibration_size_sensitivity.csv", sensitivity)

    plot_results(target_rows, fixed_rows, target99_counts, error_rows)
    manifest = {
        "dataset": "msmarco_qwen1024",
        "corpus": "MS MARCO 1M",
        "embedding": "Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "split": "seed11.cal1000.test5980",
        "graph": str(GRAPH),
        "graph_sha256": file_sha256(GRAPH),
        "graph_build": "FP32, M=24, efConstruction=200",
        "calibration_ground_truth": str(CAL_GT),
        "test_ground_truth": str(TEST_GT),
        "calibration_ground_truth_sha256": file_sha256(CAL_GT),
        "test_ground_truth_sha256": file_sha256(TEST_GT),
        "efSearch": ef,
        "candidate_pool_size": ef,
        "query_scorer": "ADC-BQ int16 compact nibble LUT, inline SIMD",
        "fp_rerank_metric": FP_RERANK_METRIC,
        "refinement": (
            f"FP32 {FP_RERANK_METRIC}, O_DIRECT io_uring, "
            "one aligned 4-KiB vector page/read"
        ),
        "threads": 1,
        "latency_mode": "strict query-local search -> selection -> I/O -> rerank wall time",
        "calibration_reference": "exact FP32 top-10 within the ADC-HNSW candidate pool",
        "important_interpretation": (
            "0.99 calibrates candidate-pool top-10 preservation, not the fraction of "
            "queries with final global Recall@10=1.0; both are reported separately"
        ),
        "targets": sorted(set(args.targets)),
        "fixed_rs": sorted(set(args.fixed_rs)),
        "outputs": {
            "target_sweep": str(RESULT_ROOT / "target_sweep.csv"),
            "fixed_r_sweep": str(RESULT_ROOT / "fixed_r_sweep.csv"),
            "per_query_target099": str(RESULT_ROOT / "per_query_target099.csv"),
            "per_query_recall_groups_target099": str(
                RESULT_ROOT / "per_query_recall_groups_target099.csv"
            ),
            "error_decomposition": str(RESULT_ROOT / "error_decomposition_target099.csv"),
            "calibration_size_sensitivity": str(RESULT_ROOT / "calibration_size_sensitivity.csv"),
            "figure": str(RESULT_ROOT / "msmarco_qwen1024_calibration_analysis.png"),
        },
    }
    write_json(RESULT_ROOT / "manifest.json", manifest)

    print("\nTARGET SWEEP", flush=True)
    for row in target_rows:
        print(
            f"target={row['target_pool_top10_coverage']} "
            f"coverage={float(row['test_empirical_pool_top10_coverage']):.6f} "
            f"covered={row['test_pool_top10_covered_queries']}/{ntest} "
            f"recall={float(row['final_mean_recall_at_10']):.6f} "
            f"perfect={row['final_perfect_queries']}/{ntest} "
            f"reads={float(row['mean_fp_reads']):.2f}",
            flush=True,
        )
    print("\nTARGET 0.99 ERROR DECOMPOSITION", flush=True)
    for row in error_rows:
        print(f"{row['category']}: {row['queries']} ({float(row['fraction']):.4%})", flush=True)


if __name__ == "__main__":
    main()
