#!/usr/bin/env python3
"""Single-thread MSMARCO/Qwen-1024 recall ablation on one FP-built graph.

The evaluated stages are:

1. Sym-BQ, without exact refinement.
2. ADC-BQ, without exact refinement.
3. ADC-BQ with fixed exact-FP refinement at R=75 (the Table-3 R*).
4. ADC-BQ with per-ef split-conformal, query-adaptive refinement.

All methods use the same M=24, efConstruction=200 FP32-built HNSW graph and
the same held-out query split.  The calibration JSON files were fit on the
disjoint calibration split by run_diskveclab_full_matrix_adcbq_pool_top10_per_ef.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-msmarco-recall-ablation")


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/msmarco_qwen1024_recall_ablation_t1"

ADCBQ_BINARY = (
    REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_same_pool_calibration"
)
SYMBQ_BINARY = REPO / "experiment/faiss_hnsw_bq_budget_paged_codes"

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
SYMBQ_CODEC = Path(
    "/home/smrc/samsung-nvme/jihyo/"
    "faiss_exact_matched_budget_compare_results/indexes/"
    "msmarco_qwen1024/msmarco_qwen1024.symbq1.faiss"
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
CALIBRATION_ROOT = (
    REPO
    / "experiment/diskveclab_full_matrix/"
    "calibrated_adcbq_pool_top10_target99_per_ef/"
    "msmarco_qwen1024/calibration"
)

DEFAULT_EFS = (256, 512, 1024)
PAGE_BYTES = 4096.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def validate_inputs(efs: list[int]) -> None:
    required = (
        ADCBQ_BINARY,
        SYMBQ_BINARY,
        TRAIN,
        GRAPH,
        SYMBQ_CODEC,
        FP_SIDECAR,
        CAL_QUERY,
        CAL_GT,
        TEST_QUERY,
        TEST_GT,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for ef in efs:
        calibration = CALIBRATION_ROOT / f"pool_top10_target99.ef{ef}.json"
        if not calibration.is_file():
            raise FileNotFoundError(calibration)


def fit_missing_calibrations(efs: list[int], cpu: int) -> None:
    missing = [
        ef
        for ef in efs
        if not (CALIBRATION_ROOT / f"pool_top10_target99.ef{ef}.json").is_file()
    ]
    if not missing:
        return

    # Reuse the exact fitting implementation used by the 16-dataset matrix,
    # but stop after fitting: no test-set benchmark and no transient 4 GiB FP
    # sidecar are needed here.
    import run_diskveclab_full_matrix_adcbq_pool_top10_per_ef as fitter

    fitter.runner.RESULT_ROOT = fitter.RESULT_ROOT
    fitter.runner.calibration.BIN = ADCBQ_BINARY
    dataset = fitter.runner.registry.DATASETS["msmarco_qwen1024"]
    split = fitter.runner.matrix.split_for("msmarco_qwen1024")
    old_affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {cpu})
        for ef in missing:
            fitted = fitter.fit_per_ef(
                dataset,
                split,
                ef,
                0.99,
                11,
                1,
                False,
                False,
            )
            if fitted.get("status") != "feasible":
                raise RuntimeError(
                    f"pool-top10 calibration is infeasible at ef={ef}: {fitted}"
                )
    finally:
        os.sched_setaffinity(0, old_affinity)


def adcbq_base(
    query: Path,
    gt: Path,
    output: Path,
    efs: list[int],
    rounds: int,
    max_queries: int,
) -> list[str]:
    return [
        str(ADCBQ_BINARY),
        "--dataset", "msmarco_qwen1024",
        "--train", str(TRAIN),
        "--query", str(query),
        "--gt", str(gt),
        "--index", str(GRAPH),
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
        "--efs", *[str(ef) for ef in efs],
    ]


def fp_rerank_args(rerank_n: int, qd: int) -> list[str]:
    return [
        "--rerank-n", str(rerank_n),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", "odirect-uring",
        "--fp-rerank-norms", "onfetch",
        "--fp-rerank-path", str(FP_SIDECAR),
        "--fp-rerank-overlap-queries",
        "--fp-rerank-qd", str(qd),
    ]


def complete_adcbq(path: Path, efs: list[int], rounds: int, nq: int) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_rows(path)
        return (
            sorted(int(row["efSearch"]) for row in rows) == sorted(efs)
            and all(int(row["rounds"]) == rounds for row in rows)
            and all(int(row["nq"]) == nq for row in rows)
            and all(int(row["search_threads"]) == 1 for row in rows)
        )
    except (KeyError, ValueError):
        return False


def run_symbq(
    root: Path,
    efs: list[int],
    rounds: int,
    nq: int,
    cpu: int,
    force: bool,
) -> Path:
    output = root / "raw/symbq_no_rerank.csv"
    if output.is_file() and not force:
        rows = read_rows(output)
        expected = {(ef, round_id) for ef in efs for round_id in range(1, rounds + 1)}
        actual = {(int(row["efSearch"]), int(row["round"])) for row in rows}
        if actual == expected and all(int(row["nq"]) == nq for row in rows):
            print(f"REUSE {output}", flush=True)
            return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(SYMBQ_BINARY),
        "--dataset", "msmarco_qwen1024",
        "--method", "symbq1",
        "--metric", "ip",
        "--query", str(TEST_QUERY),
        "--gt", str(TEST_GT),
        "--graph", str(GRAPH),
        "--codec-index", str(SYMBQ_CODEC),
        # Syntactically required but unused with resident codes.
        "--sidecar", str(root / "unused.symbq.pages4k"),
        "--out", str(output),
        "--max-queries", str(nq),
        "--rounds", str(rounds),
        "--threads", "1",
        "--bq-budget-bytes", "128000000",
        "--pinned-metadata-bytes", "0",
        "--resident-codes",
        "--efs", *[str(ef) for ef in efs],
    ]
    run(command, cpu)
    return output


def run_adc_no_rerank(
    root: Path,
    efs: list[int],
    rounds: int,
    max_queries: int,
    nq: int,
    cpu: int,
    force: bool,
) -> Path:
    output = root / "raw/adcbq_no_rerank.csv"
    if not force and complete_adcbq(output, efs, rounds, nq):
        print(f"REUSE {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = adcbq_base(TEST_QUERY, TEST_GT, output, efs, rounds, max_queries)
    command.append("--no-rerank")
    run(command, cpu)
    return output


def run_adc_fixed(
    root: Path,
    efs: list[int],
    fixed_r: int,
    rounds: int,
    max_queries: int,
    nq: int,
    cpu: int,
    force: bool,
) -> Path:
    output = root / f"raw/adcbq_fixed_r{fixed_r}.csv"
    if not force and complete_adcbq(output, efs, rounds, nq):
        print(f"REUSE {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = adcbq_base(TEST_QUERY, TEST_GT, output, efs, rounds, max_queries)
    command += fp_rerank_args(fixed_r, fixed_r)
    run(command, cpu)
    return output


def calibration_for_ef(ef: int) -> dict[str, object]:
    path = CALIBRATION_ROOT / f"pool_top10_target99.ef{ef}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != "per-ef-pool-top10-split-conformal"
        or payload.get("status") != "feasible"
        or int(payload.get("ef", -1)) != ef
        or int(payload.get("calibration_rows", -1)) != 1000
        or float(payload.get("target_pool_top10_coverage", -1.0)) != 0.99
    ):
        raise RuntimeError(f"incompatible calibration file: {path}")
    payload["path"] = str(path)
    return payload


def run_adc_calibrated(
    root: Path,
    efs: list[int],
    rounds: int,
    max_queries: int,
    nq: int,
    cpu: int,
    force: bool,
) -> dict[int, Path]:
    outputs: dict[int, Path] = {}
    for ef in efs:
        fitted = calibration_for_ef(ef)
        tau = float(fitted["tau"])
        output = root / f"raw/adcbq_calibrated_ef{ef}.csv"
        outputs[ef] = output
        reusable = not force and complete_adcbq(output, [ef], rounds, nq)
        if reusable:
            row = read_rows(output)[0]
            reusable = (
                row.get("calibrated_threshold_rerank_enabled") == "true"
                and abs(float(row["calibrated_threshold_rerank_tau"]) - tau)
                <= 1e-5 * max(1.0, abs(tau))
            )
        if reusable:
            print(f"REUSE {output}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        command = adcbq_base(
            TEST_QUERY, TEST_GT, output, [ef], rounds, max_queries
        )
        command += fp_rerank_args(ef, ef)
        command += ["--calibrated-threshold-rerank-tau", repr(tau)]
        run(command, cpu)
    return outputs


def standard_adc_row(
    source: Path,
    row: dict[str, str],
    method: str,
    stage: int,
    query_scoring: str,
    rerank_policy: str,
    fixed_r: int | str,
    tau: float | str,
    calibration: dict[str, object] | None,
) -> dict[str, object]:
    nq = int(row["nq"])
    selected = float(row.get("adaptive_rerank_count_mean", "0") or 0.0)
    submitted = float(row.get("direct_submitted_bytes_mean", "0") or 0.0)
    return {
        "dataset": "msmarco_qwen1024",
        "corpus": "MS MARCO",
        "embedding": "Qwen3-Embedding-0.6B",
        "dim": int(row["dim"]),
        "nb": int(row["nb"]),
        "nq": nq,
        "split": "seed11.cal1000.test5980.test",
        "stage": stage,
        "method": method,
        "query_scoring": query_scoring,
        "rerank_policy": rerank_policy,
        "efSearch": int(row["efSearch"]),
        "fixed_r": fixed_r,
        "calibration_tau": tau,
        "calibration_protocol": "" if calibration is None else calibration["protocol"],
        "calibration_target": "" if calibration is None else calibration["target_pool_top10_coverage"],
        "calibration_empirical_coverage": "" if calibration is None else calibration["empirical_all_top10_coverage"],
        "recall_at_10": float(row["recall_at_k"]),
        "qps": float(row.get("qps_median", row["qps"])),
        "qps_from_mean_seconds": float(row["qps"]),
        "latency_ms_per_query": 1000.0 * float(row["total_seconds_mean"]) / nq,
        "fp_vectors_per_query_mean": selected,
        "fp_vectors_p95": float(row.get("adaptive_rerank_count_p95", "0") or 0.0),
        "fp_vectors_p99": float(row.get("adaptive_rerank_count_p99", "0") or 0.0),
        "fp_vectors_max": float(row.get("adaptive_rerank_count_max", "0") or 0.0),
        "fp_4k_pages_per_query_mean": submitted / nq / PAGE_BYTES,
        "rounds": int(row["rounds"]),
        "search_threads": int(row["search_threads"]),
        "graph_M": int(row["M"]),
        "graph_efConstruction": int(row["efConstruction"]),
        "graph_build": row["graph_build"],
        "graph_path": row["index_path"],
        "source_csv": str(source),
    }


def combine(
    root: Path,
    sym_path: Path,
    no_rerank_path: Path,
    fixed_path: Path,
    calibrated_paths: dict[int, Path],
    fixed_r: int,
) -> Path:
    rows: list[dict[str, object]] = []

    sym_groups: dict[int, list[dict[str, str]]] = {}
    for row in read_rows(sym_path):
        sym_groups.setdefault(int(row["efSearch"]), []).append(row)
    for ef, group in sorted(sym_groups.items()):
        nq = int(group[0]["nq"])
        elapsed = statistics.mean(float(item["total_seconds"]) for item in group)
        rows.append(
            {
                "dataset": "msmarco_qwen1024",
                "corpus": "MS MARCO",
                "embedding": "Qwen3-Embedding-0.6B",
                "dim": int(group[0]["dim"]),
                "nb": int(group[0]["nb"]),
                "nq": nq,
                "split": "seed11.cal1000.test5980.test",
                "stage": 1,
                "method": "Sym-BQ",
                "query_scoring": "sign(query) vs sign(database), Hamming",
                "rerank_policy": "none",
                "efSearch": ef,
                "fixed_r": "",
                "calibration_tau": "",
                "calibration_protocol": "",
                "calibration_target": "",
                "calibration_empirical_coverage": "",
                "recall_at_10": statistics.mean(
                    float(item["recall_at_10"]) for item in group
                ),
                "qps": statistics.median(float(item["qps"]) for item in group),
                "qps_from_mean_seconds": nq / elapsed,
                "latency_ms_per_query": 1000.0 * elapsed / nq,
                "fp_vectors_per_query_mean": 0.0,
                "fp_vectors_p95": 0.0,
                "fp_vectors_p99": 0.0,
                "fp_vectors_max": 0.0,
                "fp_4k_pages_per_query_mean": 0.0,
                "rounds": len(group),
                "search_threads": int(group[0]["threads"]),
                "graph_M": int(group[0]["M"]),
                "graph_efConstruction": int(group[0]["efConstruction"]),
                "graph_build": "fp",
                "graph_path": group[0]["graph_path"],
                "source_csv": str(sym_path),
            }
        )

    for row in read_rows(no_rerank_path):
        rows.append(
            standard_adc_row(
                no_rerank_path,
                row,
                "ADC-BQ",
                2,
                "FP32 query vs sign(database), int16 4-bit-block LUT",
                "none",
                "",
                "",
                None,
            )
        )
    for row in read_rows(fixed_path):
        rows.append(
            standard_adc_row(
                fixed_path,
                row,
                f"ADC-BQ + Fixed R={fixed_r}",
                3,
                "FP32 query vs sign(database), int16 4-bit-block LUT",
                "fixed exact-FP32 SSD reranking",
                fixed_r,
                "",
                None,
            )
        )
    for ef, path in calibrated_paths.items():
        fitted = calibration_for_ef(ef)
        for row in read_rows(path):
            rows.append(
                standard_adc_row(
                    path,
                    row,
                    "BBQ + Calibrated R(q)",
                    4,
                    "FP32 query vs sign(database), int16 4-bit-block LUT",
                    "per-query calibrated exact-FP32 SSD reranking",
                    "adaptive",
                    float(fitted["tau"]),
                    fitted,
                )
            )

    rows.sort(key=lambda row: (int(row["stage"]), int(row["efSearch"])))
    output = root / "msmarco_qwen1024_recall_ablation_t1.csv"
    write_rows(output, rows)
    print(f"WROTE {output}", flush=True)
    return output


def plot(combined: Path, fixed_r: int) -> None:
    import matplotlib.pyplot as plt

    rows = read_rows(combined)
    methods = (
        "Sym-BQ",
        "ADC-BQ",
        f"ADC-BQ + Fixed R={fixed_r}",
        "BBQ + Calibrated R(q)",
    )
    styles = {
        "Sym-BQ": dict(color="#7f7f7f", marker="o", linestyle="--"),
        "ADC-BQ": dict(color="#1f77b4", marker="s", linestyle="-"),
        f"ADC-BQ + Fixed R={fixed_r}": dict(
            color="#ff7f0e", marker="D", linestyle="-"
        ),
        "BBQ + Calibrated R(q)": dict(
            color="#009e73", marker="^", linestyle="-"
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.7))
    all_efs = sorted({int(row["efSearch"]) for row in rows})
    labeled_efs = [
        ef for ef in (80, 120, 160, 200, 240, 320, 512, 1024) if ef in all_efs
    ]
    ax = axes[0]
    for method in methods:
        group = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["efSearch"]),
        )
        ax.plot(
            [int(row["efSearch"]) for row in group],
            [float(row["recall_at_10"]) for row in group],
            linewidth=2,
            markersize=6,
            label=method,
            **styles[method],
        )
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Recall@10")
    ax.set_xscale("log", base=2)
    ax.set_xticks(labeled_efs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    for method in (f"ADC-BQ + Fixed R={fixed_r}", "BBQ + Calibrated R(q)"):
        group = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["efSearch"]),
        )
        ax.plot(
            [int(row["efSearch"]) for row in group],
            [float(row["recall_at_10"]) for row in group],
            linewidth=2,
            markersize=6,
            label=method,
            **styles[method],
        )
    ax.set_xlabel("efSearch")
    ax.set_ylabel("Recall@10 (reranking zoom)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(labeled_efs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[2]
    for method in (f"ADC-BQ + Fixed R={fixed_r}", "BBQ + Calibrated R(q)"):
        group = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["efSearch"]),
        )
        ax.plot(
            [float(row["fp_vectors_per_query_mean"]) for row in group],
            [float(row["recall_at_10"]) for row in group],
            linewidth=2,
            markersize=6,
            label=method,
            **styles[method],
        )
        for row in group:
            if int(row["efSearch"]) not in labeled_efs:
                continue
            ax.annotate(
                f"ef={row['efSearch']}",
                (
                    float(row["fp_vectors_per_query_mean"]),
                    float(row["recall_at_10"]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
            )
    ax.set_xlabel("Average exact FP32 vectors read / query")
    ax.set_ylabel("Recall@10")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "MS MARCO · Qwen3-Embedding-0.6B (1024D), T=1, common FP-built HNSW",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        f"Fixed R={fixed_r} is the Table-3 reference; calibration uses a disjoint 1,000-query split and target pool-top10 coverage 0.99.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    for suffix in ("png", "pdf"):
        output = combined.with_name(f"msmarco_qwen1024_recall_ablation_t1.{suffix}")
        fig.savefig(output, dpi=200 if suffix == "png" else None)
        print(f"WROTE {output}", flush=True)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--fixed-r", type=int, default=75)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=46)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="0 evaluates the full 5,980-query held-out split",
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-combine", action="store_true")
    args = parser.parse_args()
    efs = list(dict.fromkeys(args.efs))
    if min(efs) < 10 or args.fixed_r < 10 or args.fixed_r > min(efs):
        parser.error("require efSearch >= fixed R >= k=10")
    if args.rounds < 1 or args.max_queries < 0:
        parser.error("rounds must be positive and max-queries nonnegative")
    # Static files must exist before missing calibration points can be fit.
    validate_inputs([ef for ef in efs if (CALIBRATION_ROOT / f"pool_top10_target99.ef{ef}.json").is_file()])
    fit_missing_calibrations(efs, args.cpu)
    validate_inputs(efs)
    root = args.result_root.resolve()
    nq = 5980 if args.max_queries == 0 else min(args.max_queries, 5980)
    main_max_queries = 0 if nq == 5980 else nq

    sym_path = root / "raw/symbq_no_rerank.csv"
    no_rerank_path = root / "raw/adcbq_no_rerank.csv"
    fixed_path = root / f"raw/adcbq_fixed_r{args.fixed_r}.csv"
    calibrated_paths = {
        ef: root / f"raw/adcbq_calibrated_ef{ef}.csv" for ef in efs
    }
    if not args.only_combine:
        sym_path = run_symbq(
            root, efs, args.rounds, nq, args.cpu, args.force
        )
        no_rerank_path = run_adc_no_rerank(
            root,
            efs,
            args.rounds,
            main_max_queries,
            nq,
            args.cpu,
            args.force,
        )
        fixed_path = run_adc_fixed(
            root,
            efs,
            args.fixed_r,
            args.rounds,
            main_max_queries,
            nq,
            args.cpu,
            args.force,
        )
        calibrated_paths = run_adc_calibrated(
            root,
            efs,
            args.rounds,
            main_max_queries,
            nq,
            args.cpu,
            args.force,
        )
    combined = combine(
        root,
        sym_path,
        no_rerank_path,
        fixed_path,
        calibrated_paths,
        args.fixed_r,
    )
    plot(combined, args.fixed_r)
    print("MSMARCO QWEN-1024 RECALL ABLATION COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
