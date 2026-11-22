#!/usr/bin/env python3
"""Interleaved hybrid-I/O QPS rerun for MSMARCO efSearch=256.

The original sweep ran complete ADC-BQ and RaBitQ blocks sequentially.  This
targeted rerun launches one fresh process per observation and alternates the
two methods in an ABBA order.  One warmup observation is discarded and five
independent observations are retained for every layout/RAM setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

import run_adcbq_calibrated_threshold_overlap as calibration
import run_msmarco_adcbq_calibrated_rabitq_memory_page_sweep as calibrated
import run_msmarco_adcbq_rabitq_memory_page_sweep as base


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/msmarco_page4k_ef256_qps_rerun_alllevels_allhitfast_async_qd16"
FIGURE_ROOT = REPO / (
    "experiment/figures/msmarco_page4k_ef256_qps_rerun_alllevels_allhitfast_async_qd16"
)
FP_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_page4k_ef256_qps_rerun"
)
RABITQ_PAGED_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_page_memory_sweep_alllevels_hybrid"
)
RABITQ_BINARIES = {
    page: REPO
    / f"experiment/rabitq_paged_adaptive_{page // 1024}k_alllevels_batchcache_benchmark"
    for page in base.PAGES
}
ADC_BINARIES = {
    page: REPO / (
        "experiment/adcbq_optimized/"
        f"faiss_adcbq_hnsw_{page // 1024}k_graph_batchcache_calibrated"
    )
    for page in base.PAGES
}
GRAPH_IO_SELECTION = (
    REPO / "experiment/msmarco_adcbq_graph_io_calibration/selection.json"
)
SOURCE = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
ADC_BUILD_SCRIPT = REPO / "experiment/adcbq_optimized/build.sh"
RABITQ_SOURCE = REPO / "experiment/rabitq_paged_adaptive_benchmark.cpp"
RABITQ_BUILD = REPO / "experiment/rabitq_v051_adaptive_odirect_build"
ADC_GRAPH_QD = 16
ADC_SYNC_MISS_MAX = 0
PAGE = 4096
EF = 256
THREADS = 24
MEASURED_TRIALS = 3
WARMUP_TRIALS = 1
METHOD_ADC = "ADC-BQ calibrated, batch-cache, all-hit-fast-async-QD16"
METHOD_RABITQ = "RaBitQ-1bit-c16-adaptive-FP-all-level-paged-batch-cache"
METHOD_ORDER = {
    0: (METHOD_ADC, METHOD_RABITQ),
    1: (METHOD_ADC, METHOD_RABITQ),
    2: (METHOD_RABITQ, METHOD_ADC),
    3: (METHOD_RABITQ, METHOD_ADC),
}


def configure_page(page: int) -> None:
    global PAGE, RESULT_ROOT, FIGURE_ROOT
    PAGE = page
    page_kib = page // 1024
    RESULT_ROOT = REPO / (
        f"experiment/msmarco_page{page_kib}k_ef256_qps_rerun_"
        "alllevels_batchcache_allhitfast_async_qd16"
    )
    FIGURE_ROOT = REPO / (
        f"experiment/figures/msmarco_page{page_kib}k_ef256_qps_rerun_"
        "alllevels_batchcache_allhitfast_async_qd16"
    )


def ensure_binaries() -> None:
    adc_binary = ADC_BINARIES[PAGE]
    if not adc_binary.exists() or adc_binary.stat().st_mtime_ns < SOURCE.stat().st_mtime_ns:
        env = os.environ.copy()
        env["PAGE_BYTES"] = str(PAGE)
        env["OUT"] = str(adc_binary)
        subprocess.run([str(ADC_BUILD_SCRIPT)], cwd=REPO, env=env, check=True)

    rabitq_binary = RABITQ_BINARIES[PAGE]
    header = RABITQ_BUILD / "include/rabitqlib/index/hnsw/hnsw.hpp"
    newest_source = max(RABITQ_SOURCE.stat().st_mtime_ns, header.stat().st_mtime_ns)
    if rabitq_binary.exists() and rabitq_binary.stat().st_mtime_ns >= newest_source:
        return
    subprocess.run(
        [
            "g++", "-std=c++17", "-O3", "-DNDEBUG",
            f"-DRABITQ_PAGED_INDEX_PAGE_BYTES={PAGE}",
            "-march=native", "-fopenmp",
            f"-I{RABITQ_BUILD / 'include'}",
            str(RABITQ_SOURCE),
            str(RABITQ_BUILD / "lib/librabitq_core.a"),
            "-luring", "-o", str(rabitq_binary),
        ],
        cwd=REPO,
        check=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rabitq_paged_path(layout: str) -> Path:
    return RABITQ_PAGED_ROOT / base.DATASET.key / (
        f"{base.DATASET.key}.rabitq_b1.alllevels.{layout}.paged{PAGE // 1024}k"
    )


def prepare_rabitq_layout(layout: str) -> dict[str, str]:
    paged = rabitq_paged_path(layout)
    output = RESULT_ROOT / "build" / f"rabitq_alllevels_{layout}.csv"
    expected_iterations = base.COLOCATION_ITERATIONS if layout == "coloc" else 0
    try:
        rows = read_csv(output)
        row = rows[0]
        if (
            len(rows) == 1
            and paged.exists()
            and int(row["page_bytes"]) == PAGE
            and row["colocated"] == ("true" if layout == "coloc" else "false")
            and int(row["colocation_iterations"]) == expected_iterations
            and row["upper_graph_paged"] == "true"
            and int(row["resident_upper_graph_bytes"]) == 0
            and int(row["paged_index_file_bytes"]) == paged.stat().st_size
        ):
            return row
    except (OSError, IndexError, KeyError, ValueError):
        pass

    paged.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(RABITQ_BINARIES[PAGE]),
        "--dataset", base.DATASET.key,
        "--train", str(base.DATASET.train),
        "--query", str(base.DATASET.query),
        "--gt", str(base.DATASET.gt),
        "--index", str(base.RABITQ_BASE_INDEX),
        "--paged-index", str(paged),
        "--out", str(output),
        "--threads", str(THREADS),
        "--rounds", "1",
        "--k", "10",
        "--efs", str(EF),
        "--max-queries", "1",
        "--cache-mib", "1",
        "--prepare-only",
    ]
    if layout == "coloc":
        command.extend(
            [
                "--colocation-iterations", str(base.COLOCATION_ITERATIONS),
                "--colocation-partition-size", str(base.COLOCATION_PARTITION_SIZE),
            ]
        )
    base.adc.run_command(command, output.with_suffix(".log"), THREADS)
    rows = read_csv(output)
    if len(rows) != 1:
        raise RuntimeError(f"invalid all-level RaBitQ build result: {output}")
    return rows[0]


def output_path(
    method: str, layout: str, ratio: float, trial: int
) -> Path:
    phase = "warmup" if trial == 0 else "measured"
    method_dir = "adcbq" if method == METHOD_ADC else "rabitq"
    return RESULT_ROOT / phase / method_dir / (
        f"search_{layout}_{base.ratio_label(ratio)}_ef{EF}_"
        f"t{THREADS}_trial{trial}.csv"
    )


def result_complete(
    path: Path,
    method: str,
    layout: str,
    ratio: float,
    upper_graph_bytes: int,
    tau: float,
) -> bool:
    try:
        rows = read_csv(path)
        if len(rows) != 1:
            return False
        row = rows[0]
        nq, _ = base.datasets_base.matrix_shape(base.DATASET.query)
        common = (
            int(row["efSearch"]) == EF
            and int(row["rounds"]) == 1
            and int(row["search_threads"]) == THREADS
            and int(row["nq"]) == nq
            and float(row["qps"]) > 0
            and int(row["index_page_bytes"]) == PAGE
            and row["paged_index_status"] == "loaded"
        )
        if not common:
            return False
        if method == METHOD_ADC:
            return (
                int(row["index_page_cache_mib"]) == base.budget_mib(ratio)
                and row["calibrated_threshold_rerank_enabled"] == "true"
                and math.isclose(
                    float(row["calibrated_threshold_rerank_tau"]),
                    tau,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
                and row["index_page_async"] == "true"
                and int(row["index_page_qd"]) == ADC_GRAPH_QD
                and int(row["index_page_async_sync_miss_max"])
                == ADC_SYNC_MISS_MAX
                and row["fp_rerank_source"] == "odirect-uring"
                and row["fp_rerank_overlap_queries"] == "false"
                and int(row["colocation_iterations"])
                == (base.COLOCATION_ITERATIONS if layout == "coloc" else 0)
            )
        return (
            int(row["index_page_cache_mib"])
            == base.budget_mib(ratio)
            and int(row["upper_graph_bytes"]) == upper_graph_bytes
            and row["upper_graph_paged"] == "true"
            and int(row["resident_upper_graph_bytes"]) == 0
            and row["colocated"]
            == ("true" if layout == "coloc" else "false")
            and "adaptive-FP" in row["method"]
        )
    except (OSError, KeyError, ValueError):
        return False


def run_adc(
    output: Path,
    layout: str,
    ratio: float,
    tau: float,
    sidecar: Path,
) -> None:
    base.adc.PAGE_BYTES = PAGE
    command = base.adc.common_command(
        base.DATASET,
        layout,
        base.ADC_ARTIFACT_ROOTS[PAGE],
        output,
        base.budget_mib(ratio),
        THREADS,
        (EF,),
        1,
        0,
    )
    command[0] = str(ADC_BINARIES[PAGE])
    calibrated.replace_option(command, "--rerank-n", str(EF))
    command.extend(
        [
            "--index-page-async",
            "--index-page-qd",
            str(ADC_GRAPH_QD),
            "--index-page-async-sync-miss-max",
            str(ADC_SYNC_MISS_MAX),
            "--rerank-dot",
            "fp32",
            "--fp-rerank-source",
            "odirect-uring",
            "--fp-rerank-norms",
            "onfetch",
            "--fp-rerank-path",
            str(sidecar),
            "--fp-rerank-qd",
            str(EF),
            "--calibrated-threshold-rerank-tau",
            repr(tau),
        ]
    )
    base.adc.run_command(command, output.with_suffix(".log"), THREADS)


def run_rabitq(
    output: Path,
    layout: str,
    ratio: float,
    upper_graph_bytes: int,
    sidecar: Path,
) -> None:
    command = [
        str(RABITQ_BINARIES[PAGE]),
        "--dataset",
        base.DATASET.key,
        "--train",
        str(base.DATASET.train),
        "--query",
        str(base.DATASET.query),
        "--gt",
        str(base.DATASET.gt),
        "--index",
        str(base.RABITQ_BASE_INDEX),
        "--paged-index",
        str(rabitq_paged_path(layout)),
        "--fp-path",
        str(sidecar),
        "--out",
        str(output),
        "--threads",
        str(THREADS),
        "--rounds",
        "1",
        "--k",
        "10",
        "--efs",
        str(EF),
    ]
    if layout == "coloc":
        command.extend(
            [
                "--colocation-iterations",
                str(base.COLOCATION_ITERATIONS),
                "--colocation-partition-size",
                str(base.COLOCATION_PARTITION_SIZE),
            ]
        )
    command.extend(
        [
            "--cache-mib",
            str(base.budget_mib(ratio)),
        ]
    )
    base.adc.run_command(command, output.with_suffix(".log"), THREADS)


def completed_count(
    build: dict[str, dict[str, str]], tau: float
) -> int:
    completed = 0
    for trial in range(WARMUP_TRIALS + MEASURED_TRIALS):
        for layout in base.LAYOUTS:
            upper = int(build[layout]["upper_graph_bytes"])
            for ratio in base.RATIOS:
                for method in (METHOD_ADC, METHOD_RABITQ):
                    if result_complete(
                        output_path(method, layout, ratio, trial),
                        method,
                        layout,
                        ratio,
                        upper,
                        tau,
                    ):
                        completed += 1
    return completed


def write_progress(
    build: dict[str, dict[str, str]], tau: float
) -> None:
    total = (
        (WARMUP_TRIALS + MEASURED_TRIALS)
        * len(base.LAYOUTS)
        * len(base.RATIOS)
        * 2
    )
    completed = completed_count(build, tau)
    calibrated.write_json(
        RESULT_ROOT / "progress.json",
        {
            "status": "complete" if completed == total else "running",
            "completed_jobs": completed,
            "total_jobs": total,
            "warmup_jobs": WARMUP_TRIALS * len(base.LAYOUTS) * len(base.RATIOS) * 2,
            "measured_jobs": MEASURED_TRIALS * len(base.LAYOUTS) * len(base.RATIOS) * 2,
        },
    )


def normalized_row(
    method: str,
    layout: str,
    ratio: float,
    trial: int,
    source: Path,
) -> dict[str, object]:
    row = read_csv(source)[0]
    nq = int(row["nq"])
    graph_4k = float(row["index_page_read_bytes_mean"]) / 4096 / nq
    if method == METHOD_ADC:
        fp_vectors = float(row["adaptive_rerank_count_mean"])
        fp_4k = float(row["direct_submitted_bytes_mean"]) / 4096 / nq
        async_wait = float(row["index_async_wait_seconds_sum_mean"])
        recall = float(row["recall_at_k"])
    else:
        fp_vectors = float(row["fp_reads_mean"])
        fp_4k = float(row["fp_physical_4k_reads_per_query"])
        async_wait = 0.0
        recall = float(row["recall_at_k"])
    return {
        "method": method,
        "layout": layout,
        "bq_ram_ratio_target": ratio,
        "trial": trial,
        "qps": float(row["qps"]),
        "recall_at_10": recall,
        "graph_page_hit_ratio": float(row["index_page_hit_ratio"]),
        "graph_physical_4k_reads_per_query": graph_4k,
        "fp_vectors_mean": fp_vectors,
        "fp_physical_4k_reads_per_query": fp_4k,
        "total_physical_4k_reads_per_query": graph_4k + fp_4k,
        "index_async_wait_seconds_sum": async_wait,
        "source_csv": str(source),
    }


def original_qps(method: str, layout: str, ratio: float) -> float:
    old_root = (
        REPO
        / "experiment/msmarco_adcbq_calibrated_rabitq_memory_page_sweep"
        / base.DATASET.key
    )
    if method == METHOD_ADC:
        source = old_root / f"adcbq/{PAGE // 1024}k" / (
            f"search_{layout}_{base.ratio_label(ratio)}_t24_async_qd48_"
            "calibrated_tau_ef256_ef256.csv"
        )
        return float(read_csv(source)[0]["qps"])
    source = old_root / f"rabitq/{PAGE // 1024}k" / (
        f"search_{layout}_{base.ratio_label(ratio)}_t24_sync.csv"
    )
    row = next(row for row in read_csv(source) if int(row["efSearch"]) == EF)
    return float(row["qps"])


def summarize() -> Path:
    rounds: list[dict[str, object]] = []
    for trial in range(1, MEASURED_TRIALS + 1):
        for layout in base.LAYOUTS:
            for ratio in base.RATIOS:
                for method in (METHOD_ADC, METHOD_RABITQ):
                    source = output_path(method, layout, ratio, trial)
                    rounds.append(
                        normalized_row(method, layout, ratio, trial, source)
                    )
    calibrated.write_csv(RESULT_ROOT / "qps_rerun_rounds.csv", rounds)

    summary: list[dict[str, object]] = []
    for layout in base.LAYOUTS:
        for ratio in base.RATIOS:
            for method in (METHOD_ADC, METHOD_RABITQ):
                selected = [
                    row
                    for row in rounds
                    if row["method"] == method
                    and row["layout"] == layout
                    and math.isclose(
                        float(row["bq_ram_ratio_target"]), ratio
                    )
                ]
                qps = [float(row["qps"]) for row in selected]
                mean = statistics.mean(qps)
                median = statistics.median(qps)
                original = original_qps(method, layout, ratio)
                summary.append(
                    {
                        "method": method,
                        "layout": layout,
                        "bq_ram_ratio_target": ratio,
                        "page_bytes": PAGE,
                        "efSearch": EF,
                        "threads": THREADS,
                        "measured_trials": MEASURED_TRIALS,
                        "qps_original_3round_mean": original,
                        "qps_rerun_mean": mean,
                        "qps_rerun_median": median,
                        "qps_rerun_stdev": statistics.stdev(qps),
                        "qps_rerun_cv": statistics.stdev(qps) / mean,
                        "qps_rerun_min": min(qps),
                        "qps_rerun_max": max(qps),
                        "mean_vs_original_percent": (mean / original - 1) * 100,
                        "median_vs_original_percent": (median / original - 1) * 100,
                        "recall_at_10": statistics.mean(
                            float(row["recall_at_10"]) for row in selected
                        ),
                        "graph_page_hit_ratio": statistics.mean(
                            float(row["graph_page_hit_ratio"])
                            for row in selected
                        ),
                        "graph_physical_4k_reads_per_query": statistics.mean(
                            float(row["graph_physical_4k_reads_per_query"])
                            for row in selected
                        ),
                        "fp_vectors_mean": statistics.mean(
                            float(row["fp_vectors_mean"]) for row in selected
                        ),
                        "total_physical_4k_reads_per_query": statistics.mean(
                            float(row["total_physical_4k_reads_per_query"])
                            for row in selected
                        ),
                    }
                )
    output = RESULT_ROOT / "qps_rerun_summary.csv"
    calibrated.write_csv(output, summary)
    return output


def plot(summary_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(summary_path)
    colors = {METHOD_ADC: "#1f77b4", METHOD_RABITQ: "#d95f02"}
    labels = {
        METHOD_ADC: "ADC-BQ calibrated, graph QD16",
        METHOD_RABITQ: "RaBitQ adaptive, all-level paged",
    }
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for axis, layout in zip(axes, base.LAYOUTS):
        for method in (METHOD_ADC, METHOD_RABITQ):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["layout"] == layout and row["method"] == method
                ),
                key=lambda row: float(row["bq_ram_ratio_target"]),
            )
            x = [float(row["bq_ram_ratio_target"]) * 100 for row in selected]
            y = [float(row["qps_rerun_mean"]) for row in selected]
            error = [float(row["qps_rerun_stdev"]) for row in selected]
            old = [float(row["qps_original_3round_mean"]) for row in selected]
            axis.errorbar(
                x,
                y,
                yerr=error,
                color=colors[method],
                marker="o",
                capsize=4,
                linewidth=2,
                label=f"{labels[method]} rerun mean (±1 SD)",
            )
            axis.plot(
                x,
                old,
                color=colors[method],
                marker="x",
                linestyle="--",
                alpha=0.55,
                label=f"{labels[method]} original hybrid mean",
            )
        axis.set_title(layout)
        axis.set_xlabel("BQ-based RAM budget (%)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("QPS")
    handles, labels_out = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_out,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"MSMARCO OpenAI-1536, {PAGE // 1024}KB, efSearch=256, T=24: hybrid-I/O QPS rerun",
        y=1.13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURE_ROOT / f"qps_rerun_{PAGE // 1024}k_ef256_t24.{suffix}",
            dpi=190 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--page-bytes", type=int, choices=base.PAGES, default=4096
    )
    args = parser.parse_args()
    configure_page(args.page_bytes)
    selection = json.loads(GRAPH_IO_SELECTION.read_text(encoding="utf-8"))
    if (
        selection.get("selected_engine") != "async"
        or int(selection.get("selected_qd", 0)) != ADC_GRAPH_QD
    ):
        raise RuntimeError(
            f"graph-I/O calibration did not select async QD{ADC_GRAPH_QD}: "
            f"{GRAPH_IO_SELECTION}"
        )
    ensure_binaries()
    tau, calibration_payload = calibrated.load_tau()
    split = calibration.make_split(
        base.DATASET,
        calibrated.CALIBRATION_QUERIES,
        calibrated.CALIBRATION_SEED,
        0,
    )
    dataset = replace(base.DATASET, query=split.test_query, gt=split.test_gt)
    calibrated.configure_base(dataset)
    base.datasets_base.validate(base.DATASET)
    if not ADC_BINARIES[PAGE].exists():
        raise FileNotFoundError(ADC_BINARIES[PAGE])
    if not RABITQ_BINARIES[PAGE].exists():
        raise FileNotFoundError(RABITQ_BINARIES[PAGE])

    sidecar = FP_ROOT / base.DATASET.key / f"{base.DATASET.key}.fp32.aligned4k.fbin"
    sidecar_existed = sidecar.exists()
    base.adc.prepare_aligned_fp(sidecar, base.DATASET)
    builds = {layout: prepare_rabitq_layout(layout) for layout in base.LAYOUTS}
    write_progress(builds, tau)

    try:
        combinations = [
            (layout, ratio)
            for layout in base.LAYOUTS
            for ratio in base.RATIOS
        ]
        for trial in range(WARMUP_TRIALS + MEASURED_TRIALS):
            ordered_combinations = (
                combinations if trial % 2 == 0 else list(reversed(combinations))
            )
            for layout, ratio in ordered_combinations:
                upper = int(builds[layout]["upper_graph_bytes"])
                for method in METHOD_ORDER[trial]:
                    output = output_path(method, layout, ratio, trial)
                    if result_complete(
                        output, method, layout, ratio, upper, tau
                    ):
                        print(f"SKIP complete rerun observation: {output}", flush=True)
                        continue
                    output.parent.mkdir(parents=True, exist_ok=True)
                    print(
                        f"RUN trial={trial} method={method} layout={layout} "
                        f"ratio={ratio}",
                        flush=True,
                    )
                    if method == METHOD_ADC:
                        run_adc(output, layout, ratio, tau, sidecar)
                    else:
                        run_rabitq(output, layout, ratio, upper, sidecar)
                    if not result_complete(
                        output, method, layout, ratio, upper, tau
                    ):
                        raise RuntimeError(f"invalid rerun result: {output}")
                    write_progress(builds, tau)

        summary = summarize()
        try:
            plot(summary)
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            print(
                "SKIP plot: matplotlib is unavailable; CSV/JSON results are complete",
                flush=True,
            )
        calibrated.write_json(
            RESULT_ROOT / "run_manifest.json",
            {
                "dataset": base.DATASET.key,
                "query_split": {
                    "seed": calibrated.CALIBRATION_SEED,
                    "calibration_queries": split.calibration_rows,
                    "test_queries": split.test_rows,
                },
                "page_bytes": PAGE,
                "efSearch": EF,
                "threads": THREADS,
                "warmup_trials_discarded": WARMUP_TRIALS,
                "measured_trials": MEASURED_TRIALS,
                "schedule": "ABBA method order with alternating configuration order",
                "adc_graph_io": (
                    f"all-cache-hit fast return; any miss uses query-internal "
                    f"async io_uring QD{ADC_GRAPH_QD} with a real in-flight cap"
                ),
                "adc_graph_io_calibration": selection,
                "adc_fp_io": "query-internal batched O_DIRECT io_uring QD256",
                "rabitq_graph_io": (
                    "all HNSW levels use the same O_DIRECT page cache; "
                    "synchronous pread across 24 search threads"
                ),
                "rabitq_resident_upper_graph_bytes": 0,
                "calibration": calibration_payload,
            },
        )
        write_progress(builds, tau)
    finally:
        if not sidecar_existed and sidecar.exists():
            size = sidecar.stat().st_size
            sidecar.unlink()
            print(
                f"REMOVED transient FP sidecar bytes={size}: {sidecar}",
                flush=True,
            )


if __name__ == "__main__":
    main()
