#!/usr/bin/env python3
"""Remeasure selected MSMARCO calibrated ADC-BQ 4 KiB / ef=256 points.

The original sweep results remain untouched.  One warm-up process per layout is
discarded, then fresh-process observations are retained in an interleaved
baseline/coloc order.  Ratios, trial count, and output roots can be overridden
with environment variables.  A summary and an adjusted copy of the 4 KiB
memory-sweep CSV/figure are produced from the retained observations.
"""

from __future__ import annotations

import csv
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
RESULT_ROOT = Path(
    os.environ.get(
        "ADCBQ_RERUN_RESULT_ROOT",
        REPO / "experiment/msmarco_adcbq_calibrated_4k_p075_ef256_rerun",
    )
)
FIGURE_ROOT = Path(
    os.environ.get(
        "ADCBQ_RERUN_FIGURE_ROOT",
        REPO / "experiment/figures/msmarco_adcbq_calibrated_4k_p075_ef256_rerun",
    )
)
FP_PATH = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_page4k_ef256_qps_rerun/"
    "msmarco_openai1536/msmarco_openai1536.fp32.aligned4k.fbin"
)
PAGE = 4096
RATIOS = tuple(
    float(value)
    for value in os.environ.get("ADCBQ_RERUN_RATIOS", "0.75").split(",")
)
EF = 256
TRIALS = int(os.environ.get("ADCBQ_RERUN_TRIALS", "3"))
LAYOUTS = ("baseline", "coloc")
RATIO_TAG = "p075" if RATIOS == (0.75,) else "allratios"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ensure_binary() -> None:
    binary = calibrated.ADC_BINARIES[PAGE]
    if binary.exists() and binary.stat().st_mtime_ns >= calibrated.SOURCE.stat().st_mtime_ns:
        return
    env = os.environ.copy()
    env["PAGE_BYTES"] = str(PAGE)
    env["OUT"] = str(binary)
    subprocess.run([str(calibrated.BUILD_SCRIPT)], cwd=REPO, env=env, check=True)


def output_path(layout: str, ratio: float, trial: int) -> Path:
    phase = "warmup" if trial == 0 else "measured"
    return RESULT_ROOT / phase / (
        f"search_{layout}_{base.ratio_label(ratio)}_ef{EF}_"
        f"t{base.THREADS}_qd{base.ASYNC_QD}_trial{trial}.csv"
    )


def result_complete(path: Path, layout: str, ratio: float, tau: float) -> bool:
    try:
        rows = read_csv(path)
        if len(rows) != 1:
            return False
        row = rows[0]
        nq, _ = base.datasets_base.matrix_shape(base.DATASET.query)
        return (
            int(row["efSearch"]) == EF
            and int(row["rounds"]) == 1
            and int(row["search_threads"]) == base.THREADS
            and int(row["nq"]) == nq
            and int(row["rerank_n"]) == EF
            and row["calibrated_threshold_rerank_enabled"] == "true"
            and math.isclose(
                float(row["calibrated_threshold_rerank_tau"]),
                tau,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and int(row["index_page_bytes"]) == PAGE
            and int(row["index_page_cache_mib"]) == base.budget_mib(ratio)
            and row["index_page_async"] == "true"
            and int(row["index_page_qd"]) == base.ASYNC_QD
            and row["paged_index_status"] == "loaded"
            and row["fp_rerank_source"] == "odirect-uring"
            and row["fp_rerank_overlap_queries"] == "false"
            and int(row["colocation_iterations"])
            == (base.COLOCATION_ITERATIONS if layout == "coloc" else 0)
            and float(row["qps"]) > 0.0
        )
    except (OSError, KeyError, ValueError):
        return False


def run_one(layout: str, ratio: float, trial: int, tau: float) -> Path:
    output = output_path(layout, ratio, trial)
    if result_complete(output, layout, ratio, tau):
        print(f"SKIP complete: {output}", flush=True)
        return output

    base.adc.PAGE_BYTES = PAGE
    command = base.adc.common_command(
        base.DATASET,
        layout,
        base.ADC_ARTIFACT_ROOTS[PAGE],
        output,
        base.budget_mib(ratio),
        base.THREADS,
        (EF,),
        1,
        0,
    )
    command[0] = str(calibrated.ADC_BINARIES[PAGE])
    calibrated.replace_option(command, "--rerank-n", str(EF))
    command.extend(
        [
            "--index-page-async",
            "--index-page-qd",
            str(base.ASYNC_QD),
            "--rerank-dot",
            "fp32",
            "--fp-rerank-source",
            "odirect-uring",
            "--fp-rerank-norms",
            "onfetch",
            "--fp-rerank-path",
            str(FP_PATH),
            "--fp-rerank-qd",
            str(EF),
            "--calibrated-threshold-rerank-tau",
            repr(tau),
        ]
    )
    base.adc.run_command(command, output.with_suffix(".log"), base.THREADS)
    if not result_complete(output, layout, ratio, tau):
        raise RuntimeError(f"invalid rerun result: {output}")
    row = read_csv(output)[0]
    print(
        f"DONE trial={trial} ratio={ratio} layout={layout} qps={row['qps']} "
        f"recall={row['recall_at_k']}",
        flush=True,
    )
    return output


def observation(
    layout: str, ratio: float, trial: int, path: Path
) -> dict[str, object]:
    row = read_csv(path)[0]
    nq = int(row["nq"])
    graph_reads = float(row["index_page_read_bytes_mean"]) / 4096 / nq
    fp_reads = float(row["direct_submitted_bytes_mean"]) / 4096 / nq
    return {
        "layout": layout,
        "bq_ram_ratio_target": ratio,
        "trial": trial,
        "qps": float(row["qps"]),
        "recall_at_10": float(row["recall_at_k"]),
        "total_seconds": float(row["total_seconds_mean"]),
        "graph_page_hit_ratio": float(row["index_page_hit_ratio"]),
        "graph_page_accesses_per_query": float(row["index_page_accesses_mean"]) / nq,
        "graph_physical_4k_reads_per_query": graph_reads,
        "fp_vectors_mean": float(row["adaptive_rerank_count_mean"]),
        "fp_physical_4k_reads_per_query": fp_reads,
        "total_physical_4k_reads_per_query": graph_reads + fp_reads,
        "source_csv": str(path),
    }


def aggregate(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for ratio in RATIOS:
        for layout in LAYOUTS:
            group = [
                row for row in observations
                if row["layout"] == layout
                and math.isclose(float(row["bq_ram_ratio_target"]), ratio)
            ]
            qps = [float(row["qps"]) for row in group]
            seconds = [float(row["total_seconds"]) for row in group]
            nq, _ = base.datasets_base.matrix_shape(base.DATASET.query)
            original_path = calibrated.adc_output(PAGE, ratio, layout, EF)
            original = read_csv(original_path)[0]
            aggregate_qps = nq / statistics.mean(seconds)
            summary.append(
                {
                    "dataset": base.DATASET.key,
                    "page_kib": PAGE // 1024,
                    "bq_ram_ratio_target": ratio,
                    "bq_ram_budget_mib": base.budget_mib(ratio),
                    "layout": layout,
                    "efSearch": EF,
                    "threads": base.THREADS,
                    "graph_async_qd": base.ASYNC_QD,
                    "calibration_queries": calibrated.CALIBRATION_QUERIES,
                    "test_queries": nq,
                    "measured_trials": TRIALS,
                    "original_qps_3round_process": float(original["qps"]),
                    "rerun_qps_from_mean_seconds": aggregate_qps,
                    "rerun_qps_arithmetic_mean": statistics.mean(qps),
                    "rerun_qps_stdev": statistics.stdev(qps),
                    "rerun_qps_min": min(qps),
                    "rerun_qps_max": max(qps),
                    "change_vs_original_percent":
                        (aggregate_qps / float(original["qps"]) - 1.0) * 100.0,
                    "recall_at_10": statistics.mean(
                        float(row["recall_at_10"]) for row in group
                    ),
                    "graph_page_hit_ratio_mean": statistics.mean(
                        float(row["graph_page_hit_ratio"]) for row in group
                    ),
                    "total_physical_4k_reads_per_query_mean": statistics.mean(
                        float(row["total_physical_4k_reads_per_query"]) for row in group
                    ),
                    "original_source_csv": str(original_path),
                }
            )
    return summary


def adjusted_rows(
    observations: list[dict[str, object]], summary: list[dict[str, object]]
) -> list[dict[str, object]]:
    source = calibrated.RESULT_ROOT / "page4k_memory_sweep_adcbq_calibrated_rabitq.csv"
    rows: list[dict[str, object]] = [dict(row) for row in read_csv(source)]
    by_key = {
        (str(row["layout"]), float(row["bq_ram_ratio_target"])): row
        for row in summary
    }
    for row in rows:
        row["qps_original"] = row["qps"]
        row["qps_stdev"] = ""
        row["qps_observations"] = ""
        row["measurement_source"] = "original_sweep"
        if not (
            row["method"] == calibrated.METHOD_ADC
            and int(row["efSearch"]) == EF
            and any(
                math.isclose(float(row["bq_ram_ratio_target"]), ratio)
                for ratio in RATIOS
            )
        ):
            continue
        layout = str(row["layout"])
        ratio = float(row["bq_ram_ratio_target"])
        group = [
            item for item in observations
            if item["layout"] == layout
            and math.isclose(float(item["bq_ram_ratio_target"]), ratio)
        ]
        aggregate_row = by_key[(layout, ratio)]
        row["qps"] = aggregate_row["rerun_qps_from_mean_seconds"]
        row["qps_stdev"] = aggregate_row["rerun_qps_stdev"]
        row["qps_observations"] = TRIALS
        row["latency_ms_batch_average"] = statistics.mean(
            float(item["total_seconds"]) for item in group
        ) * 1000 / int(row["test_queries"])
        for field in (
            "graph_page_hit_ratio",
            "graph_page_accesses_per_query",
            "graph_physical_4k_reads_per_query",
            "fp_vectors_mean",
            "fp_physical_4k_reads_per_query",
            "total_physical_4k_reads_per_query",
        ):
            row[field] = statistics.mean(float(item[field]) for item in group)
        row["measurement_source"] = "fresh_process_rerun_mean"
    return rows


def plot(rows: list[dict[str, object]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if int(row["efSearch"]) == EF]
    colors = {calibrated.METHOD_ADC: "#0072B2", calibrated.METHOD_RABITQ: "#D55E00"}
    markers = {"baseline": "s", "coloc": "o"}
    linestyles = {"baseline": "--", "coloc": "-"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for method, color in colors.items():
        for layout in LAYOUTS:
            group = sorted(
                (
                    row for row in selected
                    if row["method"] == method and row["layout"] == layout
                ),
                key=lambda row: float(row["bq_ram_ratio_target"]),
            )
            x = [float(row["bq_ram_ratio_target"]) * 100 for row in group]
            label = f"{method}, {layout}"
            axes[0].plot(
                x,
                [float(row["qps"]) for row in group],
                color=color,
                marker=markers[layout],
                linestyle=linestyles[layout],
                label=label,
            )
            axes[1].plot(
                x,
                [float(row["total_physical_4k_reads_per_query"]) for row in group],
                color=color,
                marker=markers[layout],
                linestyle=linestyles[layout],
                label=label,
            )
            if method == calibrated.METHOD_ADC:
                rerun_points = [
                    row for row in group
                    if any(
                        math.isclose(float(row["bq_ram_ratio_target"]), ratio)
                        for ratio in RATIOS
                    )
                ]
                axes[0].errorbar(
                    [float(point["bq_ram_ratio_target"]) * 100 for point in rerun_points],
                    [float(point["qps"]) for point in rerun_points],
                    yerr=[float(point["qps_stdev"]) for point in rerun_points],
                    fmt="none",
                    ecolor=color,
                    capsize=4,
                    alpha=0.8,
                )
    axes[0].set_ylabel("QPS")
    axes[1].set_ylabel("Total physical 4 KiB reads/query")
    for axis in axes:
        axis.set_xlabel("BQ-based graph/code RAM budget (%)")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "MSMARCO OpenAI-1536, 4 KiB pages, efSearch=256\n"
        f"ADC-BQ selected points replaced by {TRIALS} fresh-process calibrated reruns"
    )
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    output = FIGURE_ROOT / f"page4k_memory_sweep_ef256_{RATIO_TAG}_rerun_adjusted.png"
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    tau, _ = calibrated.load_tau()
    split = calibration.make_split(
        base.DATASET,
        calibrated.CALIBRATION_QUERIES,
        calibrated.CALIBRATION_SEED,
        0,
    )
    dataset = replace(base.DATASET, query=split.test_query, gt=split.test_gt)
    calibrated.configure_base(dataset)
    base.datasets_base.validate(base.DATASET)
    ensure_binary()
    for layout in LAYOUTS:
        base.adc_build(PAGE, layout)
    base.adc.prepare_aligned_fp(FP_PATH, base.DATASET)

    for trial in range(TRIALS + 1):
        order = LAYOUTS if trial % 2 == 0 else tuple(reversed(LAYOUTS))
        ratio_order = RATIOS if trial % 2 == 0 else tuple(reversed(RATIOS))
        for ratio in ratio_order:
            for layout in order:
                run_one(layout, ratio, trial, tau)

    observations = [
        observation(layout, ratio, trial, output_path(layout, ratio, trial))
        for trial in range(1, TRIALS + 1)
        for ratio in RATIOS
        for layout in LAYOUTS
    ]
    summary = aggregate(observations)
    write_csv(RESULT_ROOT / "observations.csv", observations)
    write_csv(RESULT_ROOT / "summary.csv", summary)
    adjusted = adjusted_rows(observations, summary)
    adjusted_csv = RESULT_ROOT / (
        f"page4k_memory_sweep_ef256_{RATIO_TAG}_rerun_adjusted.csv"
    )
    write_csv(adjusted_csv, adjusted)
    figure = plot(adjusted)
    print(f"SUMMARY {RESULT_ROOT / 'summary.csv'}", flush=True)
    print(f"ADJUSTED_CSV {adjusted_csv}", flush=True)
    print(f"FIGURE {figure}", flush=True)


if __name__ == "__main__":
    main()
