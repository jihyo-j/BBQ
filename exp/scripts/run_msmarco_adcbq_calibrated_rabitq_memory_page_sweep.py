#!/usr/bin/env python3
"""MSMARCO calibrated ADC-BQ vs. RaBitQ page/RAM sweep.

The ADC-BQ policy uses the existing single-tau calibration fitted on 1,000
queries at efSearch=256.  Both ADC-BQ and RaBitQ are measured on the same
5,980-query held-out split.  Existing graph indexes and paged layouts are
reused; only the search matrix is rerun.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import run_adcbq_calibrated_threshold_overlap as calibration
import run_msmarco_adcbq_rabitq_memory_page_sweep as base


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/msmarco_adcbq_calibrated_rabitq_memory_page_sweep"
FIGURE_ROOT = (
    REPO / "experiment/figures/msmarco_adcbq_calibrated_rabitq_memory_page_sweep"
)
FP_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/"
    "msmarco_adcbq_calibrated_rabitq_memory_page_sweep"
)
CALIBRATION_PATH = (
    REPO
    / "experiment/adcbq_calibrated_threshold_overlap/sieun_msmarco_openai1536/"
    "calibration.single_tau_ef256.json"
)
CALIBRATION_QUERIES = 1000
CALIBRATION_SEED = 11
CALIBRATION_REFERENCE_EF = 256
METHOD_ADC = "ADC-BQ calibrated"
METHOD_RABITQ = "RaBitQ-1bit-c16-adaptive-FP"
SOURCE = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
BUILD_SCRIPT = REPO / "experiment/adcbq_optimized/build.sh"
ADC_BINARIES = {
    page: REPO
    / (
        "experiment/adcbq_optimized/"
        f"faiss_adcbq_hnsw_{page // 1024}k_graph_async_calibrated"
    )
    for page in base.PAGES
}


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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_tau() -> tuple[float, dict[str, object]]:
    payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    fitted = payload["calibration"]
    if int(payload["reference_ef"]) != CALIBRATION_REFERENCE_EF:
        raise RuntimeError(f"calibration reference mismatch: {CALIBRATION_PATH}")
    if int(fitted["calibration_rows"]) != CALIBRATION_QUERIES:
        raise RuntimeError(f"calibration row mismatch: {CALIBRATION_PATH}")
    return float(fitted["tau"]), payload


def ensure_adc_binaries() -> None:
    for page, binary in ADC_BINARIES.items():
        if binary.exists() and binary.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
            continue
        env = os.environ.copy()
        env["PAGE_BYTES"] = str(page)
        env["OUT"] = str(binary)
        subprocess.run([str(BUILD_SCRIPT)], cwd=REPO, env=env, check=True)


def replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise RuntimeError(f"missing command option {option}") from exc
    command[index + 1] = value


def adc_output(page: int, ratio: float, layout: str, ef: int) -> Path:
    return RESULT_ROOT / base.DATASET.key / "adcbq" / f"{page // 1024}k" / (
        f"search_{layout}_{base.ratio_label(ratio)}_t{base.THREADS}_"
        f"async_qd{base.ASYNC_QD}_calibrated_tau_ef{CALIBRATION_REFERENCE_EF}_"
        f"ef{ef}.csv"
    )


def rabitq_output(page: int, ratio: float, layout: str) -> Path:
    return RESULT_ROOT / base.DATASET.key / "rabitq" / f"{page // 1024}k" / (
        f"search_{layout}_{base.ratio_label(ratio)}_t{base.THREADS}_sync.csv"
    )


def adc_complete(
    path: Path, page: int, ratio: float, layout: str, ef: int, tau: float
) -> bool:
    try:
        rows = read_csv(path)
        if len(rows) != 1:
            return False
        row = rows[0]
        nq, _ = base.datasets_base.matrix_shape(base.DATASET.query)
        return (
            int(row["efSearch"]) == ef
            and int(row["rounds"]) == base.ROUNDS
            and int(row["search_threads"]) == base.THREADS
            and int(row["nq"]) == nq
            and int(row["rerank_n"]) == ef
            and row["calibrated_threshold_rerank_enabled"] == "true"
            and math.isclose(
                float(row["calibrated_threshold_rerank_tau"]),
                tau,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and int(row["index_page_bytes"]) == page
            and int(row["index_page_cache_mib"]) == base.budget_mib(ratio)
            and row["index_page_async"] == "true"
            and int(row["index_page_qd"]) == base.ASYNC_QD
            and row["paged_index_status"] == "loaded"
            and row["fp_rerank_source"] == "odirect-uring"
            and row["fp_rerank_overlap_queries"] == "false"
            and float(row["adaptive_rerank_count_mean"]) >= 10.0
            and int(row["colocation_iterations"])
            == (base.COLOCATION_ITERATIONS if layout == "coloc" else 0)
        )
    except (OSError, KeyError, ValueError):
        return False


def run_adc(
    page: int,
    ratio: float,
    layout: str,
    ef: int,
    tau: float,
    sidecar: Path,
) -> Path:
    output = adc_output(page, ratio, layout, ef)
    if adc_complete(output, page, ratio, layout, ef, tau):
        print(f"SKIP complete calibrated ADC-BQ: {output}", flush=True)
        return output
    base.adc.PAGE_BYTES = page
    command = base.adc.common_command(
        base.DATASET,
        layout,
        base.ADC_ARTIFACT_ROOTS[page],
        output,
        base.budget_mib(ratio),
        base.THREADS,
        (ef,),
        base.ROUNDS,
        0,
    )
    command[0] = str(ADC_BINARIES[page])
    replace_option(command, "--rerank-n", str(ef))
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
            str(sidecar),
            "--fp-rerank-qd",
            str(ef),
            "--calibrated-threshold-rerank-tau",
            repr(tau),
        ]
    )
    base.adc.run_command(command, output.with_suffix(".log"), base.THREADS)
    if not adc_complete(output, page, ratio, layout, ef, tau):
        raise RuntimeError(f"invalid calibrated ADC-BQ result: {output}")
    print(
        f"DONE calibrated ADC-BQ page={page} ratio={ratio} "
        f"layout={layout} ef={ef}",
        flush=True,
    )
    return output


def adc_group_complete(
    page: int, ratio: float, layout: str, tau: float
) -> bool:
    return all(
        adc_complete(adc_output(page, ratio, layout, ef), page, ratio, layout, ef, tau)
        for ef in base.EFS
    )


def configure_base(dataset: object) -> None:
    base.DATASET = dataset
    base.RESULT_ROOT = RESULT_ROOT
    base.FIGURE_ROOT = FIGURE_ROOT
    base.rabitq_output = rabitq_output


def progress(
    rabitq_builds: dict[tuple[int, str], dict[str, str]], tau: float
) -> dict[str, object]:
    total = len(base.RATIOS) * len(base.PAGES) * len(base.LAYOUTS) * 2
    completed: list[dict[str, object]] = []
    for ratio in base.RATIOS:
        for page in base.PAGES:
            for layout in base.LAYOUTS:
                if adc_group_complete(page, ratio, layout, tau):
                    completed.append(
                        {
                            "method": METHOD_ADC,
                            "page_bytes": page,
                            "ram_ratio": ratio,
                            "layout": layout,
                        }
                    )
                build = rabitq_builds.get((page, layout))
                path = rabitq_output(page, ratio, layout)
                if build and base.rabitq_complete(
                    path, page, ratio, layout, int(build["upper_graph_bytes"])
                ):
                    completed.append(
                        {
                            "method": METHOD_RABITQ,
                            "page_bytes": page,
                            "ram_ratio": ratio,
                            "layout": layout,
                            "path": str(path),
                        }
                    )
    value = {
        "status": "complete" if len(completed) == total else "running",
        "completed_search_groups": len(completed),
        "total_search_groups": total,
        "completed": completed,
    }
    write_json(RESULT_ROOT / "progress.json", value)
    return value


def adc_row(
    page: int,
    ratio: float,
    layout: str,
    source: Path,
    row: dict[str, str],
    build: dict[str, object],
    tau: float,
) -> dict[str, object]:
    nq = int(row["nq"])
    graph_read_bytes = float(row["index_page_read_bytes_mean"])
    graph_reads_4k = graph_read_bytes / 4096 / nq
    fp_reads_4k = float(row["direct_submitted_bytes_mean"]) / 4096 / nq
    return {
        "dataset": base.DATASET.key,
        "corpus": base.DATASET.corpus,
        "model": base.DATASET.model,
        "method": METHOD_ADC,
        "layout": layout,
        "graph_io_engine": "async-io_uring-hybrid",
        "page_bytes": page,
        "page_kib": page // 1024,
        "bq_ram_ratio_target": ratio,
        "bq_ram_budget_mib": base.budget_mib(ratio),
        "page_cache_mib": int(row["index_page_cache_mib"]),
        "upper_graph_bytes_pinned_ram": 0,
        "charged_graph_code_ram_bytes": int(row["index_page_cache_mib"]) * base.MIB,
        "graph_code_data_bytes": int(build["paged_index_data_bytes"]),
        "threads": base.THREADS,
        "rounds": base.ROUNDS,
        "efSearch": int(row["efSearch"]),
        "calibration_queries": CALIBRATION_QUERIES,
        "test_queries": nq,
        "calibration_seed": CALIBRATION_SEED,
        "calibration_reference_ef": CALIBRATION_REFERENCE_EF,
        "calibration_tau": tau,
        "recall_at_10": float(row["recall_at_k"]),
        "qps": float(row["qps"]),
        "latency_ms_batch_average": float(row["total_seconds_mean"]) * 1000 / nq,
        "graph_page_hit_ratio": float(row["index_page_hit_ratio"]),
        "graph_page_accesses_per_query": float(row["index_page_accesses_mean"]) / nq,
        "graph_physical_pages_per_query": graph_read_bytes / page / nq,
        "graph_physical_4k_reads_per_query": graph_reads_4k,
        "fp_vectors_mean": float(row["adaptive_rerank_count_mean"]),
        "fp_vectors_min": float(row["adaptive_rerank_count_min"]),
        "fp_vectors_max": float(row["adaptive_rerank_count_max"]),
        "fp_vectors_p95": float(row["adaptive_rerank_count_p95"]),
        "fp_vectors_p99": float(row["adaptive_rerank_count_p99"]),
        "fp_physical_4k_reads_per_query": fp_reads_4k,
        "total_physical_4k_reads_per_query": graph_reads_4k + fp_reads_4k,
        "source_csv": str(source),
    }


def rabitq_row(
    page: int,
    ratio: float,
    layout: str,
    source: Path,
    row: dict[str, str],
    build: dict[str, str],
) -> dict[str, object]:
    nq = int(row["nq"])
    graph_read_bytes = float(row["index_page_read_bytes_mean"])
    graph_reads_4k = graph_read_bytes / 4096 / nq
    fp_reads_4k = float(row["fp_physical_4k_reads_per_query"])
    upper = int(build["upper_graph_bytes"])
    cache_mib = int(row["index_page_cache_mib"])
    return {
        "dataset": base.DATASET.key,
        "corpus": base.DATASET.corpus,
        "model": base.DATASET.model,
        "method": METHOD_RABITQ,
        "layout": layout,
        "graph_io_engine": "sync-pread",
        "page_bytes": page,
        "page_kib": page // 1024,
        "bq_ram_ratio_target": ratio,
        "bq_ram_budget_mib": base.budget_mib(ratio),
        "page_cache_mib": cache_mib,
        "upper_graph_bytes_pinned_ram": upper,
        "charged_graph_code_ram_bytes": cache_mib * base.MIB + upper,
        "graph_code_data_bytes": int(build["paged_index_data_pages"]) * page,
        "threads": base.THREADS,
        "rounds": base.ROUNDS,
        "efSearch": int(row["efSearch"]),
        "calibration_queries": CALIBRATION_QUERIES,
        "test_queries": nq,
        "calibration_seed": CALIBRATION_SEED,
        "calibration_reference_ef": CALIBRATION_REFERENCE_EF,
        "calibration_tau": "",
        "recall_at_10": float(row["recall_at_k"]),
        "qps": float(row["qps"]),
        "latency_ms_batch_average": float(row["total_seconds_mean"]) * 1000 / nq,
        "graph_page_hit_ratio": float(row["index_page_hit_ratio"]),
        "graph_page_accesses_per_query": float(row["index_page_accesses_mean"]) / nq,
        "graph_physical_pages_per_query": graph_read_bytes / page / nq,
        "graph_physical_4k_reads_per_query": graph_reads_4k,
        "fp_vectors_mean": float(row["fp_reads_mean"]),
        "fp_vectors_min": float(row["fp_reads_min"]),
        "fp_vectors_max": float(row["fp_reads_max"]),
        "fp_vectors_p95": float(row["fp_reads_p95"]),
        "fp_vectors_p99": float(row["fp_reads_p99"]),
        "fp_physical_4k_reads_per_query": fp_reads_4k,
        "total_physical_4k_reads_per_query": graph_reads_4k + fp_reads_4k,
        "source_csv": str(source),
    }


def combine(
    adc_builds: dict[tuple[int, str], dict[str, object]],
    rabitq_builds: dict[tuple[int, str], dict[str, str]],
    tau: float,
) -> Path:
    rows: list[dict[str, object]] = []
    for ratio in base.RATIOS:
        for page in base.PAGES:
            for layout in base.LAYOUTS:
                for ef in base.EFS:
                    source = adc_output(page, ratio, layout, ef)
                    selected = read_csv(source)
                    if len(selected) != 1:
                        raise RuntimeError(f"expected one ADC-BQ row: {source}")
                    rows.append(
                        adc_row(
                            page,
                            ratio,
                            layout,
                            source,
                            selected[0],
                            adc_builds[(page, layout)],
                            tau,
                        )
                    )
                source = rabitq_output(page, ratio, layout)
                for row in read_csv(source):
                    rows.append(
                        rabitq_row(
                            page,
                            ratio,
                            layout,
                            source,
                            row,
                            rabitq_builds[(page, layout)],
                        )
                    )

    invariant_fields = (
        "recall_at_10",
        "graph_page_accesses_per_query",
        "fp_vectors_mean",
        "fp_vectors_min",
        "fp_vectors_max",
        "fp_vectors_p95",
        "fp_vectors_p99",
        "fp_physical_4k_reads_per_query",
    )
    # Colocation remaps node ids and can legitimately change tie ordering by a
    # tiny amount.  Page size and cache budget must preserve search semantics
    # within a fixed layout, so validate those dimensions independently for
    # baseline and coloc.
    references: dict[tuple[str, str, int], dict[str, object]] = {}
    validation: list[dict[str, object]] = []
    for row in rows:
        key = (
            str(row["method"]),
            str(row["layout"]),
            int(row["efSearch"]),
        )
        reference = references.setdefault(key, row)
        equal = all(
            math.isclose(
                float(row[field]),
                float(reference[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for field in invariant_fields
        )
        validation.append(
            {
                "method": row["method"],
                "page_kib": row["page_kib"],
                "bq_ram_ratio_target": row["bq_ram_ratio_target"],
                "layout": row["layout"],
                "efSearch": row["efSearch"],
                "semantic_invariant_scope": "method+layout+efSearch",
                "semantic_invariants_equal": equal,
            }
        )
        if not equal:
            raise RuntimeError(
                "page/cache changed search semantics within a fixed layout: "
                f"{row['method']} page={row['page_kib']} "
                f"ratio={row['bq_ram_ratio_target']} layout={row['layout']} "
                f"ef={row['efSearch']}"
            )
    output = RESULT_ROOT / "all_search_results.csv"
    write_csv(output, rows)
    write_csv(RESULT_ROOT / "validation.csv", validation)
    write_csv(
        RESULT_ROOT / "page4k_memory_sweep_adcbq_calibrated_rabitq.csv",
        [row for row in rows if int(row["page_bytes"]) == 4096],
    )
    return output


def plot(combined: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(combined)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {METHOD_ADC: "#0072B2", METHOD_RABITQ: "#D55E00"}
    markers = {"baseline": "s", "coloc": "o"}
    linestyles = {"baseline": "--", "coloc": "-"}

    for ef in base.EFS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for column, page in enumerate(base.PAGES):
            for method in colors:
                for layout in base.LAYOUTS:
                    group = sorted(
                        (
                            row
                            for row in rows
                            if row["method"] == method
                            and row["layout"] == layout
                            and int(row["page_bytes"]) == page
                            and int(row["efSearch"]) == ef
                        ),
                        key=lambda row: float(row["bq_ram_ratio_target"]),
                    )
                    x = [float(row["bq_ram_ratio_target"]) * 100 for row in group]
                    label = f"{method}, {layout}"
                    axes[0][column].plot(
                        x,
                        [float(row["qps"]) for row in group],
                        color=colors[method],
                        marker=markers[layout],
                        linestyle=linestyles[layout],
                        label=label,
                    )
                    axes[1][column].plot(
                        x,
                        [float(row["total_physical_4k_reads_per_query"]) for row in group],
                        color=colors[method],
                        marker=markers[layout],
                        linestyle=linestyles[layout],
                        label=label,
                    )
            axes[0][column].set_title(f"{page // 1024}KB pages")
            axes[0][column].set_ylabel("QPS")
            axes[1][column].set_ylabel("Total physical 4KB reads/query")
            for axis in (axes[0][column], axes[1][column]):
                axis.set_xlabel("BQ-based graph/code RAM budget (%)")
                axis.grid(True, alpha=0.25)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(
            "MSMARCO OpenAI-1536 held-out test, calibrated ADC-BQ, "
            f"T=24, efSearch={ef}",
            y=0.99,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        stem = f"memory_page_sweep_ef{ef}_t24"
        fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=190, bbox_inches="tight")
        fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), squeeze=False)
        metrics = (
            ("qps", "QPS"),
            ("graph_page_hit_ratio", "Graph page-cache hit ratio"),
            ("graph_physical_4k_reads_per_query", "Graph physical 4KB reads/query"),
            ("total_physical_4k_reads_per_query", "Total physical 4KB reads/query"),
        )
        for method in colors:
            for layout in base.LAYOUTS:
                group = sorted(
                    (
                        row
                        for row in rows
                        if row["method"] == method
                        and row["layout"] == layout
                        and int(row["page_bytes"]) == 4096
                        and int(row["efSearch"]) == ef
                    ),
                    key=lambda row: float(row["bq_ram_ratio_target"]),
                )
                x = [float(row["bq_ram_ratio_target"]) * 100 for row in group]
                for column, (field, ylabel) in enumerate(metrics):
                    axes[0][column].plot(
                        x,
                        [float(row[field]) for row in group],
                        color=colors[method],
                        marker=markers[layout],
                        linestyle=linestyles[layout],
                        label=f"{method}, {layout}",
                    )
                    axes[0][column].set_ylabel(ylabel)
                    axes[0][column].set_xlabel("BQ-based RAM budget (%)")
                    axes[0][column].grid(True, alpha=0.25)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(
            "MSMARCO OpenAI-1536 held-out test, 4KB pages, calibrated ADC-BQ, "
            f"T=24, efSearch={ef}",
            y=1.01,
        )
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=4,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.83))
        stem = f"page4k_memory_sweep_adcbq_calibrated_vs_rabitq_ef{ef}_t24"
        fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=190, bbox_inches="tight")
        fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    tau, calibration_payload = load_tau()
    split = calibration.make_split(
        base.DATASET,
        CALIBRATION_QUERIES,
        CALIBRATION_SEED,
        0,
    )
    dataset = replace(base.DATASET, query=split.test_query, gt=split.test_gt)
    configure_base(dataset)
    base.datasets_base.validate(base.DATASET)
    ensure_adc_binaries()
    if not base.RABITQ_BASE_INDEX.exists():
        raise FileNotFoundError(base.RABITQ_BASE_INDEX)
    for binary in base.RABITQ_BINARIES.values():
        if not binary.exists():
            raise FileNotFoundError(binary)

    adc_builds = {
        (page, layout): base.adc_build(page, layout)
        for page in base.PAGES
        for layout in base.LAYOUTS
    }
    sidecar = FP_ROOT / base.DATASET.key / f"{base.DATASET.key}.fp32.aligned4k.fbin"
    sidecar_existed = sidecar.exists()
    base.adc.prepare_aligned_fp(sidecar, base.DATASET)
    rabitq_builds = {
        (page, layout): base.prepare_rabitq_layout(page, layout, sidecar)
        for page in base.PAGES
        for layout in base.LAYOUTS
    }
    progress(rabitq_builds, tau)

    try:
        for ratio in base.RUN_RATIO_ORDER:
            for page in base.PAGES:
                for layout in base.LAYOUTS:
                    for ef in base.EFS:
                        run_adc(page, ratio, layout, ef, tau, sidecar)
                    progress(rabitq_builds, tau)
            for page in base.PAGES:
                for layout in base.LAYOUTS:
                    base.run_rabitq(
                        page,
                        ratio,
                        layout,
                        rabitq_builds[(page, layout)],
                        sidecar,
                    )
                    progress(rabitq_builds, tau)

        combined = combine(adc_builds, rabitq_builds, tau)
        plot(combined)
        write_json(
            RESULT_ROOT / "run_manifest.json",
            {
                "dataset": base.DATASET.key,
                "model": base.DATASET.model,
                "query_split": {
                    "seed": CALIBRATION_SEED,
                    "calibration_queries": split.calibration_rows,
                    "test_queries": split.test_rows,
                    "test_query_path": str(split.test_query),
                    "test_gt_path": str(split.test_gt),
                },
                "methods": {
                    METHOD_ADC: (
                        "single tau fitted at efSearch=256; graph-page async "
                        "io_uring QD48; calibrated FP rows via O_DIRECT io_uring"
                    ),
                    METHOD_RABITQ: (
                        "1-bit c=16; synchronous graph-page and adaptive "
                        "LB-selected FP O_DIRECT reads"
                    ),
                },
                "calibration": calibration_payload,
                "page_bytes": list(base.PAGES),
                "bq_ram_ratio_targets": list(base.RATIOS),
                "canonical_bq_graph_code_bytes": base.canonical_bq_bytes(),
                "bq_ram_budgets_mib": {
                    str(ratio): base.budget_mib(ratio) for ratio in base.RATIOS
                },
                "budget_policy": (
                    "same absolute BQ-derived RAM cap; RaBitQ pinned upper "
                    "graph charged by ceil(MiB) before assigning page cache"
                ),
                "layouts": list(base.LAYOUTS),
                "threads": base.THREADS,
                "rounds": base.ROUNDS,
                "efSearch": list(base.EFS),
                "M": base.M,
                "efConstruction": base.EF_CONSTRUCTION,
                "index_rebuild": False,
                "paged_layout_materialization": "existing ADC-BQ and RaBitQ layouts reused",
            },
        )
        final_progress = progress(rabitq_builds, tau)
        if final_progress["status"] != "complete":
            raise RuntimeError("progress accounting did not reach completion")
    finally:
        if not sidecar_existed and sidecar.exists():
            size = sidecar.stat().st_size
            sidecar.unlink()
            print(f"REMOVED transient FP sidecar bytes={size}: {sidecar}", flush=True)


if __name__ == "__main__":
    main()
