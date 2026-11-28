#!/usr/bin/env python3
"""MSMARCO OpenAI-1536 page-size and BQ-RAM sweep.

ADC-BQ uses graph-page io_uring.  RaBitQ uses synchronous graph-page and
adaptive FP reads.  Both methods receive the same absolute graph/code RAM
budget derived from the canonical 4KB ADC-BQ paged data size.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import run_nq_quora_adcbq_colocation_4k as adc
import run_v051_ssd_odirect_multidataset as datasets_base


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/msmarco_adcbq_rabitq_memory_page_sweep"
FIGURE_ROOT = REPO / "experiment/figures/msmarco_adcbq_rabitq_memory_page_sweep"
ARTIFACT_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_adcbq_rabitq_memory_page_sweep"
)
DATASET = datasets_base.DATASETS["sieun_msmarco_openai1536"]
PAGES = (4096, 8192, 16384)
RATIOS = (0.10, 0.25, 0.50, 0.75)
# Produce the requested fixed-25% page sweep first, then fill the RAM sweep.
RUN_RATIO_ORDER = (0.25, 0.10, 0.50, 0.75)
LAYOUTS = ("baseline", "coloc")
EFS = (64, 128, 256, 512, 1024)
THREADS = 24
ROUNDS = 3
ADC_RERANK = 80
RABITQ_C = 16
ASYNC_QD = 48
M = 24
EF_CONSTRUCTION = 200
COLOCATION_ITERATIONS = 20
COLOCATION_PARTITION_SIZE = 64
MIB = 1024 * 1024

ADC_BINARIES = {
    page: REPO
    / f"experiment/adcbq_optimized/faiss_adcbq_hnsw_{page // 1024}k_graph_async"
    for page in PAGES
}
RABITQ_BINARIES = {
    page: REPO / f"experiment/rabitq_paged_adaptive_{page // 1024}k_benchmark"
    for page in PAGES
}
ADC_ARTIFACT_ROOTS = {
    page: Path(
        f"/home/smrc/samsung-nvme/jihyo/msmarco_adcbq_colocation_{page // 1024}k"
    )
    for page in PAGES
}
ADC_INDEX_RESULT_ROOTS = {
    page: REPO / f"experiment/msmarco_adcbq_colocation_{page // 1024}k"
    for page in PAGES
}
RABITQ_BASE_INDEX = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_rabitq_paged25_t24/indexes/"
    "sieun_msmarco_openai1536.rabitq_b1_c16.n1000000.d1536.M24.efC200.index"
)
LEGACY_ADC_ASYNC_4K_P025 = (
    REPO / "experiment/msmarco_adcbq_colocation_4k_graph_async/sieun_msmarco_openai1536"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def ratio_label(ratio: float) -> str:
    return f"p{int(round(ratio * 100)):03d}"


def adc_build(page: int, layout: str) -> dict[str, object]:
    path = ADC_INDEX_RESULT_ROOTS[page] / DATASET.key / f"build_{layout}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    paged = Path(str(manifest["paged_index_path"]))
    if (
        int(manifest["page_bytes"]) != page
        or str(manifest["layout"]) != layout
        or not paged.exists()
        or paged.stat().st_size != int(manifest["paged_index_bytes"])
    ):
        raise RuntimeError(f"invalid reusable ADC-BQ layout: {path}")
    return manifest


def canonical_bq_bytes() -> int:
    baseline = adc_build(4096, "baseline")
    return int(baseline["paged_index_data_bytes"])


def budget_mib(ratio: float) -> int:
    return max(1, math.ceil(canonical_bq_bytes() * ratio / MIB))


def shared_fp_path() -> Path:
    return ARTIFACT_ROOT / DATASET.key / f"{DATASET.key}.fp32.aligned4k.fbin"


def adc_output(page: int, ratio: float, layout: str) -> Path:
    if page == 4096 and math.isclose(ratio, 0.25):
        return LEGACY_ADC_ASYNC_4K_P025 / (
            f"search_{layout}_p025_t24_async_qd{ASYNC_QD}.csv"
        )
    return RESULT_ROOT / DATASET.key / "adcbq" / f"{page // 1024}k" / (
        f"search_{layout}_{ratio_label(ratio)}_t24_async_qd{ASYNC_QD}.csv"
    )


def rabitq_output(page: int, ratio: float, layout: str) -> Path:
    return RESULT_ROOT / DATASET.key / "rabitq" / f"{page // 1024}k" / (
        f"search_{layout}_{ratio_label(ratio)}_t24_sync.csv"
    )


def adc_complete(path: Path, page: int, ratio: float, layout: str) -> bool:
    try:
        rows = read_csv(path)
        nq, _ = datasets_base.matrix_shape(DATASET.query)
        return (
            [int(row["efSearch"]) for row in rows] == list(EFS)
            and all(int(row["rounds"]) == ROUNDS for row in rows)
            and all(int(row["search_threads"]) == THREADS for row in rows)
            and all(int(row["nq"]) == nq for row in rows)
            and all(int(row["rerank_n"]) == ADC_RERANK for row in rows)
            and all(int(row["index_page_bytes"]) == page for row in rows)
            and all(int(row["index_page_cache_mib"]) == budget_mib(ratio) for row in rows)
            and all(row["index_page_async"] == "true" for row in rows)
            and all(int(row["index_page_qd"]) == ASYNC_QD for row in rows)
            and all(row["paged_index_status"] == "loaded" for row in rows)
            and all(row["fp_rerank_source"] == "odirect-uring" for row in rows)
            and all(int(float(row["direct_rows_mean"])) == nq * ADC_RERANK for row in rows)
            and all(float(row["index_async_requests_mean"]) > 0 for row in rows)
            and all(
                int(row["colocation_iterations"])
                == (COLOCATION_ITERATIONS if layout == "coloc" else 0)
                for row in rows
            )
        )
    except (OSError, KeyError, ValueError):
        return False


def run_adc(page: int, ratio: float, layout: str, sidecar: Path) -> Path:
    output = adc_output(page, ratio, layout)
    if adc_complete(output, page, ratio, layout):
        print(f"SKIP complete ADC-BQ: {output}", flush=True)
        return output
    adc.PAGE_BYTES = page
    command = adc.common_command(
        DATASET,
        layout,
        ADC_ARTIFACT_ROOTS[page],
        output,
        budget_mib(ratio),
        THREADS,
        EFS,
        ROUNDS,
        0,
    )
    command[0] = str(ADC_BINARIES[page])
    command.extend(
        [
            "--index-page-async",
            "--index-page-qd", str(ASYNC_QD),
            "--rerank-dot", "fp32",
            "--fp-rerank-source", "odirect-uring",
            "--fp-rerank-norms", "onfetch",
            "--fp-rerank-path", str(sidecar),
            "--fp-rerank-qd", str(ADC_RERANK),
        ]
    )
    adc.run_command(command, output.with_suffix(".log"), THREADS)
    if not adc_complete(output, page, ratio, layout):
        raise RuntimeError(f"invalid ADC-BQ result: {output}")
    print(f"DONE ADC-BQ page={page} ratio={ratio} layout={layout}", flush=True)
    return output


def rabitq_paged_path(page: int, layout: str) -> Path:
    return ARTIFACT_ROOT / DATASET.key / (
        f"{DATASET.key}.rabitq_b1.level0.{layout}.paged{page // 1024}k"
    )


def rabitq_build_csv(page: int, layout: str) -> Path:
    return RESULT_ROOT / DATASET.key / "rabitq" / f"{page // 1024}k" / (
        f"paged_build_{layout}.csv"
    )


def rabitq_common_command(
    binary: Path,
    page: int,
    paged: Path,
    output: Path,
    layout: str,
    fp_path: Path | None = None,
) -> list[str]:
    command = [
        str(binary),
        "--dataset", DATASET.key,
        "--train", str(DATASET.train),
        "--query", str(DATASET.query),
        "--gt", str(DATASET.gt),
        "--index", str(RABITQ_BASE_INDEX),
        "--paged-index", str(paged),
        "--fp-path", str(fp_path if fp_path is not None else shared_fp_path()),
        "--out", str(output),
        "--threads", str(THREADS),
        "--rounds", str(ROUNDS),
        "--k", "10",
        "--efs", *map(str, EFS),
    ]
    if layout == "coloc":
        command.extend(
            [
                "--colocation-iterations", str(COLOCATION_ITERATIONS),
                "--colocation-partition-size", str(COLOCATION_PARTITION_SIZE),
            ]
        )
    return command


def rabitq_build_complete(page: int, layout: str) -> bool:
    output = rabitq_build_csv(page, layout)
    paged = rabitq_paged_path(page, layout)
    try:
        rows = read_csv(output)
        if len(rows) != 1 or not paged.exists():
            return False
        row = rows[0]
        return (
            int(row["page_bytes"]) == page
            and row["colocated"] == ("true" if layout == "coloc" else "false")
            and int(row["colocation_iterations"])
            == (COLOCATION_ITERATIONS if layout == "coloc" else 0)
            and int(row["paged_index_file_bytes"]) == paged.stat().st_size
            and int(row["upper_graph_bytes"]) > 0
        )
    except (OSError, KeyError, ValueError):
        return False


def prepare_rabitq_layout(
    page: int, layout: str, fp_path: Path | None = None
) -> dict[str, str]:
    output = rabitq_build_csv(page, layout)
    paged = rabitq_paged_path(page, layout)
    if not rabitq_build_complete(page, layout):
        paged.parent.mkdir(parents=True, exist_ok=True)
        command = rabitq_common_command(
            RABITQ_BINARIES[page], page, paged, output, layout, fp_path
        )
        command.extend(["--cache-mib", "1", "--prepare-only"])
        adc.run_command(command, output.with_suffix(".log"), THREADS)
    if not rabitq_build_complete(page, layout):
        raise RuntimeError(f"invalid RaBitQ paged layout: {output}")
    return read_csv(output)[0]


def rabitq_cache_mib(ratio: float, upper_graph_bytes: int) -> int:
    # Charge the always-resident upper graph against the shared ADC-BQ budget.
    upper_charge_mib = math.ceil(upper_graph_bytes / MIB)
    return max(1, budget_mib(ratio) - upper_charge_mib)


def rabitq_complete(
    path: Path,
    page: int,
    ratio: float,
    layout: str,
    upper_graph_bytes: int,
) -> bool:
    try:
        rows = read_csv(path)
        cache_mib = rabitq_cache_mib(ratio, upper_graph_bytes)
        nq, _ = datasets_base.matrix_shape(DATASET.query)
        return (
            [int(row["efSearch"]) for row in rows] == list(EFS)
            and all(int(row["rounds"]) == ROUNDS for row in rows)
            and all(int(row["search_threads"]) == THREADS for row in rows)
            and all(int(row["nq"]) == nq for row in rows)
            and all(int(row["num_clusters"]) == RABITQ_C for row in rows)
            and all(int(row["index_page_bytes"]) == page for row in rows)
            and all(int(row["index_page_cache_mib"]) == cache_mib for row in rows)
            and all(int(row["upper_graph_bytes"]) == upper_graph_bytes for row in rows)
            and all(row["colocated"] == ("true" if layout == "coloc" else "false") for row in rows)
            and all("adaptive-FP" in row["method"] for row in rows)
            and all(row.get("index_page_async", "false") == "false" for row in rows)
        )
    except (OSError, KeyError, ValueError):
        return False


def run_rabitq(
    page: int,
    ratio: float,
    layout: str,
    build: dict[str, str],
    fp_path: Path | None = None,
) -> Path:
    output = rabitq_output(page, ratio, layout)
    upper_graph_bytes = int(build["upper_graph_bytes"])
    if rabitq_complete(output, page, ratio, layout, upper_graph_bytes):
        print(f"SKIP complete RaBitQ: {output}", flush=True)
        return output
    command = rabitq_common_command(
        RABITQ_BINARIES[page],
        page,
        rabitq_paged_path(page, layout),
        output,
        layout,
        fp_path,
    )
    command.extend(["--cache-mib", str(rabitq_cache_mib(ratio, upper_graph_bytes))])
    adc.run_command(command, output.with_suffix(".log"), THREADS)
    if not rabitq_complete(output, page, ratio, layout, upper_graph_bytes):
        raise RuntimeError(f"invalid RaBitQ result: {output}")
    print(f"DONE RaBitQ page={page} ratio={ratio} layout={layout}", flush=True)
    return output


def progress(builds: dict[tuple[int, str], dict[str, str]]) -> dict[str, object]:
    total_jobs = len(RATIOS) * len(PAGES) * len(LAYOUTS) * 2
    completed: list[dict[str, object]] = []
    for ratio in RATIOS:
        for page in PAGES:
            for layout in LAYOUTS:
                apath = adc_output(page, ratio, layout)
                if adc_complete(apath, page, ratio, layout):
                    completed.append(
                        {"method": "ADC-BQ async", "page_bytes": page, "ram_ratio": ratio, "layout": layout, "path": str(apath)}
                    )
                build = builds.get((page, layout))
                rpath = rabitq_output(page, ratio, layout)
                if build and rabitq_complete(
                    rpath, page, ratio, layout, int(build["upper_graph_bytes"])
                ):
                    completed.append(
                        {"method": "RaBitQ sync", "page_bytes": page, "ram_ratio": ratio, "layout": layout, "path": str(rpath)}
                    )
    value = {
        "status": "complete" if len(completed) == total_jobs else "running",
        "completed_search_jobs": len(completed),
        "total_search_jobs": total_jobs,
        "completed": completed,
    }
    write_json(RESULT_ROOT / "progress.json", value)
    return value


def common_row(
    method: str,
    page: int,
    ratio: float,
    layout: str,
    source: Path,
    row: dict[str, str],
    build: dict[str, object],
    upper_graph_bytes: int,
    graph_async: bool,
) -> dict[str, object]:
    nq = int(row["nq"])
    graph_read_bytes = float(row["index_page_read_bytes_mean"])
    if method == "ADC-BQ":
        fp_mean = float(row["direct_rows_mean"]) / nq
        fp_min = fp_max = fp_p95 = fp_p99 = fp_mean
        fp_reads_4k = float(row["direct_submitted_bytes_mean"]) / 4096 / nq
    else:
        fp_mean = float(row["fp_reads_mean"])
        fp_min = float(row["fp_reads_min"])
        fp_max = float(row["fp_reads_max"])
        fp_p95 = float(row["fp_reads_p95"])
        fp_p99 = float(row["fp_reads_p99"])
        fp_reads_4k = float(row["fp_physical_4k_reads_per_query"])
    cache_mib = int(row["index_page_cache_mib"])
    graph_reads_4k = graph_read_bytes / 4096 / nq
    return {
        "dataset": DATASET.key,
        "corpus": DATASET.corpus,
        "model": DATASET.model,
        "method": method,
        "layout": layout,
        "graph_io_engine": "async-io_uring-hybrid" if graph_async else "sync-pread",
        "page_bytes": page,
        "page_kib": page // 1024,
        "bq_ram_ratio_target": ratio,
        "bq_ram_budget_mib": budget_mib(ratio),
        "page_cache_mib": cache_mib,
        "upper_graph_bytes_pinned_ram": upper_graph_bytes,
        "charged_graph_code_ram_bytes": cache_mib * MIB + upper_graph_bytes,
        "graph_code_data_bytes": int(build["paged_index_data_bytes"]),
        "threads": THREADS,
        "rounds": ROUNDS,
        "efSearch": int(row["efSearch"]),
        "recall_at_10": float(row["recall_at_k"]),
        "qps": float(row["qps"]),
        "latency_ms_batch_average": float(row["total_seconds_mean"]) * 1000 / nq,
        "graph_page_hit_ratio": float(row["index_page_hit_ratio"]),
        "graph_page_accesses_per_query": float(row["index_page_accesses_mean"]) / nq,
        "graph_physical_pages_per_query": graph_read_bytes / page / nq,
        "graph_physical_4k_reads_per_query": graph_reads_4k,
        "fp_vectors_mean": fp_mean,
        "fp_vectors_min": fp_min,
        "fp_vectors_max": fp_max,
        "fp_vectors_p95": fp_p95,
        "fp_vectors_p99": fp_p99,
        "fp_physical_4k_reads_per_query": fp_reads_4k,
        "total_physical_4k_reads_per_query": graph_reads_4k + fp_reads_4k,
        "source_csv": str(source),
    }


def combine(
    adc_builds: dict[tuple[int, str], dict[str, object]],
    rabitq_builds: dict[tuple[int, str], dict[str, str]],
) -> Path:
    rows: list[dict[str, object]] = []
    for ratio in RATIOS:
        for page in PAGES:
            for layout in LAYOUTS:
                asource = adc_output(page, ratio, layout)
                for row in read_csv(asource):
                    rows.append(
                        common_row(
                            "ADC-BQ", page, ratio, layout, asource, row,
                            adc_builds[(page, layout)], 0, True,
                        )
                    )
                rsource = rabitq_output(page, ratio, layout)
                rbuild = rabitq_builds[(page, layout)]
                rbuild_common: dict[str, object] = {
                    "paged_index_data_bytes": int(rbuild["paged_index_data_pages"]) * page
                }
                for row in read_csv(rsource):
                    rows.append(
                        common_row(
                            "RaBitQ-1bit-c16-adaptive-FP", page, ratio, layout,
                            rsource, row, rbuild_common,
                            int(rbuild["upper_graph_bytes"]), False,
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
    reference: dict[tuple[str, int], dict[str, object]] = {}
    validation: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["method"]), int(row["efSearch"]))
        if key not in reference:
            reference[key] = row
        equal = all(
            math.isclose(
                float(row[field]), float(reference[key][field]), rel_tol=0.0, abs_tol=1e-9
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
                "semantic_invariants_equal": equal,
            }
        )
        if not equal:
            raise RuntimeError(
                "page/cache/layout changed search semantics: "
                f"{row['method']} page={row['page_kib']} ratio={row['bq_ram_ratio_target']} "
                f"layout={row['layout']} ef={row['efSearch']}"
            )
    output = RESULT_ROOT / "all_search_results.csv"
    write_csv(output, rows)
    write_csv(RESULT_ROOT / "validation.csv", validation)
    return output


def plot(combined: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(combined)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    colors = {"ADC-BQ": "#0072B2", "RaBitQ-1bit-c16-adaptive-FP": "#D55E00"}
    markers = {"baseline": "s", "coloc": "o"}
    linestyles = {"baseline": "--", "coloc": "-"}
    for ef in EFS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for column, page in enumerate(PAGES):
            for method in colors:
                for layout in LAYOUTS:
                    group = sorted(
                        (
                            row for row in rows
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
                        x, [float(row["qps"]) for row in group],
                        color=colors[method], marker=markers[layout],
                        linestyle=linestyles[layout], label=label,
                    )
                    axes[1][column].plot(
                        x, [float(row["total_physical_4k_reads_per_query"]) for row in group],
                        color=colors[method], marker=markers[layout],
                        linestyle=linestyles[layout], label=label,
                    )
            axes[0][column].set_title(f"{page // 1024}KB pages")
            axes[0][column].set_ylabel("QPS")
            axes[1][column].set_ylabel("Total physical 4KB reads/query")
            for axis in (axes[0][column], axes[1][column]):
                axis.set_xlabel("BQ-based graph/code RAM budget (%)")
                axis.grid(True, alpha=0.25)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.suptitle(f"MSMARCO OpenAI-1536, T=24, efSearch={ef}", y=0.99)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.89))
        stem = f"memory_page_sweep_ef{ef}_t24"
        fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=190, bbox_inches="tight")
        fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    datasets_base.validate(DATASET)
    if not RABITQ_BASE_INDEX.exists():
        raise FileNotFoundError(RABITQ_BASE_INDEX)
    for binary in (*ADC_BINARIES.values(), *RABITQ_BINARIES.values()):
        if not binary.exists():
            raise FileNotFoundError(binary)

    adc_builds = {
        (page, layout): adc_build(page, layout)
        for page in PAGES
        for layout in LAYOUTS
    }
    sidecar = shared_fp_path()
    adc.prepare_aligned_fp(sidecar, DATASET)
    rabitq_builds = {
        (page, layout): prepare_rabitq_layout(page, layout)
        for page in PAGES
        for layout in LAYOUTS
    }
    progress(rabitq_builds)

    try:
        for ratio in RUN_RATIO_ORDER:
            for page in PAGES:
                for layout in LAYOUTS:
                    run_adc(page, ratio, layout, sidecar)
                    progress(rabitq_builds)
            for page in PAGES:
                for layout in LAYOUTS:
                    run_rabitq(page, ratio, layout, rabitq_builds[(page, layout)])
                    progress(rabitq_builds)

        combined = combine(adc_builds, rabitq_builds)
        plot(combined)
        write_json(
            RESULT_ROOT / "run_manifest.json",
            {
                "dataset": DATASET.key,
                "model": DATASET.model,
                "methods": {
                    "ADC-BQ": "graph-page async io_uring QD48; fixed FP rerank@80 via O_DIRECT io_uring",
                    "RaBitQ": "1-bit c=16; synchronous graph-page and adaptive LB-selected FP O_DIRECT reads",
                },
                "page_bytes": list(PAGES),
                "bq_ram_ratio_targets": list(RATIOS),
                "canonical_bq_graph_code_bytes": canonical_bq_bytes(),
                "bq_ram_budgets_mib": {str(ratio): budget_mib(ratio) for ratio in RATIOS},
                "budget_policy": "same absolute BQ-derived RAM cap; RaBitQ pinned upper graph charged by ceil(MiB) before assigning its page cache",
                "layouts": list(LAYOUTS),
                "threads": THREADS,
                "rounds": ROUNDS,
                "efSearch": list(EFS),
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "index_rebuild": False,
                "paged_layout_materialization": "ADC-BQ reused; RaBitQ 8/16KB layouts generated from the existing c16 base index",
            },
        )
        final_progress = progress(rabitq_builds)
        if final_progress["status"] != "complete":
            raise RuntimeError("progress accounting did not reach completion")
    finally:
        if sidecar.exists():
            size = sidecar.stat().st_size
            sidecar.unlink()
            print(f"REMOVED transient FP sidecar bytes={size}: {sidecar}", flush=True)


if __name__ == "__main__":
    main()
