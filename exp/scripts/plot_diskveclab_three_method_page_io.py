#!/usr/bin/env python3
"""Compare actual submitted 4 KiB reads for three DiskVecLab methods."""

from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


try:
    import matplotlib
except ModuleNotFoundError:
    conda_python = Path("/home/jihyo/anaconda3/bin/python")
    if Path(sys.executable) != conda_python and conda_python.is_file():
        os.execv(str(conda_python), [str(conda_python), *sys.argv])
    raise

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiment/diskveclab_full_matrix"
OUT = ROOT / "page_io_3methods"

PANELS = (
    ("nq_bge_m3_vanilla", "NQ · BGE-M3 (1024D)"),
    ("nq_qwen06_1024", "NQ · Qwen-0.6B (1024D)"),
    ("nq_nemotron4096", "NQ · Nemotron (4096D)"),
    ("nq_e5mistral4096", "NQ · E5-Mistral (4096D)"),
    ("quora_bge_m3_vanilla", "Quora · BGE-M3 (1024D)"),
    ("quora_qwen1024", "Quora · Qwen-0.6B (1024D)"),
    ("quora_nemotron4096", "Quora · Nemotron (4096D)"),
    ("quora_e5mistral4096", "Quora · E5-Mistral (4096D)"),
    ("msmarco_openai1536", "MS MARCO · OpenAI (1536D)"),
    ("msmarco_bge1024", "MS MARCO · BGE (1024D)"),
    ("msmarco_qwen1024", "MS MARCO · Qwen-0.6B (1024D)"),
    ("msmarco_e5mistral4096", "MS MARCO · E5-Mistral (4096D)"),
    ("dbpedia_openai1536", "DBpedia · OpenAI (1536D)"),
    ("dbpedia_bge1024", "DBpedia · BGE (1024D)"),
    ("dbpedia_qwen1024", "DBpedia · Qwen-0.6B (1024D)"),
    ("dbpedia_e5mistral4096", "DBpedia · E5-Mistral (4096D)"),
)
DATASETS = {key for key, _ in PANELS}

METHODS = {
    "adcbq": {
        "label": "ADC-BQ · RAM graph + adaptive FP",
        "short": "ADC-BQ",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
        "component": "FP sidecar pages",
    },
    "diskann_classic": {
        "label": "DiskANN classic · PQ64 + FP nodes",
        "short": "DiskANN",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "-",
        "component": "SSD node-record pages",
    },
    "diskann_rabitq": {
        "label": "DiskANN RaBitQ-1 + adaptive FP",
        "short": "RaBitQ-1",
        "color": "#8C564B",
        "marker": "v",
        "linestyle": "--",
        "component": "SSD node-record pages",
    },
}
METHOD_ORDER = tuple(METHODS)

NORMALIZED_FIELDS = (
    "dataset", "corpus", "model", "dim", "test_queries",
    "method_key", "method", "search_parameter", "search_parameter_value",
    "recall_at_10", "qps", "io_requests_per_query_mean",
    "graph_or_node_4k_pages_per_query_mean",
    "fp_sidecar_4k_pages_per_query_mean", "total_4k_pages_per_query_mean",
    "pages_per_request", "io_component", "io_engine", "source_csv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def collect() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    adc_path = ROOT / "calibrated_adcbq_pool_top10_target99_per_ef/summary_t1.csv"
    for row in read_csv(adc_path):
        if row["dataset"] not in DATASETS:
            continue
        requests = float(row["fp_vectors_per_query_mean"])
        fp_pages = float(row["fp_4k_pages_per_query_mean"])
        total_pages = float(row["total_4k_pages_per_query_mean"])
        output.append(
            {
                "dataset": row["dataset"], "corpus": row["corpus"],
                "model": row["model"], "dim": int(row["dim"]),
                "test_queries": int(row["test_queries"]),
                "method_key": "adcbq", "method": METHODS["adcbq"]["label"],
                "search_parameter": "efSearch",
                "search_parameter_value": int(row["efSearch"]),
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "io_requests_per_query_mean": requests,
                "graph_or_node_4k_pages_per_query_mean": 0.0,
                "fp_sidecar_4k_pages_per_query_mean": fp_pages,
                "total_4k_pages_per_query_mean": total_pages,
                "pages_per_request": total_pages / requests,
                "io_component": METHODS["adcbq"]["component"],
                "io_engine": row["io_engine"], "source_csv": relative(adc_path),
            }
        )

    for family, method_key in (
        ("diskann_classic", "diskann_classic"),
        ("diskann_rabitq_1bit_adaptive_fp", "diskann_rabitq"),
    ):
        for path in sorted((ROOT / "diskann").glob(f"*/{family}/search_t1.csv")):
            for row in read_csv(path):
                requests = float(row["graph_io_requests_per_query_mean"])
                node_pages = float(row["graph_4k_pages_per_query_mean"])
                total_pages = float(row["total_4k_pages_per_query_mean"])
                output.append(
                    {
                        "dataset": row["dataset"], "corpus": row["corpus"],
                        "model": row["model"], "dim": int(row["dim"]),
                        "test_queries": int(row["nq"]),
                        "method_key": method_key,
                        "method": METHODS[method_key]["label"],
                        "search_parameter": "Lsearch",
                        "search_parameter_value": int(row["Lsearch"]),
                        "recall_at_10": float(row["recall_at_10"]),
                        "qps": float(row["qps"]),
                        "io_requests_per_query_mean": requests,
                        "graph_or_node_4k_pages_per_query_mean": node_pages,
                        "fp_sidecar_4k_pages_per_query_mean": float(
                            row["final_rerank_4k_pages_per_query_mean"]
                        ),
                        "total_4k_pages_per_query_mean": total_pages,
                        "pages_per_request": total_pages / requests,
                        "io_component": METHODS[method_key]["component"],
                        "io_engine": row["io_engine"], "source_csv": relative(path),
                    }
                )
    return output


def validate(rows: list[dict[str, object]]) -> None:
    expected = len(PANELS) * len(METHODS) * 20
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    counts = Counter((str(row["dataset"]), str(row["method_key"])) for row in rows)
    if any(count != 20 for count in counts.values()) or len(counts) != 48:
        raise RuntimeError(f"curve coverage mismatch: {counts}")
    split = {row["dataset"]: int(row["test_queries"]) for row in read_csv(ROOT / "split_manifest.csv")}
    for row in rows:
        if int(row["test_queries"]) != split[str(row["dataset"])]:
            raise RuntimeError(f"held-out split mismatch: {row}")
        if float(row["total_4k_pages_per_query_mean"]) <= 0:
            raise RuntimeError(f"non-positive page count: {row}")
        if int(row["dim"]) == 1024:
            expected_pages = 1.0 if row["method_key"] == "adcbq" else 2.0
        elif int(row["dim"]) == 1536:
            expected_pages = 2.0
        elif int(row["dim"]) == 4096:
            expected_pages = 4.0 if row["method_key"] == "adcbq" else 5.0
        else:
            raise RuntimeError(f"unexpected dimensionality: {row}")
        # The source summaries round request counts while retaining a more precise
        # page mean, so their ratio can differ from the integral layout slightly.
        if abs(float(row["pages_per_request"]) - expected_pages) > 1e-3:
            raise RuntimeError(f"page alignment/accounting mismatch: {row}")


def write_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def grouped_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_key"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["search_parameter_value"]))
    return grouped


def write_plot_ready_recall_pages_csv(
    rows: list[dict[str, object]], path: Path, *, min_recall: float
) -> None:
    """Write the exact ordered curve points used by the recall/page plot."""
    grouped = grouped_rows(rows)
    output: list[dict[str, object]] = []
    for panel_index, (dataset, title) in enumerate(PANELS):
        for curve_index, method_key in enumerate(METHOD_ORDER):
            points = [
                row for row in grouped[(dataset, method_key)]
                if float(row["recall_at_10"]) >= min_recall
            ]
            style = METHODS[method_key]
            for point_index, row in enumerate(points):
                output.append(
                    {
                        "panel_index": panel_index,
                        "panel_row": panel_index // 4,
                        "panel_column": panel_index % 4,
                        "panel_title": title,
                        "curve_index": curve_index,
                        "point_index": point_index,
                        "dataset": dataset,
                        "dim": row["dim"],
                        "method_key": method_key,
                        "method_label": style["label"],
                        "color": style["color"],
                        "marker": style["marker"],
                        "linestyle": style["linestyle"],
                        "search_parameter": row["search_parameter"],
                        "search_parameter_value": row["search_parameter_value"],
                        "recall_at_10": row["recall_at_10"],
                        "submitted_4k_pages_per_query": row[
                            "total_4k_pages_per_query_mean"
                        ],
                        "qps": row["qps"],
                        "io_requests_per_query": row[
                            "io_requests_per_query_mean"
                        ],
                        "pages_per_request": row["pages_per_request"],
                        "reference_recall": 0.99,
                        "source_csv": row["source_csv"],
                    }
                )
    fields = tuple(output[0])
    write_csv(path, output, fields)


def method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=METHODS[key]["color"], marker=METHODS[key]["marker"],
            linestyle=METHODS[key]["linestyle"], linewidth=1.7, markersize=5.5,
            label=METHODS[key]["label"],
        )
        for key in METHOD_ORDER
    ]


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 190} if extension == "png" else {}
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_recall_pages(rows: list[dict[str, object]], stem: Path, zoom90: bool) -> None:
    grouped = grouped_rows(rows)
    fig, axes = plt.subplots(4, 4, figsize=(21, 17.5))
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        for method_key in METHOD_ORDER:
            points = grouped[(dataset, method_key)]
            if zoom90:
                points = [row for row in points if float(row["recall_at_10"]) >= 0.90]
            if not points:
                continue
            style = METHODS[method_key]
            ax.plot(
                [float(row["recall_at_10"]) for row in points],
                [float(row["total_4k_pages_per_query_mean"]) for row in points],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.45, markersize=3.8,
            )
        ax.axvline(0.99, color="#2CA02C", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(title, fontsize=10.5)
        if zoom90:
            ax.set_xlim(0.90, 1.001)
        if index // 4 == 3:
            ax.set_xlabel("Held-out Recall@10")
        if index % 4 == 0:
            ax.set_ylabel("Submitted 4 KiB pages/query (log)")
        ax.tick_params(labelsize=8)
    handles = method_handles() + [
        Line2D([0], [0], color="#2CA02C", linestyle="--", label="Recall@10 = 0.99")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.962),
               ncol=4, frameon=False, fontsize=10)
    scope = "Recall@10 ≥ 0.90" if zoom90 else "full measured recall range"
    fig.suptitle(f"Actual SSD page reads vs Recall — 3 methods, T=1 ({scope})",
                 fontsize=16, y=0.994)
    fig.text(
        0.5, 0.012,
        "Pages are actual O_DIRECT submitted bytes / 4096. ADC-BQ: RAM graph + FP sidecar; "
        "DiskANN variants: SSD node records containing routing metadata and FP32 vectors.",
        ha="center", fontsize=9.2,
    )
    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.06, top=0.90,
                        wspace=0.20, hspace=0.28)
    save_figure(fig, stem)


def plot_qps_pages(rows: list[dict[str, object]], stem: Path) -> None:
    grouped = grouped_rows(rows)
    fig, axes = plt.subplots(4, 4, figsize=(21, 17.5))
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        for method_key in METHOD_ORDER:
            points = grouped[(dataset, method_key)]
            style = METHODS[method_key]
            ax.plot(
                [float(row["total_4k_pages_per_query_mean"]) for row in points],
                [float(row["qps"]) for row in points],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.45, markersize=3.8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(title, fontsize=10.5)
        if index // 4 == 3:
            ax.set_xlabel("Submitted 4 KiB pages/query (log)")
        if index % 4 == 0:
            ax.set_ylabel("QPS (log)")
        ax.tick_params(labelsize=8)
    fig.legend(handles=method_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.962),
               ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Throughput vs actual SSD page reads — 3 methods, 16 datasets, T=1",
                 fontsize=16, y=0.994)
    fig.text(0.5, 0.012, "Lower page count and higher QPS are better.", ha="center", fontsize=9.2)
    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.06, top=0.90,
                        wspace=0.20, hspace=0.28)
    save_figure(fig, stem)


def select_common_target(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = grouped_rows(rows)
    selected: list[dict[str, object]] = []
    for dataset, _ in PANELS:
        maxima = {
            method: max(float(row["recall_at_10"]) for row in grouped[(dataset, method)])
            for method in METHOD_ORDER
        }
        target = min(maxima.values())
        for method in METHOD_ORDER:
            feasible = [
                row for row in grouped[(dataset, method)]
                if float(row["recall_at_10"]) + 1e-12 >= target
            ]
            best = min(
                feasible,
                key=lambda row: (
                    float(row["total_4k_pages_per_query_mean"]),
                    -float(row["recall_at_10"]),
                ),
            )
            selected.append(
                {
                    "dataset": dataset,
                    "dim": best["dim"],
                    "target_recall_at_10": target,
                    "method_key": method,
                    "method": best["method"],
                    "selected_parameter": best["search_parameter"],
                    "selected_parameter_value": best["search_parameter_value"],
                    "measured_recall_at_10": best["recall_at_10"],
                    "qps": best["qps"],
                    "io_requests_per_query_mean": best["io_requests_per_query_mean"],
                    "total_4k_pages_per_query_mean": best["total_4k_pages_per_query_mean"],
                    "pages_per_request": best["pages_per_request"],
                }
            )
    return selected


def plot_common_target_bars(selected: list[dict[str, object]], stem: Path) -> None:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in selected:
        grouped[str(row["dataset"])][str(row["method_key"])] = row
    fig, axes = plt.subplots(4, 4, figsize=(21, 17.5))
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        rows = grouped[dataset]
        values = [float(rows[key]["total_4k_pages_per_query_mean"]) for key in METHOD_ORDER]
        bars = ax.bar(
            np.arange(len(METHOD_ORDER)), values,
            color=[METHODS[key]["color"] for key in METHOD_ORDER], alpha=0.9,
        )
        target = float(rows[METHOD_ORDER[0]]["target_recall_at_10"])
        ax.set_title(f"{title}\nminimum pages at Recall ≥ {target:.3f}", fontsize=10)
        ax.set_xticks(np.arange(len(METHOD_ORDER)))
        ax.set_xticklabels([METHODS[key]["short"] for key in METHOD_ORDER], fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.04, f"{value:.0f}",
                    ha="center", va="bottom", fontsize=7.5)
        if index % 4 == 0:
            ax.set_ylabel("Submitted 4 KiB pages/query (log)")
    fig.suptitle(
        "Minimum SSD pages at each dataset's highest recall reachable by all 3 methods",
        fontsize=16, y=0.994,
    )
    fig.text(
        0.5, 0.012,
        "Each bar is the measured setting with the fewest pages that reaches at least the panel target; "
        "the selected method recalls may therefore be slightly above the target.",
        ha="center", fontsize=9.2,
    )
    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.06, top=0.90,
                        wspace=0.20, hspace=0.34)
    save_figure(fig, stem)


def write_summary(selected: list[dict[str, object]]) -> Path:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in selected:
        grouped[str(row["dataset"])][str(row["method_key"])] = row
    adc_page_savings: list[float] = []
    rabitq_page_savings: list[float] = []
    adc_qps_speedup: list[float] = []
    rabitq_qps_speedup: list[float] = []
    for dataset, methods in grouped.items():
        classic = float(methods["diskann_classic"]["total_4k_pages_per_query_mean"])
        classic_qps = float(methods["diskann_classic"]["qps"])
        adc_page_savings.append(1.0 - float(methods["adcbq"]["total_4k_pages_per_query_mean"]) / classic)
        rabitq_page_savings.append(1.0 - float(methods["diskann_rabitq"]["total_4k_pages_per_query_mean"]) / classic)
        adc_qps_speedup.append(float(methods["adcbq"]["qps"]) / classic_qps)
        rabitq_qps_speedup.append(float(methods["diskann_rabitq"]["qps"]) / classic_qps)

    path = OUT / "summary.md"
    text = (
        "# Three-method SSD page-I/O summary\n\n"
        "All values use the common held-out split at T=1. Page counts are actual "
        "O_DIRECT submitted bytes divided by 4096. The operating point for each method "
        "is the measured point with minimum pages that reaches the dataset-specific "
        "highest recall reachable by all three methods.\n\n"
        f"- ADC-BQ median page reduction vs DiskANN classic: "
        f"{statistics.median(adc_page_savings) * 100:.1f}%\n"
        f"- DiskANN RaBitQ-1 median page reduction vs classic: "
        f"{statistics.median(rabitq_page_savings) * 100:.1f}%\n"
        f"- ADC-BQ median QPS ratio vs classic: {statistics.median(adc_qps_speedup):.2f}x\n"
        f"- DiskANN RaBitQ-1 median QPS ratio vs classic: "
        f"{statistics.median(rabitq_qps_speedup):.2f}x\n\n"
        "Per-request physical pages are fixed by the layouts: ADC-BQ uses 1/2/4 pages "
        "for 1024D/1536D/4096D FP sidecar records; both DiskANN variants use 2/2/5 "
        "pages for their corresponding SSD node records.\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    validate(rows)
    dataset_order = {key: i for i, (key, _) in enumerate(PANELS)}
    method_order = {key: i for i, key in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda row: (
        dataset_order[str(row["dataset"])], method_order[str(row["method_key"])],
        int(row["search_parameter_value"]),
    ))
    normalized = OUT / "normalized_page_io.csv"
    write_csv(normalized, rows, NORMALIZED_FIELDS)

    plot_ready_zoom = OUT / "recall_vs_4k_pages_zoom90.csv"
    write_plot_ready_recall_pages_csv(rows, plot_ready_zoom, min_recall=0.90)

    selected = select_common_target(rows)
    selected_path = OUT / "minimum_pages_at_highest_common_recall.csv"
    selected_fields = tuple(selected[0])
    write_csv(selected_path, selected, selected_fields)

    plot_recall_pages(rows, OUT / "recall_vs_4k_pages_full", zoom90=False)
    plot_recall_pages(rows, OUT / "recall_vs_4k_pages_zoom90", zoom90=True)
    plot_qps_pages(rows, OUT / "qps_vs_4k_pages")
    plot_common_target_bars(selected, OUT / "minimum_pages_at_highest_common_recall")
    summary = write_summary(selected)
    print(normalized)
    print(plot_ready_zoom)
    print(selected_path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
