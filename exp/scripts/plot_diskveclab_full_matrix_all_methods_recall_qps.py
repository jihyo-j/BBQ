#!/usr/bin/env python3
"""Plot the T=1 DiskVecLab matrix plus the representative MRQ-SSD run."""

from __future__ import annotations

import argparse
import csv
import os
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
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiment/diskveclab_full_matrix"
COMBINED_CSV = ROOT / "recall_qps_all_14methods_16datasets_t1.csv"
OUTPUT_FULL = ROOT / "recall_qps_all_14methods_16datasets_t1_full"
OUTPUT_ZOOM = ROOT / "recall_qps_all_14methods_16datasets_t1_zoom90"

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
MRQ_REPRESENTATIVE_DATASET = "nq_bge_m3_vanilla"

METHODS = {
    "adcbq_pool_top10": {
        "label": "ADC-BQ · per-ef pool-top10@0.99",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
    },
    "diskann_classic": {
        "label": "DiskANN classic PQ64 + FP nodes",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "-",
    },
    "diskann_rabitq": {
        "label": "DiskANN RaBitQ-1 + adaptive FP",
        "color": "#8C564B",
        "marker": "v",
        "linestyle": "--",
    },
    "spann": {
        "label": "SPANN PQ128 + FP@80",
        "color": "#56B4E9",
        "marker": "P",
        "linestyle": "-",
    },
    "pipeann": {
        "label": "PipeANN RaBitQ-1",
        "color": "#E69F00",
        "marker": "X",
        "linestyle": "-",
    },
    "spann_rabitq": {
        "label": "SPANN RaBitQ-1 + adaptive FP",
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "--",
    },
    "rabitq_7bit": {
        "label": "RaBitQ 7-bit (RAM)",
        "color": "#009E73",
        "marker": "o",
        "linestyle": "-",
    },
    "rabitq_1bit_adaptive": {
        "label": "HNSW RaBitQ-1 C16 + adaptive FP",
        "color": "#6A3D9A",
        "marker": "x",
        "linestyle": "--",
    },
    "elasticsearch_bbq": {
        "label": "Elasticsearch BBQ + FP@80",
        "color": "#333333",
        "marker": "*",
        "linestyle": ":",
    },
    "mrq_ssd": {
        "label": "MRQ-SSD B320 async QD8",
        "color": "#9467BD",
        "marker": "h",
        "linestyle": "-.",
    },
    "rabitq_ssd": {
        "label": "RaBitQ-SSD B9 async QD8",
        "color": "#17BECF",
        "marker": ">",
        "linestyle": "-.",
    },
    "turboquant_tq1_fp80": {
        "label": "TurboQuant TQ1+ + FP@80 (io_uring)",
        "color": "#E69F00",
        "marker": "p",
        "linestyle": "-",
    },
    "turboquant_tq1_fp60": {
        "label": "TurboQuant TQ1+ + FP@60 (io_uring)",
        "color": "#D55E00",
        "marker": "H",
        "linestyle": "-",
    },
    "turboquant_tq1_pareto": {
        "label": "TurboQuant TQ1+ + Pareto R (io_uring)",
        "color": "#E69F00",
        "marker": "p",
        "linestyle": "-",
    },
    "turboquant_tq1_tuned_r": {
        "label": "TurboQuant TQ1+ + dataset-tuned R (io_uring)",
        "color": "#E69F00",
        "marker": "p",
        "linestyle": "-",
    },
    "faiss_pq8m128_pareto": {
        "label": "Faiss PQ8-M128 + Pareto R (io_uring)",
        "color": "#F58518",
        "marker": "d",
        "linestyle": "-",
    },
    "faiss_opq_pq8m128_pareto": {
        "label": "Faiss OPQ-PQ8-M128 + Pareto R (io_uring)",
        "color": "#54A24B",
        "marker": "<",
        "linestyle": "-",
    },
    "faiss_pq8m128_tuned_r": {
        "label": "Faiss PQ8-M128 + dataset-tuned R (io_uring)",
        "color": "#F58518",
        "marker": "d",
        "linestyle": "-",
    },
    "faiss_opq_pq8m128_tuned_r": {
        "label": "Faiss OPQ-PQ8-M128 + dataset-tuned R (io_uring)",
        "color": "#54A24B",
        "marker": "<",
        "linestyle": "-",
    },
}
METHOD_ORDER = tuple(METHODS)
COMPRESSED_METHODS = {
    "turboquant_tq1_fp80",
    "turboquant_tq1_fp60",
    "turboquant_tq1_pareto",
    "turboquant_tq1_tuned_r",
    "faiss_pq8m128_pareto",
    "faiss_opq_pq8m128_pareto",
    "faiss_pq8m128_tuned_r",
    "faiss_opq_pq8m128_tuned_r",
}
PARETO_METHODS = {
    "turboquant_tq1_pareto",
    "faiss_pq8m128_pareto",
    "faiss_opq_pq8m128_pareto",
}
TUNED_R_METHODS = {
    "turboquant_tq1_tuned_r",
    "faiss_pq8m128_tuned_r",
    "faiss_opq_pq8m128_tuned_r",
}
TUNED_R_SPECS = {
    "turboquant_tq1_tuned_r": ("tq_rerank_pareto_uring", "qdranttq1"),
    "faiss_pq8m128_tuned_r": ("pq_opq_rerank_pareto_uring", "pq8m128"),
    "faiss_opq_pq8m128_tuned_r": (
        "pq_opq_rerank_pareto_uring",
        "opq8m128",
    ),
}
BASE_C16_METHOD_ORDER = tuple(
    key for key in METHOD_ORDER if key not in COMPRESSED_METHODS
)
TQ_NO_PIPEANN_METHOD_ORDER = (
    "adcbq_pool_top10",
    "turboquant_tq1_fp80",
    "diskann_classic",
    "diskann_rabitq",
    "spann",
    "rabitq_7bit",
    "rabitq_1bit_adaptive",
    "elasticsearch_bbq",
    "mrq_ssd",
    "rabitq_ssd",
)
COMPRESSED_NO_PIPEANN_METHOD_ORDER = (
    "adcbq_pool_top10",
    "turboquant_tq1_fp80",
    "turboquant_tq1_fp60",
    "faiss_pq8m128_pareto",
    "faiss_opq_pq8m128_pareto",
    "diskann_classic",
    "diskann_rabitq",
    "spann",
    "rabitq_7bit",
    "rabitq_1bit_adaptive",
    "elasticsearch_bbq",
    "mrq_ssd",
    "rabitq_ssd",
)
DATASET_TUNED_R_NO_PIPEANN_METHOD_ORDER = (
    "adcbq_pool_top10",
    "turboquant_tq1_tuned_r",
    "faiss_pq8m128_tuned_r",
    "faiss_opq_pq8m128_tuned_r",
    "diskann_classic",
    "diskann_rabitq",
    "spann",
    "rabitq_7bit",
    "rabitq_1bit_adaptive",
    "elasticsearch_bbq",
    "mrq_ssd",
    "rabitq_ssd",
)
FIELDS = (
    "dataset",
    "corpus",
    "model",
    "method_key",
    "method",
    "search_parameter",
    "search_parameter_value",
    "efSearch",
    "rerank_n",
    "recall_at_10",
    "qps",
    "rounds",
    "search_threads",
    "test_queries",
    "storage",
    "io_engine",
    "source_csv",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def dataset_tuned_r_rows(method_key: str) -> tuple[Path, list[dict[str, str]]]:
    subdir, source_key = TUNED_R_SPECS[method_key]
    directory = ROOT / subdir
    selections = [
        row
        for row in read_csv(directory / "target99_max_qps_t1.csv")
        if row["method_key"] == source_key
    ]
    selected_r = {
        row["dataset"]: int(row["rerank_n"])
        for row in selections
    }
    if set(selected_r) != DATASETS:
        raise RuntimeError(f"dataset-tuned R coverage mismatch for {method_key}")
    summary_path = directory / "summary_t1.csv"
    rows = [
        row
        for row in read_csv(summary_path)
        if row["method_key"] == source_key
        and int(row["rerank_n"]) == selected_r[row["dataset"]]
    ]
    for dataset in DATASETS:
        curve = [row for row in rows if row["dataset"] == dataset]
        if not curve or len({int(row["rerank_n"]) for row in curve}) != 1:
            raise RuntimeError(
                f"dataset-tuned R is not fixed within curve: {method_key} {dataset}"
            )
    return summary_path, rows


def add_rows(
    output: list[dict[str, object]],
    rows: list[dict[str, str]],
    *,
    path: Path,
    method_key: str,
    parameter: str,
    parameter_field: str,
    recall_field: str = "recall_at_10",
    thread_field: str = "search_threads",
    query_field: str = "nq",
) -> None:
    style = METHODS[method_key]
    for row in rows:
        dataset = row["dataset"]
        if dataset not in DATASETS:
            continue
        output.append(
            {
                "dataset": dataset,
                "corpus": row.get("corpus", ""),
                "model": row.get("model", ""),
                "method_key": method_key,
                "method": style["label"],
                "search_parameter": parameter,
                "search_parameter_value": int(float(row[parameter_field])),
                "efSearch": row.get("efSearch", ""),
                "rerank_n": row.get("rerank_n", ""),
                "recall_at_10": float(row[recall_field]),
                "qps": float(row["qps"]),
                "rounds": int(float(row.get("rounds", 1))),
                "search_threads": int(float(row.get(thread_field, 1))),
                "test_queries": int(float(row[query_field])),
                "storage": row.get("storage", row.get("io_path", "")),
                "io_engine": row.get("io_engine", row.get("timing_scope", "")),
                "source_csv": relative(path),
            }
        )


def collect() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    adc_path = ROOT / "calibrated_adcbq_pool_top10_target99_per_ef/summary_t1.csv"
    adc_rows = read_csv(adc_path)
    if any(row["calibration_status"] != "feasible" for row in adc_rows):
        raise RuntimeError("pool-top10 ADC-BQ summary contains infeasible rows")
    add_rows(
        output,
        adc_rows,
        path=adc_path,
        method_key="adcbq_pool_top10",
        parameter="efSearch",
        parameter_field="efSearch",
        query_field="test_queries",
    )

    for variant, method_key in (
        ("diskann_classic", "diskann_classic"),
        ("diskann_rabitq_1bit_adaptive_fp", "diskann_rabitq"),
    ):
        for path in sorted((ROOT / "diskann").glob(f"*/{variant}/search_t1.csv")):
            add_rows(
                output,
                read_csv(path),
                path=path,
                method_key=method_key,
                parameter="Lsearch",
                parameter_field="Lsearch",
            )

    spann_path = ROOT / "spann_pipeann/all_search_results.csv"
    spann_rows = read_csv(spann_path)
    for family, method_key, parameter, parameter_field in (
        ("spann_pq128_fixed_fp_rerank80", "spann", "internal_heads", "internal_heads"),
        ("pipeann_native_rabitq_1bit_c16", "pipeann", "Lsearch", "Lsearch"),
    ):
        selected = [row for row in spann_rows if row["family"] == family]
        add_rows(
            output,
            selected,
            path=spann_path,
            method_key=method_key,
            parameter=parameter,
            parameter_field=parameter_field,
            thread_field="threads",
        )

    spann_rabitq_path = (
        ROOT / "spann_rabitq_1bit_adaptive_fp/spann_rabitq_results.csv"
    )
    add_rows(
        output,
        read_csv(spann_rabitq_path),
        path=spann_rabitq_path,
        method_key="spann_rabitq",
        parameter="candidate_n",
        parameter_field="candidate_n",
        query_field="test_queries",
    )

    for path in sorted((ROOT / "rabitq7").glob("*.csv")):
        add_rows(
            output,
            read_csv(path),
            path=path,
            method_key="rabitq_7bit",
            parameter="efSearch",
            parameter_field="efSearch",
            recall_field="recall_at_k",
        )

    rabitq1_path = ROOT / "rabitq1_c16_adaptive/summary_t1.csv"
    add_rows(
        output,
        read_csv(rabitq1_path),
        path=rabitq1_path,
        method_key="rabitq_1bit_adaptive",
        parameter="efSearch",
        parameter_field="efSearch",
        recall_field="recall_at_k",
    )

    elastic_path = ROOT / "elasticsearch_bbq/all_results.csv"
    add_rows(
        output,
        read_csv(elastic_path),
        path=elastic_path,
        method_key="elasticsearch_bbq",
        parameter="num_candidates",
        parameter_field="num_candidates",
        thread_field="client_concurrency",
    )

    ssd_path = ROOT / "mrq_rabitq_ssd/summary_t1.csv"
    ssd_rows = read_csv(ssd_path)
    for method_key in ("mrq_ssd", "rabitq_ssd"):
        selected = [row for row in ssd_rows if row["method_key"] == method_key]
        if method_key == "mrq_ssd":
            selected = [
                row
                for row in selected
                if row["dataset"] == MRQ_REPRESENTATIVE_DATASET
            ]
        add_rows(
            output,
            selected,
            path=ssd_path,
            method_key=method_key,
            parameter="nprobe",
            parameter_field="search_parameter_value",
            query_field="test_queries",
        )

    if "turboquant_tq1_fp80" in METHOD_ORDER:
        compressed_path = ROOT / "tq_pq_opq_uring/summary_t1.csv"
        compressed_rows = read_csv(compressed_path)
        selected = [
            row for row in compressed_rows if row["method_key"] == "qdranttq1"
        ]
        if any(int(row["rerank_n"]) != 80 for row in selected):
            raise RuntimeError("unexpected rerank budget for TurboQuant FP@80")
        if any(
            row["io_engine"]
            != "O_DIRECT io_uring one fixed-R batch per query"
            for row in selected
        ):
            raise RuntimeError("unexpected I/O engine for TurboQuant FP@80")
        add_rows(
            output,
            selected,
            path=compressed_path,
            method_key="turboquant_tq1_fp80",
            parameter="efSearch",
            parameter_field="efSearch",
            query_field="test_queries",
        )
    if "turboquant_tq1_pareto" in METHOD_ORDER:
        tq_pareto_path = (
            ROOT / "tq_rerank_pareto_uring/recall_qps_pareto_frontier_t1.csv"
        )
        selected = [
            row
            for row in read_csv(tq_pareto_path)
            if row["method_key"] == "qdranttq1"
        ]
        if any(
            row["io_engine"]
            != "O_DIRECT io_uring one fixed-R batch per query"
            for row in selected
        ):
            raise RuntimeError("unexpected I/O engine for TurboQuant Pareto")
        add_rows(
            output,
            selected,
            path=tq_pareto_path,
            method_key="turboquant_tq1_pareto",
            parameter="joint efSearch/R Pareto",
            parameter_field="efSearch",
            query_field="test_queries",
        )
    for method_key in TUNED_R_METHODS.intersection(METHOD_ORDER):
        tuned_path, selected = dataset_tuned_r_rows(method_key)
        if any(
            row["io_engine"]
            != "O_DIRECT io_uring one fixed-R batch per query"
            for row in selected
        ):
            raise RuntimeError(f"unexpected I/O engine for {method_key}")
        add_rows(
            output,
            selected,
            path=tuned_path,
            method_key=method_key,
            parameter="efSearch (dataset-fixed R)",
            parameter_field="efSearch",
            query_field="test_queries",
        )
    if PARETO_METHODS.intersection(METHOD_ORDER):
        pareto_path = ROOT / "pq_opq_rerank_pareto_uring/recall_qps_pareto_frontier_t1.csv"
        pareto_rows = read_csv(pareto_path)
        pareto_methods = (
            ("pq8m128", "faiss_pq8m128_pareto"),
            ("opq8m128", "faiss_opq_pq8m128_pareto"),
        )
        for source_key, method_key in pareto_methods:
            if method_key not in METHOD_ORDER:
                continue
            selected = [
                row for row in pareto_rows if row["method_key"] == source_key
            ]
            if any(
                row["io_engine"]
                != "O_DIRECT io_uring one fixed-R batch per query"
                for row in selected
            ):
                raise RuntimeError(f"unexpected I/O engine for {source_key} Pareto")
            add_rows(
                output,
                selected,
                path=pareto_path,
                method_key=method_key,
                parameter="joint efSearch/R Pareto",
                parameter_field="efSearch",
                query_field="test_queries",
            )
    if "turboquant_tq1_fp60" in METHOD_ORDER:
        tq60_path = ROOT / "tq60_uring/summary_t1.csv"
        tq60_rows = read_csv(tq60_path)
        selected = [
            row for row in tq60_rows if row["method_key"] == "qdranttq1"
        ]
        if any(int(row["rerank_n"]) != 60 for row in selected):
            raise RuntimeError("unexpected rerank budget for TurboQuant FP@60")
        if any(
            row["io_engine"]
            != "O_DIRECT io_uring one fixed-R batch per query"
            for row in selected
        ):
            raise RuntimeError("unexpected I/O engine for TurboQuant FP@60")
        add_rows(
            output,
            selected,
            path=tq60_path,
            method_key="turboquant_tq1_fp60",
            parameter="efSearch",
            parameter_field="efSearch",
            query_field="test_queries",
        )
    return [row for row in output if str(row["method_key"]) in METHOD_ORDER]


def validate(rows: list[dict[str, object]]) -> None:
    full_matrix_methods = set(METHOD_ORDER) - {"mrq_ssd"}
    fixed_methods = full_matrix_methods - PARETO_METHODS - TUNED_R_METHODS
    expected_counts = {
        (dataset, method): 20
        for dataset in DATASETS
        for method in fixed_methods
    }
    active_pareto = PARETO_METHODS.intersection(METHOD_ORDER)
    if active_pareto:
        source_counts: Counter[tuple[str, str]] = Counter()
        pareto_specs = (
            (
                ROOT / "tq_rerank_pareto_uring/recall_qps_pareto_frontier_t1.csv",
                "qdranttq1",
                "turboquant_tq1_pareto",
            ),
            (
                ROOT / "pq_opq_rerank_pareto_uring/recall_qps_pareto_frontier_t1.csv",
                "pq8m128",
                "faiss_pq8m128_pareto",
            ),
            (
                ROOT / "pq_opq_rerank_pareto_uring/recall_qps_pareto_frontier_t1.csv",
                "opq8m128",
                "faiss_opq_pq8m128_pareto",
            ),
        )
        for path, source_key, method_key in pareto_specs:
            if method_key not in active_pareto:
                continue
            source_counts.update(
                (row["dataset"], method_key)
                for row in read_csv(path)
                if row["method_key"] == source_key
            )
        expected_pareto_keys = {
            (dataset, method)
            for dataset in DATASETS
            for method in active_pareto
        }
        if set(source_counts) != expected_pareto_keys:
            raise RuntimeError("joint Pareto source coverage mismatch")
        if any(count <= 0 for count in source_counts.values()):
            raise RuntimeError("empty joint Pareto frontier")
        expected_counts.update(source_counts)
    for method_key in TUNED_R_METHODS.intersection(METHOD_ORDER):
        _, selected = dataset_tuned_r_rows(method_key)
        tuned_counts = Counter(row["dataset"] for row in selected)
        if set(tuned_counts) != DATASETS:
            raise RuntimeError(f"dataset-tuned source coverage mismatch: {method_key}")
        expected_counts.update(
            {
                (dataset, method_key): count
                for dataset, count in tuned_counts.items()
            }
        )
    expected_counts[(MRQ_REPRESENTATIVE_DATASET, "mrq_ssd")] = 11
    expected = sum(expected_counts.values())
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    counts = Counter((row["dataset"], row["method_key"]) for row in rows)
    if set(counts) != set(expected_counts):
        raise RuntimeError("dataset/method coverage mismatch")
    bad = {
        key: count
        for key, count in counts.items()
        if count != expected_counts[key]
    }
    if bad:
        raise RuntimeError(f"unexpected point count per curve: {bad}")
    if any(int(row["search_threads"]) != 1 for row in rows):
        raise RuntimeError("non-T=1 row found")
    if any(float(row["qps"]) <= 0 for row in rows):
        raise RuntimeError("non-positive QPS found")
    if any(not 0 <= float(row["recall_at_10"]) <= 1 for row in rows):
        raise RuntimeError("Recall@10 outside [0, 1]")

    split_rows = read_csv(ROOT / "split_manifest.csv")
    expected_queries = {row["dataset"]: int(row["test_queries"]) for row in split_rows}
    mismatches = [
        (row["dataset"], row["method_key"], row["test_queries"])
        for row in rows
        if int(row["test_queries"]) != expected_queries[row["dataset"]]
    ]
    if mismatches:
        raise RuntimeError(f"held-out split mismatch: {mismatches[:5]}")


def write_combined(rows: list[dict[str, object]]) -> None:
    order = {dataset: i for i, (dataset, _) in enumerate(PANELS)}
    method_order = {method: i for i, method in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            order[str(row["dataset"])],
            method_order[str(row["method_key"])],
            float(row["recall_at_10"])
            if row["method_key"] in PARETO_METHODS
            else float(row["search_parameter_value"]),
        )
    )
    with COMBINED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], output: Path, zoom90: bool) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_key"]))].append(row)

    fig, axes = plt.subplots(4, 4, figsize=(21, 17.5), constrained_layout=False)
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        for method_key in METHOD_ORDER:
            style = METHODS[method_key]
            points = sorted(
                grouped[(dataset, method_key)],
                key=lambda row: (
                    float(row["recall_at_10"])
                    if method_key in PARETO_METHODS
                    else float(row["search_parameter_value"])
                ),
            )
            if zoom90:
                points = [p for p in points if float(p["recall_at_10"]) >= 0.90]
            if not points:
                continue
            ax.plot(
                [float(p["recall_at_10"]) for p in points],
                [float(p["qps"]) for p in points],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.35,
                markersize=3.6 if method_key != "elasticsearch_bbq" else 5.0,
                markeredgewidth=0.45,
                alpha=0.92,
                zorder=3,
            )
        ax.axvline(0.99, color="#2CA02C", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
        ax.set_title(title, fontsize=10.5, pad=5)
        if zoom90:
            ax.set_xlim(0.90, 1.001)
        else:
            recalls = [
                float(row["recall_at_10"])
                for row in rows
                if row["dataset"] == dataset
            ]
            margin = max(0.005, (max(recalls) - min(recalls)) * 0.035)
            ax.set_xlim(max(0.0, min(recalls) - margin), min(1.001, max(recalls) + margin))
        if index // 4 == 3:
            ax.set_xlabel("Held-out Recall@10", fontsize=9.5)
        if index % 4 == 0:
            ax.set_ylabel("QPS (log scale)", fontsize=9.5)
        ax.tick_params(labelsize=8)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHODS[key]["color"],
            marker=METHODS[key]["marker"],
            linestyle=METHODS[key]["linestyle"],
            linewidth=1.6,
            markersize=5,
            label=METHODS[key]["label"],
        )
        for key in METHOD_ORDER
    ]
    handles.append(
        Line2D([0], [0], color="#2CA02C", linestyle="--", label="Recall@10 = 0.99")
    )
    dense_legend = len(handles) >= 12
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4 if dense_legend else 3,
        frameon=False,
        fontsize=9.0 if dense_legend else 9.5,
        columnspacing=1.1 if dense_legend else 1.4,
        handlelength=2.5,
    )
    scope = "high-recall zoom (Recall@10 ≥ 0.90)" if zoom90 else "full measured recall range"
    if TUNED_R_METHODS.intersection(METHOD_ORDER):
        compressed_note = (
            "TurboQuant/PQ/OPQ curves fix one dataset-specific R from the "
            "target-0.99 joint sweep (best available fallback if unattainable), "
            "then vary the measured efSearch settings satisfying efSearch >= R. "
        )
    else:
        compressed_note = (
            "PQ/OPQ curves are non-dominated frontiers from 142 measured joint "
            "(efSearch, R) settings per dataset. "
        )
    fig.suptitle(
        f"DiskVecLab Recall–QPS — {len(set(METHOD_ORDER) - {'mrq_ssd'})} "
        f"full-matrix methods + MRQ-SSD representative, T=1 ({scope})",
        fontsize=16,
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Common held-out split; non-tuned fixed-setting curves use 20 measured "
        "efSearch settings. "
        + compressed_note
        + "MRQ-SSD: 11 measured settings on representative NQ · BGE-M3 only. "
        "ADC-BQ calibration target is 0.99 query-level preservation of the exact-FP "
        "top-10 inside each ef candidate pool, not final Recall@10.",
        ha="center",
        va="bottom",
        fontsize=9.2,
    )
    # The all-method profile has fourteen legend entries.  Reserve enough
    # vertical room that its four legend rows cannot overlap the first row of
    # panel titles; the smaller profiles retain the more compact layout.
    axes_top = 0.845 if dense_legend else 0.895
    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        bottom=0.055,
        top=axes_top,
        wspace=0.20,
        hspace=0.28,
    )
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 190} if extension == "png" else {}
        fig.savefig(output.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    fig.savefig(
        output.with_name(f"{output.name}_preview").with_suffix(".jpg"),
        bbox_inches="tight",
        dpi=76,
        facecolor="white",
        pil_kwargs={"quality": 88, "optimize": True},
    )
    plt.close(fig)


def audit_adc_monotonicity(rows: list[dict[str, object]]) -> None:
    for dataset, _ in PANELS:
        points = sorted(
            (
                row
                for row in rows
                if row["dataset"] == dataset
                and row["method_key"] == "adcbq_pool_top10"
            ),
            key=lambda row: int(row["search_parameter_value"]),
        )
        decreases = sum(
            float(right["recall_at_10"]) + 1e-12 < float(left["recall_at_10"])
            for left, right in zip(points, points[1:])
        )
        print(
            f"ADC monotonic audit {dataset}: decreases={decreases}, "
            f"recall={float(points[0]['recall_at_10']):.6f}"
            f"->{float(points[-1]['recall_at_10']):.6f}"
        )


def main() -> int:
    global METHOD_ORDER, COMBINED_CSV, OUTPUT_FULL, OUTPUT_ZOOM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "base-c16",
            "all",
            "tq-no-pipeann",
            "compressed-no-pipeann",
            "tuned-r-no-pipeann",
        ),
        default="base-c16",
        help="base-c16 reproduces the attached base figure plus HNSW RaBitQ C16",
    )
    args = parser.parse_args()
    if args.profile == "base-c16":
        METHOD_ORDER = BASE_C16_METHOD_ORDER
    elif args.profile == "tq-no-pipeann":
        METHOD_ORDER = TQ_NO_PIPEANN_METHOD_ORDER
    elif args.profile == "compressed-no-pipeann":
        METHOD_ORDER = COMPRESSED_NO_PIPEANN_METHOD_ORDER
    elif args.profile == "tuned-r-no-pipeann":
        METHOD_ORDER = DATASET_TUNED_R_NO_PIPEANN_METHOD_ORDER
    method_count = len(METHOD_ORDER)
    if args.profile == "tq-no-pipeann":
        stem = "recall_qps_selected_tq_no_pipeann_10methods_16datasets_t1"
    elif args.profile == "compressed-no-pipeann":
        stem = (
            "recall_qps_selected_tq60_pq_opq_no_pipeann_"
            "13methods_16datasets_t1"
        )
    elif args.profile == "tuned-r-no-pipeann":
        stem = (
            "recall_qps_selected_dataset_r_tq_pq_opq_no_pipeann_"
            "12methods_16datasets_t1"
        )
    else:
        stem = f"recall_qps_all_{method_count}methods_16datasets_t1"
    COMBINED_CSV = ROOT / f"{stem}.csv"
    OUTPUT_FULL = ROOT / f"{stem}_full"
    OUTPUT_ZOOM = ROOT / f"{stem}_zoom90"
    rows = collect()
    validate(rows)
    audit_adc_monotonicity(rows)
    write_combined(rows)
    plot(rows, OUTPUT_FULL, zoom90=False)
    plot(rows, OUTPUT_ZOOM, zoom90=True)
    print(COMBINED_CSV)
    print(OUTPUT_FULL.with_suffix(".png"))
    print(OUTPUT_ZOOM.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
