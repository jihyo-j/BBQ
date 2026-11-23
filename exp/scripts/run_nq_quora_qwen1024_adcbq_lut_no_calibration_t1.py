#!/usr/bin/env python3
"""ADC-BQ LUT ablation without calibration/reranking.

The two scorer implementations search the same FP-built HNSW graph.  The
FP32-direct arm accumulates coordinate contributions without a query LUT;
the optimized arm uses the current int16 4-dim/16-entry nibble LUT.  Search
returns the ADC-BQ top-k directly: no calibration and no FP reranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiment"
sys.path.insert(0, str(EXPERIMENT))

import run_v051_ssd_odirect_multidataset as registry  # noqa: E402


SOURCE = EXPERIMENT / "faiss_fpgraph_packed_asym_bq_hnsw.cpp"
BUILD_SCRIPT = EXPERIMENT / "adcbq_optimized/build.sh"
BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_lut_ablation"
SCALAR_BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_scalar_direct_ablation"
RESULT_ROOT = EXPERIMENT / "nq_quora_qwen1024_adcbq_scalar_vs_lut_no_calibration_t1"
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
    scalar_build: bool = False


MODES = (
    Mode(
        "no_lut_fp32_direct_scalar",
        "No LUT (scalar direct)",
        "fp32-direct",
        "nibble",
        True,
    ),
    Mode(
        "optimized_int16_nibble_lut",
        "Optimized int16 nibble LUT",
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
QUORA_BGE_GRAPH = (
    REPO
    / "indexes/quora_bge1024_scalar_vs_lut"
    / "quora_bge_m3_vanilla.fp_built_graph.M24.efC200.faiss"
)


def datasets() -> dict[str, registry.Dataset]:
    return {
        "nq_qwen06_1024": replace(
            registry.DATASETS["nq_qwen06_1024"], adcbq_index=NQ_GRAPH
        ),
        "quora_qwen1024": replace(
            registry.DATASETS["quora_qwen1024"], adcbq_index=QUORA_GRAPH
        ),
        "quora_bge_m3_vanilla": replace(
            registry.DATASETS["quora_bge_m3_vanilla"], adcbq_index=QUORA_BGE_GRAPH
        ),
    }


def matrix_rows(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(4)
    if len(header) != 4:
        raise ValueError(f"short matrix header: {path}")
    return int.from_bytes(header, byteorder="little", signed=True)


def read_rows(path: Path) -> list[dict[str, str]]:
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


def ensure_binary(path: Path, scalar_build: bool) -> None:
    if path.exists() and path.stat().st_mtime_ns >= SOURCE.stat().st_mtime_ns:
        return
    env = {
        **os.environ,
        "OUT": str(path),
        "ADCBQ_FORCE_FP32_DIRECT_SCALAR": "1" if scalar_build else "0",
    }
    subprocess.run([str(BUILD_SCRIPT)], cwd=REPO, env=env, check=True)


def complete(
    path: Path,
    dataset: registry.Dataset,
    mode: Mode,
    efs: tuple[int, ...],
    rounds: int,
) -> bool:
    try:
        rows = read_rows(path)
        return (
            len(rows) == len(efs)
            and {int(row["efSearch"]) for row in rows} == set(efs)
            and all(int(row["nq"]) == matrix_rows(dataset.query) for row in rows)
            and all(int(row["rounds"]) == rounds for row in rows)
            and all(int(row["search_threads"]) == 1 for row in rows)
            and all(row["adc_mode"] == mode.adc_mode for row in rows)
            and all(row["rerank_enabled"] == "false" for row in rows)
            and all(row["calibrated_threshold_rerank_enabled"] == "false" for row in rows)
        )
    except (OSError, KeyError, ValueError):
        return False


def run_mode(
    dataset: registry.Dataset,
    mode: Mode,
    efs: tuple[int, ...],
    rounds: int,
    cpu: int,
    build_threads: int,
    force: bool,
) -> Path:
    binary = SCALAR_BINARY if mode.scalar_build else BINARY
    output = RESULT_ROOT / "raw" / f"{dataset.key}.{mode.key}.csv"
    if not force and complete(output, dataset, mode, efs, rounds):
        print(f"REUSE {dataset.key} {mode.key}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(binary),
        "--dataset",
        dataset.key,
        "--train",
        str(dataset.train),
        "--query",
        str(dataset.query),
        "--gt",
        str(dataset.gt),
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
        str(build_threads),
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
        *map(str, efs),
        "--no-rerank",
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)
    if not complete(output, dataset, mode, efs, rounds):
        raise RuntimeError(f"invalid or incomplete output: {output}")
    return output


def summarize(
    selected: tuple[str, ...],
    dataset_map: dict[str, registry.Dataset],
    efs: tuple[int, ...],
    rounds: int,
) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    for dataset_key in selected:
        dataset = dataset_map[dataset_key]
        for mode in MODES:
            raw_path = RESULT_ROOT / "raw" / f"{dataset.key}.{mode.key}.csv"
            for raw in read_rows(raw_path):
                rows.append(
                    {
                        "dataset": dataset.key,
                        "dataset_title": dataset.corpus,
                        "embedding": dataset.model,
                        "dim": int(raw["dim"]),
                        "mode": mode.key,
                        "mode_label": mode.label,
                        "adc_mode": raw["adc_mode"],
                        "lut_mode": "none" if mode.adc_mode == "fp32-direct" else raw["lut_mode"],
                        "efSearch": int(raw["efSearch"]),
                        "recall_at_10": float(raw["recall_at_k"]),
                        "qps_mean": float(raw["qps"]),
                        "qps_median": float(raw["qps_median"]),
                        "queries": int(raw["nq"]),
                        "rounds": int(raw["rounds"]),
                        "search_threads": int(raw["search_threads"]),
                        "calibration_enabled": False,
                        "fp_rerank_enabled": False,
                        "graph_build": raw["graph_build"],
                        "index_path": raw["index_path"],
                        "raw_csv": str(raw_path),
                    }
                )
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["efSearch"]), str(row["mode"])))
    summary_path = RESULT_ROOT / "summary_t1.csv"
    write_csv(summary_path, rows)

    by_key = {
        (str(row["dataset"]), str(row["mode"]), int(row["efSearch"])): row
        for row in rows
    }
    paired: list[dict[str, object]] = []
    for dataset_key in selected:
        for ef in efs:
            direct = by_key[(dataset_key, MODES[0].key, ef)]
            lut = by_key[(dataset_key, MODES[1].key, ef)]
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
                    "lut_speedup": float(lut["qps_median"])
                    / float(direct["qps_median"]),
                }
            )
    paired_path = RESULT_ROOT / "paired_ef_comparison_t1.csv"
    write_csv(paired_path, paired)
    return summary_path, paired_path


def plot(summary_path: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    rows = read_rows(summary_path)
    rounds = int(rows[0]["rounds"])
    dataset_keys = list(dict.fromkeys(row["dataset"] for row in rows))
    fig, axes = plt.subplots(
        1, len(dataset_keys), figsize=(4.6 * len(dataset_keys), 3.7), sharey=False
    )
    axes_list = [axes] if len(dataset_keys) == 1 else list(axes)
    styles = {
        MODES[0].key: ("#e07a25", "o", "--"),
        MODES[1].key: ("#2166ac", "s", "-"),
    }
    for axis, dataset_key in zip(axes_list, dataset_keys):
        dataset_rows = [row for row in rows if row["dataset"] == dataset_key]
        first = dataset_rows[0]
        model = first["embedding"].replace(" vanilla", "")
        title = f'{first["dataset_title"]} · {model} ({first["dim"]}D)'
        for mode in MODES:
            points = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset_key and row["mode"] == mode.key
                ),
                key=lambda row: int(row["efSearch"]),
            )
            color, marker, line = styles[mode.key]
            axis.plot(
                [float(row["recall_at_10"]) for row in points],
                [float(row["qps_median"]) for row in points],
                label=mode.label,
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.8,
                markersize=4.2,
            )
        axis.set_title(title)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel(f"QPS (median of {rounds} round{'s' if rounds != 1 else ''})")
        axis.set_yscale("log")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        axis.grid(True, which="both", alpha=0.25)
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.suptitle("ADC-BQ scorer ablation (no calibration or FP reranking)", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.76))
    png = RESULT_ROOT / "recall_qps_no_calibration_lut_ablation_t1.png"
    pdf = RESULT_ROOT / "recall_qps_no_calibration_lut_ablation_t1.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    global RESULT_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=DEFAULT_CPU)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--efs", default=",".join(map(str, DEFAULT_EFS)))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nq_qwen06_1024", "quora_qwen1024", "quora_bge_m3_vanilla"),
        default=("nq_qwen06_1024", "quora_qwen1024"),
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.result_root is not None:
        RESULT_ROOT = args.result_root.resolve()
    efs = tuple(int(value) for value in args.efs.split(",") if value)
    selected = tuple(args.datasets)
    dataset_map = datasets()

    ensure_binary(BINARY, False)
    ensure_binary(SCALAR_BINARY, True)
    for dataset_key in selected:
        dataset = dataset_map[dataset_key]
        for path in (dataset.train, dataset.query, dataset.gt, dataset.adcbq_index):
            if not path.is_file():
                raise FileNotFoundError(path)
        print(
            f"DATASET {dataset_key}: queries={matrix_rows(dataset.query)}, "
            f"graph={dataset.adcbq_index}",
            flush=True,
        )
        for mode in MODES:
            run_mode(
                dataset,
                mode,
                efs,
                args.rounds,
                args.cpu,
                args.build_threads,
                args.force,
            )

    summary_path, paired_path = summarize(selected, dataset_map, efs, args.rounds)
    png, pdf = plot(summary_path)
    manifest_path = RESULT_ROOT / "manifest.json"
    write_json(
        manifest_path,
        {
            "protocol": "ADC-BQ top-10 directly; no calibration; no FP reranking",
            "datasets": list(selected),
            "all_queries": {key: matrix_rows(dataset_map[key].query) for key in selected},
            "k": K,
            "M": M,
            "efConstruction": EF_CONSTRUCTION,
            "efs": list(efs),
            "rounds": args.rounds,
            "threads": 1,
            "cpu": args.cpu,
            "modes": [mode.__dict__ for mode in MODES],
            "summary": str(summary_path),
            "paired": str(paired_path),
            "figure_png": str(png),
            "figure_pdf": str(pdf),
            "lut_binary": str(BINARY),
            "scalar_binary": str(SCALAR_BINARY),
            "source": str(SOURCE),
        },
    )
    for path in (summary_path, paired_path, png, pdf, manifest_path):
        print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
