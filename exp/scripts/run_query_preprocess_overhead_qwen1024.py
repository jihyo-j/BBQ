#!/usr/bin/env python3
"""Benchmark method-specific query preprocessing for Qwen-1024 ANN methods."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiment/query_preprocess_overhead_qwen1024.cpp"
BINARY = REPO / "experiment/query_preprocess_overhead_qwen1024"
OUTPUT_ROOT = REPO / "experiment/query_preprocess_overhead_qwen1024_t1"
FAISS_ROOT = Path("/home/jihyo/workspace/faiss")
FAISS_BUILD = FAISS_ROOT / "build_hnsw_conda/faiss"
CONDA_LIB = Path("/home/jihyo/anaconda3/envs/hnsw/lib")
MRQ_ROOT = Path("/home/smrc/samsung-nvme/jihyo/resq_ssd_qwen1024")
OPQ_INDEX = Path(
    "/home/smrc/samsung-nvme/jihyo/faiss_exact_matched_budget_compare_results/"
    "indexes/msmarco_qwen1024/msmarco_qwen1024.opq8m128.faiss"
)
TQ_FFI = REPO / "experiment/qdrant_tq_ffi"


@dataclass(frozen=True)
class Workload:
    key: str
    query: Path

    @property
    def mrq_dir(self) -> Path:
        return MRQ_ROOT / self.key

    @property
    def pca(self) -> Path:
        return self.mrq_dir / f"{self.key}_pca.fvecs"

    @property
    def mean(self) -> Path:
        return self.mrq_dir / f"{self.key}_mean.fvecs"

    @property
    def rotation(self) -> Path:
        return self.mrq_dir / "RESP_C4096_B320.fvecs"

    @property
    def centroids(self) -> Path:
        return self.mrq_dir / "RESCentroid_C4096_B320.fvecs"


WORKLOADS = (
    Workload(
        "msmarco_qwen1024",
        REPO
        / "experiment/adcbq_calibrated_threshold_overlap/splits/msmarco_qwen1024/"
        "seed11.cal1000.test5980.test.query.fbin",
    ),
    Workload(
        "dbpedia_qwen1024",
        REPO
        / "experiment/adcbq_calibrated_threshold_overlap/splits/dbpedia_qwen1024/"
        "seed11.cal100.test300.test.query.fbin",
    ),
)


def compile_binary() -> None:
    command = [
        "g++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-march=native",
        "-fopenmp",
        f"-I{FAISS_ROOT}",
        f"-I{FAISS_BUILD}",
        f"-I{REPO / 'include'}",
        f"-I{TQ_FFI}",
        str(SOURCE),
        f"-L{FAISS_BUILD}",
        f"-L{CONDA_LIB}",
        f"-L{TQ_FFI / 'target/release'}",
        f"-Wl,-rpath,{FAISS_BUILD}:{CONDA_LIB}:{TQ_FFI / 'target/release'}",
        "-lfaiss_avx2",
        "-lqdrant_tq_ffi",
        "-lopenblas",
        "-ldl",
        "-lpthread",
        "-o",
        str(BINARY),
    ]
    subprocess.run(command, cwd=REPO, check=True)


def benchmark(workload: Workload, output: Path, args: argparse.Namespace) -> None:
    for path in (
        workload.query,
        workload.pca,
        workload.mean,
        workload.rotation,
        workload.centroids,
        OPQ_INDEX,
        TQ_FFI / "target/release/libqdrant_tq_ffi.so",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    command = [
        str(BINARY),
        "--dataset",
        workload.key,
        "--query",
        str(workload.query),
        "--mrq-pca",
        str(workload.pca),
        "--mrq-mean",
        str(workload.mean),
        "--mrq-rotation",
        str(workload.rotation),
        "--mrq-centroids",
        str(workload.centroids),
        "--opq-index",
        str(OPQ_INDEX),
        "--tq-data",
        str(output.parent / f"{workload.key}.tqplus1.tqdata"),
        "--tq-meta",
        str(output.parent / f"{workload.key}.tqplus1.meta.json"),
        "--output",
        str(output),
        "--cpu",
        str(args.cpu),
        "--rounds",
        str(args.rounds),
        "--fixed-iterations",
        str(args.fixed_iterations),
        "--mrq-iterations",
        str(args.mrq_iterations),
    ]
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    subprocess.run(command, cwd=REPO, env=env, check=True)


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive(rows: list[dict[str, str]], output_dir: Path) -> tuple[Path, Path, Path, Path]:
    derived: list[dict[str, object]] = []
    for workload in WORKLOADS:
        selected = [row for row in rows if row["dataset"] == workload.key]
        fixed = {
            row["method"]: float(row["median_us_per_query"])
            for row in selected
            if row["stage"] == "method_total"
        }
        transform = next(
            float(row["median_us_per_query"])
            for row in selected
            if row["method"] == "MRQ B320 C4096"
            and row["stage"] == "dense_transform"
        )
        for method, latency in fixed.items():
            derived.append(
                {
                    "dataset": workload.key,
                    "method": method,
                    "nprobe": "",
                    "component": "method_total",
                    "median_us_per_query": latency,
                    "preprocess_only_qps_ceiling": 1e6 / latency,
                    "excludes": "common normalization; ANN search; rerank; I/O",
                }
            )
        derived.append(
            {
                "dataset": workload.key,
                "method": "MRQ B320 C4096",
                "nprobe": 0,
                "component": "dense_transform_only",
                "median_us_per_query": transform,
                "preprocess_only_qps_ceiling": 1e6 / transform,
                "excludes": "IVF centroid routing; ANN scan; rerank; I/O",
            }
        )
        routing_by_probe = {
            int(row["parameter"]): float(row["median_us_per_query"])
            for row in selected
            if row["method"] == "MRQ B320 C4096"
            and row["stage"] == "ivf_centroid_routing"
        }
        for row in selected:
            if row["method"] != "MRQ B320 C4096" or row["stage"] != "probe_residual_luts":
                continue
            probe = int(row["parameter"])
            probe_luts = float(row["median_us_per_query"])
            routing = routing_by_probe[probe]
            total = transform + routing + probe_luts
            derived.append(
                {
                    "dataset": workload.key,
                    "method": "MRQ B320 C4096",
                    "nprobe": probe,
                    "component": "dense_transform_plus_routing_plus_probe_luts",
                    "median_us_per_query": total,
                    "preprocess_only_qps_ceiling": 1e6 / total,
                    "excludes": "ANN list scan; exact rerank; I/O",
                }
            )

    derived_csv = output_dir / "query_preprocess_overhead_derived.csv"
    write_csv(derived_csv, derived)

    import matplotlib.pyplot as plt

    labels = {
        "ADC-BQ int16 nibble": "Calibrated\nADC-BQ int16",
        "RaBitQ-1b C16": "RaBitQ-1b\nC16",
        "PQ M256x8 dsub4": "PQ8 M256\ndsub=4",
        "PQ M128x8 dsub8": "PQ8 M128\ndsub=8",
        "SQ4": "SQ4\n(pointer bind)",
        "OPQ M128x8": "OPQ-PQ8\nM=128",
        "TurboQuant TQ+ 1b": "TurboQuant\nTQ+ 1b",
        "MRQ B320 C4096": "MRQ B320\ntransform",
    }
    colors = {
        "ADC-BQ int16 nibble": "#d62728",
        "RaBitQ-1b C16": "#1f77b4",
        "PQ M256x8 dsub4": "#9467bd",
        "PQ M128x8 dsub8": "#8c564b",
        "SQ4": "#7f7f7f",
        "OPQ M128x8": "#e377c2",
        "TurboQuant TQ+ 1b": "#17becf",
        "MRQ B320 C4096": "#2ca02c",
    }
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 9.4))
    for row_index, workload in enumerate(WORKLOADS):
        selected = [row for row in derived if row["dataset"] == workload.key]
        fixed_rows = [
            row
            for row in selected
            if row["component"] in {"method_total", "dense_transform_only"}
        ]
        fixed_order = [
            "ADC-BQ int16 nibble",
            "RaBitQ-1b C16",
            "PQ M256x8 dsub4",
            "PQ M128x8 dsub8",
            "SQ4",
            "OPQ M128x8",
            "TurboQuant TQ+ 1b",
            "MRQ B320 C4096",
        ]
        fixed_by_method = {str(row["method"]): row for row in fixed_rows}
        bar_axis = axes[row_index, 0]
        bar_values = [
            float(fixed_by_method[method]["median_us_per_query"])
            for method in fixed_order
        ]
        bars = bar_axis.bar(
            [labels[method] for method in fixed_order],
            bar_values,
            color=[colors[method] for method in fixed_order],
        )
        bar_axis.set_yscale("log")
        bar_axis.set_ylabel("µs/query (median, log scale)")
        bar_axis.set_title(f"{workload.key}: fixed query preparation")
        bar_axis.grid(True, axis="y", which="both", alpha=0.25)
        for bar, value in zip(bars, bar_values):
            bar_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.3f}" if value < 0.1 else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        line_axis = axes[row_index, 1]
        mrq_rows = sorted(
            [
                row
                for row in selected
                if row["component"] == "dense_transform_plus_routing_plus_probe_luts"
            ],
            key=lambda row: int(row["nprobe"]),
        )
        line_axis.plot(
            [int(row["nprobe"]) for row in mrq_rows],
            [float(row["median_us_per_query"]) for row in mrq_rows],
            marker="o",
            linewidth=2.2,
            color=colors["MRQ B320 C4096"],
            label="MRQ transform + C4096 routing + per-probe LUTs",
        )
        for method in (
            "ADC-BQ int16 nibble",
            "RaBitQ-1b C16",
            "PQ M256x8 dsub4",
            "OPQ M128x8",
            "TurboQuant TQ+ 1b",
        ):
            value = float(fixed_by_method[method]["median_us_per_query"])
            line_axis.axhline(
                value,
                linewidth=1.7,
                linestyle="--",
                color=colors[method],
                label=labels[method].replace("\n", " "),
            )
        line_axis.set_xscale("log", base=2)
        line_axis.set_yscale("log")
        line_axis.set_xlabel("MRQ nprobe")
        line_axis.set_ylabel("µs/query (median, log scale)")
        line_axis.set_title(f"{workload.key}: MRQ nprobe scaling")
        line_axis.grid(True, which="both", alpha=0.25)
        line_axis.legend(fontsize=8)

    figure.suptitle(
        "Qwen-1024 query-dependent setup compute — T=1, CPU-pinned",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    figure.text(
        0.5,
        0.012,
        (
            "Excludes ANN list/graph traversal, exact rerank, SSD I/O, and common normalization. "
            "LUTs are included; MRQ also includes C4096 centroid scan+partial sort. "
            "SQ4 setup is pointer-only; its decode/distance work occurs per candidate."
        ),
        ha="center",
        fontsize=9,
    )
    figure_path = output_dir / "query_preprocess_overhead.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    component_rows: list[dict[str, object]] = []
    for workload in WORKLOADS:
        selected_raw = [row for row in rows if row["dataset"] == workload.key]

        def raw_value(method: str, stage: str, parameter: int | None = None) -> float:
            return next(
                float(row["median_us_per_query"])
                for row in selected_raw
                if row["method"] == method
                and row["stage"] == stage
                and (parameter is None or int(row["parameter"]) == parameter)
            )

        fixed_components = (
            ("ADC-BQ", "non_lut", raw_value(
                "ADC-BQ int16 nibble", "calibration_tau_scale_scan"
            ), "calibration tau-width scale scan"),
            ("ADC-BQ", "lut", raw_value(
                "ADC-BQ int16 nibble", "lut_build_pack"
            ), "int16 4D LUT build and hacc pack"),
            ("SQ4", "non_lut", raw_value("SQ4", "method_total"),
             "query pointer binding"),
            ("PQ8-M128", "lut", raw_value("PQ M128x8 dsub8", "method_total"),
             "M128x256 inner-product LUT"),
            ("PQ8-M256", "lut", raw_value("PQ M256x8 dsub4", "method_total"),
             "M256x256 inner-product LUT"),
            ("OPQ-PQ8-M128", "non_lut", raw_value("OPQ M128x8", "opq_rotation"),
             "1024x1024 trained OPQ rotation"),
            ("OPQ-PQ8-M128", "lut", raw_value(
                "OPQ M128x8", "pq_lut_build_isolated"
            ), "M128x256 PQ LUT on pre-rotated query"),
            ("TurboQuant TQ+ 1b", "non_lut", raw_value(
                "TurboQuant TQ+ 1b", "method_total"
            ), "Hadamard rotations plus TQ+ correction and query packing"),
            ("RaBitQ-1b C16", "non_lut", raw_value(
                "RaBitQ-1b C16", "method_total"
            ), "FHT plus query pack plus C16 metadata"),
        )
        for method, component, latency, detail in fixed_components:
            component_rows.append(
                {
                    "dataset": workload.key,
                    "method": method,
                    "nprobe": "",
                    "component": component,
                    "median_us_per_query": latency,
                    "detail": detail,
                }
            )

        transform = raw_value("MRQ B320 C4096", "dense_transform")
        for probe in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            routing = raw_value("MRQ B320 C4096", "ivf_centroid_routing", probe)
            lut = raw_value("MRQ B320 C4096", "probe_residual_luts", probe)
            component_rows.extend(
                (
                    {
                        "dataset": workload.key,
                        "method": "MRQ B320 C4096",
                        "nprobe": probe,
                        "component": "non_lut",
                        "median_us_per_query": transform + routing,
                        "detail": "dense transform plus C4096 routing",
                    },
                    {
                        "dataset": workload.key,
                        "method": "MRQ B320 C4096",
                        "nprobe": probe,
                        "component": "lut",
                        "median_us_per_query": lut,
                        "detail": "per-probe residual LUTs",
                    },
                )
            )

    component_csv = output_dir / "query_preprocess_lut_breakdown.csv"
    write_csv(component_csv, component_rows)

    component_order = (
        "ADC-BQ",
        "SQ4",
        "PQ8-M128",
        "PQ8-M256",
        "OPQ-PQ8-M128",
        "TurboQuant TQ+ 1b",
        "RaBitQ-1b C16",
    )
    component_labels = (
        "Calibrated\nADC-BQ",
        "SQ4",
        "PQ8\nM128",
        "PQ8\nM256",
        "OPQ-PQ8\nM128",
        "TurboQuant\nTQ+ 1b",
        "RaBitQ-1b\nC16",
    )
    breakdown_figure, breakdown_axes = plt.subplots(2, 2, figsize=(14.8, 9.2))
    for row_index, workload in enumerate(WORKLOADS):
        selected_components = [
            row for row in component_rows if row["dataset"] == workload.key
        ]
        fixed = {
            (str(row["method"]), str(row["component"])): float(
                row["median_us_per_query"]
            )
            for row in selected_components
            if row["method"] != "MRQ B320 C4096"
        }
        non_lut = [fixed.get((method, "non_lut"), 0.0) for method in component_order]
        lut = [fixed.get((method, "lut"), 0.0) for method in component_order]
        axis = breakdown_axes[row_index, 0]
        positions = list(range(len(component_order)))
        width = 0.36
        non_lut_bars = axis.bar(
            [position - width / 2 for position in positions],
            non_lut,
            width,
            color="#1f77b4",
            label="non-LUT query setup",
        )
        lut_bars = axis.bar(
            [position + width / 2 for position in positions],
            lut,
            width,
            color="#ff7f0e",
            label="LUT build/pack",
        )
        axis.set_yscale("log")
        axis.set_ylim(0.0015, 3000)
        axis.set_xticks(positions, component_labels)
        axis.set_ylabel("µs/query (isolated median; log scale)")
        axis.set_title(f"{workload.key}: LUT separated")
        axis.grid(True, axis="y", which="both", alpha=0.25)
        axis.legend(fontsize=9)
        for bars, values in ((non_lut_bars, non_lut), (lut_bars, lut)):
            for bar, value in zip(bars, values):
                if value <= 0:
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.12,
                    f"{value:.3f}" if value < 0.1 else f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if value >= 0.1 else 0,
                )

        mrq_non_lut = sorted(
            [
                row for row in selected_components
                if row["method"] == "MRQ B320 C4096"
                and row["component"] == "non_lut"
            ],
            key=lambda row: int(row["nprobe"]),
        )
        mrq_lut = sorted(
            [
                row for row in selected_components
                if row["method"] == "MRQ B320 C4096"
                and row["component"] == "lut"
            ],
            key=lambda row: int(row["nprobe"]),
        )
        axis = breakdown_axes[row_index, 1]
        probes = [int(row["nprobe"]) for row in mrq_non_lut]
        non_lut_values = [float(row["median_us_per_query"]) for row in mrq_non_lut]
        lut_values = [float(row["median_us_per_query"]) for row in mrq_lut]
        axis.plot(
            probes, non_lut_values, marker="o", color="#1f77b4",
            label="transform + C4096 routing",
        )
        axis.plot(
            probes, lut_values, marker="s", color="#ff7f0e",
            label="per-probe LUT build",
        )
        axis.plot(
            probes,
            [base + table for base, table in zip(non_lut_values, lut_values)],
            marker="^",
            color="#2ca02c",
            label="sum",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("MRQ nprobe")
        axis.set_ylabel("µs/query (median; log scale)")
        axis.set_title(f"{workload.key}: MRQ component split")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=9)

    breakdown_figure.suptitle(
        "Qwen-1024 query setup: transform/quantization vs LUT — T=1 CPU-pinned",
        fontsize=15,
    )
    breakdown_figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    breakdown_figure.text(
        0.5,
        0.012,
        (
            "Bars are isolated component timings and are not forced to sum to the production end-to-end median. "
            "Zero-height means that method has no separate component of that type."
        ),
        ha="center",
        fontsize=9,
    )
    breakdown_path = output_dir / "query_preprocess_lut_breakdown.png"
    breakdown_figure.savefig(breakdown_path, dpi=180, bbox_inches="tight")
    breakdown_figure.savefig(breakdown_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(breakdown_figure)
    return derived_csv, figure_path, component_csv, breakdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--fixed-iterations", type=int, default=6000)
    parser.add_argument("--mrq-iterations", type=int, default=512)
    parser.add_argument("--tag", default="clean_20261110_cpu20")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_binary()

    raw_paths: list[Path] = []
    for workload in WORKLOADS:
        output = output_dir / f"{workload.key}.raw.csv"
        raw_paths.append(output)
        if output.exists() and not args.force:
            print(f"REUSE {output}", flush=True)
        else:
            benchmark(workload, output, args)

    raw_rows = load_rows(raw_paths)
    combined = output_dir / "query_preprocess_overhead_raw.csv"
    write_csv(combined, raw_rows)
    derived, figure, component_csv, breakdown_figure = derive(raw_rows, output_dir)
    manifest = {
        "created": datetime.now().astimezone().isoformat(),
        "cpu": args.cpu,
        "threads": 1,
        "rounds": args.rounds,
        "fixed_iterations": args.fixed_iterations,
        "mrq_iterations": args.mrq_iterations,
        "model_sources": {
            "OPQ-PQ8-M128": str(OPQ_INDEX),
            "TurboQuant": (
                "Qdrant canonical TQ+ Bits1 model fit outside timing on each "
                "workload query set; timing matches production set_query by "
                "calling encode_query and export_bits1_wide_planes"
            ),
        },
        "definitions": {
            "ADC-BQ": "calibrated production int16+nibble set_query, hacc byte pack, and tau-width scale scan",
            "RaBitQ": "production FHT rotation + SplitSingleQuery 4-bit pack + C16 metadata and temporaries",
            "PQ": "Faiss PQ M=256, nbits=8, dsub=4 inner-product LUT",
            "PQ-M128": "Faiss production IndexPQ M=128, nbits=8 set_query LUT using the trained OPQ index codebook",
            "SQ4": "Faiss production IndexScalarQuantizer QT_4bit set_query; pointer binding only and no query LUT",
            "OPQ": "Faiss production OPQ8-M128 apply_chain allocation and rotation plus the inner PQ set_query LUT",
            "TurboQuant": "Qdrant production TQ+ Bits1 encode_query with padded Hadamard rotation and EC correction, followed by AVX-512 wide-plane query packing",
            "MRQ": "production 1024-D PCA + B320 rotation + C4096 routing + per-probe residual 4-bit LUT",
        },
        "excluded": [
            "common query normalization",
            "ANN list/graph traversal after routing",
            "exact reranking",
            "SSD I/O",
        ],
        "outputs": [
            str(combined),
            str(derived),
            str(figure),
            str(component_csv),
            str(breakdown_figure),
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
