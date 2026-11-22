#!/usr/bin/env python3
"""Clean T=1 rerun of calibrated ADC-BQ/SSD and resident FP32 HNSW."""

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
EXPERIMENT = REPO / "experiment"
ADC_BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_calibrated"
FP_BINARY = EXPERIMENT / "faiss_fp_hnsw_resident_from_binary_graph"
SPLIT_ROOT = EXPERIMENT / "adcbq_calibrated_threshold_overlap/splits"
OUTPUT_ROOT = EXPERIMENT / "calibrated_adcbq_hnsw_ssd_vs_fp_hnsw_ram_t1"

ADC_EFS = (64, 128, 256, 512, 1024)
FP_EFS = (
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    512,
    640,
    768,
    1024,
)


@dataclass(frozen=True)
class Workload:
    key: str
    title: str
    train: Path
    graph: Path
    split_stem: str
    calibration: Path
    rounds: int
    sidecar: Path

    @property
    def query(self) -> Path:
        return SPLIT_ROOT / self.key / f"{self.split_stem}.test.query.fbin"

    @property
    def gt(self) -> Path:
        return SPLIT_ROOT / self.key / f"{self.split_stem}.test.gt.ibin"


WORKLOADS = (
    Workload(
        key="msmarco_qwen1024",
        title="MSMARCO Qwen-1024 (1M)",
        train=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_train.fbin"
        ),
        graph=Path(
            "/home/smrc/samsung-nvme/jihyo/msmarco_qwen1024_scale_v014/"
            "indexes/faiss_adcbq/msmarco-qwen1024-1m.fpgraph_pgbit_binary_hnsw."
            "n1000000.d1024.M24.efC200.faiss"
        ),
        split_stem="seed11.cal1000.test5980",
        calibration=EXPERIMENT
        / "adcbq_vs_fp_hnsw_fullvec_ram/clean_t1_cpu20_20261109/"
        "msmarco_qwen1024/calibration_target099.json",
        rounds=5,
        sidecar=Path(
            "/home/smrc/samsung-nvme/jihyo/adcbq_aligned4k_fp_transient/"
            "msmarco_qwen1024.fp32.aligned4k.fbin"
        ),
    ),
    Workload(
        key="dbpedia_qwen1024",
        title="DBpedia Qwen-1024 (1M)",
        train=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "dbpedia-entity__vanilla-qwen06b_train.fbin"
        ),
        graph=REPO
        / "indexes/dbpedia_graph_build_bq_adcbq/"
        "dbpedia_qwen1024.fp_graph_bq_storage.n1000000.d1024.M24.efC200.faiss",
        split_stem="seed11.cal100.test300",
        calibration=EXPERIMENT
        / "adcbq_vs_fp_hnsw_fullvec_ram/clean_t1_cpu20_20261109/"
        "dbpedia_qwen1024/calibration_target099.json",
        rounds=20,
        sidecar=Path(
            "/home/smrc/samsung-nvme/jihyo/adcbq_aligned4k_fp_transient/"
            "dbpedia_qwen1024.fp32.aligned4k.fbin"
        ),
    ),
)


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_PROC_BIND": "true",
        }
    )
    subprocess.run(command, cwd=REPO, env=env, check=True)


def calibration_fit(workload: Workload, ef: int) -> dict[str, object]:
    payload = json.loads(workload.calibration.read_text(encoding="utf-8"))
    if payload.get("protocol") != "per_ef_pool_top10_split_conformal":
        raise RuntimeError(f"unexpected calibration protocol: {workload.calibration}")
    if float(payload.get("target", 0.0)) != 0.99:
        raise RuntimeError(f"unexpected calibration target: {workload.calibration}")
    return payload["operating_points"][str(ef)]


def adc_command(
    workload: Workload, ef: int, fit: dict[str, object], output: Path
) -> list[str]:
    return [
        str(ADC_BINARY),
        "--dataset",
        workload.key,
        "--train",
        str(workload.train),
        "--query",
        str(workload.query),
        "--gt",
        str(workload.gt),
        "--index",
        str(workload.graph),
        "--out",
        str(output),
        "--m",
        "24",
        "--ef-construction",
        "200",
        "--k",
        "10",
        "--max-queries",
        "0",
        "--rounds",
        str(workload.rounds),
        "--build-threads",
        "1",
        "--search-threads",
        "1",
        "--adc-mode",
        "int16",
        "--lut-mode",
        "nibble",
        "--graph-build",
        "fp",
        "--distance-mode",
        "adcbq",
        "--search-impl",
        "inline",
        "--code-layout",
        "block32",
        "--efs",
        str(ef),
        "--rerank-n",
        str(ef),
        "--rerank-dot",
        "fp32",
        "--fp-rerank-source",
        "odirect-uring",
        "--fp-rerank-norms",
        "onfetch",
        "--fp-rerank-path",
        str(workload.sidecar),
        "--fp-rerank-overlap-queries",
        "--fp-rerank-qd",
        str(ef),
        "--calibrated-threshold-rerank-tau",
        repr(float(fit["tau"])),
    ]


def fp_command(workload: Workload, output: Path) -> list[str]:
    return [
        str(FP_BINARY),
        "--dataset",
        workload.key,
        "--train",
        str(workload.train),
        "--query",
        str(workload.query),
        "--gt",
        str(workload.gt),
        "--graph",
        str(workload.graph),
        "--out",
        str(output),
        "--m",
        "24",
        "--ef-construction",
        "200",
        "--k",
        "10",
        "--rounds",
        str(workload.rounds),
        "--threads",
        "1",
        "--efs",
        *[str(ef) for ef in FP_EFS],
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def qps_median(row: dict[str, str]) -> float:
    value = row.get("qps_median", "")
    return float(value) if value else float(row["qps"])


def plot_and_combine(output_dir: Path) -> tuple[Path, Path, Path]:
    import matplotlib.pyplot as plt

    combined: list[dict[str, object]] = []
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for axis, workload in zip(axes, WORKLOADS):
        adc_rows = []
        for ef in ADC_EFS:
            rows = read_rows(
                output_dir
                / workload.key
                / "adcbq_ssd_per_ef_target099"
                / f"ef{ef}.csv"
            )
            if len(rows) != 1:
                raise RuntimeError(f"expected one ADC row for {workload.key} ef={ef}")
            adc_rows.append(rows[0])
        fp_rows = read_rows(output_dir / workload.key / "fp_hnsw_ram.csv")

        axis.plot(
            [float(row["recall_at_k"]) for row in adc_rows],
            [qps_median(row) for row in adc_rows],
            marker="o",
            linewidth=2.3,
            markersize=6.5,
            color="#d62728",
            label="Calibrated ADC-BQ HNSW + SSD FP32 rerank",
        )
        axis.plot(
            [float(row["recall_at_k"]) for row in fp_rows],
            [qps_median(row) for row in fp_rows],
            marker="s",
            linewidth=2.3,
            markersize=5.5,
            color="#1f77b4",
            label="Full FP32 HNSW (in-memory)",
        )
        for method, rows, storage in (
            ("Calibrated ADC-BQ HNSW + SSD FP32 rerank", adc_rows, "SSD O_DIRECT"),
            ("Full FP32 HNSW (in-memory)", fp_rows, "process_owned_ram"),
        ):
            for row in rows:
                ef = int(row["efSearch"])
                fit = calibration_fit(workload, ef) if storage == "SSD O_DIRECT" else None
                combined.append(
                    {
                        "dataset": workload.key,
                        "method": method,
                        "efSearch": ef,
                        "threads": 1,
                        "rounds": int(row["rounds"]),
                        "queries": int(row["nq"]),
                        "recall_at_10": float(row["recall_at_k"]),
                        "qps_median": qps_median(row),
                        "qps_mean": float(row["qps"]),
                        "pages_4k_per_query": (
                            float(row["direct_requests_mean"]) / int(row["nq"])
                            if storage == "SSD O_DIRECT"
                            else 0.0
                        ),
                        "full_vector_storage": storage,
                        "calibration_target": 0.99 if fit else "",
                        "calibration_tau": float(fit["tau"]) if fit else "",
                        "calibration_query_level_all_top10_coverage": (
                            float(fit["empirical_all_top10_coverage"]) if fit else ""
                        ),
                    }
                )

        axis.set_title(workload.title)
        axis.set_xlabel("Recall@10")
        axis.set_xlim(0.96, 1.0)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.28)
        axis.legend(fontsize=9)

    axes[0].set_ylabel("End-to-end median QPS (log scale)")
    figure.suptitle(
        "Calibrated ADC-BQ HNSW SSD vs in-memory FP32 HNSW — clean T=1 rerun",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.94))
    figure.text(
        0.5,
        0.018,
        (
            "Same FP-built HNSW topology (M=24, efConstruction=200), held-out split, "
            "and pinned CPU 20. ADC-BQ: io_uring O_DIRECT SSD; FP-HNSW: process-owned RAM."
        ),
        ha="center",
        fontsize=9.2,
    )
    png = output_dir / "recall_qps_t1_from_recall096_rerun.png"
    pdf = output_dir / "recall_qps_t1_from_recall096_rerun.pdf"
    csv_path = output_dir / "recall_qps_t1_from_recall096_rerun.csv"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    return png, pdf, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-tag", default="rerun_20261110_cpu20")
    parser.add_argument("--cpu", type=int, default=20)
    parser.add_argument("--stage", choices=("all", "adc", "fp", "plot"), default="all")
    args = parser.parse_args()
    if Path(args.result_tag).name != args.result_tag:
        raise SystemExit("--result-tag must be one directory name")
    os.sched_setaffinity(0, {args.cpu})
    output_dir = OUTPUT_ROOT / args.result_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    started_load = os.getloadavg()

    if args.stage in ("all", "adc"):
        for workload in WORKLOADS:
            for ef in ADC_EFS:
                fit = calibration_fit(workload, ef)
                output = (
                    output_dir
                    / workload.key
                    / "adcbq_ssd_per_ef_target099"
                    / f"ef{ef}.csv"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                run(adc_command(workload, ef, fit, output))
    if args.stage in ("all", "fp"):
        for workload in WORKLOADS:
            output = output_dir / workload.key / "fp_hnsw_ram.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            run(fp_command(workload, output))
    figures = None
    if args.stage in ("all", "plot"):
        figures = [str(path) for path in plot_and_combine(output_dir)]

    manifest = {
        "protocol": "calibrated_adcbq_ssd_vs_fp_hnsw_ram_clean_t1_rerun",
        "created": datetime.now().astimezone().isoformat(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average_start": started_load,
        "load_average_end": os.getloadavg(),
        "adc_efs": ADC_EFS,
        "fp_efs": FP_EFS,
        "rounds": {workload.key: workload.rounds for workload in WORKLOADS},
        "qps_statistic": "median across rounds",
        "M": 24,
        "efConstruction": 200,
        "same_topology": True,
        "calibration_protocol": "per_ef_pool_top10_split_conformal",
        "calibration_target": 0.99,
        "calibration_event": "all exact-FP top-10 within each ef candidate pool",
        "sidecars": {workload.key: str(workload.sidecar) for workload in WORKLOADS},
        "outputs": figures,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
