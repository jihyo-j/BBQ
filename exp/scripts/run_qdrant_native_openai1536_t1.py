#!/usr/bin/env python3
"""Native-Qdrant TurboQuant calibration/fixed-R benchmark on OpenAI-1536.

The runner reuses the exact seed-11 calibration/test splits from the ADC-BQ
experiment, builds a TurboQuant-1bit native Qdrant HNSW, evaluates both the
per-ef calibrated adaptive policy and a Pareto-selected fixed-R policy, then
combines those curves with the matching ADC-BQ results.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BINARY = Path("/tmp/qdrant-native-check-target/release/qdrant_native_hnsw_bench")
RESULT_ROOT = REPO / "experiment/qdrant_native_openai1536_t1"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/qdrant_native_openai1536_t1")
STATUS = RESULT_ROOT / "status.json"
EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
RERANKS = (10, 20, 30, 40, 60, 80, 120, 160, 200)

DATASETS = (
    {
        "key": "sieun_dbpedia_entity_openai1536",
        "corpus": "DBpedia",
        "model": "OpenAI text-embedding-3-small",
        "train": Path(
            "/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1/data/"
            "sieun_dbpedia_entity_openai1536/"
            "sieun_dbpedia_entity_openai1536.train.fbin"
        ),
        "fp_sidecar": Path(
            "/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1/data/"
            "sieun_dbpedia_entity_openai1536/"
            "sieun_dbpedia_entity_openai1536.train.row_aligned4k.fbin"
        ),
        "split": REPO / (
            "experiment/openai_beir_11methods_t1/four_methods/"
            "sieun_dbpedia_entity_openai1536/"
            "adcbq_pool_top10_target99_per_ef_raw/splits/"
            "sieun_dbpedia_entity_openai1536/seed11.cal200.test200"
        ),
        "adc": REPO / (
            "experiment/openai_beir_11methods_t1/four_methods/"
            "sieun_dbpedia_entity_openai1536/summary_t1.csv"
        ),
    },
    {
        "key": "sieun_msmarco_openai1536",
        "corpus": "MSMARCO",
        "model": "OpenAI text-embedding-3-small",
        "train": Path(
            "/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1/data/"
            "sieun_msmarco_openai1536/sieun_msmarco_openai1536.train.fbin"
        ),
        "fp_sidecar": Path(
            "/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1/data/"
            "sieun_msmarco_openai1536/"
            "sieun_msmarco_openai1536.train.row_aligned4k.fbin"
        ),
        "split": REPO / (
            "experiment/openai_beir_11methods_t1/four_methods/"
            "sieun_msmarco_openai1536/"
            "adcbq_pool_top10_target99_per_ef_raw/splits/"
            "sieun_msmarco_openai1536/seed11.cal1000.test5980"
        ),
        "adc": REPO / (
            "experiment/openai_beir_11methods_t1/four_methods/"
            "sieun_msmarco_openai1536/summary_t1.csv"
        ),
    },
)


def update_status(**updates: object) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if STATUS.is_file():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    if updates.get("stage") in {"running", "dataset_complete", "complete"}:
        payload.pop("error", None)
    payload.update(updates)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
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
    os.replace(temporary, path)


def summary_complete(path: Path, fixed_only: bool = False) -> bool:
    if not path.is_file():
        return False
    rows = read_csv(path)
    methods = {row.get("method_key") for row in rows}
    points = {(row.get("method_key"), int(row["efSearch"])) for row in rows}
    expected_methods = (
        {"qdrant_native_tq1_fixed_r"}
        if fixed_only
        else {"qdrant_native_tq1_calibrated", "qdrant_native_tq1_fixed_r"}
    )
    expected_points = len(EFS) if fixed_only else 2 * len(EFS)
    return (
        methods == expected_methods
        and len(points) == expected_points
        and all(
            row.get("io_engine") == "O_DIRECT io_uring one candidate batch per query"
            for row in rows
        )
    )


def ensure_fp_sidecar(spec: dict[str, object]) -> Path:
    source = Path(spec["train"])
    target = Path(spec["fp_sidecar"])
    with source.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise RuntimeError(f"invalid fbin header: {source}")
    rows = int.from_bytes(header[:4], "little", signed=True)
    dim = int.from_bytes(header[4:], "little", signed=True)
    expected = 4096 + rows * dim * 4
    if target.is_file() and target.stat().st_size == expected:
        print(f"REUSE FP SIDECAR {target}", flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    print(f"CREATE FP SIDECAR {target}", flush=True)
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        input_header = input_handle.read(8)
        output_handle.write(input_header)
        output_handle.write(bytes(4096 - len(input_header)))
        shutil.copyfileobj(input_handle, output_handle, length=64 * 1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if temporary.stat().st_size != expected:
        raise RuntimeError(
            f"FP sidecar size mismatch: expected {expected}, got {temporary.stat().st_size}"
        )
    os.replace(temporary, target)
    return target


def run_dataset(spec: dict[str, object]) -> Path:
    key = str(spec["key"])
    result_dir = RESULT_ROOT / key
    summary = result_dir / "summary_t1.csv"
    fixed_only = bool(spec.get("fixed_only", False))
    if summary_complete(summary, fixed_only=fixed_only):
        print(f"SKIP complete {key}", flush=True)
        return summary

    fp_sidecar = ensure_fp_sidecar(spec)
    index_root = EXTERNAL_ROOT / key / "index"
    if index_root.exists():
        resolved = index_root.resolve()
        if EXTERNAL_ROOT.resolve() not in resolved.parents:
            raise RuntimeError(f"refusing cleanup outside owned root: {resolved}")
        print(f"REMOVE incomplete owned index {resolved}", flush=True)
        shutil.rmtree(resolved)

    result_dir.mkdir(parents=True, exist_ok=True)
    split = Path(spec["split"])
    command = [
        str(BINARY),
        "--dataset", key,
        "--train", str(spec["train"]),
        "--fp-sidecar", str(fp_sidecar),
        "--cal-query", str(Path(f"{split}.cal.query.fbin")),
        "--cal-gt", str(Path(f"{split}.cal.gt.ibin")),
        "--test-query", str(Path(f"{split}.test.query.fbin")),
        "--test-gt", str(Path(f"{split}.test.gt.ibin")),
        "--root", str(index_root),
        "--calibration-out", str(result_dir / "calibration_sweep.csv"),
        "--selection-out", str(result_dir / "fixed_r_selection.csv"),
        "--out", str(summary),
        "--metric", "cosine",
        "--m", "24",
        "--ef-construction", "200",
        "--build-threads", "24",
        "--k", "10",
        "--rounds", "3",
        "--pilot-rounds", "1",
        "--target-recall", "0.99",
        "--efs", ",".join(map(str, EFS)),
        "--reranks", ",".join(map(str, RERANKS)),
    ]
    if fixed_only:
        command.append("--fixed-only")
    if bool(spec.get("build_graph_fp", False)):
        command.append("--build-graph-fp")
    if bool(spec.get("stream_base", False)):
        command.append("--stream-base")
    update_status(stage="running", dataset=key, command=command)
    print("RUN " + " ".join(command), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=REPO, env=environment, check=True)
    if not summary_complete(summary, fixed_only=fixed_only):
        raise RuntimeError(f"incomplete native Qdrant result: {summary}")
    update_status(stage="dataset_complete", dataset=key, summary=str(summary))
    print(f"DATASET_COMPLETE {key} {summary}", flush=True)
    return summary


def combine(summaries: dict[str, Path]) -> Path:
    rows: list[dict[str, object]] = []
    for spec in DATASETS:
        key = str(spec["key"])
        for row in read_csv(Path(spec["adc"])):
            if row.get("method_key") != "adc_bq_calibrated":
                continue
            rows.append(
                {
                    "dataset": key,
                    "corpus": spec["corpus"],
                    "model": spec["model"],
                    "method_key": row["method_key"],
                    "method": row["method"],
                    "efSearch": int(row["search_parameter_value"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "rerank_count_mean": float(row["fp_vectors_per_query"]),
                    "rerank_policy": row["calibration_protocol"],
                    "calibration_queries": int(row["calibration_queries"]),
                    "test_queries": int(row["test_queries"]),
                    "storage": row["storage"],
                    "io_engine": row["io_engine"],
                    "fp_4k_pages_per_query": float(row["fp_4k_pages_per_query"]),
                    "source_csv": str(Path(spec["adc"]).relative_to(REPO)),
                }
            )
        for row in read_csv(summaries[key]):
            rows.append(
                {
                    "dataset": key,
                    "corpus": spec["corpus"],
                    "model": spec["model"],
                    "method_key": row["method_key"],
                    "method": row["method"],
                    "efSearch": int(row["efSearch"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "rerank_count_mean": float(row["rerank_count_mean"]),
                    "rerank_policy": row["rerank_policy"],
                    "selected_r": row["selected_r"],
                    "tau": row["tau"],
                    "calibration_queries": int(row["calibration_queries"]),
                    "test_queries": int(row["nq"]),
                    "storage": "SSD O_DIRECT FP sidecar; resident TQ1 codes and native Qdrant HNSW",
                    "io_engine": row["io_engine"],
                    "fp_4k_pages_per_query": float(row["fp_4k_pages_per_query"]),
                    "source_csv": str(summaries[key].relative_to(REPO)),
                }
            )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["method_key"]), int(row["efSearch"])))
    output = RESULT_ROOT / "adcbq_vs_qdrant_native_tq1_openai1536_t1.csv"
    write_csv(output, rows)
    return output


def plot(combined: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    rows = read_csv(combined)
    labels = {
        "adc_bq_calibrated": "ADC-BQ calibrated",
        "qdrant_native_tq1_calibrated": "Qdrant native TQ1 calibrated",
        "qdrant_native_tq1_fixed_r": "Qdrant native TQ1 fixed R",
    }
    styles = {
        "adc_bq_calibrated": ("o", "#0072B2"),
        "qdrant_native_tq1_calibrated": ("s", "#D55E00"),
        "qdrant_native_tq1_fixed_r": ("^", "#009E73"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=False)
    for axis, spec in zip(axes, DATASETS, strict=True):
        key = str(spec["key"])
        for method_key, label in labels.items():
            selected = [
                row for row in rows
                if row["dataset"] == key and row["method_key"] == method_key
            ]
            selected.sort(key=lambda row: float(row["recall_at_10"]))
            marker, color = styles[method_key]
            axis.plot(
                [float(row["recall_at_10"]) for row in selected],
                [float(row["qps"]) for row in selected],
                marker=marker,
                markersize=3.5,
                linewidth=1.4,
                color=color,
                label=label,
            )
        axis.axvline(0.99, color="0.55", linestyle="--", linewidth=0.9)
        axis.set_title(str(spec["corpus"]))
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("QPS (T=1)")
        axis.grid(True, alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    png = RESULT_ROOT / "adcbq_vs_qdrant_native_tq1_openai1536_t1.png"
    pdf = RESULT_ROOT / "adcbq_vs_qdrant_native_tq1_openai1536_t1.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    if not BINARY.is_file():
        raise FileNotFoundError(BINARY)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Path] = {}
    try:
        for spec in DATASETS:
            summaries[str(spec["key"])] = run_dataset(spec)
        combined = combine(summaries)
        png, pdf = plot(combined)
        update_status(
            stage="complete",
            dataset=None,
            combined_csv=str(combined),
            figure_png=str(png),
            figure_pdf=str(pdf),
        )
        print(f"COMPLETE {combined} {png} {pdf}", flush=True)
    except BaseException as error:
        update_status(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
