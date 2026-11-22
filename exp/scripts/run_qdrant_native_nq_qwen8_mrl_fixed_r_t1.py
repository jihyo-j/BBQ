#!/usr/bin/env python3
"""Benchmark Pareto-selected fixed-R TurboQuant on native Qdrant NQ MRL.

The four dimensions are normalized prefixes of the same Qwen3-Embedding-8B
vectors.  Each dimension uses its own exact ground truth and the existing
seed-11 calibration/test split (1,000/2,452 queries).  A joint efSearch/R
calibration sweep selects one fixed R per dimension at Recall@10 >= 0.99;
that R is then held fixed for the full held-out efSearch sweep.
"""

from __future__ import annotations

import csv
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiment"))

import run_qdrant_native_openai1536_t1 as native  # noqa: E402


RESULT_ROOT = REPO / "experiment/qdrant_native_nq_qwen8_mrl_fixed_r_t1"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/qdrant_native_nq_qwen8_mrl_fixed_r_t1"
)
STATUS = RESULT_ROOT / "status.json"
DATA_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_qwen8_mrl_four_methods_t1/data")
SOURCE_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq")
SOURCE_FP_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp")
SPLIT_ROOT = REPO / "experiment/four_methods_scalability_mrl_t1/nq_mrl"

DIMS = (512, 1024, 2560, 4096)
EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
RERANKS = (10, 20, 30, 40, 60, 80, 120, 160, 200)


def dataset_spec(dim: int) -> dict[str, object]:
    key = f"nq_qwen8_mrl{dim}"
    if dim == 4096:
        train = SOURCE_ROOT / "nq_vanilla-qw8_train.fbin"
        fp_sidecar = SOURCE_FP_ROOT / "nq_qwen8_4096.fp32.header4096.fbin"
    else:
        data = DATA_ROOT / key
        train = data / f"{key}.train.header4096.fbin"
        fp_sidecar = train
    source_split = (
        SPLIT_ROOT / key / "adcbq_pool_top10_target99_per_ef_raw/splits"
        / key / "seed11.cal1000.test2452"
    )
    split = RESULT_ROOT / "splits" / key / "seed11.cal1000.test2452.top10"
    return {
        "key": key,
        "corpus": "NQ",
        "model": f"Qwen3-Embedding-8B MRL-{dim}",
        "dim": dim,
        "train": train,
        "fp_sidecar": fp_sidecar,
        "split": split,
        "source_split": source_split,
        "fixed_only": True,
    }


DATASETS = tuple(dataset_spec(dim) for dim in DIMS)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def update_status(**updates: object) -> None:
    payload: dict[str, object] = {}
    if STATUS.is_file():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(updates)
    payload["updated_at"] = now()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def matrix_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"invalid matrix header: {path}")
    return struct.unpack("<ii", raw)


def prepare_top10_splits() -> None:
    """Reuse query files and convert the existing top-100 GT to top-10."""
    for spec in DATASETS:
        source = Path(spec["source_split"])
        target = Path(spec["split"])
        target.parent.mkdir(parents=True, exist_ok=True)
        for phase, rows in (("cal", 1000), ("test", 2452)):
            source_query = Path(f"{source}.{phase}.query.fbin")
            target_query = Path(f"{target}.{phase}.query.fbin")
            if not target_query.exists():
                target_query.symlink_to(source_query)
            source_gt = Path(f"{source}.{phase}.gt.ibin")
            target_gt = Path(f"{target}.{phase}.gt.ibin")
            if target_gt.is_file() and matrix_header(target_gt) == (rows, 10):
                continue
            if matrix_header(source_gt) != (rows, 100):
                raise RuntimeError(f"unexpected source GT shape: {source_gt}")
            values = np.memmap(
                source_gt,
                dtype=np.int32,
                mode="r",
                offset=8,
                shape=(rows, 100),
            )
            temporary = target_gt.with_suffix(target_gt.suffix + ".tmp")
            temporary.unlink(missing_ok=True)
            with temporary.open("wb") as handle:
                handle.write(struct.pack("<ii", rows, 10))
                np.ascontiguousarray(values[:, :10]).tofile(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target_gt)


def validate_inputs() -> None:
    if not native.BINARY.is_file():
        raise FileNotFoundError(native.BINARY)
    for spec in DATASETS:
        dim = int(spec["dim"])
        train = Path(spec["train"])
        fp_sidecar = Path(spec["fp_sidecar"])
        if not train.is_file():
            raise FileNotFoundError(train)
        if not fp_sidecar.is_file():
            raise FileNotFoundError(fp_sidecar)
        if matrix_header(train) != (1_000_000, dim):
            raise RuntimeError(f"unexpected train shape: {train} {matrix_header(train)}")
        if matrix_header(fp_sidecar) != (1_000_000, dim):
            raise RuntimeError(
                f"unexpected FP sidecar shape: {fp_sidecar} {matrix_header(fp_sidecar)}"
            )
        split = Path(spec["split"])
        expected = {
            ".cal.query.fbin": (1000, dim),
            ".cal.gt.ibin": (1000, 10),
            ".test.query.fbin": (2452, dim),
            ".test.gt.ibin": (2452, 10),
        }
        for suffix, shape in expected.items():
            path = Path(f"{split}{suffix}")
            if not path.is_file():
                raise FileNotFoundError(path)
            if matrix_header(path) != shape:
                raise RuntimeError(f"unexpected split shape: {path} {matrix_header(path)}")


def combine_fixed(summaries: dict[str, Path]) -> tuple[Path, Path]:
    curves: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for spec in DATASETS:
        key = str(spec["key"])
        selection_path = RESULT_ROOT / key / "fixed_r_selection.csv"
        selection_rows = read_csv(selection_path)
        if len(selection_rows) != 1:
            raise RuntimeError(f"invalid fixed-R selection: {selection_path}")
        selection = selection_rows[0]
        if selection["target_reached"].lower() != "true":
            raise RuntimeError(f"target not reached during calibration: {selection_path}")
        selections.append(
            {
                "dataset": key,
                "dim": int(spec["dim"]),
                "selected_r": int(selection["selected_r"]),
                "selection_ef": int(selection["selected_ef"]),
                "calibration_recall_at_10": float(selection["selected_recall_at_10"]),
                "calibration_qps": float(selection["selected_qps"]),
                "target_recall": float(selection["target_recall"]),
                "selection_protocol": selection["protocol"],
                "calibration_queries": int(selection["calibration_queries"]),
                "rerank_sweep": selection["rerank_sweep"],
                "ef_sweep": selection["ef_sweep"],
                "source_csv": str(selection_path.relative_to(REPO)),
            }
        )
        fixed_rows = [
            row for row in read_csv(summaries[key])
            if row["method_key"] == "qdrant_native_tq1_fixed_r"
        ]
        if len(fixed_rows) != len(EFS):
            raise RuntimeError(f"incomplete fixed-R curve: {summaries[key]}")
        for row in fixed_rows:
            curves.append(
                {
                    "dataset": key,
                    "corpus": "NQ",
                    "model": spec["model"],
                    "dim": int(spec["dim"]),
                    "method_key": row["method_key"],
                    "method": "Qdrant native TurboQuant-1bit + Pareto fixed R",
                    "efSearch": int(row["efSearch"]),
                    "selected_r": int(row["selected_r"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "rerank_count_mean": float(row["rerank_count_mean"]),
                    "fp_4k_pages_per_query": float(row["fp_4k_pages_per_query"]),
                    "calibration_queries": int(row["calibration_queries"]),
                    "test_queries": int(row["nq"]),
                    "rounds": int(row["rounds"]),
                    "search_threads": int(row["search_threads"]),
                    "graph_implementation": "native Qdrant TQ1 HNSW",
                    "io_engine": row["io_engine"],
                    "source_csv": str(summaries[key].relative_to(REPO)),
                }
            )
    curves.sort(key=lambda row: (int(row["dim"]), int(row["efSearch"])))
    selections.sort(key=lambda row: int(row["dim"]))
    curve_path = RESULT_ROOT / "fixed_r_summary_t1.csv"
    selection_path = RESULT_ROOT / "selected_r_by_dimension.csv"
    write_csv(curve_path, curves)
    write_csv(selection_path, selections)
    return curve_path, selection_path


def plot(curve_path: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    rows = read_csv(curve_path)
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    colors = {512: "#0072B2", 1024: "#E69F00", 2560: "#009E73", 4096: "#D55E00"}
    for dim in DIMS:
        selected = [row for row in rows if int(row["dim"]) == dim]
        selected.sort(key=lambda row: int(row["efSearch"]))
        fixed_r = int(selected[0]["selected_r"])
        axis.plot(
            [float(row["recall_at_10"]) for row in selected],
            [float(row["qps"]) for row in selected],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=colors[dim],
            label=f"{dim}D (R={fixed_r})",
        )
    axis.axvline(0.99, color="0.45", linestyle="--", linewidth=0.9)
    axis.set_xlabel("Recall@10")
    axis.set_ylabel("QPS (T=1)")
    axis.set_title("NQ Qwen3-Embedding-8B MRL: native Qdrant TQ1")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    png = RESULT_ROOT / "recall_qps_fixed_r.png"
    pdf = RESULT_ROOT / "recall_qps_fixed_r.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_top10_splits()
    validate_inputs()
    native.RESULT_ROOT = RESULT_ROOT
    native.EXTERNAL_ROOT = EXTERNAL_ROOT
    native.STATUS = STATUS
    native.EFS = EFS
    native.RERANKS = RERANKS
    summaries: dict[str, Path] = {}
    try:
        update_status(
            stage="running",
            datasets=[spec["key"] for spec in DATASETS],
            dimensions=list(DIMS),
            policy="max_qps_on_joint_ef_rerank_pareto",
            target_recall=0.99,
            calibration_queries=1000,
            test_queries=2452,
            efs=list(EFS),
            reranks=list(RERANKS),
            search_threads=1,
            rounds=3,
        )
        for spec in DATASETS:
            summaries[str(spec["key"])] = native.run_dataset(spec)
        curve_path, selection_path = combine_fixed(summaries)
        png, pdf = plot(curve_path)
        update_status(
            stage="complete",
            dataset=None,
            fixed_r_summary=str(curve_path),
            selected_r_by_dimension=str(selection_path),
            figure_png=str(png),
            figure_pdf=str(pdf),
        )
        print(f"COMPLETE {curve_path} {selection_path} {pdf}", flush=True)
    except BaseException as error:
        update_status(stage="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
