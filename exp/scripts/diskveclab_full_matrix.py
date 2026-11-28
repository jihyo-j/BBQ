"""Shared specification for the 16-dataset held-out DiskVecLab matrix."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import run_adcbq_calibrated_threshold_overlap as calibration
import run_v051_ssd_odirect_multidataset as registry


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(
    os.environ.get(
        "DISKVECLAB_RESULT_ROOT",
        str(REPO / "experiment/diskveclab_full_matrix"),
    )
)
SEED = 11

# Calibration sizes preserve the already fitted ADC-BQ protocol.  Every other
# method is evaluated only on the exact remaining test rows.
CALIBRATION_ROWS = {
    "nq_bge_m3_vanilla": 1000,
    "nq_qwen06_1024": 1000,
    "nq_nemotron4096": 1000,
    "nq_e5mistral4096": 1000,
    "quora_bge_m3_vanilla": 1000,
    "quora_qwen1024": 1000,
    "quora_nemotron4096": 1000,
    "quora_e5mistral4096": 1000,
    "msmarco_bge1024": 1000,
    "msmarco_qwen1024": 1000,
    "msmarco_e5mistral4096": 1000,
    "dbpedia_openai1536": 200,
    "dbpedia_bge1024": 200,
    "dbpedia_qwen1024": 100,
    "dbpedia_e5mistral4096": 100,
}


def _load_campaign_selection() -> None:
    manifest_path = os.environ.get("DISKVECLAB_EXTRA_DATASETS_MANIFEST", "").strip()
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for item in payload.get("datasets", []):
            CALIBRATION_ROWS[str(item["key"])] = int(item["calibration_rows"])
    selected_raw = os.environ.get("DISKVECLAB_DATASETS", "").strip()
    if selected_raw:
        selected = [item.strip() for item in selected_raw.split(",") if item.strip()]
        missing = [key for key in selected if key not in CALIBRATION_ROWS]
        if missing:
            raise KeyError(f"unknown DISKVECLAB_DATASETS keys: {missing}")
        chosen = {key: CALIBRATION_ROWS[key] for key in selected}
        CALIBRATION_ROWS.clear()
        CALIBRATION_ROWS.update(chosen)


registry.load_extra_datasets_from_manifest()
_load_campaign_selection()

METHODS = (
    "calibrated_adcbq",
    "diskann_classic",
    "diskann_rabitq_1bit_adaptive_fp",
    "pipeann",
    "spann_pq_fp80",
    "spann_rabitq_1bit_adaptive_fp",
    "rabitq_7bit_ram",
    "elasticsearch_bbq_fp80",
)

COMMON_RESULT_FIELDS = (
    "dataset",
    "corpus",
    "model",
    "method",
    "search_parameter",
    "search_parameter_value",
    "recall_at_10",
    "qps",
    "rounds",
    "search_threads",
    "test_queries",
    "nb",
    "dim",
    "graph_4k_pages_per_query_mean",
    "fp_vectors_per_query_mean",
    "fp_4k_pages_per_query_mean",
    "total_4k_pages_per_query_mean",
    "storage",
    "io_engine",
)


def split_for(key: str) -> calibration.SplitFiles:
    if key not in CALIBRATION_ROWS:
        raise KeyError(f"dataset is not in the full matrix: {key}")
    dataset = registry.DATASETS[key]
    return calibration.make_split(dataset, CALIBRATION_ROWS[key], SEED, 0)


def heldout_dataset(key: str) -> registry.Dataset:
    dataset = registry.DATASETS[key]
    split = split_for(key)
    return replace(dataset, query=split.test_query, gt=split.test_gt)


def write_split_manifest() -> Path:
    rows: list[dict[str, object]] = []
    for key, calibration_rows in CALIBRATION_ROWS.items():
        dataset = registry.DATASETS[key]
        split = split_for(key)
        nb, dim = registry.matrix_shape(dataset.train)
        rows.append(
            {
                "dataset": key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "nb": nb,
                "dim": dim,
                "seed": SEED,
                "calibration_queries": calibration_rows,
                "test_queries": split.test_rows,
                "test_query_path": split.test_query,
                "test_gt_path": split.test_gt,
            }
        )
    path = RESULT_ROOT / "split_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    print(write_split_manifest())
