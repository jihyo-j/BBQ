#!/usr/bin/env python3
"""Join the 101M frozen ADC-BQ and RaBitQ B7 C=16 insertion results."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "dynamic_insert_101m"
ADC = ROOT / "adcbq_online_frozen_1m_tau_vs_rebuild_recalibrated_t32.csv"
RABITQ = ROOT / "rabitq_b7_c16_random_independent_t32.csv"
TRANSFORM = Path(__file__).resolve().parent / "dynamic_insert_8m/vector_transform_c16.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", "nan", "NaN") else float("nan")


def key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["mode"], int(row["corpus_size"]), int(row["efSearch"])


def policy(mode: str) -> str:
    if mode == "online_add_no_rebuild":
        return "frozen_1m_tau"
    if mode == "full_rebuild_oracle":
        return "recalibrated_for_rebuild"
    return "frozen_1m_tau_base"


def operation_seconds(row: dict[str, str]) -> float:
    if row["mode"] == "online_add_no_rebuild":
        return number(row, "add_seconds")
    return number(row, "build_or_rebuild_seconds")


def main() -> None:
    adc_rows = read_csv(ADC)
    rabitq_rows = read_csv(RABITQ)
    adc = {key(row): row for row in adc_rows}
    rabitq = {key(row): row for row in rabitq_rows}
    common = sorted(
        set(adc) & set(rabitq),
        key=lambda item: (item[1], item[0], item[2]),
    )
    if len(common) != 45:
        raise RuntimeError(f"expected 45 matched rows, found {len(common)}")

    comparison: list[dict[str, object]] = []
    for item in common:
        a = adc[item]
        r = rabitq[item]
        adc_recall = number(a, "recall_current_at_k")
        rabitq_recall = number(r, "recall_current_at_k")
        comparison.append(
            {
                "dataset": "msmarco_qwen1024_1m_to_101m",
                "mode": a["mode"],
                "insert_policy": "random_independent",
                "base_size": int(a["base_size"]),
                "insert_size": int(a["insert_size"]),
                "corpus_size": int(a["corpus_size"]),
                "dim": int(a["dim"]),
                "M": int(a["M"]),
                "efConstruction": int(a["efConstruction"]),
                "efSearch": int(a["efSearch"]),
                "k": int(a["k"]),
                "queries": int(a["nq"]),
                "threads": int(a["search_threads"]),
                "adc_calibration_policy": policy(a["mode"]),
                "adc_calibration_target": number(a, "calibration_target"),
                "adc_calibration_tau": number(a, "calibration_tau"),
                "adc_calibration_queries": int(a["calibration_queries"]),
                "adc_candidate_cap": int(a["candidate_cap"]),
                "adc_recall_at_10": adc_recall,
                "rabitq_recall_at_10": rabitq_recall,
                "adc_minus_rabitq_recall": adc_recall - rabitq_recall,
                "adc_minus_rabitq_recall_pp": 100.0 * (adc_recall - rabitq_recall),
                "adc_qps": number(a, "qps"),
                "rabitq_qps": number(r, "qps"),
                "adc_latency_ms_per_query": number(a, "latency_ms_per_query"),
                "rabitq_latency_ms_per_query": number(r, "latency_ms_per_query"),
                "adc_mean_fp_reads": number(a, "physical_reads_per_query"),
                "adc_mean_adaptive_r": number(a, "adaptive_rerank_count_mean"),
                "adc_p95_adaptive_r": number(a, "adaptive_rerank_count_p95"),
                "rabitq_num_clusters": int(r["num_clusters"]),
                "rabitq_total_bits": int(r["total_bits"]),
                "adc_build_or_rebuild_seconds": number(a, "build_or_rebuild_seconds"),
                "rabitq_build_or_rebuild_seconds": number(r, "build_or_rebuild_seconds"),
                "adc_online_add_seconds": number(a, "add_seconds"),
                "rabitq_online_add_seconds": number(r, "add_seconds"),
                "adc_input_copy_seconds": number(a, "input_copy_seconds"),
                "adc_normalize_seconds": number(a, "normalize_seconds"),
                "adc_bq_encode_seconds": number(a, "bq_encode_seconds"),
                "adc_graph_add_seconds": number(a, "graph_add_seconds"),
                "adc_fp_append_seconds": number(a, "fp_append_seconds"),
                "rabitq_input_load_normalize_seconds": number(
                    r, "input_load_normalize_seconds"
                ),
                "rabitq_centroid_train_seconds": number(r, "centroid_train_seconds"),
                "rabitq_cluster_assignment_seconds": number(
                    r, "cluster_assignment_seconds"
                ),
                "rabitq_graph_and_quantization_seconds": number(
                    r, "graph_and_rabitq_quantization_seconds"
                ),
                "rabitq_index_build_total_seconds": number(
                    r, "index_build_total_seconds"
                ),
                "rabitq_online_insert_graph_and_quantization_seconds": number(
                    r, "online_insert_graph_and_quantization_seconds"
                ),
                "rabitq_online_insert_us_per_vector": number(
                    r, "online_insert_us_per_vector"
                ),
                "adc_source_csv": str(ADC),
                "rabitq_source_csv": str(RABITQ),
            }
        )
    comparison_path = ROOT / "frozen_adcbq_vs_rabitq_c16_comparison.csv"
    write_csv(comparison_path, comparison)

    timing: list[dict[str, object]] = []
    unique_modes = sorted(
        {(mode, corpus) for mode, corpus, _ in common},
        key=lambda item: (item[1], item[0]),
    )
    for mode, corpus in unique_modes:
        # Build and insertion timings are repeated across efSearch values.
        a = adc[(mode, corpus, 256)]
        r = rabitq[(mode, corpus, 256)]
        insert_size = int(a["insert_size"])
        adc_seconds = operation_seconds(a)
        rabitq_seconds = operation_seconds(r)
        timing.append(
            {
                "dataset": "msmarco_qwen1024_1m_to_101m",
                "mode": mode,
                "adc_calibration_policy": policy(mode),
                "base_size": int(a["base_size"]),
                "insert_size": insert_size,
                "corpus_size": corpus,
                "adc_operation_seconds": adc_seconds,
                "rabitq_operation_seconds": rabitq_seconds,
                "rabitq_over_adc_operation_time": rabitq_seconds / adc_seconds,
                "adc_online_insert_us_per_vector": (
                    1e6 * number(a, "add_seconds") / insert_size
                    if mode == "online_add_no_rebuild" else float("nan")
                ),
                "rabitq_online_insert_us_per_vector": number(
                    r, "online_insert_us_per_vector"
                ),
                "adc_input_copy_seconds": number(a, "input_copy_seconds"),
                "adc_normalize_seconds": number(a, "normalize_seconds"),
                "adc_bq_encode_seconds": number(a, "bq_encode_seconds"),
                "adc_graph_add_seconds": number(a, "graph_add_seconds"),
                "adc_fp_append_seconds": number(a, "fp_append_seconds"),
                "rabitq_input_load_normalize_seconds": number(
                    r, "input_load_normalize_seconds"
                ),
                "rabitq_centroid_train_seconds": number(r, "centroid_train_seconds"),
                "rabitq_cluster_assignment_seconds": number(
                    r, "cluster_assignment_seconds"
                ),
                "rabitq_graph_and_quantization_seconds": number(
                    r, "graph_and_rabitq_quantization_seconds"
                ),
                "rabitq_index_build_total_seconds": number(
                    r, "index_build_total_seconds"
                ),
                "timing_definition": (
                    "online: add_seconds; base/rebuild: build_or_rebuild_seconds"
                ),
            }
        )
    timing_path = ROOT / "frozen_adcbq_vs_rabitq_c16_insert_timing.csv"
    write_csv(timing_path, timing)

    transform_rows = read_csv(TRANSFORM)
    transform_output: list[dict[str, object]] = []
    for row in transform_rows:
        transform_output.append(
            {
                **row,
                "benchmark_scope": "standalone single-vector transform microbenchmark",
                "applicability": (
                    "same Qwen-1024, ADC-BQ or RaBitQ-B7-C16 transform; "
                    "measured separately from the 101M end-to-end insertion run"
                ),
                "source_csv": str(TRANSFORM),
            }
        )
    transform_path = ROOT / "adcbq_vs_rabitq_c16_vector_transform_microbenchmark.csv"
    write_csv(transform_path, transform_output)

    print(comparison_path)
    print(timing_path)
    print(transform_path)


if __name__ == "__main__":
    main()
