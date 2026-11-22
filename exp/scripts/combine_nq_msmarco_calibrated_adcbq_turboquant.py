#!/usr/bin/env python3
"""Combine the matched NQ/MSMARCO calibrated ADC-BQ and TurboQuant curves."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCES = (
    REPO / "experiment/nq_qwen1024_calibrated_adcbq_optimized_pair_t1/summary_t1.csv",
    REPO / "experiment/turboquant_calibrated_target099_nq_optimized_inline_t1/summary_t1.csv",
    REPO / "experiment/msmarco_qwen1024_calibrated_adcbq_optimized_pair_t1/summary_t1.csv",
    REPO / "experiment/turboquant_calibrated_target099_msmarco_optimized_inline_pair_t1/summary_t1.csv",
)
OUTPUT = (
    REPO
    / "experiment/nq_msmarco_calibrated_adcbq_turboquant_optimized_pair_t1"
    / "summary_t1.csv"
)
DATASET_ORDER = {"nq_qwen06_1024": 0, "msmarco_qwen1024": 1}
METHOD_ORDER = {"adc_bq_calibrated": 0, "turboquant_tq1_calibrated": 1}
EXPECTED_EFS = {
    80, 84, 88, 92, 96, 100, 104, 108, 112, 116,
    120, 128, 144, 160, 192, 224, 256, 320, 512, 1024,
}


def main() -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for source in SOURCES:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"missing CSV header: {source}")
            for field in reader.fieldnames:
                if field not in fields:
                    fields.append(field)
            for row in reader:
                row["combined_source_csv"] = str(source.relative_to(REPO))
                rows.append(row)

    expected_pairs = {
        (dataset, method)
        for dataset in DATASET_ORDER
        for method in METHOD_ORDER
    }
    counts = Counter((row["dataset"], row["method_key"]) for row in rows)
    if set(counts) != expected_pairs or any(count != 20 for count in counts.values()):
        raise RuntimeError(f"unexpected dataset/method counts: {dict(counts)}")
    if len(rows) != 80:
        raise RuntimeError(f"expected 80 rows, found {len(rows)}")

    for pair in expected_pairs:
        selected = [
            row for row in rows
            if (row["dataset"], row["method_key"]) == pair
        ]
        efs = {int(row["efSearch"]) for row in selected}
        if efs != EXPECTED_EFS:
            raise RuntimeError(f"unexpected efSearch grid for {pair}: {sorted(efs)}")
        for row in selected:
            if (
                row["rounds"] != "3"
                or row["search_threads"] != "1"
                or row["first_stage_search_impl"] != "inline_batched_prefetch"
                or row["calibration_protocol"]
                != "per-ef-pool-top10-split-conformal"
                or row["target_pool_top10_coverage"] != "0.99"
            ):
                raise RuntimeError(f"unmatched paired protocol row: {row}")

    rows.sort(
        key=lambda row: (
            DATASET_ORDER[row["dataset"]],
            METHOD_ORDER[row["method_key"]],
            int(row["efSearch"]),
        )
    )
    source_index = fields.index("source_csv")
    fields.insert(source_index, "combined_source_csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, OUTPUT)
    print(f"WROTE {OUTPUT} ({len(rows)} rows)")
    for pair in sorted(
        counts,
        key=lambda item: (DATASET_ORDER[item[0]], METHOD_ORDER[item[1]]),
    ):
        print(f"  {pair[0]} / {pair[1]}: {counts[pair]} rows")


if __name__ == "__main__":
    main()
