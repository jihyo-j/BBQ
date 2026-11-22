#!/usr/bin/env python3
"""Paired MS MARCO Qwen-1024 ADC-BQ/TurboQuant target-.99 sweep."""

from pathlib import Path

import run_turboquant_calibrated_target099_main4_t1 as benchmark


REPO = Path(__file__).resolve().parents[1]
TQ_ROOT = (
    REPO
    / "experiment/turboquant_calibrated_target099_msmarco_optimized_inline_pair_t1"
)
ADC_ROOT = (
    REPO
    / "experiment/msmarco_qwen1024_calibrated_adcbq_optimized_pair_t1"
)

benchmark.DATASETS = ("msmarco_qwen1024",)

benchmark.RESULT_ROOT = TQ_ROOT
# Candidate order is bit-exact between the generic and optimized inline TQ
# traversal, so reuse the already completed target-.99 fit and only remeasure
# the optimized search/QPS path under the paired server load.
benchmark.CALIBRATION_ROOT = (
    REPO / "experiment/turboquant_calibrated_target099_main4_t1/calibration"
)
benchmark.RAW_ROOT = TQ_ROOT / "raw"
benchmark.SUMMARY = TQ_ROOT / "summary_t1.csv"
benchmark.STATUS = TQ_ROOT / "status.json"

benchmark.ADC_RESULT_ROOT = ADC_ROOT
benchmark.ADC_CALIBRATION_ROOT = ADC_ROOT / "calibration"
benchmark.ADC_RAW_ROOT = ADC_ROOT / "raw"
benchmark.ADC_SUMMARY = ADC_ROOT / "summary_t1.csv"

benchmark.search.BINARY = (
    REPO / "experiment/faiss_hnsw_bq_budget_paged_codes_tq_inline"
)
benchmark.TQ_INLINE_HNSW = True


if __name__ == "__main__":
    raise SystemExit(benchmark.main())
