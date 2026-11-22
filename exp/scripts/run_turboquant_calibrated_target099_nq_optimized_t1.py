#!/usr/bin/env python3
"""Run the paired NQ target-.99 comparison with optimized TQ HNSW search."""

from pathlib import Path

import run_turboquant_calibrated_target099_main4_t1 as benchmark


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPO / "experiment/turboquant_calibrated_target099_nq_optimized_inline_t1"
)
ADC_RESULT_ROOT = (
    REPO / "experiment/nq_qwen1024_calibrated_adcbq_optimized_pair_t1"
)

benchmark.RESULT_ROOT = RESULT_ROOT
benchmark.CALIBRATION_ROOT = RESULT_ROOT / "calibration"
benchmark.RAW_ROOT = RESULT_ROOT / "raw"
benchmark.SUMMARY = RESULT_ROOT / "summary_t1.csv"
benchmark.STATUS = RESULT_ROOT / "status.json"

benchmark.ADC_RESULT_ROOT = ADC_RESULT_ROOT
benchmark.ADC_CALIBRATION_ROOT = ADC_RESULT_ROOT / "calibration"
benchmark.ADC_RAW_ROOT = ADC_RESULT_ROOT / "raw"
benchmark.ADC_SUMMARY = ADC_RESULT_ROOT / "summary_t1.csv"

benchmark.search.BINARY = (
    REPO / "experiment/faiss_hnsw_bq_budget_paged_codes_tq_inline"
)
benchmark.TQ_INLINE_HNSW = True


if __name__ == "__main__":
    raise SystemExit(benchmark.main())
