#!/usr/bin/env python3
"""Complete/reuse calibrated ADC-BQ measurements for the held-out matrix."""

from __future__ import annotations

import argparse

import diskveclab_full_matrix as matrix
import run_remaining_dbpedia_msmarco_adcbq_calibration as calibrated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.build_threads < 1:
        parser.error("rounds and build threads must be positive")

    old_threads = calibrated.THREADS
    old_result_root = calibrated.RESULT_ROOT
    calibrated.THREADS = (1,)
    calibrated.RESULT_ROOT = matrix.RESULT_ROOT / "calibrated_adcbq"
    calibrated.calibration.ensure_binary()
    calibrated.calibration.ALIGNED_FP_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        for key in args.datasets:
            calibrated.benchmark_dataset(
                key,
                matrix.CALIBRATION_ROWS[key],
                args.rounds,
                args.build_threads,
                args.force,
            )
    finally:
        calibrated.THREADS = old_threads
        calibrated.RESULT_ROOT = old_result_root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
