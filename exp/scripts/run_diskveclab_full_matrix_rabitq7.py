#!/usr/bin/env python3
"""Run in-memory total-7-bit RaBitQ on the common held-out matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import diskveclab_full_matrix as matrix
import run_v051_ssd_odirect_multidataset as rabitq


DEFAULT_EFS = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)

NVME_ROOT = Path("/home/smrc/samsung-nvme/jihyo").resolve()


def cleanup_b7_index(dataset: object) -> None:
    """Remove only the exact registered, reproducible b7 index."""
    path = dataset.b7_index.resolve()
    if NVME_ROOT not in path.parents:
        raise RuntimeError(f"refusing unsafe RaBitQ cleanup outside {NVME_ROOT}: {path}")
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        print(f"CLEANED completed RaBitQ b7 index {path} bytes={size}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--keep-index", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.search_threads != 1:
        parser.error("rounds must be positive and the matrix requires T=1")

    old_result = rabitq.RESULT_DIR
    rabitq.RESULT_DIR = matrix.RESULT_ROOT / "rabitq7"
    run_args = SimpleNamespace(
        efs=list(dict.fromkeys(args.efs)),
        rounds=args.rounds,
        max_queries=0,
        build_threads=args.build_threads,
        search_threads=args.search_threads,
        adc_rerank_n=80,
        parts=("b7",),
        force_search=args.force_search,
        rebuild=args.rebuild,
    )
    try:
        for key in args.datasets:
            original = rabitq.DATASETS[key]
            rabitq.validate(original)
            heldout = matrix.heldout_dataset(key)
            print(
                f"DATASET {key}: test_queries={matrix.split_for(key).test_rows}",
                flush=True,
            )
            result = rabitq.output_csv(heldout, "rabitq_b7_ram")
            if (
                rabitq.complete_csv(
                    result,
                    run_args.efs,
                    run_args.search_threads,
                    run_args.rounds,
                )
                and not run_args.force_search
            ):
                print(f"SKIP complete {result}", flush=True)
                if not args.keep_index:
                    cleanup_b7_index(original)
                continue
            rabitq.ensure_rabitq_indexes(original, run_args)
            rabitq.run_rabitq(heldout, 7, run_args)
            if not args.keep_index:
                cleanup_b7_index(original)
    finally:
        rabitq.RESULT_DIR = old_result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
