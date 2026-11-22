#!/usr/bin/env python3
"""Elasticsearch BBQ on the common held-out 16-dataset matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import diskveclab_full_matrix as matrix


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiment/run_nq_quora_elasticsearch_bbq_rerank80.py"
RESULT_ROOT = matrix.RESULT_ROOT / "elasticsearch_bbq"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_full_matrix_elasticsearch_bbq"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument(
        "--phase", choices=("build", "search", "report", "cleanup", "all"), default="all"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-index", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    command = [
        sys.executable,
        str(RUNNER),
        "--phase", args.phase,
        "--heldout-matrix",
        "--build-workers", "24",
        "--search-concurrency", "1",
        "--rounds", str(args.rounds),
        "--result-root", str(RESULT_ROOT),
        "--external-root", str(EXTERNAL_ROOT),
        "--reset-incomplete",
    ]
    if not args.keep_index and args.phase in {"search", "all"}:
        command.append("--cleanup-after-search")
    if args.force:
        command.append("--force")
    for key in args.datasets:
        command.extend(("--dataset", key))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
