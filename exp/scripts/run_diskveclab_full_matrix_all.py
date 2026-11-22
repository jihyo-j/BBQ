#!/usr/bin/env python3
"""Restartable sequential launcher for the common held-out method matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import diskveclab_full_matrix as matrix


REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = REPO / "experiment"
PYTHON = sys.executable
STANDARD_RESULT = matrix.RESULT_ROOT / "spann_pipeann"
STANDARD_WORK = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_full_matrix_spann_pipeann"
)
METHODS = (
    "adcbq",
    "diskann",
    "spann_pipeann",
    "spann_rabitq",
    "rabitq7",
    "elasticsearch",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def emit(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    matrix.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (matrix.RESULT_ROOT / "full_matrix_progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str]) -> None:
    emit("RUN " + shlex.join(command))
    subprocess.run(command, cwd=REPO, check=True)
    emit("DONE " + shlex.join(command))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(METHODS)
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="rounds for the full 20-point curve; repeat only selected Pareto points later",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    datasets = args.datasets
    common = ["--datasets", *datasets, "--rounds", str(args.rounds)]
    if "adcbq" in args.methods:
        run([PYTHON, str(EXPERIMENT / "run_diskveclab_full_matrix_adcbq.py"), *common])
    if "diskann" in args.methods:
        run([PYTHON, str(EXPERIMENT / "run_diskveclab_full_matrix_diskann.py"), *common])
    if "spann_pipeann" in args.methods:
        command = [
            PYTHON,
            str(EXPERIMENT / "run_nq_quora_spann_pipeann.py"),
            "--method", "all",
            "--rounds", str(args.rounds),
            "--search-threads", "1",
            "--heldout-matrix",
            "--result-root", str(STANDARD_RESULT),
            "--external-root", str(STANDARD_WORK),
        ]
        for key in datasets:
            command.extend(("--dataset", key))
        run(command)
    if "spann_rabitq" in args.methods:
        run(
            [
                PYTHON,
                str(EXPERIMENT / "run_diskveclab_full_matrix_spann_rabitq.py"),
                *common,
            ]
        )
    if "rabitq7" in args.methods:
        run([PYTHON, str(EXPERIMENT / "run_diskveclab_full_matrix_rabitq7.py"), *common])
    if "elasticsearch" in args.methods:
        run(
            [
                PYTHON,
                str(EXPERIMENT / "run_diskveclab_full_matrix_elasticsearch.py"),
                *common,
            ]
        )
    emit("ALL REQUESTED FULL-MATRIX METHODS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
