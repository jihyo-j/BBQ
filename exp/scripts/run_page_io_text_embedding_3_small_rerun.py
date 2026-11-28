#!/usr/bin/env python3
"""Restartable four-method page-I/O rerun on MS MARCO text-embedding-3-small."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiment"
CAMPAIGN = EXP / "page_io_text_embedding_3_small_rerun"
EXTERNAL = Path("/home/smrc/samsung-nvme/jihyo/page_io_text_embedding_3_small_rerun")
MANIFEST = Path("/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1/dataset_manifest.json")
KEY = "sieun_msmarco_openai1536"
PYTHON = Path("/home/jihyo/anaconda3/envs/rabitq-cfr/bin/python")
PLOT_PYTHON = Path("/home/jihyo/anaconda3/bin/python")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def status(**updates: object) -> None:
    path = CAMPAIGN / "status.json"
    payload: dict[str, object] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(updates)
    payload["updated_at"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def wait_for_idle(stage: str, threshold: float = 8.0) -> None:
    while True:
        load = os.getloadavg()[0]
        if load <= threshold:
            status(stage=f"starting_{stage}", load1=load)
            return
        status(stage=f"waiting_for_idle_before_{stage}", load1=load, threshold=threshold)
        print(f"WAIT stage={stage} load1={load:.2f} threshold={threshold:.2f}", flush=True)
        time.sleep(30)


def run(stage: str, command: list[str], environment: dict[str, str] | None = None) -> None:
    wait_for_idle(stage)
    log = CAMPAIGN / "logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status(stage=f"running_{stage}", command=command, log=str(log))
    print("RUN " + " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] RUN {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"[{now()}] returncode={completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"stage {stage} failed; see {log}")
    status(stage=f"complete_{stage}")


def main() -> None:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(EXTERNAL).free
    if free < 50 * 1024**3:
        raise RuntimeError(f"need at least 50 GiB free under {EXTERNAL}, found {free / 1024**3:.1f}")
    status(
        stage="starting",
        dataset=KEY,
        model="OpenAI text-embedding-3-small",
        calibration_queries=1000,
        test_queries=5980,
        search_threads=1,
        rounds=3,
        seed=11,
    )
    common_env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    run(
        "adcbq_io_modes",
        [
            str(PYTHON), "-u", str(EXP / "run_bbq_calibrated_io_modes_t1.py"),
            "--rounds", "3", "--build-threads", "24", "--cpu", "39",
            "--result-root", str(CAMPAIGN / "adc_io_modes"),
            "--datasets", KEY,
        ],
        common_env,
    )
    run(
        "qdrant_tq1_fixed_r",
        [
            str(PLOT_PYTHON), "-u", str(EXP / "run_qdrant_tq1_text_embedding_3_small_rerun.py"),
            "--result-root", str(CAMPAIGN / "qdrant_tq1"),
            "--external-root", str(EXTERNAL / "qdrant_tq1"),
        ],
        common_env,
    )

    disk_env = {
        **common_env,
        "DISKVECLAB_EXTRA_DATASETS_MANIFEST": str(MANIFEST),
        "DISKVECLAB_DATASETS": KEY,
        "DISKVECLAB_RESULT_ROOT": str(CAMPAIGN / "disk_matrix"),
    }
    run(
        "diskann_pq192_and_rabitq1",
        [
            "/usr/bin/python3", "-u", str(EXP / "run_diskveclab_full_matrix_diskann.py"),
            "--datasets", KEY,
            "--rounds", "3",
            "--build-threads", "24",
            "--search-threads", "1",
            "--pq-code-budget", "one-bit",
            "--external-root", str(EXTERNAL / "diskann"),
        ],
        disk_env,
    )
    run(
        "plot",
        [
            str(PLOT_PYTHON), "-u", str(EXP / "plot_page_io_text_embedding_3_small.py"),
            "--campaign", str(CAMPAIGN),
        ],
        common_env,
    )
    status(
        stage="complete",
        normalized_csv=str(CAMPAIGN / "figure/normalized_page_io.csv"),
        operating_points_csv=str(CAMPAIGN / "figure/operating_points.csv"),
        figure_pdf=str(CAMPAIGN / "figure/page_io.pdf"),
    )
    print(f"COMPLETE {CAMPAIGN}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status(stage="failed", error=repr(error))
        raise
