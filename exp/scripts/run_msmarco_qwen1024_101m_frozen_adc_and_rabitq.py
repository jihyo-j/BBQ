#!/usr/bin/env python3
"""Run frozen-base-calibration ADC-BQ online add, then RaBitQ online/rebuild."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiment/dynamic_insert_101m"
INPUTS = EXP / "inputs"
PYTHON = Path("/home/jihyo/anaconda3/bin/python")
PREPARE = REPO / "experiment/prepare_msmarco_qwen1024_101m_dynamic_inputs.py"
BUILD_ADC = REPO / "experiment/build_adcbq_fpgraph_dynamic_101m.sh"
ADC = REPO / "experiment/adcbq_fpgraph_dynamic_101m"
RABITQ = REPO / "experiment/rabitq_qwen3m_online_add_rebuild.py"
BASE = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_base.fbin")
COMBINED = Path("/dev/shm/msmarco_qwen1024_dynamic_101m/base101m.aligned4k.fbin")
CAL_QUERY = INPUTS / "seed11.cal100.query.fbin"
TEST_QUERY = INPUTS / "seed11.test1000.query.fbin"
BASE_GT = INPUTS / "seed11.test1000.base1m.gt_k10.ibin"
BASE_MAP = INPUTS / "base_1m_to_combined_ids.i64bin"
SELECTED = INPUTS / "new_100m_selected_ids.i64bin"
OLD_CALIBRATED = EXP / "adcbq_calibrated_target099_random_independent_t32.csv"
ADC_FROZEN = EXP / "adcbq_online_frozen_1m_tau_random_independent_t32.csv"
ADC_MERGED = EXP / "adcbq_online_frozen_1m_tau_vs_rebuild_recalibrated_t32.csv"
RABITQ_OUT = EXP / "rabitq_b7_centroid_schedule_random_independent_t32.csv"
STATUS = EXP / "frozen_adc_rabitq_status.json"
INSERTS = (10_000_000, 30_000_000, 50_000_000, 100_000_000)
REBUILD_CLUSTERS = (3_072, 6_144, 7_168, 10_240)
EFS = (64, 128, 256, 512, 1024)
THREADS = 32


def update_status(stage: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, STATUS)
    print(f"STATUS {stage} {extra}", flush=True)


def run(command: list[str], env: dict[str, str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def csv_keys(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def result_complete(path: Path, modes: tuple[str, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        _, rows = csv_keys(path)
        expected = len(EFS) * sum(1 if mode == "base_only" else len(INSERTS) for mode in modes)
        keys = {(r["mode"], int(r["insert_size"]), int(r["efSearch"])) for r in rows}
        return len(rows) == expected and len(keys) == expected
    except (OSError, UnicodeDecodeError, KeyError, ValueError):
        return False


def merge_adc() -> None:
    frozen_fields, frozen_rows = csv_keys(ADC_FROZEN)
    old_fields, old_rows = csv_keys(OLD_CALIBRATED)
    rebuild_rows = [row for row in old_rows if row["mode"] == "full_rebuild_oracle"]
    rows = frozen_rows + rebuild_rows
    fields = frozen_fields + [name for name in old_fields if name not in frozen_fields]
    expected = len(EFS) * (1 + 2 * len(INSERTS))
    unique = {(r["mode"], r["insert_size"], r["efSearch"]) for r in rows}
    if len(rows) != expected or len(unique) != expected:
        raise RuntimeError(f"ADC merge has {len(rows)} rows/{len(unique)} keys; expected {expected}")
    mode_order = {"base_only": 0, "online_add_no_rebuild": 1, "full_rebuild_oracle": 2}
    rows.sort(key=lambda r: (int(r["insert_size"]), mode_order[r["mode"]], int(r["efSearch"])))
    tmp = ADC_MERGED.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, ADC_MERGED)


def main() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        }
    )
    env.pop("OMP_PLACES", None)
    EXP.mkdir(parents=True, exist_ok=True)

    try:
        update_status("building_adc_latest_lut")
        run(["bash", str(BUILD_ADC)], env)

        update_status("materializing_combined_fbin_streaming")
        run(
            [
                str(PYTHON),
                str(PREPARE),
                "--threads",
                str(THREADS),
                "--combined-fbin",
                str(COMBINED),
            ],
            env,
        )

        if not result_complete(ADC_FROZEN, ("base_only", "online_add_no_rebuild")):
            update_status("running_adc_online_frozen_1m_tau")
            run(
                [
                    str(ADC),
                    "--dataset", "msmarco-qwen1024-v1-1m-plus-v2-100m",
                    "--base-train", str(BASE),
                    "--larger-train", str(COMBINED),
                    "--query", str(TEST_QUERY),
                    "--calibration-query", str(CAL_QUERY),
                    "--calibration-target", "0.99",
                    "--base-gt", str(BASE_GT),
                    "--base-to-larger-ids", str(BASE_MAP),
                    "--selected-ids", str(SELECTED),
                    "--gt-dir", str(INPUTS),
                    "--out", str(ADC_FROZEN),
                    "--policy", "random",
                    "--fp-rerank-source", "mmap",
                    "--fp-rerank-path", str(COMBINED),
                    "--online-graph-add", "adcbq",
                    "--insert-sizes", *map(str, INSERTS),
                    "--efs", *map(str, EFS),
                    "--m", "24",
                    "--ef-construction", "200",
                    "--rerank-n", "80",
                    "--max-queries", "1000",
                    "--rounds", "1",
                    "--build-threads", str(THREADS),
                    "--add-threads", str(THREADS),
                    "--search-threads", str(THREADS),
                    "--io-queue-depth", "1024",
                    "--add-batch-size", "50000",
                    "--independent-online-checkpoints",
                    "--no-overlap",
                    "--freeze-online-calibration-from-base",
                    "--skip-rebuild",
                ],
                env,
            )
        if not result_complete(ADC_FROZEN, ("base_only", "online_add_no_rebuild")):
            raise RuntimeError("frozen ADC online CSV is incomplete")
        if not result_complete(OLD_CALIBRATED, ("base_only", "online_add_no_rebuild", "full_rebuild_oracle")):
            raise RuntimeError("existing recalibrated ADC rebuild CSV is incomplete")
        merge_adc()
        update_status("adc_complete", output=str(ADC_MERGED))

        if not result_complete(RABITQ_OUT, ("base_only", "online_add_no_rebuild", "full_rebuild_oracle")):
            update_status("running_rabitq_online_and_rebuild")
            run(
                [
                    str(PYTHON),
                    str(RABITQ),
                    "--base-train", str(BASE),
                    "--larger-train", str(COMBINED),
                    "--query", str(TEST_QUERY),
                    "--base-gt", str(BASE_GT),
                    "--selected-ids", str(SELECTED),
                    "--stage-gt-dir", str(INPUTS),
                    "--out", str(RABITQ_OUT),
                    "--dataset", "msmarco-qwen1024-v1-1m-plus-v2-100m-rabitq-b7",
                    "--insert-sizes", *map(str, INSERTS),
                    "--policy", "random",
                    "--num-clusters", "1024",
                    "--rebuild-clusters", *map(str, REBUILD_CLUSTERS),
                    "--m", "24",
                    "--ef-construction", "200",
                    "--total-bits", "7",
                    "--efs", *map(str, EFS),
                    "--max-queries", "1000",
                    "--rounds", "1",
                    "--build-threads", str(THREADS),
                    "--add-threads", str(THREADS),
                    "--search-threads", str(THREADS),
                    "--independent-online-checkpoints",
                    "--reuse-larger-as-raw-data",
                    "--unlink-larger-after-load",
                ],
                env,
            )
        if not result_complete(RABITQ_OUT, ("base_only", "online_add_no_rebuild", "full_rebuild_oracle")):
            raise RuntimeError("RaBitQ CSV is incomplete")
        update_status("complete", adc_output=str(ADC_MERGED), rabitq_output=str(RABITQ_OUT))
    except BaseException as exc:
        update_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
