#!/usr/bin/env python3
"""Run 1M -> 101M RaBitQ-1 adaptive-FP online-add/rebuild at fixed C=16."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiment/dynamic_insert_101m"
INPUTS = EXP / "inputs"
PYTHON = Path("/home/jihyo/anaconda3/envs/rabitq-cfr/bin/python")
PREPARE_PYTHON = Path("/home/jihyo/anaconda3/bin/python")
PREPARE = REPO / "experiment/prepare_msmarco_qwen1024_101m_dynamic_inputs.py"
RABITQ = REPO / "experiment/rabitq_qwen3m_online_add_rebuild.py"
BINDING = (
    REPO / "experiment/rabitq_v051_adaptive_odirect_aligned4k_add_build/rabitqlib"
)
BASE = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_base.fbin")
COMBINED = Path("/dev/shm/msmarco_qwen1024_dynamic_101m/base101m.aligned4k.fbin")
QUERY = INPUTS / "seed11.test1000.query.fbin"
BASE_GT = INPUTS / "seed11.test1000.base1m.gt_k10.ibin"
SELECTED = INPUTS / "new_100m_selected_ids.i64bin"
OUTPUT = EXP / "rabitq_b1_adaptive_c16_random_independent_t32.csv"
MANIFEST = OUTPUT.with_suffix(OUTPUT.suffix + ".manifest.json")
STATUS = EXP / "rabitq_b1_adaptive_c16_status.json"
ASSIGNMENT_CACHE = EXP / "rabitq_b1_c16_frozen_selected100m"
INSERTS = (10_000_000, 30_000_000, 50_000_000, 100_000_000)
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


def complete() -> bool:
    if not OUTPUT.is_file() or not MANIFEST.is_file():
        return False
    try:
        with OUTPUT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(r["mode"], int(r["insert_size"]), int(r["efSearch"])) for r in rows}
        expected = {("base_only", 0, ef) for ef in EFS} | {
            (mode, insert_size, ef)
            for mode in ("online_add_no_rebuild", "full_rebuild_oracle")
            for insert_size in INSERTS
            for ef in EFS
        }
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return (
            len(rows) == len(expected) == 45
            and keys == expected
            and all(
                int(r["num_clusters"]) == 16
                and int(r["total_bits"]) == 1
                and int(r["M"]) == 24
                and int(r["efConstruction"]) == 200
                and int(r["nq"]) == 1000
                and r["rerank_policy"] == "rabitq_1bit_adaptive_fp_odirect"
                and 0.0 <= float(r["recall_current_at_k"]) <= 1.0
                and float(r["fp_reads_mean"]) >= 10.0
                and float(r["odirect_read_bytes_per_query"]) > 0.0
                and float(r["odirect_read_requests_per_query"]) > 0.0
                and int(r["fbin_data_offset"]) == 4096
                and math.isfinite(float(r["qps"]))
                and float(r["qps"]) > 0.0
                and (
                    r["mode"] != "online_add_no_rebuild"
                    or float(r["online_insert_graph_and_quantization_seconds"]) > 0.0
                )
                for r in rows
            )
            and manifest["base_num_clusters"] == 16
            and manifest["rebuild_clusters"] == [16] * len(INSERTS)
            and manifest["online_checkpoint_source"]
            == "base_index_rebuilt_for_each_checkpoint"
            and manifest["adaptive_fp_path"] == str(COMBINED)
            and manifest["saved_index_paths"] == []
            and b"\0" not in OUTPUT.read_bytes()
        )
    except (OSError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        return False


def resume_final_rebuild_only() -> bool:
    if not OUTPUT.is_file():
        return False
    try:
        with OUTPUT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(r["mode"], int(r["insert_size"]), int(r["efSearch"])) for r in rows}
        final_online = {
            ("online_add_no_rebuild", 100_000_000, ef) for ef in EFS
        }
        final_rebuild = {
            ("full_rebuild_oracle", 100_000_000, ef) for ef in EFS
        }
        return final_online <= keys and not (final_rebuild <= keys)
    except (OSError, UnicodeDecodeError, KeyError, ValueError):
        return False


def main() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
            "RABITQ_ALLOW_PY311": "1",
        }
    )
    env.pop("LD_PRELOAD", None)
    env.pop("OMP_PLACES", None)
    EXP.mkdir(parents=True, exist_ok=True)
    try:
        if not complete():
            update_status("preparing_101m_aligned_fp_sidecar", path=str(COMBINED))
            run(
                [
                    str(PREPARE_PYTHON),
                    str(PREPARE),
                    "--threads", str(THREADS),
                    "--combined-fbin", str(COMBINED),
                    "--materialize-mode", "memory",
                ],
                env,
            )
            update_status("running_b1_adaptive_online_no_rebuild_and_rebuild_oracle")
            resume_args = (
                ["--resume-existing", "--skip-online"]
                if resume_final_rebuild_only()
                else []
            )
            run(
                [
                    str(PYTHON),
                    str(RABITQ),
                    "--base-train", str(BASE),
                    "--larger-train", str(COMBINED),
                    "--query", str(QUERY),
                    "--base-gt", str(BASE_GT),
                    "--selected-ids", str(SELECTED),
                    "--stage-gt-dir", str(INPUTS),
                    "--out", str(OUTPUT),
                    "--dataset", "msmarco-qwen1024-v1-1m-plus-v2-100m-rabitq-b1-adaptive-c16",
                    "--insert-sizes", *map(str, INSERTS),
                    "--policy", "random",
                    "--num-clusters", "16",
                    "--rebuild-clusters", "16", "16", "16", "16",
                    "--m", "24",
                    "--ef-construction", "200",
                    "--total-bits", "1",
                    "--efs", *map(str, EFS),
                    "--max-queries", "1000",
                    "--rounds", "1",
                    "--build-threads", str(THREADS),
                    "--add-threads", str(THREADS),
                    "--search-threads", str(THREADS),
                    "--independent-online-checkpoints",
                    "--reuse-larger-as-raw-data",
                    "--adaptive-fp-path", str(COMBINED),
                    "--rabitq-binding-dir", str(BINDING),
                    "--selected-assignment-cache-prefix", str(ASSIGNMENT_CACHE),
                    *resume_args,
                ],
                env,
            )
        if not complete():
            raise RuntimeError("RaBitQ-1 adaptive C=16 result is incomplete")
        update_status("complete", output=str(OUTPUT))
    except BaseException as exc:
        update_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
