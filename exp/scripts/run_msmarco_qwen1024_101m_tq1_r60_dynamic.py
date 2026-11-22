#!/usr/bin/env python3
"""Run fixed-R60 or base-calibrated Qdrant TQ1+ online insertion."""

from __future__ import annotations

import csv
import ctypes
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiment/dynamic_insert_101m"
INPUTS = EXP / "inputs"
FFI = REPO / "experiment/qdrant_tq_ffi/target/release/libqdrant_tq_ffi.so"
FFI_DIR = REPO / "experiment/qdrant_tq_ffi"
BUILD = REPO / "experiment/build_adcbq_fpgraph_dynamic_101m.sh"
DYNAMIC = REPO / "experiment/adcbq_fpgraph_dynamic_101m"
BASE = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_base.fbin")
INSERT_EMB_ROOT = Path("/home/smrc/samsung-nvme/sieun/qwen100m/emb")
INSERT_PERMUTATION = Path(
    "/home/smrc/samsung-nvme/sieun/qwen100m/insert_exp/gt/perm.npy"
)
QUERY = INPUTS / "seed11.test1000.query.fbin"
CAL_QUERY = INPUTS / "seed11.cal100.query.fbin"
BASE_GT = INPUTS / "seed11.test1000.base1m.gt_k10.ibin"
BASE_MAP = INPUTS / "base_1m_to_combined_ids.i64bin"
SELECTED = INPUTS / "new_100m_selected_ids.i64bin"
TQ_CODES = EXP / "tq1plus_frozen_base1m.tqdata"
TQ_META = EXP / "tq1plus_frozen_base1m.meta"
FIXED_OUTPUT = EXP / "tq1plus_r60_random_independent_t32.csv"
FIXED_STATUS = EXP / "tq1plus_r60_status.json"
CALIBRATED_OUTPUT = EXP / "tq1plus_calibrated_target099_frozen_1m_tau_random_independent_t32.csv"
CALIBRATED_STATUS = EXP / "tq1plus_calibrated_target099_status.json"
OUTPUT = FIXED_OUTPUT
STATUS = FIXED_STATUS
CALIBRATED = False
INSERTS = (10_000_000, 30_000_000, 50_000_000, 100_000_000)
EFS = (64, 128, 256, 512, 1024)
THREADS = 32
RERANK = 60
EXPECTED_ROWS = 5 + len(INSERTS) * len(EFS)


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
    if not OUTPUT.is_file() or OUTPUT.stat().st_size == 0:
        return False
    try:
        with OUTPUT.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {(row["mode"], int(row["insert_size"]), int(row["efSearch"])) for row in rows}
        expected = {("base_only", 0, ef) for ef in EFS} | {
            (mode, insert, ef)
            for mode in ("online_add_no_rebuild",)
            for insert in INSERTS
            for ef in EFS
        }
        return (
            len(rows) == EXPECTED_ROWS
            and keys == expected
            and all(
                int(row["candidate_cap"])
                == (int(row["efSearch"]) if CALIBRATED else RERANK)
                and row["calibrated_target_enabled"]
                == ("true" if CALIBRATED else "false")
                and (
                    not CALIBRATED
                    or (
                        abs(float(row["calibration_target"]) - 0.99) < 1e-12
                        and row["calibration_reference"]
                        in (
                            "exact_fp_top10_within_graph_candidate_pool",
                            "frozen_1m_base_tau_no_online_recalibration",
                        )
                    )
                )
                and row["online_graph_add"] in ("none", "tq1", "fp")
                and float(row["recall_current_at_k"]) >= 0.0
                for row in rows
            )
        )
    except (OSError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        return False


def load_ffi() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(FFI))
    lib.qdrant_tq_build.restype = ctypes.c_void_p
    lib.qdrant_tq_build.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int32,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    lib.qdrant_tq_load.restype = ctypes.c_void_p
    lib.qdrant_tq_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.qdrant_tq_free.argtypes = [ctypes.c_void_p]
    lib.qdrant_tq_code_size.restype = ctypes.c_size_t
    lib.qdrant_tq_code_size.argtypes = [ctypes.c_void_p]
    lib.qdrant_tq_codes.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.qdrant_tq_codes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.qdrant_tq_last_error.restype = ctypes.c_char_p
    return lib


def valid_base_codec(lib: ctypes.CDLL, rows: int) -> bool:
    if not TQ_CODES.is_file() or not TQ_META.is_file():
        return False
    index = lib.qdrant_tq_load(str(TQ_CODES).encode(), str(TQ_META).encode())
    if not index:
        return False
    try:
        stride = int(lib.qdrant_tq_code_size(index))
        byte_len = ctypes.c_size_t()
        codes = lib.qdrant_tq_codes(index, ctypes.byref(byte_len))
        return bool(codes) and stride > 0 and byte_len.value == rows * stride
    finally:
        lib.qdrant_tq_free(index)


def ensure_base_codec(lib: ctypes.CDLL) -> None:
    with BASE.open("rb") as handle:
        rows, dim = struct.unpack("<ii", handle.read(8))
    if rows <= 0 or dim <= 0:
        raise RuntimeError(f"invalid base fbin header: {BASE}")
    if valid_base_codec(lib, rows):
        print(f"REUSE {TQ_CODES} and {TQ_META}", flush=True)
        return

    update_status("building_frozen_base1m_tq1plus", rows=rows, dim=dim)
    values = np.memmap(BASE, mode="r", dtype="<f4", offset=8, shape=(rows, dim))
    tmp_codes = TQ_CODES.with_suffix(TQ_CODES.suffix + ".tmp")
    tmp_meta = TQ_META.with_suffix(TQ_META.suffix + ".tmp")
    index = lib.qdrant_tq_build(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        dim,
        0,
        THREADS,
        str(tmp_codes).encode(),
        str(tmp_meta).encode(),
    )
    if not index:
        message = lib.qdrant_tq_last_error()
        raise RuntimeError(message.decode() if message else "TQ base build failed")
    lib.qdrant_tq_free(index)
    os.replace(tmp_codes, TQ_CODES)
    os.replace(tmp_meta, TQ_META)
    if not valid_base_codec(lib, rows):
        raise RuntimeError("built TQ base codec failed validation")


def main() -> None:
    global OUTPUT, STATUS, CALIBRATED
    options = set(sys.argv[1:])
    unknown = options - {"--calibrated", "--wait-for-fixed"}
    if unknown:
        raise SystemExit("unknown options: " + " ".join(sorted(unknown)))
    CALIBRATED = "--calibrated" in options
    if "--wait-for-fixed" in options and not CALIBRATED:
        raise SystemExit("--wait-for-fixed requires --calibrated")
    if CALIBRATED:
        OUTPUT = CALIBRATED_OUTPUT
        STATUS = CALIBRATED_STATUS

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(THREADS),
            "OPENBLAS_NUM_THREADS": str(THREADS),
            "MKL_NUM_THREADS": str(THREADS),
            "OMP_DYNAMIC": "FALSE",
            "OMP_PROC_BIND": "FALSE",
        }
    )
    env.pop("OMP_PLACES", None)
    EXP.mkdir(parents=True, exist_ok=True)

    try:
        if "--wait-for-fixed" in options:
            update_status("waiting_for_fixed_r60", fixed_status=str(FIXED_STATUS))
            while True:
                try:
                    fixed = json.loads(FIXED_STATUS.read_text())
                except (OSError, json.JSONDecodeError):
                    fixed = {}
                if fixed.get("stage") == "complete":
                    break
                if fixed.get("stage") == "failed":
                    raise RuntimeError(f"fixed-R60 predecessor failed: {fixed!r}")
                time.sleep(30)
        if complete():
            update_status("complete", output=str(OUTPUT), reused=True)
            return
        update_status("building_tq_dynamic_runner")
        run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(FFI_DIR / "Cargo.toml"),
            ],
            env | {"CARGO_TERM_COLOR": "never"},
        )
        run(["bash", str(BUILD)], env)
        lib = load_ffi()
        ensure_base_codec(lib)

        update_status(
            "running_tq1plus_calibrated_adaptive_insertion"
            if CALIBRATED
            else "running_tq1plus_rerank60_insertion",
            output=str(OUTPUT),
        )
        command = [
                str(DYNAMIC),
                "--dataset",
                (
                    "msmarco-qwen1024-v1-1m-plus-v2-100m-tq1plus-calibrated-target099"
                    if CALIBRATED
                    else "msmarco-qwen1024-v1-1m-plus-v2-100m-tq1plus-r60"
                ),
                "--base-train", str(BASE),
                # Chunked mode reads the 100M vectors directly from the source
                # npy files, so larger-train is only a small interface fallback.
                "--larger-train", str(BASE),
                "--query", str(QUERY),
                "--base-gt", str(BASE_GT),
                "--base-to-larger-ids", str(BASE_MAP),
                "--selected-ids", str(SELECTED),
                "--gt-dir", str(INPUTS),
                "--out", str(OUTPUT),
                "--policy", "random",
                "--fp-rerank-source", "mmap",
                "--fp-rerank-path", str(BASE),
                "--online-graph-add", "tq1",
                "--tq-code-path", str(TQ_CODES),
                "--tq-meta-path", str(TQ_META),
                "--insert-emb-root", str(INSERT_EMB_ROOT),
                "--insert-permutation", str(INSERT_PERMUTATION),
                "--insert-sizes", *map(str, INSERTS),
                "--efs", *map(str, EFS),
                "--m", "24",
                "--ef-construction", "200",
                "--rerank-n", str(RERANK),
                "--max-queries", "1000",
                "--rounds", "1",
                "--build-threads", str(THREADS),
                "--add-threads", str(THREADS),
                "--search-threads", str(THREADS),
                "--io-queue-depth", "1024",
                "--add-batch-size", "50000",
                "--independent-online-checkpoints",
                "--skip-rebuild",
                "--no-overlap",
            ]
        if CALIBRATED:
            command.extend(
                [
                    "--calibration-query", str(CAL_QUERY),
                    "--calibration-target", "0.99",
                    "--freeze-online-calibration-from-base",
                ]
            )
        run(command, env)
        if not complete():
            raise RuntimeError(f"TQ1+ dynamic result is incomplete: {OUTPUT}")
        update_status("complete", output=str(OUTPUT))
    except BaseException as exc:
        update_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
