#!/usr/bin/env python3
"""Run the NQ/MSMARCO Qwen-1024 disk-vector baseline pilot.

The shared DiskANN index stores FP32 graph records on SSD and emits both
PQ64 and RaBitQ 1-bit (ex_bits=0) in-memory routing artifacts.  PipeANN and
SPANN reuse the instrumented implementations already used by this workspace.
All builds use 24 workers and every timed search uses one worker.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RABITQ_REPO = Path("/home/jihyo/workspace/RaBitQ-Library")
sys.path.insert(0, str(RABITQ_REPO / "experiment"))

import run_nq_quora_spann_pipeann as legacy  # noqa: E402
import run_v051_ssd_odirect_multidataset as datasets_module  # noqa: E402


RESULT_ROOT = REPO / "results/nq_msmarco_qwen1024"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/diskveclab_nq_msmarco_qwen1024")
DISKANN = REPO / "src/DiskAnn"
DISKANN_BUILD = DISKANN / "build-release/apps/build_disk_index"
DISKANN_SEARCH = DISKANN / "build-release/apps/search_disk_index"
OPENBLAS_LIB = Path("/home/jihyo/anaconda3/pkgs/libopenblas-0.3.33-pthreads_h94d23a6_0/lib")
FAISS_LIB = Path("/home/jihyo/anaconda3/pkgs/libfaiss-1.10.0-cpu_openblas_hfcc2109_0/lib")

DATASET_KEYS = ("nq_qwen06_1024", "msmarco_qwen1024")
METHODS = ("diskann", "diskann-rabitq", "pipeann", "spann")
BUILD_THREADS = 24
SEARCH_THREADS = 1
R = 24
LBUILD = 200
PQ_BYTES = 64
RABITQ_BITS = 1
RABITQ_CLUSTERS = 16
BUILD_DRAM_GIB = 64
SEARCH_L = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)
K = 10
BEAM = 2

ROW_RE = re.compile(
    r"^\s*(?P<L>\d+)\s+(?P<beam>\d+)\s+"
    r"(?P<qps>[0-9.eE+-]+)\s+(?P<mean_latency>[0-9.eE+-]+)\s+"
    r"(?P<p999_latency>[0-9.eE+-]+)\s+(?P<mean_ios>[0-9.eE+-]+)\s+"
    r"(?P<mean_io_us>[0-9.eE+-]+)\s+(?P<mean_cpu_us>[0-9.eE+-]+)\s+"
    r"(?P<mean_io_latency>[0-9.eE+-]+)\s+(?P<io_latency_ratio>[0-9.eE+-]+)\s+"
    r"(?P<mean_dcs>[0-9.eE+-]+)\s+(?P<recall_pct>[0-9.eE+-]+)\s*$"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def emit(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def diskann_env(threads: int) -> dict[str, str]:
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    libs = f"{OPENBLAS_LIB}:{FAISS_LIB}"
    return {
        **os.environ,
        "LD_LIBRARY_PATH": f"{libs}:{old_ld}" if old_ld else libs,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
    }


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit(f"RUN {shlex.join(cmd)}")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={now()}\ncommand={shlex.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        returncode = process.wait()
        elapsed = time.perf_counter() - started
        log.write(f"\nfinished_at={now()}\nreturncode={returncode}\nelapsed_s={elapsed:.9f}\n")
    if returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
        raise RuntimeError(f"command failed ({returncode}): {shlex.join(cmd)}\n{tail}")
    emit(f"DONE elapsed={elapsed:.1f}s log={log_path}")
    return elapsed


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset(key: str):
    return datasets_module.DATASETS[key]


def diskann_paths(data) -> tuple[Path, Path]:
    work = EXTERNAL_ROOT / data.key / "diskann_shared_pq64_rabitq1"
    return work / "index", RESULT_ROOT / data.key


def diskann_required(prefix: Path) -> list[Path]:
    return [
        Path(f"{prefix}_disk.index"),
        Path(f"{prefix}_pq_pivots.bin"),
        Path(f"{prefix}_pq_compressed.bin"),
        Path(f"{prefix}_faiss_rabitq.index"),
    ]


def build_diskann(data) -> None:
    prefix, result = diskann_paths(data)
    manifest = result / "diskann_shared_build.json"
    required = diskann_required(prefix)
    if manifest.exists() and all(path.is_file() for path in required):
        emit(f"REUSE DiskANN shared build dataset={data.key}")
        return
    present = [path for path in required if path.exists()]
    if present:
        raise RuntimeError(f"partial DiskANN build at {prefix}; inspect before retry: {present}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n, dim = datasets_module.matrix_shape(data.train)
    elapsed = run_logged(
        [
            str(DISKANN_BUILD),
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", str(data.train),
            "--index_path_prefix", str(prefix),
            "--max_degree", str(R),
            "--Lbuild", str(LBUILD),
            "--search_DRAM_budget", str(PQ_BYTES),
            "--build_DRAM_budget", str(BUILD_DRAM_GIB),
            "--num_threads", str(BUILD_THREADS),
            "--QD", str(PQ_BYTES),
            "--PQ_disk_bytes", "0",
            "--build_PQ_bytes", "0",
            "--rabitq_bits", str(RABITQ_BITS),
            "--rabitq_clusters", str(RABITQ_CLUSTERS),
        ],
        result / "diskann_shared_build.log",
        diskann_env(BUILD_THREADS),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"DiskANN build completed without required files: {missing}")
    write_json(
        manifest,
        {
            "dataset": data.key,
            "nb": n,
            "dim": dim,
            "build_threads": BUILD_THREADS,
            "R": R,
            "Lbuild": LBUILD,
            "routing_pq_bytes": PQ_BYTES,
            "rabitq_total_bits_per_dim": RABITQ_BITS,
            "rabitq_extra_bits_per_dim": 0,
            "rabitq_clusters": RABITQ_CLUSTERS,
            "disk_records": "FP32 vector + graph neighbors",
            "shared_graph_between_methods": True,
            "build_seconds": elapsed,
            "files": {path.name: path.stat().st_size for path in required},
        },
    )


def parse_diskann(data, method: str, round_id: int, log_path: Path) -> list[dict[str, object]]:
    parsed: dict[int, dict[str, str]] = {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match:
            values = match.groupdict()
            parsed[int(values["L"])] = values
    if tuple(parsed) != SEARCH_L:
        raise RuntimeError(f"bad DiskANN rows {tuple(parsed)} in {log_path}")
    n, dim = datasets_module.matrix_shape(data.train)
    nq, _ = datasets_module.matrix_shape(data.query)
    family = "diskann_classic_pq64_fp32_nodes" if method == "diskann" else "diskann_rabitq_1bit_fp32_nodes"
    label = "DiskANN classic PQ64" if method == "diskann" else "DiskANN + RaBitQ 1-bit"
    rows: list[dict[str, object]] = []
    for search_l in SEARCH_L:
        raw = parsed[search_l]
        rows.append(
            {
                "dataset": data.key,
                "corpus": data.corpus,
                "model": data.model,
                "method": label,
                "family": family,
                "metric": "L2 on normalized embeddings (cosine-equivalent ranking)",
                "threads": SEARCH_THREADS,
                "build_threads": BUILD_THREADS,
                "Lsearch": search_l,
                "beam_width": int(raw["beam"]),
                "R": R,
                "Lbuild": LBUILD,
                "routing_bytes": PQ_BYTES if method == "diskann" else dim // 8,
                "rabitq_extra_bits": None if method == "diskann" else 0,
                "k": K,
                "nq": nq,
                "nb": n,
                "dim": dim,
                "round": round_id,
                "qps": float(raw["qps"]),
                "recall_at_10": float(raw["recall_pct"]) / 100.0,
                "mean_latency_us": float(raw["mean_latency"]),
                "p999_latency_us": float(raw["p999_latency"]),
                "mean_ios": float(raw["mean_ios"]),
                "mean_io_us": float(raw["mean_io_us"]),
                "mean_cpu_us": float(raw["mean_cpu_us"]),
                "mean_io_latency_us": float(raw["mean_io_latency"]),
                "io_latency_ratio": float(raw["io_latency_ratio"]),
                "mean_distance_computations": float(raw["mean_dcs"]),
                "fp_read_policy": "FP32 graph record read when candidate is expanded",
            }
        )
    return rows


def search_diskann(data, method: str, rounds: int) -> None:
    prefix, result = diskann_paths(data)
    method_dir = result / method.replace("-", "_")
    output = method_dir / "search_t1.csv"
    if output.exists():
        emit(f"REUSE search dataset={data.key} method={method}")
        return
    quantizer = "PQ" if method == "diskann" else "RABITQ"
    rows: list[dict[str, object]] = []
    for round_id in range(1, rounds + 1):
        log = method_dir / f"search_t1_round{round_id}.log"
        run_logged(
            [
                str(DISKANN_SEARCH),
                "--data_type", "float",
                "--dist_fn", "l2",
                "--index_path_prefix", str(prefix),
                "--result_path", str(method_dir / "native_result"),
                "--query_file", str(data.query),
                "--gt_file", str(data.gt),
                "--recall_at", str(K),
                "--search_list", *map(str, SEARCH_L),
                "--quantification_type", quantizer,
                "--beamwidth", str(BEAM),
                "--num_threads", str(SEARCH_THREADS),
                "--num_nodes_to_cache", "0",
            ],
            log,
            diskann_env(SEARCH_THREADS),
        )
        rows.extend(parse_diskann(data, method, round_id, log))
    write_csv(method_dir / "search_t1_all_rounds.csv", rows)
    if rounds == 1:
        write_csv(output, rows)
    else:
        legacy.write_csv(output, legacy.aggregate_rows(rows, "Lsearch"))


def configure_legacy() -> None:
    legacy.RESULT_ROOT = RESULT_ROOT
    legacy.EXTERNAL_ROOT = EXTERNAL_ROOT
    legacy.SEARCH_THREADS = (SEARCH_THREADS,)
    legacy.BUILD_THREADS = BUILD_THREADS


def collect_all_reports(selected) -> None:
    method_dirs = ("diskann", "diskann_rabitq", "pipeann", "spann")
    all_rows: list[dict[str, object]] = []
    matched: list[dict[str, object]] = []
    for data in selected:
        for method_dir in method_dirs:
            path = RESULT_ROOT / data.key / method_dir / "search_t1.csv"
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            all_rows.extend(rows)
            knob_name = "Lsearch" if rows and rows[0].get("Lsearch") else "internal_heads"
            for target in (0.95, 0.99):
                feasible = [row for row in rows if float(row["recall_at_10"]) >= target]
                target_reached = bool(feasible)
                candidates = feasible or rows
                chosen = max(candidates, key=lambda row: float(row["qps"])) if target_reached else max(
                    candidates, key=lambda row: float(row["recall_at_10"])
                )
                matched.append(
                    {
                        "dataset": data.key,
                        "method": chosen["method"],
                        "family": chosen["family"],
                        "target_recall": target,
                        "target_reached": target_reached,
                        "knob": knob_name,
                        "knob_value": chosen[knob_name],
                        "recall_at_10": chosen["recall_at_10"],
                        "qps": chosen["qps"],
                        "threads": chosen["threads"],
                    }
                )
    if all_rows:
        write_csv(RESULT_ROOT / "all_search_results.csv", all_rows)
    if matched:
        write_csv(RESULT_ROOT / "matched_recall_summary.csv", matched)


def verify() -> None:
    required = [DISKANN_BUILD, DISKANN_SEARCH, legacy.SPTAG_QUANTIZER, legacy.SPTAG_BUILDER,
                legacy.SPTAG_SEARCH, legacy.PIPEANN_BUILD, legacy.PIPEANN_SEARCH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing executables: {missing}")
    for key in DATASET_KEYS:
        data = dataset(key)
        for path in (data.train, data.query, data.gt):
            if not path.is_file():
                raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS, default=list(DATASET_KEYS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    configure_legacy()
    verify()
    selected = [dataset(key) for key in args.datasets]
    emit(
        f"START datasets={args.datasets} methods={args.methods} "
        f"build_threads={BUILD_THREADS} search_threads={SEARCH_THREADS} rounds={args.rounds}"
    )
    for data in selected:
        emit(f"DATASET start {data.key}")
        if "diskann" in args.methods or "diskann-rabitq" in args.methods:
            build_diskann(data)
        if "diskann" in args.methods:
            search_diskann(data, "diskann", args.rounds)
        if "diskann-rabitq" in args.methods:
            search_diskann(data, "diskann-rabitq", args.rounds)
        if "pipeann" in args.methods:
            legacy.build_pipeann(data, False)
            legacy.search_pipeann(data, args.rounds)
        if "spann" in args.methods:
            legacy.build_spann(data, False)
            legacy.search_spann(data, args.rounds)
        emit(f"DATASET done {data.key}")
    legacy.collect_reports(selected)
    collect_all_reports([dataset(key) for key in DATASET_KEYS])
    emit("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
