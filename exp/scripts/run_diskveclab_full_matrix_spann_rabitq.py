#!/usr/bin/env python3
"""Held-out SPANN + RaBitQ-1bit + adaptive O_DIRECT FP matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import diskveclab_full_matrix as matrix
import run_nq_quora_spann_pipeann as helpers
import run_v051_ssd_odirect_multidataset as registry


REPO = Path(__file__).resolve().parents[1]
DISKVECLAB = Path("/home/jihyo/workspace/DiskVecLab")
SPANN_EXT = REPO / "experiment/vendor/SPANN-ext"
QUANTIZER = SPANN_EXT / "Release/quantizer"
BUILDER = SPANN_EXT / "Release/ssdserving"
SEARCH = SPANN_EXT / "Release/spann_rabitq_adaptive_benchmark"
VALIDATOR = REPO / "experiment/validate_spann_rabitq_layout.py"
FAISS_LIB = Path("/home/jihyo/anaconda3/envs/rabitq-cfr/lib")
BUILD_DIR = Path("/tmp/diskveclab_spann_ext_build")

RESULT_ROOT = matrix.RESULT_ROOT / "spann_rabitq_1bit_adaptive_fp"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/diskveclab_full_matrix_spann_rabitq")

BUILD_THREADS = 24
SEARCH_THREADS = 1
ROUNDS = 3
K = 10
TRAIN_SAMPLES = 10_000
HEAD_RATIO = 0.10
HEAD_DEGREE = 24
BUILD_INTERNAL = 64
POSTING_PAGES = 12
CANDIDATE_COUNTS = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)

RESULT_RE = re.compile(r"^SPANN_RABITQ_RESULT (?P<body>.*)$")
STATS_RE = re.compile(
    r"^SPANN_RABITQ_QUERY_STATS metric=(?P<metric>[a-z0-9_]+) (?P<body>.*)$"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def emit(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def runtime_env(threads: int) -> dict[str, str]:
    old = os.environ.get("LD_LIBRARY_PATH", "")
    return {
        **os.environ,
        "LD_LIBRARY_PATH": f"{FAISS_LIB}:{old}" if old else str(FAISS_LIB),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
    }


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> float:
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
    if returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
        raise RuntimeError(f"command failed ({returncode}): {shlex.join(cmd)}\n{tail}")
    emit(f"DONE elapsed={elapsed:.1f}s log={log_path}")
    return elapsed


def parse_kv(body: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in body.split())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safe_cleanup(path: Path) -> None:
    target = path.resolve()
    root = EXTERNAL_ROOT.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(f"refusing unsafe cleanup: {target}")
    if target.exists():
        size = directory_bytes(target)
        shutil.rmtree(target)
        emit(f"CLEANED transient artifacts path={target} bytes={size}")


def paths(key: str) -> tuple[Path, Path]:
    return EXTERNAL_ROOT / key, RESULT_ROOT / key


def config_text(dataset: registry.Dataset, work: Path) -> str:
    _, dim = registry.matrix_shape(dataset.train)
    return f"""[Base]
ValueType=Float
DistCalcMethod=L2
IndexAlgoType=BKT
Dim={dim}
VectorPath={work / 'base.rabitq1.bin'}
FullVectorPath={work / 'base.normalized.fbin'}
VectorType=DEFAULT
FullVectorType=DEFAULT
IndexDirectory={work / 'index'}
QuantizerFilePath={work / 'rabitq1.quantizer.bin'}
DeleteHeadVectors=true

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
SelectThreshold=10
SplitFactor=6
SplitThreshold=25
Ratio={HEAD_RATIO}
NumberOfThreads={BUILD_THREADS}
BKTLambdaFactor=1.0

[BuildHead]
isExecute=true
NeighborhoodSize={HEAD_DEGREE}
TPTNumber=32
TPTLeafSize=2000
MaxCheck=4096
MaxCheckForRefineGraph=4096
RefineIterations=3
NumberOfThreads={BUILD_THREADS}
BKTLambdaFactor=-1.0

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum={BUILD_INTERNAL}
ReplicaCount=8
PostingPageLimit={POSTING_PAGES}
NumberOfThreads={BUILD_THREADS}
MaxCheck=4096
TmpDir={work / 'tmp'}
SearchInternalResultNum=80
SearchPostingPageLimit={POSTING_PAGES}
UseDirectIO=true
IOThreadsPerHandler=4
EnableDataCompression=false

[SearchSSDIndex]
isExecute=false
"""


def validate_layout(dataset: registry.Dataset, work: Path, include_ssd: bool, log: Path) -> None:
    quantizer = (
        work / "index/rabitq1.quantizer.bin"
        if include_ssd
        else work / "rabitq1.quantizer.bin"
    )
    cmd = [
        "python3", str(VALIDATOR),
        "--base", str(work / "base.normalized.fbin"),
        "--quantized", str(work / "base.rabitq1.bin"),
        "--quantizer", str(quantizer),
        "--bits", "1",
        "--fp-sidecar", str(work / "base.normalized.fp32.padded4k.bin"),
    ]
    if include_ssd:
        cmd += [
            "--spann-ssd", str(work / "index/SPTAGFullList.bin"),
            "--index-loader", str(work / "index/indexloader.ini"),
        ]
    run_logged(cmd, log)


def build(dataset: registry.Dataset, force: bool) -> None:
    work, result = paths(dataset.key)
    manifest = result / "build_manifest.json"
    if not force and (work / "index/indexloader.ini").exists() and manifest.exists():
        emit(f"VALIDATE reusable SPANN+RaBitQ build {dataset.key}")
        validate_layout(dataset, work, True, result / "validate_reuse.log")
        emit(f"REUSE SPANN+RaBitQ build {dataset.key}")
        return
    if work.exists():
        safe_cleanup(work)
    (work / "index").mkdir(parents=True)
    (work / "tmp").mkdir()
    result.mkdir(parents=True, exist_ok=True)
    n, dim = registry.matrix_shape(dataset.train)
    quantize_seconds = run_logged(
        [
            str(QUANTIZER), "-d", str(dim), "-v", "Float", "-f", "DEFAULT",
            "-i", str(dataset.train), "-o", str(work / "base.rabitq1.bin"),
            "-t", str(BUILD_THREADS), "-norm", "true",
            "-oq", str(work / "rabitq1.quantizer.bin"),
            "-qt", "RaBitQ", "-rbqbits", "1",
            "-ts", str(min(TRAIN_SAMPLES, n)), "-dist", "L2",
            "-ofv", str(work / "base.normalized.fbin"),
        ],
        result / "quantize.log",
        env=runtime_env(BUILD_THREADS),
    )
    normalized_dataset = replace(dataset, train=work / "base.normalized.fbin")
    sidecar_seconds, record_bytes = helpers.normalized_padded_fp_sidecar(
        normalized_dataset, work / "base.normalized.fp32.padded4k.bin"
    )
    validate_layout(dataset, work, False, result / "validate_prebuild.log")
    config = result / "build_config.ini"
    config.write_text(config_text(dataset, work), encoding="utf-8")
    build_seconds = run_logged(
        [str(BUILDER), str(config)],
        result / "build.log",
        env=runtime_env(BUILD_THREADS),
    )
    validate_layout(dataset, work, True, result / "validate_postbuild.log")
    payload = {
        "dataset": dataset.key,
        "method": "SPANN + RaBitQ 1-bit + adaptive O_DIRECT FP",
        "nb": n,
        "dim": dim,
        "build_threads": BUILD_THREADS,
        "rabitq_bits": 1,
        "training_samples": min(TRAIN_SAMPLES, n),
        "quantize_seconds": quantize_seconds,
        "sidecar_seconds": sidecar_seconds,
        "spann_build_seconds": build_seconds,
        "build_end_to_end_seconds": quantize_seconds + sidecar_seconds + build_seconds,
        "fp_record_bytes": record_bytes,
        "fp_pages_per_vector": record_bytes // 4096,
        "index_bytes": directory_bytes(work / "index"),
        "layout_validation": "passed before and after SPANN build",
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_search(dataset: registry.Dataset, path: Path, candidate_n: int, round_id: int) -> dict[str, object]:
    result: dict[str, str] | None = None
    stats: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := RESULT_RE.match(line.strip()):
            result = parse_kv(match.group("body"))
        if match := STATS_RE.match(line.strip()):
            stats[match.group("metric")] = {
                key: float(value) for key, value in parse_kv(match.group("body")).items()
            }
    if result is None:
        raise RuntimeError(f"missing SPANN_RABITQ_RESULT: {path}")
    required = {
        "latency_us", "head_us", "posting_us", "rerank_io_us", "rerank_cpu_us",
        "posting_requests", "posting_pages_4k", "fp_vectors", "fp_pages_4k",
        "total_pages_4k",
    }
    if missing := sorted(required - set(stats)):
        raise RuntimeError(f"missing query stats {missing}: {path}")
    nb, dim = registry.matrix_shape(dataset.train)
    nq, _ = registry.matrix_shape(dataset.query)
    row: dict[str, object] = {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method": "SPANN + RaBitQ 1-bit + adaptive O_DIRECT FP",
        "family": "spann_rabitq_1bit_adaptive_fp",
        "search_parameter": "candidate_n",
        "search_parameter_value": candidate_n,
        "candidate_n": candidate_n,
        "round": round_id,
        "rounds": 1,
        "search_threads": SEARCH_THREADS,
        "build_threads": BUILD_THREADS,
        "test_queries": nq,
        "nb": nb,
        "dim": dim,
        "recall_at_10": float(result["recall_at_k"]),
        "qps": float(result["qps"]),
        "wall_seconds": float(result["wall_seconds"]),
        "storage": "SPANN postings + normalized FP sidecar on SSD",
        "io_engine": "SPANN DirectIO/io_uring postings; O_DIRECT io_uring FP in at most two batches",
    }
    for metric, distribution in stats.items():
        for statistic, value in distribution.items():
            row[f"{metric}_{statistic}"] = value
    row["graph_4k_pages_per_query_mean"] = row["posting_pages_4k_mean"]
    row["fp_vectors_per_query_mean"] = row["fp_vectors_mean"]
    row["fp_4k_pages_per_query_mean"] = row["fp_pages_4k_mean"]
    row["total_4k_pages_per_query_mean"] = row["total_pages_4k_mean"]
    return row


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate_n in CANDIDATE_COUNTS:
        group = [row for row in rows if int(row["candidate_n"]) == candidate_n]
        result = dict(group[0])
        result.pop("round", None)
        result["rounds"] = len(group)
        for key, value in list(result.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in {
                "candidate_n", "search_parameter_value", "search_threads", "build_threads",
                "test_queries", "nb", "dim", "rounds",
            }:
                result[key] = statistics.fmean(float(row[key]) for row in group)
        qps = [float(row["qps"]) for row in group]
        result["qps_stddev"] = statistics.stdev(qps) if len(qps) > 1 else 0.0
        output.append(result)
    return output


def search(dataset: registry.Dataset, rounds: int) -> None:
    work, result = paths(dataset.key)
    rows: list[dict[str, object]] = []
    for round_id in range(1, rounds + 1):
        for candidate_n in CANDIDATE_COUNTS:
            log = result / "raw_t1" / f"candidates{candidate_n}.round{round_id}.log"
            row: dict[str, object] | None = None
            if log.is_file():
                # run_logged appends this footer only after a clean child exit.
                # An interrupted log can already contain a result line, so do
                # not accept it based on parse_search alone.
                tail = log.read_text(encoding="utf-8", errors="replace")[-4096:]
                if "returncode=0" in tail:
                    try:
                        row = parse_search(dataset, log, candidate_n, round_id)
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        row = None
            if row is None:
                run_logged(
                    [
                        str(SEARCH), str(work / "index"), str(dataset.query), str(dataset.gt),
                        str(work / "base.normalized.fp32.padded4k.bin"), "1",
                        str(candidate_n), str(POSTING_PAGES), str(K),
                    ],
                    log,
                    env=runtime_env(1),
                )
                row = parse_search(dataset, log, candidate_n, round_id)
            else:
                emit(
                    f"REUSE completed SPANN+RaBitQ search {dataset.key} "
                    f"candidate_n={candidate_n} round={round_id}"
                )
            rows.append(row)
            # Keep an inspectable checkpoint even if a later point is stopped.
            write_csv(result / "search_t1_all_rounds.partial.csv", rows)
    write_csv(result / "search_t1_all_rounds.csv", rows)
    write_csv(result / "search_t1.csv", aggregate(rows))


def collect() -> None:
    rows: list[dict[str, object]] = []
    for key in matrix.CALIBRATION_ROWS:
        path = RESULT_ROOT / key / "search_t1.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    if rows:
        write_csv(RESULT_ROOT / "spann_rabitq_results.csv", rows)


def search_complete(path: Path, rounds: int) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        len(rows) == len(CANDIDATE_COUNTS)
        and all(int(row.get("rounds", 0)) == rounds for row in rows)
        and tuple(int(row.get("candidate_n", -1)) for row in rows) == CANDIDATE_COUNTS
    )


def verify_tools() -> None:
    missing = [str(path) for path in (QUANTIZER, BUILDER, SEARCH, VALIDATOR) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SPANN+RaBitQ tools: {missing}")


def ensure_tools_built() -> None:
    tracked_sources = (
        SPANN_EXT / "AnnService/inc/Core/Common/rabitqlib/utils/fht_avx.hpp",
        SPANN_EXT / "AnnService/inc/Core/Common/rabitqlib/utils/rotator.hpp",
        SPANN_EXT / "AnnService/inc/Core/Common/rabitqlib/fastscan/highacc_fastscan.hpp",
        REPO / "experiment/spann_rabitq_adaptive_odirect_benchmark.cpp",
    )
    binaries = (QUANTIZER, BUILDER, SEARCH)
    if (
        all(path.is_file() for path in binaries)
        and max(path.stat().st_mtime for path in tracked_sources)
        <= min(path.stat().st_mtime for path in binaries)
    ):
        return
    if not BUILD_DIR.is_dir():
        raise FileNotFoundError(f"missing configured SPANN build directory: {BUILD_DIR}")
    run_logged(
        [
            "cmake", "--build", str(BUILD_DIR),
            "--target", "quantizer", "ssdserving", "spann_rabitq_adaptive_benchmark",
            "-j24",
        ],
        RESULT_ROOT / "binary_build.log",
        runtime_env(BUILD_THREADS),
    )


def main() -> None:
    global RESULT_ROOT, EXTERNAL_ROOT, CANDIDATE_COUNTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(matrix.CALIBRATION_ROWS))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=list(CANDIDATE_COUNTS))
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--keep-index", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0 or any(value < K for value in args.candidate_counts):
        raise ValueError("invalid rounds/candidate counts")
    RESULT_ROOT = args.result_root.resolve()
    EXTERNAL_ROOT = args.external_root.resolve()
    helpers.RESULT_ROOT = RESULT_ROOT
    helpers.EXTERNAL_ROOT = EXTERNAL_ROOT
    CANDIDATE_COUNTS = tuple(dict.fromkeys(args.candidate_counts))
    keys = args.datasets or list(matrix.CALIBRATION_ROWS)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        collect()
        return
    ensure_tools_built()
    verify_tools()

    for key in keys:
        result_csv = RESULT_ROOT / key / "search_t1.csv"
        if search_complete(result_csv, args.rounds) and not args.force_build:
            emit(f"SKIP complete SPANN+RaBitQ {key}")
            continue
        original = registry.DATASETS[key]
        heldout = matrix.heldout_dataset(key)
        emit(f"DATASET {key} corpus={original.corpus} model={original.model}")
        build(original, args.force_build)
        search(heldout, args.rounds)
        if not args.keep_index:
            safe_cleanup(paths(key)[0])
        collect()
    emit("ALL REQUESTED SPANN+RABITQ EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
