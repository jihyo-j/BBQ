#!/usr/bin/env python3
"""Run MRQ-SSD B320 and RaBitQ-SSD B9 on the 16-dataset held-out matrix.

The two methods are deliberately built and measured one dataset at a time.
All large transient files live below one owned external-SSD directory and are
removed after their standardized result rows have been committed.  A disk
guard terminates the active subprocess before the configured free-space floor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

import diskveclab_full_matrix as matrix
import run_v051_ssd_odirect_multidataset as registry


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = matrix.RESULT_ROOT / "mrq_rabitq_ssd"
SUMMARY = RESULT_ROOT / "summary_t1.csv"
STATUS = RESULT_ROOT / "queue_status.json"
SEARCH_THREADS = 1

EXTERNAL_MOUNT = Path("/home/smrc/samsung-nvme")
WORK_ROOT = EXTERNAL_MOUNT / "jihyo/diskveclab_full_matrix_mrq_rabitq_ssd"
ROOT_MOUNT = Path("/")
DEFAULT_EXTERNAL_FLOOR_GIB = 25.0
DEFAULT_ROOT_FLOOR_GIB = 3.0

RESQ_REPO = Path("/home/jihyo/workspace/RESQ")
RESQ_BUILD = WORK_ROOT / "_resq_build"
RESQ_PREPARE = RESQ_REPO / "script/RESQ/prepare_qwen1024_ssd.py"
RESQ_INDEX = RESQ_BUILD / "src/res_ssd_index"
RESQ_SEARCH = RESQ_BUILD / "src/res_ssd_search"
RESQ_PACK = RESQ_BUILD / "src/prepare_ssd_vectors"

RABITQ_REPO = Path("/home/jihyo/workspace/RaBitQ-SSD")
RABITQ_BUILD_TREE = RABITQ_REPO / "build"
RABITQ_CLUSTER = RABITQ_REPO / "clustering/run_kmeans.sh"
RABITQ_BUILD = RABITQ_REPO / "scripts/build_index/build_index.sh"
RABITQ_QUERY = RABITQ_REPO / "bin/querying"
PYTHON = Path("/home/jihyo/anaconda3/bin/python")

MRQ_BITS = 320
MRQ_CLUSTERS = 4096
MRQ_PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
MRQ_ASYNC_QD = 8
RABITQ_QD_PER_THREAD = 8

FIELDS = (
    "dataset",
    "corpus",
    "model",
    "method_key",
    "method",
    "search_parameter",
    "search_parameter_value",
    "nlist",
    "recall_at_10",
    "qps",
    "rounds",
    "search_threads",
    "test_queries",
    "nb",
    "dim",
    "storage",
    "io_engine",
    "source_csv",
)


class DiskFloorReached(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def gib_free(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024.0 ** 3)


def write_status(phase: str, **extra: object) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now(),
        "phase": phase,
        "external_free_gib": round(gib_free(EXTERNAL_MOUNT), 3),
        "root_free_gib": round(gib_free(ROOT_MOUNT), 3),
        **extra,
    }
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)
    print(json.dumps(payload, sort_keys=True), flush=True)


def process_start_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = raw[raw.rfind(")") + 2 :].split()
        return int(suffix[19])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def wait_for_process(pid: int, expected_start_ticks: int | None, poll: float) -> None:
    actual = process_start_ticks(pid)
    if actual is None:
        write_status("wait_target_already_finished", wait_pid=pid)
        return
    if expected_start_ticks is not None and actual != expected_start_ticks:
        raise RuntimeError(
            f"PID {pid} no longer names the queued process "
            f"(expected start {expected_start_ticks}, found {actual})"
        )
    expected = actual
    write_status("waiting", wait_pid=pid, wait_start_ticks=expected)
    while process_start_ticks(pid) == expected:
        time.sleep(poll)
    write_status("wait_target_finished", wait_pid=pid, wait_start_ticks=expected)


def check_disk(external_floor_gib: float, root_floor_gib: float) -> None:
    external = gib_free(EXTERNAL_MOUNT)
    root = gib_free(ROOT_MOUNT)
    if external < external_floor_gib or root < root_floor_gib:
        raise DiskFloorReached(
            f"disk floor reached: external={external:.2f} GiB "
            f"(floor {external_floor_gib:.2f}), root={root:.2f} GiB "
            f"(floor {root_floor_gib:.2f})"
        )


def run_guarded(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str] | None,
    external_floor_gib: float,
    root_floor_gib: float,
    poll_seconds: float = 10.0,
) -> None:
    check_disk(external_floor_gib, root_floor_gib)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, command)
                    return
                try:
                    check_disk(external_floor_gib, root_floor_gib)
                except DiskFloorReached:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                time.sleep(poll_seconds)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(rows: list[dict[str, object]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    existing = read_rows(SUMMARY) if SUMMARY.is_file() else []
    replacement = {(str(row["dataset"]), str(row["method_key"])) for row in rows}
    merged: list[dict[str, object]] = [
        row
        for row in existing
        if (row["dataset"], row["method_key"]) not in replacement
    ]
    merged.extend(rows)
    dataset_order = {key: index for index, key in enumerate(matrix.CALIBRATION_ROWS)}
    method_order = {"mrq_ssd": 0, "rabitq_ssd": 1}
    merged.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method_key"])],
            int(float(row["search_parameter_value"])),
        )
    )
    temporary = SUMMARY.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    temporary.replace(SUMMARY)


def method_complete(dataset: str, method: str, points: int, test_queries: int) -> bool:
    if not SUMMARY.is_file():
        return False
    selected = [
        row
        for row in read_rows(SUMMARY)
        if row["dataset"] == dataset and row["method_key"] == method
    ]
    return (
        len(selected) == points
        and all(int(row["search_threads"]) == SEARCH_THREADS for row in selected)
        and all(int(row["test_queries"]) == test_queries for row in selected)
        and all(float(row["qps"]) > 0 for row in selected)
    )


def safe_cleanup(path: Path) -> None:
    resolved_root = WORK_ROOT.resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise RuntimeError(f"refusing cleanup outside owned work root: {resolved}")
    if resolved.exists():
        before = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
        shutil.rmtree(resolved)
        print(f"REMOVED owned transient bytes={before}: {resolved}", flush=True)


def preserve_failure_logs(work: Path, destination: Path) -> None:
    """Copy small nested build logs before owned transient work is removed."""
    if not work.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source in work.rglob("*.log"):
        if not source.is_file():
            continue
        relative = source.relative_to(work)
        target = destination / ("__".join(relative.parts))
        shutil.copy2(source, target)


def remove_owned_files(paths: list[Path], owner_root: Path, label: str) -> None:
    resolved_owner = owner_root.resolve()
    removed_bytes = 0
    removed_count = 0
    for path in paths:
        resolved = path.resolve()
        if resolved_owner != resolved and resolved_owner not in resolved.parents:
            raise RuntimeError(f"refusing file cleanup outside {resolved_owner}: {resolved}")
        if resolved.is_file():
            removed_bytes += resolved.stat().st_size
            resolved.unlink()
            removed_count += 1
    if removed_count:
        print(
            f"REMOVED {label}: files={removed_count} bytes={removed_bytes}",
            flush=True,
        )


def nprobes_for(nlist: int) -> list[int]:
    if nlist >= 500:
        # C=511 would collapse both 512 and the final nlist to 511.  Keep the
        # sweep at 20 distinct measured points by using 448 below that edge.
        penultimate = 512 if nlist > 512 else 448
        values = [
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32,
            48, 64, 96, 128, 160, 192, 256, 384, penultimate, nlist,
        ]
    else:
        values = [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 16,
            20, 24, 32, 40, 48, 56, 64, 72, 80, nlist,
        ]
    result = sorted(set(min(nlist, value) for value in values))
    if len(result) != 20:
        raise RuntimeError(f"expected 20 RaBitQ-SSD nprobes for C={nlist}: {result}")
    return result


def ensure_resq_binaries(args: argparse.Namespace) -> None:
    sources = [
        RESQ_REPO / "src/ResBitQ/res_index.cpp",
        RESQ_REPO / "src/ResBitQ/res_search.cpp",
        RESQ_REPO / "include/ivf_res.h",
    ]
    binaries = [RESQ_INDEX, RESQ_SEARCH, RESQ_PACK]
    newest_source = max(path.stat().st_mtime for path in sources)
    rebuild = any(not path.is_file() for path in binaries) or any(
        path.stat().st_mtime < newest_source for path in binaries[:2]
    )
    if not rebuild:
        return
    if not (RESQ_BUILD / "CMakeCache.txt").is_file():
        RESQ_BUILD.mkdir(parents=True, exist_ok=True)
        run_guarded(
            [
                "cmake", "-S", str(RESQ_REPO), "-B", str(RESQ_BUILD),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            RESULT_ROOT / "logs/resq_configure.log",
            cwd=RESQ_REPO,
            env=os.environ.copy(),
            external_floor_gib=args.external_floor_gib,
            root_floor_gib=args.root_floor_gib,
        )
    run_guarded(
        [
            "cmake", "--build", str(RESQ_BUILD), "--target",
            "res_ssd_index", "res_ssd_search", "prepare_ssd_vectors",
            "-j", "2",
        ],
        RESULT_ROOT / "logs/resq_build.log",
        cwd=RESQ_REPO,
        env=os.environ.copy(),
        external_floor_gib=args.external_floor_gib,
        root_floor_gib=args.root_floor_gib,
    )


def ensure_rabitq_binaries(args: argparse.Namespace) -> None:
    if not (RABITQ_BUILD_TREE / "CMakeCache.txt").is_file():
        raise FileNotFoundError(f"missing RaBitQ-SSD CMake build: {RABITQ_BUILD_TREE}")
    run_guarded(
        [
            "cmake", "--build", str(RABITQ_BUILD_TREE), "--target",
            "build_invlist", "build_coarse", "querying", "-j", "2",
        ],
        RESULT_ROOT / "logs/rabitq_build.log",
        cwd=RABITQ_REPO,
        env=os.environ.copy(),
        external_floor_gib=args.external_floor_gib,
        root_floor_gib=args.root_floor_gib,
    )


def run_method_with_retries(
    key: str,
    method: str,
    operation: Callable[[str, argparse.Namespace], None],
    args: argparse.Namespace,
) -> None:
    for attempt in range(1, args.max_method_attempts + 1):
        try:
            operation(key, args)
            return
        except DiskFloorReached:
            raise
        except Exception as error:
            if attempt >= args.max_method_attempts:
                raise
            write_status(
                "retrying",
                dataset=key,
                method=method,
                failed_attempt=attempt,
                next_attempt=attempt + 1,
                max_attempts=args.max_method_attempts,
                error=f"{type(error).__name__}: {error}",
            )
            time.sleep(args.retry_delay_seconds)


def dataset_info(key: str) -> tuple[registry.Dataset, object, int, int]:
    dataset = registry.DATASETS[key]
    split = matrix.split_for(key)
    nb, dim = registry.matrix_shape(dataset.train)
    return dataset, split, nb, dim


def run_mrq(key: str, args: argparse.Namespace) -> None:
    dataset, split, nb, dim = dataset_info(key)
    if method_complete(key, "mrq_ssd", len(MRQ_PROBES), split.test_rows):
        safe_cleanup(WORK_ROOT / "mrq" / key)
        print(f"REUSE completed MRQ-SSD summary: {key}", flush=True)
        return

    work = WORK_ROOT / "mrq" / key
    mrq_name = f"diskveclab_{key}"
    manifest_path = work / "prepare_manifest.json"
    required = [
        manifest_path,
        work / f"{mrq_name}.ssdvec",
        work / f"{mrq_name}_query.fvecs",
        work / f"{mrq_name}_groundtruth.ivecs",
        work / f"{mrq_name}_pca.fvecs",
        work / f"{mrq_name}_mean.fvecs",
        work / f"RESP_C{MRQ_CLUSTERS}_B{MRQ_BITS}.fvecs",
        work / f"ivf_res{MRQ_CLUSTERS}_B{MRQ_BITS}.ssd.index",
    ]
    reusable = False
    if all(path.is_file() and path.stat().st_size > 0 for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reusable = (
            manifest.get("dataset") == mrq_name
            and int(manifest.get("bits", -1)) == MRQ_BITS
            and int(manifest.get("clusters", -1)) == MRQ_CLUSTERS
            and int(manifest.get("seed", -1)) == args.seed
            and manifest.get("train") == str(dataset.train)
            and manifest.get("query") == str(split.test_query)
            and manifest.get("gt") == str(split.test_gt)
        )
    if work.exists() and not reusable:
        safe_cleanup(work)
    work.mkdir(parents=True, exist_ok=True)
    logs = RESULT_ROOT / key / "mrq_ssd/logs"
    raw_results = RESULT_ROOT / key / "mrq_ssd/raw"
    raw_results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(args.build_threads),
            "OPENBLAS_NUM_THREADS": str(args.build_threads),
            "MKL_NUM_THREADS": str(args.build_threads),
            "NUMEXPR_NUM_THREADS": str(args.build_threads),
        }
    )
    write_status("running", dataset=key, method="mrq_ssd", stage="prepare")
    try:
        projected = work / f"{mrq_name}_proj.fvecs"
        ssd_vectors = work / f"{mrq_name}.ssdvec"
        if reusable:
            print(f"REUSE completed MRQ-SSD artifacts: {work}", flush=True)
        else:
            run_guarded(
                [
                    str(PYTHON), str(RESQ_PREPARE),
                    "--dataset", mrq_name,
                    "--train", str(dataset.train),
                    "--query", str(split.test_query),
                    "--gt", str(split.test_gt),
                    "--output", str(work),
                    "--bits", str(MRQ_BITS),
                    "--clusters", str(MRQ_CLUSTERS),
                    "--pca-sample", str(args.pca_sample),
                    "--kmeans-sample", str(args.kmeans_sample),
                    "--kmeans-iterations", str(args.kmeans_iterations),
                    "--threads", str(args.build_threads),
                    "--seed", str(args.seed),
                ],
                logs / "prepare.log",
                cwd=RESQ_REPO,
                env=env,
                external_floor_gib=args.external_floor_gib,
                root_floor_gib=args.root_floor_gib,
            )
            write_status("running", dataset=key, method="mrq_ssd", stage="build_index")
            run_guarded(
                [
                    str(RESQ_INDEX), "-d", mrq_name, "-s", str(work) + "/",
                    "-b", str(MRQ_BITS), "-c", str(MRQ_CLUSTERS),
                ],
                logs / "build_index.log",
                cwd=RESQ_REPO,
                env=env,
                external_floor_gib=args.external_floor_gib,
                root_floor_gib=args.root_floor_gib,
            )
            write_status("running", dataset=key, method="mrq_ssd", stage="pack_ssd")
            run_guarded(
                [str(RESQ_PACK), str(projected), str(ssd_vectors)],
                logs / "pack_ssd.log",
                cwd=RESQ_REPO,
                env=env,
                external_floor_gib=args.external_floor_gib,
                root_floor_gib=args.root_floor_gib,
            )
        remove_owned_files(
            [
                projected,
                work / f"p{mrq_name}_dist_to_centroid_{MRQ_CLUSTERS}.fvecs",
                work / f"p{mrq_name}_cluster_id_{MRQ_CLUSTERS}.ivecs",
                work / f"RES_Rand_C{MRQ_CLUSTERS}_B{MRQ_BITS}.Ivecs",
                work / f"RES_x0_C{MRQ_CLUSTERS}_B{MRQ_BITS}.fvecs",
                work / f"RESCentroid_C{MRQ_CLUSTERS}_B{MRQ_BITS}.fvecs",
            ],
            work,
            "MRQ build-only artifacts",
        )
        search_env = env.copy()
        search_env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        write_status("running", dataset=key, method="mrq_ssd", stage="search")
        run_guarded(
            [
                str(RESQ_SEARCH), "-d", mrq_name, "-s", str(work) + "/",
                "-r", str(raw_results) + "/", "-k", "10",
                "-b", str(MRQ_BITS), "-c", str(MRQ_CLUSTERS),
                "--rounds", str(args.rounds), "--io-mode", "direct",
                "--async-qd", str(MRQ_ASYNC_QD),
                "--vector-file", str(ssd_vectors),
                "--probes", ",".join(map(str, MRQ_PROBES)),
            ],
            logs / "search.log",
            cwd=RESQ_REPO,
            env=search_env,
            external_floor_gib=args.external_floor_gib,
            root_floor_gib=args.root_floor_gib,
        )
        raw = raw_results / (
            f"{mrq_name}_mrq_ssd_C{MRQ_CLUSTERS}_B{MRQ_BITS}_direct.csv"
        )
        measured = read_rows(raw)
        if [int(row["nprobe"]) for row in measured] != list(MRQ_PROBES):
            raise RuntimeError(f"unexpected MRQ sweep in {raw}")
        if any(int(row["queries"]) != split.test_rows for row in measured):
            raise RuntimeError(f"MRQ held-out query mismatch in {raw}")
        if any(row.get("io_mode") != "direct" or int(row.get("io_qd", 0)) != MRQ_ASYNC_QD
               for row in measured):
            raise RuntimeError(f"MRQ async O_DIRECT/QD mismatch in {raw}")
        rows: list[dict[str, object]] = []
        for row in measured:
            rows.append(
                {
                    "dataset": key,
                    "corpus": dataset.corpus,
                    "model": dataset.model,
                    "method_key": "mrq_ssd",
                    "method": "MRQ-SSD B320 async QD8",
                    "search_parameter": "nprobe",
                    "search_parameter_value": int(row["nprobe"]),
                    "nlist": MRQ_CLUSTERS,
                    "recall_at_10": float(row["recall"]),
                    "qps": float(row["end_to_end_qps"]),
                    "rounds": int(row["rounds"]),
                    "search_threads": 1,
                    "test_queries": int(row["queries"]),
                    "nb": nb,
                    "dim": dim,
                    "storage": "RAM MRQ B320 codes + SSD FP32 projected vectors",
                    "io_engine": "libaio O_DIRECT QD8",
                    "source_csv": str(raw),
                }
            )
        write_summary(rows)
    finally:
        safe_cleanup(work)


def rabitq_paths(key: str, nlist: int, dim: int, work: Path) -> tuple[Path, Path, Path]:
    token = f"C{nlist}_cpu_B9_memdim{dim}_ip_ordered"
    index_dir = work / key / f"idx_{token}"
    stem = index_dir / f"ss_{key}_{token}"
    return stem.with_suffix(".base"), stem.with_suffix(".ssd"), stem.with_suffix(".coarse_flat")


def run_rabitq(key: str, args: argparse.Namespace) -> None:
    dataset, split, nb, dim = dataset_info(key)
    nlist = (nb + 1023) // 1024
    nprobes = nprobes_for(nlist)
    if method_complete(key, "rabitq_ssd", len(nprobes), split.test_rows):
        safe_cleanup(WORK_ROOT / "rabitq" / key)
        print(f"REUSE completed RaBitQ-SSD summary: {key}", flush=True)
        return

    method_work = WORK_ROOT / "rabitq" / key
    if method_work.exists():
        safe_cleanup(method_work)
    method_work.mkdir(parents=True, exist_ok=True)
    index_root = WORK_ROOT / "rabitq"
    clustering = method_work / "clustering"
    logs = RESULT_ROOT / key / "rabitq_ssd/logs"
    raw_results = RESULT_ROOT / key / "rabitq_ssd/raw"
    raw_results.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "DATA_ENV": str(dataset.train),
            "OUTDIR_ENV": str(clustering),
            "INDEX_ROOT": str(index_root),
            "CL_ENV": str(clustering),
            "C_ENV": str(nlist),
            "MEMDIM_ENV": str(dim),
            "METRIC_ENV": "ip",
            "ORDER_ENV": "ordered",
            "STORE_ENV": "exbits",
            "B_ENV": "9",
            "COARSE_ENV": "flat",
            "BUDGET_GB_ENV": str(args.build_budget_gib),
            "PYTHON_ENV": str(PYTHON),
            "RABITQ_BUILD_SCRATCH_DIR": str(method_work / "scratch"),
            "OMP_NUM_THREADS": str(args.build_threads),
        }
    )
    try:
        write_status("running", dataset=key, method="rabitq_ssd", stage="clustering")
        run_guarded(
            [str(RABITQ_CLUSTER), key],
            logs / "clustering.log",
            cwd=RABITQ_REPO,
            env=env,
            external_floor_gib=args.external_floor_gib,
            root_floor_gib=args.root_floor_gib,
        )
        write_status("running", dataset=key, method="rabitq_ssd", stage="build_index")
        run_guarded(
            [str(RABITQ_BUILD), key],
            logs / "build_index.log",
            cwd=RABITQ_REPO,
            env=env,
            external_floor_gib=args.external_floor_gib,
            root_floor_gib=args.root_floor_gib,
        )
        base, ssd, coarse = rabitq_paths(key, nlist, dim, index_root)
        missing = [str(path) for path in (base, ssd, coarse) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing RaBitQ-SSD artifacts: {missing}")
        round_rows: list[list[dict[str, str]]] = []
        search_env = env.copy()
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            search_env[variable] = str(args.search_threads)
        search_env["RABITQ_QD_PER_THREAD"] = str(RABITQ_QD_PER_THREAD)
        for round_id in range(1, args.rounds + 1):
            raw = raw_results / f"round{round_id}.csv"
            write_status(
                "running", dataset=key, method="rabitq_ssd",
                stage="search", round=round_id,
            )
            run_guarded(
                [
                    str(RABITQ_QUERY), str(base), str(ssd), str(coarse),
                    str(split.test_query), str(split.test_gt),
                    str(args.search_threads),
                    "topk=10", f"memdim={dim}", "use_hacc=true",
                    "drain_alpha=0.4",
                    "nprobes=" + ",".join(map(str, nprobes)),
                    f"csv={raw}",
                    f"run_id={key}_b9_t{args.search_threads}_round{round_id}",
                    "system=RaBitQ-SSD-B9", f"dataset={key}",
                    "metric_space=ip",
                ],
                logs / f"search_round{round_id}.log",
                cwd=RABITQ_REPO,
                env=search_env,
                external_floor_gib=args.external_floor_gib,
                root_floor_gib=args.root_floor_gib,
            )
            measured = read_rows(raw)
            if [int(row["op_param"]) for row in measured] != nprobes:
                raise RuntimeError(f"unexpected RaBitQ-SSD sweep in {raw}")
            if any(int(row["nq"]) != split.test_rows for row in measured):
                raise RuntimeError(f"RaBitQ-SSD held-out query mismatch in {raw}")
            round_rows.append(measured)

        rows: list[dict[str, object]] = []
        for point_index, nprobe in enumerate(nprobes):
            selected = [round_rows[index][point_index] for index in range(args.rounds)]
            rows.append(
                {
                    "dataset": key,
                    "corpus": dataset.corpus,
                    "model": dataset.model,
                    "method_key": "rabitq_ssd",
                    "method": "RaBitQ-SSD B9",
                    "search_parameter": "nprobe",
                    "search_parameter_value": nprobe,
                    "nlist": nlist,
                    "recall_at_10": float(np.mean([float(row["recall_at_topk"]) for row in selected])),
                    "qps": float(np.mean([float(row["qps"]) for row in selected])),
                    "rounds": args.rounds,
                    "search_threads": args.search_threads,
                    "test_queries": split.test_rows,
                    "nb": nb,
                    "dim": dim,
                    "storage": "RAM RaBitQ sign prefix + SSD B9 extra bits",
                    "io_engine": f"libaio O_DIRECT QD{RABITQ_QD_PER_THREAD}/thread",
                    "source_csv": ";".join(str(raw_results / f"round{i}.csv") for i in range(1, args.rounds + 1)),
                }
            )
        write_summary(rows)
    except BaseException:
        preserve_failure_logs(method_work, logs / "failed_work")
        raise
    finally:
        safe_cleanup(method_work)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(matrix.CALIBRATION_ROWS))
    parser.add_argument("--methods", nargs="+", choices=("mrq", "rabitq"), default=("mrq", "rabitq"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=20)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--build-budget-gib", type=int, default=128)
    parser.add_argument("--pca-sample", type=int, default=100_000)
    parser.add_argument("--kmeans-sample", type=int, default=262_144)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--external-floor-gib", type=float, default=DEFAULT_EXTERNAL_FLOOR_GIB)
    parser.add_argument("--root-floor-gib", type=float, default=DEFAULT_ROOT_FLOOR_GIB)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--wait-start-ticks", type=int)
    parser.add_argument("--wait-poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-method-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.rounds <= 0 or args.build_threads <= 0 or args.search_threads <= 0:
        parser.error("rounds, build threads, and search threads must be positive")
    if args.wait_poll_seconds <= 0:
        parser.error("wait poll interval must be positive")
    if args.max_method_attempts <= 0 or args.retry_delay_seconds < 0:
        parser.error("max method attempts must be positive and retry delay non-negative")
    return args


def main() -> int:
    global SUMMARY, SEARCH_THREADS
    args = parse_args()
    SEARCH_THREADS = args.search_threads
    SUMMARY = RESULT_ROOT / f"summary_t{SEARCH_THREADS}.csv"
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    if args.wait_pid is not None:
        wait_for_process(args.wait_pid, args.wait_start_ticks, args.wait_poll_seconds)
    datasets = args.datasets or list(matrix.CALIBRATION_ROWS)
    try:
        check_disk(args.external_floor_gib, args.root_floor_gib)
        ensure_resq_binaries(args)
        ensure_rabitq_binaries(args)
        for key in datasets:
            if "mrq" in args.methods:
                run_method_with_retries(key, "mrq_ssd", run_mrq, args)
                write_status("checkpoint", dataset=key, completed_method="mrq_ssd")
            if "rabitq" in args.methods:
                run_method_with_retries(key, "rabitq_ssd", run_rabitq, args)
                write_status("checkpoint", dataset=key, completed_method="rabitq_ssd")
        if datasets == list(matrix.CALIBRATION_ROWS) and set(args.methods) == {"mrq", "rabitq"}:
            write_status("plotting")
            run_guarded(
                [str(PYTHON), str(REPO / "experiment/plot_diskveclab_full_matrix_all_methods_recall_qps.py")],
                RESULT_ROOT / "logs/plot_all_10methods.log",
                cwd=REPO,
                env=os.environ.copy(),
                external_floor_gib=args.external_floor_gib,
                root_floor_gib=args.root_floor_gib,
            )
            safe_cleanup(RESQ_BUILD)
        write_status("complete", datasets=datasets, methods=args.methods)
        return 0
    except DiskFloorReached as error:
        write_status("stopped_disk_floor", error=str(error))
        print(str(error), file=sys.stderr, flush=True)
        return 75
    except BaseException as error:
        write_status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
