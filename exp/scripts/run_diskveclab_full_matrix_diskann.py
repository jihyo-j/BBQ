#!/usr/bin/env python3
"""DiskANN PQ and DiskANN+RaBitQ-1bit on common held-out queries.

One shared FP32-node disk graph is built per dataset.  PQ and RaBitQ-1bit are
alternative in-memory routing representations over that same graph.  PQ can
use the historical fixed 64-byte code or a one-bit-equivalent code budget of
``dimension / 8`` bytes per vector.  The instrumented DiskVecLab searcher
reports submitted graph requests, actual 4 KiB sectors, and therefore the FP
vectors fetched while expanding SSD nodes.
"""

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
from pathlib import Path

import diskveclab_full_matrix as matrix
import run_v051_ssd_odirect_multidataset as registry


REPO = Path(__file__).resolve().parents[1]
DISKVEC = REPO.parent / "DiskVecLab"
DISKANN = DISKVEC / "src/DiskAnn"
BUILD = DISKANN / "build-release/apps/build_disk_index"
SEARCH = DISKANN / "build-release/apps/search_disk_index"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/diskveclab_full_matrix/diskann")
RESULT_ROOT = matrix.RESULT_ROOT / "diskann"
OPENBLAS_LIB = Path("/home/jihyo/anaconda3/pkgs/libopenblas-0.3.33-pthreads_h94d23a6_0/lib")
FAISS_LIB = Path("/home/jihyo/anaconda3/pkgs/libfaiss-1.10.0-cpu_openblas_hfcc2109_0/lib")

SEARCH_L = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)
M = 24
LBUILD = 200
FIXED_RAM_PQ_BYTES = 64
PQ_CODE_BUDGET = "fixed64"
RABITQ_BITS = 1
RABITQ_CLUSTERS = 16
BEAM = 2

REUSED_PREFIXES = {
    "nq_qwen06_1024": Path(
        "/home/smrc/samsung-nvme/jihyo/diskveclab_nq_msmarco_qwen1024/"
        "nq_qwen06_1024/diskann_shared_pq64_rabitq1/index"
    ),
    "msmarco_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/diskveclab_nq_msmarco_qwen1024/"
        "msmarco_qwen1024/diskann_shared_pq64_rabitq1/index"
    ),
}

ROW_RE = re.compile(
    r"^\s*(?P<L>\d+)\s+(?P<beam>\d+)\s+"
    r"(?P<qps>[0-9.eE+-]+)\s+(?P<mean_latency>[0-9.eE+-]+)\s+"
    r"(?P<p999_latency>[0-9.eE+-]+)\s+(?P<mean_ios>[0-9.eE+-]+)\s+"
    r"(?P<mean_io_us>[0-9.eE+-]+)\s+(?P<mean_cpu_us>[0-9.eE+-]+)\s+"
    r"(?P<mean_io_latency>[0-9.eE+-]+)\s+(?P<io_latency_ratio>[0-9.eE+-]+)\s+"
    r"(?P<mean_dcs>[0-9.eE+-]+)\s+(?P<recall_pct>[0-9.eE+-]+)\s*$"
)
STATS_RE = re.compile(
    r"^QUERY_STATS L=(?P<L>\d+) metric=(?P<metric>[a-z0-9_]+) "
    r"mean=(?P<mean>[-+0-9.eE]+) min=(?P<min>[-+0-9.eE]+) "
    r"max=(?P<max>[-+0-9.eE]+) p95=(?P<p95>[-+0-9.eE]+) "
    r"p99=(?P<p99>[-+0-9.eE]+)$"
)
REQUIRED_STATS = (
    "fp_vectors",
    "graph_io_requests",
    "graph_4k_pages",
    "total_4k_pages",
    "latency_us",
)


def environment(threads: int) -> dict[str, str]:
    old = os.environ.get("LD_LIBRARY_PATH", "")
    libs = f"{OPENBLAS_LIB}:{FAISS_LIB}"
    return {
        **os.environ,
        "LD_LIBRARY_PATH": f"{libs}:{old}" if old else libs,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
    }


def pq_bytes_for_dim(dim: int) -> int:
    if PQ_CODE_BUDGET == "fixed64":
        return FIXED_RAM_PQ_BYTES
    if PQ_CODE_BUDGET != "one-bit":
        raise ValueError(f"unknown PQ code budget: {PQ_CODE_BUDGET}")
    if dim <= 0 or dim % 8:
        raise ValueError(f"one-bit-equivalent PQ requires dimension divisible by 8: {dim}")
    return dim // 8


def prefix(key: str) -> tuple[Path, bool]:
    if PQ_CODE_BUDGET == "fixed64" and key in REUSED_PREFIXES:
        return REUSED_PREFIXES[key], False
    return EXTERNAL_ROOT / key / "index/index", True


def required_files(index_prefix: Path) -> tuple[Path, ...]:
    return (
        Path(f"{index_prefix}_disk.index"),
        Path(f"{index_prefix}_pq_pivots.bin"),
        Path(f"{index_prefix}_pq_compressed.bin"),
        Path(f"{index_prefix}_faiss_rabitq.index"),
    )


def run_logged(command: list[str], log: Path, threads: int) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {shlex.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment(threads),
        )
        assert process.stdout is not None
        chunks: list[str] = []
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            chunks.append(line)
        if process.wait():
            raise RuntimeError(f"command failed; see {log}")
    return "".join(chunks)


def build(dataset: registry.Dataset, build_threads: int, build_dram_gib: int) -> tuple[Path, bool]:
    index_prefix, owned = prefix(dataset.key)
    nb, dim = registry.matrix_shape(dataset.train)
    pq_bytes = pq_bytes_for_dim(dim)
    search_dram_budget_gib = nb * pq_bytes / 1024**3
    required = required_files(index_prefix)
    manifest_path = RESULT_ROOT / dataset.key / "build_manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compatible = (
        int(manifest.get("ram_pq_bytes_per_vector", -1)) == pq_bytes
        and str(manifest.get("pq_code_budget", "")) == PQ_CODE_BUDGET
        and str(manifest.get("train", "")) == str(dataset.train)
    )
    if all(path.is_file() for path in required) and (compatible or not owned):
        print(f"REUSE shared DiskANN index {dataset.key}: {index_prefix}", flush=True)
        return index_prefix, owned
    partial = [path for path in required if path.exists()]
    if partial:
        if not owned:
            raise RuntimeError(f"partial reused index for {dataset.key}: {partial}")
        # This directory is an experiment-owned transient build.  A stopped
        # DiskANN builder cannot resume these files safely, so rebuild only
        # this validated dataset/index directory while preserving all result
        # CSVs and logs in RESULT_ROOT.
        target = index_prefix.parent.resolve()
        external_root = EXTERNAL_ROOT.resolve()
        if target.name != "index" or external_root not in target.parents:
            raise RuntimeError(f"refusing unsafe partial-index cleanup: {target}")
        shutil.rmtree(target)
        print(
            f"REMOVED interrupted owned DiskANN index {dataset.key}: {target}",
            flush=True,
        )
    index_prefix.parent.mkdir(parents=True, exist_ok=True)
    run_logged(
        [
            str(BUILD),
            "--data_type", "float",
            "--dist_fn", "l2",
            "--data_path", str(dataset.train),
            "--index_path_prefix", str(index_prefix),
            "--max_degree", str(M),
            "--Lbuild", str(LBUILD),
            "--search_DRAM_budget", f"{search_dram_budget_gib:.9f}",
            "--build_DRAM_budget", str(build_dram_gib),
            "--num_threads", str(build_threads),
            "--QD", str(pq_bytes),
            "--PQ_disk_bytes", "0",
            "--build_PQ_bytes", "0",
            "--rabitq_bits", str(RABITQ_BITS),
            "--rabitq_clusters", str(RABITQ_CLUSTERS),
        ],
        RESULT_ROOT / dataset.key / "build.log",
        build_threads,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"build completed with missing files: {missing}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "dataset": dataset.key,
                "train": str(dataset.train),
                "nb": nb,
                "dim": dim,
                "M": M,
                "Lbuild": LBUILD,
                "ram_pq_bytes_per_vector": pq_bytes,
                "search_DRAM_budget_GiB": search_dram_budget_gib,
                "pq_code_budget": PQ_CODE_BUDGET,
                "pq_nominal_bits_per_dimension": 8.0 * pq_bytes / dim,
                "rabitq_data_bits_per_dimension": RABITQ_BITS,
                "rabitq_clusters": RABITQ_CLUSTERS,
                "disk_node_layout": "FP32 vector plus graph neighbors",
                "num_nodes_to_cache": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return index_prefix, owned


def parse_round(
    dataset: registry.Dataset,
    method: str,
    round_id: int,
    text: str,
    search_threads: int,
) -> list[dict[str, object]]:
    table: dict[int, dict[str, str]] = {}
    distributions: dict[int, dict[str, dict[str, float]]] = {}
    for line in text.splitlines():
        match = ROW_RE.match(line)
        if match:
            raw = match.groupdict()
            table[int(raw["L"])] = raw
        match = STATS_RE.match(line.strip())
        if match:
            raw = match.groupdict()
            distributions.setdefault(int(raw["L"]), {})[raw["metric"]] = {
                name: float(raw[name]) for name in ("mean", "min", "max", "p95", "p99")
            }
    if tuple(table) != SEARCH_L:
        raise RuntimeError(f"bad search rows for {dataset.key}/{method}: {tuple(table)}")
    nb, dim = registry.matrix_shape(dataset.train)
    nq, _ = registry.matrix_shape(dataset.query)
    pq_bytes = pq_bytes_for_dim(dim)
    rows: list[dict[str, object]] = []
    for search_l in SEARCH_L:
        missing = set(REQUIRED_STATS) - set(distributions.get(search_l, {}))
        if missing:
            raise RuntimeError(f"missing query stats L={search_l}: {sorted(missing)}")
        raw = table[search_l]
        stats = distributions[search_l]
        latency = stats["latency_us"]
        # SIGSTOP/SIGCONT (or a comparable multi-minute suspension) is counted
        # by DiskANN's wall-clock timer.  Reject that round instead of quietly
        # aggregating a poisoned QPS point.  The absolute guard leaves ordinary
        # SSD tail latency alone, while the ratio guard identifies a single
        # suspension-sized query relative to the measured p99.
        suspension_limit_us = max(
            60_000_000.0,
            1_000.0 * max(latency["p99"], 1.0),
        )
        if latency["max"] > suspension_limit_us:
            raise RuntimeError(
                f"suspension-like latency outlier for {dataset.key}/{method} "
                f"round={round_id} L={search_l}: max_us={latency['max']:.0f} "
                f"p99_us={latency['p99']:.0f} limit_us={suspension_limit_us:.0f}"
            )
        row: dict[str, object] = {
            "dataset": dataset.key,
            "corpus": dataset.corpus,
            "model": dataset.model,
            "method": (
                (
                    f"DiskANN PQ{pq_bytes} routing (1-bit-equivalent budget) + FP32 SSD nodes"
                    if PQ_CODE_BUDGET == "one-bit"
                    else "DiskANN classic PQ64 routing + FP32 SSD nodes"
                )
                if method == "diskann_classic"
                else "DiskANN RaBitQ 1-bit routing + adaptive FP32 SSD nodes"
            ),
            "family": method,
            "Lsearch": search_l,
            "beam_width": int(raw["beam"]),
            "M": M,
            "Lbuild": LBUILD,
            "k": 10,
            "nq": nq,
            "nb": nb,
            "dim": dim,
            "round": round_id,
            "recall_at_10": float(raw["recall_pct"]) / 100.0,
            "qps": float(raw["qps"]),
            "mean_latency_us": float(raw["mean_latency"]),
            "mean_io_us": float(raw["mean_io_us"]),
            "fp_read_policy": "read FP32 node when candidate is expanded",
            "storage": "SSD O_DIRECT",
            "io_engine": "Linux native AIO batch (io_submit/io_getevents)",
            "search_threads": search_threads,
            "routing_representation": (
                "PQ8 subspace IDs" if method == "diskann_classic" else "RaBitQ 1-bit"
            ),
            "routing_code_budget": (
                PQ_CODE_BUDGET if method == "diskann_classic" else "actual-one-bit"
            ),
            "routing_code_bytes_per_vector": (
                pq_bytes if method == "diskann_classic" else dim // 8
            ),
            "routing_metadata_bytes_per_vector": (
                0 if method == "diskann_classic" else 16
            ),
            "routing_nominal_bits_per_dimension": (
                8.0 * pq_bytes / dim if method == "diskann_classic" else 1.0
            ),
        }
        for metric in REQUIRED_STATS:
            for statistic, value in stats[metric].items():
                row[f"{metric}_per_query_{statistic}"] = value
        # There is no separate final rerank: graph-node FP reads are the exact
        # refinements.  Keep explicit zero-valued final-rerank columns too.
        row["final_rerank_vectors_per_query_mean"] = 0
        row["final_rerank_4k_pages_per_query_mean"] = 0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    numeric = [
        key
        for key in rows[0]
        if key not in {
            "dataset", "corpus", "model", "method", "family", "Lsearch",
            "beam_width", "M", "Lbuild", "k", "nq", "nb", "dim", "round",
            "fp_read_policy", "storage", "io_engine", "search_threads",
            "routing_representation", "routing_code_budget",
        }
    ]
    for search_l in SEARCH_L:
        selected = [row for row in rows if int(row["Lsearch"]) == search_l]
        row = dict(selected[0])
        row.pop("round")
        row["rounds"] = len(selected)
        for key in numeric:
            row[key] = statistics.fmean(float(item[key]) for item in selected)
        row["qps_stddev"] = statistics.stdev(float(item["qps"]) for item in selected) if len(selected) > 1 else 0.0
        output.append(row)
    return output


def complete(path: Path, rounds: int, search_threads: int) -> bool:
    if not path.is_file():
        return False
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    return (
        len(rows) == len(SEARCH_L)
        and all(int(row.get("rounds", 0)) == rounds for row in rows)
        and all(int(row.get("search_threads", 0)) == search_threads for row in rows)
    )


def reusable_round(
    dataset: registry.Dataset,
    method: str,
    round_id: int,
    log: Path,
    search_threads: int,
) -> list[dict[str, object]] | None:
    """Return a previously completed round only when it is provably current.

    A supervisor retry used to repeat every round of an incomplete method.  In
    particular, stopping a live search poisoned only its current wall-clock
    sample but unnecessarily discarded earlier clean rounds.  A round log is
    safe to reuse when it is newer than the index manifest and held-out inputs,
    and the normal strict parser accepts all expected L values and statistics.
    The parser also rejects suspension-sized latency outliers.
    """
    manifest = RESULT_ROOT / dataset.key / "build_manifest.json"
    required = (log, manifest, dataset.query, dataset.gt, SEARCH)
    if not all(path.is_file() for path in required):
        return None
    freshness_floor = max(
        manifest.stat().st_mtime_ns,
        dataset.query.stat().st_mtime_ns,
        dataset.gt.stat().st_mtime_ns,
        SEARCH.stat().st_mtime_ns,
    )
    if log.stat().st_mtime_ns < freshness_floor:
        return None
    try:
        rows = parse_round(
            dataset,
            method,
            round_id,
            log.read_text(encoding="utf-8"),
            search_threads,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(
            f"RERUN invalid cached round {dataset.key}/{method} "
            f"round={round_id}: {error}",
            flush=True,
        )
        return None
    print(
        f"REUSE valid round {dataset.key}/{method} round={round_id}: {log}",
        flush=True,
    )
    return rows


def search_method(
    dataset: registry.Dataset,
    index_prefix: Path,
    method: str,
    rounds: int,
    search_threads: int,
) -> None:
    result = RESULT_ROOT / dataset.key / method / f"search_t{search_threads}.csv"
    if complete(result, rounds, search_threads):
        print(f"SKIP complete {dataset.key}/{method}", flush=True)
        return
    quantizer = "PQ" if method == "diskann_classic" else "RABITQ"
    all_rows: list[dict[str, object]] = []
    for round_id in range(1, rounds + 1):
        output_prefix = RESULT_ROOT / dataset.key / method / f"native_round{round_id}"
        round_log = RESULT_ROOT / dataset.key / method / f"round{round_id}.log"
        cached_rows = reusable_round(
            dataset,
            method,
            round_id,
            round_log,
            search_threads,
        )
        if cached_rows is not None:
            all_rows.extend(cached_rows)
            continue
        text = run_logged(
            [
                str(SEARCH),
                "--data_type", "float",
                "--dist_fn", "l2",
                "--index_path_prefix", str(index_prefix),
                "--result_path", str(output_prefix),
                "--query_file", str(dataset.query),
                "--gt_file", str(dataset.gt),
                "--recall_at", "10",
                "--search_list", *map(str, SEARCH_L),
                "--quantification_type", quantizer,
                "--beamwidth", str(BEAM),
                "--num_threads", str(search_threads),
                "--num_nodes_to_cache", "0",
            ],
            round_log,
            search_threads,
        )
        all_rows.extend(
            parse_round(dataset, method, round_id, text, search_threads)
        )
    write_csv(
        RESULT_ROOT / dataset.key / method / f"all_rounds_t{search_threads}.csv",
        all_rows,
    )
    write_csv(result, aggregate(all_rows))


def cleanup_owned(key: str, index_prefix: Path, owned: bool) -> None:
    if not owned:
        return
    target = index_prefix.parent.parent.resolve()
    root = EXTERNAL_ROOT.resolve()
    if target == root or root not in target.parents or target.name != key:
        raise RuntimeError(f"refusing unsafe cleanup: {target}")
    if target.exists():
        shutil.rmtree(target)
        print(f"CLEANED transient shared index {target}", flush=True)


def main() -> int:
    global EXTERNAL_ROOT, PQ_CODE_BUDGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--build-dram-gib", type=int, default=64)
    parser.add_argument(
        "--pq-code-budget",
        choices=("fixed64", "one-bit"),
        default=PQ_CODE_BUDGET,
        help="PQ bytes/vector: historical 64 bytes or dimension/8 bytes",
    )
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--keep-index", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.build_threads != 24 or args.search_threads < 1:
        parser.error(
            "rounds/search threads must be positive; matrix build threads must be 24"
        )
    PQ_CODE_BUDGET = args.pq_code_budget
    EXTERNAL_ROOT = args.external_root.resolve()
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    for executable in (BUILD, SEARCH):
        if not executable.is_file():
            raise FileNotFoundError(executable)

    for key in args.datasets:
        completed_methods = (
            "diskann_classic",
            "diskann_rabitq_1bit_adaptive_fp",
        )
        if all(
            complete(
                RESULT_ROOT / key / method / f"search_t{args.search_threads}.csv",
                args.rounds,
                args.search_threads,
            )
            for method in completed_methods
        ):
            print(
                f"SKIP complete dataset {key}: both DiskANN methods already exist",
                flush=True,
            )
            continue
        original = registry.DATASETS[key]
        registry.validate(original)
        heldout = matrix.heldout_dataset(key)
        index_prefix, owned = build(original, args.build_threads, args.build_dram_gib)
        search_method(
            heldout,
            index_prefix,
            "diskann_classic",
            args.rounds,
            args.search_threads,
        )
        search_method(
            heldout,
            index_prefix,
            "diskann_rabitq_1bit_adaptive_fp",
            args.rounds,
            args.search_threads,
        )
        if not args.keep_index:
            cleanup_owned(key, index_prefix, owned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
