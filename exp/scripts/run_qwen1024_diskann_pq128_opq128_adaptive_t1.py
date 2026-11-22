#!/usr/bin/env python3
"""Matched-memory DiskANN classic PQ/OPQ adaptive-FP T=1 benchmark.

The on-disk graph node contains its FP32 vector.  During beam search a node is
read only when it is expanded, so there is no separate fixed final rerank.
PQ128 and OPQ128 both use 128 routing-code bytes per database vector, matching
the 1024-bit payload of a 1024-D one-bit code.  OPQ additionally owns its
1024x1024 rotation matrix.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import statistics
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORK = REPO / "experiment/vendor/DiskANN-cpp-rerank80"
BUILD = FORK / "build-rerank80/apps/build_disk_index"
SEARCH = FORK / "build-rerank80/apps/search_disk_index"
GENERATE_PQ = FORK / "build-rerank80/apps/utils/generate_pq"
RESULT_ROOT = REPO / "experiment/qwen1024_diskann_pq128_opq128_adaptive_t1"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/qwen1024_diskann_pq128_opq128_adaptive"
)
SEARCH_L = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
ROUTING_BYTES = 128
M = 24
LBUILD = 200
BEAM = 2
MAX_QUERIES = 1000


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    train: Path
    query: Path
    gt: Path
    reusable_disk_prefix: Path | None = None


DATASETS = {
    "msmarco_qwen1024": Dataset(
        "msmarco_qwen1024",
        "MSMARCO Qwen-1024",
        Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_train.fbin"
        ),
        Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_query.fbin"
        ),
        Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_neighbors.ibin"
        ),
    ),
    "nq_qwen1024": Dataset(
        "nq_qwen1024",
        "NQ Qwen-1024",
        Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_train.fbin"
        ),
        Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_query.fbin"
        ),
        Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_neighbors_top100.ibin"
        ),
        Path(
            "/home/smrc/samsung-nvme/jihyo/nq_diskann_variants_io/"
            "nq_qwen06_1024/classic/index/"
            "nq_qwen06_1024.M24.L200.ramPQ64.classic_fpnode"
        ),
    ),
}


ROW_RE = re.compile(
    r"^\s*(?P<L>\d+)\s+(?P<beam>\d+)\s+"
    r"(?P<qps>[0-9.eE+-]+)\s+(?P<mean_latency>[0-9.eE+-]+)\s+"
    r"(?P<p999_latency>[0-9.eE+-]+)\s+(?P<mean_ios>[0-9.eE+-]+)\s+"
    r"(?P<mean_reorder_ios>[0-9.eE+-]+)\s+(?P<mean_reorder_4k>[0-9.eE+-]+)\s+"
    r"(?P<mean_io_us>[0-9.eE+-]+)\s+(?P<mean_cpu_us>[0-9.eE+-]+)\s+"
    r"(?P<recall_pct>[0-9.eE+-]+)\s*$"
)
STATS_RE = re.compile(
    r"^QUERY_STATS L=(?P<L>\d+) metric=(?P<metric>[a-z0-9_]+) "
    r"mean=(?P<mean>[-+0-9.eE]+) min=(?P<min>[-+0-9.eE]+) "
    r"max=(?P<max>[-+0-9.eE]+) p95=(?P<p95>[-+0-9.eE]+) "
    r"p99=(?P<p99>[-+0-9.eE]+)$"
)
REQUIRED_STATS = (
    "graph_io_requests",
    "graph_4k_pages",
    "total_4k_pages",
    "latency_us",
    "preprocess_us",
    "io_us",
)


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"short matrix header: {path}")
    return struct.unpack("ii", raw)


def prefix(dataset: Dataset, method: str) -> Path:
    return EXTERNAL_ROOT / dataset.key / method / "index"


def subset_path(dataset: Dataset, kind: str) -> Path:
    suffix = "query.fbin" if kind == "query" else "gt.ibin"
    return RESULT_ROOT / "subsets" / f"{dataset.key}.first{MAX_QUERIES}.{suffix}"


def write_prefix_rows(source: Path, target: Path, rows: int) -> None:
    n, dim = matrix_shape(source)
    if n < rows:
        raise RuntimeError(f"{source} has only {n} rows")
    element_bytes = 4
    expected = 8 + n * dim * element_bytes
    if source.stat().st_size < expected:
        raise RuntimeError(f"short matrix payload: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and matrix_shape(target) == (rows, dim):
        return
    with source.open("rb") as src, target.open("wb") as dst:
        src.seek(8)
        dst.write(struct.pack("ii", rows, dim))
        remaining = rows * dim * element_bytes
        while remaining:
            block = src.read(min(16 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError(f"short read while subsetting {source}")
            dst.write(block)
            remaining -= len(block)


def run_logged(command: list[str], log: Path, threads: int) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + shlex.join(command), flush=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with log.open("w", encoding="utf-8") as handle:
        handle.write("command=" + shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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


def required_index_files(index_prefix: Path) -> tuple[Path, ...]:
    return (
        Path(f"{index_prefix}_disk.index"),
        Path(f"{index_prefix}_pq_pivots.bin"),
        Path(f"{index_prefix}_pq_compressed.bin"),
    )


def ensure_pq128(dataset: Dataset, build_threads: int) -> Path:
    target = prefix(dataset, "pq128")
    required = required_index_files(target)
    if all(path.is_file() for path in required):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    disk_target = Path(f"{target}_disk.index")
    if dataset.reusable_disk_prefix is None:
        run_logged(
            [
                str(BUILD),
                "--data_type", "float",
                "--dist_fn", "cosine",
                "--data_path", str(dataset.train),
                "--index_path_prefix", str(target),
                "--max_degree", str(M),
                "--Lbuild", str(LBUILD),
                "--search_DRAM_budget", "1",
                "--build_DRAM_budget", "64",
                "--num_threads", str(build_threads),
                "--QD", str(ROUTING_BYTES),
                "--PQ_disk_bytes", "0",
                "--build_PQ_bytes", "0",
            ],
            RESULT_ROOT / dataset.key / "pq128_build.log",
            build_threads,
        )
    else:
        source_disk = Path(f"{dataset.reusable_disk_prefix}_disk.index")
        if not source_disk.is_file():
            raise FileNotFoundError(source_disk)
        if not disk_target.exists():
            os.link(source_disk, disk_target)
        run_logged(
            [
                str(GENERATE_PQ), "float", str(dataset.train), str(target),
                str(ROUTING_BYTES), "0.256", "0",
            ],
            RESULT_ROOT / dataset.key / "pq128_generate.log",
            1,
        )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing PQ128 files: {missing}")
    return target


def ensure_opq128(dataset: Dataset, pq_prefix: Path) -> Path:
    target = prefix(dataset, "opq128")
    required = (*required_index_files(target), Path(f"{target}_pq_pivots.bin_rotation_matrix.bin"))
    if all(path.is_file() for path in required):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    disk_target = Path(f"{target}_disk.index")
    if not disk_target.exists():
        os.link(Path(f"{pq_prefix}_disk.index"), disk_target)
    run_logged(
        [
            str(GENERATE_PQ), "float", str(dataset.train), str(target),
            str(ROUTING_BYTES), "0.256", "1",
        ],
        RESULT_ROOT / dataset.key / "opq128_generate.log",
        1,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing OPQ128 files: {missing}")
    return target


def parse_round(dataset: Dataset, method: str, round_id: int, text: str) -> list[dict[str, object]]:
    table: dict[int, dict[str, str]] = {}
    distributions: dict[int, dict[str, dict[str, float]]] = {}
    for line in text.splitlines():
        row_match = ROW_RE.match(line)
        if row_match:
            raw = row_match.groupdict()
            table[int(raw["L"])] = raw
        stat_match = STATS_RE.match(line.strip())
        if stat_match:
            raw = stat_match.groupdict()
            distributions.setdefault(int(raw["L"]), {})[raw["metric"]] = {
                key: float(raw[key]) for key in ("mean", "min", "max", "p95", "p99")
            }
    if tuple(table) != SEARCH_L:
        raise RuntimeError(f"parsed L={tuple(table)}, expected {SEARCH_L}")
    n, dim = matrix_shape(dataset.train)
    rows: list[dict[str, object]] = []
    for search_l in SEARCH_L:
        missing = set(REQUIRED_STATS) - set(distributions.get(search_l, {}))
        if missing:
            raise RuntimeError(f"missing stats L={search_l}: {sorted(missing)}")
        raw = table[search_l]
        stats = distributions[search_l]
        row: dict[str, object] = {
            "dataset": dataset.key,
            "title": dataset.title,
            "method_key": f"diskann_{method}",
            "method": f"DiskANN classic {method.upper()}128 + adaptive FP nodes",
            "routing_bytes_per_vector": ROUTING_BYTES,
            "Lsearch": search_l,
            "beam_width": int(raw["beam"]),
            "M": M,
            "Lbuild": LBUILD,
            "k": 10,
            "nq": MAX_QUERIES,
            "nb": n,
            "dim": dim,
            "round": round_id,
            "recall_at_10": float(raw["recall_pct"]) / 100.0,
            "qps": float(raw["qps"]),
            "mean_latency_us": float(raw["mean_latency"]),
            "mean_io_wait_us": float(raw["mean_io_us"]),
            "mean_query_preprocess_us": stats["preprocess_us"]["mean"],
            "adaptive_fp_node_reads_per_query": stats["graph_io_requests"]["mean"],
            "adaptive_fp_4k_pages_per_query": stats["graph_4k_pages"]["mean"],
            "storage": "SSD O_DIRECT FP32 graph nodes",
            "io_engine": "Linux native AIO beam batch",
            "search_threads": 1,
        }
        for metric in REQUIRED_STATS:
            for statistic, value in stats[metric].items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    identifiers = {
        "dataset", "title", "method_key", "method", "routing_bytes_per_vector",
        "Lsearch", "beam_width", "M", "Lbuild", "k", "nq", "nb", "dim",
        "round", "storage", "io_engine", "search_threads",
    }
    numeric = [key for key in rows[0] if key not in identifiers]
    output: list[dict[str, object]] = []
    for search_l in SEARCH_L:
        selected = [row for row in rows if int(row["Lsearch"]) == search_l]
        row = dict(selected[0])
        row.pop("round")
        row["rounds"] = len(selected)
        for key in numeric:
            row[key] = statistics.fmean(float(item[key]) for item in selected)
        row["qps_stddev"] = statistics.stdev(float(item["qps"]) for item in selected)
        output.append(row)
    return output


def result_path(dataset: Dataset, method: str) -> Path:
    return RESULT_ROOT / dataset.key / method / "search_t1.csv"


def complete(path: Path, rounds: int) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == len(SEARCH_L) and all(int(row["rounds"]) == rounds for row in rows)


def search(dataset: Dataset, method: str, index_prefix: Path, rounds: int) -> Path:
    result = result_path(dataset, method)
    if complete(result, rounds):
        print(f"REUSE {result}", flush=True)
        return result
    query = subset_path(dataset, "query")
    gt = subset_path(dataset, "gt")
    all_rows: list[dict[str, object]] = []
    for round_id in range(1, rounds + 1):
        output_prefix = RESULT_ROOT / dataset.key / method / f"native_round{round_id}"
        text = run_logged(
            [
                "taskset", "-c", "46", str(SEARCH),
                "--data_type", "float",
                "--dist_fn", "cosine",
                "--index_path_prefix", str(index_prefix),
                "--result_path", str(output_prefix),
                "--query_file", str(query),
                "--gt_file", str(gt),
                "--recall_at", "10",
                "--search_list", *map(str, SEARCH_L),
                "--beamwidth", str(BEAM),
                "--num_threads", "1",
                "--num_nodes_to_cache", "0",
            ],
            RESULT_ROOT / dataset.key / method / f"round{round_id}.log",
            1,
        )
        all_rows.extend(parse_round(dataset, method, round_id, text))
    write_csv(RESULT_ROOT / dataset.key / method / "all_rounds.csv", all_rows)
    write_csv(result, aggregate(all_rows))
    return result


def write_memory_summary(prefixes: list[tuple[Dataset, str, Path]]) -> Path:
    rows: list[dict[str, object]] = []
    for dataset, method, index_prefix in prefixes:
        nb, _ = matrix_shape(dataset.train)
        pivots = Path(f"{index_prefix}_pq_pivots.bin")
        codes = Path(f"{index_prefix}_pq_compressed.bin")
        rotation = Path(f"{index_prefix}_pq_pivots.bin_rotation_matrix.bin")
        rows.append(
            {
                "dataset": dataset.key,
                "method": method,
                "routing_code_payload_bytes": codes.stat().st_size - 8,
                "routing_code_bytes_per_vector": (codes.stat().st_size - 8) / nb,
                "pivots_bytes": pivots.stat().st_size,
                "rotation_bytes": rotation.stat().st_size if rotation.exists() else 0,
                "routing_total_bytes_excluding_graph": codes.stat().st_size
                + pivots.stat().st_size
                + (rotation.stat().st_size if rotation.exists() else 0),
                "disk_index_bytes": Path(f"{index_prefix}_disk.index").stat().st_size,
            }
        )
    path = RESULT_ROOT / "memory_summary.csv"
    write_csv(path, rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument(
        "--phase",
        choices=("prepare", "search", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("use at least two rounds")
    for executable in (BUILD, SEARCH, GENERATE_PQ):
        if not executable.is_file():
            raise FileNotFoundError(executable)

    prepared: list[tuple[Dataset, str, Path]] = []
    for key in args.datasets:
        dataset = DATASETS[key]
        for source in (dataset.train, dataset.query, dataset.gt):
            if not source.is_file():
                raise FileNotFoundError(source)
        write_prefix_rows(dataset.query, subset_path(dataset, "query"), MAX_QUERIES)
        write_prefix_rows(dataset.gt, subset_path(dataset, "gt"), MAX_QUERIES)
        pq = ensure_pq128(dataset, args.build_threads)
        opq = ensure_opq128(dataset, pq)
        prepared.extend(((dataset, "pq128", pq), (dataset, "opq128", opq)))

    write_memory_summary(prepared)
    if args.phase == "prepare":
        return
    outputs: list[Path] = []
    for dataset, method, index_prefix in prepared:
        outputs.append(search(dataset, method, index_prefix, args.rounds))
    rows: list[dict[str, object]] = []
    for path in outputs:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    write_csv(RESULT_ROOT / "diskann_pq128_opq128_adaptive_t1.csv", rows)


if __name__ == "__main__":
    main()
