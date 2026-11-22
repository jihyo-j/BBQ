#!/usr/bin/env python3
"""Local Elasticsearch 9.5.5 BBQ benchmark for NQ/Quora.

This starts the checked local tar distribution (no Docker) with a dedicated
data/log path on the 3.5 TiB NVMe.  Each empty index uses one shard,
M=24/ef_construction=200 and 24 parallel bulk clients.  Search sends one HTTP
request at a time and uses query-time oversample=8, i.e. exact reranking of 80
vectors for k=10.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

import run_elasticsearch_bbq_msmarco_qwen1024_1m as esbase
import diskveclab_full_matrix as matrix
import run_nq_quora_aligned4k_rerank50_80 as previous
import run_v051_ssd_odirect_multidataset as base


REPO = Path(__file__).resolve().parents[1]
ES_HOME = REPO / "tools/elasticsearch-dist/elasticsearch-9.5.5"
ES_BIN = ES_HOME / "bin/elasticsearch"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_elasticsearch_bbq_es955")
ES_DATA = EXTERNAL_ROOT / "data"
ES_LOGS = EXTERNAL_ROOT / "logs"
RESULT_ROOT = REPO / "experiment/nq_quora_elasticsearch_bbq_rerank80"

URL = "http://127.0.0.1:19255"
M = 24
EF_CONSTRUCTION = 200
K = 10
OVERSAMPLE = 8.0
RERANK = 80
NUM_CANDIDATES = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)
DISK_WATERMARK_LOW = "50gb"
DISK_WATERMARK_HIGH = "35gb"
DISK_WATERMARK_FLOOD = "20gb"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def selected_datasets(keys: list[str] | None) -> list[base.Dataset]:
    datasets = previous.target_datasets() if not keys else list(base.DATASETS.values())
    if not keys:
        return datasets
    by_key = {dataset.key: dataset for dataset in datasets}
    missing = sorted(set(keys) - set(by_key))
    if missing:
        raise ValueError(f"unknown datasets {missing}; choices={sorted(by_key)}")
    return [by_key[key] for key in keys]


def result_dir(dataset: base.Dataset) -> Path:
    return RESULT_ROOT / dataset.key


def index_name(dataset: base.Dataset) -> str:
    return f"{dataset.key.replace('_', '-')}-bbq-m24-efc200"


def configure_base(dataset: base.Dataset) -> None:
    _, dim = base.matrix_shape(dataset.train)
    esbase.BASE_PATH = dataset.train
    esbase.QUERY_PATH = dataset.query
    esbase.GT_PATH = dataset.gt
    esbase.RESULT_DIR = result_dir(dataset)
    esbase.DIM = dim
    esbase.M = M
    esbase.EF_CONSTRUCTION = EF_CONSTRUCTION
    esbase.OVERSAMPLE = OVERSAMPLE
    esbase.RERANK = RERANK
    esbase.NUM_CANDIDATES = NUM_CANDIDATES


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_ground_truth(path: Path, max_rows: int) -> np.ndarray:
    """Read int32 IDs from an ibin, allowing an appended float32 distance block."""
    n, width = base.matrix_shape(path)
    item_count = n * width
    payload_bytes = path.stat().st_size - 8
    ids_bytes = item_count * np.dtype(np.int32).itemsize
    if payload_bytes not in (ids_bytes, ids_bytes * 2):
        raise ValueError(
            f"bad ground-truth size: {path}: payload={payload_bytes}, "
            f"shape=({n}, {width})"
        )
    rows = min(n, max_rows) if max_rows else n
    return np.asarray(
        np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(n, width))[:rows]
    )


def start_server() -> tuple[subprocess.Popen[str], Any]:
    try:
        response = requests.get(URL, timeout=2)
        if response.ok:
            raise RuntimeError(
                f"refusing to reuse/alter an Elasticsearch already listening at {URL}"
            )
    except requests.ConnectionError:
        pass
    ES_DATA.mkdir(parents=True, exist_ok=True)
    ES_LOGS.mkdir(parents=True, exist_ok=True)
    server_log_path = ES_LOGS / "launcher-stdout.log"
    server_log = server_log_path.open("a", encoding="utf-8")
    server_log.write(f"\n[{now()}] starting local Elasticsearch\n")
    server_log.flush()
    env = dict(os.environ)
    env["ES_JAVA_OPTS"] = "-Xms16g -Xmx16g"
    cmd = [
        str(ES_BIN),
        "-Ecluster.name=jihyo-nq-quora-bbq",
        "-Enode.name=jihyo-nq-quora-bbq-1",
        f"-Epath.data={ES_DATA}",
        f"-Epath.logs={ES_LOGS}",
        "-Enetwork.host=127.0.0.1",
        "-Ehttp.port=19255",
        "-Etransport.port=19355",
        "-Ediscovery.type=single-node",
        "-Expack.security.enabled=false",
        "-Expack.security.enrollment.enabled=false",
        f"-Ecluster.routing.allocation.disk.watermark.low={DISK_WATERMARK_LOW}",
        f"-Ecluster.routing.allocation.disk.watermark.high={DISK_WATERMARK_HIGH}",
        f"-Ecluster.routing.allocation.disk.watermark.flood_stage={DISK_WATERMARK_FLOOD}",
    ]
    print(f"[{now()}] START local Elasticsearch 9.5.5 (no Docker)", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=ES_HOME,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 180
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            server_log.flush()
            raise RuntimeError(
                f"Elasticsearch exited early rc={process.returncode}; see {server_log_path}"
            )
        try:
            response = requests.get(URL, timeout=3)
            if response.ok:
                info = response.json()
                if info.get("cluster_name") != "jihyo-nq-quora-bbq":
                    raise RuntimeError(f"unexpected cluster: {info}")
                print(
                    f"[{now()}] READY Elasticsearch {info['version']['number']} "
                    f"cluster={info['cluster_name']}",
                    flush=True,
                )
                return process, server_log
        except (requests.ConnectionError, requests.Timeout, ValueError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Elasticsearch startup timed out: {last_error}; see {server_log_path}")


def stop_server(process: subprocess.Popen[str], server_log: Any) -> None:
    print(f"[{now()}] STOP local Elasticsearch pid={process.pid}", flush=True)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    server_log.flush()
    server_log.close()


def finish_index_timed(
    es: esbase.ES,
    index: str,
    target: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Finalize an index while preserving wall time for each observable stage."""
    print("FINALIZE refresh + force-merge(max_num_segments=1)", flush=True)
    stage_seconds: dict[str, float] = {}

    started = time.perf_counter()
    es.request("PUT", f"{index}/_settings", body={"index.refresh_interval": "1s"})
    stage_seconds["restore_refresh_interval_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    es.request("POST", f"{index}/_refresh", timeout=600)
    stage_seconds["explicit_refresh_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    count = int(es.request("GET", f"{index}/_count")["count"])
    stage_seconds["count_validation_seconds"] = time.perf_counter() - started
    if count != target:
        raise RuntimeError(f"indexed count {count:,} != target {target:,}")

    started = time.perf_counter()
    es.request(
        "POST", f"{index}/_forcemerge?max_num_segments=1&flush=true", timeout=7200
    )
    stage_seconds["force_merge_api_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    stats = es.request("GET", f"{index}/_stats/store,segments,docs")
    segments = es.request("GET", f"_cat/segments/{index}?format=json&bytes=b")
    mapping = es.request("GET", f"{index}/_mapping")
    settings = es.request("GET", f"{index}/_settings")
    stage_seconds["post_merge_metadata_seconds"] = time.perf_counter() - started

    manifest = {
        "index": index,
        "count": count,
        "stage_seconds": stage_seconds,
        "force_merge_seconds": stage_seconds["force_merge_api_seconds"],
        "index_stats": stats,
        "segments": segments,
        "mapping": mapping,
        "settings": settings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    store_bytes = stats["_all"]["primaries"]["store"]["size_in_bytes"]
    print(
        f"FINALIZED docs={count:,}, store={store_bytes/(1024**3):.3f} GiB, "
        f"force_merge={stage_seconds['force_merge_api_seconds']/60:.1f} min, "
        f"segments={len(segments)}",
        flush=True,
    )
    return manifest


def build_dataset(dataset: base.Dataset, es: esbase.ES, args: argparse.Namespace) -> None:
    configure_base(dataset)
    out = result_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    name = index_name(dataset)
    timing_path = out / "build_times.json"
    checkpoint = esbase.checkpoint_path(name)
    exists = es.session.head(f"{es.url}/{name}", timeout=30)
    if timing_path.exists() and exists.status_code == 200 and not args.force:
        print(f"SKIP complete Elasticsearch build: {dataset.key}", flush=True)
        return
    if args.force and exists.status_code == 200:
        print(f"RESET forced Elasticsearch index: {name}", flush=True)
        result = es.request("DELETE", name)
        if not result.get("acknowledged"):
            raise RuntimeError(f"index delete not acknowledged: {result}")
        checkpoint.unlink(missing_ok=True)
        timing_path.unlink(missing_ok=True)
        exists = es.session.head(f"{es.url}/{name}", timeout=30)
    if args.reset_incomplete and not timing_path.exists():
        if exists.status_code == 200:
            print(f"RESET incomplete Elasticsearch index: {name}", flush=True)
            result = es.request("DELETE", name)
            if not result.get("acknowledged"):
                raise RuntimeError(f"index delete not acknowledged: {result}")
        checkpoint.unlink(missing_ok=True)
        exists = es.session.head(f"{es.url}/{name}", timeout=30)
    if exists.status_code == 200:
        raise RuntimeError(
            f"index {name} exists without a complete build manifest; refusing checkpoint resume/mixing"
        )
    if exists.status_code != 404:
        raise RuntimeError(f"HEAD {name}: HTTP {exists.status_code}: {exists.text}")
    n, d = base.matrix_shape(dataset.train)
    pipeline_started_at = now()
    pipeline_started = time.perf_counter()
    started = time.perf_counter()
    esbase.create_index(es, name)
    create_seconds = time.perf_counter() - started
    started = time.perf_counter()
    health = es.request(
        "GET",
        f"_cluster/health/{name}?wait_for_status=yellow&wait_for_active_shards=1&timeout=120s",
        timeout=130,
    )
    primary_shard_activation_seconds = time.perf_counter() - started
    if health.get("timed_out") or int(health.get("active_primary_shards", 0)) != 1:
        raise RuntimeError(f"primary shard did not activate for {name}: {health}")
    started = time.perf_counter()
    esbase.bulk_index(es, name, n, args.bulk_docs, args.build_workers)
    bulk_seconds = time.perf_counter() - started
    started = time.perf_counter()
    finish_manifest = finish_index_timed(es, name, n, out)
    finish_seconds = time.perf_counter() - started
    stats = finish_manifest["index_stats"]
    started = time.perf_counter()
    version = es.request("GET", "")["version"]["number"]
    version_lookup_seconds = time.perf_counter() - started
    total_seconds = time.perf_counter() - pipeline_started
    finalize_stages = finish_manifest["stage_seconds"]
    record = {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "nb": n,
        "dim": d,
        "elasticsearch_version": version,
        "index": name,
        "index_type": "bbq_hnsw",
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "shards": 1,
        "replicas": 0,
        "build_workers": args.build_workers,
        "bulk_docs": args.bulk_docs,
        "started_at": pipeline_started_at,
        "finished_at": now(),
        "create_seconds": create_seconds,
        "primary_shard_activation_seconds": primary_shard_activation_seconds,
        "bulk_preprocess_graph_seconds": bulk_seconds,
        "bulk_timing_scope": "client mmap/read + normalize + NDJSON serialization + HTTP + server BBQ HNSW indexing",
        "finalize_force_merge_seconds": finish_seconds,
        "restore_refresh_interval_seconds": float(finalize_stages["restore_refresh_interval_seconds"]),
        "explicit_refresh_seconds": float(finalize_stages["explicit_refresh_seconds"]),
        "count_validation_seconds": float(finalize_stages["count_validation_seconds"]),
        "force_merge_api_seconds": float(finalize_stages["force_merge_api_seconds"]),
        "post_merge_metadata_seconds": float(finalize_stages["post_merge_metadata_seconds"]),
        "version_lookup_seconds": version_lookup_seconds,
        "disk_watermark_low": DISK_WATERMARK_LOW,
        "disk_watermark_high": DISK_WATERMARK_HIGH,
        "disk_watermark_flood_stage": DISK_WATERMARK_FLOOD,
        "end_to_end_build_seconds": total_seconds,
        "timing_scope": "empty index create + client normalize/JSON + bulk BBQ graph build + refresh + force merge",
        "store_bytes_after_finish": int(stats["_all"]["primaries"]["store"]["size_in_bytes"]),
        "final_segment_bytes": sum(
            int(segment["size"]) for segment in finish_manifest["segments"]
        ),
    }
    timing_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run_queries(
    index: str,
    queries: np.ndarray,
    gt: np.ndarray,
    candidates: int,
    round_id: int,
    concurrency: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if concurrency == 1:
        return esbase.run_queries(
            esbase.ES(URL), index, queries, gt, candidates, round_id
        )

    local = threading.local()
    clients: list[esbase.ES] = []
    clients_lock = threading.Lock()

    def client() -> esbase.ES:
        value = getattr(local, "client", None)
        if value is None:
            value = esbase.ES(URL)
            local.client = value
            with clients_lock:
                clients.append(value)
        return value

    def query_one(item: tuple[int, np.ndarray]) -> dict[str, Any]:
        query_id, vector = item
        started = time.perf_counter()
        result = client().request(
            "POST",
            f"{index}/_search?filter_path=took,hits.hits._id",
            body=esbase.query_body(vector, candidates),
            timeout=120,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        ids = [int(hit["_id"]) for hit in result["hits"]["hits"]]
        hits = len(set(ids).intersection(int(value) for value in gt[query_id, :K]))
        return {
            "round": round_id,
            "num_candidates": candidates,
            "query_id": query_id,
            "hits_at_10": hits,
            "latency_ms": latency_ms,
            "server_took_ms": int(result.get("took", 0)),
        }

    started = time.perf_counter()
    details: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for completed, detail in enumerate(
                executor.map(query_one, enumerate(queries)), start=1
            ):
                details.append(detail)
                if completed % 1000 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"QUERY nc={candidates} round={round_id} concurrency={concurrency}: "
                        f"{completed:,}/{len(queries):,}, QPS={completed/elapsed:.1f}",
                        flush=True,
                    )
    finally:
        for value in clients:
            value.session.close()
    elapsed = time.perf_counter() - started
    latencies = [float(item["latency_ms"]) for item in details]
    server_took = [int(item["server_took_ms"]) for item in details]
    row = {
        "round": round_id,
        "num_candidates": candidates,
        "client_concurrency": concurrency,
        "recall_at_10": sum(int(item["hits_at_10"]) for item in details)
        / (len(queries) * K),
        "qps": len(queries) / elapsed,
        "elapsed_seconds": elapsed,
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_max_ms": max(latencies),
        "server_took_mean_ms": statistics.fmean(server_took),
    }
    print(
        f"RESULT nc={candidates} round={round_id} concurrency={concurrency}: "
        f"R@10={row['recall_at_10']:.6f}, QPS={row['qps']:.2f}, "
        f"p95={row['latency_p95_ms']:.3f} ms, p99={row['latency_p99_ms']:.3f} ms",
        flush=True,
    )
    return row, details


def search_dataset(dataset: base.Dataset, es: esbase.ES, args: argparse.Namespace) -> None:
    configure_base(dataset)
    name = index_name(dataset)
    exists = es.session.head(f"{es.url}/{name}", timeout=30)
    if exists.status_code != 200:
        raise RuntimeError(f"missing Elasticsearch index {name}: HTTP {exists.status_code}")
    queries = esbase.normalize(esbase.matrix_memmap(dataset.query, np.float32))
    gt = read_ground_truth(dataset.gt, len(queries))
    n, d = base.matrix_shape(dataset.train)
    round_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    print(
        f"SEARCH {dataset.key}: no explicit warmup, client concurrency="
        f"{args.search_concurrency}, all {len(queries):,} queries",
        flush=True,
    )
    for candidates in NUM_CANDIDATES:
        for round_id in range(1, args.rounds + 1):
            row, details = run_queries(
                name, queries, gt, candidates, round_id, args.search_concurrency
            )
            round_rows.append(row)
            for detail in details:
                detail["dataset"] = dataset.key
                per_query.append(detail)
            write_csv(result_dir(dataset) / "round_results.csv", round_rows)
            write_csv(result_dir(dataset) / "per_query.csv", per_query)
            write_csv(
                result_dir(dataset)
                / f"round_results_t{args.search_concurrency}.csv",
                round_rows,
            )
            write_csv(
                result_dir(dataset) / f"per_query_t{args.search_concurrency}.csv",
                per_query,
            )
    aggregate: list[dict[str, object]] = []
    version = es.request("GET", "")["version"]["number"]
    for candidates in NUM_CANDIDATES:
        selected = [row for row in round_rows if row["num_candidates"] == candidates]
        aggregate.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method": "Elasticsearch BBQ HNSW + FP rerank@80",
                "family": "elasticsearch_bbq_rerank80",
                "elasticsearch_version": version,
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "k": K,
                "num_candidates": candidates,
                "oversample": OVERSAMPLE,
                "fp_rerank": RERANK,
                "nq": len(queries),
                "nb": n,
                "dim": d,
                "client_concurrency": args.search_concurrency,
                "rounds": args.rounds,
                "explicit_warmup": False,
                "recall_at_10": statistics.fmean(float(row["recall_at_10"]) for row in selected),
                "qps": statistics.fmean(float(row["qps"]) for row in selected),
                "qps_stddev": statistics.stdev(float(row["qps"]) for row in selected) if len(selected) > 1 else 0.0,
                "latency_mean_ms": statistics.fmean(float(row["latency_mean_ms"]) for row in selected),
                "latency_p50_ms_mean_of_rounds": statistics.fmean(float(row["latency_p50_ms"]) for row in selected),
                "latency_p95_ms_mean_of_rounds": statistics.fmean(float(row["latency_p95_ms"]) for row in selected),
                "latency_p99_ms_mean_of_rounds": statistics.fmean(float(row["latency_p99_ms"]) for row in selected),
                "fp_vectors_per_query_mean": RERANK,
                "graph_4k_pages_per_query_mean": "",
                "fp_4k_pages_per_query_mean": "",
                "total_4k_pages_per_query_mean": "",
                "page_reads_observable": False,
                "io_path": "Lucene mmap / OS page cache; not O_DIRECT",
                "timing_scope": "end-to-end localhost HTTP including JSON/REST",
            }
        )
    write_csv(result_dir(dataset) / "all_results.csv", aggregate)
    write_csv(
        result_dir(dataset) / f"all_results_t{args.search_concurrency}.csv",
        aggregate,
    )


def search_complete(dataset: base.Dataset, concurrency: int, rounds: int) -> bool:
    path = result_dir(dataset) / f"all_results_t{concurrency}.csv"
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(NUM_CANDIDATES):
        return False
    return all(
        int(row.get("rounds", 0)) == rounds
        and int(row.get("num_candidates", -1)) == expected
        for row, expected in zip(rows, NUM_CANDIDATES)
    )


def delete_dataset_index(dataset: base.Dataset, es: esbase.ES) -> None:
    """Delete only this benchmark's transient Elasticsearch index."""
    name = index_name(dataset)
    exists = es.session.head(f"{es.url}/{name}", timeout=30)
    if exists.status_code == 404:
        configure_base(dataset)
        esbase.checkpoint_path(name).unlink(missing_ok=True)
        print(f"SKIP missing Elasticsearch index during cleanup: {name}", flush=True)
        return
    if exists.status_code != 200:
        raise RuntimeError(f"HEAD {name}: HTTP {exists.status_code}: {exists.text}")
    result = es.request("DELETE", name)
    if not result.get("acknowledged"):
        raise RuntimeError(f"index delete not acknowledged: {result}")
    configure_base(dataset)
    esbase.checkpoint_path(name).unlink(missing_ok=True)
    print(f"DELETED transient Elasticsearch index: {name}", flush=True)


def report(datasets: list[base.Dataset]) -> None:
    builds: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for dataset in datasets:
        timing = result_dir(dataset) / "build_times.json"
        tagged_results = sorted(result_dir(dataset).glob("all_results_t*.csv"))
        csv_paths = tagged_results or [result_dir(dataset) / "all_results.csv"]
        if timing.exists():
            builds.append(json.loads(timing.read_text(encoding="utf-8")))
        for csv_path in csv_paths:
            if csv_path.exists():
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    results.extend(dict(row) for row in csv.DictReader(handle))
    if builds:
        write_csv(RESULT_ROOT / "build_times.csv", builds)
    if results:
        write_csv(RESULT_ROOT / "all_results.csv", results)


def main() -> None:
    global ES_DATA, ES_LOGS, EXTERNAL_ROOT, RESULT_ROOT, NUM_CANDIDATES
    global DISK_WATERMARK_LOW, DISK_WATERMARK_HIGH, DISK_WATERMARK_FLOOD
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("build", "search", "report", "cleanup", "all"), default="all"
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--build-workers", type=int, default=24)
    parser.add_argument("--bulk-docs", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--search-concurrency", type=int, default=1)
    parser.add_argument(
        "--num-candidates",
        nargs="+",
        type=int,
        default=list(NUM_CANDIDATES),
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--disk-watermark-low", default=DISK_WATERMARK_LOW)
    parser.add_argument("--disk-watermark-high", default=DISK_WATERMARK_HIGH)
    parser.add_argument("--disk-watermark-flood", default=DISK_WATERMARK_FLOOD)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reset-incomplete", action="store_true")
    parser.add_argument("--heldout-matrix", action="store_true")
    parser.add_argument("--cleanup-after-search", action="store_true")
    args = parser.parse_args()
    if args.search_concurrency <= 0:
        raise ValueError("--search-concurrency must be positive")
    NUM_CANDIDATES = tuple(dict.fromkeys(args.num_candidates))
    RESULT_ROOT = args.result_root.resolve()
    EXTERNAL_ROOT = args.external_root.resolve()
    DISK_WATERMARK_LOW = args.disk_watermark_low
    DISK_WATERMARK_HIGH = args.disk_watermark_high
    DISK_WATERMARK_FLOOD = args.disk_watermark_flood
    ES_DATA = EXTERNAL_ROOT / "data"
    ES_LOGS = EXTERNAL_ROOT / "logs"
    datasets = selected_datasets(args.datasets)
    if args.heldout_matrix:
        datasets = [matrix.heldout_dataset(dataset.key) for dataset in datasets]
    if args.phase == "report":
        report(datasets)
        return
    process, server_log = start_server()
    try:
        es = esbase.ES(URL)
        for dataset in datasets:
            print(f"[{now()}] DATASET {dataset.key} {dataset.model}", flush=True)
            complete = search_complete(dataset, args.search_concurrency, args.rounds)
            if complete and not args.force and args.phase in {"search", "all"}:
                print(f"SKIP complete Elasticsearch search: {dataset.key}", flush=True)
                if args.cleanup_after_search:
                    delete_dataset_index(dataset, es)
                continue
            if args.phase in {"build", "all"}:
                build_dataset(dataset, es, args)
            if args.phase in {"search", "all"}:
                search_dataset(dataset, es, args)
                if args.cleanup_after_search:
                    delete_dataset_index(dataset, es)
            if args.phase == "cleanup":
                delete_dataset_index(dataset, es)
        report(datasets)
    finally:
        stop_server(process, server_log)


if __name__ == "__main__":
    main()
