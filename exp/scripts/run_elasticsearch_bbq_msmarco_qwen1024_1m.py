#!/usr/bin/env python3
"""Elasticsearch BBQ benchmark for MS MARCO Qwen-1024 1M.

The benchmark deliberately uses one Elasticsearch shard and one sequential
client request at a time.  With k=10 and ``rescore_vector.oversample=8.0``,
Elasticsearch reranks 80 full-precision vectors per query on that shard.

The input fbin vectors are already unit length, but each indexing/query block
is normalized again so Elasticsearch's dot-product requirements are explicit.
Bulk-index progress is checkpointed after every successful request, allowing a
safe resume after interruption without recreating the index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import requests


REPO = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = REPO / "tools/python-packages"
if str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))
try:
    import orjson
except ImportError as exc:  # pragma: no cover - actionable CLI error
    raise SystemExit(
        "orjson is required; install it into tools/python-packages"
    ) from exc


DATA_ROOT = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024")
BASE_PATH = DATA_ROOT / "msmarco-1m_base.fbin"
QUERY_PATH = DATA_ROOT / "msmarco-1m_query.fbin"
GT_PATH = DATA_ROOT / "msmarco-1m_gt.ibin"
RESULT_DIR = REPO / "experiment/elasticsearch_bbq_msmarco_qwen1024_1m"

DEFAULT_URL = "http://127.0.0.1:19255"
DEFAULT_INDEX = "msmarco-qwen1024-1m-bbq-m24-efc200"
DIM = 1024
K = 10
M = 24
EF_CONSTRUCTION = 200
OVERSAMPLE = 8.0
RERANK = int(K * OVERSAMPLE)
NUM_CANDIDATES = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)
ROUNDS = 3
BULK_DOCS = 256
BUILD_WORKERS = 24


def fbin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        shape = np.fromfile(handle, dtype=np.int32, count=2)
    if shape.size != 2:
        raise ValueError(f"short fbin header: {path}")
    return int(shape[0]), int(shape[1])


def matrix_memmap(path: Path, dtype: np.dtype[Any]) -> np.memmap:
    n, d = fbin_shape(path)
    itemsize = np.dtype(dtype).itemsize
    expected = 8 + n * d * itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"bad matrix size: {path}: {actual} != {expected}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def normalize(rows: np.ndarray) -> np.ndarray:
    result = np.asarray(rows, dtype=np.float32).copy(order="C")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero vector found")
    result /= norms
    return result


class ES:
    def __init__(self, url: str, timeout: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any | None = None,
        data: bytes | None = None,
        content_type: str = "application/json",
        timeout: float | None = None,
    ) -> Any:
        headers = {"Content-Type": content_type}
        encoded = None if body is None else orjson.dumps(body)
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.session.request(
                    method,
                    f"{self.url}/{path.lstrip('/')}",
                    data=data if data is not None else encoded,
                    headers=headers,
                    timeout=self.timeout if timeout is None else timeout,
                )
                if response.status_code >= 500 and attempt < 4:
                    time.sleep(2**attempt)
                    continue
                if not response.ok:
                    raise RuntimeError(
                        f"{method} {path}: HTTP {response.status_code}: "
                        f"{response.text[:2000]}"
                    )
                if not response.content:
                    return None
                return orjson.loads(response.content)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"{method} {path} failed after retries: {last_error}")


def validate_inputs(limit_docs: int | None) -> tuple[int, int]:
    for path in (BASE_PATH, QUERY_PATH, GT_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    nb, dim = fbin_shape(BASE_PATH)
    nq, qdim = fbin_shape(QUERY_PATH)
    ngt, gt_width = fbin_shape(GT_PATH)
    if (nb, dim) != (1_000_000, DIM):
        raise ValueError(f"unexpected base shape: {nb}x{dim}")
    if qdim != DIM or nq != ngt or gt_width < K:
        raise ValueError(
            f"query/GT mismatch: query={nq}x{qdim}, gt={ngt}x{gt_width}"
        )
    target = nb if limit_docs is None else min(nb, limit_docs)
    print(
        f"INPUT base={target:,}/{nb:,}x{dim}, query={nq:,}x{qdim}, "
        f"gt={ngt:,}x{gt_width}",
        flush=True,
    )
    return target, nq


def create_index(es: ES, index: str) -> None:
    exists = es.session.head(f"{es.url}/{index}", timeout=30)
    if exists.status_code == 200:
        print(f"INDEX exists: {index}", flush=True)
        return
    if exists.status_code != 404:
        raise RuntimeError(f"HEAD {index}: HTTP {exists.status_code}: {exists.text}")
    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": DIM,
                    "index": True,
                    "similarity": "dot_product",
                    "index_options": {
                        "type": "bbq_hnsw",
                        "m": M,
                        "ef_construction": EF_CONSTRUCTION,
                    },
                }
            },
        },
    }
    result = es.request("PUT", index, body=body)
    if not result.get("acknowledged"):
        raise RuntimeError(f"index create not acknowledged: {result}")
    print(f"CREATED {index}: BBQ HNSW M={M}, efConstruction={EF_CONSTRUCTION}")


def checkpoint_path(index: str) -> Path:
    return RESULT_DIR / f"{index}.bulk_checkpoint.json"


def read_checkpoint(index: str) -> int:
    path = checkpoint_path(index)
    if not path.exists():
        return 0
    value = json.loads(path.read_text())
    return int(value["next_id"])


def write_checkpoint(index: str, next_id: int, target: int) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(index)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"index": index, "next_id": next_id, "target": target}, indent=2
        )
        + "\n"
    )
    os.replace(temporary, path)


def bulk_payload(start: int, rows: np.ndarray) -> bytes:
    lines: list[bytes] = []
    option = orjson.OPT_SERIALIZE_NUMPY
    for offset, vector in enumerate(rows):
        doc_id = start + offset
        lines.append(b'{"index":{"_id":"' + str(doc_id).encode() + b'"}}\n')
        lines.append(
            b'{"vector":' + orjson.dumps(vector, option=option) + b"}\n"
        )
    return b"".join(lines)


_worker_state = threading.local()


def bulk_batch(
    url: str,
    index: str,
    vectors: np.ndarray,
    start: int,
    stop: int,
) -> tuple[int, int]:
    worker_es = getattr(_worker_state, "es", None)
    if worker_es is None:
        worker_es = ES(url)
        _worker_state.es = worker_es
    rows = normalize(vectors[start:stop])
    payload = bulk_payload(start, rows)
    result = worker_es.request(
        "POST",
        f"{index}/_bulk?refresh=false&filter_path=errors,items.*.error",
        data=payload,
        content_type="application/x-ndjson",
        timeout=600,
    )
    if result.get("errors"):
        errors = [
            item
            for item in result.get("items", [])
            if next(iter(item.values())).get("error")
        ]
        raise RuntimeError(f"bulk errors for [{start}, {stop}): {errors[:3]}")
    return start, stop


def bulk_index(
    es: ES,
    index: str,
    target: int,
    batch_docs: int,
    build_workers: int,
) -> None:
    vectors = matrix_memmap(BASE_PATH, np.float32)
    checkpoint = read_checkpoint(index)
    if checkpoint > target:
        raise ValueError(f"checkpoint {checkpoint} exceeds target {target}")
    if checkpoint:
        print(f"RESUME bulk indexing at id={checkpoint:,}", flush=True)
    print(
        f"BUILD parallel bulk workers={build_workers}, batch_docs={batch_docs}",
        flush=True,
    )
    started = time.monotonic()
    interval_started = started
    interval_completed = 0
    completed_docs = 0
    completed_ranges: dict[int, int] = {}
    next_submit = checkpoint
    pending: dict[Future[tuple[int, int]], tuple[int, int]] = {}
    with ThreadPoolExecutor(max_workers=build_workers) as executor:
        while next_submit < target and len(pending) < build_workers:
            stop = min(target, next_submit + batch_docs)
            future = executor.submit(
                bulk_batch, es.url, index, vectors, next_submit, stop
            )
            pending[future] = (next_submit, stop)
            next_submit = stop
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                expected = pending.pop(future)
                start, stop = future.result()
                if (start, stop) != expected:
                    raise RuntimeError(
                        f"bulk completion mismatch: {(start, stop)} != {expected}"
                    )
                completed_ranges[start] = stop
                completed_docs += stop - start
                while checkpoint in completed_ranges:
                    checkpoint = completed_ranges.pop(checkpoint)
                    write_checkpoint(index, checkpoint, target)
                if next_submit < target:
                    submit_stop = min(target, next_submit + batch_docs)
                    new_future = executor.submit(
                        bulk_batch,
                        es.url,
                        index,
                        vectors,
                        next_submit,
                        submit_stop,
                    )
                    pending[new_future] = (next_submit, submit_stop)
                    next_submit = submit_stop
            now = time.monotonic()
            if now - interval_started >= 15 or not pending:
                recent_done = completed_docs - interval_completed
                recent_rate = recent_done / max(1e-9, now - interval_started)
                total_rate = completed_docs / max(1e-9, now - started)
                eta = (target - checkpoint) / max(1e-9, recent_rate)
                print(
                    f"BULK checkpoint={checkpoint:,}/{target:,} "
                    f"({100*checkpoint/target:.2f}%), "
                    f"recent={recent_rate:.1f} vec/s "
                    f"parallel_total={total_rate:.1f} vec/s "
                    f"inflight={len(pending)} ETA={eta/60:.1f} min",
                    flush=True,
                )
                interval_started = now
                interval_completed = completed_docs
    if checkpoint != target:
        raise RuntimeError(f"non-contiguous bulk completion: {checkpoint} != {target}")


def finish_index(es: ES, index: str, target: int) -> dict[str, Any]:
    print("FINALIZE refresh + force-merge(max_num_segments=1)", flush=True)
    es.request("PUT", f"{index}/_settings", body={"index.refresh_interval": "1s"})
    es.request("POST", f"{index}/_refresh", timeout=600)
    count = int(es.request("GET", f"{index}/_count")["count"])
    if count != target:
        raise RuntimeError(f"indexed count {count:,} != target {target:,}")
    merge_started = time.monotonic()
    es.request(
        "POST", f"{index}/_forcemerge?max_num_segments=1&flush=true", timeout=7200
    )
    merge_seconds = time.monotonic() - merge_started
    stats = es.request("GET", f"{index}/_stats/store,segments,docs")
    segments = es.request("GET", f"_cat/segments/{index}?format=json&bytes=b")
    manifest = {
        "index": index,
        "count": count,
        "force_merge_seconds": merge_seconds,
        "index_stats": stats,
        "segments": segments,
        "mapping": es.request("GET", f"{index}/_mapping"),
        "settings": es.request("GET", f"{index}/_settings"),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    store_bytes = stats["_all"]["primaries"]["store"]["size_in_bytes"]
    print(
        f"FINALIZED docs={count:,}, store={store_bytes/(1024**3):.3f} GiB, "
        f"force_merge={merge_seconds/60:.1f} min, segments={len(segments)}",
        flush=True,
    )
    return manifest


def query_body(vector: np.ndarray, num_candidates: int) -> dict[str, Any]:
    return {
        "size": K,
        "_source": False,
        "stored_fields": [],
        "track_total_hits": False,
        "knn": {
            "field": "vector",
            "query_vector": vector.tolist(),
            "k": K,
            "num_candidates": num_candidates,
            "rescore_vector": {"oversample": OVERSAMPLE},
        },
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_queries(
    es: ES,
    index: str,
    queries: np.ndarray,
    gt: np.ndarray,
    num_candidates: int,
    round_no: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    latencies_ms: list[float] = []
    server_took_ms: list[int] = []
    correct = 0
    details: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query_id, vector in enumerate(queries):
        query_started = time.perf_counter()
        result = es.request(
            "POST",
            f"{index}/_search?filter_path=took,hits.hits._id",
            body=query_body(vector, num_candidates),
            timeout=120,
        )
        latency_ms = (time.perf_counter() - query_started) * 1000.0
        ids = [int(hit["_id"]) for hit in result["hits"]["hits"]]
        hits = len(set(ids).intersection(int(value) for value in gt[query_id, :K]))
        correct += hits
        took = int(result.get("took", 0))
        latencies_ms.append(latency_ms)
        server_took_ms.append(took)
        details.append(
            {
                "round": round_no,
                "num_candidates": num_candidates,
                "query_id": query_id,
                "hits_at_10": hits,
                "latency_ms": latency_ms,
                "server_took_ms": took,
            }
        )
        if (query_id + 1) % 1000 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"QUERY nc={num_candidates} round={round_no}: "
                f"{query_id+1:,}/{len(queries):,}, "
                f"QPS={(query_id+1)/elapsed:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    row = {
        "round": round_no,
        "num_candidates": num_candidates,
        "recall_at_10": correct / (len(queries) * K),
        "qps": len(queries) / elapsed,
        "elapsed_seconds": elapsed,
        "latency_mean_ms": statistics.fmean(latencies_ms),
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_p99_ms": percentile(latencies_ms, 99),
        "latency_max_ms": max(latencies_ms),
        "server_took_mean_ms": statistics.fmean(server_took_ms),
    }
    print(
        f"RESULT nc={num_candidates} round={round_no}: "
        f"R@10={row['recall_at_10']:.6f}, QPS={row['qps']:.2f}, "
        f"p95={row['latency_p95_ms']:.3f} ms, "
        f"p99={row['latency_p99_ms']:.3f} ms",
        flush=True,
    )
    return row, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(es: ES, index: str, rounds: int) -> None:
    queries = normalize(matrix_memmap(QUERY_PATH, np.float32))
    gt = np.asarray(matrix_memmap(GT_PATH, np.int32))
    round_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    print(
        "SEARCH no explicit warmup; round 1 is the first measured pass. "
        "All requests are sequential (client concurrency=1).",
        flush=True,
    )
    for num_candidates in NUM_CANDIDATES:
        for round_no in range(1, rounds + 1):
            row, details = run_queries(
                es, index, queries, gt, num_candidates, round_no
            )
            round_rows.append(row)
            detail_rows.extend(details)
            write_csv(RESULT_DIR / "round_results.csv", round_rows)
            write_csv(RESULT_DIR / "per_query.csv", detail_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for num_candidates in NUM_CANDIDATES:
        selected = [
            row for row in round_rows if row["num_candidates"] == num_candidates
        ]
        aggregate_rows.append(
            {
                "dataset": "msmarco_qwen1024_1m",
                "method": "Elasticsearch BBQ HNSW + FP rerank@80",
                "elasticsearch_version": es.request("GET", "")["version"][
                    "number"
                ],
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "k": K,
                "num_candidates": num_candidates,
                "oversample": OVERSAMPLE,
                "fp_rerank": RERANK,
                "nq": len(queries),
                "nb": 1_000_000,
                "dim": DIM,
                "client_concurrency": 1,
                "rounds": rounds,
                "explicit_warmup": False,
                "recall_at_10": statistics.fmean(
                    float(row["recall_at_10"]) for row in selected
                ),
                "qps": statistics.fmean(float(row["qps"]) for row in selected),
                "latency_mean_ms": statistics.fmean(
                    float(row["latency_mean_ms"]) for row in selected
                ),
                "latency_p50_ms_mean_of_rounds": statistics.fmean(
                    float(row["latency_p50_ms"]) for row in selected
                ),
                "latency_p95_ms_mean_of_rounds": statistics.fmean(
                    float(row["latency_p95_ms"]) for row in selected
                ),
                "latency_p99_ms_mean_of_rounds": statistics.fmean(
                    float(row["latency_p99_ms"]) for row in selected
                ),
            }
        )
    write_csv(RESULT_DIR / "all_results.csv", aggregate_rows)
    stats = {
        "nodes": es.request("GET", "_nodes/stats/jvm,process,indices,fs"),
        "index": es.request("GET", f"{index}/_stats/store,segments,docs"),
    }
    (RESULT_DIR / "post_search_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("create", "index", "finish", "search", "all")
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--bulk-docs", type=int, default=BULK_DOCS)
    parser.add_argument("--build-workers", type=int, default=BUILD_WORKERS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target, _ = validate_inputs(args.limit_docs)
    es = ES(args.url)
    info = es.request("GET", "")
    print(
        f"ELASTICSEARCH {info['version']['number']} at {args.url}; "
        f"cluster={info['cluster_name']}",
        flush=True,
    )
    if args.phase in {"create", "all"}:
        create_index(es, args.index)
    if args.phase in {"index", "all"}:
        create_index(es, args.index)
        bulk_index(
            es,
            args.index,
            target,
            args.bulk_docs,
            args.build_workers,
        )
    if args.phase in {"finish", "all"}:
        finish_index(es, args.index, target)
    if args.phase in {"search", "all"}:
        if target != 1_000_000:
            raise ValueError("recall benchmark requires the full 1M corpus")
        benchmark(es, args.index, args.rounds)


if __name__ == "__main__":
    main()
