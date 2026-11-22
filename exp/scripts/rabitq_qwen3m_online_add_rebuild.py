#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

if sys.version_info[:2] != (3, 12) and os.environ.get("RABITQ_ALLOW_PY311") != "1":
    python312 = "/home/jihyo/anaconda3/bin/python"
    system_libstdcxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    bootstrap_env = os.environ.copy()
    previous_preload = bootstrap_env.get("LD_PRELOAD", "").strip()
    bootstrap_env["LD_PRELOAD"] = (
        f"{system_libstdcxx}:{previous_preload}"
        if previous_preload
        else system_libstdcxx
    )
    os.execve(python312, [python312, __file__, *sys.argv[1:]], bootstrap_env)

import numpy as np

from rabitq_centroid_insert_sweep import (
    FbinView,
    assign_and_residual_distance,
    compute_recall,
    exact_topk_ip,
    exact_topk_l2,
    fbin_mmap,
    import_faiss,
    import_hnsw_index,
    l2_normalize_rows,
    load_rows,
    parse_int_list,
    read_fbin,
    summarize,
    train_centroids,
    write_csv,
)


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024")
DEFAULT_BASE = DATA_ROOT / "msmarco-1m_base.fbin"
DEFAULT_LARGER = DATA_ROOT / "msmarco-3m_base.fbin"
DEFAULT_QUERY = DATA_ROOT / "msmarco-3m_query.fbin"
DEFAULT_BASE_GT = DATA_ROOT / "msmarco-1m_gt.ibin"
DEFAULT_OUT = (
    REPO
    / "experiment/online_insert_sweeps/"
    / "msmarco_qwen1024_1m_plus_3mnew_rabitq_b7_far_online_vs_rebuild_q1000.csv"
)
DEFAULT_INDEX_DIR = (
    Path("/home/smrc/samsung-nvme/jihyo")
    / "msmarco_qwen1024_online_insert_rebuild"
    / "indexes"
    / "rabitq"
)
DEFAULT_INSERT_SIZES = [200_000, 500_000, 1_000_000]
DEFAULT_EFS = [64, 128, 256, 512, 1024]
DEFAULT_FINGERPRINT_DIM_COUNT = 1024


def read_ibin(path: Path, max_rows: int = 0) -> np.ndarray:
    with path.open("rb") as fp:
        header = np.fromfile(fp, dtype=np.uint32, count=2)
        if header.size != 2:
            raise ValueError(f"Bad ibin header: {path}")
        n, d = int(header[0]), int(header[1])
        rows = min(n, max_rows) if max_rows and max_rows > 0 else n
        data = np.fromfile(fp, dtype=np.int32, count=rows * d)
    if data.size != rows * d:
        raise ValueError(f"Bad ibin payload: {path}")
    return data.reshape(rows, d).astype(np.int64, copy=False)


def void_keys(x: np.ndarray, dims: list[int]) -> np.ndarray:
    if len(dims) == x.shape[1] and all(i == d for i, d in enumerate(dims)):
        subset = np.ascontiguousarray(x).view(np.uint32)
    else:
        subset = np.ascontiguousarray(x[:, dims]).view(np.uint32)
    return subset.reshape(subset.shape[0], -1).view(
        np.dtype((np.void, subset.shape[1] * subset.dtype.itemsize))
    ).reshape(-1)


def find_larger_rows_not_in_base(
    base_view: FbinView,
    larger_view: FbinView,
    dims: list[int],
    batch_size: int,
) -> tuple[np.ndarray, int, float]:
    if base_view.dim != larger_view.dim:
        raise ValueError("base and larger dimensions differ")
    bad_dims = [d for d in dims if d < 0 or d >= base_view.dim]
    if bad_dims:
        raise ValueError(f"fingerprint dims out of range: {bad_dims}")

    t0 = time.perf_counter()
    print(f"building base fingerprint set with {len(dims)} dims", flush=True)
    base_keys = np.unique(void_keys(base_view.data, dims))

    new_ids: list[np.ndarray] = []
    matched = 0
    for start in range(0, larger_view.n, batch_size):
        end = min(start + batch_size, larger_view.n)
        keys = void_keys(larger_view.data[start:end], dims)
        in_base = np.isin(keys, base_keys, assume_unique=False)
        matched += int(np.count_nonzero(in_base))
        local_new = np.flatnonzero(~in_base).astype(np.int64) + start
        if local_new.size:
            new_ids.append(local_new)
        print(
            f"  membership scanned {end}/{larger_view.n} "
            f"matched={matched} new={sum(x.size for x in new_ids)}",
            flush=True,
        )

    out = np.concatenate(new_ids) if new_ids else np.empty(0, dtype=np.int64)
    return out, matched, time.perf_counter() - t0


def find_larger_rows_not_in_base_multiset(
    base_view: FbinView,
    larger_view: FbinView,
    dims: list[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Match every base row to exactly one larger row and return the complement.

    The old set-membership helper intentionally removed every copy of a vector
    occurring in the base corpus.  That makes a nominal 1M -> 5M experiment a
    few rows short when the corpus contains duplicate embeddings.  Here the
    base fingerprints are treated as a multiset: only as many larger rows as
    occur in the base are consumed.  The selected dimensions use exact float
    bits.  Callers should use enough dimensions (the 5M driver uses 16) to make
    accidental collisions negligible.
    """
    if base_view.dim != larger_view.dim:
        raise ValueError("base and larger dimensions differ")
    bad_dims = [d for d in dims if d < 0 or d >= base_view.dim]
    if bad_dims:
        raise ValueError(f"fingerprint dims out of range: {bad_dims}")

    t0 = time.perf_counter()
    print(f"building base fingerprint multiset with {len(dims)} dims", flush=True)
    base_keys = void_keys(base_view.data, dims)
    base_order = np.argsort(base_keys, kind="stable")
    sorted_keys = base_keys[base_order]
    unique_keys, first, required = np.unique(
        sorted_keys,
        return_index=True,
        return_counts=True,
    )
    consumed = np.zeros(required.size, dtype=np.int64)
    base_to_larger = np.full(base_view.n, -1, dtype=np.int64)
    new_chunks: list[np.ndarray] = []
    matched = 0

    for start in range(0, larger_view.n, batch_size):
        end = min(start + batch_size, larger_view.n)
        keys = void_keys(larger_view.data[start:end], dims)
        pos = np.searchsorted(unique_keys, keys)
        valid = pos < unique_keys.size
        valid_ids = np.flatnonzero(valid)
        if valid_ids.size:
            valid_pos = pos[valid_ids]
            equal = unique_keys[valid_pos] == keys[valid_ids]
            valid_ids = valid_ids[equal]
            valid_pos = valid_pos[equal]

        take_local = np.zeros(end - start, dtype=bool)
        if valid_ids.size:
            order = np.argsort(valid_pos, kind="stable")
            group_sorted = valid_pos[order]
            local_sorted = valid_ids[order]
            group_start_mask = np.empty(group_sorted.size, dtype=bool)
            group_start_mask[0] = True
            group_start_mask[1:] = group_sorted[1:] != group_sorted[:-1]
            group_starts = np.flatnonzero(group_start_mask)
            group_counts = np.diff(np.append(group_starts, group_sorted.size))
            group_ids = group_sorted[group_starts]
            rank = np.arange(group_sorted.size) - np.repeat(group_starts, group_counts)
            before = consumed[group_sorted]
            allowed = rank + before < required[group_sorted]

            chosen_local = local_sorted[allowed]
            chosen_group = group_sorted[allowed]
            chosen_rank = rank[allowed]
            take_local[chosen_local] = True
            larger_ids = chosen_local.astype(np.int64) + start
            base_sorted_positions = first[chosen_group] + consumed[chosen_group] + chosen_rank
            base_ids = base_order[base_sorted_positions]
            base_to_larger[base_ids] = larger_ids
            matched += int(chosen_local.size)

            available = required[group_ids] - consumed[group_ids]
            consumed[group_ids] += np.minimum(group_counts, available)

        local_new = np.flatnonzero(~take_local).astype(np.int64) + start
        if local_new.size:
            new_chunks.append(local_new)
        print(
            f"  multiset membership scanned {end}/{larger_view.n} "
            f"matched={matched} new={sum(x.size for x in new_chunks)}",
            flush=True,
        )

    if matched != base_view.n or np.any(base_to_larger < 0):
        raise RuntimeError(
            f"one-to-one base matching failed: matched={matched}, expected={base_view.n}"
        )
    new_ids = np.concatenate(new_chunks) if new_chunks else np.empty(0, dtype=np.int64)
    if new_ids.size != larger_view.n - base_view.n:
        raise RuntimeError(
            f"multiset complement has {new_ids.size} rows; expected {larger_view.n - base_view.n}"
        )
    return new_ids, base_to_larger, time.perf_counter() - t0


def read_i64bin(path: Path) -> np.ndarray:
    with path.open("rb") as fp:
        header = np.fromfile(fp, dtype=np.uint64, count=1)
        if header.size != 1:
            raise ValueError(f"Bad i64bin header: {path}")
        values = np.fromfile(fp, dtype=np.int64, count=int(header[0]))
    if values.size != int(header[0]):
        raise ValueError(f"Bad i64bin payload: {path}")
    return values


def assign_new_rows_to_base_centroids(
    view: FbinView,
    row_ids: np.ndarray,
    centroids: np.ndarray,
    metric: str,
    normalize: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    cluster_ids = np.empty(row_ids.size, dtype=np.uint32)
    residual_d2 = np.empty(row_ids.size, dtype=np.float32)

    faiss = import_faiss()
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    quantizer = faiss.IndexFlat(view.dim, metric_type)
    quantizer.add(np.ascontiguousarray(centroids, dtype=np.float32))
    centroid_norm2 = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)

    for start in range(0, row_ids.size, batch_size):
        end = min(start + batch_size, row_ids.size)
        block = np.array(view.data[row_ids[start:end]], dtype=np.float32, order="C", copy=True)
        if normalize:
            l2_normalize_rows(block)
        _, ids2 = quantizer.search(block, 1)
        ids = ids2.reshape(-1).astype(np.int64)
        cluster_ids[start:end] = ids.astype(np.uint32)
        dots = np.einsum("ij,ij->i", block, centroids[ids]).astype(np.float32)
        x_norm2 = np.einsum("ij,ij->i", block, block).astype(np.float32)
        d2 = x_norm2 + centroid_norm2[ids] - 2.0 * dots
        residual_d2[start:end] = np.maximum(d2, 0.0)
        print(f"  assigned new rows {end}/{row_ids.size}", flush=True)

    return cluster_ids, np.sqrt(residual_d2).astype(np.float32), time.perf_counter() - t0


def choose_insert_order(
    policy: str,
    new_ids: np.ndarray,
    new_dist: np.ndarray,
    seed: int,
) -> np.ndarray:
    if policy == "far":
        return np.argsort(-new_dist, kind="mergesort").astype(np.int64)
    if policy == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(new_ids.size).astype(np.int64)
    if policy == "source-order":
        return np.arange(new_ids.size, dtype=np.int64)
    raise ValueError(f"unknown policy: {policy}")


def default_fingerprint_dims(dim: int, count: int) -> list[int]:
    count = max(1, min(int(count), int(dim)))
    dims = np.linspace(0, dim - 1, num=count, dtype=np.int64)
    return sorted({int(v) for v in dims.tolist()})


def recall_for_gt_component(
    labels: np.ndarray,
    gt: np.ndarray,
    k: int,
    threshold: int,
    want_insert: bool,
) -> float:
    labels = np.asarray(labels[:, :k], dtype=np.int64)
    gt = np.asarray(gt[:, :k], dtype=np.int64)
    hits = 0
    total = 0
    for i in range(gt.shape[0]):
        found = set(int(v) for v in labels[i])
        for v in gt[i]:
            is_insert = int(v) >= int(threshold)
            if is_insert != want_insert:
                continue
            total += 1
            hits += int(int(v) in found)
    return float(hits / total) if total else math.nan


def label_fraction_at_or_above(labels: np.ndarray, threshold: int, k: int) -> float:
    return float(np.mean(np.asarray(labels[:, :k], dtype=np.int64) >= int(threshold)))


def result_stage_complete(
    rows: list[dict[str, object]], mode: str, insert_size: int, efs: list[int]
) -> bool:
    measured = {
        int(row["efSearch"])
        for row in rows
        if row.get("mode") == mode and int(row.get("insert_size", -1)) == insert_size
    }
    return measured == set(efs)


def search_rows(
    *,
    rows: list[dict[str, object]],
    out: Path,
    idx,
    mode: str,
    dataset: str,
    policy: str,
    queries: np.ndarray,
    gt: np.ndarray,
    base_gt: np.ndarray,
    efs: list[int],
    rounds: int,
    k: int,
    search_threads: int,
    base_size: int,
    insert_size: int,
    corpus_size: int,
    dim: int,
    m: int,
    ef_construction: int,
    nbits: int,
    num_clusters: int,
    metric: str,
    normalize: bool,
    build_seconds: float,
    add_seconds: float,
    exact_seconds: float,
    selected_summary: dict[str, float],
    selected_source_ids: np.ndarray,
    index_path: str,
    index_bytes: int,
    note: str,
    timing_breakdown: dict[str, float] | None = None,
    adaptive_fp_path: Path | None = None,
) -> None:
    timing_breakdown = timing_breakdown or {}
    if hasattr(idx, "search_with_routing_stats"):
        _, _, routing_stats = idx.search_with_routing_stats(
            queries,
            k=k,
            ef=efs[0],
            num_threads=search_threads,
            max_trace=0,
        )
        preprocess_ns = np.asarray(
            routing_stats["preprocess_time_ns"], dtype=np.float64
        )
        preprocess_us = preprocess_ns / 1_000.0
    else:
        preprocess_us = np.zeros(queries.shape[0], dtype=np.float64)
    preprocess_summary = {
        "query_preprocess_us_mean": float(np.mean(preprocess_us)),
        "query_preprocess_us_median": float(np.median(preprocess_us)),
        "query_preprocess_us_p95": float(np.percentile(preprocess_us, 95.0)),
    }
    print(
        f"  {mode} insert={insert_size} query preprocess "
        f"mean={preprocess_summary['query_preprocess_us_mean']:.3f} us "
        f"median={preprocess_summary['query_preprocess_us_median']:.3f} us "
        f"p95={preprocess_summary['query_preprocess_us_p95']:.3f} us",
        flush=True,
    )
    for ef in efs:
        recalls: list[float] = []
        base_gt_recalls: list[float] = []
        insert_gt_recalls: list[float] = []
        current_base_gt_recalls: list[float] = []
        returned_insert_fracs: list[float] = []
        search_seconds_values: list[float] = []
        qps_values: list[float] = []
        fp_read_counts: list[np.ndarray] = []
        adaptive_query_latency_ns: list[np.ndarray] = []
        odirect_read_bytes_values: list[float] = []
        odirect_read_requests_values: list[float] = []
        odirect_read_ns_values: list[float] = []
        fbin_data_offsets: list[int] = []

        for r in range(rounds):
            t0 = time.perf_counter()
            if adaptive_fp_path is None:
                labels, _ = idx.search(
                    queries,
                    k=k,
                    ef=ef,
                    num_threads=search_threads,
                )
            else:
                labels, _, adaptive_stats = idx.search_1bit_adaptive_fp_odirect(
                    queries,
                    str(adaptive_fp_path),
                    k=k,
                    ef=ef,
                    num_threads=search_threads,
                )
                fp_read_counts.append(
                    np.asarray(adaptive_stats["fp_read_count"], dtype=np.float64)
                )
                adaptive_query_latency_ns.append(
                    np.asarray(adaptive_stats["query_latency_ns"], dtype=np.float64)
                )
                odirect_read_bytes_values.append(float(adaptive_stats["odirect_read_bytes"]))
                odirect_read_requests_values.append(
                    float(adaptive_stats["odirect_read_requests"])
                )
                odirect_read_ns_values.append(
                    float(adaptive_stats.get("odirect_read_ns", 0.0))
                )
                fbin_data_offsets.append(int(adaptive_stats["fbin_data_offset"]))
            search_seconds = time.perf_counter() - t0
            labels = np.asarray(labels, dtype=np.int64)
            recall = compute_recall(labels, gt, k)
            base_recall = compute_recall(labels, base_gt, k)
            insert_recall = recall_for_gt_component(labels, gt, k, base_size, True)
            current_base_recall = recall_for_gt_component(labels, gt, k, base_size, False)
            qps = queries.shape[0] / max(search_seconds, 1e-12)

            recalls.append(float(recall))
            base_gt_recalls.append(float(base_recall))
            insert_gt_recalls.append(float(insert_recall))
            current_base_gt_recalls.append(float(current_base_recall))
            returned_insert_fracs.append(label_fraction_at_or_above(labels, base_size, k))
            search_seconds_values.append(float(search_seconds))
            qps_values.append(float(qps))
            print(
                f"  {mode} insert={insert_size} ef={ef} round={r + 1}/{rounds} "
                f"recall_current@{k}={recall:.6f} "
                f"insert_gt={insert_recall:.6f} current_base_gt={current_base_recall:.6f} "
                f"qps={qps:.2f}",
                flush=True,
            )

        all_fp_reads = (
            np.concatenate(fp_read_counts)
            if fp_read_counts
            else np.asarray([], dtype=np.float64)
        )
        all_adaptive_latency_ns = (
            np.concatenate(adaptive_query_latency_ns)
            if adaptive_query_latency_ns
            else np.asarray([], dtype=np.float64)
        )

        row = {
                "dataset": dataset,
                "mode": mode,
                "policy": policy,
                "base_size": int(base_size),
                "insert_size": int(insert_size),
                "corpus_size": int(corpus_size),
                "dim": int(dim),
                "num_clusters": int(num_clusters),
                "M": int(m),
                "efConstruction": int(ef_construction),
                "total_bits": int(nbits),
                "metric": metric,
                "normalize": bool(normalize),
                "efSearch": int(ef),
                "k": int(k),
                "nq": int(queries.shape[0]),
                "rounds": int(rounds),
                "build_or_rebuild_seconds": float(build_seconds),
                "add_seconds": float(add_seconds),
                "exact_gt_seconds": float(exact_seconds),
                "search_seconds_mean": float(np.mean(search_seconds_values)),
                "latency_ms_per_query": float(np.mean(search_seconds_values) * 1000.0 / queries.shape[0]),
                "recall_current_at_k": float(np.mean(recalls)),
                "recall_current_std": float(np.std(recalls)),
                "insert_gt_recall_at_k": float(np.nanmean(insert_gt_recalls))
                if not np.all(np.isnan(insert_gt_recalls))
                else math.nan,
                "current_gt_base_recall_at_k": float(np.nanmean(current_base_gt_recalls))
                if not np.all(np.isnan(current_base_gt_recalls))
                else math.nan,
                "base_gt_recall_at_k": float(np.mean(base_gt_recalls)),
                "qps": float(np.mean(qps_values)),
                "qps_std": float(np.std(qps_values)),
                "rerank_policy": (
                    "rabitq_1bit_adaptive_fp_odirect"
                    if adaptive_fp_path is not None
                    else "none"
                ),
                "fp_reads_mean": float(np.mean(all_fp_reads)) if all_fp_reads.size else 0.0,
                "fp_reads_p50": float(np.percentile(all_fp_reads, 50.0)) if all_fp_reads.size else 0.0,
                "fp_reads_p95": float(np.percentile(all_fp_reads, 95.0)) if all_fp_reads.size else 0.0,
                "fp_reads_p99": float(np.percentile(all_fp_reads, 99.0)) if all_fp_reads.size else 0.0,
                "fp_reads_max": float(np.max(all_fp_reads)) if all_fp_reads.size else 0.0,
                "adaptive_query_latency_ms_mean": (
                    float(np.mean(all_adaptive_latency_ns) / 1e6)
                    if all_adaptive_latency_ns.size
                    else 0.0
                ),
                "adaptive_query_latency_ms_p95": (
                    float(np.percentile(all_adaptive_latency_ns, 95.0) / 1e6)
                    if all_adaptive_latency_ns.size
                    else 0.0
                ),
                "odirect_read_bytes_per_query": (
                    float(np.mean(odirect_read_bytes_values) / queries.shape[0])
                    if odirect_read_bytes_values
                    else 0.0
                ),
                "odirect_read_requests_per_query": (
                    float(np.mean(odirect_read_requests_values) / queries.shape[0])
                    if odirect_read_requests_values
                    else 0.0
                ),
                "odirect_read_ms_per_query": (
                    float(np.mean(odirect_read_ns_values) / 1e6 / queries.shape[0])
                    if odirect_read_ns_values
                    else 0.0
                ),
                "fp_sidecar_path": str(adaptive_fp_path) if adaptive_fp_path else "",
                "fbin_data_offset": fbin_data_offsets[0] if fbin_data_offsets else 0,
                "returned_insert_fraction": float(np.mean(returned_insert_fracs)),
                "current_gt_insert_fraction": label_fraction_at_or_above(gt, base_size, k),
                "selected_centroid_dist_mean": selected_summary["mean"],
                "selected_centroid_dist_p50": selected_summary["p50"],
                "selected_centroid_dist_p90": selected_summary["p90"],
                "selected_centroid_dist_max": selected_summary["max"],
                "selected_source_id_min": int(selected_source_ids.min()) if selected_source_ids.size else -1,
                "selected_source_id_max": int(selected_source_ids.max()) if selected_source_ids.size else -1,
                "index_path": index_path,
                "index_bytes": int(index_bytes),
                "note": note,
            }
        row.update(timing_breakdown)
        row.update(preprocess_summary)
        rows.append(row)
        write_csv(out, rows)


def run() -> None:
    parser = argparse.ArgumentParser(
        description="RaBitQ b=7 online-add vs full-rebuild experiment from MSMARCO Qwen1024 1M -> 3M-new rows."
    )
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--larger-train", type=Path, default=DEFAULT_LARGER)
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--base-gt", type=Path, default=DEFAULT_BASE_GT)
    parser.add_argument(
        "--selected-ids",
        type=Path,
        default=None,
        help="Optional precomputed larger-corpus row order (uint64 count + int64 ids)",
    )
    parser.add_argument(
        "--stage-gt-dir",
        type=Path,
        default=None,
        help="Optional directory containing precomputed qwen1024_1m_plus_* GT files",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--dataset", default="msmarco-qwen1024-1m-plus-3mnew")
    parser.add_argument("--insert-sizes", type=int, nargs="+", default=DEFAULT_INSERT_SIZES)
    parser.add_argument("--policy", choices=["far", "random", "source-order"], default="far")
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument(
        "--rebuild-clusters",
        type=int,
        nargs="+",
        default=None,
        help="Per-checkpoint centroid counts for full rebuilds; defaults to --num-clusters",
    )
    parser.add_argument("--cluster-train-samples", type=int, default=0)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--total-bits", type=int, default=7)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    parser.add_argument("--efs", type=int, nargs="+", default=DEFAULT_EFS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--add-threads", type=int, default=1)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument(
        "--rabitq-binding-dir",
        type=Path,
        default=None,
        help="Optional directory containing a compatible _rabitqlib Python extension",
    )
    parser.add_argument(
        "--adaptive-fp-path",
        type=Path,
        default=None,
        help="Enable RaBitQ 1-bit adaptive FP reranking using this raw fbin via O_DIRECT",
    )
    parser.add_argument(
        "--selected-assignment-cache-prefix",
        type=Path,
        default=None,
        help="Optional prefix for cached selected-stream uint32 cluster ids and float32 residuals",
    )
    parser.add_argument("--membership-batch-size", type=int, default=250_000)
    parser.add_argument("--assign-batch-size", type=int, default=65_536)
    parser.add_argument("--selected-load-batch-size", type=int, default=250_000)
    parser.add_argument("--exact-query-batch", type=int, default=32)
    parser.add_argument("--exact-data-batch", type=int, default=65_536)
    parser.add_argument("--fingerprint-dims", type=int, nargs="+", default=None)
    parser.add_argument("--fingerprint-dim-count", type=int, default=DEFAULT_FINGERPRINT_DIM_COUNT)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-quantization", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--skip-online", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Load existing CSV rows and skip already complete mode/checkpoint stages",
    )
    parser.add_argument(
        "--independent-online-checkpoints",
        action="store_true",
        help="Reload the same base-index snapshot and insert the full prefix independently at each checkpoint",
    )
    parser.add_argument("--save-indexes", action="store_true")
    parser.add_argument(
        "--unlink-larger-after-load",
        action="store_true",
        help="Close and unlink the larger fbin after copying selected rows into RAM",
    )
    parser.add_argument(
        "--reuse-larger-as-raw-data",
        action="store_true",
        help="Use a sequential selected stream directly from the normalized larger fbin instead of copying it",
    )
    args = parser.parse_args()

    if args.adaptive_fp_path is not None:
        if int(args.total_bits) != 1:
            raise ValueError("--adaptive-fp-path requires --total-bits 1")
        if args.adaptive_fp_path.resolve() != args.larger_train.resolve():
            raise ValueError(
                "adaptive FP labels must address the same sequential fbin as --larger-train"
            )
        if args.unlink_larger_after_load:
            raise ValueError("cannot unlink --larger-train while adaptive FP search uses it")

    insert_sizes = parse_int_list(args.insert_sizes)
    if any(v <= 0 for v in insert_sizes):
        raise ValueError("--insert-sizes must be positive")
    max_insert = max(insert_sizes)
    efs = parse_int_list(args.efs)
    rebuild_clusters = (
        [int(args.num_clusters)] * len(insert_sizes)
        if args.rebuild_clusters is None
        # Unlike insert sizes and efSearch values, a per-checkpoint cluster
        # schedule is positional and may intentionally repeat a value (for
        # example fixed C=16 at every checkpoint).  parse_int_list() sorts and
        # deduplicates, which would collapse such a schedule to one entry.
        else [int(value) for value in args.rebuild_clusters]
    )
    if len(rebuild_clusters) != len(insert_sizes) or any(v <= 0 for v in rebuild_clusters):
        raise ValueError("--rebuild-clusters must contain one positive value per insert size")

    faiss = import_faiss()
    try:
        faiss.omp_set_num_threads(max(1, int(args.build_threads)))
    except Exception:
        pass

    base_view = fbin_mmap(args.base_train)
    larger_view = fbin_mmap(args.larger_train)
    larger_shape = [larger_view.n, larger_view.dim]
    if base_view.dim != larger_view.dim:
        raise ValueError("base and larger train dimensions differ")
    queries = read_fbin(args.query, int(args.max_queries))
    if args.normalize:
        l2_normalize_rows(queries)
    base_gt = read_ibin(args.base_gt, queries.shape[0])[:, : int(args.k)]
    print(f"base={args.base_train} shape=({base_view.n}, {base_view.dim})", flush=True)
    print(f"larger={args.larger_train} shape=({larger_view.n}, {larger_view.dim})", flush=True)
    print(f"queries={args.query} shape={queries.shape}", flush=True)

    rows: list[dict[str, object]] = []
    if args.resume_existing and args.out.is_file():
        with args.out.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        print(f"resuming from {len(rows)} existing CSV rows -> {args.out}", flush=True)
    manifest: dict[str, object] = {
        "argv": sys.argv,
        "base_train": str(args.base_train),
        "larger_train": str(args.larger_train),
        "query": str(args.query),
        "base_gt": str(args.base_gt),
        "out": str(args.out),
        "index_dir": str(args.index_dir),
        "save_indexes": bool(args.save_indexes),
        "insert_sizes": insert_sizes,
        "base_num_clusters": int(args.num_clusters),
        "rebuild_clusters": rebuild_clusters,
        "efs": efs,
        "policy": args.policy,
        "selected_ids": str(args.selected_ids) if args.selected_ids else "",
        "stage_gt_dir": str(args.stage_gt_dir) if args.stage_gt_dir else "",
        "rabitq_binding_dir": str(args.rabitq_binding_dir) if args.rabitq_binding_dir else "",
        "adaptive_fp_path": str(args.adaptive_fp_path) if args.adaptive_fp_path else "",
        "note": "a precomputed shared insert stream is used when --selected-ids is supplied",
    }

    fingerprint_dims = (
        list(args.fingerprint_dims)
        if args.fingerprint_dims is not None
        else default_fingerprint_dims(base_view.dim, int(args.fingerprint_dim_count))
    )
    manifest["fingerprint_dims"] = fingerprint_dims

    new_ids: np.ndarray | None = None
    matched_count = 0
    duplicate_vector_matches = 0
    membership_seconds = 0.0
    if args.selected_ids is not None:
        selected_larger_ids = read_i64bin(args.selected_ids)
        if selected_larger_ids.size < max_insert:
            raise ValueError(
                f"selected id file has {selected_larger_ids.size} rows; need {max_insert}"
            )
        selected_larger_ids = selected_larger_ids[:max_insert]
        if np.any(selected_larger_ids < 0) or np.any(selected_larger_ids >= larger_view.n):
            raise ValueError("selected ids are out of range for the larger corpus")
        if np.unique(selected_larger_ids).size != selected_larger_ids.size:
            raise ValueError("selected ids contain duplicates")
        manifest["new_row_count"] = int(selected_larger_ids.size)
    else:
        new_ids, matched_count, membership_seconds = find_larger_rows_not_in_base(
            base_view,
            larger_view,
            fingerprint_dims,
            int(args.membership_batch_size),
        )
        if matched_count < base_view.n:
            raise RuntimeError(
                f"expected to match at least all {base_view.n} base rows inside larger corpus, got {matched_count}; "
                "check corpus pairing"
            )
        duplicate_vector_matches = int(matched_count - base_view.n)
        if duplicate_vector_matches:
            print(
                f"note: found {duplicate_vector_matches} additional larger rows with vectors identical "
                "to existing base rows; excluding them from insert candidates",
                flush=True,
            )
        if max_insert > new_ids.size:
            raise ValueError(f"max insert {max_insert} exceeds new row count {new_ids.size}")
        manifest["new_row_count"] = int(new_ids.size)
        manifest["matched_base_rows_in_larger"] = int(matched_count)
        manifest["duplicate_vector_matches_excluded"] = int(duplicate_vector_matches)
        manifest["membership_seconds"] = float(membership_seconds)

    print("loading existing 1M base rows", flush=True)
    t0 = time.perf_counter()
    base_ids = np.arange(base_view.n, dtype=np.int64)
    base = load_rows(base_view, base_ids, bool(args.normalize))
    base_load_seconds = time.perf_counter() - t0

    print("training centroids on existing 1M base", flush=True)
    base_cluster_started = time.perf_counter()
    centroids, base_cluster_ids, cluster_seconds = train_centroids(
        base,
        int(args.num_clusters),
        args.metric,
        int(args.cluster_train_samples),
        int(args.seed),
    )
    base_cluster_total_seconds = time.perf_counter() - base_cluster_started
    base_cluster_assignment_seconds = max(0.0, base_cluster_total_seconds - cluster_seconds)
    print(f"base centroid training seconds={cluster_seconds:.3f}", flush=True)

    if args.selected_ids is not None:
        cache_cluster = (
            Path(str(args.selected_assignment_cache_prefix) + ".cluster_ids.npy")
            if args.selected_assignment_cache_prefix is not None
            else None
        )
        cache_dist = (
            Path(str(args.selected_assignment_cache_prefix) + ".residual_dist.npy")
            if args.selected_assignment_cache_prefix is not None
            else None
        )
        cache_ok = False
        if cache_cluster is not None and cache_dist is not None:
            try:
                cached_cluster = np.load(cache_cluster, mmap_mode="r")
                cached_dist = np.load(cache_dist, mmap_mode="r")
                cache_ok = (
                    cached_cluster.shape == (selected_larger_ids.size,)
                    and cached_cluster.dtype == np.uint32
                    and cached_dist.shape == (selected_larger_ids.size,)
                    and cached_dist.dtype == np.float32
                )
            except (OSError, ValueError):
                cache_ok = False
        if cache_ok:
            print(f"reusing selected-stream assignment cache -> {cache_cluster}", flush=True)
            selected_cluster_ids = cached_cluster
            selected_dist = cached_dist
            assign_seconds = 0.0
        else:
            print("assigning the shared selected stream to this run's base centroids", flush=True)
            selected_cluster_ids, selected_dist, assign_seconds = assign_new_rows_to_base_centroids(
                larger_view,
                selected_larger_ids,
                centroids,
                args.metric,
                bool(args.normalize),
                int(args.assign_batch_size),
            )
            if cache_cluster is not None and cache_dist is not None:
                cache_cluster.parent.mkdir(parents=True, exist_ok=True)
                tmp_cluster = cache_cluster.with_name(cache_cluster.name + ".tmp.npy")
                tmp_dist = cache_dist.with_name(cache_dist.name + ".tmp.npy")
                np.save(tmp_cluster, np.asarray(selected_cluster_ids, dtype=np.uint32))
                np.save(tmp_dist, np.asarray(selected_dist, dtype=np.float32))
                os.replace(tmp_cluster, cache_cluster)
                os.replace(tmp_dist, cache_dist)
                print(f"saved selected-stream assignment cache -> {cache_cluster}", flush=True)
        new_summary = summarize(selected_dist)
    else:
        assert new_ids is not None
        print("assigning all new larger-corpus rows to existing base centroids", flush=True)
        new_cluster_ids, new_dist, assign_seconds = assign_new_rows_to_base_centroids(
            larger_view,
            new_ids,
            centroids,
            args.metric,
            bool(args.normalize),
            int(args.assign_batch_size),
        )
        new_summary = summarize(new_dist)
        print(
            "new-row residual distance "
            f"mean={new_summary['mean']:.6f} p50={new_summary['p50']:.6f} "
            f"p90={new_summary['p90']:.6f} max={new_summary['max']:.6f}",
            flush=True,
        )
        order = choose_insert_order(args.policy, new_ids, new_dist, int(args.seed) + 17)
        chosen_pos = order[:max_insert]
        selected_larger_ids = new_ids[chosen_pos]
        selected_cluster_ids = new_cluster_ids[chosen_pos]
        selected_dist = new_dist[chosen_pos]
    selected_summary_max = summarize(selected_dist)
    print(f"loading selected insert rows max_insert={max_insert}", flush=True)
    t0 = time.perf_counter()
    if args.reuse_larger_as_raw_data:
        expected = base_view.n
        check_batch = int(args.selected_load_batch_size)
        for start in range(0, max_insert, check_batch):
            end = min(start + check_batch, max_insert)
            if not np.array_equal(
                selected_larger_ids[start:end],
                np.arange(expected + start, expected + end, dtype=np.int64),
            ):
                raise ValueError(
                    "--reuse-larger-as-raw-data requires selected ids to be the sequential "
                    "rows immediately following the base"
                )
        raw_data_max = larger_view.data[: base_view.n + max_insert]
        if args.normalize:
            sample_ids = np.linspace(
                0, base_view.n + max_insert - 1, num=4096, dtype=np.int64
            )
            sample_norms = np.linalg.norm(raw_data_max[sample_ids], axis=1)
            max_norm_error = float(np.max(np.abs(sample_norms - 1.0)))
            if max_norm_error > 2e-3:
                raise ValueError(
                    f"larger fbin is not already L2 normalized (sample max error={max_norm_error})"
                )
            print(
                f"  reusing normalized larger mmap directly; sample max norm error={max_norm_error:.3g}",
                flush=True,
            )
    else:
        raw_data_max = np.empty(
            (base_view.n + max_insert, base_view.dim), dtype=np.float32
        )
        raw_data_max[: base_view.n] = base
        load_batch = int(args.selected_load_batch_size)
        for start in range(0, max_insert, load_batch):
            end = min(start + load_batch, max_insert)
            block = np.array(
                larger_view.data[selected_larger_ids[start:end]],
                dtype=np.float32,
                order="C",
                copy=True,
            )
            if args.normalize:
                l2_normalize_rows(block)
            raw_data_max[base_view.n + start : base_view.n + end] = block
            if end == max_insert or end % 5_000_000 == 0:
                print(f"  loaded selected rows {end}/{max_insert}", flush=True)
    del base
    selected_load_seconds = time.perf_counter() - t0
    print(f"raw_data_max shape={raw_data_max.shape}", flush=True)
    if args.unlink_larger_after_load:
        if not args.reuse_larger_as_raw_data:
            mmap_obj = getattr(larger_view.data, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            del larger_view
        args.larger_train.unlink()
        print(f"unlinked larger tmpfs path -> {args.larger_train}", flush=True)

    HnswIndex = import_hnsw_index(args.rabitq_binding_dir)

    print("building online base index with headroom for inserts", flush=True)
    online_idx = HnswIndex(
        dim=base_view.dim,
        max_elements=(
            base_view.n
            if args.independent_online_checkpoints
            else base_view.n + max_insert
        ),
        M=int(args.m),
        ef_construction=int(args.ef_construction),
        nbits=int(args.total_bits),
        metric=args.metric,
    )
    t0 = time.perf_counter()
    online_idx.build(
        raw_data_max[: base_view.n],
        centroids,
        base_cluster_ids.astype(np.uint32, copy=False),
        num_threads=int(args.build_threads),
        fast_quantization=bool(args.fast_quantization),
    )
    online_base_build_seconds = time.perf_counter() - t0
    print(f"online base build seconds={online_base_build_seconds:.3f}", flush=True)
    saved_index_paths: list[str] = []

    online_base_index_path = ""
    online_base_index_bytes = 0
    if args.save_indexes:
        args.index_dir.mkdir(parents=True, exist_ok=True)
        online_base_index = args.index_dir / (
            f"{args.dataset}.base_only.n{base_view.n}.d{base_view.dim}."
            f"b{args.total_bits}.M{args.m}.efC{args.ef_construction}.index"
        )
        print(f"saving base-only index -> {online_base_index}", flush=True)
        online_idx.save(str(online_base_index))
        online_base_index_path = str(online_base_index)
        online_base_index_bytes = online_base_index.stat().st_size
        saved_index_paths.append(online_base_index_path)

    empty_summary = summarize(np.asarray([], dtype=np.float32))
    if not result_stage_complete(rows, "base_only", 0, efs):
        print("baseline search on existing 1M index", flush=True)
        search_rows(
        rows=rows,
        out=args.out,
        idx=online_idx,
        mode="base_only",
        dataset=args.dataset,
        policy=args.policy,
        queries=queries,
        gt=base_gt,
        base_gt=base_gt,
        efs=efs,
        rounds=int(args.rounds),
        k=int(args.k),
        search_threads=int(args.search_threads),
        base_size=base_view.n,
        insert_size=0,
        corpus_size=base_view.n,
        dim=base_view.dim,
        m=int(args.m),
        ef_construction=int(args.ef_construction),
        nbits=int(args.total_bits),
        num_clusters=int(args.num_clusters),
        metric=args.metric,
        normalize=bool(args.normalize),
        build_seconds=online_base_build_seconds,
        add_seconds=0.0,
        exact_seconds=0.0,
        selected_summary=empty_summary,
        selected_source_ids=np.asarray([], dtype=np.int64),
        index_path=online_base_index_path,
        index_bytes=online_base_index_bytes,
        note="existing 1M baseline using provided 1M GT",
        timing_breakdown={
            "input_load_normalize_seconds": float(base_load_seconds),
            "centroid_train_seconds": float(cluster_seconds),
            "cluster_assignment_seconds": float(base_cluster_assignment_seconds),
            "selected_stream_assignment_seconds": float(assign_seconds),
            "selected_stream_load_normalize_seconds": float(selected_load_seconds),
            "graph_and_rabitq_quantization_seconds": float(online_base_build_seconds),
            "index_build_total_seconds": float(
                base_load_seconds
                + cluster_seconds
                + base_cluster_assignment_seconds
                + online_base_build_seconds
            ),
            "online_insert_graph_and_quantization_seconds": 0.0,
            "online_insert_us_per_vector": 0.0,
        },
            adaptive_fp_path=args.adaptive_fp_path,
        )
    else:
        print("skip completed base_only search", flush=True)

    if args.independent_online_checkpoints and (
        not args.skip_online
        or any(row.get("mode") == "online_add_no_rebuild" for row in rows)
    ):
        del online_idx
        manifest["online_checkpoint_source"] = "base_index_rebuilt_for_each_checkpoint"
    else:
        manifest["online_checkpoint_source"] = "cumulative_reuse"

    prev_insert = 0
    for insert_size, rebuild_num_clusters in zip(insert_sizes, rebuild_clusters):
        corpus_size = base_view.n + insert_size
        stage_data = raw_data_max[:corpus_size]
        stage_selected_dist = selected_dist[:insert_size]
        stage_selected_ids = selected_larger_ids[:insert_size]
        stage_summary = summarize(stage_selected_dist)

        if args.stage_gt_dir is not None:
            gt_path = args.stage_gt_dir / (
                f"qwen1024_1m_plus_{insert_size}_{args.policy}_gt_k{args.k}_q{queries.shape[0]}.ibin"
            )
            print(f"\nloading precomputed GT -> {gt_path}", flush=True)
            current_gt = read_ibin(gt_path, queries.shape[0])[:, : int(args.k)]
            exact_seconds = 0.0
        else:
            print(f"\ncomputing exact GT for custom corpus insert={insert_size}", flush=True)
            t0 = time.perf_counter()
            if args.metric == "ip":
                current_gt = exact_topk_ip(
                    queries,
                    stage_data,
                    int(args.k),
                    int(args.exact_query_batch),
                    int(args.exact_data_batch),
                )
            else:
                current_gt = exact_topk_l2(
                    queries,
                    stage_data,
                    int(args.k),
                    int(args.exact_query_batch),
                    int(args.exact_data_batch),
                )
            exact_seconds = time.perf_counter() - t0

        if not args.skip_online:
            online_stage_idx = None if args.independent_online_checkpoints else online_idx
            online_stage_base_build_seconds = online_base_build_seconds
            add_start = prev_insert
            if args.independent_online_checkpoints:
                print(
                    f"building fixed 1M base for independent insert={insert_size}",
                    flush=True,
                )
                online_stage_idx = HnswIndex(
                    dim=base_view.dim,
                    max_elements=corpus_size,
                    M=int(args.m),
                    ef_construction=int(args.ef_construction),
                    nbits=int(args.total_bits),
                    metric=args.metric,
                )
                stage_base_started = time.perf_counter()
                online_stage_idx.build(
                    raw_data_max[: base_view.n],
                    centroids,
                    base_cluster_ids.astype(np.uint32, copy=False),
                    num_threads=int(args.build_threads),
                    fast_quantization=bool(args.fast_quantization),
                )
                online_stage_base_build_seconds = time.perf_counter() - stage_base_started
                add_start = 0
            assert online_stage_idx is not None
            print(f"online add {add_start}->{insert_size}", flush=True)
            add_labels = np.arange(base_view.n + add_start, corpus_size, dtype=np.uint32)
            add_cluster_ids = selected_cluster_ids[add_start:insert_size].astype(np.uint32, copy=False)
            t0 = time.perf_counter()
            online_stage_idx.add(
                raw_data_max,
                add_labels,
                add_cluster_ids,
                num_threads=int(args.add_threads),
                fast_quantization=bool(args.fast_quantization),
            )
            add_seconds = time.perf_counter() - t0
            online_index_path = ""
            online_index_bytes = 0
            if args.save_indexes:
                online_index = args.index_dir / (
                    f"{args.dataset}.online_no_rebuild.{args.policy}.ins{insert_size}."
                    f"n{corpus_size}.d{base_view.dim}.b{args.total_bits}.M{args.m}."
                    f"efC{args.ef_construction}.index"
                )
                print(f"saving online-add index -> {online_index}", flush=True)
                online_stage_idx.save(str(online_index))
                online_index_path = str(online_index)
                online_index_bytes = online_index.stat().st_size
                saved_index_paths.append(online_index_path)
            search_rows(
                rows=rows,
                out=args.out,
                idx=online_stage_idx,
                mode="online_add_no_rebuild",
                dataset=args.dataset,
                policy=args.policy,
                queries=queries,
                gt=current_gt,
                base_gt=base_gt,
                efs=efs,
                rounds=int(args.rounds),
                k=int(args.k),
                search_threads=int(args.search_threads),
                base_size=base_view.n,
                insert_size=insert_size,
                corpus_size=corpus_size,
                dim=base_view.dim,
                m=int(args.m),
                ef_construction=int(args.ef_construction),
                nbits=int(args.total_bits),
                num_clusters=int(args.num_clusters),
                metric=args.metric,
                normalize=bool(args.normalize),
                build_seconds=online_stage_base_build_seconds,
                add_seconds=add_seconds,
                exact_seconds=exact_seconds,
                selected_summary=stage_summary,
                selected_source_ids=stage_selected_ids,
                index_path=online_index_path,
                index_bytes=online_index_bytes,
                note=(
                    "fixed base snapshot reloaded; full insert prefix added without reclustering"
                    if args.independent_online_checkpoints
                    else "base centroids/graph kept; new rows are added online without reclustering"
                ),
                timing_breakdown={
                    "input_load_normalize_seconds": float(base_load_seconds),
                    "centroid_train_seconds": float(cluster_seconds),
                    "cluster_assignment_seconds": float(base_cluster_assignment_seconds),
                    "selected_stream_assignment_seconds": float(assign_seconds),
                    "selected_stream_load_normalize_seconds": float(selected_load_seconds),
                    "graph_and_rabitq_quantization_seconds": float(online_stage_base_build_seconds),
                    "index_build_total_seconds": float(
                        base_load_seconds
                        + cluster_seconds
                        + base_cluster_assignment_seconds
                        + online_stage_base_build_seconds
                    ),
                    "online_insert_graph_and_quantization_seconds": float(add_seconds),
                    "online_insert_us_per_vector": float(
                        add_seconds * 1e6 / max(insert_size - add_start, 1)
                    ),
                },
                adaptive_fp_path=args.adaptive_fp_path,
            )
            if args.independent_online_checkpoints:
                del online_stage_idx
        prev_insert = insert_size

        if not args.skip_rebuild and not result_stage_complete(
            rows, "full_rebuild_oracle", insert_size, efs
        ):
            print(f"full rebuild oracle insert={insert_size}", flush=True)
            recluster_started = time.perf_counter()
            rebuild_centroids, rebuild_cluster_ids, rebuild_cluster_seconds = train_centroids(
                stage_data,
                int(rebuild_num_clusters),
                args.metric,
                int(args.cluster_train_samples),
                int(args.seed) + insert_size,
            )
            rebuild_cluster_total_seconds = time.perf_counter() - recluster_started
            rebuild_assignment_seconds = max(
                0.0, rebuild_cluster_total_seconds - rebuild_cluster_seconds
            )
            rebuild_idx = HnswIndex(
                dim=base_view.dim,
                max_elements=corpus_size,
                M=int(args.m),
                ef_construction=int(args.ef_construction),
                nbits=int(args.total_bits),
                metric=args.metric,
            )
            rebuild_graph_started = time.perf_counter()
            rebuild_idx.build(
                stage_data,
                rebuild_centroids,
                rebuild_cluster_ids.astype(np.uint32, copy=False),
                num_threads=int(args.build_threads),
                fast_quantization=bool(args.fast_quantization),
            )
            rebuild_graph_quantization_seconds = time.perf_counter() - rebuild_graph_started
            rebuild_seconds = (
                rebuild_cluster_total_seconds + rebuild_graph_quantization_seconds
            )
            rebuild_index_path = ""
            rebuild_index_bytes = 0
            if args.save_indexes:
                rebuild_index = args.index_dir / (
                    f"{args.dataset}.full_rebuild_oracle.{args.policy}.ins{insert_size}."
                    f"n{corpus_size}.d{base_view.dim}.b{args.total_bits}.M{args.m}."
                    f"efC{args.ef_construction}.index"
                )
                print(f"saving rebuild oracle index -> {rebuild_index}", flush=True)
                rebuild_idx.save(str(rebuild_index))
                rebuild_index_path = str(rebuild_index)
                rebuild_index_bytes = rebuild_index.stat().st_size
                saved_index_paths.append(rebuild_index_path)
            search_rows(
                rows=rows,
                out=args.out,
                idx=rebuild_idx,
                mode="full_rebuild_oracle",
                dataset=args.dataset,
                policy=args.policy,
                queries=queries,
                gt=current_gt,
                base_gt=base_gt,
                efs=efs,
                rounds=int(args.rounds),
                k=int(args.k),
                search_threads=int(args.search_threads),
                base_size=base_view.n,
                insert_size=insert_size,
                corpus_size=corpus_size,
                dim=base_view.dim,
                m=int(args.m),
                ef_construction=int(args.ef_construction),
                nbits=int(args.total_bits),
                num_clusters=int(rebuild_num_clusters),
                metric=args.metric,
                normalize=bool(args.normalize),
                build_seconds=rebuild_seconds,
                add_seconds=0.0,
                exact_seconds=exact_seconds,
                selected_summary=stage_summary,
                selected_source_ids=stage_selected_ids,
                index_path=rebuild_index_path,
                index_bytes=rebuild_index_bytes,
                note=(
                    f"full rebuild C={rebuild_num_clusters} with "
                    f"recluster_seconds={rebuild_cluster_seconds:.6f}"
                ),
                timing_breakdown={
                    "input_load_normalize_seconds": 0.0,
                    "centroid_train_seconds": float(rebuild_cluster_seconds),
                    "cluster_assignment_seconds": float(rebuild_assignment_seconds),
                    "selected_stream_assignment_seconds": 0.0,
                    "selected_stream_load_normalize_seconds": 0.0,
                    "graph_and_rabitq_quantization_seconds": float(
                        rebuild_graph_quantization_seconds
                    ),
                    "index_build_total_seconds": float(rebuild_seconds),
                    "online_insert_graph_and_quantization_seconds": 0.0,
                    "online_insert_us_per_vector": 0.0,
                },
                adaptive_fp_path=args.adaptive_fp_path,
            )
            del rebuild_idx
        elif not args.skip_rebuild:
            print(f"skip completed full rebuild oracle insert={insert_size}", flush=True)

    manifest.update(
        {
            "base_shape": [base_view.n, base_view.dim],
            "larger_shape": larger_shape,
            "query_shape": list(queries.shape),
            "base_load_seconds": float(base_load_seconds),
            "base_cluster_seconds": float(cluster_seconds),
            "base_centroid_train_seconds": float(cluster_seconds),
            "base_cluster_assignment_seconds": float(base_cluster_assignment_seconds),
            "new_assign_seconds": float(assign_seconds),
            "selected_load_seconds": float(selected_load_seconds),
            "online_base_build_seconds": float(online_base_build_seconds),
            "new_residual_summary": new_summary,
            "selected_max_insert_residual_summary": selected_summary_max,
            "saved_index_paths": saved_index_paths,
        }
    )
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_csv(args.out, rows)
    print(f"\nCSV saved -> {args.out}", flush=True)
    print(f"Manifest saved -> {manifest_path}", flush=True)


if __name__ == "__main__":
    run()
