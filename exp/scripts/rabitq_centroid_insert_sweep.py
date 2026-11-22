#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_base.fbin")
DEFAULT_QUERY = Path("/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_query.fbin")
DEFAULT_OUT = Path("/tmp/rabitq_centroid_insert_sweep.csv")
DEFAULT_INDEX_DIR = Path("/tmp/rabitq_centroid_insert_indexes")
DEFAULT_EFS = [64, 128, 256, 512]
DEFAULT_POLICIES = ["none", "random", "near", "far", "far-balanced"]


@dataclass(frozen=True)
class FbinView:
    path: Path
    n: int
    dim: int
    data: np.memmap


def import_hnsw_index(binding_dir: Path | None = None):
    binding_dir = binding_dir or (REPO / "build_stats/python_bindings")
    if str(binding_dir) not in sys.path:
        sys.path.insert(0, str(binding_dir))
    from _rabitqlib import HnswIndex  # type: ignore

    return HnswIndex


def import_faiss():
    try:
        import faiss  # type: ignore
    except Exception:
        faiss_path = Path("/home/jihyo/workspace/faiss/build_hnsw_conda/faiss/python")
        conda_site = Path("/home/jihyo/anaconda3/lib/python3.12/site-packages")
        if faiss_path.exists() and str(faiss_path) not in sys.path:
            sys.path.insert(0, str(faiss_path))
        if conda_site.exists() and str(conda_site) not in sys.path:
            sys.path.append(str(conda_site))
        import faiss  # type: ignore

    return faiss


def fbin_mmap(path: Path) -> FbinView:
    with path.open("rb") as fp:
        header = np.fromfile(fp, dtype=np.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"Bad fbin header: {path}")
    n, dim = int(header[0]), int(header[1])
    payload = n * dim * np.dtype(np.float32).itemsize
    legacy_size = 8 + payload
    aligned_size = 4096 + payload
    actual = path.stat().st_size
    if actual == legacy_size:
        offset = 8
    elif actual == aligned_size:
        offset = 4096
    else:
        raise ValueError(
            f"Bad fbin size for {path}: expected {legacy_size} (legacy) or "
            f"{aligned_size} (4KiB-aligned), got {actual}"
        )
    data = np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(n, dim))
    return FbinView(path=path, n=n, dim=dim, data=data)


def read_fbin(path: Path, max_rows: int = 0) -> np.ndarray:
    view = fbin_mmap(path)
    n = min(view.n, max_rows) if max_rows and max_rows > 0 else view.n
    return np.array(view.data[:n], dtype=np.float32, order="C", copy=True)


def l2_normalize_rows(x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    x = np.array(x, dtype=np.float32, order="C", copy=True) if not x.flags.writeable else np.ascontiguousarray(x, dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        block = x[start : start + batch_size]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        block /= norms
    return x


def choose_split(n: int, base_size: int, holdout_size: int, seed: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    need = base_size + holdout_size
    if need > n:
        raise ValueError(f"base_size + holdout_size exceeds source rows: {need} > {n}")
    if mode == "prefix":
        base_ids = np.arange(base_size, dtype=np.int64)
        holdout_ids = np.arange(base_size, need, dtype=np.int64)
        return base_ids, holdout_ids
    if mode != "random":
        raise ValueError("--split-mode must be prefix or random")
    rng = np.random.default_rng(seed)
    ids = rng.choice(n, size=need, replace=False).astype(np.int64)
    return ids[:base_size], ids[base_size:]


def load_rows(view: FbinView, ids: np.ndarray, normalize: bool) -> np.ndarray:
    x = np.array(view.data[ids], dtype=np.float32, order="C", copy=True)
    if normalize:
        l2_normalize_rows(x)
    return x


def train_centroids(base: np.ndarray, num_clusters: int, metric: str, train_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    faiss = import_faiss()
    dim = base.shape[1]
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    index = faiss.index_factory(dim, f"IVF{num_clusters},Flat", metric_type)
    index.verbose = True
    index.cp.seed = int(seed)

    train_x = base
    if train_samples > 0 and train_samples < base.shape[0]:
        rng = np.random.default_rng(seed + 17)
        sample_ids = rng.choice(base.shape[0], size=train_samples, replace=False)
        train_x = np.ascontiguousarray(base[sample_ids], dtype=np.float32)

    t0 = time.perf_counter()
    index.train(train_x)
    seconds = time.perf_counter() - t0
    centroids = np.ascontiguousarray(index.quantizer.reconstruct_n(0, index.nlist), dtype=np.float32)
    _, base_cluster_2d = index.quantizer.search(base, 1)
    base_cluster_ids = np.ascontiguousarray(base_cluster_2d.reshape(-1), dtype=np.uint32)
    return centroids, base_cluster_ids, seconds


def assign_and_residual_distance(x: np.ndarray, centroids: np.ndarray, metric: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    faiss = import_faiss()
    dim = x.shape[1]
    metric_type = faiss.METRIC_INNER_PRODUCT if metric == "ip" else faiss.METRIC_L2
    quantizer = faiss.IndexFlat(dim, metric_type)
    quantizer.add(centroids)

    cluster_ids = np.empty(x.shape[0], dtype=np.uint32)
    residual_d2 = np.empty(x.shape[0], dtype=np.float32)
    centroid_norm2 = np.einsum("ij,ij->i", centroids, centroids).astype(np.float32)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        block = np.ascontiguousarray(x[start:end], dtype=np.float32)
        _, ids2 = quantizer.search(block, 1)
        ids = ids2.reshape(-1).astype(np.int64)
        cluster_ids[start:end] = ids.astype(np.uint32)
        dots = np.einsum("ij,ij->i", block, centroids[ids]).astype(np.float32)
        x_norm2 = np.einsum("ij,ij->i", block, block).astype(np.float32)
        d2 = x_norm2 + centroid_norm2[ids] - 2.0 * dots
        residual_d2[start:end] = np.maximum(d2, 0.0)
    return cluster_ids, residual_d2


def quota_select_by_cluster(
    candidate_positions: np.ndarray,
    cluster_ids: np.ndarray,
    distances: np.ndarray,
    take: int,
) -> np.ndarray:
    if take <= 0:
        return np.empty(0, dtype=np.int64)
    clusters, counts = np.unique(cluster_ids[candidate_positions], return_counts=True)
    exact = counts.astype(np.float64) * (float(take) / float(candidate_positions.size))
    quotas = np.floor(exact).astype(np.int64)
    remainder = int(take - quotas.sum())
    if remainder > 0:
        frac_order = np.argsort(-(exact - quotas))
        quotas[frac_order[:remainder]] += 1

    selected: list[np.ndarray] = []
    for cid, quota in zip(clusters, quotas):
        if quota <= 0:
            continue
        pos = candidate_positions[cluster_ids[candidate_positions] == cid]
        order = np.argsort(-distances[pos], kind="stable")
        selected.append(pos[order[:quota]])
    if not selected:
        return np.empty(0, dtype=np.int64)
    out = np.concatenate(selected)
    if out.size > take:
        out = out[:take]
    elif out.size < take:
        selected_set = set(int(v) for v in out.tolist())
        fill_order = np.argsort(-distances[candidate_positions], kind="stable")
        fill = [int(candidate_positions[i]) for i in fill_order if int(candidate_positions[i]) not in selected_set]
        out = np.concatenate([out, np.asarray(fill[: take - out.size], dtype=np.int64)])
    return out.astype(np.int64, copy=False)


def select_insert_positions(
    policy: str,
    insert_size: int,
    holdout_cluster_ids: np.ndarray,
    holdout_distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = holdout_distances.shape[0]
    if insert_size < 0 or insert_size > n:
        raise ValueError(f"bad insert size {insert_size}; holdout has {n} rows")
    if policy == "none":
        if insert_size != 0:
            raise ValueError("policy=none only supports insert_size=0")
        return np.empty(0, dtype=np.int64)
    if insert_size == 0:
        return np.empty(0, dtype=np.int64)
    candidate_positions = np.arange(n, dtype=np.int64)
    if policy == "random":
        return rng.choice(n, size=insert_size, replace=False).astype(np.int64)
    if policy == "near":
        return np.argsort(holdout_distances, kind="stable")[:insert_size].astype(np.int64)
    if policy == "far":
        return np.argsort(-holdout_distances, kind="stable")[:insert_size].astype(np.int64)
    if policy == "far-balanced":
        return quota_select_by_cluster(candidate_positions, holdout_cluster_ids, holdout_distances, insert_size)
    raise ValueError(f"unknown policy: {policy}")


def exact_topk_ip(queries: np.ndarray, data: np.ndarray, k: int, query_batch: int, data_batch: int) -> np.ndarray:
    nq = queries.shape[0]
    labels = np.empty((nq, k), dtype=np.int64)
    for qs in range(0, nq, query_batch):
        qe = min(qs + query_batch, nq)
        q = np.ascontiguousarray(queries[qs:qe], dtype=np.float32)
        best_scores = np.full((qe - qs, k), -np.inf, dtype=np.float32)
        best_ids = np.full((qe - qs, k), -1, dtype=np.int64)
        for ds in range(0, data.shape[0], data_batch):
            de = min(ds + data_batch, data.shape[0])
            scores = q @ data[ds:de].T
            block_k = min(k, scores.shape[1])
            part = np.argpartition(scores, -block_k, axis=1)[:, -block_k:]
            part_scores = np.take_along_axis(scores, part, axis=1)
            part_ids = part.astype(np.int64) + ds
            merged_scores = np.concatenate([best_scores, part_scores], axis=1)
            merged_ids = np.concatenate([best_ids, part_ids], axis=1)
            keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(merged_scores, keep, axis=1)
            best_ids = np.take_along_axis(merged_ids, keep, axis=1)
        order = np.argsort(-best_scores, axis=1)
        labels[qs:qe] = np.take_along_axis(best_ids, order, axis=1)
    return labels


def exact_topk_l2(queries: np.ndarray, data: np.ndarray, k: int, query_batch: int, data_batch: int) -> np.ndarray:
    nq = queries.shape[0]
    labels = np.empty((nq, k), dtype=np.int64)
    data_norm2 = np.einsum("ij,ij->i", data, data).astype(np.float32)
    for qs in range(0, nq, query_batch):
        qe = min(qs + query_batch, nq)
        q = np.ascontiguousarray(queries[qs:qe], dtype=np.float32)
        q_norm2 = np.einsum("ij,ij->i", q, q).astype(np.float32)[:, None]
        best_dist = np.full((qe - qs, k), np.inf, dtype=np.float32)
        best_ids = np.full((qe - qs, k), -1, dtype=np.int64)
        for ds in range(0, data.shape[0], data_batch):
            de = min(ds + data_batch, data.shape[0])
            dist = q_norm2 + data_norm2[ds:de][None, :] - 2.0 * (q @ data[ds:de].T)
            block_k = min(k, dist.shape[1])
            part = np.argpartition(dist, block_k - 1, axis=1)[:, :block_k]
            part_dist = np.take_along_axis(dist, part, axis=1)
            part_ids = part.astype(np.int64) + ds
            merged_dist = np.concatenate([best_dist, part_dist], axis=1)
            merged_ids = np.concatenate([best_ids, part_ids], axis=1)
            keep = np.argpartition(merged_dist, k - 1, axis=1)[:, :k]
            best_dist = np.take_along_axis(merged_dist, keep, axis=1)
            best_ids = np.take_along_axis(merged_ids, keep, axis=1)
        order = np.argsort(best_dist, axis=1)
        labels[qs:qe] = np.take_along_axis(best_ids, order, axis=1)
    return labels


def compute_recall(labels: np.ndarray, gt: np.ndarray, k: int) -> float:
    labels = np.asarray(labels[:, :k], dtype=np.int64)
    gt = np.asarray(gt[:, :k], dtype=np.int64)
    hits = 0
    for i in range(labels.shape[0]):
        truth = set(int(v) for v in gt[i])
        hits += sum(int(v) in truth for v in labels[i])
    return hits / float(labels.shape[0] * k)


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(values: Iterable[int]) -> list[int]:
    out = sorted({int(v) for v in values})
    if not out:
        raise ValueError("empty integer list")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RaBitQ HNSW centroid-distance insertion stress sweep."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--save-indexes", action="store_true")
    parser.add_argument("--dataset", default="msmarco-qwen1024-insert")
    parser.add_argument("--base-size", type=int, default=100000)
    parser.add_argument("--holdout-size", type=int, default=50000)
    parser.add_argument("--insert-sizes", type=int, nargs="+", default=[0, 1000, 5000, 10000])
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--split-mode", choices=["random", "prefix"], default="random")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--cluster-train-samples", type=int, default=0)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--total-bits", type=int, default=7)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    parser.add_argument("--efs", type=int, nargs="+", default=DEFAULT_EFS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=1, help="Use 1 to preserve base-then-insert order.")
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--assign-batch-size", type=int, default=65536)
    parser.add_argument("--exact-query-batch", type=int, default=32)
    parser.add_argument("--exact-data-batch", type=int, default=65536)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-quantization", action="store_true")
    args = parser.parse_args()

    insert_sizes = parse_int_list(args.insert_sizes)
    efs = parse_int_list(args.efs)
    policies = list(dict.fromkeys(args.policies))
    if 0 not in insert_sizes:
        insert_sizes = [0] + insert_sizes
    if "none" not in policies:
        policies = ["none"] + policies

    source = fbin_mmap(args.train)
    queries = read_fbin(args.query, args.max_queries)
    if args.normalize:
        l2_normalize_rows(queries)
    print(f"source={source.path} shape=({source.n}, {source.dim})", flush=True)
    print(f"queries={args.query} shape={queries.shape}", flush=True)

    base_ids, holdout_ids = choose_split(
        source.n,
        int(args.base_size),
        int(args.holdout_size),
        int(args.seed),
        args.split_mode,
    )
    print(
        f"split mode={args.split_mode} base={base_ids.size} holdout={holdout_ids.size} seed={args.seed}",
        flush=True,
    )

    t0 = time.perf_counter()
    base = load_rows(source, base_ids, args.normalize)
    holdout = load_rows(source, holdout_ids, args.normalize)
    load_seconds = time.perf_counter() - t0
    print(f"loaded base+holdout in {load_seconds:.3f}s", flush=True)

    centroids, base_cluster_ids, cluster_seconds = train_centroids(
        base,
        int(args.num_clusters),
        args.metric,
        int(args.cluster_train_samples),
        int(args.seed),
    )
    holdout_cluster_ids, holdout_d2 = assign_and_residual_distance(
        holdout,
        centroids,
        args.metric,
        int(args.assign_batch_size),
    )
    holdout_dist = np.sqrt(holdout_d2).astype(np.float32)
    all_holdout_summary = summarize(holdout_dist)
    print(
        "holdout residual distance "
        f"mean={all_holdout_summary['mean']:.6f} "
        f"p50={all_holdout_summary['p50']:.6f} "
        f"p90={all_holdout_summary['p90']:.6f} "
        f"max={all_holdout_summary['max']:.6f}",
        flush=True,
    )

    HnswIndex = import_hnsw_index()
    rng = np.random.default_rng(int(args.seed) + 1009)
    rows: list[dict[str, object]] = []
    args.index_dir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        for insert_size in insert_sizes:
            if policy == "none" and insert_size != 0:
                continue
            if policy != "none" and insert_size == 0:
                continue

            print(f"\npolicy={policy} insert_size={insert_size}", flush=True)
            policy_seed = int(args.seed) + 1009 + sum((i + 1) * ord(ch) for i, ch in enumerate(policy)) + int(insert_size)
            select_rng = np.random.default_rng(policy_seed)
            selected_pos = select_insert_positions(
                policy,
                int(insert_size),
                holdout_cluster_ids,
                holdout_dist,
                select_rng,
            )
            selected_source_ids = holdout_ids[selected_pos]
            selected_d = holdout_dist[selected_pos]
            selected_summary = summarize(selected_d)

            data = np.vstack([base, holdout[selected_pos]]).astype(np.float32, copy=False)
            cluster_ids = np.concatenate([base_cluster_ids, holdout_cluster_ids[selected_pos]]).astype(np.uint32)
            print(f"build data shape={data.shape}", flush=True)

            idx = HnswIndex(
                dim=source.dim,
                max_elements=data.shape[0],
                M=int(args.m),
                ef_construction=int(args.ef_construction),
                nbits=int(args.total_bits),
                metric=args.metric,
            )
            t_build = time.perf_counter()
            idx.build(
                data,
                centroids,
                cluster_ids,
                num_threads=int(args.build_threads),
                fast_quantization=bool(args.fast_quantization),
            )
            build_seconds = time.perf_counter() - t_build
            index_path = ""
            index_bytes = 0
            if args.save_indexes:
                index_path_obj = args.index_dir / (
                    f"{args.dataset}.{policy}.ins{insert_size}.n{data.shape[0]}."
                    f"b{args.total_bits}.M{args.m}.efC{args.ef_construction}.index"
                )
                idx.save(str(index_path_obj))
                index_path = str(index_path_obj)
                index_bytes = index_path_obj.stat().st_size

            print("computing exact GT for selected corpus", flush=True)
            t_exact = time.perf_counter()
            if args.metric == "ip":
                gt = exact_topk_ip(
                    queries,
                    data,
                    int(args.k),
                    int(args.exact_query_batch),
                    int(args.exact_data_batch),
                )
            else:
                gt = exact_topk_l2(
                    queries,
                    data,
                    int(args.k),
                    int(args.exact_query_batch),
                    int(args.exact_data_batch),
                )
            exact_seconds = time.perf_counter() - t_exact

            for ef in efs:
                recalls: list[float] = []
                qps_values: list[float] = []
                search_seconds_values: list[float] = []
                for r in range(int(args.rounds)):
                    t_search = time.perf_counter()
                    labels, _ = idx.search(
                        queries,
                        k=int(args.k),
                        ef=int(ef),
                        num_threads=int(args.search_threads),
                    )
                    search_seconds = time.perf_counter() - t_search
                    labels = np.asarray(labels, dtype=np.int64)
                    recall = compute_recall(labels, gt, int(args.k))
                    qps = queries.shape[0] / max(search_seconds, 1e-12)
                    recalls.append(float(recall))
                    qps_values.append(float(qps))
                    search_seconds_values.append(float(search_seconds))
                    print(
                        f"  ef={ef} round={r + 1}/{args.rounds} "
                        f"recall@{args.k}={recall:.6f} qps={qps:.2f}",
                        flush=True,
                    )

                rows.append(
                    {
                        "dataset": args.dataset,
                        "train": str(args.train),
                        "query": str(args.query),
                        "split_mode": args.split_mode,
                        "seed": int(args.seed),
                        "policy": policy,
                        "base_size": int(base_ids.size),
                        "holdout_size": int(holdout_ids.size),
                        "insert_size": int(insert_size),
                        "corpus_size": int(data.shape[0]),
                        "dim": int(source.dim),
                        "num_clusters": int(args.num_clusters),
                        "M": int(args.m),
                        "efConstruction": int(args.ef_construction),
                        "total_bits": int(args.total_bits),
                        "metric": args.metric,
                        "efSearch": int(ef),
                        "k": int(args.k),
                        "nq": int(queries.shape[0]),
                        "rounds": int(args.rounds),
                        "build_threads": int(args.build_threads),
                        "search_threads": int(args.search_threads),
                        "normalize": bool(args.normalize),
                        "fast_quantization": bool(args.fast_quantization),
                        "load_seconds": float(load_seconds),
                        "cluster_seconds": float(cluster_seconds),
                        "build_seconds": float(build_seconds),
                        "exact_gt_seconds": float(exact_seconds),
                        "search_seconds_mean": float(np.mean(search_seconds_values)),
                        "latency_ms_per_query": float(np.mean(search_seconds_values) * 1000.0 / queries.shape[0]),
                        "recall_at_k": float(np.mean(recalls)),
                        "recall_std": float(np.std(recalls)),
                        "qps": float(np.mean(qps_values)),
                        "qps_std": float(np.std(qps_values)),
                        "holdout_centroid_dist_mean": all_holdout_summary["mean"],
                        "holdout_centroid_dist_p50": all_holdout_summary["p50"],
                        "holdout_centroid_dist_p90": all_holdout_summary["p90"],
                        "selected_centroid_dist_mean": selected_summary["mean"],
                        "selected_centroid_dist_p50": selected_summary["p50"],
                        "selected_centroid_dist_p90": selected_summary["p90"],
                        "selected_centroid_dist_max": selected_summary["max"],
                        "selected_source_id_min": int(selected_source_ids.min()) if selected_source_ids.size else -1,
                        "selected_source_id_max": int(selected_source_ids.max()) if selected_source_ids.size else -1,
                        "index_bytes": int(index_bytes),
                        "index_path": index_path,
                    }
                )
                write_csv(args.out, rows)

            del idx
            del data
            del cluster_ids

    write_csv(args.out, rows)
    manifest = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "out": str(args.out),
                "source_shape": [source.n, source.dim],
                "query_shape": list(queries.shape),
                "policies": policies,
                "insert_sizes": insert_sizes,
                "efs": efs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nCSV saved -> {args.out}", flush=True)
    print(f"Manifest saved -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
