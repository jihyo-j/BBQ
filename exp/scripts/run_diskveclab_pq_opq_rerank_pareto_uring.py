#!/usr/bin/env python3
"""Sweep Faiss PQ/OPQ rerank budgets on all 16 DiskVecLab held-out sets.

For each codec and dataset, measure the joint (efSearch, R) space with
resident compressed codes and one O_DIRECT io_uring FP batch per query.  The
already measured PQ FP@80 and OPQ FP@60 curves are imported.  Outputs include
the per-codec Recall/QPS Pareto frontier and the maximum-QPS point reaching
Recall@10 >= 0.99 for every method/dataset pair.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import math
import os
import sys
from pathlib import Path

import diskveclab_full_matrix as matrix
import run_diskveclab_tq_pq_opq_uring as run


RESULT_ROOT = matrix.RESULT_ROOT / "pq_opq_rerank_pareto_uring"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_pq_opq_rerank_pareto_uring"
)
SUMMARY = RESULT_ROOT / "summary_t1.csv"
FRONTIER = RESULT_ROOT / "recall_qps_pareto_frontier_t1.csv"
TARGET = RESULT_ROOT / "target99_max_qps_t1.csv"
STATUS = RESULT_ROOT / "queue_status.json"
LOCK = RESULT_ROOT / "queue.lock"
RERANKS = (10, 20, 30, 40, 60, 80, 120, 160, 200)
METHOD_KEYS = ("pq8m128", "opq8m128")
METHOD_LABELS = {
    "pq8m128": "Faiss PQ8-M128",
    "opq8m128": "Faiss OPQ-PQ8-M128",
}
TARGET_RECALL = 0.99
FIXED_ROOT = matrix.RESULT_ROOT / "tq_pq_opq_uring/raw"
FIXED_INDEX_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_tq_pq_opq_uring/indexes"
)
SOURCE_CURVES = {
    ("pq8m128", 80): FIXED_ROOT,
    ("opq8m128", 60): FIXED_ROOT,
}


def configure() -> None:
    run.RESULT_ROOT = RESULT_ROOT
    run.EXTERNAL_ROOT = EXTERNAL_ROOT
    run.INDEX_ROOT = EXTERNAL_ROOT / "indexes"
    run.SIDECAR_ROOT = EXTERNAL_ROOT / "transient_aligned_fp"
    run.SUMMARY = SUMMARY
    run.STATUS = STATUS
    run.LOCK = LOCK


def method(method_key: str, rerank_n: int) -> run.Method:
    return run.Method(
        method_key,
        f"{METHOD_LABELS[method_key]} + FP@{rerank_n}",
        rerank_n,
    )


def link_reusable_codec(dataset_key: str, method_key: str) -> bool:
    """Link an intact fixed-run codec into this run without copying it."""
    if dataset_key in run.PREBUILT_DATASET_NAMES:
        return False
    destination, _ = run.codec_paths(dataset_key, method_key)
    if destination.exists():
        return destination.is_symlink()
    source = (
        FIXED_INDEX_ROOT
        / dataset_key
        / f"{dataset_key}.{method_key}.faiss"
    )
    if not source.is_file() or source.stat().st_size <= 4096:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)
    print(
        f"LINKED reusable codec {dataset_key}/{method_key}: {source}",
        flush=True,
    )
    return True


def reclaim_codec(dataset_key: str, arm: run.Method, linked: bool) -> None:
    if linked:
        index, _ = run.codec_paths(dataset_key, arm.key)
        index.unlink(missing_ok=True)
        print(
            f"REMOVED reusable codec link {dataset_key}/{arm.key}",
            flush=True,
        )
        return
    run.reclaim_codec(dataset_key, arm)


def efs_for(rerank_n: int) -> list[int]:
    return [ef for ef in run.DEFAULT_EFS if ef >= rerank_n]


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def import_existing(dataset_key: str, method_key: str, rerank_n: int) -> None:
    source_root = SOURCE_CURVES[(method_key, rerank_n)]
    source = source_root / dataset_key / f"{method_key}.r{rerank_n}.csv"
    arm = method(method_key, rerank_n)
    expected_efs = efs_for(rerank_n)
    destination = run.raw_path(dataset_key, arm)
    if run.raw_complete(destination, dataset_key, arm, expected_efs):
        return
    if not run.raw_complete(source, dataset_key, arm, expected_efs):
        raise RuntimeError(f"missing/incompatible reusable curve: {source}")
    fieldnames, rows = run.read_raw_rows(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run.write_raw_rows(destination, fieldnames, rows)
    print(
        f"IMPORTED {dataset_key} {method_key} FP@{rerank_n}: {source}",
        flush=True,
    )


def pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_at_recall: dict[float, dict[str, object]] = {}
    for row in rows:
        recall = float(row["recall_at_10"])
        old = best_at_recall.get(recall)
        if old is None or float(row["qps"]) > float(old["qps"]):
            best_at_recall[recall] = row
    frontier_desc: list[dict[str, object]] = []
    best_qps = -math.inf
    for recall in sorted(best_at_recall, reverse=True):
        row = best_at_recall[recall]
        qps = float(row["qps"])
        if qps > best_qps:
            frontier_desc.append(row)
            best_qps = qps
    return list(reversed(frontier_desc))


def rebuild_outputs() -> tuple[int, int]:
    all_rows: list[dict[str, object]] = []
    for dataset_key in matrix.CALIBRATION_ROWS:
        for method_key in METHOD_KEYS:
            for rerank_n in RERANKS:
                arm = method(method_key, rerank_n)
                efs = efs_for(rerank_n)
                path = run.raw_path(dataset_key, arm)
                if run.raw_complete(path, dataset_key, arm, efs):
                    all_rows.extend(run.normalized_rows(path, dataset_key, arm))
    dataset_order = {key: i for i, key in enumerate(matrix.CALIBRATION_ROWS)}
    method_order = {key: i for i, key in enumerate(METHOD_KEYS)}
    all_rows.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method_key"])],
            int(row["rerank_n"]),
            int(row["efSearch"]),
        )
    )
    atomic_csv(SUMMARY, all_rows)

    frontier_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for dataset_key in matrix.CALIBRATION_ROWS:
        for method_key in METHOD_KEYS:
            method_rows = [
                row
                for row in all_rows
                if row["dataset"] == dataset_key
                and row["method_key"] == method_key
            ]
            method_frontier = pareto(method_rows)
            frontier_rows.extend(method_frontier)
            feasible = [
                row
                for row in method_frontier
                if float(row["recall_at_10"]) >= TARGET_RECALL
            ]
            if feasible:
                chosen = max(feasible, key=lambda row: float(row["qps"]))
                reached = True
            elif method_frontier:
                chosen = max(
                    method_frontier,
                    key=lambda row: (
                        float(row["recall_at_10"]),
                        float(row["qps"]),
                    ),
                )
                reached = False
            else:
                continue
            target_rows.append(
                {
                    **chosen,
                    "target_recall": TARGET_RECALL,
                    "target_reached": reached,
                    "selection_objective": "max_qps_on_joint_ef_rerank_pareto",
                }
            )
    frontier_rows.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method_key"])],
            float(row["recall_at_10"]),
        )
    )
    target_rows.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order[str(row["method_key"])],
        )
    )
    atomic_csv(FRONTIER, frontier_rows)
    atomic_csv(TARGET, target_rows)
    return len(all_rows), len(target_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--opq-niter", type=int, default=25)
    parser.add_argument("--opq-blas-threads", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    configure()
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another PQ/OPQ rerank Pareto queue is running")

    for stale in RESULT_ROOT.rglob("*.supplement.*.tmp"):
        print(f"REMOVED stale supplement: {stale}", flush=True)
        stale.unlink()

    expected_points_per_arm = sum(len(efs_for(r)) for r in RERANKS)
    expected_points = (
        expected_points_per_arm
        * len(matrix.CALIBRATION_ROWS)
        * len(METHOD_KEYS)
    )
    total_new_curves = len(matrix.CALIBRATION_ROWS) * (
        len(METHOD_KEYS) * len(RERANKS) - len(SOURCE_CURVES)
    )
    completed_new_curves = sum(
        run.raw_complete(
            run.raw_path(dataset_key, method(method_key, rerank_n)),
            dataset_key,
            method(method_key, rerank_n),
            efs_for(rerank_n),
        )
        for dataset_key in matrix.CALIBRATION_ROWS
        for method_key in METHOD_KEYS
        for rerank_n in RERANKS
        if (method_key, rerank_n) not in SOURCE_CURVES
    )
    run.write_status(
        phase="compiling",
        methods=list(METHOD_KEYS),
        reranks=list(RERANKS),
        efs=list(run.DEFAULT_EFS),
        target_recall=TARGET_RECALL,
        expected_points=expected_points,
        completed_new_curves=completed_new_curves,
        total_new_curves=total_new_curves,
    )
    try:
        run.check_disk("PQ/OPQ rerank sweep startup")
        run.codec_builder.compile_binary()
        run.search_runner.compile_binary()
        for dataset_index, dataset_key in enumerate(
            matrix.CALIBRATION_ROWS, start=1
        ):
            matrix.registry.validate(matrix.registry.DATASETS[dataset_key])
            matrix.split_for(dataset_key)
            for method_key, rerank_n in SOURCE_CURVES:
                import_existing(dataset_key, method_key, rerank_n)

            missing_by_method = {
                method_key: [
                    rerank_n
                    for rerank_n in RERANKS
                    if not run.raw_complete(
                        run.raw_path(
                            dataset_key, method(method_key, rerank_n)
                        ),
                        dataset_key,
                        method(method_key, rerank_n),
                        efs_for(rerank_n),
                    )
                ]
                for method_key in METHOD_KEYS
            }
            fp_sidecar: Path | None = None
            try:
                for method_index, method_key in enumerate(METHOD_KEYS, start=1):
                    missing = missing_by_method[method_key]
                    if not missing:
                        continue
                    codec_arm = method(method_key, 10)
                    linked_codec = link_reusable_codec(dataset_key, method_key)
                    try:
                        run.write_status(
                            phase="building_codec",
                            current_dataset=dataset_key,
                            dataset_index=dataset_index,
                            total_datasets=len(matrix.CALIBRATION_ROWS),
                            current_method=method_key,
                            method_index=method_index,
                            total_methods=len(METHOD_KEYS),
                        )
                        run.retry(
                            lambda codec_arm=codec_arm: run.build_codec(
                                dataset_key, codec_arm, args
                            ),
                            f"codec {dataset_key}/{method_key}",
                        )
                        if fp_sidecar is None:
                            run.write_status(
                                phase="building_fp_sidecar",
                                current_dataset=dataset_key,
                                current_method=method_key,
                            )
                            fp_sidecar = run.retry(
                                lambda: run.build_sidecar(dataset_key),
                                f"FP sidecar {dataset_key}",
                            )
                        for rerank_n in missing:
                            arm = method(method_key, rerank_n)
                            arm_args = argparse.Namespace(**vars(args))
                            arm_args.efs = efs_for(rerank_n)
                            run.write_status(
                                phase="searching",
                                current_dataset=dataset_key,
                                current_method=method_key,
                                current_rerank_n=rerank_n,
                                completed_new_curves=completed_new_curves,
                                total_new_curves=total_new_curves,
                            )
                            assert fp_sidecar is not None
                            run.retry(
                                lambda arm=arm, arm_args=arm_args: run.run_search(
                                    dataset_key, arm, fp_sidecar, arm_args
                                ),
                                f"search {dataset_key}/{method_key}/R{rerank_n}",
                            )
                            completed_new_curves += 1
                            rows, selected = rebuild_outputs()
                            run.write_status(
                                completed_new_curves=completed_new_curves,
                                completed_points=rows,
                                selected_configurations=selected,
                            )
                    finally:
                        reclaim_codec(dataset_key, codec_arm, linked_codec)
            finally:
                owned_sidecar = run.sidecar_path(dataset_key)
                if owned_sidecar.exists():
                    size = owned_sidecar.stat().st_size
                    owned_sidecar.unlink()
                    print(
                        f"REMOVED transient FP sidecar bytes={size}: "
                        f"{owned_sidecar}",
                        flush=True,
                    )

        rows, selected = rebuild_outputs()
        expected_selected = len(matrix.CALIBRATION_ROWS) * len(METHOD_KEYS)
        if rows != expected_points or selected != expected_selected:
            raise RuntimeError(
                f"incomplete sweep: rows={rows}/{expected_points}, "
                f"selected={selected}/{expected_selected}"
            )
        run.write_status(
            phase="complete",
            current_dataset=None,
            current_method=None,
            current_rerank_n=None,
            completed_new_curves=total_new_curves,
            completed_points=rows,
            selected_configurations=selected,
            summary=str(SUMMARY),
            frontier=str(FRONTIER),
            target_summary=str(TARGET),
        )
        print(f"ALL COMPLETE: {rows} points; {TARGET}", flush=True)
        return 0
    except run.StorageRisk as error:
        run.write_status(phase="stopped_storage_floor", error=str(error))
        print(f"STOPPED FOR STORAGE SAFETY: {error}", file=sys.stderr, flush=True)
        return 75
    except Exception as error:
        run.write_status(phase="failed", error=repr(error))
        raise
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
