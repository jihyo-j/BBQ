#!/usr/bin/env python3
"""Sweep TurboQuant fixed rerank budgets on all 16 DiskVecLab held-out sets.

The already measured FP@60 and FP@80 curves are imported.  Missing budgets
are measured with resident TQ1+ codes and one O_DIRECT io_uring FP batch per
query.  Outputs include the joint (efSearch, R) Recall/QPS Pareto frontier and
the maximum-QPS point reaching Recall@10 >= 0.99 for every dataset.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sys
from pathlib import Path

import diskveclab_full_matrix as matrix
import run_diskveclab_tq_pq_opq_uring as run


RESULT_ROOT = matrix.RESULT_ROOT / "tq_rerank_pareto_uring"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_tq_rerank_pareto_uring"
)
SUMMARY = RESULT_ROOT / "summary_t1.csv"
FRONTIER = RESULT_ROOT / "recall_qps_pareto_frontier_t1.csv"
TARGET = RESULT_ROOT / "target99_max_qps_t1.csv"
STATUS = RESULT_ROOT / "queue_status.json"
LOCK = RESULT_ROOT / "queue.lock"
RERANKS = (10, 20, 30, 40, 60, 80, 120, 160, 200)
TARGET_RECALL = 0.99
SOURCE_ROOTS = {
    60: matrix.RESULT_ROOT / "tq60_uring/raw",
    80: matrix.RESULT_ROOT / "tq_pq_opq_uring/raw",
}


def configure() -> None:
    run.RESULT_ROOT = RESULT_ROOT
    run.EXTERNAL_ROOT = EXTERNAL_ROOT
    run.INDEX_ROOT = EXTERNAL_ROOT / "indexes"
    run.SIDECAR_ROOT = EXTERNAL_ROOT / "transient_aligned_fp"
    run.SUMMARY = SUMMARY
    run.STATUS = STATUS
    run.LOCK = LOCK


def method(rerank_n: int) -> run.Method:
    return run.Method(
        "qdranttq1", f"TurboQuant TQ1+ + FP@{rerank_n}", rerank_n
    )


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


def import_existing(dataset_key: str, rerank_n: int) -> None:
    source = (
        SOURCE_ROOTS[rerank_n]
        / dataset_key
        / f"qdranttq1.r{rerank_n}.csv"
    )
    arm = method(rerank_n)
    expected_efs = efs_for(rerank_n)
    destination = run.raw_path(dataset_key, arm)
    if run.raw_complete(destination, dataset_key, arm, expected_efs):
        return
    if not run.raw_complete(source, dataset_key, arm, expected_efs):
        raise RuntimeError(f"missing/incompatible reusable FP@{rerank_n} curve: {source}")
    fieldnames, rows = run.read_raw_rows(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run.write_raw_rows(destination, fieldnames, rows)
    print(f"IMPORTED {dataset_key} FP@{rerank_n}: {source}", flush=True)


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
        for rerank_n in RERANKS:
            arm = method(rerank_n)
            efs = efs_for(rerank_n)
            path = run.raw_path(dataset_key, arm)
            if run.raw_complete(path, dataset_key, arm, efs):
                all_rows.extend(run.normalized_rows(path, dataset_key, arm))
    dataset_order = {key: i for i, key in enumerate(matrix.CALIBRATION_ROWS)}
    all_rows.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            int(row["rerank_n"]),
            int(row["efSearch"]),
        )
    )
    atomic_csv(SUMMARY, all_rows)

    frontier_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for dataset_key in matrix.CALIBRATION_ROWS:
        dataset_rows = [
            row for row in all_rows if row["dataset"] == dataset_key
        ]
        dataset_frontier = pareto(dataset_rows)
        frontier_rows.extend(dataset_frontier)
        feasible = [
            row
            for row in dataset_frontier
            if float(row["recall_at_10"]) >= TARGET_RECALL
        ]
        if feasible:
            chosen = max(feasible, key=lambda row: float(row["qps"]))
            reached = True
        elif dataset_frontier:
            chosen = max(
                dataset_frontier,
                key=lambda row: (float(row["recall_at_10"]), float(row["qps"])),
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
            float(row["recall_at_10"]),
        )
    )
    target_rows.sort(key=lambda row: dataset_order[str(row["dataset"])])
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
        raise SystemExit("another TurboQuant rerank Pareto queue is running")

    # Supplements are private, atomic scratch outputs.  A surviving one means
    # the previous process died before it could publish a complete raw curve.
    for stale in RESULT_ROOT.rglob("*.supplement.*.tmp"):
        print(f"REMOVED stale supplement: {stale}", flush=True)
        stale.unlink()

    expected_points_per_dataset = sum(len(efs_for(r)) for r in RERANKS)
    expected_points = expected_points_per_dataset * len(matrix.CALIBRATION_ROWS)
    total_new_curves = len(matrix.CALIBRATION_ROWS) * (len(RERANKS) - 2)
    completed_new_curves = sum(
        run.raw_complete(
            run.raw_path(dataset_key, method(rerank_n)),
            dataset_key,
            method(rerank_n),
            efs_for(rerank_n),
        )
        for dataset_key in matrix.CALIBRATION_ROWS
        for rerank_n in RERANKS
        if rerank_n not in SOURCE_ROOTS
    )
    run.write_status(
        phase="compiling",
        reranks=list(RERANKS),
        efs=list(run.DEFAULT_EFS),
        target_recall=TARGET_RECALL,
        expected_points=expected_points,
        completed_new_curves=completed_new_curves,
        total_new_curves=total_new_curves,
    )
    try:
        run.check_disk("rerank sweep startup")
        run.codec_builder.compile_binary()
        run.search_runner.compile_binary()
        for dataset_index, dataset_key in enumerate(
            matrix.CALIBRATION_ROWS, start=1
        ):
            matrix.registry.validate(matrix.registry.DATASETS[dataset_key])
            matrix.split_for(dataset_key)
            import_existing(dataset_key, 60)
            import_existing(dataset_key, 80)
            missing = [
                r
                for r in RERANKS
                if not run.raw_complete(
                    run.raw_path(dataset_key, method(r)),
                    dataset_key,
                    method(r),
                    efs_for(r),
                )
            ]
            fp_sidecar: Path | None = None
            codec_arm = method(10)
            try:
                if missing:
                    run.write_status(
                        phase="building_codec",
                        current_dataset=dataset_key,
                        dataset_index=dataset_index,
                        total_datasets=len(matrix.CALIBRATION_ROWS),
                    )
                    run.retry(
                        lambda: run.build_codec(dataset_key, codec_arm, args),
                        f"TQ codec {dataset_key}",
                    )
                    run.write_status(
                        phase="building_fp_sidecar",
                        current_dataset=dataset_key,
                    )
                    fp_sidecar = run.retry(
                        lambda: run.build_sidecar(dataset_key),
                        f"FP sidecar {dataset_key}",
                    )
                for rerank_n in missing:
                    arm = method(rerank_n)
                    arm_args = argparse.Namespace(**vars(args))
                    arm_args.efs = efs_for(rerank_n)
                    run.write_status(
                        phase="searching",
                        current_dataset=dataset_key,
                        current_rerank_n=rerank_n,
                        completed_new_curves=completed_new_curves,
                        total_new_curves=total_new_curves,
                    )
                    run.retry(
                        lambda arm=arm, arm_args=arm_args: run.run_search(
                            dataset_key, arm, fp_sidecar, arm_args
                        ),
                        f"TQ search {dataset_key}/R{rerank_n}",
                    )
                    completed_new_curves += 1
                    rows, selected = rebuild_outputs()
                    run.write_status(
                        completed_new_curves=completed_new_curves,
                        completed_points=rows,
                        selected_datasets=selected,
                    )
            finally:
                owned_sidecar = run.sidecar_path(dataset_key)
                if owned_sidecar.exists():
                    size = owned_sidecar.stat().st_size
                    owned_sidecar.unlink()
                    print(
                        f"REMOVED transient FP sidecar bytes={size}: {owned_sidecar}",
                        flush=True,
                    )
                run.reclaim_codec(dataset_key, codec_arm)

        rows, selected = rebuild_outputs()
        if rows != expected_points or selected != len(matrix.CALIBRATION_ROWS):
            raise RuntimeError(
                f"incomplete sweep: rows={rows}/{expected_points}, "
                f"selected={selected}/{len(matrix.CALIBRATION_ROWS)}"
            )
        run.write_status(
            phase="complete",
            current_dataset=None,
            current_rerank_n=None,
            completed_new_curves=total_new_curves,
            completed_points=rows,
            selected_datasets=selected,
            summary=str(SUMMARY),
            frontier=str(FRONTIER),
            target_summary=str(TARGET),
        )
        print(
            f"ALL COMPLETE: {rows} points; {TARGET}",
            flush=True,
        )
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
