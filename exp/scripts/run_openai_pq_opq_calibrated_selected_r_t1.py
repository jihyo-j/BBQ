#!/usr/bin/env python3
"""Calibrate one fixed rerank R for exact-1-bit/dim Faiss PQ/OPQ.

R is selected independently for every dataset/method from the joint (ef, R)
Pareto frontier on calibration queries.  Only that selected R is measured on
the held-out queries.  PQ and OPQ use M=d/8 eight-bit subquantizers, hence an
exact d/8-byte routing code for every vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import diskveclab_full_matrix as matrix
import run_diskveclab_tq_pq_opq_uring as low
import run_nq_qwen_multidim_calibrated_adcbq_tq_t1 as selection


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/openai_beir_11methods_t1/faiss_pq_opq_calibrated"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1/faiss_pq_opq_work"
)
EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
RERANKS = (10, 20, 30, 40, 60, 80, 120, 160, 200)
METHODS = {
    "pq8d8": "Faiss PQ8 (d/8-byte code)",
    "opq8d8": "Faiss OPQ-PQ8 (d/8-byte code)",
}
RAW_METHOD_NAMES = {
    "pq8d8": "Faiss-PQ8-d/8",
    "opq8d8": "Faiss-OPQ-PQ8-d/8",
}
PAGE_BYTES = 4096.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def configure() -> None:
    low.RESULT_ROOT = RESULT_ROOT / "work"
    low.EXTERNAL_ROOT = EXTERNAL_ROOT
    low.INDEX_ROOT = EXTERNAL_ROOT / "indexes"
    low.SIDECAR_ROOT = EXTERNAL_ROOT / "sidecars"
    low.SUMMARY = RESULT_ROOT / "work/unused_summary.csv"
    low.STATUS = RESULT_ROOT / "work/status.json"
    low.LOCK = RESULT_ROOT / "work/queue.lock"


def shared_sidecar(dataset_key: str) -> tuple[Path, bool]:
    manifest_path = os.environ.get("DISKVECLAB_EXTRA_DATASETS_MANIFEST", "").strip()
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for item in payload.get("datasets", []):
            if item.get("key") == dataset_key:
                candidate = Path(str(item["fp_sidecar"]))
                if low.valid_sidecar(candidate, dataset_key):
                    print(f"REUSE shared aligned FP sidecar {candidate}", flush=True)
                    return candidate, False
    return low.build_sidecar(dataset_key), True


def arm(method_key: str, rerank_n: int) -> low.Method:
    return low.Method(method_key, f"{METHODS[method_key]} + FP@{rerank_n}", rerank_n)


def one_bit_budget(dataset_key: str) -> tuple[int, int, int]:
    rows, dim = matrix.registry.matrix_shape(
        matrix.registry.DATASETS[dataset_key].train
    )
    if dim % 8:
        raise RuntimeError(
            f"exact 1-bit/dim Faiss code requires dimension divisible by 8: "
            f"{dataset_key} d={dim}"
        )
    code_bytes = dim // 8
    return rows, code_bytes, rows * code_bytes


def raw_path(
    dataset: str, method: str, phase: str, rerank_n: int, search_threads: int
) -> Path:
    return (
        RESULT_ROOT / "raw" / dataset / method
        / f"{phase}.r{rerank_n}.t{search_threads}.csv"
    )


def raw_complete(
    path: Path,
    dataset: str,
    method: str,
    rerank_n: int,
    rounds: int,
    search_threads: int,
) -> bool:
    rows = read_csv(path)
    expected = {(ef, rid) for ef in EFS for rid in range(1, rounds + 1)}
    _, code_bytes, budget_bytes = one_bit_budget(dataset)
    try:
        return (
            {(int(row["efSearch"]), int(row["round"])) for row in rows} == expected
            and all(
                row["dataset"] == dataset
                and row["method"] == RAW_METHOD_NAMES[method]
                and int(row["threads"]) == search_threads
                and int(row["rerank_n"]) == rerank_n
                and row["code_storage"] == "resident"
                and row["fp_io_mode"] == "uring-batch"
                and int(row["code_bytes_per_vector"]) == code_bytes
                and int(row["bq_budget_bytes"]) == budget_bytes
                for row in rows
            )
        )
    except (KeyError, ValueError):
        return False


def run_curve(
    dataset_key: str,
    method_key: str,
    rerank_n: int,
    *,
    query: Path,
    gt: Path,
    nq: int,
    rounds: int,
    phase: str,
    sidecar: Path,
    cpu: int,
    search_threads: int,
    force: bool,
) -> Path:
    output = raw_path(
        dataset_key, method_key, phase, rerank_n, search_threads
    )
    if not force and raw_complete(
        output, dataset_key, method_key, rerank_n, rounds, search_threads
    ):
        print(f"REUSE {output}", flush=True)
        return output
    dataset = matrix.registry.DATASETS[dataset_key]
    _, _, budget_bytes = one_bit_budget(dataset_key)
    codec, _ = low.codec_paths(dataset_key, method_key)
    for path in (query, gt, dataset.adcbq_index, codec, sidecar):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset", "-c",
        ",".join(map(str, selection.affinity_cpus(cpu, search_threads))),
        str(low.search_runner.BINARY),
        "--dataset", dataset_key,
        "--method", method_key,
        "--metric", "ip",
        "--query", str(query),
        "--gt", str(gt),
        "--graph", str(dataset.adcbq_index),
        "--codec-index", str(codec),
        "--sidecar", str(EXTERNAL_ROOT / "unused_resident_code_sidecar"),
        "--fp-sidecar", str(sidecar),
        "--out", str(output),
        "--max-queries", str(nq),
        "--rounds", str(rounds),
        "--threads", str(search_threads),
        "--rerank-n", str(rerank_n),
        "--resident-codes",
        "--fp-io-mode", "uring-batch",
        "--fp-rerank-qd", str(rerank_n),
        "--bq-budget-bytes", str(budget_bytes),
        "--efs", *map(str, EFS),
    ]
    print("RUN " + " ".join(command), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(search_threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=REPO, env=environment, check=True)
    if not raw_complete(
        output, dataset_key, method_key, rerank_n, rounds, search_threads
    ):
        raise RuntimeError(f"incomplete result: {output}")
    return output


def summarize(
    path: Path,
    dataset_key: str,
    method_key: str,
    rerank_n: int,
    phase: str,
    search_threads: int,
) -> list[dict[str, object]]:
    dataset = matrix.registry.DATASETS[dataset_key]
    _, expected_code_bytes, expected_budget_bytes = one_bit_budget(dataset_key)
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(int(row["efSearch"]), []).append(row)
    output: list[dict[str, object]] = []
    for ef in EFS:
        rows = grouped[ef]
        output.append(
            {
                "dataset": dataset_key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method_key": method_key,
                "method": f"{METHODS[method_key]} + FP@{rerank_n}",
                "search_parameter": "efSearch",
                "search_parameter_value": ef,
                "efSearch": ef,
                "rerank_n": rerank_n,
                "recall_at_10": statistics.median(float(row["recall_at_10"]) for row in rows),
                "qps": statistics.median(float(row["qps"]) for row in rows),
                "qps_stddev": statistics.pstdev(float(row["qps"]) for row in rows),
                "rounds": len(rows),
                "search_threads": search_threads,
                "code_bytes_per_vector": expected_code_bytes,
                "routing_bits_per_dimension": 1.0,
                "routing_code_budget_bytes": expected_budget_bytes,
                "routing_code_budget_policy": "d/8 bytes per vector",
                "fp_vectors_per_query": rerank_n,
                "fp_4k_pages_per_query": statistics.median(
                    float(row["fp_read_bytes_per_query"]) / PAGE_BYTES for row in rows
                ),
                "phase": phase,
                "source_csv": str(path.relative_to(REPO)),
            }
        )
    return output


def main() -> int:
    global RESULT_ROOT, EXTERNAL_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pilot-rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--opq-niter", type=int, default=25)
    parser.add_argument("--opq-blas-threads", type=int, default=8)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(
        args.rounds,
        args.pilot_rounds,
        args.build_threads,
        args.search_threads,
        args.train_size,
    ) < 1:
        parser.error("numeric arguments must be positive")
    RESULT_ROOT = args.result_root.resolve()
    EXTERNAL_ROOT = args.external_root.resolve()
    configure()
    low.codec_builder.compile_binary()
    low.search_runner.compile_binary()
    datasets = list(matrix.CALIBRATION_ROWS)
    all_pilot: list[dict[str, object]] = []
    all_selected: list[dict[str, object]] = []
    all_heldout: list[dict[str, object]] = []
    build_args = SimpleNamespace(
        build_threads=args.build_threads,
        train_size=args.train_size,
        opq_niter=args.opq_niter,
        opq_blas_threads=args.opq_blas_threads,
    )
    for dataset_key in datasets:
        dataset = matrix.registry.DATASETS[dataset_key]
        matrix.registry.validate(dataset)
        split = matrix.split_for(dataset_key)
        sidecar, transient_sidecar = shared_sidecar(dataset_key)
        try:
            for method_key in args.methods:
                codec_arm = arm(method_key, RERANKS[0])
                low.build_codec(dataset_key, codec_arm, build_args)
                try:
                    pilot_rows: list[dict[str, object]] = []
                    for rerank_n in RERANKS:
                        raw = run_curve(
                            dataset_key,
                            method_key,
                            rerank_n,
                            query=split.calibration_query,
                            gt=split.calibration_gt,
                            nq=split.calibration_rows,
                            rounds=args.pilot_rounds,
                            phase="calibration_pareto",
                            sidecar=sidecar,
                            cpu=args.cpu,
                            search_threads=args.search_threads,
                            force=args.force,
                        )
                        rows = summarize(
                            raw,
                            dataset_key,
                            method_key,
                            rerank_n,
                            "calibration_pareto",
                            args.search_threads,
                        )
                        pilot_rows.extend(rows)
                        all_pilot.extend(rows)
                        write_csv(
                            RESULT_ROOT
                            / f"calibration_r_sweep_t{args.search_threads}.csv",
                            all_pilot,
                        )
                    chosen = selection.choose_r(dataset_key, pilot_rows)
                    chosen["method_key"] = method_key
                    all_selected.append(chosen)
                    write_csv(RESULT_ROOT / "selected_r.csv", all_selected)
                    selected_r = int(chosen["selected_r"])
                    print(
                        f"SELECTED {dataset_key}/{method_key} R={selected_r} "
                        f"ef={chosen['selection_ef']}",
                        flush=True,
                    )
                    raw = run_curve(
                        dataset_key,
                        method_key,
                        selected_r,
                        query=split.test_query,
                        gt=split.test_gt,
                        nq=split.test_rows,
                        rounds=args.rounds,
                        phase="heldout_selected",
                        sidecar=sidecar,
                        cpu=args.cpu,
                        search_threads=args.search_threads,
                        force=args.force,
                    )
                    heldout = summarize(
                        raw,
                        dataset_key,
                        method_key,
                        selected_r,
                        "heldout_selected",
                        args.search_threads,
                    )
                    for row in heldout:
                        row.update(
                            {
                                "selected_r": selected_r,
                                "calibration_queries": split.calibration_rows,
                                "test_queries": split.test_rows,
                                "selection_objective": chosen["selection_objective"],
                            }
                        )
                    all_heldout.extend(heldout)
                    write_csv(
                        RESULT_ROOT
                        / f"selected_r_heldout_t{args.search_threads}.csv",
                        all_heldout,
                    )
                finally:
                    low.reclaim_codec(dataset_key, codec_arm)
        finally:
            if transient_sidecar and sidecar.is_file():
                sidecar.unlink()
                print(f"REMOVED transient sidecar {sidecar}", flush=True)
    status = RESULT_ROOT / "status.json"
    status.write_text(
        json.dumps(
            {
                "stage": "complete",
                "datasets": datasets,
                "methods": args.methods,
                "search_threads": args.search_threads,
                "protocol": "calibration joint-(ef,R) Pareto; fixed-R held-out",
                "routing_code_budget_policy": "exactly d/8 bytes per vector (1 bit/dim)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
