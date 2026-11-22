#!/usr/bin/env python3
"""Remeasure per-ef calibrated ADC-BQ and TurboQuant on NQ Qwen dimensions.

The three Qwen embedding dimensions use the existing held-out split
(1,000 calibration queries, seed 11, remaining queries for test).  At every
efSearch independently, ADC-BQ uses the latest split-conformal protocol to
fit a threshold that retains all ten exact-FP nearest candidates inside the
ADC candidate pool with target probability 0.99.

TurboQuant first sweeps fixed rerank budgets on calibration queries.  For
each dataset, the selected R is the maximum-QPS point on the joint (ef, R)
Pareto frontier that reaches Recall@10 >= 0.99.  The selected R is then
remeasured for three rounds on held-out test queries.  Compressed codes are
resident and exact FP32 reranking uses one O_DIRECT io_uring batch per query.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import diskveclab_full_matrix as matrix
import run_adcbq_calibrated_threshold_overlap as adc
import run_adcbq_vs_fp_hnsw_fullvec_ram as adc_pool
import run_diskveclab_tq_pq_opq_uring as tq
import run_nq_quora_adcbq_calibrated_io_rerank as adc_panel


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/nq_qwen_multidim_calibrated_adcbq_tq_t1"
ADC_PROTOCOL = "per-ef-pool-top10-split-conformal"
ADC_CALIBRATION_ROOT = RESULT_ROOT / "adcbq_pool_top10_target99_per_ef"
ADC_RAW_ROOT = RESULT_ROOT / "adcbq_pool_top10_target99_per_ef_raw"
ADC_SUMMARY = RESULT_ROOT / "adcbq_pool_top10_target99_per_ef_t1.csv"
TQ_RAW_ROOT = RESULT_ROOT / "tq_raw"
TMP_ROOT = Path("/tmp/nq_qwen_multidim_calibrated_adcbq_tq_t1")
EXISTING_PANEL = (
    REPO / "experiment/figures/nq_quora_search_t1_odirect/all_methods_results.csv"
)
FIGURE_ROOT = REPO / "experiment/figures/nq_qwen_multidim_calibrated_adcbq_tq_t1"

DATASETS = (
    "nq_qwen06_1024",
    "nq_qwen4_2560",
    "nq_qwen8_4096",
)
EFS = (64, 128, 256, 512, 1024)
TQ_RERANKS = (10, 20, 30, 40, 60, 80, 120, 160, 200)
TARGET_RECALL = 0.99
CALIBRATION_ROWS = 1000
SEED = 11
PAGE_BYTES = 4096
BASELINE_METHODS = {"diskann", "spann", "rabitq_b1", "rabitq_b7", "bbq"}
EXISTING_ALIGNED_FP_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def configure_modules() -> None:
    for key in DATASETS:
        matrix.CALIBRATION_ROWS[key] = CALIBRATION_ROWS
    tq.RESULT_ROOT = RESULT_ROOT / "tq_work"
    tq.EXTERNAL_ROOT = TMP_ROOT
    tq.INDEX_ROOT = TMP_ROOT / "indexes"
    tq.SIDECAR_ROOT = TMP_ROOT / "sidecars"
    tq.SUMMARY = RESULT_ROOT / "tq_work" / "unused_summary.csv"
    tq.STATUS = RESULT_ROOT / "tq_work" / "status.json"
    tq.LOCK = RESULT_ROOT / "tq_work" / "queue.lock"
    # /tmp is intentionally used for the one transient 2560-D padded sidecar.
    tq.EXTERNAL_FLOOR_GIB = 8.0
    tq.ROOT_FLOOR_GIB = 8.0
    adc.RESULT_DIR = ADC_RAW_ROOT
    adc.BIN = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_same_pool_calibration"


def split_for(dataset_key: str) -> adc.SplitFiles:
    dataset = matrix.registry.DATASETS[dataset_key]
    return adc.make_split(dataset, CALIBRATION_ROWS, SEED, 0)


def aligned_fp_path(dataset_key: str) -> Path:
    path = EXISTING_ALIGNED_FP_ROOT / f"{dataset_key}.fp32.header4096.fbin"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def adc_calibration_path(dataset_key: str, ef: int) -> Path:
    return ADC_CALIBRATION_ROOT / dataset_key / f"pool_top10_target99.ef{ef}.json"


def adc_candidate_paths(dataset_key: str, ef: int) -> tuple[Path, Path]:
    root = ADC_CALIBRATION_ROOT / dataset_key / "candidates"
    return root / f"ef{ef}.summary.csv", root / f"ef{ef}.csv"


def fit_adcbq_per_ef(
    dataset: matrix.registry.Dataset,
    split: adc.SplitFiles,
    ef: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    path = adc_calibration_path(dataset.key, ef)
    if path.is_file() and not args.force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol") == ADC_PROTOCOL
            and payload.get("status") == "feasible"
            and int(payload.get("ef", -1)) == ef
            and int(payload.get("calibration_rows", -1)) == split.calibration_rows
            and float(payload.get("target_pool_top10_coverage", -1.0))
            == TARGET_RECALL
        ):
            print(
                f"REUSE ADC_CALIBRATION {dataset.key} ef={ef} "
                f"tau={float(payload['tau']):.9g}",
                flush=True,
            )
            return payload

    summary_path, candidate_path = adc_candidate_paths(dataset.key, ef)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    command = adc.common_command(
        dataset,
        split.calibration_query,
        split.calibration_gt,
        summary_path,
        ef,
        1,
        1,
        args.build_threads,
    )
    command += [
        "--rerank-n", str(ef),
        "--candidate-only",
        "--no-rerank",
        "--candidates-out", str(candidate_path),
    ]
    adc.run(command, 1)

    adc_pool.TARGET_COVERAGE = TARGET_RECALL
    fitted = adc_pool.fit_pool_top10_conformal(
        SimpleNamespace(dataset=dataset), split, candidate_path, ef
    )
    payload: dict[str, object] = {
        "protocol": ADC_PROTOCOL,
        "status": "feasible",
        "target_pool_top10_coverage": TARGET_RECALL,
        "ef": ef,
        "calibration_rows": split.calibration_rows,
        **fitted,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    if not args.keep_candidates:
        candidate_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    print(
        f"ADC_CALIBRATED {dataset.key} ef={ef} "
        f"tau={float(payload['tau']):.9g} "
        f"coverage={float(payload['empirical_all_top10_coverage']):.4f} "
        f"Rmean={float(payload['rerank_count_mean']):.2f}",
        flush=True,
    )
    return payload


def affinity_cpus(cpu: int, threads: int) -> tuple[int, ...]:
    allowed = sorted(os.sched_getaffinity(0))
    if cpu not in allowed:
        raise RuntimeError(f"CPU {cpu} is outside allowed affinity {allowed}")
    start = allowed.index(cpu)
    ordered = allowed[start:] + allowed[:start]
    if threads > len(ordered):
        raise RuntimeError(
            f"requested {threads} search threads but only {len(ordered)} CPUs are allowed"
        )
    return tuple(ordered[:threads])


def pinned_adc_run(cpu: int):
    def run(command: list[str], search_threads: int) -> None:
        cpus = affinity_cpus(cpu, search_threads)
        pinned = ["taskset", "-c", ",".join(map(str, cpus)), *command]
        print("RUN " + " ".join(pinned), flush=True)
        env = {
            **os.environ,
            "OMP_NUM_THREADS": str(search_threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        subprocess.run(pinned, cwd=REPO, env=env, check=True)

    return run


def run_adcbq(args: argparse.Namespace) -> list[dict[str, object]]:
    adc.ensure_binary()
    original_run = adc.run
    adc.run = pinned_adc_run(args.cpu)
    rows: list[dict[str, object]] = []
    try:
        for dataset_key in DATASETS:
            dataset = matrix.registry.DATASETS[dataset_key]
            matrix.registry.validate(dataset)
            split = split_for(dataset_key)
            sidecar = aligned_fp_path(dataset_key)
            for ef in EFS:
                fitted = fit_adcbq_per_ef(dataset, split, ef, args)
                tau = float(fitted["tau"])
                run_args = SimpleNamespace(
                    force=args.force,
                    fixed_r=ef,
                    rounds=args.rounds,
                    search_threads=args.search_threads,
                    build_threads=args.build_threads,
                )
                raw = adc.benchmark_policy(
                    dataset,
                    split,
                    ef,
                    "calibrated",
                    tau,
                    sidecar,
                    run_args,
                    f".pool_top10_target99_per_ef.t{args.search_threads}",
                )
                summary = adc_panel.summarize(
                    dataset,
                    split,
                    ef=ef,
                    threads=args.search_threads,
                    rounds=args.rounds,
                    tau=tau,
                    result_path=raw,
                )
                row = {
                    "dataset": dataset_key,
                    "corpus": dataset.corpus,
                    "model": dataset.model,
                    "method_key": "adc_bq_calibrated",
                    "method": "ADC-BQ per-ef pool-top10@0.99",
                    "search_parameter": "efSearch",
                    "search_parameter_value": ef,
                    "recall_at_10": summary["recall_at_10"],
                    "qps": summary["qps"],
                    "qps_stddev": "",
                    "rounds": args.rounds,
                    "M": 24,
                    "efConstruction_or_Lbuild": 200,
                    "search_threads": args.search_threads,
                    "fp_vectors_per_query": summary["rerank_vectors_per_query_mean"],
                    "fp_4k_pages_per_query": summary[
                        "fp_submitted_4k_pages_per_query_mean"
                    ],
                    "storage": summary["storage"],
                    "io_engine": summary["io_engine"],
                    "query_scope": summary["query_scope"],
                    "calibration_queries": CALIBRATION_ROWS,
                    "test_queries": split.test_rows,
                    "calibration_protocol": ADC_PROTOCOL,
                    "target_pool_top10_coverage": TARGET_RECALL,
                    "empirical_pool_top10_coverage": fitted[
                        "empirical_all_top10_coverage"
                    ],
                    "calibration_rerank_count_mean": fitted["rerank_count_mean"],
                    "tau": tau,
                    "calibration_reference_ef": ef,
                    "source_csv": str(raw.relative_to(REPO)),
                }
                rows.append(row)
                write_csv(ADC_SUMMARY, rows)
                print(
                    f"ADC_RESULT {dataset_key} ef={ef} "
                    f"recall={float(row['recall_at_10']):.6f} "
                    f"qps={float(row['qps']):.2f} "
                    f"reads={float(row['fp_vectors_per_query']):.2f}",
                    flush=True,
                )
    finally:
        adc.run = original_run
    return rows


def tq_method(rerank_n: int) -> tq.Method:
    return tq.Method("qdranttq1", f"TurboQuant TQ1+ + FP@{rerank_n}", rerank_n)


def tq_codec_ready(dataset_key: str) -> tuple[Path, Path]:
    return tq.codec_paths(dataset_key, "qdranttq1")


def tq_sidecar(dataset_key: str) -> tuple[Path, bool]:
    existing = aligned_fp_path(dataset_key)
    if tq.valid_sidecar(existing, dataset_key):
        print(f"REUSE TQ FP sidecar {existing}", flush=True)
        return existing, False
    path = tq.build_sidecar(dataset_key)
    return path, True


def tq_raw_complete(
    path: Path,
    dataset_key: str,
    rerank_n: int,
    efs: tuple[int, ...],
    rounds: int,
    search_threads: int,
) -> bool:
    rows = read_csv(path)
    expected = {(ef, round_id) for ef in efs for round_id in range(1, rounds + 1)}
    try:
        actual = {(int(row["efSearch"]), int(row["round"])) for row in rows}
        return actual == expected and all(
            row["dataset"] == dataset_key
            and int(row["threads"]) == search_threads
            and int(row["rerank_n"]) == rerank_n
            and row["code_storage"] == "resident"
            and row["fp_io_mode"] == "uring-batch"
            and int(row["fp_rerank_qd"]) >= rerank_n
            for row in rows
        )
    except (KeyError, ValueError):
        return False


def run_tq_curve(
    dataset_key: str,
    split: adc.SplitFiles,
    *,
    rerank_n: int,
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
    output = TQ_RAW_ROOT / dataset_key / f"{phase}.r{rerank_n}.t{search_threads}.csv"
    if not force and tq_raw_complete(
        output, dataset_key, rerank_n, EFS, rounds, search_threads
    ):
        print(f"REUSE {output}", flush=True)
        return output
    dataset = matrix.registry.DATASETS[dataset_key]
    codec, meta = tq_codec_ready(dataset_key)
    for required in (query, gt, dataset.adcbq_index, codec, meta, sidecar):
        if not required.is_file():
            raise FileNotFoundError(required)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset", "-c", ",".join(map(str, affinity_cpus(cpu, search_threads))),
        str(tq.search_runner.BINARY),
        "--dataset", dataset_key,
        "--method", "qdranttq1",
        "--metric", "ip",
        "--query", str(query),
        "--gt", str(gt),
        "--graph", str(dataset.adcbq_index),
        "--codec-index", str(codec),
        "--meta", str(meta),
        "--sidecar", str(TMP_ROOT / "unused_resident_code_sidecar"),
        "--fp-sidecar", str(sidecar),
        "--out", str(output),
        "--max-queries", str(nq),
        "--rounds", str(rounds),
        "--threads", str(search_threads),
        "--rerank-n", str(rerank_n),
        "--resident-codes",
        "--fp-io-mode", "uring-batch",
        "--fp-rerank-qd", str(rerank_n),
        "--efs", *[str(ef) for ef in EFS],
    ]
    print("RUN " + " ".join(command), flush=True)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(search_threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=REPO, env=env, check=True)
    if not tq_raw_complete(
        output, dataset_key, rerank_n, EFS, rounds, search_threads
    ):
        raise RuntimeError(f"incomplete TQ result: {output}")
    return output


def summarize_tq(
    path: Path,
    dataset_key: str,
    rerank_n: int,
    *,
    phase: str,
    query_scope: str,
    search_threads: int,
) -> list[dict[str, object]]:
    dataset = matrix.registry.DATASETS[dataset_key]
    groups: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        groups.setdefault(int(row["efSearch"]), []).append(row)
    output: list[dict[str, object]] = []
    for ef in EFS:
        raw = groups[ef]
        qps_values = [float(row["qps"]) for row in raw]
        recalls = [float(row["recall_at_10"]) for row in raw]
        pages = [float(row["fp_read_bytes_per_query"]) / PAGE_BYTES for row in raw]
        output.append(
            {
                "dataset": dataset_key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method_key": "turboquant_tq1",
                "method": f"TurboQuant TQ1+ + FP@{rerank_n}",
                "search_parameter": "efSearch",
                "search_parameter_value": ef,
                "efSearch": ef,
                "rerank_n": rerank_n,
                "recall_at_10": statistics.median(recalls),
                "qps": statistics.median(qps_values),
                "qps_stddev": statistics.pstdev(qps_values),
                "rounds": len(raw),
                "M": 24,
                "efConstruction_or_Lbuild": 200,
                "search_threads": search_threads,
                "fp_vectors_per_query": rerank_n,
                "fp_4k_pages_per_query": statistics.median(pages),
                "storage": "resident TQ1+ codes + SSD FP32 sidecar",
                "io_engine": "O_DIRECT io_uring one fixed-R batch per query",
                "query_scope": query_scope,
                "phase": phase,
                "source_csv": str(path.relative_to(REPO)),
            }
        )
    return output


def pareto(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best_at_recall: dict[float, dict[str, object]] = {}
    for row in rows:
        recall = float(row["recall_at_10"])
        if recall not in best_at_recall or float(row["qps"]) > float(
            best_at_recall[recall]["qps"]
        ):
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


def choose_r(dataset_key: str, rows: list[dict[str, object]]) -> dict[str, object]:
    frontier = pareto(rows)
    feasible = [row for row in frontier if float(row["recall_at_10"]) >= TARGET_RECALL]
    if feasible:
        chosen = max(feasible, key=lambda row: float(row["qps"]))
        reached = True
    else:
        chosen = max(
            frontier,
            key=lambda row: (float(row["recall_at_10"]), float(row["qps"])),
        )
        reached = False
    return {
        "dataset": dataset_key,
        "selected_r": int(chosen["rerank_n"]),
        "selection_ef": int(chosen["efSearch"]),
        "calibration_recall_at_10": float(chosen["recall_at_10"]),
        "calibration_qps": float(chosen["qps"]),
        "target_recall": TARGET_RECALL,
        "target_reached": reached,
        "selection_objective": "max_qps_on_calibration_joint_ef_r_pareto_at_target",
    }


def run_turboquant(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tq.codec_builder.compile_binary()
    tq.search_runner.compile_binary()
    all_pilot: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    build_args = SimpleNamespace(
        build_threads=args.build_threads,
        train_size=args.tq_train_size,
        opq_niter=1,
        opq_blas_threads=1,
    )
    for dataset_key in DATASETS:
        dataset = matrix.registry.DATASETS[dataset_key]
        matrix.registry.validate(dataset)
        split = split_for(dataset_key)
        codec_arm = tq_method(TQ_RERANKS[0])
        tq.build_codec(dataset_key, codec_arm, build_args)
        sidecar, transient_sidecar = tq_sidecar(dataset_key)
        try:
            pilot_rows: list[dict[str, object]] = []
            for rerank_n in TQ_RERANKS:
                raw = run_tq_curve(
                    dataset_key,
                    split,
                    rerank_n=rerank_n,
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
                pilot_rows.extend(
                    summarize_tq(
                        raw,
                        dataset_key,
                        rerank_n,
                        phase="calibration_pareto",
                        query_scope="1000-query calibration split",
                        search_threads=args.search_threads,
                    )
                )
                all_pilot.extend(pilot_rows[-len(EFS):])
                write_csv(
                    RESULT_ROOT / f"tq_calibration_r_sweep_t{args.search_threads}.csv",
                    all_pilot,
                )
            selection = choose_r(dataset_key, pilot_rows)
            selections.append(selection)
            write_csv(RESULT_ROOT / "tq_selected_r.csv", selections)
            selected_r = int(selection["selected_r"])
            print(
                f"TQ_SELECTED {dataset_key} R={selected_r} "
                f"ef={selection['selection_ef']} "
                f"recall={float(selection['calibration_recall_at_10']):.6f} "
                f"qps={float(selection['calibration_qps']):.2f}",
                flush=True,
            )
            raw = run_tq_curve(
                dataset_key,
                split,
                rerank_n=selected_r,
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
            selected_curve = summarize_tq(
                raw,
                dataset_key,
                selected_r,
                phase="heldout_selected",
                query_scope="held-out test queries after 1000-query R selection split",
                search_threads=args.search_threads,
            )
            for row in selected_curve:
                row["calibration_queries"] = split.calibration_rows
                row["test_queries"] = split.test_rows
                row["selected_r"] = selected_r
            final_rows.extend(selected_curve)
            write_csv(
                RESULT_ROOT / f"tq_selected_r_heldout_t{args.search_threads}.csv",
                final_rows,
            )
        finally:
            if transient_sidecar and sidecar.exists():
                size = sidecar.stat().st_size
                sidecar.unlink()
                print(f"REMOVED transient TQ sidecar bytes={size}: {sidecar}", flush=True)
            tq.reclaim_codec(dataset_key, codec_arm)
    return final_rows, selections


def build_overlay(
    adcbq_rows: list[dict[str, object]] | None = None,
    tq_rows: list[dict[str, object]] | None = None,
) -> Path:
    if adcbq_rows is None:
        adcbq_rows = [dict(row) for row in read_csv(ADC_SUMMARY)]
    if tq_rows is None:
        tq_rows = [dict(row) for row in read_csv(RESULT_ROOT / "tq_selected_r_heldout_t1.csv")]
    baselines = [
        dict(row)
        for row in read_csv(EXISTING_PANEL)
        if row.get("dataset") in DATASETS and row.get("method_key") in BASELINE_METHODS
    ]
    if not baselines:
        raise RuntimeError(f"no reusable NQ Qwen baselines in {EXISTING_PANEL}")
    rows: list[dict[str, object]] = [*adcbq_rows, *tq_rows, *baselines]
    dataset_order = {key: index for index, key in enumerate(DATASETS)}
    method_order = {
        "adc_bq_calibrated": 0,
        "turboquant_tq1": 1,
        "rabitq_b1": 2,
        "rabitq_b7": 3,
        "diskann": 4,
        "spann": 5,
        "bbq": 6,
    }
    rows.sort(
        key=lambda row: (
            dataset_order[str(row["dataset"])],
            method_order.get(str(row["method_key"]), 99),
            float(row["recall_at_10"]),
        )
    )
    out = FIGURE_ROOT / "nq_qwen_3dims_calibrated_adcbq_tq_7methods_t1.csv"
    write_csv(out, rows)
    return out


def plot_overlay(csv_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("SKIP plot: matplotlib is not installed in this Python", flush=True)
        return
    rows = read_csv(csv_path)
    styles = {
        "adc_bq_calibrated": (
            "#1f77b4", "o", "-", "ADC-BQ per-ef pool-top10@0.99"
        ),
        "turboquant_tq1": ("#8c564b", "D", "-", "TurboQuant TQ1+"),
        "rabitq_b1": ("#9467bd", "X", "--", "RaBitQ 1-bit + adaptive FP"),
        "rabitq_b7": ("#009E73", "o", "-", "RaBitQ 7-bit RAM"),
        "diskann": ("#2ca02c", "s", "--", "DiskANN"),
        "spann": ("#56B4E9", "*", "-", "SPANN"),
        "bbq": ("#333333", "x", ":", "Elasticsearch BBQ"),
    }
    titles = {
        "nq_qwen06_1024": "NQ · Qwen-0.6B (1024D)",
        "nq_qwen4_2560": "NQ · Qwen-4B (2560D)",
        "nq_qwen8_4096": "NQ · Qwen-8B (4096D)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharey=False)
    for axis, dataset_key in zip(axes, DATASETS):
        for method_key, (color, marker, linestyle, label) in styles.items():
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset_key and row["method_key"] == method_key
            ]
            if not selected:
                continue
            selected.sort(key=lambda row: float(row["recall_at_10"]))
            axis.plot(
                [float(row["recall_at_10"]) for row in selected],
                [float(row["qps"]) for row in selected],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                label=label,
            )
        axis.axvline(0.99, color="#777777", linestyle=":", linewidth=1.1)
        axis.set_title(titles[dataset_key])
        axis.set_xlabel("Held-out Recall@10")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("QPS (T=1, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "NQ Qwen dimensions: calibrated ADC-BQ and per-dataset Pareto-selected TurboQuant",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    png = FIGURE_ROOT / "nq_qwen_3dims_calibrated_adcbq_tq_7methods_t1.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {png}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pilot-rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--tq-train-size", type=int, default=100000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("adcbq", "turboquant", "overlay"),
        default=("adcbq", "turboquant", "overlay"),
    )
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_modules()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        overlay = build_overlay()
        plot_overlay(overlay)
        return 0
    if min(
        args.cpu,
        args.rounds,
        args.pilot_rounds,
        args.build_threads,
        args.search_threads,
        args.tq_train_size,
    ) <= 0:
        raise SystemExit("all numeric arguments must be positive")
    adcbq_rows = run_adcbq(args) if "adcbq" in args.parts else None
    tq_rows = None
    if "turboquant" in args.parts:
        tq_rows, _ = run_turboquant(args)
    if "overlay" in args.parts:
        overlay = build_overlay(adcbq_rows, tq_rows)
        plot_overlay(overlay)
        print(f"COMPLETE overlay={overlay}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
