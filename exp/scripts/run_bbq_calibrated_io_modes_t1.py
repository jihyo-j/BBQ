#!/usr/bin/env python3
"""Compare three exact-FP I/O paths for calibrated ADC-BQ at strict T=1.

The ADC graph, held-out split, efSearch, per-ef split-conformal threshold, and
selected candidates are identical across the three measurements.  Only the
exact FP32 candidate-fetch path changes:

* sequential: one synchronous O_DIRECT pread per selected vector;
* async_batch: one per-query io_uring batch, without inter-query overlap;
* async_batch_overlap: the same batch I/O overlapped with the next query's
  resident ADC-BQ graph traversal.

This runner is restartable.  It covers the existing MSMARCO Qwen-1024 data and
the latest MSMARCO OpenAI text-embedding-3-small (1536-D) data prepared for the
main OpenAI/BEIR campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiment"))

import run_four_methods_series_t1 as four  # noqa: E402
import run_nq_qwen_multidim_calibrated_adcbq_tq_t1 as calibrated  # noqa: E402
import run_v051_ssd_odirect_multidataset as registry  # noqa: E402


RESULT_ROOT = REPO / "experiment/bbq_calibrated_io_modes_t1"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/openai_beir_11methods_t1")
MANIFEST = EXTERNAL_ROOT / "dataset_manifest.json"
OPENAI_KEY = "sieun_msmarco_openai1536"
QWEN_KEY = "msmarco_qwen1024"
QWEN_SIDECAR = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/aligned_fp/"
    "msmarco_qwen1024.fp32.header4096.fbin"
)
EFS = (
    64, 72, 80, 88, 96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 320, 512, 1024,
)
CALIBRATION_ROWS = 1000
TARGET_RECALL = 0.99
SEED = 11
PAGE_BYTES = 4096


@dataclass(frozen=True)
class IoMode:
    key: str
    label: str
    source: str
    overlap_queries: bool
    description: str


MODES = (
    IoMode(
        "sequential",
        "BBQ-Sequential",
        "odirect",
        False,
        "one synchronous O_DIRECT FP32 candidate read at a time",
    ),
    IoMode(
        "async_batch",
        "BBQ-Async Batch",
        "odirect-uring",
        False,
        "one io_uring FP32 candidate batch per query; no query overlap",
    ),
    IoMode(
        "async_batch_overlap",
        "BBQ-Async Batch + Overlap",
        "odirect-uring",
        True,
        "io_uring FP32 batches overlapped with the next query traversal",
    ),
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def openai_spec() -> dict[str, object]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"OpenAI input manifest is missing; run prepare_openai_beir_11methods_inputs.py: {MANIFEST}"
        )
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    matches = [item for item in payload["datasets"] if item["key"] == OPENAI_KEY]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {OPENAI_KEY} entry in {MANIFEST}, found {len(matches)}")
    return dict(matches[0])


def datasets(selected_keys: set[str]) -> list[tuple[registry.Dataset, Path, int]]:
    configured: list[tuple[registry.Dataset, Path, int]] = []
    if QWEN_KEY in selected_keys:
        qwen = registry.DATASETS[QWEN_KEY]
        registry.validate(qwen)
        if not QWEN_SIDECAR.is_file():
            raise FileNotFoundError(QWEN_SIDECAR)
        configured.append((qwen, QWEN_SIDECAR, CALIBRATION_ROWS))
    if OPENAI_KEY in selected_keys:
        spec = openai_spec()
        openai = four.register(spec)
        configured.append(
            (openai, Path(str(spec["fp_sidecar"])), int(spec["calibration_rows"]))
        )
    return configured


def raw_path(dataset_key: str, mode: IoMode, ef: int) -> Path:
    return RESULT_ROOT / dataset_key / "raw" / mode.key / f"ef{ef}.csv"


def raw_complete(
    path: Path,
    *,
    dataset_key: str,
    mode: IoMode,
    ef: int,
    rounds: int,
    test_rows: int,
) -> bool:
    rows = read_csv(path)
    if len(rows) != 1:
        return False
    try:
        row = rows[0]
        return (
            row["dataset"] == dataset_key
            and int(row["efSearch"]) == ef
            and int(row["rounds"]) == rounds
            and int(row["search_threads"]) == 1
            and int(row["nq"]) == test_rows
            and row["fp_rerank_source"] == mode.source
            and (row["fp_rerank_overlap_queries"] == "true")
            == mode.overlap_queries
            and row["fp_rerank_norms"] == "onfetch"
            and row["rerank_dot"] == "fp32"
            and row["calibrated_threshold_rerank_enabled"] == "true"
        )
    except (KeyError, ValueError):
        return False


def run_mode(
    dataset: registry.Dataset,
    split: calibrated.adc.SplitFiles,
    sidecar: Path,
    mode: IoMode,
    ef: int,
    tau: float,
    args: argparse.Namespace,
) -> Path:
    output = raw_path(dataset.key, mode, ef)
    if not args.force and raw_complete(
        output,
        dataset_key=dataset.key,
        mode=mode,
        ef=ef,
        rounds=args.rounds,
        test_rows=split.test_rows,
    ):
        print(f"REUSE {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = calibrated.adc.common_command(
        dataset,
        split.test_query,
        split.test_gt,
        output,
        ef,
        args.rounds,
        1,
        args.build_threads,
    )
    command += [
        "--rerank-n", str(ef),
        "--rerank-dot", "fp32",
        "--fp-rerank-source", mode.source,
        "--fp-rerank-norms", "onfetch",
        "--fp-rerank-path", str(sidecar),
        "--fp-rerank-qd", str(ef),
        "--calibrated-threshold-rerank-tau", repr(tau),
    ]
    if mode.overlap_queries:
        command.append("--fp-rerank-overlap-queries")
    pinned = ["taskset", "-c", str(args.cpu), *command]
    print("RUN " + " ".join(pinned), flush=True)
    subprocess.run(
        pinned,
        cwd=REPO,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=True,
    )
    if not raw_complete(
        output,
        dataset_key=dataset.key,
        mode=mode,
        ef=ef,
        rounds=args.rounds,
        test_rows=split.test_rows,
    ):
        raise RuntimeError(f"invalid or incomplete raw output: {output}")
    return output


def number(raw: dict[str, str], key: str, default: float = 0.0) -> float:
    value = raw.get(key, "")
    return default if value == "" else float(value)


def summarize(
    dataset: registry.Dataset,
    split: calibrated.adc.SplitFiles,
    sidecar: Path,
    mode: IoMode,
    ef: int,
    fitted: dict[str, object],
    output: Path,
    rounds: int,
    order: int,
) -> dict[str, object]:
    raw = read_csv(output)[0]
    nq = int(raw["nq"])
    dim = int(raw["dim"])
    rerank_mean = float(raw["adaptive_rerank_count_mean"])
    row_stride = math.ceil(dim * 4 / PAGE_BYTES) * PAGE_BYTES
    derived_rows = rerank_mean
    derived_bytes = derived_rows * row_stride
    reported_rows = number(raw, "direct_rows_mean") / nq
    reported_bytes = number(raw, "direct_submitted_bytes_mean") / nq
    if mode.source == "odirect-uring":
        if not math.isclose(reported_rows, derived_rows, rel_tol=2e-5, abs_tol=2e-4):
            raise RuntimeError(
                f"{dataset.key}/{mode.key}/ef{ef}: direct rows/q={reported_rows} "
                f"!= rerank/q={derived_rows}"
            )
        if not math.isclose(reported_bytes, derived_bytes, rel_tol=2e-5, abs_tol=1.0):
            raise RuntimeError(
                f"{dataset.key}/{mode.key}/ef{ef}: bytes/q={reported_bytes} "
                f"!= derived={derived_bytes}"
            )
    return {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "dim": dim,
        "nb": int(raw["nb"]),
        "method_key": mode.key,
        "method": mode.label,
        "io_description": mode.description,
        "efSearch": ef,
        "tau": float(fitted["tau"]),
        "target_pool_top10_coverage": TARGET_RECALL,
        "empirical_pool_top10_coverage": fitted["empirical_all_top10_coverage"],
        "calibration_protocol": calibrated.ADC_PROTOCOL,
        "calibration_queries": split.calibration_rows,
        "test_queries": split.test_rows,
        "seed": SEED,
        "search_threads": 1,
        "rounds": rounds,
        "measurement_order_within_ef": order,
        "recall_at_10": float(raw["recall_at_k"]),
        "qps": float(raw["qps"]),
        "qps_median": number(raw, "qps_median"),
        "search_seconds_mean": number(raw, "search_seconds_mean"),
        "rerank_seconds_mean": number(raw, "rerank_seconds_mean"),
        "total_seconds_mean": number(raw, "total_seconds_mean"),
        "rerank_vectors_per_query_mean": rerank_mean,
        "rerank_vectors_per_query_min": float(raw["adaptive_rerank_count_min"]),
        "rerank_vectors_per_query_p95": float(raw["adaptive_rerank_count_p95"]),
        "rerank_vectors_per_query_p99": float(raw["adaptive_rerank_count_p99"]),
        "rerank_vectors_per_query_max": float(raw["adaptive_rerank_count_max"]),
        "fp_row_stride_bytes": row_stride,
        "fp_logical_rows_per_query": derived_rows,
        "fp_submitted_bytes_per_query": derived_bytes,
        "fp_submitted_4k_pages_per_query": derived_bytes / PAGE_BYTES,
        "kernel_read_bytes_per_query": number(raw, "rerank_read_bytes_mean") / nq,
        "fp_io_operations_per_query": (
            derived_rows
            if mode.source == "odirect"
            else number(raw, "direct_requests_mean") / nq
        ),
        "fp_io_submission_style": (
            "one blocking pread per candidate"
            if mode.source == "odirect"
            else "one io_uring batch per query"
        ),
        "fp_rerank_source": mode.source,
        "fp_rerank_overlap_queries": mode.overlap_queries,
        "fp_rerank_qd": int(raw["fp_rerank_qd"]),
        "intra_query_io_wait_us_per_query": (
            number(raw, "intra_query_wait_seconds_sum_mean") * 1e6 / nq
        ),
        "pipeline_queries_overlapped_per_round": number(
            raw, "pipeline_overlapped_queries_mean"
        ),
        "pipeline_io_operations_overlapped_per_round": number(
            raw, "pipeline_overlapped_io_requests_mean"
        ),
        "pipeline_ready_ratio": number(raw, "pipeline_ready_ratio"),
        "pipeline_wait_us_per_query": (
            number(raw, "pipeline_wait_seconds_sum_mean") * 1e6 / nq
        ),
        "fp_sidecar": str(sidecar),
        "raw_result_csv": str(output.relative_to(REPO)),
    }


def validate_same_work(rows: list[dict[str, object]], dataset_key: str, ef: int) -> None:
    selected = [
        row for row in rows
        if row["dataset"] == dataset_key and int(row["efSearch"]) == ef
    ]
    if len(selected) != len(MODES):
        raise RuntimeError(f"{dataset_key}/ef{ef}: expected {len(MODES)} modes")
    exact_fields = (
        "recall_at_10",
        "rerank_vectors_per_query_mean",
        "rerank_vectors_per_query_min",
        "rerank_vectors_per_query_p95",
        "rerank_vectors_per_query_p99",
        "rerank_vectors_per_query_max",
        "fp_logical_rows_per_query",
        "fp_submitted_bytes_per_query",
    )
    for field in exact_fields:
        values = [float(row[field]) for row in selected]
        if max(values) - min(values) > 1e-9:
            raise RuntimeError(
                f"{dataset_key}/ef{ef}: modes did different logical work for "
                f"{field}: {values}"
            )


def run_dataset(
    dataset: registry.Dataset,
    sidecar: Path,
    calibration_rows: int,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    result_root = RESULT_ROOT / dataset.key
    four.configure_calibrated(
        dataset,
        sidecar,
        result_root / "calibration_work",
        EXTERNAL_ROOT / "bbq_io_ablation" / dataset.key,
        EFS,
        calibration_rows,
        1,
    )
    split = calibrated.split_for(dataset.key)
    calibrated.adc.ensure_binary()
    original_run = calibrated.adc.run
    calibrated.adc.run = calibrated.pinned_adc_run(args.cpu)
    rows: list[dict[str, object]] = []
    try:
        for ef_index, ef in enumerate(EFS):
            fitted = calibrated.fit_adcbq_per_ef(dataset, split, ef, args)
            tau = float(fitted["tau"])
            rotated = MODES[ef_index % len(MODES):] + MODES[:ef_index % len(MODES)]
            for order, mode in enumerate(rotated, start=1):
                output = run_mode(dataset, split, sidecar, mode, ef, tau, args)
                rows.append(
                    summarize(
                        dataset,
                        split,
                        sidecar,
                        mode,
                        ef,
                        fitted,
                        output,
                        args.rounds,
                        order,
                    )
                )
            validate_same_work(rows, dataset.key, ef)
            rows.sort(key=lambda row: (int(row["efSearch"]), str(row["method_key"])))
            write_csv(result_root / "summary_t1.csv", rows)
            print(f"CHECKPOINT {dataset.key} ef={ef} modes=3", flush=True)
    finally:
        calibrated.adc.run = original_run
    return rows


def main() -> int:
    global RESULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=(QWEN_KEY, OPENAI_KEY),
        default=[QWEN_KEY, OPENAI_KEY],
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.build_threads, args.cpu) < 1:
        parser.error("numeric arguments must be positive")
    args.pilot_rounds = 1
    args.tq_train_size = 100_000
    RESULT_ROOT = args.result_root.resolve()
    selected_keys = tuple(dict.fromkeys(args.datasets))

    status = RESULT_ROOT / "campaign_status.json"
    write_json(
        status,
        {
            "stage": "running",
            "updated_at": now(),
            "datasets": list(selected_keys),
            "threads": 1,
            "rounds": args.rounds,
            "efs": list(EFS),
            "modes": [mode.key for mode in MODES],
        },
    )
    all_rows: list[dict[str, object]] = []
    for dataset, sidecar, calibration_rows in datasets(set(selected_keys)):
        all_rows.extend(run_dataset(dataset, sidecar, calibration_rows, args))
    all_rows.sort(
        key=lambda row: (
            selected_keys.index(str(row["dataset"])),
            int(row["efSearch"]),
            tuple(mode.key for mode in MODES).index(str(row["method_key"])),
        )
    )
    expected = len(selected_keys) * len(EFS) * len(MODES)
    if len(all_rows) != expected:
        raise RuntimeError(f"expected {expected} summary rows, found {len(all_rows)}")
    combined = RESULT_ROOT / "summary_t1.csv"
    write_csv(combined, all_rows)
    write_json(
        status,
        {
            "stage": "complete",
            "updated_at": now(),
            "datasets": list(selected_keys),
            "threads": 1,
            "rounds": args.rounds,
            "efs": list(EFS),
            "modes": [mode.key for mode in MODES],
            "rows": len(all_rows),
            "summary": str(combined),
        },
    )
    print(f"COMPLETE {combined}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
