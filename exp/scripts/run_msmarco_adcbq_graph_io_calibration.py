#!/usr/bin/env python3
"""Tune ADC-BQ graph-page I/O on calibration queries only.

The selected setting is global across the four RAM budgets.  The held-out
test queries are never used for this choice.  Candidate settings include the
24-thread synchronous pread path and query-internal io_uring with a real
per-query in-flight cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

import run_adcbq_calibrated_threshold_overlap as calibration
import run_msmarco_adcbq_calibrated_rabitq_memory_page_sweep as calibrated
import run_msmarco_adcbq_rabitq_memory_page_sweep as base


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/msmarco_adcbq_graph_io_calibration"
FP_PATH = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_page4k_ef256_qps_rerun/"
    "msmarco_openai1536/msmarco_openai1536.fp32.aligned4k.fbin"
)
PAGE = 4096
EF = 256
THREADS = 24
TRIALS = 2
SETTINGS: tuple[tuple[str, int], ...] = (
    ("sync", 0),
    ("async", 1),
    ("async", 2),
    ("async", 4),
    ("async", 8),
    ("async", 16),
    ("async", 32),
    ("async", 48),
)


def configure_page(page_bytes: int) -> None:
    global PAGE, RESULT_ROOT
    if page_bytes not in base.PAGES:
        raise ValueError(f"unsupported page size: {page_bytes}")
    PAGE = page_bytes
    if PAGE == 4096:
        RESULT_ROOT = REPO / "experiment/msmarco_adcbq_graph_io_calibration"
    else:
        RESULT_ROOT = REPO / (
            f"experiment/msmarco_adcbq_graph_io_calibration_{PAGE // 1024}k"
        )


def ensure_binary() -> None:
    binary = calibrated.ADC_BINARIES[PAGE]
    source = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
    if binary.exists() and binary.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return
    env = os.environ.copy()
    env["PAGE_BYTES"] = str(PAGE)
    env["OUT"] = str(binary)
    subprocess.run(
        [str(REPO / "experiment/adcbq_optimized/build.sh")],
        cwd=REPO,
        env=env,
        check=True,
    )


def setting_name(engine: str, qd: int) -> str:
    return "sync" if engine == "sync" else f"async_qd{qd}"


def read_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one result row: {path}")
    return rows[0]


def output_path(engine: str, qd: int, ratio: float, trial: int) -> Path:
    return RESULT_ROOT / "raw" / (
        f"{setting_name(engine, qd)}_{base.ratio_label(ratio)}_trial{trial}.csv"
    )


def complete(path: Path, engine: str, qd: int, ratio: float) -> bool:
    try:
        row = read_row(path)
        return (
            int(row["nq"]) == calibrated.CALIBRATION_QUERIES
            and int(row["efSearch"]) == EF
            and int(row["search_threads"]) == THREADS
            and int(row["index_page_cache_mib"]) == base.budget_mib(ratio)
            and row["index_page_async"] == ("true" if engine == "async" else "false")
            and (engine == "sync" or int(row["index_page_qd"]) == qd)
            and row["calibrated_threshold_rerank_enabled"] == "true"
            and row["fp_rerank_source"] == "odirect-uring"
        )
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def run_one(
    dataset: object,
    tau: float,
    engine: str,
    qd: int,
    ratio: float,
    trial: int,
) -> Path:
    output = output_path(engine, qd, ratio, trial)
    if complete(output, engine, qd, ratio):
        print(f"SKIP {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    base.adc.PAGE_BYTES = PAGE
    command = base.adc.common_command(
        dataset,
        "coloc",
        base.ADC_ARTIFACT_ROOTS[PAGE],
        output,
        base.budget_mib(ratio),
        THREADS,
        (EF,),
        1,
        0,
    )
    command[0] = str(calibrated.ADC_BINARIES[PAGE])
    calibrated.replace_option(command, "--rerank-n", str(EF))
    command.extend(
        [
            "--rerank-dot", "fp32",
            "--fp-rerank-source", "odirect-uring",
            "--fp-rerank-norms", "onfetch",
            "--fp-rerank-path", str(FP_PATH),
            "--fp-rerank-qd", str(EF),
            "--calibrated-threshold-rerank-tau", repr(tau),
        ]
    )
    if engine == "async":
        command.extend(["--index-page-async", "--index-page-qd", str(qd)])
    base.adc.run_command(command, output.with_suffix(".log"), THREADS)
    if not complete(output, engine, qd, ratio):
        raise RuntimeError(f"invalid graph-I/O calibration result: {output}")
    return output


def summarize() -> tuple[Path, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for engine, qd in SETTINGS:
        name = setting_name(engine, qd)
        per_ratio_medians: list[float] = []
        per_ratio: dict[str, float] = {}
        for ratio in base.RATIOS:
            samples = [
                float(read_row(output_path(engine, qd, ratio, trial))["qps"])
                for trial in range(TRIALS)
            ]
            median = statistics.median(samples)
            per_ratio_medians.append(median)
            per_ratio[str(ratio)] = median
            rows.append(
                {
                    "setting": name,
                    "engine": engine,
                    "qd": qd,
                    "bq_ram_ratio": ratio,
                    "trials": TRIALS,
                    "qps_samples": ";".join(f"{value:.9g}" for value in samples),
                    "qps_median": median,
                    "qps_min": min(samples),
                    "qps_max": max(samples),
                }
            )
        geometric_mean = math.exp(
            statistics.mean(math.log(value) for value in per_ratio_medians)
        )
        for row in rows:
            if row["setting"] == name:
                row["qps_geomean_across_ram"] = geometric_mean

    calibrated.write_csv(RESULT_ROOT / "graph_io_calibration.csv", rows)
    scores: dict[str, float] = {}
    for row in rows:
        scores[str(row["setting"])] = float(row["qps_geomean_across_ram"])
    best_name = max(scores, key=scores.get)
    best_engine, best_qd = next(
        (engine, qd)
        for engine, qd in SETTINGS
        if setting_name(engine, qd) == best_name
    )
    selection: dict[str, object] = {
        "selection_source": "1000 calibration queries only",
        "page_bytes": PAGE,
        "layout": "coloc",
        "ram_ratios": list(base.RATIOS),
        "trials_per_setting_ratio": TRIALS,
        "objective": "geometric mean of per-RAM median QPS",
        "selected_setting": best_name,
        "selected_engine": best_engine,
        "selected_qd": best_qd,
        "scores": scores,
        "source_csv": str(RESULT_ROOT / "graph_io_calibration.csv"),
    }
    calibrated.write_json(RESULT_ROOT / "selection.json", selection)
    return RESULT_ROOT / "graph_io_calibration.csv", selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--page-bytes",
        type=int,
        choices=base.PAGES,
        default=4096,
    )
    args = parser.parse_args()
    configure_page(args.page_bytes)
    ensure_binary()
    tau, _ = calibrated.load_tau()
    split = calibration.make_split(
        base.DATASET,
        calibrated.CALIBRATION_QUERIES,
        calibrated.CALIBRATION_SEED,
        0,
    )
    dataset = replace(
        base.DATASET,
        query=split.calibration_query,
        gt=split.calibration_gt,
    )
    base.datasets_base.validate(dataset)
    if not calibrated.ADC_BINARIES[PAGE].exists():
        raise FileNotFoundError(calibrated.ADC_BINARIES[PAGE])
    base.adc.prepare_aligned_fp(FP_PATH, dataset)

    # Reverse both setting and RAM order in the second trial to reduce order
    # bias from SSD/device state.
    for trial in range(TRIALS):
        settings = SETTINGS if trial == 0 else tuple(reversed(SETTINGS))
        ratios = base.RATIOS if trial == 0 else tuple(reversed(base.RATIOS))
        for engine, qd in settings:
            for ratio in ratios:
                print(
                    f"CAL trial={trial} setting={setting_name(engine, qd)} "
                    f"ratio={ratio}",
                    flush=True,
                )
                run_one(dataset, tau, engine, qd, ratio, trial)

    csv_path, selection = summarize()
    print(f"SELECTED {json.dumps(selection, sort_keys=True)}", flush=True)
    print(f"CSV {csv_path}", flush=True)


if __name__ == "__main__":
    main()
