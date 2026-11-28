#!/usr/bin/env python3
"""ADC-BQ colocation ablation on MSMARCO text-embedding-3-small 1536D.

Fit tau independently for the requested ef=256 using calibration queries.
Compare 4/8/16 KiB pages and 10/25/50/75% BQ-derived cache budgets with and
without LDG colocation. All graph levels are paged; graph I/O uses the
existing batch-cache QD16 implementation. RaBitQ is intentionally absent.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from types import SimpleNamespace

import run_adcbq_calibrated_threshold_overlap as calibration
import run_nq_qwen_multidim_calibrated_adcbq_tq_t1 as per_ef
import run_msmarco_page4k_ef256_qps_rerun as search

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiment/msmarco_te3small1536_adcbq_colocation_ef256_t24"
EXTERNAL = Path("/home/smrc/samsung-nvme/jihyo/msmarco_te3small1536_adcbq_colocation_ef256_t24")
KEY = "sieun_msmarco_openai1536"
EF = 256
THREADS = 24
PAGES = (4096, 8192, 16384)
RATIOS = (0.10, 0.25, 0.50, 0.75)
LAYOUTS = ("baseline", "coloc")
TRIALS = (0, 1, 2, 3)  # Discard trial zero; use three fresh measured processes.
SIDE = EXTERNAL / "fp32.header4096.fbin"
MAP = EXTERNAL / "layouts/shared.ldg20.p64.map"
STATUS = ROOT / "status.json"
base = search.base
adc = base.adc
write_json = search.calibrated.write_json
write_csv = search.calibrated.write_csv


def status(stage: str, **extra: object) -> None:
    payload = {"stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "pid": os.getpid(), "dataset": KEY, "efSearch": EF, "threads": THREADS, **extra}
    write_json(STATUS, payload)
    print(json.dumps(payload), flush=True)


def dataset():
    manifest = REPO / "experiment/full16_12methods_t24/dataset_manifest.json"
    spec = next(x for x in json.loads(manifest.read_text())["datasets"] if x["key"] == KEY)
    assert spec["model"] == "OpenAI text-embedding-3-small" and int(spec["dim"]) == 1536
    item = replace(base.datasets_base.DATASETS[KEY],
                   key=KEY, corpus=spec["corpus"], model=spec["model"],
                   title="MSMARCO text-embedding-3-small 1536D",
                   train=Path(spec["train"]), query=Path(spec["query"]), gt=Path(spec["gt"]),
                   adcbq_index=EXTERNAL / "indexes" / f"{KEY}.adcbq.M24.efC200.faiss",
                   b1_index=EXTERNAL / "unused.rabitq_b1.index", b7_index=EXTERNAL / "unused.rabitq_b7.index",
                   index_provenance="fresh graph for text-embedding-3-small colocation")
    base.datasets_base.validate(item)
    assert base.datasets_base.matrix_shape(item.train) == (1000000, 1536)
    assert base.datasets_base.matrix_shape(item.query) == (6980, 1536)
    return item


def configure(item) -> None:
    base.DATASET = item
    base.THREADS = THREADS
    base.RATIOS = RATIOS
    base.PAGES = PAGES
    base.ADC_ARTIFACT_ROOTS = {p: EXTERNAL / "layouts" / f"{p // 1024}k" for p in PAGES}
    base.ADC_INDEX_RESULT_ROOTS = {p: ROOT / "layouts" / f"{p // 1024}k" for p in PAGES}
    adc.colocation_map_path = lambda _item, _root: MAP
    per_ef.ADC_CALIBRATION_ROOT = ROOT / "calibration"
    per_ef.EFS = (EF,)
    per_ef.matrix.registry.DATASETS[KEY] = item
    per_ef.matrix.CALIBRATION_ROWS[KEY] = 1000
    search.THREADS = THREADS
    search.EF = EF
    search.ADC_GRAPH_QD = 16
    search.ADC_SYNC_MISS_MAX = 0


def ensure_binaries() -> None:
    source = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw.cpp"
    for page in PAGES:
        binary = search.ADC_BINARIES[page]
        if not binary.exists() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
            status("compiling", page_bytes=page, binary=str(binary))
            env = {**os.environ, "PAGE_BYTES": str(page), "OUT": str(binary)}
            subprocess.run([str(search.ADC_BUILD_SCRIPT)], cwd=REPO, env=env, check=True)
    calibration.BIN = search.ADC_BINARIES[4096]


def ensure_graph(item, split) -> dict:
    output = ROOT / "build/fp_graph.csv"
    if item.adcbq_index.exists() and output.exists():
        return adc.read_csv(output)[0]
    status("building_fp_graph", M=24, efConstruction=200, build_threads=24)
    item.adcbq_index.parent.mkdir(parents=True, exist_ok=True)
    command = calibration.common_command(item, split.calibration_query, split.calibration_gt,
                                         output, EF, 1, 1, THREADS)
    search.calibrated.replace_option(command, "--max-queries", "1")
    command += ["--no-rerank", "--rebuild"]
    adc.run_command(command, ROOT / "build/fp_graph.log", THREADS)
    raw = adc.read_csv(output)[0]
    assert raw["index_status"] == "built" and int(raw["nb"]) == 1000000 and int(raw["dim"]) == 1536
    return raw


def prepare_layout(item, page: int, layout: str, graph: dict) -> dict:
    adc.PAGE_BYTES = page
    adc.BIN = search.ADC_BINARIES[page]
    adc.RESULT_ROOT = base.ADC_INDEX_RESULT_ROOTS[page]
    destination = adc.build_manifest_path(item, layout)
    paged = adc.paged_index_path(item, layout, base.ADC_ARTIFACT_ROOTS[page])
    if destination.exists():
        payload = json.loads(destination.read_text())
        if paged.exists() and paged.stat().st_size == int(payload["paged_index_bytes"]):
            return payload
    status("materializing_layout", page_bytes=page, layout=layout)
    output = destination.with_name(f"build_{layout}_smoke.csv")
    paged.parent.mkdir(parents=True, exist_ok=True)
    command = adc.common_command(item, layout, base.ADC_ARTIFACT_ROOTS[page], output, 1, 1, (EF,), 1, 1)
    command += ["--no-rerank", "--rebuild-paged-index"]
    # All page sizes share one permutation of the exact same graph.
    adc.run_command(command, output.with_suffix(".log"), THREADS)
    raw = adc.read_csv(output)[0]
    adc.validate_build_row(raw, layout)
    payload = {"dataset": KEY, "model": item.model, "page_bytes": page, "layout": layout,
               "paged_index_path": str(paged), "paged_index_bytes": int(raw["paged_index_file_bytes"]),
               "paged_index_data_bytes": int(raw["paged_index_data_pages"]) * page,
               "base_index_build_seconds": float(graph["build_or_load_seconds"]),
               "base_index_bytes": item.adcbq_index.stat().st_size,
               "colocation_map_path": str(MAP) if layout == "coloc" else "",
               "source_smoke_csv": str(output)}
    write_json(destination, payload)
    return payload


def observation(page: int, layout: str, ratio: float, trial: int) -> Path:
    phase = "warmup" if trial == 0 else "measured"
    return ROOT / phase / f"{page // 1024}k" / f"{layout}_{base.ratio_label(ratio)}_trial{trial}.csv"


def complete(path: Path, page: int, layout: str, ratio: float, tau: float) -> bool:
    search.PAGE = page
    # The C++ CSV writer uses defaultfloat with six significant digits.
    # Commands and the calibration manifest retain the full fitted value.
    csv_tau = float(format(tau, ".6g"))
    if not search.result_complete(path, search.METHOD_ADC, layout, ratio, 0, csv_tau):
        return False
    row = adc.read_csv(path)[0]
    return row["dataset"] == KEY and int(row["nb"]) == 1000000 and int(row["dim"]) == 1536


def collect(tau: float) -> tuple[list, list]:
    measured = []
    for page in PAGES:
        search.PAGE = page
        for layout in LAYOUTS:
            for ratio in RATIOS:
                for trial in TRIALS[1:]:
                    path = observation(page, layout, ratio, trial)
                    if not complete(path, page, layout, ratio, tau):
                        continue
                    row = search.normalized_row(search.METHOD_ADC, layout, ratio, trial, path)
                    row.update(dataset=KEY, model=base.DATASET.model, page_bytes=page,
                               efSearch=EF, threads=THREADS, calibration_tau=tau,
                               calibration_protocol=per_ef.ADC_PROTOCOL, calibration_queries=1000,
                               test_queries=5980, cache_mib=base.budget_mib(ratio))
                    measured.append(row)
    grouped = collections.defaultdict(list)
    for row in measured:
        grouped[(row["page_bytes"], row["layout"], row["bq_ram_ratio_target"])].append(row)
    summary = []
    for (page, layout, ratio), group in sorted(grouped.items()):
        if len(group) != 3:
            continue
        qps = [row["qps"] for row in group]
        summary.append({"dataset": KEY, "model": base.DATASET.model, "method": "ADC-BQ calibrated",
                        "page_bytes": page, "layout": layout, "bq_ram_ratio_target": ratio,
                        "cache_mib": base.budget_mib(ratio), "efSearch": EF, "threads": THREADS,
                        "measured_trials": 3, "qps": statistics.median(qps), "qps_median": statistics.median(qps),
                        "qps_mean": statistics.mean(qps), "qps_stddev": statistics.pstdev(qps),
                        "qps_min": min(qps), "qps_max": max(qps),
                        "recall_at_10": statistics.mean(r["recall_at_10"] for r in group),
                        "graph_page_hit_ratio": statistics.mean(r["graph_page_hit_ratio"] for r in group),
                        "graph_physical_4k_reads_per_query": statistics.mean(r["graph_physical_4k_reads_per_query"] for r in group),
                        "fp_vectors_mean": statistics.mean(r["fp_vectors_mean"] for r in group),
                        "fp_physical_4k_reads_per_query": statistics.mean(r["fp_physical_4k_reads_per_query"] for r in group),
                        "total_physical_4k_reads_per_query": statistics.mean(r["total_physical_4k_reads_per_query"] for r in group),
                        "calibration_tau": tau, "calibration_protocol": per_ef.ADC_PROTOCOL,
                        "source_csv": ";".join(str(r["source_csv"]) for r in group)})
    if measured:
        write_csv(ROOT / "msmarco_te3small1536_colocation_measured_trials.csv", measured)
    if summary:
        write_csv(ROOT / "msmarco_te3small1536_colocation_summary.csv", summary)
    return measured, summary


def prepare() -> tuple:
    ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    item = dataset()
    configure(item)
    split = calibration.make_split(item, 1000, 11, 0)
    assert split.test_rows == 5980
    manifest = {"dataset": KEY, "model": item.model, "dim": 1536, "nb": 1000000,
                "train": str(item.train), "query": str(item.query), "gt": str(item.gt),
                "heldout_query": str(split.test_query), "heldout_gt": str(split.test_gt),
                "calibration_queries": 1000, "test_queries": 5980, "seed": 11,
                "methods": ["ADC-BQ calibrated"], "efSearch": [EF], "threads": THREADS,
                "page_bytes": PAGES, "ram_ratios": RATIOS, "layouts": LAYOUTS,
                "warmup_trials_discarded": 1, "measured_trials": 3,
                "measured_configurations": 24, "measured_observations": 72, "total_observations": 96,
                "calibration_protocol": per_ef.ADC_PROTOCOL, "target_pool_top10_coverage": 0.99,
                "tau_policy": "fit tau_256 on calibration queries only; freeze across layouts, pages and cache budgets",
                "graph_io": "batch-cache, all-cache-hit fast return, async io_uring QD16",
                "fp_io": "O_DIRECT io_uring, on-fetch norms, QD256 per query",
                "cache_policy": "shared cold LRU in each fresh process; cap derived from 4KB baseline footprint",
                "colocation": {"iterations": 20, "partition_size": 64, "shared_permutation": str(MAP)},
                "artifact_root": str(EXTERNAL)}
    write_json(ROOT / "run_manifest.json", manifest)
    return item, split, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    item, split, manifest = prepare()
    if args.prepare_only:
        status("prepared", total_observations=96)
        return
    with (ROOT / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("colocation campaign is already running")
        ensure_binaries()
        graph = ensure_graph(item, split)
        status("calibrating_tau", calibration_queries=1000, target_pool_top10_coverage=0.99)
        fitted = per_ef.fit_adcbq_per_ef(item, split, EF,
                                        SimpleNamespace(force=False, build_threads=24, keep_candidates=True))
        tau = float(fitted["tau"])
        manifest["calibration"] = fitted
        write_json(ROOT / "run_manifest.json", manifest)
        adc.prepare_aligned_fp(SIDE, item)
        heldout = replace(item, query=split.test_query, gt=split.test_gt)
        configure(heldout)
        completed = 0
        for page in PAGES:
            for layout in LAYOUTS:
                prepare_layout(heldout, page, layout, graph)
            search.PAGE = page
            combinations = [(layout, ratio) for layout in LAYOUTS for ratio in RATIOS]
            for trial in TRIALS:
                ordered = combinations if trial % 2 == 0 else list(reversed(combinations))
                for layout, ratio in ordered:
                    path = observation(page, layout, ratio, trial)
                    if not complete(path, page, layout, ratio, tau):
                        status("measuring", page_bytes=page, layout=layout, ram_ratio=ratio, trial=trial,
                               completed_observations=completed, total_observations=96, tau=tau)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        search.run_adc(path, layout, ratio, tau, SIDE)
                        if not complete(path, page, layout, ratio, tau):
                            raise RuntimeError(f"invalid observation: {path}")
                    completed += 1
            collect(tau)
        measured, summary = collect(tau)
        assert len(measured) == 72 and len(summary) == 24
        references = {}
        for row in measured:
            key = row["layout"]
            ref = references.setdefault(key, row)
            for field in ("recall_at_10", "fp_vectors_mean", "fp_physical_4k_reads_per_query"):
                assert math.isclose(row[field], ref[field], rel_tol=0, abs_tol=1e-8), (key, field, row[field], ref[field])
            assert math.isfinite(row["qps"]) and row["qps"] > 0
        write_json(ROOT / "validation.json", {"status": "passed", "measured_observations": 72,
                   "configurations": 24, "search_semantics_invariant_across_pages_and_budgets": True})
        destination = Path("/home/jihyo/msmarco_te3small1536_adcbq_colocation_t24.csv")
        temporary = destination.with_suffix(".csv.tmp")
        shutil.copyfile(ROOT / "msmarco_te3small1536_colocation_summary.csv", temporary)
        os.replace(temporary, destination)
        status("complete", completed_observations=96, measured_observations=72,
               configurations=24, summary_csv=str(destination), tau=tau)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
