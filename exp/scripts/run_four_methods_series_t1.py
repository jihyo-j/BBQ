#!/usr/bin/env python3
"""Run the common four-method T=1 protocol on one or more dataset specs.

Methods:
  * ADC-BQ with per-ef pool-top10 split-conformal target 0.99.
  * TurboQuant TQ1+ with a calibration-query joint (ef, R) Pareto sweep and
    one selected fixed R on the held-out queries.
  * RaBitQ 1-bit c16 with official adaptive exact-FP refinement.
  * RaBitQ total-7-bit c16 in RAM.

The runner is restartable and removes only explicitly owned indexes after a
complete durable result curve has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiment"))

import diskveclab_full_matrix as matrix  # noqa: E402
import run_nq_quora_c16_rabitq as rq  # noqa: E402
import run_nq_qwen_multidim_calibrated_adcbq_tq_t1 as calibrated  # noqa: E402
import run_v051_ssd_odirect_multidataset as registry  # noqa: E402


DEFAULT_EFS = (
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    320,
    512,
    1024,
)
DEFAULT_EXTERNAL = Path(
    "/home/smrc/samsung-nvme/jihyo/four_methods_scalability_mrl_t1"
)
DEFAULT_RESULT = REPO / "experiment/four_methods_scalability_mrl_t1"
PREPARE_100M = REPO / "experiment/prepare_msmarco_qwen1024_100m_scalability_inputs.py"
SOURCE_100M = Path("/home/smrc/samsung-nvme/sieun/qwen100m")
MSMARCO_EXTERNAL = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_qwen1024_100m_scalability_t1"
)
MSMARCO_PREFIX = MSMARCO_EXTERNAL / "data/msmarco_qwen1024_v2_permuted_prefix.aligned4k.fbin"
MRL_EXTERNAL = Path("/home/smrc/samsung-nvme/jihyo/nq_qwen8_mrl_four_methods_t1")
MRL_PREPARE = REPO / "experiment/prepare_nq_qwen8_mrl_inputs.py"
LARGE_FP_CACHE_EVICT_BYTES = 256 * 1024**3


def log(message: str) -> None:
    print(message, flush=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(command: list[str], threads: int) -> None:
    log("RUN " + " ".join(command))
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    subprocess.run(command, cwd=REPO, env=environment, check=True)


def msmarco_spec(scale: int, build_threads: int) -> dict[str, object]:
    if scale not in (10_000_000, 50_000_000, 100_000_000):
        raise ValueError(f"unsupported MSMARCO scale: {scale}")
    split_root = (
        REPO / "experiment/msmarco_qwen1024_100m_scalability_t1/inputs"
    )
    run(
        [
            "/home/jihyo/anaconda3/envs/hnsw/bin/python3.11",
            str(PREPARE_100M),
            "--target-rows",
            str(scale),
            "--output",
            str(MSMARCO_PREFIX),
            "--split-dir",
            str(split_root),
            "--threads",
            str(build_threads),
            "--reserve-gib",
            "20",
        ],
        build_threads,
    )
    tag = f"{scale // 1_000_000}m"
    index_root = MSMARCO_EXTERNAL / "indexes" / tag
    return {
        "key": f"msmarco_qwen1024_v2_{tag}",
        "corpus": "MSMARCO",
        "model": f"Qwen3-Embedding-0.6B 1024D ({tag.upper()})",
        "dim": 1024,
        "train": str(MSMARCO_PREFIX),
        "query": str(SOURCE_100M / "insert_exp/queries/queries.fbin"),
        "gt": str(SOURCE_100M / f"insert_exp/gt/gt_{scale}.ibin"),
        "fp_sidecar": str(MSMARCO_PREFIX),
        "adcbq_index": str(index_root / f"{tag}.adcbq.M24.efC200.faiss"),
        "b1_index": str(index_root / f"{tag}.rabitq_b1_c16.M24.efC200.index"),
        "b7_index": str(index_root / f"{tag}.rabitq_b7_c16.M24.efC200.index"),
        "owned_indexes": True,
        "scale_rows": scale,
    }


def mrl_specs(build_threads: int) -> list[dict[str, object]]:
    manifest = MRL_EXTERNAL / "dataset_manifest.json"
    run(
        [
            # Use the runner environment here: its FAISS is linked against
            # the bundled OpenBLAS.  The legacy hnsw environment resolves a
            # mixed-version system MKL and fails during larger exact searches.
            sys.executable,
            str(MRL_PREPARE),
            "--output-root",
            str(MRL_EXTERNAL),
            "--threads",
            str(build_threads),
            "--dims",
            "256",
            "512",
            "1024",
            "2560",
            "4096",
        ],
        build_threads,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return list(payload["datasets"])


def register(spec: dict[str, object]) -> registry.Dataset:
    dataset = registry.Dataset(
        key=str(spec["key"]),
        corpus=str(spec["corpus"]),
        model=str(spec["model"]),
        title=f"{spec['corpus']} {spec['model']}",
        train=Path(str(spec["train"])),
        query=Path(str(spec["query"])),
        gt=Path(str(spec["gt"])),
        b1_index=Path(str(spec["b1_index"])),
        b7_index=Path(str(spec["b7_index"])),
        adcbq_index=Path(str(spec["adcbq_index"])),
        index_provenance="four-method-series-t1",
    )
    registry.DATASETS[dataset.key] = dataset
    matrix.registry.DATASETS[dataset.key] = dataset
    matrix.CALIBRATION_ROWS[dataset.key] = int(spec.get("calibration_rows", 1000))
    registry.validate(dataset)
    fp_sidecar = Path(str(spec["fp_sidecar"]))
    if not fp_sidecar.is_file():
        raise FileNotFoundError(fp_sidecar)
    return dataset


def configure_calibrated(
    dataset: registry.Dataset,
    fp_sidecar: Path,
    result_root: Path,
    external_root: Path,
    efs: tuple[int, ...],
    calibration_rows: int,
    search_threads: int,
) -> None:
    calibrated.DATASETS = (dataset.key,)
    calibrated.EFS = efs
    calibrated.RESULT_ROOT = result_root
    calibrated.ADC_CALIBRATION_ROOT = result_root / "adcbq_pool_top10_target99_per_ef"
    calibrated.ADC_RAW_ROOT = result_root / "adcbq_pool_top10_target99_per_ef_raw"
    calibrated.ADC_SUMMARY = (
        result_root / f"adcbq_pool_top10_target99_per_ef_t{search_threads}.csv"
    )
    # v2 results use FP sidecars built with the source payload offset checked.
    # Keep any pre-fix CSVs untouched and never consider them reusable.
    calibrated.TQ_RAW_ROOT = result_root / "tq_raw_v2"
    calibrated.TMP_ROOT = external_root / "tq_work"
    calibrated.FIGURE_ROOT = result_root / "figures"
    calibrated.aligned_fp_path = lambda _key: fp_sidecar
    calibrated.CALIBRATION_ROWS = calibration_rows
    calibrated.configure_modules()


def configure_rabitq(
    result_root: Path,
    external_root: Path,
    fp_sidecar: Path,
    efs: tuple[int, ...],
) -> None:
    rq.RESULT_ROOT = result_root / "rabitq_c16"
    rq.EXTERNAL_ROOT = external_root / "rabitq_c16"
    rq.INDEX_ROOT = rq.EXTERNAL_ROOT / "indexes"
    rq.FP_ROOT = rq.EXTERNAL_ROOT / "aligned_fp"
    rq.EFS = efs
    rq.index_path = lambda dataset, bits: (
        dataset.b1_index if bits == 1 else dataset.b7_index
    )
    rq.fp_path = lambda _dataset: fp_sidecar


def normalize_adc_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        output.append({**row, "method_key": "adc_bq_calibrated"})
    return output


def normalize_rq_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in rows:
        row = dict(raw)
        family = str(row["family"])
        row["method_key"] = "rabitq_b1_adaptive" if "b1" in family else "rabitq_b7"
        row["search_parameter"] = "efSearch"
        row["search_parameter_value"] = row["efSearch"]
        row["test_queries"] = row["nq"]
        output.append(row)
    return output


def rabitq_bit_search_complete(
    dataset: registry.Dataset,
    bits: int,
    *,
    rounds: int,
    search_threads: int,
) -> bool:
    """Recognize a durable single-bit result before rebuilding a deleted index."""
    family = (
        "rabitq_b1_c16_adaptive_odirect_aligned4k"
        if bits == 1
        else "rabitq_b7_c16_ram"
    )
    root = rq.result_dir(dataset)
    rows = [
        row for row in read_csv(root / "search_results.csv")
        if row.get("family") == family
    ]
    try:
        if (
            {int(row["efSearch"]) for row in rows} != set(efs := rq.EFS)
            or len(rows) != len(efs)
            or any(int(row["rounds"]) != rounds for row in rows)
            or any(int(row["search_threads"]) != search_threads for row in rows)
        ):
            return False
        if bits == 1:
            nq, _ = registry.matrix_shape(dataset.query)
            with (root / "per_query_fp_reads.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                details = sum(1 for _ in csv.DictReader(handle))
            return details == nq * len(efs) * rounds
        return True
    except (OSError, KeyError, ValueError):
        return False


def remove_owned(path: Path, owned_root: Path) -> int:
    resolved = path.resolve()
    root = owned_root.resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"refusing cleanup outside owned root: {resolved}")
    if not path.is_file():
        return 0
    size = path.stat().st_size
    path.unlink()
    log(f"REMOVED completed owned index bytes={size}: {path}")
    return size


def evict_large_fp_clean_cache(path: Path) -> None:
    """Release TQ's clean page cache before loading a huge FP array.

    The 100M arm maps the 409.6 GB FP sidecar while building TurboQuant, then
    immediately reads the same file into an anonymous NumPy array for RaBitQ.
    Keeping both copies resident can push the kernel into sustained reclaim
    even though the file-backed pages are clean and disposable.  Small inputs
    benefit from the cache, so only evict exceptionally large sidecars.
    """
    try:
        size = path.stat().st_size
    except OSError as error:
        log(f"SKIP FP cache eviction (stat failed): {path}: {error}")
        return
    if size < LARGE_FP_CACHE_EVICT_BYTES:
        return
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        log(f"SKIP FP cache eviction (unsupported): {path}")
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError as error:
        log(f"SKIP FP cache eviction (fadvise failed): {path}: {error}")
        return
    log(f"EVICTED clean FP cache bytes={size}: {path}")


def run_one(
    spec: dict[str, object],
    *,
    result_base: Path,
    external_base: Path,
    efs: tuple[int, ...],
    rounds: int,
    pilot_rounds: int,
    build_threads: int,
    cpu: int,
    cleanup_indexes: bool,
    search_threads: int = 1,
) -> Path:
    expected_result_root = result_base / str(spec["key"])
    expected_summary = expected_result_root / f"summary_t{search_threads}.csv"
    expected_status = expected_result_root / "status.json"
    if expected_summary.is_file() and expected_status.is_file():
        try:
            status_payload = json.loads(expected_status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status_payload = {}
        if status_payload.get("stage") == "complete":
            log(f"SKIP complete {spec['key']}: {expected_summary}")
            return expected_summary
    dataset = register(spec)
    fp_sidecar = Path(str(spec["fp_sidecar"]))
    result_root = result_base / dataset.key
    external_root = external_base / dataset.key
    result_root.mkdir(parents=True, exist_ok=True)
    external_root.mkdir(parents=True, exist_ok=True)
    dataset.adcbq_index.parent.mkdir(parents=True, exist_ok=True)
    dataset.b1_index.parent.mkdir(parents=True, exist_ok=True)
    dataset.b7_index.parent.mkdir(parents=True, exist_ok=True)
    configure_calibrated(
        dataset,
        fp_sidecar,
        result_root,
        external_root,
        efs,
        int(spec.get("calibration_rows", 1000)),
        search_threads,
    )
    args = SimpleNamespace(
        cpu=cpu,
        rounds=rounds,
        pilot_rounds=pilot_rounds,
        build_threads=build_threads,
        tq_train_size=100_000,
        force=False,
        keep_candidates=False,
        search_threads=search_threads,
    )

    # Build the shared FP graph before entering the CPU-pinned measurement
    # phase.  The generated calibration point is reused below.
    split = calibrated.split_for(dataset.key)
    calibrated.adc.ensure_binary()
    calibrated.fit_adcbq_per_ef(dataset, split, efs[0], args)
    adc_rows = calibrated.run_adcbq(args)
    tq_rows, _ = calibrated.run_turboquant(args)

    # TurboQuant no longer needs the ADC graph.  Releasing it before the two
    # RaBitQ builds materially lowers peak SSD usage for the 100M arm.
    if cleanup_indexes and bool(spec.get("owned_indexes", True)):
        remove_owned(dataset.adcbq_index, external_base)

    # TurboQuant just scanned the FP sidecar.  On the 100M workload, discard
    # those clean file-backed pages before RaBitQ allocates its 409.6 GB
    # writable training array; otherwise page-cache reclaim can dominate the
    # subsequent sequential load.  This is a cache hint only and never alters
    # the source file.
    evict_large_fp_clean_cache(fp_sidecar)

    configure_rabitq(result_root, external_root, fp_sidecar, efs)
    heldout = replace(dataset, query=split.test_query, gt=split.test_gt)
    rq_args = SimpleNamespace(
        build_threads=build_threads,
        search_threads=search_threads,
        rounds=rounds,
        force=False,
    )
    shared_indexes = not bool(spec.get("owned_indexes", True))
    rq_rows: list[dict[str, object]]
    if cleanup_indexes and not shared_indexes:
        # Build, measure, and release one c16 index at a time.  This preserves
        # the exact benchmark protocol while avoiding the b1+b7 on-disk peak.
        for partial in (dataset.b1_index, dataset.b7_index):
            if partial.is_file():
                remove_owned(partial, external_base)
        for bits in (1, 7):
            rq_args.only_bits = (bits,)
            if rabitq_bit_search_complete(
                heldout,
                bits,
                rounds=rounds,
                search_threads=search_threads,
            ):
                log(f"SKIP durable RaBitQ b{bits} search: {dataset.key}")
                rq_rows = [
                    dict(row)
                    for row in read_csv(rq.result_dir(heldout) / "search_results.csv")
                ]
                continue
            rq.build_dataset(heldout, rq_args)
            original_affinity = os.sched_getaffinity(0)
            try:
                if cpu not in original_affinity:
                    raise RuntimeError(
                        f"search CPU {cpu} outside affinity {sorted(original_affinity)}"
                    )
                os.sched_setaffinity(
                    0, set(calibrated.affinity_cpus(cpu, search_threads))
                )
                rq_rows = rq.search_dataset(heldout, rq_args)
            finally:
                os.sched_setaffinity(0, original_affinity)
            index_path = dataset.b1_index if bits == 1 else dataset.b7_index
            remove_owned(index_path, external_base)
        delattr(rq_args, "only_bits")
    else:
        if not (shared_indexes and dataset.b1_index.is_file() and dataset.b7_index.is_file()):
            if bool(spec.get("owned_indexes", True)) and (
                dataset.b1_index.is_file() != dataset.b7_index.is_file()
            ):
                # A single surviving file is an unpublished partial build.  Both
                # indexes are deterministic and owned by this campaign, so clear
                # the exact pair before restarting their shared build pipeline.
                for partial in (dataset.b1_index, dataset.b7_index):
                    remove_owned(partial, external_base)
            rq.build_dataset(heldout, rq_args)
        original_affinity = os.sched_getaffinity(0)
        try:
            if cpu not in original_affinity:
                raise RuntimeError(f"search CPU {cpu} outside affinity {sorted(original_affinity)}")
            os.sched_setaffinity(
                0, set(calibrated.affinity_cpus(cpu, search_threads))
            )
            rq_rows = rq.search_dataset(heldout, rq_args)
        finally:
            os.sched_setaffinity(0, original_affinity)
    rq.combine([heldout])

    combined = [
        *normalize_adc_rows(adc_rows),
        *[dict(row) for row in tq_rows],
        *normalize_rq_rows(rq_rows),
    ]
    method_order = {
        "adc_bq_calibrated": 0,
        "turboquant_tq1": 1,
        "rabitq_b1_adaptive": 2,
        "rabitq_b7": 3,
    }
    combined.sort(
        key=lambda row: (
            method_order.get(str(row.get("method_key")), 99),
            int(row.get("efSearch", row.get("search_parameter_value", 0))),
        )
    )
    summary = result_root / f"summary_t{search_threads}.csv"
    write_csv(summary, combined)
    status = result_root / "status.json"
    status.write_text(
        json.dumps(
            {
                "stage": "complete",
                "dataset": dataset.key,
                "methods": list(method_order),
                "efs": list(efs),
                "rounds": rounds,
                "search_threads": search_threads,
                "summary": str(summary),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if cleanup_indexes and bool(spec.get("owned_indexes", True)):
        for path in (dataset.adcbq_index, dataset.b1_index, dataset.b7_index):
            remove_owned(path, external_base)
    log(f"COMPLETE {dataset.key}: {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", choices=("msmarco100m", "nq_mrl"), required=True
    )
    parser.add_argument(
        "--scales", nargs="+", type=int, default=[10_000_000, 50_000_000, 100_000_000]
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[256, 512, 1024, 2560, 4096])
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pilot-rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=39)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--keep-indexes", action="store_true")
    parser.add_argument("--remove-final-prefix", action="store_true")
    args = parser.parse_args()
    if (
        args.rounds < 1
        or args.pilot_rounds < 1
        or args.build_threads < 1
        or args.search_threads < 1
    ):
        parser.error("round and thread counts must be positive")
    efs = tuple(dict.fromkeys(args.efs))
    if len(efs) != 20:
        parser.error("this protocol requires the 20-point front-loaded ef sweep")

    summaries: list[str] = []
    if args.campaign == "msmarco100m":
        scales = list(dict.fromkeys(args.scales))
        for scale in scales:
            spec = msmarco_spec(scale, args.build_threads)
            summary = run_one(
                spec,
                result_base=args.result_root / "msmarco100m",
                external_base=MSMARCO_EXTERNAL,
                efs=efs,
                rounds=args.rounds,
                pilot_rounds=args.pilot_rounds,
                build_threads=args.build_threads,
                cpu=args.cpu,
                cleanup_indexes=not args.keep_indexes,
                search_threads=args.search_threads,
            )
            summaries.append(str(summary))
        if args.remove_final_prefix and scales and max(scales) == 100_000_000:
            remove_owned(MSMARCO_PREFIX, MSMARCO_EXTERNAL)
    else:
        wanted = set(args.dims)
        specs = [spec for spec in mrl_specs(args.build_threads) if int(spec["dim"]) in wanted]
        for spec in specs:
            summary = run_one(
                spec,
                result_base=args.result_root / "nq_mrl",
                external_base=MRL_EXTERNAL,
                efs=efs,
                rounds=args.rounds,
                pilot_rounds=args.pilot_rounds,
                build_threads=args.build_threads,
                cpu=args.cpu,
                cleanup_indexes=not args.keep_indexes,
                search_threads=args.search_threads,
            )
            summaries.append(str(summary))
    campaign_status = args.result_root / f"{args.campaign}.status.json"
    campaign_status.parent.mkdir(parents=True, exist_ok=True)
    campaign_status.write_text(
        json.dumps({"stage": "complete", "summaries": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
