#!/usr/bin/env python3
"""Full-query NQ/Quora SSD comparison with a shared 4 KiB FP store.

The measured methods are ADC-BQ fixed FP rerank@50, ADC-BQ fixed FP
rerank@80, and RaBitQ 1-bit adaptive FP reranking.  Every method reads the
same fbin sidecar whose FP32 payload begins at byte 4096.  SYN and the NQ
Qwen-4B MRL-1024 variants are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import run_v051_ssd_odirect_multidataset as base


REPO = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO / "experiment/nq_quora_aligned4k_rerank50_80"
FIGURE_DIR = REPO / "experiment/figures/nq_quora_aligned4k_rerank50_80"
ALIGNED_FP_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/adcbq_aligned4k_fp_transient"
)
RABITQ_BINDING_DIR = (
    REPO
    / "experiment/rabitq_v051_adaptive_odirect_aligned4k_build/rabitqlib"
)
ADCBQ_BIN = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw"

EFS = (64, 128, 256, 512, 1024)
RERANKS = (50, 80)
ROUNDS = 3
SEARCH_THREADS = 1
QD = 80
PAGE_BYTES = 4096

B1_FAMILY = "rabitq_b1_adaptive_odirect_aligned4k"


def adc_family(rerank: int) -> str:
    return f"adcbq_int16_aligned4k_uring_overlap_qd{QD}_rerank{rerank}"


FAMILIES = (adc_family(50), adc_family(80), B1_FAMILY)
METHOD_LABELS = {
    adc_family(50): "ADC-BQ + fixed FP rerank@50",
    adc_family(80): "ADC-BQ + fixed FP rerank@80",
    B1_FAMILY: "RaBitQ 1-bit + adaptive FP",
}


def target_datasets() -> list[base.Dataset]:
    return [
        item
        for item in base.DATASETS.values()
        if item.corpus in {"NQ", "Quora"}
        and "syn" not in item.key.lower()
        and "SYN" not in item.model
        and item.key != "nq_qwen4_mrl1024"
    ]


def available_datasets(requested: list[str] | None = None) -> list[base.Dataset]:
    """Keep the old default while accepting explicit shared-registry keys."""
    if requested:
        missing = sorted(set(requested) - set(base.DATASETS))
        if missing:
            raise ValueError(f"unknown datasets: {missing}")
        return [base.DATASETS[key] for key in requested]
    return target_datasets()


def output_csv(dataset: base.Dataset, family: str) -> Path:
    return RESULT_DIR / f"{dataset.key}.{family}.M24.efC200.csv"


def aligned_fp_path(dataset: base.Dataset, root: Path) -> Path:
    return root / f"{dataset.key}.fp32.aligned4k.fbin"


def validate_aligned_fp(path: Path, dataset: base.Dataset) -> None:
    n, d = base.matrix_shape(dataset.train)
    expected = PAGE_BYTES + n * d * np.dtype(np.float32).itemsize
    if not path.exists() or path.stat().st_size != expected:
        raise RuntimeError(
            f"missing or invalid aligned FP store: {path} "
            f"(expected {expected} bytes)"
        )
    if base.matrix_shape(path) != (n, d):
        raise RuntimeError(f"aligned FP header mismatch: {path}")


def prepare_aligned_fp(
    path: Path, dataset: base.Dataset, reuse_existing: bool
) -> None:
    if reuse_existing:
        try:
            validate_aligned_fp(path, dataset)
            print(f"REUSE fsynced aligned FP store: {path}", flush=True)
            return
        except RuntimeError:
            pass

    n, d = base.matrix_shape(dataset.train)
    payload_bytes = n * d * np.dtype(np.float32).itemsize
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    copied = 0
    print(f"PREPARE aligned FP store outside timed search: {path}", flush=True)
    try:
        with dataset.train.open("rb", buffering=0) as source, temporary.open(
            "wb", buffering=0
        ) as destination:
            header = source.read(8)
            if len(header) != 8:
                raise RuntimeError(f"short fbin header: {dataset.train}")
            destination.write(header)
            destination.write(bytes(PAGE_BYTES - 8))
            while copied < payload_bytes:
                block = source.read(min(8 * 1024 * 1024, payload_bytes - copied))
                if not block:
                    raise RuntimeError(f"short fbin payload: {dataset.train}")
                destination.write(block)
                copied += len(block)
            destination.flush()
            os.fsync(destination.fileno())
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(
                    destination.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_aligned_fp(path, dataset)
    print(
        f"PREPARED+FSYNC aligned FP store ({path.stat().st_size} bytes): {path}",
        flush=True,
    )


def adc_complete(path: Path, rerank: int, expected_nq: int) -> bool:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return (
            [int(row["efSearch"]) for row in rows] == list(EFS)
            and all(int(row["nq"]) == expected_nq for row in rows)
            and all(int(row["rounds"]) == ROUNDS for row in rows)
            and all(int(row["search_threads"]) == SEARCH_THREADS for row in rows)
            and all(int(row["rerank_n"]) == rerank for row in rows)
            and all("NativeCodes" in row["method"] for row in rows)
            and all(row["fp_rerank_source"] == "odirect-uring" for row in rows)
            and all(row["fp_rerank_overlap_queries"].lower() == "true" for row in rows)
            and all(int(row["fp_rerank_qd"]) == QD for row in rows)
            and all(Path(row["fp_rerank_path"]).name.endswith("aligned4k.fbin") for row in rows)
        )
    except (OSError, KeyError, ValueError):
        return False


def rabitq_complete(path: Path, expected_nq: int) -> bool:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return (
            [int(row["efSearch"]) for row in rows] == list(EFS)
            and all(int(row["nq"]) == expected_nq for row in rows)
            and all(int(row["rounds"]) == ROUNDS for row in rows)
            and all(int(row["search_threads"]) == SEARCH_THREADS for row in rows)
            and all(int(row["fbin_data_offset"]) == PAGE_BYTES for row in rows)
            and all(row["fp_store_layout"] == "header4096_rows_contiguous" for row in rows)
        )
    except (OSError, KeyError, ValueError):
        return False


def dataset_complete(dataset: base.Dataset) -> bool:
    nq, _ = base.matrix_shape(dataset.query)
    return (
        all(adc_complete(output_csv(dataset, adc_family(r)), r, nq) for r in RERANKS)
        and (
            B1_FAMILY not in FAMILIES
            or rabitq_complete(output_csv(dataset, B1_FAMILY), nq)
        )
    )


def read_cpu_totals() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3]
    total = sum(values)
    return idle, total


def cpu_idle_percent(seconds: float) -> float:
    idle0, total0 = read_cpu_totals()
    time.sleep(seconds)
    idle1, total1 = read_cpu_totals()
    delta_total = max(1, total1 - total0)
    return 100.0 * (idle1 - idle0) / delta_total


def wait_for_idle(args: argparse.Namespace, label: str) -> None:
    if args.min_idle_percent <= 0:
        return
    consecutive = 0
    started = time.monotonic()
    last_report = 0.0
    while consecutive < args.idle_consecutive:
        idle = cpu_idle_percent(args.idle_sample_seconds)
        consecutive = consecutive + 1 if idle >= args.min_idle_percent else 0
        now = time.monotonic()
        if now - last_report >= 30 or consecutive == args.idle_consecutive:
            print(
                f"CPU-IDLE {label}: {idle:.1f}% "
                f"({consecutive}/{args.idle_consecutive} samples >= "
                f"{args.min_idle_percent:.1f}%)",
                flush=True,
            )
            last_report = now
        if args.idle_timeout_seconds and now - started > args.idle_timeout_seconds:
            raise TimeoutError(f"CPU idle guard timed out before {label}")


def run_adc(
    dataset: base.Dataset, rerank: int, sidecar: Path, args: argparse.Namespace
) -> Path:
    family = adc_family(rerank)
    out = output_csv(dataset, family)
    index_path = (
        args.index_root
        / dataset.key
        / f"{dataset.key}.fpgraph_adcbq.n{base.matrix_shape(dataset.train)[0]}.d{base.matrix_shape(dataset.train)[1]}.M24.efC200.faiss"
        if args.index_root is not None
        else dataset.adcbq_index
    )
    expected_nq, _ = base.matrix_shape(dataset.query)
    if adc_complete(out, rerank, expected_nq) and sidecar.exists() and not args.force:
        print(f"SKIP complete {out}", flush=True)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    def command_for(output: Path, rounds: int) -> list[str]:
        return [
            str(ADCBQ_BIN),
            "--dataset", dataset.key,
            "--train", str(dataset.train),
            "--query", str(dataset.query),
            "--gt", str(dataset.gt),
            "--index", str(index_path),
            "--out", str(output),
            "--m", "24",
            "--ef-construction", "200",
            "--k", "10",
            "--rerank-n", str(rerank),
            "--max-queries", "0",
            "--rounds", str(rounds),
            "--build-threads", str(args.build_threads),
            "--search-threads", str(SEARCH_THREADS),
            "--adc-mode", "int16",
            "--lut-mode", "nibble",
            "--graph-build", "fp",
            "--distance-mode", "adcbq",
            "--search-impl", "inline",
            "--code-layout", "vector",
            "--rerank-dot", "fp32",
            "--fp-rerank-source", "odirect-uring",
            "--fp-rerank-norms", "onfetch",
            "--fp-rerank-path", str(sidecar),
            "--fp-rerank-overlap-queries",
            "--fp-rerank-qd", str(QD),
            "--efs", *map(str, EFS),
        ]

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(SEARCH_THREADS),
            "OPENBLAS_NUM_THREADS": str(SEARCH_THREADS),
            "MKL_NUM_THREADS": str(SEARCH_THREADS),
        }
    )
    wait_for_idle(args, f"{dataset.key} ADC-BQ rerank@{rerank}")
    command = command_for(out, ROUNDS)
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)
    validate_aligned_fp(sidecar, dataset)
    if not adc_complete(out, rerank, expected_nq):
        raise RuntimeError(f"ADC-BQ output failed invariant checks: {out}")
    return out


def import_rabitq():
    sys.path.insert(0, str(RABITQ_BINDING_DIR))
    from _rabitqlib import HnswIndex  # type: ignore

    return HnswIndex


def run_rabitq(
    dataset: base.Dataset, sidecar: Path, args: argparse.Namespace
) -> Path:
    out = output_csv(dataset, B1_FAMILY)
    expected_nq, _ = base.matrix_shape(dataset.query)
    if rabitq_complete(out, expected_nq) and not args.force:
        print(f"SKIP complete {out}", flush=True)
        return out
    validate_aligned_fp(sidecar, dataset)
    wait_for_idle(args, f"{dataset.key} RaBitQ 1-bit adaptive")

    HnswIndex = import_rabitq()
    load_start = time.perf_counter()
    index = HnswIndex.load(str(dataset.b1_index))
    load_seconds = time.perf_counter() - load_start
    if index.nbits != 1 or index.dim != base.matrix_shape(dataset.train)[1]:
        raise ValueError(f"incompatible RaBitQ index: {dataset.b1_index}")

    queries = base.read_fbin(dataset.query)
    base.normalize_rows(queries)
    gt = base.read_ibin(dataset.gt, len(queries))
    rows: list[dict[str, object]] = []
    for ef in EFS:
        recalls: list[float] = []
        times: list[float] = []
        fp_means: list[float] = []
        fp_mins: list[int] = []
        fp_maxs: list[int] = []
        byte_counts: list[int] = []
        request_counts: list[int] = []
        offsets: list[int] = []
        for round_id in range(ROUNDS):
            start = time.perf_counter()
            ids, _, stats = index.search_1bit_adaptive_fp_odirect(
                queries,
                str(sidecar),
                k=10,
                ef=ef,
                num_threads=SEARCH_THREADS,
            )
            elapsed = time.perf_counter() - start
            offset = int(stats["fbin_data_offset"])
            if offset != PAGE_BYTES:
                raise RuntimeError(f"RaBitQ used fbin offset {offset}, expected 4096")
            fp_reads = np.asarray(stats["fp_read_count"], dtype=np.uint64)
            recall = base.recall_at_10(np.asarray(ids), gt)
            reads = int(stats["odirect_read_requests"])
            direct_bytes = int(stats["odirect_read_bytes"])
            recalls.append(recall)
            times.append(elapsed)
            fp_means.append(float(np.mean(fp_reads)))
            fp_mins.append(int(np.min(fp_reads)))
            fp_maxs.append(int(np.max(fp_reads)))
            byte_counts.append(direct_bytes)
            request_counts.append(reads)
            offsets.append(offset)
            print(
                f"[{dataset.key}] RaBitQ aligned4k ef={ef} "
                f"round={round_id + 1}/{ROUNDS} recall={recall:.6f} "
                f"qps={len(queries) / elapsed:.2f} fp_reads={np.mean(fp_reads):.2f} "
                f"pages/q={direct_bytes / PAGE_BYTES / len(queries):.2f}",
                flush=True,
            )
        mean_seconds = float(np.mean(times))
        mean_bytes = float(np.mean(byte_counts))
        mean_requests = float(np.mean(request_counts))
        rows.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method": "RaBitQ v0.5.1 1-bit + adaptive FP O_DIRECT aligned4k",
                "library_commit": base.RABITQ_COMMIT,
                "binding_path": RABITQ_BINDING_DIR,
                "storage": "SSD O_DIRECT",
                "io_engine": "sync pread O_DIRECT",
                "fp_store_layout": "header4096_rows_contiguous",
                "fbin_data_offset": min(offsets),
                "M": 24,
                "efConstruction": 200,
                "efSearch": ef,
                "k": 10,
                "nq": len(queries),
                "nb": index.max_elements,
                "dim": index.dim,
                "bits": 1,
                "rerank_policy": "adaptive_lower_bound",
                "rerank_n": "adaptive",
                "fp_reads_mean": float(np.mean(fp_means)),
                "fp_reads_min": min(fp_mins),
                "fp_reads_max": max(fp_maxs),
                "direct_bytes_mean": mean_bytes,
                "direct_bytes_per_query": mean_bytes / len(queries),
                "direct_pages_per_query": mean_bytes / PAGE_BYTES / len(queries),
                "direct_requests_mean": mean_requests,
                "direct_requests_per_query": mean_requests / len(queries),
                "rounds": ROUNDS,
                "search_threads": SEARCH_THREADS,
                "load_seconds": load_seconds,
                "total_seconds_mean": mean_seconds,
                "recall_at_k": float(np.mean(recalls)),
                "qps": len(queries) / mean_seconds,
                "index_bytes": dataset.b1_index.stat().st_size,
                "index_path": dataset.b1_index,
                "fp_rerank_path": sidecar,
            }
        )
    base.write_rows(out, rows)
    if not rabitq_complete(out, expected_nq):
        raise RuntimeError(f"RaBitQ output failed invariant checks: {out}")
    del index, queries, gt
    gc.collect()
    return out


def combine(selected: list[base.Dataset]) -> Path:
    combined: list[dict[str, object]] = []
    for dataset in selected:
        for family in FAMILIES:
            path = output_csv(dataset, family)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                nq = int(row["nq"])
                seconds = float(row["total_seconds_mean"])
                if family == B1_FAMILY:
                    fp_reads = float(row["fp_reads_mean"])
                    direct_bytes = float(row["direct_bytes_mean"])
                    direct_pages = float(row["direct_pages_per_query"])
                    rerank_n: object = "adaptive"
                    io_engine = "sync pread O_DIRECT"
                    offset = int(row["fbin_data_offset"])
                    ready_ratio = 0.0
                else:
                    rerank_n = int(row["rerank_n"])
                    fp_reads = float(rerank_n)
                    direct_bytes = float(row["direct_submitted_bytes_mean"])
                    direct_pages = direct_bytes / PAGE_BYTES / nq
                    io_engine = f"io_uring inter-query overlap QD={QD}"
                    offset = PAGE_BYTES
                    ready_ratio = float(row["pipeline_ready_ratio"])
                combined.append(
                    {
                        "dataset": dataset.key,
                        "corpus": dataset.corpus,
                        "model": dataset.model,
                        "title": dataset.title,
                        "family": family,
                        "method": METHOD_LABELS[family],
                        "storage": "SSD O_DIRECT",
                        "io_engine": io_engine,
                        "fp_store_layout": "header4096_rows_contiguous",
                        "fbin_data_offset": offset,
                        "M": 24,
                        "efConstruction": 200,
                        "efSearch": int(row["efSearch"]),
                        "k": 10,
                        "nq": nq,
                        "nb": int(row["nb"]),
                        "dim": int(row["dim"]),
                        "search_threads": SEARCH_THREADS,
                        "rerank_n": rerank_n,
                        "fp_reads_mean": fp_reads,
                        "direct_bytes_mean": direct_bytes,
                        "direct_pages_per_query": direct_pages,
                        "pipeline_ready_ratio": ready_ratio,
                        "rounds": int(row["rounds"]),
                        "recall_at_10": float(row["recall_at_k"]),
                        "qps": float(row["qps"]),
                        "latency_ms_per_query": seconds * 1000.0 / nq,
                        "index_path": row["index_path"],
                        "fp_rerank_path": row["fp_rerank_path"],
                        "source_csv": path,
                    }
                )
    out = RESULT_DIR / "all_results.csv"
    base.write_rows(out, combined)
    return out


def target_recall_table(combined: Path) -> Path:
    with combined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary: list[dict[str, object]] = []
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for target in (0.90, 0.95, 0.99):
            for family in FAMILIES:
                eligible = [
                    row for row in dataset_rows
                    if row["family"] == family and float(row["recall_at_10"]) >= target
                ]
                if not eligible:
                    continue
                best = max(eligible, key=lambda row: float(row["qps"]))
                summary.append(
                    {
                        "dataset": dataset,
                        "corpus": best["corpus"],
                        "model": best["model"],
                        "target_recall": target,
                        "family": family,
                        "method": METHOD_LABELS[family],
                        "efSearch": int(best["efSearch"]),
                        "recall_at_10": float(best["recall_at_10"]),
                        "qps": float(best["qps"]),
                    }
                )
    out = RESULT_DIR / "target_recall_summary.csv"
    base.write_rows(out, summary)
    return out


def plot(combined: Path, selected: list[base.Dataset]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with combined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    styles = {
        adc_family(50): ("#d62728", "s", "-"),
        adc_family(80): ("#ff7f0e", "D", "--"),
        B1_FAMILY: ("#2ca02c", "^", "-."),
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for corpus in ("Quora", "NQ"):
        datasets = [item for item in selected if item.corpus == corpus]
        fig, axes = plt.subplots(1, 4, figsize=(20.8, 5.4), squeeze=False)
        for axis, dataset in zip(axes.flat, datasets):
            for family, (color, marker, linestyle) in styles.items():
                group = sorted(
                    (row for row in rows if row["dataset"] == dataset.key and row["family"] == family),
                    key=lambda row: int(row["efSearch"]),
                )
                xs = [float(row["recall_at_10"]) for row in group]
                ys = [float(row["qps"]) for row in group]
                axis.plot(
                    xs, ys, color=color, marker=marker, linestyle=linestyle,
                    linewidth=2, markersize=5, label=METHOD_LABELS[family]
                )
                for x, y, row in zip(xs, ys, group):
                    axis.annotate(
                        row["efSearch"], (x, y), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=color
                    )
            axis.set_title(dataset.model)
            axis.set_xlabel("Recall@10")
            axis.set_ylabel("QPS (log scale)")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(
            f"{corpus}: ADC-BQ rerank@50/@80 vs RaBitQ adaptive FP\n"
            f"same 4 KiB-header FP32 store, all queries, T={SEARCH_THREADS}, "
            "M=24, efC=200",
            fontsize=15, y=0.99,
        )
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
            ncol=3, frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        for extension in ("png", "pdf"):
            path = FIGURE_DIR / f"{corpus.lower()}_recall_qps.{extension}"
            fig.savefig(path, dpi=190 if extension == "png" else None, bbox_inches="tight")
            print(f"WROTE {path}", flush=True)
            created.append(path)
        plt.close(fig)
    return created


def write_summary(combined: Path, selected: list[base.Dataset]) -> Path:
    with combined.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# NQ / Quora aligned-4KiB rerank@50/@80 benchmark",
        "",
        f"All {len(selected)} datasets use every query, {ROUNDS} timed rounds per "
        f"efSearch point, and {SEARCH_THREADS} search thread.",
        "",
        "ADC-BQ uses int16/nibble traversal, O_DIRECT + io_uring QD=80, and "
        "inter-query CPU/SSD overlap. RaBitQ uses synchronous O_DIRECT and "
        "lower-bound adaptive FP reranking. All methods read the same FP32 "
        "sidecar with data offset 4096.",
        "",
        "| Dataset | Method | efSearch | Recall@10 | QPS | FP/q | pages/q |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | {row['method']} | {row['efSearch']} | "
            f"{float(row['recall_at_10']):.6f} | {float(row['qps']):.2f} | "
            f"{float(row['fp_reads_mean']):.2f} | "
            f"{float(row['direct_pages_per_query']):.2f} |"
        )
    out = RESULT_DIR / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"WROTE {out}", flush=True)
    return out


def write_manifest(selected: list[base.Dataset], args: argparse.Namespace) -> Path:
    manifest = {
        "datasets": [item.key for item in selected],
        "query_mode": "all",
        "efSearch": list(EFS),
        "rounds": ROUNDS,
        "search_threads": SEARCH_THREADS,
        "adc_reranks": list(RERANKS),
        "adc_qd": QD,
        "search_warmup_rounds": 0,
        "sidecar_preparation": "atomic_copy_fsync_posix_fadvise_dontneed",
        "fp_store_data_offset": PAGE_BYTES,
        "rabitq_commit": base.RABITQ_COMMIT,
        "rabitq_binding": str(RABITQ_BINDING_DIR),
        "adc_binary": str(ADCBQ_BIN),
        "min_cpu_idle_percent": args.min_idle_percent,
    }
    out = RESULT_DIR / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    global FIGURE_DIR, FAMILIES, RESULT_DIR, RERANKS, SEARCH_THREADS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=None,
    )
    parser.add_argument("--build-threads", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--search-threads", type=int, default=SEARCH_THREADS)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--reranks", type=int, nargs="+", choices=(50, 80), default=list(RERANKS))
    parser.add_argument("--skip-rabitq", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--keep-aligned-fp", action="store_true")
    parser.add_argument("--aligned-fp-root", type=Path, default=ALIGNED_FP_ROOT)
    parser.add_argument(
        "--index-root", type=Path, default=None,
        help="Optional dedicated ADC-BQ index root; missing indexes are built with --build-threads.",
    )
    parser.add_argument("--reuse-aligned-fp", action="store_true")
    parser.add_argument("--min-idle-percent", type=float, default=85.0)
    parser.add_argument("--idle-consecutive", type=int, default=3)
    parser.add_argument("--idle-sample-seconds", type=float, default=1.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.search_threads <= 0:
        parser.error("--search-threads must be positive")
    SEARCH_THREADS = args.search_threads
    RESULT_DIR = args.result_dir.resolve()
    FIGURE_DIR = args.figure_dir.resolve()
    RERANKS = tuple(dict.fromkeys(args.reranks))
    FAMILIES = tuple(adc_family(value) for value in RERANKS)
    if not args.skip_rabitq:
        FAMILIES += (B1_FAMILY,)

    if not ADCBQ_BIN.exists():
        raise FileNotFoundError(ADCBQ_BIN)
    if not any(RABITQ_BINDING_DIR.glob("_rabitqlib*.so")):
        raise FileNotFoundError(RABITQ_BINDING_DIR)
    available = target_datasets()
    selected = available_datasets(args.datasets)
    for dataset in selected:
        base.validate(dataset)
        if B1_FAMILY in FAMILIES and not dataset.b1_index.exists():
            raise FileNotFoundError(f"missing RaBitQ index for {dataset.key}")
        if args.index_root is None and not dataset.adcbq_index.exists():
            raise FileNotFoundError(f"missing ADC-BQ index for {dataset.key}")
    write_manifest(selected, args)

    if not args.plot_only:
        for number, dataset in enumerate(selected, start=1):
            print(f"=== DATASET {number}/{len(selected)}: {dataset.key} ===", flush=True)
            if dataset_complete(dataset) and not args.force:
                print(f"SKIP fully complete {dataset.key}", flush=True)
                continue
            sidecar = aligned_fp_path(dataset, args.aligned_fp_root)
            completed = False
            try:
                prepare_aligned_fp(sidecar, dataset, args.reuse_aligned_fp)
                for rerank in RERANKS:
                    run_adc(dataset, rerank, sidecar, args)
                if B1_FAMILY in FAMILIES:
                    run_rabitq(dataset, sidecar, args)
                if not dataset_complete(dataset):
                    raise RuntimeError(f"incomplete dataset after runs: {dataset.key}")
                completed = True
            finally:
                if (
                    completed
                    and not args.keep_aligned_fp
                    and sidecar.exists()
                ):
                    size = sidecar.stat().st_size
                    sidecar.unlink()
                    print(f"REMOVED transient aligned FP sidecar ({size} bytes): {sidecar}", flush=True)

    combined = combine(selected)
    target_recall_table(combined)
    plot(combined, selected)
    write_summary(combined, selected)


if __name__ == "__main__":
    main()
