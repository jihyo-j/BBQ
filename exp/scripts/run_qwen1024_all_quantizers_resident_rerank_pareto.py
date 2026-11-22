#!/usr/bin/env python3
"""Resident-code efSearch x SSD FP-refinement sweep for Qwen-1024.

The common FP-built M=24, efConstruction=200 HNSW graph and every method's
compressed database representation stay resident in DRAM.  Only exact FP32
refinement uses 4 KiB O_DIRECT reads.  Fixed-refinement methods sweep both
efSearch and R; the official RaBitQ c16 baseline uses its native adaptive FP
policy and sweeps efSearch.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import run_faiss_hnsw_bq_budget_paged_codes as base


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/qwen1024_all_quantizers_resident_rerank_results"
E2E_RESULT_ROOT = (
    REPO / "experiment/qwen1024_all_quantizers_resident_e2e_t1_results"
)
FIXED_COMBINED = RESULT_ROOT / "qwen1024_fixed_resident_ef_rerank_sweep_t24.csv"
RABITQ_ROOT = RESULT_ROOT / "adaptive_rabitq"

DATASETS = ("msmarco_qwen1024", "nq_qwen1024")
FIXED_METHODS = (
    "adcbq1",
    "symbq1",
    "pq8m128",
    "opq8m128",
    "qdranttq1",
    "sq4",
    "sq8",
    "rabitq1",
)
DEFAULT_FIXED_METHODS = FIXED_METHODS[:-1]
RERANKS = {
    "adcbq1": (10, 20, 30, 40, 60, 80, 120),
    "symbq1": (40, 60, 80, 120, 160, 200),
    "pq8m128": (10, 20, 30, 40, 60, 80, 120),
    "opq8m128": (10, 20, 30, 40, 60, 80, 120),
    "qdranttq1": (20, 30, 40, 60, 80),
    "sq4": (10, 20),
    "sq8": (10, 20),
    "rabitq1": (10, 20, 30, 40, 60, 80, 120),
}
DEFAULT_EFS = (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
FP_FALLBACK = {
    "msmarco_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/"
        "aligned_fp/msmarco_qwen1024.fp32.header4096.fbin"
    ),
    "nq_qwen1024": Path(
        "/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/"
        "aligned_fp/nq_qwen06_1024.fp32.header4096.fbin"
    ),
}


def selected_result_root(args: argparse.Namespace) -> Path:
    if args.result_root is not None:
        return args.result_root
    return E2E_RESULT_ROOT if args.query_e2e else RESULT_ROOT


def output_path(
    dataset: str,
    method: str,
    rerank_n: int,
    args: argparse.Namespace,
) -> Path:
    return selected_result_root(args) / "fixed" / dataset / method / f"r{rerank_n}.csv"


def complete(
    path: Path,
    efs: list[int],
    rounds: int,
    fp_io_mode: str,
    minimum_qd: int,
    query_e2e: bool,
) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        (str(ef), str(round_id))
        for ef in efs
        for round_id in range(1, rounds + 1)
    }
    actual = {(row.get("efSearch", ""), row.get("round", "")) for row in rows}
    return actual == expected and all(
        row.get("code_storage") == "resident"
        and row.get("fp_io_mode") == fp_io_mode
        and int(row.get("fp_rerank_qd", "0")) >= minimum_qd
        and (
            row.get("measurement_mode") == "query_e2e"
            if query_e2e
            else row.get("measurement_mode", "bulk_phased") == "bulk_phased"
        )
        and (not query_e2e or int(row.get("query_batch_size", "0")) == 1)
        and (
            not query_e2e
            or (
                bool(row.get("latency_csv"))
                and Path(row["latency_csv"]).exists()
            )
        )
        for row in rows
    )


def run_fixed_one(
    dataset_key: str,
    method: str,
    rerank_n: int,
    args: argparse.Namespace,
) -> Path:
    dataset = base.DATASETS[dataset_key]
    fp_sidecar = dataset.fp_sidecar if dataset.fp_sidecar.exists() else FP_FALLBACK[dataset_key]
    codec, meta = base.codec_paths(dataset_key, method)
    sidecar = base.SIDECAR_ROOT / dataset_key / f"{dataset_key}.{method}.pages4k"
    output = output_path(dataset_key, method, rerank_n, args)
    rerank_qd = max(rerank_n, args.fp_rerank_qd)
    if not args.force and complete(
        output,
        args.efs,
        args.rounds,
        args.fp_io_mode,
        rerank_qd if args.fp_io_mode == "uring-batch" else 0,
        args.query_e2e,
    ):
        print(f"REUSE {output}", flush=True)
        return output

    required = (dataset.query, dataset.gt, dataset.graph, fp_sidecar, codec)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if method in ("qdranttq1", "rabitq1") and not meta.exists():
        raise FileNotFoundError(meta)

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(base.BINARY),
        "--dataset", dataset_key,
        "--method", method,
        "--metric", "ip",
        "--query", str(dataset.query),
        "--gt", str(dataset.gt),
        "--graph", str(dataset.graph),
        "--codec-index", str(codec),
        "--sidecar", str(sidecar),
        "--fp-sidecar", str(fp_sidecar),
        "--out", str(output),
        "--max-queries", str(args.max_queries),
        "--rounds", str(args.rounds),
        "--threads", str(args.threads),
        "--rerank-n", str(rerank_n),
        "--resident-codes",
        "--fp-io-mode", args.fp_io_mode,
        "--fp-rerank-qd", str(rerank_qd),
        "--efs", *[str(ef) for ef in args.efs],
    ]
    if method in ("qdranttq1", "rabitq1"):
        command += ["--meta", str(meta)]
    if args.query_e2e:
        command.append("--query-e2e")
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REPO,
        env={
            **os.environ,
            "OMP_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=True,
    )
    return output


def combine_fixed(
    outputs: list[tuple[str, str, int, Path]],
    args: argparse.Namespace,
) -> Path:
    rows: list[dict[str, str]] = []
    fields = ["sweep_rerank_n", "sweep_method", "source_csv"]
    for dataset, method, rerank_n, path in outputs:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {
                    "sweep_rerank_n": str(rerank_n),
                    "sweep_method": method,
                    "source_csv": str(path),
                    **raw,
                }
                rows.append(row)
                for field in row:
                    if field not in fields:
                        fields.append(field)
    combined = (
        selected_result_root(args)
        / f"qwen1024_fixed_resident_ef_rerank_sweep_t{args.threads}.csv"
    )
    combined.parent.mkdir(parents=True, exist_ok=True)
    with combined.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {combined} ({len(rows)} rows)", flush=True)
    return combined


def run_rabitq(args: argparse.Namespace) -> Path:
    import run_qwen1024_c16_adaptive_rabitq_t24 as rabitq

    output_root = selected_result_root(args) / "adaptive_rabitq"
    rabitq.RESULT_ROOT = output_root
    paths = [rabitq.run(rabitq.DATASETS[key], args) for key in args.datasets]
    rabitq.combine(paths, args.threads)
    return output_root / (
        f"qwen1024_rabitq1_c16_adaptive_t{args.threads}_combined.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=FIXED_METHODS,
        default=list(DEFAULT_FIXED_METHODS),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument(
        "--fp-io-mode",
        choices=("sync", "uring", "uring-batch"),
        default="sync",
    )
    parser.add_argument(
        "--fp-rerank-qd",
        type=int,
        default=0,
        help="minimum io_uring queue depth; uring-batch raises it to at least R",
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--parts", nargs="+", choices=("fixed", "rabitq"), default=["fixed", "rabitq"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--query-e2e",
        action="store_true",
        help="measure strict T=1, query-batch-1 end-to-end latency",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query_e2e and args.threads != 1:
        raise SystemExit("--query-e2e requires --threads 1")
    if args.query_e2e and "rabitq" in args.parts:
        raise SystemExit(
            "adaptive RaBitQ uses a separate benchmark path; run this pass with "
            "--parts fixed (the fixed-R rabitq1 method is available via --methods)"
        )
    if "fixed" in args.parts:
        base.compile_binary()
        outputs: list[tuple[str, str, int, Path]] = []
        for dataset in args.datasets:
            for method in args.methods:
                for rerank_n in RERANKS[method]:
                    path = run_fixed_one(dataset, method, rerank_n, args)
                    outputs.append((dataset, method, rerank_n, path))
                    combine_fixed(outputs, args)
    if "rabitq" in args.parts:
        run_rabitq(args)


if __name__ == "__main__":
    main()
