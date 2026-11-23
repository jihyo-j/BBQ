#!/usr/bin/env python3
"""Query bit-width sweep for calibrated ADC-BQ with a true integer query path.

Same protocol as run_nq_quora_query_bitwidth_sweep_t1.py (same B-bit query
files, seed-11 split, 1,000 calibration queries, target 0.99, efSearch 256,
one thread, original FP32-query ground truth), except that for B in
{2, 4, 8, 16} the int16 nibble LUT is built from the B-bit integer query codes
with integer arithmetic only (binary faiss_adcbq_hnsw_intquery, flag
--int-query-bits B).  FP32 (B = 32) runs the unchanged FP32 LUT path.

The FP32 rerank still uses the B-bit query (dequantized), exactly as in the
original sweep, so the only difference between the two sweeps is how the LUT
is built.  RaBitQ is not rerun (its query path is unchanged).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))

import run_nq_quora_query_bitwidth_sweep_t1 as sweep  # noqa: E402  (re-execs under python3.11)

adcbq_runner = sweep.adcbq_runner
calibration = sweep.calibration

RESULT_ROOT = EXPERIMENT / "nq_quora_qwen1024_query_bitwidth_int_path_t1"
BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_intquery"
SOURCE = EXPERIMENT / "faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp"
OLD_ROOT = sweep.RESULT_ROOT
BITS = sweep.BITS
CURRENT_BITS = {"bits": 32}


def int_query_scales(query_path: Path, bits: int) -> np.ndarray:
    """LUT units per FP distance unit, identical to the C++ integer path."""
    q = adcbq_runner.read_fbin(query_path).astype(np.float64)
    levels = (1 << bits) - 1
    max_abs = np.abs(q).max(axis=1, keepdims=True)
    step = 2.0 * max_abs / levels
    c = np.rint((q + max_abs) / step)
    if np.max(np.abs((q + max_abs) / step - c)) > 0.05:
        raise RuntimeError(f"{query_path} is not a {bits}-bit query file")
    qi = 2 * c - levels
    unit = (max_abs[:, 0] / levels).astype(np.float32)
    l1 = np.abs(qi).reshape(len(qi), -1, 4).sum(axis=2).max(axis=1).astype(np.int64)
    sh = np.zeros(len(qi), dtype=np.int64)
    while True:
        over = ((l1 + ((1 << sh) >> 1)) >> sh) > 32767
        if not over.any():
            break
        sh[over] += 1
    return (1.0 / (unit * (2.0 ** sh).astype(np.float32))).astype(np.float32)


_orig_common_command = adcbq_runner.common_command


def common_command(*args, **kwargs):
    command = _orig_common_command(*args, **kwargs)
    if CURRENT_BITS["bits"] < 32:
        command += ["--int-query-bits", str(CURRENT_BITS["bits"])]
    return command


def query_scales(mode, query_path: Path) -> np.ndarray:
    if CURRENT_BITS["bits"] < 32:
        return int_query_scales(query_path, CURRENT_BITS["bits"])
    rows, _ = calibration.matrix_header(query_path)
    return calibration.int16_query_scales(query_path) if mode.uses_query_scale else np.ones(rows, np.float32)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=sweep.DEFAULT_CPU)
    parser.add_argument("--datasets", nargs="+", default=("nq_qwen06_1024", "quora_qwen1024"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not BINARY.exists() or BINARY.stat().st_mtime_ns < SOURCE.stat().st_mtime_ns:
        raise RuntimeError(f"build {BINARY} from {SOURCE} first (adcbq_optimized/build_intquery.sh)")
    adcbq_runner.BINARY = BINARY
    adcbq_runner.common_command = common_command
    adcbq_runner.query_scales = query_scales
    adcbq_runner.run = sweep.run_adcbq
    sweep.ADCBQ_ROOT = RESULT_ROOT / "adcbq"

    dataset_map = sweep.datasets()
    summaries: list[dict[str, object]] = []
    per_query: list[dict[str, object]] = []
    for key in args.datasets:
        dataset = dataset_map[key]
        for bits in BITS:
            CURRENT_BITS["bits"] = bits
            print(f"INT-PATH SWEEP {key} {sweep.bit_label(bits)}", flush=True)
            split = sweep.prepare_split(dataset, bits)          # reuses the original query files
            assert str(split.files.test_query).startswith(str(OLD_ROOT / "queries"))
            summary, recalls = sweep.adcbq_measure(dataset, bits, split, args.cpu, args.force)
            summary["method"] = "calibrated_adcbq_int_query_path" if bits < 32 else "calibrated_adcbq"
            summary["lut_build"] = "integer from B-bit query codes" if bits < 32 else "FP32 LUT + int16 rounding"
            summaries.append(summary)
            per_query.extend(
                {
                    "dataset": key,
                    "query_bits": bits,
                    "query_precision": sweep.bit_label(bits),
                    "split_query_id": qi,
                    "original_query_id": int(split.test_ids[qi]),
                    "recall_at_10": float(v),
                }
                for qi, v in enumerate(recalls)
            )

    # comparison against the original (dequantize -> FP32 LUT) sweep, same queries
    old_rows = sweep.read_csv(OLD_ROOT / "per_query_recall_t1.csv")
    old = {
        (r["dataset"], int(r["query_bits"]), int(r["split_query_id"])): float(r["recall_at_10"])
        for r in old_rows
        if r["method"] == "calibrated_adcbq"
    }
    old_summary = {
        (r["dataset"], int(r["query_bits"])): r
        for r in sweep.read_csv(OLD_ROOT / "summary_t1.csv")
        if r["method"] == "calibrated_adcbq"
    }
    new = {(r["dataset"], r["query_bits"], r["split_query_id"]): r["recall_at_10"] for r in per_query}
    compare: list[dict[str, object]] = []
    for s in summaries:
        key, bits = s["dataset"], int(s["query_bits"])
        ids = sorted(q for (d, b, q) in new if d == key and b == bits)
        a = np.array([old[(key, bits, q)] for q in ids])
        b = np.array([new[(key, bits, q)] for q in ids])
        o = old_summary[(key, bits)]
        compare.append(
            {
                "dataset": key,
                "query_bits": bits,
                "queries": len(ids),
                "recall_fp32_lut_path": float(o["recall_at_10"]),
                "recall_int_query_path": float(s["recall_at_10"]),
                "recall_diff_int_minus_fp": float(s["recall_at_10"]) - float(o["recall_at_10"]),
                "qps_fp32_lut_path": float(o["qps"]),
                "qps_int_query_path": float(s["qps"]),
                "refine_mean_fp32_lut_path": float(o["refinement_count_mean"]),
                "refine_mean_int_query_path": float(s["refinement_count_mean"]),
                "tau_fp32_lut_path": float(o["tau"]),
                "tau_int_query_path": float(s["tau"]),
                "per_query_same_recall_fraction": float(np.mean(a == b)),
                "per_query_int_better_fraction": float(np.mean(b > a)),
                "per_query_int_worse_fraction": float(np.mean(b < a)),
            }
        )
    sweep.write_csv(RESULT_ROOT / "summary_t1.csv", summaries)
    sweep.write_csv(RESULT_ROOT / "per_query_recall_t1.csv", per_query)
    sweep.write_csv(RESULT_ROOT / "int_vs_fp32_lut_path_t1.csv", compare)
    sweep.write_json(
        RESULT_ROOT / "manifest.json",
        {
            "protocol": "same as nq_quora_qwen1024_query_bitwidth_sweep_t1 (BBQ only); LUT built from integer query codes",
            "binary": str(BINARY),
            "source": str(SOURCE),
            "query_files": str(OLD_ROOT / "queries"),
            "query_bits": list(BITS),
            "efSearch": sweep.EF,
            "threads": 1,
            "rerank_query": "B-bit query (dequantized), as in the original sweep",
        },
    )
    for c in compare:
        print(
            f"{c['dataset']:16s} {c['query_bits']:2d}b  recall fp-LUT {c['recall_fp32_lut_path']:.4f} "
            f"int {c['recall_int_query_path']:.4f}  qps {c['qps_fp32_lut_path']:.0f} -> {c['qps_int_query_path']:.0f}  "
            f"same/better/worse {c['per_query_same_recall_fraction']:.3f}/"
            f"{c['per_query_int_better_fraction']:.3f}/{c['per_query_int_worse_fraction']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
