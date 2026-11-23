#!/usr/bin/env python3
"""NQ Qwen-1024: BBQ (as is) vs a no-LUT scorer with an int16 query and 1-bit vectors.

  (A) BBQ                 : FP32 query -> int16 4-dim/16-entry nibble LUT (production path)
  (B) int16 query, no LUT : query quantized to int16 (per query, max|q| -> 32767), 1-bit code
                            scored directly by exact int32 accumulation (AVX-512 sign-bit masks,
                            same kernel structure as the FP32-direct no-LUT ablation)

Everything else is identical and taken from run_nq_quora_qwen1024_adcbq_lut_ablation_t1.py:
same FP-built M24/efC200 graph and 1-bit codes, same seed-11 split (1,000 calibration /
2,452 test queries), same per-ef pool-top10 split-conformal calibration (target 0.99),
same FP32 RAM rerank, one search thread, 3 rounds (QPS = median).  The two modes are run
back-to-back at each ef on the same pinned CPU.  Binary: adcbq_optimized/faiss_adcbq_hnsw_intquery
(source faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp; mode (B) = --adc-mode fp32-direct
--direct-int16-query; mode (A) uses no extra flag and is the unchanged LUT path).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT))

import run_nq_quora_qwen1024_adcbq_lut_ablation_t1 as abl  # noqa: E402

RESULT_ROOT = EXPERIMENT / "nq_qwen1024_bbq_vs_int16query_nolut_t1"
BINARY = EXPERIMENT / "adcbq_optimized/faiss_adcbq_hnsw_intquery"
SOURCE = EXPERIMENT / "faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp"
EFS = (80, 128, 256, 512, 1024)

BBQ = abl.Mode("bbq_int16_nibble_lut", "BBQ (int16 nibble LUT)", "int16", "nibble")
NOLUT = abl.Mode("int16_query_no_lut", "int16 query, 1-bit, no LUT", "fp32-direct", "nibble")

_orig_common = abl.common_command


def common_command(dataset, mode, *args, **kwargs):
    command = _orig_common(dataset, mode, *args, **kwargs)
    if mode.key == NOLUT.key:
        command.append("--direct-int16-query")
    return command


def query_scales(mode, query_path: Path) -> np.ndarray:
    if mode.key == NOLUT.key:   # int units per FP unit, identical to the C++ side
        q = abl.read_fbin(query_path)
        m = np.abs(q).max(axis=1)
        return np.divide(32767.0, m, out=np.ones_like(m), where=m > 0).astype(np.float32)
    return abl.calibration.int16_query_scales(query_path)


def main() -> None:
    if not BINARY.exists() or BINARY.stat().st_mtime_ns < SOURCE.stat().st_mtime_ns:
        raise SystemExit(f"build {BINARY} first (adcbq_optimized/build_intquery.sh)")
    abl.BINARY = BINARY
    abl.SOURCE = SOURCE
    abl.RESULT_ROOT = RESULT_ROOT
    abl.MODES = (BBQ, NOLUT)
    abl.common_command = common_command
    abl.query_scales = query_scales
    abl.ensure_binary = lambda: None
    abl.save_comparisons = lambda rows: (RESULT_ROOT / "summary_t1.csv",) * 2
    abl.plot = lambda rows: None
    sys.argv = [sys.argv[0], "--datasets", "nq_qwen06_1024", "--efs", *map(str, EFS), *sys.argv[1:]]
    abl.main()

    import csv

    with open(RESULT_ROOT / "summary_t1.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by = {(r["method_key"], int(r["efSearch"])): r for r in rows}
    out = []
    for ef in EFS:
        a, b = by[(BBQ.key, ef)], by[(NOLUT.key, ef)]
        out.append(
            {
                "efSearch": ef,
                "bbq_recall_at_10": a["recall_at_10"],
                "nolut_int16q_recall_at_10": b["recall_at_10"],
                "bbq_qps_median": a["qps_median"],
                "nolut_int16q_qps_median": b["qps_median"],
                "bbq_over_nolut_qps": float(a["qps_median"]) / float(b["qps_median"]),
                "bbq_rerank_mean": a["rerank_count_mean"],
                "nolut_int16q_rerank_mean": b["rerank_count_mean"],
                "bbq_tau": a["tau"],
                "nolut_int16q_tau": b["tau"],
            }
        )
    abl.write_csv(RESULT_ROOT / "paired_ef_comparison_t1.csv", out)
    for r in out:
        print(
            f"ef={r['efSearch']:5d}  recall BBQ {float(r['bbq_recall_at_10']):.4f} / noLUT-int16q "
            f"{float(r['nolut_int16q_recall_at_10']):.4f}   QPS {float(r['bbq_qps_median']):8.1f} / "
            f"{float(r['nolut_int16q_qps_median']):8.1f}  (x{r['bbq_over_nolut_qps']:.2f})   "
            f"R {float(r['bbq_rerank_mean']):.1f} / {float(r['nolut_int16q_rerank_mean']):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
