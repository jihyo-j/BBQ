#!/usr/bin/env python3
"""Generate exp/claims.csv: every paper number -> (file, unique row selector, column, expected, tolerance)."""
import csv, sys
from pathlib import Path

EXP = Path(sys.argv[1])
# id, section, text, file(rel. exp/), csv line (1-based, header=1), column, paper value as printed, scale
C = [
 # ---- Sec 4 / Fig 5
 ("F5.1", "Fig 5", "ADC-BQ query LUT build, MS MARCO", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 2, "query_preprocess_us", "4.7"),
 ("F5.2", "Fig 5", "TurboQuant query preprocessing, MS MARCO", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 3, "query_preprocess_us", "9.6"),
 ("F5.3", "Fig 5", "ADC-BQ exact-FP I/O wait (R=120), MS MARCO", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 2, "exact_fp_io_wait_us", "517.6"),
 ("F5.4", "Fig 5", "TurboQuant exact-FP I/O wait (R=80), MS MARCO", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 3, "exact_fp_io_wait_us", "280"),
 ("S4.1", "Sec 4", "ADC-BQ QPS @R>=0.99, MS MARCO", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 2, "qps", "349"),
 ("S4.2", "Sec 4", "TurboQuant QPS @R>=0.99, MS MARCO", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 3, "qps", "362"),
 ("S4.3", "Sec 4", "ADC-BQ QPS @R>=0.99, NQ", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 7, "qps", "984"),
 ("S4.4", "Sec 4", "TurboQuant QPS @R>=0.99, NQ", "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv", 8, "qps", "1063"),
 # ---- Fig 8 / Table 4
 ("F8.1", "Fig 8", "FP32-built graph Recall@10, MS MARCO ef=128", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 11, "fp_recall_at_10", "0.8015"),
 ("F8.2", "Fig 8", "1-bit-built graph Recall@10, MS MARCO ef=128", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 11, "binary_recall_at_10", "0.8010"),
 ("F8.3", "Fig 8", "FP32-built graph QPS, MS MARCO ef=128", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 11, "fp_qps", "2982"),
 ("F8.4", "Fig 8", "1-bit-built graph QPS, MS MARCO ef=128", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 11, "binary_qps", "2715"),
 ("F8.5", "Fig 8", "FP32-built graph QPS, Quora ef=128", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 31, "fp_qps", "3744"),
 ("F8.6", "Fig 8", "1-bit-built graph QPS, Quora ef=128", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 31, "binary_qps", "3415"),
 ("T4.1", "Table 4", "MS MARCO distance evals ef=256, FP32 graph", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 15, "fp_ndis_mean", "5221"),
 ("T4.2", "Table 4", "MS MARCO distance evals ef=256, 1-bit graph", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv", 15, "binary_ndis_mean", "6136"),
 ("T4.3", "Table 4", "MS MARCO build time, FP32 graph (s)", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_build_times.csv", 2, "index_build_seconds", "141.9"),
 ("T4.4", "Table 4", "MS MARCO build time, 1-bit graph (s)", "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_build_times.csv", 3, "index_build_seconds", "32.7"),
 # ---- Fig 9
 ("F9.1", "Fig 9", "BBQ QPS, MS MARCO @0.9926", "fig09_vs_inmemory_fp_hnsw/results/paper_run_20261109/recall_qps_t1_from_recall096.csv", 5, "qps", "635"),
 ("F9.2", "Fig 9", "in-memory FP32 HNSW QPS, MS MARCO @0.9926", "fig09_vs_inmemory_fp_hnsw/results/paper_run_20261109/recall_qps_t1_from_recall096.csv", 22, "qps", "335"),
 ("F9.3", "Fig 9", "BBQ QPS, DBpedia @0.9875", "fig09_vs_inmemory_fp_hnsw/results/paper_run_20261109/recall_qps_t1_from_recall096.csv", 28, "qps", "1547"),
 ("F9.4", "Fig 9", "in-memory FP32 HNSW QPS, DBpedia @0.9875", "fig09_vs_inmemory_fp_hnsw/results/paper_run_20261109/recall_qps_t1_from_recall096.csv", 41, "qps", "727"),
 ("F9.5", "Sec 6.2", "BBQ distance evals, DBpedia ef=1024", "fig09_vs_inmemory_fp_hnsw/results/rerun_ndis_20261120/dbpedia_matched_recall_summary.csv", 2, "distance_evaluations_mean", "22977"),
 ("F9.6", "Sec 6.2", "FP32 HNSW distance evals, DBpedia ef=256", "fig09_vs_inmemory_fp_hnsw/results/rerun_ndis_20261120/dbpedia_matched_recall_summary.csv", 3, "distance_evaluations_mean", "6834"),
 # ---- Fig 10
 ("F10.1", "Sec 6.3", "TurboQuant QPS @>=0.99, DBpedia BGE-1024", "fig10_overall_recall_qps/results/recall_qps_12methods_16datasets_t1.csv", 2676, "qps", "1869"),
 ("F10.2", "Sec 6.3", "BBQ QPS @>=0.99, DBpedia BGE-1024", "fig10_overall_recall_qps/results/recall_qps_12methods_16datasets_t1.csv", 2652, "qps", "1196"),
 ("F10.3", "Sec 6.3", "MRQ FP reads/query @0.9903, NQ BGE-M3", "fig10_overall_recall_qps/results/mrq_nq_bge_m3_C4096_B320_direct.csv", 12, "ssd_reads_per_query", "1657.6"),
 # ---- Fig 11
 ("F11.1", "Fig 11", "BBQ QPS @Recall@30>=0.99", "fig11_recall_at_k/results/nq_qwen1024_recall_k30_50_100_t1.csv", 7, "qps", "933"),
 ("F11.2", "Fig 11", "TurboQuant QPS @Recall@30>=0.99", "fig11_recall_at_k/results/nq_qwen1024_recall_k30_50_100_t1.csv", 49, "qps", "841"),
 ("F11.3", "Fig 11", "BBQ QPS @Recall@50>=0.99", "fig11_recall_at_k/results/nq_qwen1024_recall_k30_50_100_t1.csv", 67, "qps", "578"),
 ("F11.4", "Fig 11", "TurboQuant QPS @Recall@50>=0.99", "fig11_recall_at_k/results/nq_qwen1024_recall_k30_50_100_t1.csv", 106, "qps", "580"),
 ("F11.5", "Fig 11", "BBQ QPS @Recall@100>=0.99", "fig11_recall_at_k/results/nq_qwen1024_recall_k30_50_100_t1.csv", 131, "qps", "303"),
 # ---- Fig 12
 ("F12.1", "Fig 12a", "sym-BQ Recall@10, ef=256", "fig12_ablation/results/msmarco_qwen1024_recall_ablation_t1.csv", 11, "recall_at_10", "0.7231"),
 ("F12.2", "Fig 12a", "ADC-BQ Recall@10, ef=256", "fig12_ablation/results/msmarco_qwen1024_recall_ablation_t1.csv", 23, "recall_at_10", "0.8030"),
 ("F12.3", "Fig 12a", "ADC-BQ + fixed R=75 Recall@10, ef=256", "fig12_ablation/results/msmarco_qwen1024_recall_ablation_t1.csv", 37, "recall_at_10", "0.9893"),
 ("F12.4", "Fig 12a", "BBQ calibrated Recall@10, ef=256", "fig12_ablation/results/msmarco_qwen1024_recall_ablation_t1.csv", 50, "recall_at_10", "0.9897"),
 ("F12.5", "Fig 12a", "BBQ calibrated FP reads/query, ef=256", "fig12_ablation/results/msmarco_qwen1024_recall_ablation_t1.csv", 50, "fp_vectors_per_query_mean", "54.0"),
 ("F12.6", "Fig 12a", "fixed R=75 QPS (equal pool/QD)", "fig12_ablation/results/equal_pool_qd_ablation/equal_pool_qd_fixed75_vs_calibrated.csv", 11, "fixed_qps_median", "1109"),
 ("F12.7", "Fig 12a", "calibrated QPS (equal pool/QD)", "fig12_ablation/results/equal_pool_qd_ablation/equal_pool_qd_fixed75_vs_calibrated.csv", 11, "calibrated_qps_median", "1191"),
 ("F12.8", "Fig 12b", "share of queries with Recall@10=1.0 (%)", "fig12_ablation/results/per_query_recall_groups_target099.ip_aligned.csv", 2, "query_percent", "91.79"),
 ("F12.9", "Fig 12b", "FP reads/query, Recall@10=1.0 group", "fig12_ablation/results/per_query_recall_groups_target099.ip_aligned.csv", 2, "mean_fp_reads", "51.5"),
 ("F12.10", "Fig 12b", "FP reads/query, Recall@10=0.9 group", "fig12_ablation/results/per_query_recall_groups_target099.ip_aligned.csv", 3, "mean_fp_reads", "75.4"),
 ("F12.11", "Fig 12b", "FP reads/query, Recall@10<=0.8 group", "fig12_ablation/results/per_query_recall_groups_target099.ip_aligned.csv", 4, "mean_fp_reads", "119.8"),
 # ---- Fig 13
 ("F13.1", "Fig 13b", "mean FP reads at target 0.90", "fig13_calibration_analysis/results/exact_cosine_aligned/target_sweep.csv", 2, "mean_fp_reads", "28.3"),
 ("F13.2", "Fig 13b", "mean FP reads at target 0.995", "fig13_calibration_analysis/results/exact_cosine_aligned/target_sweep.csv", 6, "mean_fp_reads", "58.6"),
 ("F13.3", "Fig 13b", "pool top-10 preserved at target 0.90", "fig13_calibration_analysis/results/exact_cosine_aligned/target_sweep.csv", 2, "test_empirical_pool_top10_coverage", "0.9117"),
 ("F13.4", "Fig 13b", "pool top-10 preserved at target 0.995", "fig13_calibration_analysis/results/exact_cosine_aligned/target_sweep.csv", 6, "test_empirical_pool_top10_coverage", "0.9953"),
 ("F13.5", "Sec 6.5", "queries whose pool misses part of global top-10 (of 5980)", "fig13_calibration_analysis/results/exact_cosine_aligned/error_decomposition_target099.csv", 5, "queries", "307"),
 # ---- Fig 14
 ("F14.1", "Fig 14", "TurboQuant+calibration QPS, MS MARCO @0.9853", "fig14_bbq_vs_tq_same_calibration/results/summary_t1.csv", 74, "qps", "2060"),
 ("F14.2", "Fig 14", "BBQ QPS, MS MARCO @0.9853", "fig14_bbq_vs_tq_same_calibration/results/summary_t1.csv", 54, "qps", "1809"),
 ("F14.3", "Fig 14", "BBQ QPS, NQ @0.9976", "fig14_bbq_vs_tq_same_calibration/results/summary_t1.csv", 20, "qps", "651"),
 ("F14.4", "Fig 14", "TurboQuant FP reads/query, NQ", "fig14_bbq_vs_tq_same_calibration/results/summary_t1.csv", 41, "fp_reads_per_query", "42.1"),
 ("F14.5", "Fig 14", "BBQ FP reads/query, NQ", "fig14_bbq_vs_tq_same_calibration/results/summary_t1.csv", 20, "fp_reads_per_query", "61.3"),
 # ---- Fig 15
 *[(f"F15.{i+1}", "Fig 15", t, "fig15_page_io/results/page_io/operating_points.csv", ln, "pages_per_query", v) for i, (t, ln, v) in enumerate([
   ("BBQ pages/query, OpenAI-1536 @0.98", 2, "64.5"), ("TurboQuant pages/query, OpenAI-1536 @0.98", 3, "120.0"),
   ("DiskANN-RaBitQ pages/query, OpenAI-1536 @0.98", 4, "184.4"), ("DiskANN pages/query, OpenAI-1536 @0.98", 5, "184.2"),
   ("BBQ pages/query, OpenAI-1536 @0.99", 10, "67.6"), ("DiskANN-RaBitQ pages/query, OpenAI-1536 @0.99", 12, "342.4"),
   ("DiskANN pages/query, OpenAI-1536 @0.99", 13, "373.9"), ("BBQ pages/query, Qwen-1024 @0.98", 6, "46.2"),
   ("BBQ pages/query, Qwen-1024 @0.99", 14, "54.3"), ("DiskANN-RaBitQ pages/query, Qwen-1024 @0.99", 16, "913.6"),
   ("DiskANN pages/query, Qwen-1024 @0.99", 17, "1042.0")])],
 # ---- Fig 16
 *[(f"F16.{i+1}", "Fig 16", t, "fig16_io_modes/results/qwen1024_summary_t1.csv", ln, "qps", v) for i, (t, ln, v) in enumerate([
   ("Sequential QPS, Qwen-1024", 53, "222.0"), ("Async Batch QPS, Qwen-1024", 54, "928.0"),
   ("Async Batch + Overlap QPS, Qwen-1024", 55, "993.5"), ("Sequential QPS, OpenAI-1536", 71, "376.9"),
   ("Async Batch QPS, OpenAI-1536", 72, "1898.3"), ("Async Batch + Overlap QPS, OpenAI-1536", 73, "2322.8")])],
 # ---- Fig 17 / 18
 ("F17.1", "Fig 17", "BBQ QPS @>=0.995, 512D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl512/summary_t1.csv", 19, "qps", "1178"),
 ("F17.2", "Fig 17", "BBQ QPS @>=0.995, 4096D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl4096/summary_t1.csv", 10, "qps", "972"),
 ("F17.3", "Fig 17", "BBQ FP reads/query @>=0.995, 512D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl512/summary_t1.csv", 19, "fp_vectors_per_query", "97.5"),
 ("F17.8", "Fig 17", "BBQ FP reads/query @>=0.995, 4096D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl4096/summary_t1.csv", 10, "fp_vectors_per_query", "23.4"),
 ("F17.4", "Fig 17", "BBQ Recall@10 at ef=128, 512D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl512/summary_t1.csv", 10, "recall_at_10", "0.9868"),
 ("F17.5", "Fig 17", "BBQ Recall@10 at ef=128, 4096D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl4096/summary_t1.csv", 10, "recall_at_10", "0.9952"),
 ("F17.6", "Fig 17", "BBQ max Recall@10, 4096D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl4096/summary_t1.csv", 21, "recall_at_10", "0.9988"),
 ("F17.7", "Fig 17", "TurboQuant max Recall@10, 4096D", "fig17_18_dimension_scale/results/nq_mrl/nq_qwen8_mrl4096/summary_t1.csv", 41, "recall_at_10", "0.9990"),
 ("F18.1", "Fig 18", "BBQ QPS @>=0.99, 10M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_10m/summary_t1.csv", 20, "qps", "563"),
 ("F18.2", "Fig 18", "BBQ QPS @>=0.99, 50M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_50m/summary_t1.csv", 20, "qps", "541"),
 ("F18.3", "Fig 18", "BBQ QPS @>=0.99, 100M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_100m/summary_t1.csv", 20, "qps", "539"),
 ("F18.4", "Fig 18", "BBQ FP reads/query @>=0.99, 10M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_10m/summary_t1.csv", 20, "fp_vectors_per_query", "51.7"),
 ("F18.5", "Fig 18", "BBQ FP reads/query @>=0.99, 100M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_100m/summary_t1.csv", 20, "fp_vectors_per_query", "71.5"),
 ("F18.6", "Fig 18", "BBQ max Recall@10, 100M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_100m/summary_t1.csv", 21, "recall_at_10", "0.9944"),
 ("F18.7", "Fig 18", "RaBitQ-7bit max Recall@10, 100M", "fig17_18_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_100m/summary_t1.csv", 81, "recall_at_10", "0.9849"),
 # ---- Fig 19
 ("F19.1", "Fig 19", "BBQ cumulative insertion time, 10M (s)", "fig19_insertion/results/frozen_adcbq_vs_rabitq_c16_comparison.csv", 12, "adc_online_add_seconds", "521.7"),
 ("F19.2", "Fig 19", "BBQ cumulative insertion time, 100M (s)", "fig19_insertion/results/frozen_adcbq_vs_rabitq_c16_comparison.csv", 42, "adc_online_add_seconds", "5762.2"),
 ("F19.3", "Fig 19", "RaBitQ-1bit cumulative insertion time, 10M (s)", "fig19_insertion/results/rabitq_b1_adaptive_c16_random_independent_t32.csv", 7, "add_seconds", "1073.5"),
 ("F19.4", "Fig 19", "RaBitQ-1bit cumulative insertion time, 100M (s)", "fig19_insertion/results/rabitq_b1_adaptive_c16_random_independent_t32.csv", 37, "add_seconds", "8568.5"),
 ("F19.5", "Fig 19", "BBQ Recall@10 after 100M inserts", "fig19_insertion/results/frozen_adcbq_vs_rabitq_c16_comparison.csv", 44, "adc_recall_at_10", "0.9853"),
 ("F19.6", "Fig 19", "RaBitQ-1bit Recall@10 after 100M inserts", "fig19_insertion/results/rabitq_b1_adaptive_c16_random_independent_t32.csv", 39, "recall_current_at_k", "0.9842"),
 ("F19.7", "Fig 19", "TurboQuant Recall@10 after 100M inserts", "fig19_insertion/results/tq1plus_r60_random_independent_t32.csv", 24, "recall_current_at_k", "0.9766"),
 # ---- Fig 20
 ("F20.1", "Fig 20", "16KB baseline QPS at 50% DRAM", "fig20_colocation/results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv", 22, "qps", "112.2"),
 ("F20.2", "Fig 20", "16KB colocated QPS at 50% DRAM", "fig20_colocation/results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv", 23, "qps", "170.6"),
 ("F20.3", "Fig 20", "16KB baseline graph page hit ratio at 50% DRAM", "fig20_colocation/results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv", 22, "graph_page_hit_ratio", "0.576"),
 ("F20.4", "Fig 20", "16KB colocated graph page hit ratio at 50% DRAM", "fig20_colocation/results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv", 23, "graph_page_hit_ratio", "0.751"),
]

KEYS = ["dataset", "dataset_key", "dataset_title", "dataset_label", "method_key", "method", "family", "series", "mode",
        "io_mode", "stage", "policy", "rerank_policy", "layout", "page_size", "page_bytes", "page_kb", "memory_ratio",
        "memory_percent", "target", "target_recall", "calibration_target", "k", "efSearch", "calibration_reference_ef", "ef_search", "rerank_n",
        "fixed_r", "inserted_vectors", "inserted", "total_vectors", "scale", "step", "per_query_recall_at_10",
        "decomposition", "component", "category", "graph_build", "dim", "recall_at_10"]


def selector(rows, idx):
    row = rows[idx]
    cols = [k for k in KEYS if k in row] + [k for k in row if k not in KEYS]
    chosen = []
    for c in cols:
        cand = chosen + [c]
        if any(all(r[x] == row[x] for x in chosen) for j, r in enumerate(rows) if j != idx):
            # still ambiguous; only add columns that reduce ambiguity
            before = sum(all(r[x] == row[x] for x in chosen) for r in rows)
            after = sum(all(r[x] == row[x] for x in cand) for r in rows)
            if after < before:
                chosen = cand
        if sum(all(r[x] == row[x] for x in chosen) for r in rows) == 1:
            return ";".join(f"{c}={row[c]}" for c in chosen)
    raise SystemExit(f"cannot build a unique selector for row {idx + 2}")


out = []
for cid, sec, text, f, line, col, printed in C:
    rows = list(csv.DictReader(open(EXP / f, newline="")))
    r = rows[line - 2]
    if col not in r:
        raise SystemExit(f"{cid}: column {col} missing in {f}")
    tol = 10 ** -(len(printed.split(".")[1]) if "." in printed else 0)  # one unit of the printed last digit
    out.append(dict(id=cid, section=sec, claim=text, file=f, select=selector(rows, line - 2), column=col,
                    expected=printed, tolerance=f"{tol:g}"))
with open(EXP / "claims.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]), quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(out)
print(f"{len(out)} claims written")
