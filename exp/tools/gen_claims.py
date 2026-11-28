#!/usr/bin/env python3
"""Generate exp/claims.csv for the manuscript named in exp/claims.source.

Every claim says where its number lives; this script resolves that to a unique row selector and checks it.
Claim kinds:
  R   (id, sec, text, file, {col: val}, column, printed[, agg])   a cell (or min/max/... over matched rows)
  BI  (id, sec, text, file, {filter}, recall_col, th, column, printed, key_col)
                                                                 the max-QPS row with recall >= th, then `column`
  B   (id, sec, text, summary, dataset, method, th, column, printed)   a 'best' row of full16_summary_*.csv
  M   (id, sec, text, summary, dataset, method, printed)               a 'max_recall' row
  G   (id, sec, text, summary, metric, printed)                        an 'agg' row
  REF (id, sec, text, ...)       like R / BI / B but with printed=None: measured only, used inside D
  D   (id, sec, text, expression, printed)                             arithmetic over other claim ids
Printed values keep the paper's precision; tolerance = one unit of the last printed digit. A trailing 'K'
multiplies by 1,000 (tolerance scaled too).
"""
import csv
import sys
from pathlib import Path

EXP = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
T1 = "fig09_20_overall_t1/results/full16_summary_t1.csv"          # the table the paper's Figs 9/20 were drawn from
T1R = "fig09_20_overall_t1/results/full16_summary_rerun_t1.csv"   # the later rerun of four curves
T1S = "fig09_20_overall_t1/results/with_spann_es/full16_summary_t1.csv"
T24 = "fig21_overall_t24/results/full16_summary_t24.csv"
PR = "fig09_20_overall_t1/results/recall_qps_as_printed_t1.csv"
B, TQ, FP, R7, RSSD = ("BBQ (calibrated ADC-BQ)", "TurboQuant TQ1+", "FP32 HNSW RAM", "RaBitQ 7-bit (RAM)",
                       "RaBitQ-SSD B9")
DPQ, DRBQ, PIPE, PQ, OPQ = "DiskANN-PQ d/8", "DiskANN-RaBitQ 1-bit", "PipeANN-RaBitQ 1-bit", "Faiss PQ d/8", "Faiss OPQ-PQ d/8"
F5 = "fig05_preprocessing_io/results/qwen1024_five_method_recall099_timing_t1.csv"
F7 = "fig07_adc_lut/results/quora_bge_m3/paired_ef_comparison_t1.csv"
F8 = "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv"
F8B = "fig08_graph_build_precision/results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_build_times.csv"
F10 = "fig10_recall_at_k/results/nq_recall_k30_50_100_with_inmemory_fp.csv"
F11 = "fig11_ablation/results/msmarco_qwen1024_recall_ablation_t1.csv"
F11E = "fig11_ablation/results/equal_pool_qd_ablation/equal_pool_qd_fixed75_vs_calibrated.csv"
F11G = "fig11_ablation/results/per_query_recall_groups_target099.legacy_gt.csv"   # plot/make_recall_groups.py
F12 = "fig12_calibration/results/legacy_gt/target_sweep.csv"
F12L = "fig12_calibration/results/legacy_gt/loss_split_target099.csv"
F13 = "fig13_bbq_vs_tq_same_calibration/results/summary_t1.csv"
F14 = "fig14_page_io/results/page_io/operating_points.csv"
F15 = "fig15_io_modes/results/qwen1024_summary_t1.csv"
F16 = "fig16_17_dimension_scale/results/nq_mrl/nq_qwen8_mrl{}/summary_t1.csv"
F17 = "fig16_17_dimension_scale/results/msmarco100m/msmarco_qwen1024_v2_{}/summary_t1.csv"
F18 = "fig18_insertion/results/frozen_adcbq_vs_rabitq_c16_comparison.csv"
F18R = "fig18_insertion/results/rabitq_b1_adaptive_c16_random_independent_t32.csv"
F18T = "fig18_insertion/results/tq1plus_r60_random_independent_t32.csv"
F19 = "fig19_colocation/results/msmarco_te3small1536_adcbq_colocation_ef256_t24/msmarco_te3small1536_colocation_summary.csv"
S1 = "sec1_hnsw_vs_ivf/results/first_measured_point_at_recall099.csv"
S3 = "sec2_3_analysis/results/sec3_values.csv"
S3D = "sec2_3_analysis/results/sec3_derived.csv"   # tools/sec3_derived.py
S51 = "sec5_1_query_bits/results/summary_t1.csv"
S63 = "sec6_3_traversal_efficiency/results/time_breakdown_20261124/matched_recall_breakdown.csv"

def sv(source, key):
    return {"source": source, "key": key}

C = [
 # ---------------- Sec 1 (Intro)
 ("R", "I.1", "Sec 1", "HNSW 1-bit distance evaluations at first point >= 0.99 (MS MARCO-1M)", S1, {"method": "HNSW"}, "onebit_distance_evals_per_query", "9.5K"),
 ("R", "I.2", "Sec 1", "HNSW Recall@10 at that point", S1, {"method": "HNSW"}, "recall_at_10", "0.9928"),
 ("R", "I.3", "Sec 1", "IVF codes scanned per query", S1, {"method": "IVF"}, "onebit_distance_evals_per_query", "778K"),
 ("R", "I.4", "Sec 1", "IVF nprobe (of 977 lists)", S1, {"method": "IVF"}, "parameter_value", "768"),
 ("R", "I.5", "Sec 1", "IVF Recall@10 at that point", S1, {"method": "IVF"}, "recall_at_10", "0.9959"),
 ("G", "I.6", "Sec 1", "configurations where BBQ beats TurboQuant-1bit @0.99", T1, "bbq_faster_than_tq_at_0.99", "14"),
 ("G", "I.7", "Sec 1", "max speedup over in-memory FP32 HNSW @0.99", T1, "bbq_over_fp_hnsw_max_speedup_at_0.99", "2.7"),
 ("R", "I.8", "Sec 1", "calibration reads fewer than fixed R=75, low end (%)", F11E, {"dataset": "msmarco_qwen1024"}, "read_reduction_pct", "26.0", "min"),
 ("R", "I.9", "Sec 1", "calibration reads fewer than fixed R=75, high end (%)", F11E, {"dataset": "msmarco_qwen1024"}, "read_reduction_pct", "40.5", "max"),
 ("D", "I.10", "Sec 1", "page-read reduction vs DiskANN-RaBitQ, Qwen-1024 @0.99", "F14.10/F14.9", "16.8"),
 ("D", "I.11", "Sec 1", "page-read reduction vs DiskANN, Qwen-1024 @0.99", "F14.11/F14.9", "19.2"),
 # ---------------- Sec 3
 ("R", "S3.1", "Sec 3.1", "sym-BQ CandidateRecall@30, MS MARCO BGE-M3", S3, sv("msmarco_bgem3.json", "arms.sym_bq.candrec.30"), "value", "0.848"),
 ("R", "S3.2", "Sec 3.1", "ADC-BQ CandidateRecall@30, MS MARCO BGE-M3", S3, sv("msmarco_bgem3.json", "arms.adc_bq.candrec.30"), "value", "0.947"),
 ("R", "S3.3", "Sec 3.1", "sym-BQ p95 R*, MS MARCO BGE-M3", S3, sv("msmarco_bgem3.json", "arms.sym_bq.Rstar_p95"), "value", "416"),
 ("R", "S3.4", "Sec 3.1", "ADC-BQ p95 R*, MS MARCO BGE-M3", S3, sv("msmarco_bgem3.json", "arms.adc_bq.Rstar_p95"), "value", "99"),
 ("R", "S3.5", "Sec 3.2", "kappa at d=32 (Qwen3-8B MRL)", S3, sv("dimsweep_msmarco_qw8.json", "rows[d=32].arms.adc_bq.kappa"), "value", "0.782"),
 ("R", "S3.6", "Sec 3.2", "kappa at d=4096", S3, sv("dimsweep_msmarco_qw8.json", "rows[d=4096].arms.adc_bq.kappa"), "value", "0.958"),
 ("R", "S3.7", "Sec 3.2", "beta at d=32", S3, sv("dimsweep_msmarco_qw8.json", "rows[d=32].beta"), "value", "0.17"),
 ("R", "S3.8", "Sec 3.2", "beta at d=4096", S3, sv("dimsweep_msmarco_qw8.json", "rows[d=4096].beta"), "value", "1.07"),
 ("R", "S3.9", "Sec 3.2", "sign-code alignment f, low end", S3, sv("dimsweep_msmarco_qw8.json", "rows[d=4096].f_sign_cos"), "value", "0.775"),
 ("R", "S3.10", "Sec 3.2", "sign-code alignment f, high end", S3, sv("dimsweep_msmarco_qw8.json", "rows[d=32].f_sign_cos"), "value", "0.802"),
 *[("R", f"T2.{i+1}", "Table 2", f"p95 R* at {b} B/vector, {arm}", S3, sv("bitsvsdims.txt", f"{b}B.{arm}.p95_Rstar"), "value", v)
   for i, (b, arm, v) in enumerate([(512, "BQ", "21"), (512, "PQ", "23"), (512, "SQ-4bit", "37"), (512, "SQ-8bit", "87"),
                                    (128, "BQ", "73"), (128, "PQ", "822"), (128, "SQ-4bit", "352"), (128, "SQ-8bit", "1596")])],
 ("R", "S3.11", "Sec 3.3", "median R*, MS MARCO BGE-M3", S3, sv("msmarco_bgem3.json", "arms.adc_bq.Rstar_med"), "value", "23"),
 *[("R", f"T3.{i+1}", "Table 3", f"p95 R*, MS MARCO {enc}, {arm}", S3, sv(f"msmarco_{f}.json", f"arms.{arm}.Rstar_p95"), "value", v)
   for i, (enc, f, arm, v) in enumerate([
       ("Qwen", "qw06", "sym_bq", "140"), ("Qwen", "qw06", "adc_bq", "47"), ("Qwen", "qw06", "adc_bq_c", "61"),
       ("Qwen", "qw06", "tq1", "43"), ("Qwen", "qw06", "rabitq1", "43"), ("Qwen", "qw06", "pq", "48"),
       ("Qwen", "qw06", "opq", "33"), ("Qwen", "qw06", "sq4", "14"),
       ("BGE-M3", "bgem3", "sym_bq", "416"), ("BGE-M3", "bgem3", "adc_bq", "99"), ("BGE-M3", "bgem3", "adc_bq_c", "148"),
       ("BGE-M3", "bgem3", "tq1", "112"), ("BGE-M3", "bgem3", "rabitq1", "89"), ("BGE-M3", "bgem3", "pq", "81"),
       ("BGE-M3", "bgem3", "opq", "44"), ("BGE-M3", "bgem3", "sq4", "15")])],
 # ---------------- Sec 3: remaining sentences (sources mapped by Sieun Kim, BBQETA)
 ("R", "S3.12", "Sec 3.1", "sym-BQ CandidateRecall@30, NQ BGE-M3", S3, sv("nq_bgem3.json", "arms.sym_bq.candrec.30"), "value", "0.865"),
 ("R", "S3.13", "Sec 3.1", "ADC-BQ CandidateRecall@30, NQ BGE-M3", S3, sv("nq_bgem3.json", "arms.adc_bq.candrec.30"), "value", "0.955"),
 ("R", "S3.14", "Sec 3.1", "sym-BQ p95 R*, NQ BGE-M3", S3, sv("nq_bgem3.json", "arms.sym_bq.Rstar_p95"), "value", "367"),
 ("R", "S3.15", "Sec 3.1", "ADC-BQ p95 R*, NQ BGE-M3", S3, sv("nq_bgem3.json", "arms.adc_bq.Rstar_p95"), "value", "84"),
 ("R", "S3.16", "Sec 3.1", "ADC-BQ CandidateRecall@30, GloVe", S3, sv("glove.json", "arms.adc_bq.candrec.30"), "value", "0.412"),
 ("R", "S3.17", "Sec 3.1", "ADC-BQ CandidateRecall@30, GIST", S3, sv("gist.json", "arms.adc_bq.candrec.30"), "value", "0.000"),
 ("R", "S3.18", "Sec 3.1", "Spearman(eta, p95 R*), pooled (configuration, quantizer)", S3D, sv("tables.py", "eta.paper.spearman.pooled"), "value", "0.921"),
 ("R", "S3.19", "Sec 3.1", "number of (configuration, quantizer) pairs", S3D, sv("tables.py", "eta.paper.n.pooled"), "value", "90"),
 ("R", "S3.20", "Sec 3.1", "Spearman(eta, p95 R*), ADC-BQ only", S3D, sv("tables.py", "eta.paper.spearman.adc_bq"), "value", "0.912"),
 *[("R", f"S3.{21+i}", "Sec 3.2", f"p95 R* at d={d} (Qwen3-8B MRL)", S3, sv("dimsweep_msmarco_qw8.json", f"rows[d={d}].arms.adc_bq.Rstar_p95"), "value", v)
   for i, (d, v) in enumerate([(128, "2534"), (512, "124"), (1024, "47"), (4096, "22")])],
 *[("R", f"S3.{25+i}", "Sec 3.2", t, S3, sv(f, k), "value", v) for i, (t, f, k, v) in enumerate([
     ("beta, Quora Qwen3-0.6B", "quora_qw06.json", "beta", "0.889"),
     ("beta, Quora BGE-M3", "quora_bgem3.json", "beta", "0.896"),
     ("p95 R*, Quora Qwen3-0.6B", "quora_qw06.json", "arms.adc_bq.Rstar_p95", "35"),
     ("p95 R*, Quora BGE-M3", "quora_bgem3.json", "arms.adc_bq.Rstar_p95", "29"),
     ("beta, MS MARCO Qwen3-0.6B", "msmarco_qw06.json", "beta", "0.825"),
     ("beta, MS MARCO BGE-M3", "msmarco_bgem3.json", "beta", "0.620"),
     ("p95 R*, MS MARCO Qwen3-0.6B", "msmarco_qw06.json", "arms.adc_bq.Rstar_p95", "47"),
     ("p95 R*, MS MARCO BGE-M3", "msmarco_bgem3.json", "arms.adc_bq.Rstar_p95", "99"),
     ("beta without query instruction, MS MARCO Qwen3-0.6B", "placebo.json", "msm/none(stored)[0]", "0.523"),
     ("p95 R* without query instruction, MS MARCO", "placebo.json", "msm/none(stored)[1]", "74"),
     ("beta without query instruction, Quora Qwen3-0.6B", "placebo.json", "quora/none[0]", "0.740"),
     ("p95 R* without query instruction, Quora", "placebo.json", "quora/none[1]", "30")])],
 *[("R", f"S3.{37+i}", "Sec 3.3", t, S3D, sv(f"{k}_perq.npz", f"perq.{k}.adc_bq.{q}"), "value", v) for i, (t, k, q, v) in enumerate([
     ("median R*, MS MARCO Qwen-1024", "msmarco_qw06", "median", "16"),
     ("p95 R*, MS MARCO Qwen-1024", "msmarco_qw06", "p95", "47"),
     ("p99 R*, MS MARCO Qwen-1024", "msmarco_qw06", "p99", "87"),
     ("p95 R*, MS MARCO BGE-M3 (per-query array)", "msmarco_bgem3", "p95", "99"),
     ("p99 R*, MS MARCO BGE-M3", "msmarco_bgem3", "p99", "195")])],
 ("REF", "S3R.1", "Sec 3.3", "query share of var(log R*), levels.py", S3D, sv("levels.py", "levels.paper.query"), "value"),
 ("REF", "S3R.2", "Sec 3.3", "embedding share of var(log R*), levels.py", S3D, sv("levels.py", "levels.paper.setting"), "value"),
 ("REF", "S3R.3", "Sec 3.3", "quantizer share of var(log R*), levels.py", S3D, sv("levels.py", "levels.paper.arm"), "value"),
 ("D", "S3.42", "Sec 3.3", "query share of var(log R*) (%)", "S3R.1*100", "89.4"),
 ("D", "S3.43", "Sec 3.3", "embedding share of var(log R*) (%)", "S3R.2*100", "7.7"),
 ("D", "S3.44", "Sec 3.3", "quantizer share of var(log R*) (%)", "S3R.3*100", "1.8"),
 # ---------------- Sec 4 / Fig 5
 ("R", "F5.1", "Fig 5", "ADC-BQ query LUT build, MS MARCO (us)", F5, {"dataset": "msmarco_qwen1024", "method_key": "adcbq1"}, "query_preprocess_us", "4.7"),
 ("R", "F5.2", "Fig 5", "TurboQuant query preprocessing, MS MARCO (us)", F5, {"dataset": "msmarco_qwen1024", "method_key": "qdranttq1"}, "query_preprocess_us", "9.6"),
 ("R", "F5.3", "Fig 5", "ADC-BQ exact-FP I/O wait, MS MARCO (us)", F5, {"dataset": "msmarco_qwen1024", "method_key": "adcbq1"}, "exact_fp_io_wait_us", "517.6"),
 ("R", "F5.4", "Fig 5", "TurboQuant exact-FP I/O wait, MS MARCO (us)", F5, {"dataset": "msmarco_qwen1024", "method_key": "qdranttq1"}, "exact_fp_io_wait_us", "280"),
 ("R", "S4.1", "Sec 4", "ADC-BQ QPS @0.99, MS MARCO", F5, {"dataset": "msmarco_qwen1024", "method_key": "adcbq1"}, "qps", "349"),
 ("R", "S4.2", "Sec 4", "TurboQuant QPS @0.99, MS MARCO", F5, {"dataset": "msmarco_qwen1024", "method_key": "qdranttq1"}, "qps", "362"),
 ("R", "S4.3", "Sec 4", "ADC-BQ QPS @0.99, NQ", F5, {"dataset": "nq_qwen1024", "method_key": "adcbq1"}, "qps", "984"),
 ("R", "S4.4", "Sec 4", "TurboQuant QPS @0.99, NQ", F5, {"dataset": "nq_qwen1024", "method_key": "qdranttq1"}, "qps", "1063"),
 # ---------------- Sec 5.1 / Fig 7
 ("R", "F7.1", "Sec 5.1", "8-bit query Recall@10, NQ ef=256", S51, {"dataset": "nq_qwen06_1024", "method": "calibrated_adcbq", "query_bits": "8"}, "recall_at_10", "0.993"),
 ("REF", "R7.1", "Sec 5.1", "FP32 query Recall@10, NQ ef=256", S51, {"dataset": "nq_qwen06_1024", "method": "calibrated_adcbq", "query_bits": "32"}, "recall_at_10"),
 ("D", "F7.2", "Sec 5.1", "8-bit vs FP32 query gap (pp)", "(R7.1-F7.1)*100", "0.2"),
 ("R", "F7.3", "Fig 7b", "scalar ADC QPS, Quora ef=256", F7, {"efSearch": "256"}, "no_lut_qps_median", "71.1"),
 ("R", "F7.4", "Fig 7b", "LUT ADC QPS, Quora ef=256", F7, {"efSearch": "256"}, "lut_qps_median", "1906.3"),
 ("R", "F7.5", "Fig 7b", "LUT speedup, Quora ef=256", F7, {"efSearch": "256"}, "lut_speedup", "26.8"),
 ("R", "F7.6", "Fig 7b", "LUT speedup, low end (ef 80-1024)", F7, {"dataset": "quora_bge_m3_vanilla"}, "lut_speedup", "20.5", "min"),
 ("R", "F7.7", "Fig 7b", "LUT speedup, high end", F7, {"dataset": "quora_bge_m3_vanilla"}, "lut_speedup", "29.4", "max"),
 # ---------------- Fig 8 / Table 4
 ("R", "F8.1", "Fig 8", "FP32-built Recall@10, MS MARCO ef=128", F8, {"dataset": "msmarco", "efSearch": "128"}, "fp_recall_at_10", "0.8015"),
 ("R", "F8.2", "Fig 8", "1-bit-built Recall@10, MS MARCO ef=128", F8, {"dataset": "msmarco", "efSearch": "128"}, "binary_recall_at_10", "0.8010"),
 ("R", "F8.3", "Fig 8", "FP32-built QPS, MS MARCO ef=128", F8, {"dataset": "msmarco", "efSearch": "128"}, "fp_qps", "2982"),
 ("R", "F8.4", "Fig 8", "1-bit-built QPS, MS MARCO ef=128", F8, {"dataset": "msmarco", "efSearch": "128"}, "binary_qps", "2715"),
 ("R", "F8.5", "Fig 8", "FP32-built Recall@10, Quora ef=128", F8, {"dataset": "quora", "efSearch": "128"}, "fp_recall_at_10", "0.8277"),
 ("R", "F8.6", "Fig 8", "1-bit-built Recall@10, Quora ef=128", F8, {"dataset": "quora", "efSearch": "128"}, "binary_recall_at_10", "0.8275"),
 ("R", "F8.7", "Fig 8", "FP32-built QPS, Quora ef=128", F8, {"dataset": "quora", "efSearch": "128"}, "fp_qps", "3744"),
 ("R", "F8.8", "Fig 8", "1-bit-built QPS, Quora ef=128", F8, {"dataset": "quora", "efSearch": "128"}, "binary_qps", "3415"),
 *[("R", f"T4.{i+1}", "Table 4", f"{ds} distance evals ef={ef}, {g}", F8, {"dataset": ds, "efSearch": ef}, col, v)
   for i, (ds, ef, g, col, v) in enumerate([
       ("msmarco", "256", "FP32", "fp_ndis_mean", "5221"), ("msmarco", "256", "1-bit", "binary_ndis_mean", "6136"),
       ("msmarco", "1024", "FP32", "fp_ndis_mean", "17433"), ("msmarco", "1024", "1-bit", "binary_ndis_mean", "20379"),
       ("quora", "256", "FP32", "fp_ndis_mean", "3716"), ("quora", "256", "1-bit", "binary_ndis_mean", "4491"),
       ("quora", "1024", "FP32", "fp_ndis_mean", "12153"), ("quora", "1024", "1-bit", "binary_ndis_mean", "14634")])],
 ("R", "T4.9", "Table 4", "MS MARCO build time, FP32 graph (s)", F8B, {"dataset": "msmarco", "graph_build": "fp"}, "index_build_seconds", "141.9"),
 ("R", "T4.10", "Table 4", "MS MARCO build time, 1-bit graph (s)", F8B, {"dataset": "msmarco", "graph_build": "binary"}, "index_build_seconds", "32.7"),
 ("R", "T4.11", "Table 4", "Quora build time, FP32 graph (s)", F8B, {"dataset": "quora", "graph_build": "fp"}, "index_build_seconds", "55.7"),
 ("R", "T4.12", "Table 4", "Quora build time, 1-bit graph (s)", F8B, {"dataset": "quora", "graph_build": "binary"}, "index_build_seconds", "10.5"),
 ("D", "D8.1", "Sec 5.2", "1-bit build speedup, MS MARCO", "T4.9/T4.10", "4.3"),
 ("D", "D8.2", "Sec 5.2", "1-bit build speedup, Quora", "T4.11/T4.12", "5.3"),
 ("D", "D8.3", "Sec 5.2", "extra distance evals of 1-bit graph, MS MARCO (%)", "(T4.2/T4.1-1)*100", "17"),
 ("D", "D8.4", "Sec 5.2", "extra distance evals of 1-bit graph, Quora (%)", "(T4.6/T4.5-1)*100", "20"),
 # ---------------- Sec 6.2 / Figs 9, 20 (the table the figures were drawn from)
 ("G", "F9.1", "Sec 6.2", "BBQ beats TurboQuant @0.99 (configurations)", T1, "bbq_faster_than_tq_at_0.99", "14"),
 ("G", "F9.2", "Sec 6.2", "configurations where both reach 0.995", T1, "both_reach_0.995_bbq_tq", "7"),
 ("G", "F9.3", "Sec 6.2", "...of which BBQ is faster ('all seven')", T1, "bbq_faster_than_tq_at_0.995", "7"),
 ("B", "F9.4", "Sec 6.2", "BBQ QPS @0.995, NQ Qwen", T1, "nq_qwen06_1024", B, "0.995", "value", "1275"),
 ("B", "F9.5", "Sec 6.2", "TurboQuant QPS @0.995, NQ Qwen", T1, "nq_qwen06_1024", TQ, "0.995", "value", "636"),
 ("D", "F9.6", "Sec 6.2", "BBQ / TurboQuant @0.995, NQ Qwen", "F9.4/F9.5", "2.00"),
 ("B", "F9.7", "Sec 6.2", "BBQ QPS @0.99, MS MARCO Qwen", T1, "msmarco_qwen1024", B, "0.99", "value", "1055"),
 ("B", "F9.8", "Sec 6.2", "TurboQuant QPS @0.99, MS MARCO Qwen", T1, "msmarco_qwen1024", TQ, "0.99", "value", "664"),
 ("D", "F9.9", "Sec 6.2", "BBQ / TurboQuant @0.99, MS MARCO Qwen", "F9.7/F9.8", "1.59"),
 ("G", "F9.10", "Sec 6.2", "mean FP32 reads, BBQ (where recorded)", T1, "mean_fp_reads_bbq_at_0.99", "41.8"),
 ("G", "F9.11", "Sec 6.2", "mean FP32 reads, TurboQuant", T1, "mean_fp_reads_tq_at_0.99", "76.0"),
 ("R", "F9.12", "Sec 6.2", "OPQ FP32 reads, Quora Qwen ef=80", PR, {"dataset": "quora_qwen1024", "method": OPQ, "search_parameter_value": "80"}, "fp_vectors_per_query", "30"),
 ("R", "F9.13", "Sec 6.2", "PQ FP32 reads, Quora Qwen ef=80", PR, {"dataset": "quora_qwen1024", "method": PQ, "search_parameter_value": "80"}, "fp_vectors_per_query", "80"),
 ("R", "F9.14", "Sec 6.2", "OPQ QPS, Quora Qwen ef=80", PR, {"dataset": "quora_qwen1024", "method": OPQ, "search_parameter_value": "80"}, "qps", "980"),
 ("R", "F9.15", "Sec 6.2", "PQ QPS, Quora Qwen ef=80", PR, {"dataset": "quora_qwen1024", "method": PQ, "search_parameter_value": "80"}, "qps", "1886"),
 ("M", "F9.16", "Sec 6.2", "PipeANN max Recall@10, MS MARCO Qwen", T1, "msmarco_qwen1024", PIPE, "0.9925"),
 ("M", "F9.17", "Sec 6.2", "BBQ max Recall@10, MS MARCO Qwen", T1, "msmarco_qwen1024", B, "0.9945"),
 ("B", "F9.18", "Sec 6.2", "DiskANN-PQ QPS @0.99, MS MARCO Qwen", T1, "msmarco_qwen1024", DPQ, "0.99", "value", "33"),
 ("B", "F9.19", "Sec 6.2", "DiskANN-RaBitQ QPS @0.99, MS MARCO Qwen", T1, "msmarco_qwen1024", DRBQ, "0.99", "value", "35"),
 ("B", "F9.20", "Sec 6.2", "PipeANN QPS @0.99, MS MARCO Qwen", T1, "msmarco_qwen1024", PIPE, "0.99", "value", "38"),
 ("B", "F9.21", "Sec 6.2", "RaBitQ-7bit QPS @0.99, Quora Qwen", T1, "quora_qwen1024", R7, "0.99", "value", "4198"),
 ("B", "F9.22", "Sec 6.2", "BBQ QPS @0.99, Quora Qwen", T1, "quora_qwen1024", B, "0.99", "value", "2803"),
 ("M", "F9.23", "Sec 6.2", "RaBitQ-7bit max Recall@10, Quora Qwen", T1, "quora_qwen1024", R7, "0.9948"),
 ("M", "F9.24", "Sec 6.2", "BBQ max Recall@10, Quora Qwen", T1, "quora_qwen1024", B, "0.9981"),
 ("G", "F9.25", "Sec 6.2", "configurations where RaBitQ-7bit never reaches 0.99", T1, "rabitq7_never_reaches_0.99", "3"),
 ("G", "F9.26", "Sec 6.2", "RaBitQ-SSD max recall higher than BBQ's (configurations)", T1, "rabitq_ssd_higher_max_recall_than_bbq", "15"),
 ("G", "F9.27", "Sec 6.2", "BBQ over RaBitQ-SSD @0.99, low end of '9-20x'", T1, "bbq_over_rabitq_ssd_min_at_0.99", "9"),
 ("G", "F9.28", "Sec 6.2", "BBQ over RaBitQ-SSD @0.99, high end of '9-20x'", T1, "bbq_over_rabitq_ssd_max_at_0.99", "20"),
 ("B", "R9.1", "Sec 6.2", "BBQ QPS @0.95, NQ Qwen", T1, "nq_qwen06_1024", B, "0.95", "value", None),
 ("R", "R9.2", "Sec 6.2", "MRQ best QPS @0.95, NQ Qwen", "fig09_20_overall_t1/results/mrq_rabitq_ssd_summary_t1.csv",
  {"dataset": "nq_qwen06_1024", "method_key": "mrq_ssd_bd", "search_parameter_value": None}, "qps", None),
 ("B", "R9.3", "Sec 6.2", "SPANN-PQ best QPS @0.95, NQ Qwen", T1S, "nq_qwen06_1024", "SPANN-PQ d/8", "0.95", "value", None),
 ("D", "F9.29", "Sec 6.2", "BBQ over MRQ @0.95, NQ Qwen", "R9.1/R9.2", "11.8"),
 ("D", "F9.30", "Sec 6.2", "BBQ over SPANN-PQ @0.95, NQ Qwen", "R9.1/R9.3", "3.3"),
 ("B", "F9.31", "Sec 6.2", "TurboQuant QPS @0.99, DBpedia BGE", T1, "sieun_dbpedia_entity_bge_large1024_beir400", TQ, "0.99", "value", "250"),
 ("B", "F9.32", "Sec 6.2", "BBQ QPS @0.99, DBpedia BGE", T1, "sieun_dbpedia_entity_bge_large1024_beir400", B, "0.99", "value", "181"),
 ("B", "F9.33", "Sec 6.2", "TurboQuant QPS @0.99, DBpedia E5", T1, "dbpedia_e5mistral4096", TQ, "0.99", "value", "201"),
 ("B", "F9.34", "Sec 6.2", "BBQ QPS @0.99, DBpedia E5", T1, "dbpedia_e5mistral4096", B, "0.99", "value", "182"),
 ("B", "F9.35", "Sec 6.2", "BBQ FP32 reads @0.99, DBpedia BGE", T1, "sieun_dbpedia_entity_bge_large1024_beir400", B, "0.99", "fp_vectors_per_query", "77.5"),
 ("B", "F9.36", "Sec 6.2", "TurboQuant FP32 reads @0.99, DBpedia BGE", T1, "sieun_dbpedia_entity_bge_large1024_beir400", TQ, "0.99", "fp_vectors_per_query", "200"),
 ("B", "F9.37", "Sec 6.2", "BBQ efSearch @0.99, DBpedia BGE", T1, "sieun_dbpedia_entity_bge_large1024_beir400", B, "0.99", "search_parameter_value", "1024"),
 ("B", "F9.38", "Sec 6.2", "TurboQuant efSearch @0.99, DBpedia BGE", T1, "sieun_dbpedia_entity_bge_large1024_beir400", TQ, "0.99", "search_parameter_value", "512"),
 ("G", "F9.39", "Sec 6.2", "BBQ beats in-memory FP32 HNSW @0.99 (configurations)", T1, "bbq_faster_than_fp_hnsw_at_0.99", "15"),
 ("G", "F9.40", "Sec 6.2", "speedup over FP32 HNSW, low end", T1, "bbq_over_fp_hnsw_min_speedup_at_0.99", "1.21"),
 ("G", "F9.41", "Sec 6.2", "speedup over FP32 HNSW, high end", T1, "bbq_over_fp_hnsw_max_speedup_at_0.99", "2.70"),
 *[x for i, (ds, b, f, r) in enumerate([("msmarco_qwen1024", "1055", "555", "1.90"), ("dbpedia_qwen1024", "959", "540", "1.78"),
                                        ("sieun_dbpedia_entity_openai1536", "1639", "608", "2.70")])
   for x in (("B", f"F9.{42+3*i}", "Sec 6.2", f"BBQ QPS @0.99, {ds}", T1, ds, B, "0.99", "value", b),
             ("B", f"F9.{43+3*i}", "Sec 6.2", f"FP32 HNSW QPS @0.99, {ds}", T1, ds, FP, "0.99", "value", f),
             ("D", f"F9.{44+3*i}", "Sec 6.2", f"BBQ / FP32 HNSW, {ds}", f"F9.{42+3*i}/F9.{43+3*i}", r))],
 ("G", "F9.51", "Sec 6.2", "max-recall gap to FP32 HNSW, largest (pp)", T1, "max_recall_gap_fp_minus_bbq_max_pp", "0.20"),
 ("G", "F9.52", "Sec 6.2", "max-recall gap to FP32 HNSW, mean (pp, 'below 0.10')", T1, "max_recall_gap_fp_minus_bbq_mean_pp", "0.10"),
 ("B", "R9.4", "Sec 6.2", "FP32 HNSW QPS @0.99, DBpedia BGE (for 0.92x)", T1, "sieun_dbpedia_entity_bge_large1024_beir400", FP, "0.99", "value", None),
 ("D", "F9.53", "Sec 6.2", "BBQ / FP32 HNSW, DBpedia BGE", "F9.32/R9.4", "0.92"),
 # ---------------- Fig 10 (recall@k, with in-memory FP32 HNSW)
 *[x for i, (k, b, t, fpr, mb, mf) in enumerate([("30", "933", "841", "1.60", "0.9984", "0.9986"),
                                                 ("50", "578", "580", "1.56", "0.9975", "0.9979"),
                                                 ("100", "303", "320", "1.26", "0.9955", "0.9970")])
   for x in (("BI", f"F10.{1+6*i}", "Fig 10", f"BBQ QPS @Recall@{k}>=0.99", F10, {"method_key": "adc_bq_calibrated", "k": k}, "recall_at_k", .99, "qps", b, "efSearch"),
             ("BI", f"F10.{2+6*i}", "Fig 10", f"TurboQuant QPS @Recall@{k}>=0.99", F10, {"method_key": "turboquant_tq1_fp", "k": k}, "recall_at_k", .99, "qps", t, "efSearch"),
             ("BI", f"R10.{1+i}", "Fig 10", f"FP32 HNSW QPS @Recall@{k}>=0.99", F10, {"method_key": "fp32_hnsw_ram", "k": k}, "recall_at_k", .99, "qps", None, "efSearch"),
             ("D", f"F10.{3+6*i}", "Fig 10", f"BBQ / FP32 HNSW @Recall@{k}>=0.99", f"F10.{1+6*i}/R10.{1+i}", fpr),
             ("R", f"F10.{4+6*i}", "Fig 10", f"BBQ max Recall@{k}", F10, {"method_key": "adc_bq_calibrated", "k": k}, "recall_at_k", mb, "max"),
             ("R", f"F10.{5+6*i}", "Fig 10", f"FP32 HNSW max Recall@{k}", F10, {"method_key": "fp32_hnsw_ram", "k": k}, "recall_at_k", mf, "max"))],
 ("R", "F10.19", "Fig 10", "TurboQuant max Recall@100", F10, {"method_key": "turboquant_tq1_fp", "k": "100"}, "recall_at_k", "0.9933", "max"),
 ("R", "F10.20", "Fig 10", "RaBitQ-1bit max Recall@100", F10, {"method_key": "rabitq_1bit_adaptive_fp", "k": "100"}, "recall_at_k", "0.9914", "max"),
 ("BI", "R10.4", "Fig 10", "BBQ QPS @Recall@30>=0.995", F10, {"method_key": "adc_bq_calibrated", "k": "30"}, "recall_at_k", .995, "qps", None, "efSearch"),
 ("BI", "R10.5", "Fig 10", "TurboQuant QPS @Recall@30>=0.995", F10, {"method_key": "turboquant_tq1_fp", "k": "30"}, "recall_at_k", .995, "qps", None, "efSearch"),
 ("BI", "R10.6", "Fig 10", "BBQ QPS @Recall@50>=0.995", F10, {"method_key": "adc_bq_calibrated", "k": "50"}, "recall_at_k", .995, "qps", None, "efSearch"),
 ("BI", "R10.7", "Fig 10", "TurboQuant QPS @Recall@50>=0.995", F10, {"method_key": "turboquant_tq1_fp", "k": "50"}, "recall_at_k", .995, "qps", None, "efSearch"),
 ("D", "F10.21", "Fig 10", "BBQ / TurboQuant @Recall@30>=0.995", "R10.4/R10.5", "1.34"),
 ("D", "F10.22", "Fig 10", "BBQ / TurboQuant @Recall@50>=0.995", "R10.6/R10.7", "1.14"),
 # ---------------- Fig 11 / Sec 6.3 ablation
 ("R", "F11.1", "Fig 11a", "sym-BQ Recall@10, ef=256", F11, {"method": "Sym-BQ", "efSearch": "256"}, "recall_at_10", "0.7231"),
 ("R", "F11.2", "Fig 11a", "ADC-BQ Recall@10, ef=256", F11, {"method": "ADC-BQ", "efSearch": "256"}, "recall_at_10", "0.8030"),
 ("R", "F11.3", "Fig 11a", "fixed R=75 Recall@10, ef=256", F11, {"method": "ADC-BQ + Fixed R=75", "efSearch": "256"}, "recall_at_10", "0.9893"),
 ("R", "F11.4", "Fig 11a", "calibrated Recall@10, ef=256", F11, {"method": "BBQ + Calibrated R(q)", "efSearch": "256"}, "recall_at_10", "0.9897"),
 ("R", "F11.5", "Fig 11a", "calibrated FP32 reads, ef=256", F11, {"method": "BBQ + Calibrated R(q)", "efSearch": "256"}, "fp_vectors_per_query_mean", "54.0"),
 ("R", "F11.6", "Fig 11a", "fixed R=75 QPS, ef=256", F11E, {"efSearch": "256"}, "fixed_qps_median", "1109"),
 ("R", "F11.7", "Fig 11a", "calibrated QPS, ef=256", F11E, {"efSearch": "256"}, "calibrated_qps_median", "1191"),
 ("D", "F11.8", "Sec 6.3", "calibrated reads fewer than R=75 (%)", "(75-F11.5)/75*100", "28.0"),
 ("D", "F11.9", "Sec 6.3", "calibrated QPS gain (%)", "(F11.7/F11.6-1)*100", "7.4"),
 ("R", "F11.10", "Sec 6.3", "QPS gain over the sweep, low end (%)", F11E, {"dataset": "msmarco_qwen1024"}, "calibrated_qps_change_pct", "2.7", "min"),
 ("R", "F11.11", "Sec 6.3", "QPS gain over the sweep, high end (%)", F11E, {"dataset": "msmarco_qwen1024"}, "calibrated_qps_change_pct", "19.9", "max"),
 ("R", "F11.12", "Fig 11b", "share of queries with Recall@10=1.0 (%)", F11G, {"per_query_recall_at_10": "1.0"}, "query_percent", "91.79"),
 ("R", "F11.13", "Fig 11b", "FP32 reads, Recall@10=1.0 group", F11G, {"per_query_recall_at_10": "1.0"}, "mean_fp_reads", "51.5"),
 ("R", "F11.14", "Fig 11b", "FP32 reads, Recall@10=0.9 group", F11G, {"per_query_recall_at_10": "0.9"}, "mean_fp_reads", "75.4"),
 ("R", "F11.15", "Fig 11b", "FP32 reads, Recall@10<=0.8 group", F11G, {"per_query_recall_at_10": "<=0.8"}, "mean_fp_reads", "119.8"),
 ("R", "F11.16", "Fig 11b", "latency, Recall@10=1.0 group (ms)", F11G, {"per_query_recall_at_10": "1.0"}, "mean_latency_ms", "0.921"),
 ("R", "F11.17", "Fig 11b", "latency, Recall@10<=0.8 group (ms)", F11G, {"per_query_recall_at_10": "<=0.8"}, "mean_latency_ms", "1.192"),
 # ---------------- Sec 6.3 traversal efficiency
 ("R", "S63.1", "Sec 6.3", "BBQ Recall@10 at ef=512", S63, {"mode": "calibrated_ssd_refinement"}, "recall_at_10", "0.992575"),
 ("R", "S63.2", "Sec 6.3", "FP32 HNSW Recall@10 at ef=384", S63, {"mode": "full_search"}, "recall_at_10", "0.992642"),
 ("R", "S63.3", "Sec 6.3", "BBQ neighbours evaluated per query", S63, {"mode": "calibrated_ssd_refinement"}, "distance_evaluations_mean", "9533"),
 ("R", "S63.4", "Sec 6.3", "FP32 HNSW neighbours evaluated per query", S63, {"mode": "full_search"}, "distance_evaluations_mean", "7256"),
 ("D", "S63.5", "Sec 6.3", "BBQ / FP32 HNSW evaluations", "S63.3/S63.4", "1.31"),
 ("R", "R63.1", "Sec 6.3", "BBQ QPS", S63, {"mode": "calibrated_ssd_refinement"}, "qps_median", None),
 ("R", "R63.2", "Sec 6.3", "FP32 HNSW QPS", S63, {"mode": "full_search"}, "qps_median", None),
 ("D", "S63.6", "Sec 6.3", "BBQ / FP32 HNSW QPS", "R63.1/R63.2", "1.93"),
 ("R", "R63.3", "Sec 6.3", "FP32 HNSW scored payload (B)", S63, {"mode": "full_search"}, "nominal_scored_vector_bytes_per_query", None),
 ("R", "R63.4", "Sec 6.3", "BBQ scored payload (B)", S63, {"mode": "calibrated_ssd_refinement"}, "nominal_scored_vector_bytes_per_query", None),
 ("D", "S63.7", "Sec 6.3", "FP32 HNSW scored payload (MB)", "R63.3/1000000", "29.72"),
 ("D", "S63.8", "Sec 6.3", "BBQ scored payload (MB)", "R63.4/1000000", "1.22"),
 ("D", "S63.9", "Sec 6.3", "payload ratio", "R63.3/R63.4", "24.4"),
 ("R", "S63.10", "Sec 6.3", "BBQ traversal time (ms)", S63, {"mode": "traversal_only"}, "time_ms_per_query", "1.31"),
 ("R", "S63.11", "Sec 6.3", "BBQ total time (ms)", S63, {"mode": "calibrated_ssd_refinement"}, "time_ms_per_query", "1.55"),
 ("D", "S63.12", "Sec 6.3", "exposed refinement time (ms)", "S63.11-S63.10", "0.24"),
 ("R", "S63.13", "Sec 6.3", "FP32 HNSW time (ms)", S63, {"mode": "full_search"}, "time_ms_per_query", "2.98"),
 ("R", "S63.14", "Sec 6.3", "BBQ FP32 reads per query", S63, {"mode": "calibrated_ssd_refinement"}, "fp32_reads_mean", "50.4"),
 ("R", "S63.15", "Sec 6.3", "BBQ Recall@10, DBpedia Qwen ef=1024", PR, {"dataset": "dbpedia_qwen1024", "method": B, "search_parameter_value": "1024"}, "recall_at_10", "0.9943"),
 ("R", "S63.16", "Sec 6.3", "BBQ QPS, DBpedia Qwen ef=1024", PR, {"dataset": "dbpedia_qwen1024", "method": B, "search_parameter_value": "1024"}, "qps", "280"),
 ("R", "S63.17", "Sec 6.3", "FP32 HNSW QPS, DBpedia Qwen ef=320", PR, {"dataset": "dbpedia_qwen1024", "method": FP, "search_parameter_value": "320"}, "qps", "302"),
 # ---------------- Fig 12 calibration (legacy GT, as plotted)
 ("R", "F12.1", "Fig 12a", "mean FP32 reads at target 0.90", F12, {"target_pool_top10_coverage": "0.9"}, "mean_fp_reads", "28.3"),
 ("R", "F12.2", "Fig 12a", "mean FP32 reads at target 0.995", F12, {"target_pool_top10_coverage": "0.995"}, "mean_fp_reads", "58.6"),
 ("R", "F12.3", "Fig 12a", "pool top-10 preserved at target 0.99", F12, {"target_pool_top10_coverage": "0.99"}, "test_empirical_pool_top10_coverage", "0.9928"),
 ("REF", "R12.1", "Fig 12a", "reads at target 0.99", F12, {"target_pool_top10_coverage": "0.99"}, "mean_fp_reads"),
 ("REF", "R12.2", "Fig 12a", "Recall@10 at target 0.99", F12, {"target_pool_top10_coverage": "0.99"}, "final_mean_recall_at_10"),
 ("REF", "R12.3", "Fig 12a", "Recall@10 at target 0.995", F12, {"target_pool_top10_coverage": "0.995"}, "final_mean_recall_at_10"),
 ("D", "F12.4", "Sec 6.3", "extra reads 0.99 -> 0.995", "F12.2-R12.1", "4.6"),
 ("D", "F12.5", "Sec 6.3", "Recall@10 gain 0.99 -> 0.995", "R12.3-R12.2", "0.0002"),
 ("R", "F12.6", "Fig 12b", "L_visit at target 0.99", F12L, {"component": "L_visit"}, "loss", "0.0096"),
 ("R", "F12.7", "Fig 12b", "L_select at target 0.99", F12L, {"component": "L_select"}, "loss", "0.0007"),
 ("R", "F12.8", "Fig 12b", "total 1 - Recall@10", F12L, {"component": "total"}, "loss", "0.0103"),
 ("R", "F12.9", "Fig 12b", "queries with traversal loss", F12L, {"component": "L_visit"}, "queries", "454"),
 ("R", "F12.10", "Fig 12b", "queries with selection loss", F12L, {"component": "L_select"}, "queries", "40"),
 ("R", "F12.11", "Sec 6.3", "traversal share of the loss (%)", F12L, {"component": "visit_share"}, "loss", "0.93"),
 # ---------------- Fig 13 same calibration
 ("R", "F13.1", "Fig 13", "TurboQuant QPS, MS MARCO @0.9853", F13, {"dataset": "msmarco_qwen1024", "method_key": "turboquant_tq1_calibrated", "efSearch": "144"}, "qps", "2060"),
 ("R", "F13.2", "Fig 13", "BBQ QPS, MS MARCO @0.9853", F13, {"dataset": "msmarco_qwen1024", "method_key": "adc_bq_calibrated", "efSearch": "144"}, "qps", "1809"),
 ("R", "F13.3", "Fig 13", "BBQ Recall@10, NQ", F13, {"dataset": "nq_qwen06_1024", "method_key": "adc_bq_calibrated", "efSearch": "512"}, "recall_at_10", "0.9976"),
 ("R", "F13.4", "Fig 13", "TurboQuant Recall@10, NQ", F13, {"dataset": "nq_qwen06_1024", "method_key": "turboquant_tq1_calibrated", "efSearch": "1024"}, "recall_at_10", "0.9973"),
 ("REF", "R13.1", "Fig 13", "BBQ QPS, NQ", F13, {"dataset": "nq_qwen06_1024", "method_key": "adc_bq_calibrated", "efSearch": "512"}, "qps"),
 ("REF", "R13.2", "Fig 13", "TurboQuant QPS, NQ", F13, {"dataset": "nq_qwen06_1024", "method_key": "turboquant_tq1_calibrated", "efSearch": "1024"}, "qps"),
 ("REF", "R13.3", "Fig 13", "BBQ distance evals, NQ", F13, {"dataset": "nq_qwen06_1024", "method_key": "adc_bq_calibrated", "efSearch": "512"}, "first_stage_distance_evaluations_per_query"),
 ("REF", "R13.4", "Fig 13", "TurboQuant distance evals, NQ", F13, {"dataset": "nq_qwen06_1024", "method_key": "turboquant_tq1_calibrated", "efSearch": "1024"}, "first_stage_distance_evaluations_per_query"),
 ("D", "F13.5", "Sec 6.3", "TurboQuant / BBQ QPS, MS MARCO", "F13.1/F13.2", "1.14"),
 ("D", "F13.6", "Sec 6.3", "BBQ / TurboQuant QPS, NQ", "R13.1/R13.2", "2.01"),
 ("D", "F13.7", "Sec 6.3", "TurboQuant / BBQ distance evals, NQ", "R13.4/R13.3", "1.77"),
 # ---------------- Fig 14 page I/O
 *[("R", f"F14.{i+1}", "Fig 14", t, F14, {"dataset": ds, "method_key": m, "target_recall": tr}, "pages_per_query", v)
   for i, (t, ds, m, tr, v) in enumerate([
       ("BBQ pages, OpenAI-1536 @0.98", "sieun_msmarco_openai1536", "adcbq", "0.98", "64.5"),
       ("TurboQuant pages, OpenAI-1536 @0.98", "sieun_msmarco_openai1536", "turboquant", "0.98", "120.0"),
       ("DiskANN-RaBitQ pages, OpenAI-1536 @0.98", "sieun_msmarco_openai1536", "diskann_rabitq", "0.98", "184.4"),
       ("DiskANN pages, OpenAI-1536 @0.98", "sieun_msmarco_openai1536", "diskann_classic", "0.98", "184.2"),
       ("BBQ pages, OpenAI-1536 @0.99", "sieun_msmarco_openai1536", "adcbq", "0.99", "67.6"),
       ("DiskANN-RaBitQ pages, OpenAI-1536 @0.99", "sieun_msmarco_openai1536", "diskann_rabitq", "0.99", "342.4"),
       ("DiskANN pages, OpenAI-1536 @0.99", "sieun_msmarco_openai1536", "diskann_classic", "0.99", "373.9"),
       ("BBQ pages, Qwen-1024 @0.98", "msmarco_qwen1024", "adcbq", "0.98", "46.2"),
       ("BBQ pages, Qwen-1024 @0.99", "msmarco_qwen1024", "adcbq", "0.99", "54.3"),
       ("DiskANN-RaBitQ pages, Qwen-1024 @0.99", "msmarco_qwen1024", "diskann_rabitq", "0.99", "913.6"),
       ("DiskANN pages, Qwen-1024 @0.99", "msmarco_qwen1024", "diskann_classic", "0.99", "1042.0"),
       ("TurboQuant pages, Qwen-1024 @0.99", "msmarco_qwen1024", "turboquant", "0.99", "60.0"),
       ("DiskANN-RaBitQ pages, Qwen-1024 @0.98", "msmarco_qwen1024", "diskann_rabitq", "0.98", "341.0"),
       ("DiskANN pages, Qwen-1024 @0.98", "msmarco_qwen1024", "diskann_classic", "0.98", "373.2")])],
 ("D", "F14.15", "Sec 6.4", "TurboQuant / BBQ pages, OpenAI @0.98 (low end of 1.86-2.86x)", "F14.2/F14.1", "1.86"),
 ("D", "F14.16", "Sec 6.4", "DiskANN / BBQ pages, OpenAI @0.98 (high end)", "F14.3/F14.1", "2.86"),
 ("D", "F14.17", "Sec 6.4", "BBQ page growth 0.98->0.99, OpenAI", "F14.5/F14.1", "1.05"),
 ("D", "F14.18", "Sec 6.4", "BBQ page growth 0.98->0.99, Qwen", "F14.9/F14.8", "1.18"),
 ("D", "F14.19", "Sec 6.4", "DiskANN-RaBitQ growth, OpenAI (low end of 1.86-2.68x)", "F14.6/F14.3", "1.86"),
 ("D", "F14.20", "Sec 6.4", "DiskANN-RaBitQ growth, Qwen (high end)", "F14.10/F14.13", "2.68"),
 ("D", "F14.21", "Sec 6.4", "DiskANN growth, OpenAI (low end of 2.03-2.79x)", "F14.7/F14.4", "2.03"),
 ("D", "F14.22", "Sec 6.4", "DiskANN growth, Qwen (high end)", "F14.11/F14.14", "2.79"),
 # ---------------- Fig 15 I/O modes
 *[("R", f"F15.{i+1}", "Fig 15", t, F15, {"dataset": ds, "method_key": m, "efSearch": ef}, "qps", v)
   for i, (t, ds, m, ef, v) in enumerate([
       ("Sequential QPS, Qwen-1024", "msmarco_qwen1024", "sequential", "320", "222.0"),
       ("Async Batch QPS, Qwen-1024", "msmarco_qwen1024", "async_batch", "320", "928.0"),
       ("Async Batch + Overlap QPS, Qwen-1024", "msmarco_qwen1024", "async_batch_overlap", "320", "993.5"),
       ("Sequential QPS, OpenAI-1536", "sieun_msmarco_openai1536", "sequential", "88", "376.9"),
       ("Async Batch QPS, OpenAI-1536", "sieun_msmarco_openai1536", "async_batch", "88", "1898.3"),
       ("Async Batch + Overlap QPS, OpenAI-1536", "sieun_msmarco_openai1536", "async_batch_overlap", "88", "2322.8")])],
 ("D", "F15.7", "Sec 6.4", "batching speedup, Qwen", "F15.2/F15.1", "4.18"),
 ("D", "F15.8", "Sec 6.4", "batching speedup, OpenAI", "F15.5/F15.4", "5.04"),
 ("D", "F15.9", "Sec 6.4", "overlap gain, Qwen (%)", "(F15.3/F15.2-1)*100", "7.1"),
 ("D", "F15.10", "Sec 6.4", "overlap gain, OpenAI (%)", "(F15.6/F15.5-1)*100", "22.4"),
 # ---------------- Figs 16-17
 ("R", "F16.1", "Fig 16", "BBQ QPS @>=0.995, 512D", F16.format(512), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "320"}, "qps", "1178"),
 ("R", "F16.2", "Fig 16", "BBQ QPS @>=0.995, 4096D", F16.format(4096), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "128"}, "qps", "972"),
 ("R", "F16.3", "Fig 16", "BBQ FP32 reads, 512D", F16.format(512), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "320"}, "fp_vectors_per_query", "97.5"),
 ("R", "F16.4", "Fig 16", "BBQ FP32 reads, 4096D", F16.format(4096), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "128"}, "fp_vectors_per_query", "23.4"),
 ("R", "F16.5", "Fig 16", "BBQ Recall@10 at ef=128, 512D", F16.format(512), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "128"}, "recall_at_10", "0.9868"),
 ("R", "F16.6", "Fig 16", "BBQ Recall@10 at ef=128, 4096D", F16.format(4096), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "128"}, "recall_at_10", "0.9952"),
 ("R", "F16.7", "Fig 16", "BBQ max Recall@10, 4096D", F16.format(4096), {"method_key": "adc_bq_calibrated"}, "recall_at_10", "0.9988", "max"),
 ("R", "F16.8", "Fig 16", "TurboQuant max Recall@10, 4096D", F16.format(4096), {"method_key": "turboquant_tq1"}, "recall_at_10", "0.9990", "max"),
 ("R", "F17.1", "Fig 17", "BBQ QPS @>=0.99, 10M", F17.format("10m"), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "512"}, "qps", "563"),
 ("R", "F17.2", "Fig 17", "BBQ QPS @>=0.99, 50M", F17.format("50m"), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "512"}, "qps", "541"),
 ("R", "F17.3", "Fig 17", "BBQ QPS @>=0.99, 100M", F17.format("100m"), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "512"}, "qps", "539"),
 ("D", "F17.4", "Sec 6.5", "QPS decline 10M -> 100M (%)", "(1-F17.3/F17.1)*100", "4.3"),
 ("BI", "R17.1", "Fig 17", "TurboQuant QPS @0.99, 100M", F17.format("100m"), {"method_key": "turboquant_tq1"}, "recall_at_10", .99, "qps", None, "search_parameter_value"),
 ("BI", "R17.2", "Fig 17", "RaBitQ-1bit QPS @0.99, 100M", F17.format("100m"), {"method_key": "rabitq_b1_adaptive"}, "recall_at_10", .99, "qps", None, "search_parameter_value"),
 ("D", "F17.5", "Sec 6.5", "BBQ / TurboQuant @0.99, 100M", "F17.3/R17.1", "1.98"),
 ("D", "F17.6", "Sec 6.5", "BBQ / RaBitQ-1bit @0.99, 100M", "F17.3/R17.2", "9.59"),
 ("R", "F17.7", "Fig 17", "BBQ max Recall@10, 100M", F17.format("100m"), {"method_key": "adc_bq_calibrated"}, "recall_at_10", "0.9944", "max"),
 ("R", "F17.8", "Fig 17", "TurboQuant max Recall@10, 100M", F17.format("100m"), {"method_key": "turboquant_tq1"}, "recall_at_10", "0.9931", "max"),
 ("R", "F17.9", "Fig 17", "RaBitQ-1bit max Recall@10, 100M", F17.format("100m"), {"method_key": "rabitq_b1_adaptive"}, "recall_at_10", "0.9923", "max"),
 ("R", "F17.10", "Fig 17", "RaBitQ-7bit max Recall@10, 100M", F17.format("100m"), {"method_key": "rabitq_b7"}, "recall_at_10", "0.9849", "max"),
 ("R", "F17.11", "Fig 17", "BBQ FP32 reads @0.99, 10M", F17.format("10m"), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "512"}, "fp_vectors_per_query", "51.7"),
 ("R", "F17.12", "Fig 17", "BBQ FP32 reads @0.99, 100M", F17.format("100m"), {"method_key": "adc_bq_calibrated", "calibration_reference_ef": "512"}, "fp_vectors_per_query", "71.5"),
 # ---------------- Fig 18 insertion
 ("R", "F18.1", "Fig 18", "BBQ insertion time, 10M (s)", F18, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "10000000"}, "adc_online_add_seconds", "521.7"),
 ("R", "F18.2", "Fig 18", "BBQ insertion time, 100M (s)", F18, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "100000000"}, "adc_online_add_seconds", "5762.2"),
 ("R", "F18.3", "Fig 18", "RaBitQ-1bit insertion time, 10M (s)", F18R, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "10000000"}, "add_seconds", "1073.5"),
 ("R", "F18.4", "Fig 18", "RaBitQ-1bit insertion time, 100M (s)", F18R, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "100000000"}, "add_seconds", "8568.5"),
 ("D", "F18.5", "Sec 6.5", "insertion speedup vs RaBitQ-1bit, 10M", "F18.3/F18.1", "2.06"),
 ("D", "F18.6", "Sec 6.5", "insertion speedup vs RaBitQ-1bit, 100M", "F18.4/F18.2", "1.49"),
 ("R", "F18.7", "Fig 18", "BBQ Recall@10 after 100M", F18, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "100000000"}, "adc_recall_at_10", "0.9853"),
 ("R", "F18.8", "Fig 18", "RaBitQ-1bit Recall@10 after 100M", F18R, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "100000000"}, "recall_current_at_k", "0.9842"),
 ("R", "F18.9", "Fig 18", "TurboQuant Recall@10 after 100M", F18T, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "100000000"}, "recall_current_at_k", "0.9766"),
 ("R", "F18.10", "Fig 18", "TurboQuant Recall@10 at 10M", F18T, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "10000000"}, "recall_current_at_k", "0.9883"),
 ("R", "F18.11", "Fig 18", "BBQ min Recall@10 over checkpoints", F18, {"mode": "online_add_no_rebuild", "efSearch": "256"}, "adc_recall_at_10", "0.9849", "min"),
 ("R", "F18.12", "Fig 18a", "RaBitQ-1bit Recall@10 at 50M (as plotted: 0.9852)", F18R, {"mode": "online_add_no_rebuild", "efSearch": "256", "insert_size": "50000000"}, "recall_current_at_k", "0.9852"),
 # ---------------- Fig 19 colocation
 ("R", "F19.1", "Fig 19", "16KB baseline QPS @50% DRAM", F19, {"layout": "baseline", "page_bytes": "16384", "bq_ram_ratio_target": "0.5"}, "qps_median", "112.2"),
 ("R", "F19.2", "Fig 19", "16KB colocated QPS @50% DRAM", F19, {"layout": "coloc", "page_bytes": "16384", "bq_ram_ratio_target": "0.5"}, "qps_median", "170.6"),
 ("R", "F19.3", "Fig 19", "16KB baseline graph page hit ratio @50%", F19, {"layout": "baseline", "page_bytes": "16384", "bq_ram_ratio_target": "0.5"}, "graph_page_hit_ratio", "0.576"),
 ("R", "F19.4", "Fig 19", "16KB colocated graph page hit ratio @50%", F19, {"layout": "coloc", "page_bytes": "16384", "bq_ram_ratio_target": "0.5"}, "graph_page_hit_ratio", "0.751"),
 ("D", "F19.5", "Sec 6.5", "colocation QPS gain (%)", "(F19.2/F19.1-1)*100", "52.0"),
 # ---------------- Sec 6.6 method selection
 ("G", "S66.1", "Sec 6.6", "RaBitQ-7bit over BBQ, median @0.95", T1, "rabitq7_over_bbq_median_at_0.95", "2.10"),
 ("G", "S66.2", "Sec 6.6", "RaBitQ-7bit over BBQ, median @0.99", T1, "rabitq7_over_bbq_median_at_0.99", "1.19"),
 ("G", "S66.3", "Sec 6.6", "RaBitQ-SSD raises max recall in (configurations)", T1, "rabitq_ssd_higher_max_recall_than_bbq", "15"),
 ("G", "S66.4", "Sec 6.6", "RaBitQ-SSD max-recall gain, mean (pp)", T1, "rabitq_ssd_minus_bbq_max_recall_mean_pp", "0.20"),
 ("G", "S66.5", "Sec 6.6", "BBQ over RaBitQ-SSD @0.99, median", T1, "bbq_over_rabitq_ssd_median_at_0.99", "9.1"),
 ("G", "S66.6", "Sec 6.6", "TurboQuant max-recall change vs BBQ, mean (pp; 'at most 0.06')", T1, "tq_minus_bbq_max_recall_mean_pp", "-0.06"),
 ("G", "S66.7", "Sec 6.6", "SSD-resident graphs slower @0.99, low end (= DiskANN-PQ range)", T1, "bbq_over_diskann_pq_min_at_0.99", "12.7"),
 ("G", "S66.8", "Sec 6.6", "SSD-resident graphs slower @0.99, high end (= DiskANN-PQ range)", T1, "bbq_over_diskann_pq_max_at_0.99", "31.6"),
 ("G", "S66.9", "Sec 6.6", "BBQ over DiskANN-PQ @0.99, median", T1, "bbq_over_diskann_pq_median_at_0.99", "22.6"),
 ("G", "S66.10", "Sec 6.6", "BBQ over PipeANN @0.99, median", T1, "bbq_over_pipeann_median_at_0.99", "19.7"),
 ("G", "S66.11", "Sec 6.6", "(configuration, recall-target) cells at the d/8-byte budget", T1, "dbudget_cells", "74"),
 ("G", "S66.12", "Sec 6.6", "cells where BBQ is fastest", T1, "dbudget_bbq_fastest_cells", "58"),
 ("G", "S66.13", "Sec 6.6", "BBQ margin over the runner-up, median", T1, "dbudget_bbq_margin_over_runner_up_median", "1.4"),
 ("G", "S66.14", "Sec 6.6", "BBQ losses in the d/8-byte class", T1, "dbudget_bbq_losses", "16"),
 ("G", "S66.15", "Sec 6.6", "losses whose target is above BBQ's recall ceiling", T1, "dbudget_losses_above_bbq_ceiling", "12"),
 ("G", "S66.16", "Sec 6.6", "cells where BBQ clears the target by >= 0.3 pp", T1, "dbudget_cells_bbq_headroom_ge_0.3pp", "42"),
 ("G", "S66.17", "Sec 6.6", "of those, cells where BBQ is fastest", T1, "dbudget_bbq_fastest_headroom_ge_0.3pp", "40"),
 # ---------------- Appendix B (single thread, same table)
 *[("B", f"AB.{i+1}", "App B", t, T1, ds, m, th, "value", v) for i, (t, ds, m, th, v) in enumerate([
     ("BBQ QPS @0.99, NQ Qwen", "nq_qwen06_1024", B, "0.99", "1753"), ("TurboQuant @0.99, NQ Qwen", "nq_qwen06_1024", TQ, "0.99", "1253"),
     ("FP32 HNSW @0.99, NQ Qwen", "nq_qwen06_1024", FP, "0.99", "1234"), ("BBQ @0.99, Quora Qwen", "quora_qwen1024", B, "0.99", "2803"),
     ("TurboQuant @0.99, Quora Qwen", "quora_qwen1024", TQ, "0.99", "2500"), ("FP32 HNSW @0.99, Quora Qwen", "quora_qwen1024", FP, "0.99", "2309"),
     ("FP32 HNSW @0.995, NQ Qwen", "nq_qwen06_1024", FP, "0.995", "669"), ("BBQ @0.995, Quora Qwen", "quora_qwen1024", B, "0.995", "2386"),
     ("TurboQuant @0.995, NQ BGE", "nq_bge_m3_vanilla", TQ, "0.995", "416"), ("BBQ @0.995, NQ BGE", "nq_bge_m3_vanilla", B, "0.995", "376")])],
 # ---------------- Appendix B (24 threads) / Fig 21
 ("G", "F21.1", "App B", "BBQ beats TurboQuant @0.99, 24 threads", T24, "bbq_faster_than_tq_at_0.99", "14"),
 ("G", "F21.2", "App B", "configurations where both reach 0.99, 24 threads", T24, "both_reach_0.99_bbq_tq", "15"),
 ("G", "F21.3", "App B", "BBQ thread speedup, median", T24, "bbq_thread_speedup_median_at_0.99", "13.7"),
 ("G", "F21.4", "App B", "FP32 HNSW thread speedup, median", T24, "fp_hnsw_thread_speedup_median_at_0.99", "9.2"),
 *[("B", f"F21.{5+i}", "App B", t, T24, ds, m, "0.99", "value", v) for i, (t, ds, m, v) in enumerate([
     ("BBQ QPS, NQ Qwen", "nq_qwen06_1024", B, "23062"), ("TurboQuant, NQ Qwen", "nq_qwen06_1024", TQ, "18193"),
     ("FP32 HNSW, NQ Qwen", "nq_qwen06_1024", FP, "10126"), ("BBQ, Quora Qwen", "quora_qwen1024", B, "52083"),
     ("TurboQuant, Quora Qwen", "quora_qwen1024", TQ, "39510"), ("FP32 HNSW, Quora Qwen", "quora_qwen1024", FP, "22032"),
     ("BBQ, MS MARCO Qwen", "msmarco_qwen1024", B, "14416"), ("DiskANN-PQ, MS MARCO Qwen", "msmarco_qwen1024", DPQ, "677"),
     ("DiskANN-RaBitQ, MS MARCO Qwen", "msmarco_qwen1024", DRBQ, "648"), ("PipeANN, MS MARCO Qwen", "msmarco_qwen1024", PIPE, "669"),
     ("TurboQuant, DBpedia BGE", "sieun_dbpedia_entity_bge_large1024_beir400", TQ, "6470"),
     ("BBQ, DBpedia BGE", "sieun_dbpedia_entity_bge_large1024_beir400", B, "2742")])],
]

# colocation rows are identified by their position in the 4K/8K/16K x ratio x layout sweep
COLOC = {}
KEYS = ["dataset", "method_key", "method", "layout", "page_size", "page_kb", "memory_ratio", "efSearch",
        "calibration_reference_ef", "search_parameter_value", "target_recall", "insert_size", "mode"]


def rows_of(f):
    return list(csv.DictReader(open(EXP / f, newline="")))


def unique_selector(rows, row):
    for n in range(1, len(KEYS) + 1):
        cols = [k for k in KEYS if k in row][:n]
        if sum(all(r[c] == row[c] for c in cols) for r in rows) == 1:
            return ";".join(f"{c}={row[c]}" for c in cols)
    raise SystemExit("no unique selector")


def sel_str(d):
    return ";".join(f"{k}={v}" for k, v in d.items())


def check(f, sel, agg):
    rows = rows_of(f)
    n = sum(all(r.get(k) == v for k, v in sel.items()) for r in rows)
    if (agg and n < 1) or (not agg and n != 1):
        raise SystemExit(f"{f}: selector {sel} matched {n} rows")


def tol_of(printed):
    mult = 1000 if printed.endswith("K") else 1
    p = printed.rstrip("K")
    dec = len(p.split(".")[1]) if "." in p else 0
    return float(p) * mult, 10 ** -dec * mult


out = []
def emit(cid, sec, text, file="", select="", column="", derive="", agg="", printed=None):
    exp, tol = tol_of(printed) if printed is not None else ("", "")
    out.append(dict(id=cid, section=sec, claim=text, file=file, select=select, column=column, agg=agg, derive=derive,
                    expected=f"{exp:g}" if printed is not None else "", tolerance=f"{tol:g}" if printed is not None else ""))

for c in C:
    kind, cid, sec, text = c[:4]
    if kind in ("R", "REF"):
        f, sel, col = c[4], dict(c[5]), c[6]
        printed = c[7] if kind == "R" else None
        agg = c[8] if len(c) > 8 else ""
        if cid in COLOC:
            rows = rows_of(f); sel = {}; s = unique_selector(rows, rows[COLOC[cid] - 2])
            emit(cid, sec, text, f, s, col, printed=printed); continue
        if any(v is None for v in sel.values()):                 # best row at recall >= 0.95 (MRQ)
            rows = [r for r in rows_of(f) if all(r.get(k) == v for k, v in sel.items() if v is not None)
                    and float(r["recall_at_10"]) >= 0.95]
            best = max(rows, key=lambda r: float(r[col]))
            sel = {k: (best[k] if v is None else v) for k, v in sel.items()}
        check(f, sel, agg)
        emit(cid, sec, text, f, sel_str(sel), col, agg=agg, printed=printed)
    elif kind == "BI":
        f, flt, rcol, th, col, printed, key = c[4:11]
        rows = [r for r in rows_of(f) if all(r.get(k) == v for k, v in flt.items()) and float(r[rcol]) >= th]
        best = max(rows, key=lambda r: float(r["qps"]))
        sel = dict(flt, **{key: best[key]})
        check(f, sel, "")
        emit(cid, sec, text, f, sel_str(sel), col, printed=printed)
    elif kind == "B":
        f, ds, m, th, col, printed = c[4:10]
        sel = {"kind": "best", "dataset": ds, "method": m, "threshold": th}
        check(f, sel, "")
        emit(cid, sec, text, f, sel_str(sel), col, printed=printed)
    elif kind == "M":
        f, ds, m, printed = c[4:8]
        sel = {"kind": "max_recall", "dataset": ds, "method": m}
        check(f, sel, "")
        emit(cid, sec, text, f, sel_str(sel), "value", printed=printed)
    elif kind == "G":
        f, metric, printed = c[4:7]
        sel = {"kind": "agg", "metric": metric}
        check(f, sel, "")
        emit(cid, sec, text, f, sel_str(sel), "value", printed=printed)
    elif kind == "D":
        emit(cid, sec, text, derive=c[4], printed=c[5])

with open(EXP / "claims.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"{len(out)} claims written")
