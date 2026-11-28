# Sec 6.2-6.3: BBQ vs in-memory FP32 HNSW at matched recall; traversal-efficiency profile (MS MARCO / DBpedia Qwen-1024; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only S63.`  (14 claims here) |
| Re-plot | `./exp/sec6_3_traversal_efficiency/plot/regen.sh` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/rerun_calibrated_adcbq_ssd_vs_fp_hnsw_ram_t1.py`
- `exp/scripts/run_adcbq_vs_fp_hnsw_fullvec_ram.py`
- `exp/scripts/plot_calibrated_adcbq_hnsw_ssd_vs_fp_hnsw_ram_t1.py`
- `exp/scripts/faiss_fp_hnsw_resident_from_binary_graph.cpp`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/calibration/msmarco_qwen1024/calibration_target099.json`
- `results/clean_rerun_20261110_cpu20/dbpedia_qwen1024/adcbq_ssd_per_ef_target099/ef1024.csv`
- `results/clean_rerun_20261110_cpu20/dbpedia_qwen1024/adcbq_ssd_per_ef_target099/ef128.csv`
- `results/clean_rerun_20261110_cpu20/dbpedia_qwen1024/adcbq_ssd_per_ef_target099/ef256.csv`
- `results/clean_rerun_20261110_cpu20/dbpedia_qwen1024/adcbq_ssd_per_ef_target099/ef512.csv`
- `results/clean_rerun_20261110_cpu20/dbpedia_qwen1024/adcbq_ssd_per_ef_target099/ef64.csv`
- `results/clean_rerun_20261110_cpu20/dbpedia_qwen1024/fp_hnsw_ram.csv`
- `results/clean_rerun_20261110_cpu20/msmarco_qwen1024/adcbq_ssd_per_ef_target099/ef1024.csv`
- `results/clean_rerun_20261110_cpu20/msmarco_qwen1024/adcbq_ssd_per_ef_target099/ef128.csv`
- `results/clean_rerun_20261110_cpu20/msmarco_qwen1024/adcbq_ssd_per_ef_target099/ef256.csv`
- `results/clean_rerun_20261110_cpu20/msmarco_qwen1024/adcbq_ssd_per_ef_target099/ef512.csv`
- `results/clean_rerun_20261110_cpu20/msmarco_qwen1024/adcbq_ssd_per_ef_target099/ef64.csv`
- `results/clean_rerun_20261110_cpu20/msmarco_qwen1024/fp_hnsw_ram.csv`
- `results/clean_rerun_20261110_cpu20/recall_qps_t1_from_recall096_rerun.csv`
- `results/clean_rerun_20261110_cpu20/recall_qps_t1_from_recall096_rerun.png`
- `results/clean_rerun_20261110_cpu20/run_manifest.json`
- `results/paper_run_20261109/recall_qps_fullvec_ram_t1.csv`
- `results/paper_run_20261109/recall_qps_t1_from_recall096.csv`
- `results/paper_run_20261109/run_manifest.json`
- `results/paper_run_20261109/target_recall_summary_t1.csv`
- `results/rerun_ndis_20261120/README.md`
- `results/rerun_ndis_20261120/dbpedia_adcbq_ef1024.csv`
- `results/rerun_ndis_20261120/dbpedia_adcbq_ef1024.search_stats.csv`
- `results/rerun_ndis_20261120/dbpedia_fp_hnsw_ef256.csv`
- `results/rerun_ndis_20261120/dbpedia_matched_recall_summary.csv`
- `results/time_breakdown_20261124/README.md`
- `results/time_breakdown_20261124/dbpedia_traversal_only_ef128.csv`
- `results/time_breakdown_20261124/matched_recall_breakdown.csv`
- `results/time_breakdown_20261124/msmarco_fp_hnsw_ef384_stats.csv`
- `results/time_breakdown_20261124/msmarco_full_ef512.csv`
- `results/time_breakdown_20261124/msmarco_traversal_only_ef512.csv`
- `results/time_breakdown_20261124/msmarco_traversal_stats_ef512.csv`
- `results/time_breakdown_20261124/msmarco_traversal_stats_ef512.search_stats.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| S63.1 | BBQ Recall@10 at ef=512 | 0.992575 |
| S63.2 | FP32 HNSW Recall@10 at ef=384 | 0.992642 |
| S63.3 | BBQ neighbours evaluated per query | 9533 |
| S63.4 | FP32 HNSW neighbours evaluated per query | 7256 |
| S63.5 | BBQ / FP32 HNSW evaluations (= `S63.3/S63.4`) | 1.31 |
| S63.6 | BBQ / FP32 HNSW QPS (= `R63.1/R63.2`) | 1.93 |
| S63.7 | FP32 HNSW scored payload (MB) (= `R63.3/1000000`) | 29.72 |
| S63.8 | BBQ scored payload (MB) (= `R63.4/1000000`) | 1.22 |
| S63.9 | payload ratio (= `R63.3/R63.4`) | 24.4 |
| S63.10 | BBQ traversal time (ms) | 1.31 |
| S63.11 | BBQ total time (ms) | 1.55 |
| S63.12 | exposed refinement time (ms) (= `S63.11-S63.10`) | 0.24 |
| S63.13 | FP32 HNSW time (ms) | 2.98 |
| S63.14 | BBQ FP32 reads per query | 50.4 |

## Notes

- `results/time_breakdown_20261124/` is the matched-recall profile quoted in Sec. 6.3 (9,533 vs 7,256 evaluations, 1.31 ms traversal, 0.24 ms exposed refinement, 29.72 vs 1.22 MB nominal payload). The older full curves (Nov 9 / Nov 10 runs) are kept alongside.
