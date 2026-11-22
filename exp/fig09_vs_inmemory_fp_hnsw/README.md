# Fig 9 / Sec 6.2: BBQ (SSD) vs in-memory FP32 HNSW (MS MARCO, DBpedia; Qwen-1024; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F9.`  (9 claims) |
| Re-plot | `./exp/fig09_vs_inmemory_fp_hnsw/plot/regen.sh` |
| Paper figure | `plot/comparision_hnsw.paper.pdf` |

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
- `results/clean_rerun_20261110_cpu20/recall_qps_t1_from_recall096_rerun.pdf`
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

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F9.1 | BBQ QPS, MS MARCO @0.9926 | 635 |
| F9.2 | in-memory FP32 HNSW QPS, MS MARCO @0.9926 | 335 |
| F9.3 | BBQ QPS, DBpedia @0.9875 | 1547 |
| F9.4 | in-memory FP32 HNSW QPS, DBpedia @0.9875 | 727 |
| F9.5 | BBQ distance evals, DBpedia ef=1024 | 22977 |
| F9.6 | FP32 HNSW distance evals, DBpedia ef=256 | 6834 |
| D9.1 | BBQ / FP32 HNSW QPS, MS MARCO (= `F9.1/F9.2`) | 1.90 |
| D9.2 | BBQ / FP32 HNSW QPS, DBpedia (= `F9.3/F9.4`) | 2.13 |
| D9.3 | BBQ / FP32 HNSW distance evals, DBpedia (= `F9.5/F9.6`) | 3.36 |

## Notes

- **Two runs.** The numbers quoted in Sec. 6.2 (635 vs 335 QPS, 1547 vs 727 QPS) come from `results/paper_run_20261109/`; the plotted curves come from the 2026-11-10 rerun in `results/clean_rerun_20261110_cpu20/`. Both are kept.
- `results/rerun_ndis_20261120/` holds the DBpedia distance-evaluation counters (22,977 vs 6,834) quoted in the text.
- `results/calibration/` holds the per-efSearch tau fitted by `fit_pool_top10_conformal` (= `bbq/calibration/calibrate.py`).
