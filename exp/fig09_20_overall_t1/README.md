# Fig 9 (MS MARCO, DBpedia) + Fig 20 (NQ, Quora): Recall@10-QPS, 16 dataset-embedding configurations x 11 methods, single thread

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F9.`  (76 claims here) |
| Re-plot | `./exp/fig09_20_overall_t1/plot/regen.sh` |
| Paper figure | `plot/recall-qps_main.paper.pdf`, `plot/recall-qps_msmarco_dbpedia.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/build_latest_full16_11methods_t1_csv.py`
- `exp/scripts/build_latest_full16_10methods_t1_csv.py`
- `exp/scripts/build_onebit_t1_combined_csv.py`
- `exp/scripts/run_fp_hnsw_ram_main16_t1_t24.py`
- `exp/scripts/run_dbpedia_bge_beir_existing5_t1.py`
- `exp/scripts/run_openai_beir_11methods_t1.py`
- `exp/scripts/run_diskveclab_full_matrix_adcbq_pool_top10_per_ef.py`
- `exp/scripts/run_diskveclab_full_matrix_mrq_rabitq_ssd.py`
- `exp/scripts/run_onebit_disk_baselines_t1.py`
- `exp/scripts/summarize_full16.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/coverage_latest_t1.csv`
- `results/full16_summary_rerun_t1.csv`
- `results/full16_summary_t1.csv`
- `results/manifest.json`
- `results/mrq_rabitq_ssd_summary_t1.csv`
- `results/recall_qps_all_latest_t1.csv`
- `results/recall_qps_as_printed_t1.csv`
- `results/with_spann_es/coverage_latest_t1.csv`
- `results/with_spann_es/full16_summary_t1.csv`
- `results/with_spann_es/manifest.json`
- `results/with_spann_es/recall_qps_all_latest_t1.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| I.6 | configurations where BBQ beats TurboQuant-1bit @0.99 | 14 |
| I.7 | max speedup over in-memory FP32 HNSW @0.99 | 2.7 |
| F9.1 | BBQ beats TurboQuant @0.99 (configurations) | 14 |
| F9.2 | configurations where both reach 0.995 | 7 |
| F9.3 | ...of which BBQ is faster ('all seven') | 7 |
| F9.4 | BBQ QPS @0.995, NQ Qwen | 1275 |
| F9.5 | TurboQuant QPS @0.995, NQ Qwen | 636 |
| F9.6 | BBQ / TurboQuant @0.995, NQ Qwen (= `F9.4/F9.5`) | 2 |
| F9.7 | BBQ QPS @0.99, MS MARCO Qwen | 1055 |
| F9.8 | TurboQuant QPS @0.99, MS MARCO Qwen | 664 |
| F9.9 | BBQ / TurboQuant @0.99, MS MARCO Qwen (= `F9.7/F9.8`) | 1.59 |
| F9.10 | mean FP32 reads, BBQ (where recorded) | 41.8 |
| F9.11 | mean FP32 reads, TurboQuant | 76 |
| F9.12 | OPQ FP32 reads, Quora Qwen ef=80 | 30 |
| F9.13 | PQ FP32 reads, Quora Qwen ef=80 | 80 |
| F9.14 | OPQ QPS, Quora Qwen ef=80 | 980 |
| F9.15 | PQ QPS, Quora Qwen ef=80 | 1886 |
| F9.16 | PipeANN max Recall@10, MS MARCO Qwen | 0.9925 |
| F9.17 | BBQ max Recall@10, MS MARCO Qwen | 0.9945 |
| F9.18 | DiskANN-PQ QPS @0.99, MS MARCO Qwen | 33 |
| F9.19 | DiskANN-RaBitQ QPS @0.99, MS MARCO Qwen | 35 |
| F9.20 | PipeANN QPS @0.99, MS MARCO Qwen | 38 |
| F9.21 | RaBitQ-7bit QPS @0.99, Quora Qwen | 4198 |
| F9.22 | BBQ QPS @0.99, Quora Qwen | 2803 |
| F9.23 | RaBitQ-7bit max Recall@10, Quora Qwen | 0.9948 |
| F9.24 | BBQ max Recall@10, Quora Qwen | 0.9981 |
| F9.25 | configurations where RaBitQ-7bit never reaches 0.99 | 3 |
| F9.26 | RaBitQ-SSD max recall higher than BBQ's (configurations) | 15 |
| F9.27 | BBQ over RaBitQ-SSD @0.99, low end of '9-20x' | 9 |
| F9.28 | BBQ over RaBitQ-SSD @0.99, high end of '9-20x' | 20 |
| F9.29 | BBQ over MRQ @0.95, NQ Qwen (= `R9.1/R9.2`) | 11.8 |
| F9.30 | BBQ over SPANN-PQ @0.95, NQ Qwen (= `R9.1/R9.3`) | 3.3 |
| F9.31 | TurboQuant QPS @0.99, DBpedia BGE | 250 |
| F9.32 | BBQ QPS @0.99, DBpedia BGE | 181 |
| F9.33 | TurboQuant QPS @0.99, DBpedia E5 | 201 |
| F9.34 | BBQ QPS @0.99, DBpedia E5 | 182 |
| F9.35 | BBQ FP32 reads @0.99, DBpedia BGE | 77.5 |
| F9.36 | TurboQuant FP32 reads @0.99, DBpedia BGE | 200 |
| F9.37 | BBQ efSearch @0.99, DBpedia BGE | 1024 |
| F9.38 | TurboQuant efSearch @0.99, DBpedia BGE | 512 |
| F9.39 | BBQ beats in-memory FP32 HNSW @0.99 (configurations) | 15 |
| F9.40 | speedup over FP32 HNSW, low end | 1.21 |
| F9.41 | speedup over FP32 HNSW, high end | 2.7 |
| F9.42 | BBQ QPS @0.99, msmarco_qwen1024 | 1055 |
| F9.43 | FP32 HNSW QPS @0.99, msmarco_qwen1024 | 555 |
| F9.44 | BBQ / FP32 HNSW, msmarco_qwen1024 (= `F9.42/F9.43`) | 1.9 |
| F9.45 | BBQ QPS @0.99, dbpedia_qwen1024 | 959 |
| F9.46 | FP32 HNSW QPS @0.99, dbpedia_qwen1024 | 540 |
| F9.47 | BBQ / FP32 HNSW, dbpedia_qwen1024 (= `F9.45/F9.46`) | 1.78 |
| F9.48 | BBQ QPS @0.99, sieun_dbpedia_entity_openai1536 | 1639 |
| F9.49 | FP32 HNSW QPS @0.99, sieun_dbpedia_entity_openai1536 | 608 |
| F9.50 | BBQ / FP32 HNSW, sieun_dbpedia_entity_openai1536 (= `F9.48/F9.49`) | 2.7 |
| F9.51 | max-recall gap to FP32 HNSW, largest (pp) | 0.2 |
| F9.52 | max-recall gap to FP32 HNSW, mean (pp, 'below 0.10') | 0.1 |
| F9.53 | BBQ / FP32 HNSW, DBpedia BGE (= `F9.32/R9.4`) | 0.92 |
| S63.15 | BBQ Recall@10, DBpedia Qwen ef=1024 | 0.9943 |
| S63.16 | BBQ QPS, DBpedia Qwen ef=1024 | 280 |
| S63.17 | FP32 HNSW QPS, DBpedia Qwen ef=320 | 302 |
| S66.1 | RaBitQ-7bit over BBQ, median @0.95 | 2.1 |
| S66.2 | RaBitQ-7bit over BBQ, median @0.99 | 1.19 |
| S66.3 | RaBitQ-SSD raises max recall in (configurations) | 15 |
| S66.4 | RaBitQ-SSD max-recall gain, mean (pp) | 0.2 |
| S66.5 | BBQ over RaBitQ-SSD @0.99, median | 9.1 |
| S66.6 | TurboQuant max-recall change vs BBQ, mean (pp; 'at most 0.06') | -0.06 |
| S66.7 | SSD-resident graphs slower than BBQ @0.99, low end | 12.7 |
| S66.8 | SSD-resident graphs slower than BBQ @0.99, high end | 31.6 |
| AB.1 | BBQ QPS @0.99, NQ Qwen | 1753 |
| AB.2 | TurboQuant @0.99, NQ Qwen | 1253 |
| AB.3 | FP32 HNSW @0.99, NQ Qwen | 1234 |
| AB.4 | BBQ @0.99, Quora Qwen | 2803 |
| AB.5 | TurboQuant @0.99, Quora Qwen | 2500 |
| AB.6 | FP32 HNSW @0.99, Quora Qwen | 2309 |
| AB.7 | FP32 HNSW @0.995, NQ Qwen | 669 |
| AB.8 | BBQ @0.995, Quora Qwen | 2386 |
| AB.9 | TurboQuant @0.995, NQ BGE | 416 |
| AB.10 | BBQ @0.995, NQ BGE | 376 |

## Notes

- **Which table.** The paper's Figs 9 and 20 were drawn from `results/recall_qps_as_printed_t1.csv` (3,477 rows). `regen.sh` rebuilds both figures from it identically (data and panel files). `results/recall_qps_all_latest_t1.csv` is the later table in which four curves were re-measured with 20 points (NQ BGE TurboQuant and PQ, MS MARCO Qwen OPQ, DBpedia E5 TurboQuant); draw it with `CSV=../results/recall_qps_all_latest_t1.csv ./regen.sh`.
- The text mixes the two tables: Sec. 6.2's 'DBpedia E5 TurboQuant 201 QPS' and 'all seven above 0.995' hold in the printed table, while Appendix B's 'NQ BGE TurboQuant 416 QPS at 0.995' comes from the rerun (352.8 in the printed table; claim AB.9).
- `results/full16_summary_t1.csv` (printed table) and `full16_summary_rerun_t1.csv` are written by `exp/scripts/summarize_full16.py`: best QPS per (configuration, method, recall target), maximum recall, and the corpus-wide counts / medians / ranges the text quotes. SPANN and MRQ (dropped from the figures) are in `results/with_spann_es/` and `results/mrq_rabitq_ssd_summary_t1.csv`.
- Markers and curves are a monotone trend through each method's Pareto frontier (see the docstring of `plot/update_recall-qps_full16.py`); quote numbers from the CSV, not from the plot.
- 'RaBitQ-SSD attains the highest maximum recall in 15 of 16' is relative to BBQ (claim F9.26); against all methods it is highest in 11 (FP32 HNSW has the higher maximum in 5). Not reproduced: '9-20x lower throughput at 0.99' (3.1-27.5x, median 9.1x).
