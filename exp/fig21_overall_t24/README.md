# Fig 21: Recall@10-QPS, 16 dataset-embedding configurations x 11 methods, 24 search threads

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F21.`  (16 claims here) |
| Re-plot | `./exp/fig21_overall_t24/plot/regen.sh` |
| Paper figure | `plot/recall-qps_main_t24.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/build_latest_full16_11methods_t24_csv.py`
- `exp/scripts/run_full16_12methods_t24.py`
- `exp/scripts/run_fp_hnsw_ram_main16_t1_t24.py`
- `exp/scripts/summarize_full16.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/coverage_latest_t24.csv`
- `results/full16_summary_t24.csv`
- `results/manifest.json`
- `results/recall_qps_all_latest_t24.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F21.1 | BBQ beats TurboQuant @0.99, 24 threads | 14 |
| F21.2 | configurations where both reach 0.99, 24 threads | 15 |
| F21.3 | BBQ thread speedup, median | 13.7 |
| F21.4 | FP32 HNSW thread speedup, median | 9.2 |
| F21.5 | BBQ QPS, NQ Qwen | 23062 |
| F21.6 | TurboQuant, NQ Qwen | 18193 |
| F21.7 | FP32 HNSW, NQ Qwen | 10126 |
| F21.8 | BBQ, Quora Qwen | 52083 |
| F21.9 | TurboQuant, Quora Qwen | 39510 |
| F21.10 | FP32 HNSW, Quora Qwen | 22032 |
| F21.11 | BBQ, MS MARCO Qwen | 14416 |
| F21.12 | DiskANN-PQ, MS MARCO Qwen | 677 |
| F21.13 | DiskANN-RaBitQ, MS MARCO Qwen | 648 |
| F21.14 | PipeANN, MS MARCO Qwen | 669 |
| F21.15 | TurboQuant, DBpedia BGE | 6470 |
| F21.16 | BBQ, DBpedia BGE | 2742 |

## Notes

- `regen.sh` rebuilds the figure from `results/recall_qps_all_latest_t24.csv` identically. `results/full16_summary_t24.csv` carries the Appendix B numbers, including the median thread speedups against the single-thread table.
