# Secs 2-3: memory/score geometry (Fig 2), candidate preservation and score geometry (Fig 3, Tables 2-3), long-tailed demand (Fig 4)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only T3.`  (35 claims here) |
| Re-plot | see `plot/` |
| Paper figure | — |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/sec2_3_analysis/harness/src/measure.py`
- `exp/sec2_3_analysis/harness/src/kcurve.py`
- `exp/sec2_3_analysis/harness/src/dimsweep.py`
- `exp/sec2_3_analysis/harness/src/dimperq.py`
- `exp/sec2_3_analysis/harness/src/bitsvsdims.py`
- `exp/sec2_3_analysis/harness/src/levels.py`
- `exp/sec2_3_analysis/harness/src/export_fig4.py`
- `exp/sec2_3_analysis/harness/src/export_perq.py`
- `exp/sec2_3_analysis/harness/src/export_dimperq.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/sec3_values.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| S3.1 | sym-BQ CandidateRecall@30, MS MARCO BGE-M3 | 0.848 |
| S3.2 | ADC-BQ CandidateRecall@30, MS MARCO BGE-M3 | 0.947 |
| S3.3 | sym-BQ p95 R*, MS MARCO BGE-M3 | 416 |
| S3.4 | ADC-BQ p95 R*, MS MARCO BGE-M3 | 99 |
| S3.5 | kappa at d=32 (Qwen3-8B MRL) | 0.782 |
| S3.6 | kappa at d=4096 | 0.958 |
| S3.7 | beta at d=32 | 0.17 |
| S3.8 | beta at d=4096 | 1.07 |
| S3.9 | sign-code alignment f, low end | 0.775 |
| S3.10 | sign-code alignment f, high end | 0.802 |
| T2.1 | p95 R* at 512 B/vector, BQ | 21 |
| T2.2 | p95 R* at 512 B/vector, PQ | 23 |
| T2.3 | p95 R* at 512 B/vector, SQ-4bit | 37 |
| T2.4 | p95 R* at 512 B/vector, SQ-8bit | 87 |
| T2.5 | p95 R* at 128 B/vector, BQ | 73 |
| T2.6 | p95 R* at 128 B/vector, PQ | 822 |
| T2.7 | p95 R* at 128 B/vector, SQ-4bit | 352 |
| T2.8 | p95 R* at 128 B/vector, SQ-8bit | 1596 |
| S3.11 | median R*, MS MARCO BGE-M3 | 23 |
| T3.1 | p95 R*, MS MARCO Qwen, sym_bq | 140 |
| T3.2 | p95 R*, MS MARCO Qwen, adc_bq | 47 |
| T3.3 | p95 R*, MS MARCO Qwen, adc_bq_c | 61 |
| T3.4 | p95 R*, MS MARCO Qwen, tq1 | 43 |
| T3.5 | p95 R*, MS MARCO Qwen, rabitq1 | 43 |
| T3.6 | p95 R*, MS MARCO Qwen, pq | 48 |
| T3.7 | p95 R*, MS MARCO Qwen, opq | 33 |
| T3.8 | p95 R*, MS MARCO Qwen, sq4 | 14 |
| T3.9 | p95 R*, MS MARCO BGE-M3, sym_bq | 416 |
| T3.10 | p95 R*, MS MARCO BGE-M3, adc_bq | 99 |
| T3.11 | p95 R*, MS MARCO BGE-M3, adc_bq_c | 148 |
| T3.12 | p95 R*, MS MARCO BGE-M3, tq1 | 112 |
| T3.13 | p95 R*, MS MARCO BGE-M3, rabitq1 | 89 |
| T3.14 | p95 R*, MS MARCO BGE-M3, pq | 81 |
| T3.15 | p95 R*, MS MARCO BGE-M3, opq | 44 |
| T3.16 | p95 R*, MS MARCO BGE-M3, sq4 | 15 |

## Notes

- Workload analysis harness of Secs. 2-3 (Sieun Kim): `harness/src` produces `harness/out/*.json` from exhaustive scoring of each configuration; per-query arrays (`*_perq.npz`, 6.6 MB) live with the data (`$BBQ_DATA2/bbqeta`). `harness/gp` and `plot/` hold the gnuplot sources of Figs 2-4; `kcurve/` the Fig. 3(a) candidate-expansion data; `paper_figures/` the printed panels.
- `tools/flatten_outputs.py` flattens every harness output into `results/sec3_values.csv` so that verify can check them (CandidateRecall, kappa / beta / f across MRL widths, Table 2 from `bitsvsdims.txt`, Table 3 p95 R*).
- **Not found in any output:** the text-embedding-3-large width sweep (3072D p95 R* 17-31), the Haar-rotation sweep (25,123 -> 21), the whitening result (c_Q 0.74-0.76 on DBpedia), p99 R* = 195, the 89.4 % variance share (levels.json gives 85.7 % for its 7 settings) and the Spearman correlations 0.921 / 0.912. The raw text-embedding-3-large vectors exist (`data_prep/beir_embeddings`).
