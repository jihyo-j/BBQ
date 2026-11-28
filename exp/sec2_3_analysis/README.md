# Secs 2-3: memory/score geometry (Fig 2), candidate preservation and score geometry (Fig 3, Tables 2-3), long-tailed demand (Fig 4)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only S3.`  (65 claims here) |
| Re-plot | see `plot/` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/measure.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/kcurve.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/dimsweep.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/dimperq.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/bitsvsdims.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/levels.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/export_fig4.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/export_perq.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/export_dimperq.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/tools/sec3_derived.py`
- `exp/sec2_3_analysis/exp/sec2_3_analysis/harness/src/tables.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/sec3_derived.csv`
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
| S3.12 | sym-BQ CandidateRecall@30, NQ BGE-M3 | 0.865 |
| S3.13 | ADC-BQ CandidateRecall@30, NQ BGE-M3 | 0.955 |
| S3.14 | sym-BQ p95 R*, NQ BGE-M3 | 367 |
| S3.15 | ADC-BQ p95 R*, NQ BGE-M3 | 84 |
| S3.16 | ADC-BQ CandidateRecall@30, GloVe | 0.412 |
| S3.17 | ADC-BQ CandidateRecall@30, GIST | 0 |
| S3.18 | Spearman(eta, p95 R*), pooled (configuration, quantizer) | 0.921 |
| S3.19 | number of (configuration, quantizer) pairs | 90 |
| S3.20 | Spearman(eta, p95 R*), ADC-BQ only | 0.912 |
| S3.21 | p95 R* at d=128 (Qwen3-8B MRL) | 2534 |
| S3.22 | p95 R* at d=512 (Qwen3-8B MRL) | 124 |
| S3.23 | p95 R* at d=1024 (Qwen3-8B MRL) | 47 |
| S3.24 | p95 R* at d=4096 (Qwen3-8B MRL) | 22 |
| S3.25 | beta, Quora Qwen3-0.6B | 0.889 |
| S3.26 | beta, Quora BGE-M3 | 0.896 |
| S3.27 | p95 R*, Quora Qwen3-0.6B | 35 |
| S3.28 | p95 R*, Quora BGE-M3 | 29 |
| S3.29 | beta, MS MARCO Qwen3-0.6B | 0.825 |
| S3.30 | beta, MS MARCO BGE-M3 | 0.62 |
| S3.31 | p95 R*, MS MARCO Qwen3-0.6B | 47 |
| S3.32 | p95 R*, MS MARCO BGE-M3 | 99 |
| S3.33 | beta without query instruction, MS MARCO Qwen3-0.6B | 0.523 |
| S3.34 | p95 R* without query instruction, MS MARCO | 74 |
| S3.35 | beta without query instruction, Quora Qwen3-0.6B | 0.74 |
| S3.36 | p95 R* without query instruction, Quora | 30 |
| S3.37 | median R*, MS MARCO Qwen-1024 | 16 |
| S3.38 | p95 R*, MS MARCO Qwen-1024 | 47 |
| S3.39 | p99 R*, MS MARCO Qwen-1024 | 87 |
| S3.40 | p95 R*, MS MARCO BGE-M3 (per-query array) | 99 |
| S3.41 | p99 R*, MS MARCO BGE-M3 | 195 |

## Notes

- Workload analysis harness of Secs. 2-3 (Sieun Kim, BBQETA): `harness/src` produces `harness/out/*.json` from exhaustive scoring of 20 configurations x up to 10 quantizers; `harness/work/*_perq.npz` (6.6 MB) holds the per-query R* arrays behind Figs 3-4 and the Sec 3.3 quantiles. `harness/gp` and `plot/` hold the gnuplot sources of Figs 2-4; `kcurve/` the Fig. 3(a) data.
- Sources, per Sieun's map: CandidateRecall / R* / kappa / nu / eta per configuration and quantizer = `out/<corpus>_<enc>.json` (`src/measure.py`, `src/add_arms.py`); Fig 3(a) = `src/kcurve.py`; Fig 3(b) = `src/export_cq.py`; Fig 3(c) MRL = `out/dimsweep_msmarco_qw8.json` (`src/dimsweep.py`); Fig 3(d) = `out/dimperq_msmarco_qw8.json` (`src/export_dimperq.py`); Table 2 = `out/bitsvsdims.txt` (`src/bitsvsdims.py`); query-instruction removal = `out/placebo.json` / `placebo.log`; Fig 4 = `src/export_fig4.py`.
- Two tables feed verify: `tools/flatten_outputs.py` copies every harness value into `results/sec3_values.csv`; `tools/sec3_derived.py` (needs scipy) recomputes, with the harness's own functions, what the harness prints but does not store: the eta Spearman of `src/tables.py` `diagnostics()`, per-query median / p95 / p99 R*, and the `src/levels.py` decomposition. It writes `results/sec3_derived.csv`.
- **Configuration set of the Sec 3 statistics.** The sweep has 20 configurations, two of them OpenAI ada-002 (`msmarco_ada2`, `dbpedia_ada2`, kept as data). The manuscript's Sec 3 statistics reproduce on the other 18: ADC-BQ eta Spearman 0.912 over 17 configurations (S3.20) and the `src/levels.py` variance shares 89.4 / 7.7 / 1.8 % on its settings without `msmarco_ada2` (S3.42-44). `tools/sec3_derived.py` writes both scopes (`eta.*` / `levels.7settings.*` over all, `eta.paper.*` / `levels.paper.*` without ada-002); the claims use the latter. Over all 20: 0.903, 0.959 over 153 pairs, 85.7 / 10.3 / 2.1 %.
- **Pooled eta Spearman (S3.18-19) is stale in the text.** The sentence quotes 0.921 over 90 (configuration, quantizer) pairs; the 18 configurations give 0.957 over 133 pairs (135 pairs; GIST's sym-BQ / ADC-BQ drop out because kappa is undefined). No 6-quantizer subset of the 13 full-arm configurations lands within 0.006 of 0.921, so the 90-pair set cannot be recovered. Suggested text: *Spearman 0.957 across 133 (configuration, quantizer) pairs, and 0.912 for ADC-BQ across the 17 configurations where kappa is defined.*
- **Not found in any output:** the text-embedding-3-large width sweep (3072D p95 R* 17-31), the Haar-rotation sweep (25,123 -> 21) and the whitening result (c_Q 0.74-0.76 on DBpedia). The raw text-embedding-3-large vectors exist (`data_prep/beir_embeddings`). Sieun's evidence bundle (`../supplementary/sieun_evidence`) confirms that the 2026-11-25 manuscript has no such paragraphs. Its closest evidence is the `tq1` arm (Haar rotation + sign code): on GIST it takes p95 R* from 995,299 to 3,136, which is not the printed 25,123 -> 21.
