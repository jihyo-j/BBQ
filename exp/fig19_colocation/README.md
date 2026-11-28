# Fig 19 / Sec 6.5: colocation under constrained DRAM (MS MARCO text-embedding-3-small 1536; ef=256, 24 threads)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F19.`  (5 claims here) |
| Re-plot | `./exp/fig19_colocation/plot/regen.sh` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_msmarco_te3small1536_adcbq_colocation.py`
- `exp/scripts/plot_msmarco_te3small1536_adcbq_colocation.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/calibration/ef256.summary.csv`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/calibration/pool_top10_target99.ef256.json`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/campaign.log`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/fp_graph.csv`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/msmarco_te3small1536_colocation_comparison.csv`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/msmarco_te3small1536_colocation_measured_trials.csv`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/msmarco_te3small1536_colocation_summary.csv`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/run_manifest.json`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/status.json`
- `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/validation.json`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F19.1 | 16KB baseline QPS @50% DRAM | 112.2 |
| F19.2 | 16KB colocated QPS @50% DRAM | 170.6 |
| F19.3 | 16KB baseline graph page hit ratio @50% | 0.576 |
| F19.4 | 16KB colocated graph page hit ratio @50% | 0.751 |
| F19.5 | colocation QPS gain (%) (= `(F19.2/F19.1-1)*100`) | 52 |

## Notes

- Data: `results/msmarco_te3small1536_adcbq_colocation_ef256_t24/` (CSV prefix `msmarco_te3small1536_colocation_`) from `run_msmarco_te3small1536_adcbq_colocation.py` (MS MARCO BEIR, text-embedding-3-small, the Table 5 set `sieun_msmarco_openai1536`; ef=256, 24 threads, tau_256 = 1.5349 frozen across all 24 configurations; Recall@10 0.9952 everywhere). QPS is the median of 3 fresh processes after one discarded warm-up process; `validation.json` passed. `plot/make_colocation_dat.py` builds `colocation.dat` from `msmarco_te3small1536_colocation_summary.csv`.
- The runner takes its data paths from `scripts/manifests/full16_12methods_t24.dataset_manifest.json`, which `make workspace` links to `experiment/full16_12methods_t24/dataset_manifest.json`. It stops unless the entry is text-embedding-3-small at 1536 dimensions. Its record template is the text-embedding-3-small entry `sieun_msmarco_openai1536` of `run_v051_ssd_odirect_multidataset.py`; the helper modules it imports default to the same entry.
- **Text not yet updated.** Sec 6.5 quotes 16 KB @ 50% DRAM as 112.2 -> 170.6 QPS (+52.0%) with hit ratio 57.6% -> 75.1%. Those numbers come from an earlier run that is not part of this repository. This run gives 157.0 -> 200.1 QPS (+27.5%) and 57.1% -> 71.9% (claims F19.1-5). The largest gain is +53.7% at 16 KB @ 25% (97.4 -> 149.6); colocation wins in all 12 (page, ratio) cells, +11.5% to +53.7%. `plot/colocation.recovered.dat` holds the coordinates of the earlier printed figure.
- Colocation is `build_colocation_physical_rows()` in `bbq/src/bbq_hnsw.cpp` (LDG partitioning + page-aware reorder), ported from the pgvector implementation of Shim et al.
