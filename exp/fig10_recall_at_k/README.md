# Fig 10: Recall@k for k=30/50/100, with in-memory FP32 HNSW (NQ Qwen-1024; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F10.`  (19 claims here) |
| Re-plot | `./exp/fig10_recall_at_k/plot/regen.sh` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_msmarco_quora_qwen1024_recall_k30_50_100_t1.py`
- `exp/scripts/run_nq_inmemory_fp_hnsw_recall_k30_50_100_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/inmemory_fp_hnsw_manifest.json`
- `results/inmemory_fp_hnsw_summary_t1.csv`
- `results/nq_qwen1024_recall_k30_50_100_t1.csv`
- `results/nq_recall_k30_50_100_with_inmemory_fp.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F10.1 | BBQ QPS @Recall@30>=0.99 | 933 |
| F10.2 | TurboQuant QPS @Recall@30>=0.99 | 841 |
| F10.3 | BBQ / FP32 HNSW @Recall@30>=0.99 (= `F10.1/R10.1`) | 1.6 |
| F10.4 | BBQ max Recall@30 | 0.9984 |
| F10.5 | FP32 HNSW max Recall@30 | 0.9986 |
| F10.7 | BBQ QPS @Recall@50>=0.99 | 578 |
| F10.8 | TurboQuant QPS @Recall@50>=0.99 | 580 |
| F10.9 | BBQ / FP32 HNSW @Recall@50>=0.99 (= `F10.7/R10.2`) | 1.56 |
| F10.10 | BBQ max Recall@50 | 0.9975 |
| F10.11 | FP32 HNSW max Recall@50 | 0.9979 |
| F10.13 | BBQ QPS @Recall@100>=0.99 | 303 |
| F10.14 | TurboQuant QPS @Recall@100>=0.99 | 320 |
| F10.15 | BBQ / FP32 HNSW @Recall@100>=0.99 (= `F10.13/R10.3`) | 1.26 |
| F10.16 | BBQ max Recall@100 | 0.9955 |
| F10.17 | FP32 HNSW max Recall@100 | 0.997 |
| F10.19 | TurboQuant max Recall@100 | 0.9933 |
| F10.20 | RaBitQ-1bit max Recall@100 | 0.9914 |
| F10.21 | BBQ / TurboQuant @Recall@30>=0.995 (= `R10.4/R10.5`) | 1.34 |
| F10.22 | BBQ / TurboQuant @Recall@50>=0.995 (= `R10.6/R10.7`) | 1.14 |

## Notes

- `results/nq_recall_k30_50_100_with_inmemory_fp.csv` adds the in-memory FP32 HNSW curves to the BBQ / TurboQuant / RaBitQ runs.
- TurboQuant uses the Pareto-selected fixed R = 160 / 320 / 512 for k = 30 / 50 / 100.
