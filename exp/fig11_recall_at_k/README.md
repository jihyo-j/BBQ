# Fig 11: Recall@k for k=30/50/100, with in-memory FP32 HNSW (NQ Qwen-1024; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F11.`  (5 claims) |
| Re-plot | `./exp/fig11_recall_at_k/plot/regen.sh` |
| Paper figure | `plot/recall_k.paper.pdf` |

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
| F11.1 | BBQ QPS @Recall@30>=0.99 | 933 |
| F11.2 | TurboQuant QPS @Recall@30>=0.99 | 841 |
| F11.3 | BBQ QPS @Recall@50>=0.99 | 578 |
| F11.4 | TurboQuant QPS @Recall@50>=0.99 | 580 |
| F11.5 | BBQ QPS @Recall@100>=0.99 | 303 |

## Notes

- Despite its name, `run_msmarco_quora_qwen1024_recall_k30_50_100_t1.py` writes the NQ result (`DATASET_KEYS=("nq_qwen06_1024",)`).
- TurboQuant uses the Pareto-selected fixed R = 160 / 320 / 512 for k = 30 / 50 / 100.
