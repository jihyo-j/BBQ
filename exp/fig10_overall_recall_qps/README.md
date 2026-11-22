# Fig 10 + Fig 21: Recall@10-QPS on 16 dataset-embedding configs, 12 methods (T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F10.`  (3 claims) |
| Re-plot | `./exp/fig10_overall_recall_qps/plot/regen.sh` |
| Paper figure | `plot/recall-qps_main.paper.pdf`, `plot/recall-qps_msmarco_dbpedia.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_diskveclab_full_matrix_all.py`
- `exp/scripts/run_diskveclab_full_matrix_adcbq_pool_top10_per_ef.py`
- `exp/scripts/run_diskveclab_full_matrix_adcbq_target99_per_ef.py`
- `exp/scripts/run_diskveclab_full_matrix_elasticsearch.py`
- `exp/scripts/run_diskveclab_tq_pq_opq_uring.py`
- `exp/scripts/run_diskveclab_tq_rerank_pareto_uring.py`
- `exp/scripts/run_diskveclab_pq_opq_rerank_pareto_uring.py`
- `exp/scripts/run_diskveclab_full_matrix_rabitq7.py`
- `exp/scripts/run_diskveclab_full_matrix_rabitq1_adaptive.py`
- `exp/scripts/run_diskveclab_full_matrix_diskann.py`
- `exp/scripts/run_nq_quora_spann_pipeann.py`
- `exp/scripts/run_diskveclab_full_matrix_mrq_rabitq_ssd.py`
- `exp/scripts/cluster_fbin_ivf256.py`
- `exp/scripts/plot_diskveclab_full_matrix_all_methods_recall_qps.py`
- `exp/scripts/spann_odirect_rerank_benchmark.cpp`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- 331 files under `results/`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F10.1 | TurboQuant QPS @>=0.99, DBpedia BGE-1024 | 1869 |
| F10.2 | BBQ QPS @>=0.99, DBpedia BGE-1024 | 1196 |
| F10.3 | MRQ FP reads/query @0.9903, NQ BGE-M3 | 1657.6 |

## Notes

- Baselines run through each system's own search and I/O path; DiskANN/SPANN/PipeANN through DiskVecLab. Patches against every external tree are in `exp/baselines/patches/`.
- `recall-qps_main.dat` starts from curves recovered from the earlier matplotlib PDF (`recall-qps_main_orig.dat`); `update_recall-qps_main_tq_pq_opq.py` replaces the TQ/PQ/OPQ curves and `add_recall-qps_main_pipeann.py` adds PipeANN. `regen.sh` reproduces the shipped `.dat` exactly.
- MRQ (NQ BGE-M3, 1,657.6 FP reads/query at Recall@10 0.9903) is in `results/mrq_nq_bge_m3_C4096_B320_direct.csv`, line 12.
