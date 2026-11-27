| Experiment | Paper item | Claims |
|---|---|---|
| [`fig05_preprocessing_io`](fig05_preprocessing_io/README.md) | Fig 5 + Sec 4 end-to-end: query preprocessing vs exact-FP I/O wait (Qwen-1024) | 8 |
| [`fig07_adc_lut`](fig07_adc_lut/README.md) | Fig 7: (a) ADC vs sym-BQ candidate preservation (MS MARCO BGE); (b) LUT vs scalar ADC throughput (Quora BGE-M3, no calibration, T=1) | 5 |
| [`fig08_graph_build_precision`](fig08_graph_build_precision/README.md) | Fig 8 + Table 4: FP32-built vs 1-bit-built HNSW graph (MS MARCO, Quora; Qwen-1024) | 24 |
| [`fig09_20_overall_t1`](fig09_20_overall_t1/README.md) | Fig 9 (MS MARCO, DBpedia) + Fig 20 (NQ, Quora): Recall@10-QPS, 16 dataset-embedding configurations x 11 methods, single thread | 76 |
| [`fig10_recall_at_k`](fig10_recall_at_k/README.md) | Fig 10: Recall@k for k=30/50/100, with in-memory FP32 HNSW (NQ Qwen-1024; T=1) | 19 |
| [`fig11_ablation`](fig11_ablation/README.md) | Fig 11 / Sec 6.3: ablation sym-BQ -> ADC-BQ -> fixed R=75 -> calibrated R(q), per-query reads (MS MARCO Qwen-1024) | 19 |
| [`fig12_calibration`](fig12_calibration/README.md) | Fig 12 / Sec 6.3: calibration target sweep and L_visit / L_select loss split (MS MARCO Qwen-1024, ef=256) | 11 |
| [`fig13_bbq_vs_tq_same_calibration`](fig13_bbq_vs_tq_same_calibration/README.md) | Fig 13 / Sec 6.3: BBQ vs TurboQuant-1bit under the same calibration (NQ, MS MARCO; Qwen-1024; T=1) | 7 |
| [`fig14_page_io`](fig14_page_io/README.md) | Fig 14 / Sec 6.4: 4-KiB SSD pages per query vs Recall@10 (MS MARCO OpenAI-1536, Qwen-1024) | 24 |
| [`fig15_io_modes`](fig15_io_modes/README.md) | Fig 15 / Sec 6.4: sequential vs async batch vs async batch + overlap SSD refinement (MS MARCO; T=1) | 10 |
| [`fig16_17_dimension_scale`](fig16_17_dimension_scale/README.md) | Fig 16 (MRL dimension, NQ Qwen3-8B) + Fig 17 (scale 10M/50M/100M, MS MARCO Qwen-1024), Sec 6.5 | 20 |
| [`fig18_insertion`](fig18_insertion/README.md) | Fig 18 / Sec 6.5: incremental insertion 1M + 10M..100M (MS MARCO Qwen-1024; 32 threads) | 12 |
| [`fig19_colocation`](fig19_colocation/README.md) | Fig 19 / Sec 6.5: colocation under constrained DRAM (MS MARCO OpenAI-1536; 24 threads) | 5 |
| [`fig21_overall_t24`](fig21_overall_t24/README.md) | Fig 21: Recall@10-QPS, 16 dataset-embedding configurations x 11 methods, 24 search threads | 16 |
| [`sec1_hnsw_vs_ivf`](sec1_hnsw_vs_ivf/README.md) | Sec 1: HNSW vs IVF candidate generation at Recall@10 >= 0.99 (MS MARCO Qwen-1024, 1-bit ADC, T=1) | 5 |
| [`sec2_3_analysis`](sec2_3_analysis/README.md) | Secs 2-3: memory/score geometry (Fig 2), candidate preservation and score geometry (Fig 3, Tables 2-3), long-tailed demand (Fig 4) | 35 |
| [`sec5_1_query_bits`](sec5_1_query_bits/README.md) | Sec 5.1: query-side bit-width sweep with 1-bit database codes (NQ / Quora Qwen-1024, ef=256, T=1) | 2 |
| [`sec6_3_traversal_efficiency`](sec6_3_traversal_efficiency/README.md) | Sec 6.2-6.3: BBQ vs in-memory FP32 HNSW at matched recall; traversal-efficiency profile (MS MARCO / DBpedia Qwen-1024; T=1) | 14 |
| [`supplementary/ram_bbq_rbq7`](supplementary/ram_bbq_rbq7/README.md) | Supplementary (not in the paper): all in DRAM, BBQ vs RaBitQ-7bit (NQ Qwen-1024, T=1) | 0 |
