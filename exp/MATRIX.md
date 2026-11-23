| Experiment | Paper item | Claims |
|---|---|---|
| [`fig05_preprocessing_io`](fig05_preprocessing_io/README.md) | Fig 5 + Sec 4 end-to-end: query preprocessing vs exact-FP I/O wait (Qwen-1024) | 8 |
| [`fig08_graph_build_precision`](fig08_graph_build_precision/README.md) | Fig 8 + Table 4: FP32-built vs 1-bit-built HNSW graph (MS MARCO, Quora; Qwen-1024) | 10 |
| [`fig09_vs_inmemory_fp_hnsw`](fig09_vs_inmemory_fp_hnsw/README.md) | Fig 9 / Sec 6.2: BBQ (SSD) vs in-memory FP32 HNSW (MS MARCO, DBpedia; Qwen-1024; T=1) | 9 |
| [`fig10_overall_recall_qps`](fig10_overall_recall_qps/README.md) | Fig 10 + Fig 21: Recall@10-QPS on 16 dataset-embedding configs, 12 methods (T=1) | 3 |
| [`fig11_recall_at_k`](fig11_recall_at_k/README.md) | Fig 11: Recall@k for k=30/50/100, with in-memory FP32 HNSW (NQ Qwen-1024; T=1) | 5 |
| [`fig12_ablation`](fig12_ablation/README.md) | Fig 12 / Sec 6.4: ablation sym-BQ -> ADC-BQ -> fixed R=75 -> calibrated R (MS MARCO Qwen-1024) | 13 |
| [`fig13_calibration_analysis`](fig13_calibration_analysis/README.md) | Fig 13 / Sec 6.5: calibration target sweep and L_visit / L_select (MS MARCO Qwen-1024, ef=256) | 5 |
| [`fig14_bbq_vs_tq_same_calibration`](fig14_bbq_vs_tq_same_calibration/README.md) | Fig 14: BBQ vs TurboQuant-1bit under the same calibration (NQ, MS MARCO; Qwen-1024; T=1) | 8 |
| [`fig15_page_io`](fig15_page_io/README.md) | Fig 15 / Sec 6.6: 4-KiB SSD pages per query vs Recall@10 (MS MARCO OpenAI-1536, Qwen-1024) | 24 |
| [`fig16_io_modes`](fig16_io_modes/README.md) | Fig 16: sequential vs async batch vs async batch + overlap (MS MARCO; T=1) | 10 |
| [`fig17_18_dimension_scale`](fig17_18_dimension_scale/README.md) | Fig 17 (MRL dimension, NQ Qwen3-8B) + Fig 18 (scale 10M/50M/100M, MS MARCO Qwen-1024) | 18 |
| [`fig19_insertion`](fig19_insertion/README.md) | Fig 19 / Sec 6.8: online insertion 1M -> 100M (MS MARCO Qwen-1024; 32 threads) | 11 |
| [`fig20_colocation`](fig20_colocation/README.md) | Fig 20 / Sec 6.9: colocation under constrained DRAM (MS MARCO OpenAI-1536; 24 threads) | 5 |
| [`fig_lut_ablation`](fig_lut_ablation/README.md) | Sec 5.1: ADC-BQ with the int16 nibble LUT vs scalar FP32 ADC (no calibration, no rerank; NQ / Quora Qwen-1024; T=1) | 0 |
| [`fig_query_bits`](fig_query_bits/README.md) | Sec 3.1: query-side bit-width sweep with 1-bit database codes (NQ / Quora Qwen-1024, ef=256, T=1) | 0 |
| [`fig_ram_bbq_rbq7`](fig_ram_bbq_rbq7/README.md) | All in DRAM: BBQ (calibrated, FP32 rerank from RAM) vs RaBitQ-7bit (NQ Qwen-1024, T=1) | 0 |
