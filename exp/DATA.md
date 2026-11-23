# Data

The datasets are **not** in this repository: about 120 GB of FP32 vectors for the 16 main configurations, plus the
MRL ladder, the 100M scale set and prebuilt indexes. This file lists:
- exactly what every runner reads,
- where it lived on the original testbed (paths relative to the roots in `env.sh`),
- what is known about how each file was produced.

## File formats

| ext | layout |
|---|---|
| `.fbin` | `int32 n, int32 d`, then `n*d` float32, row-major. Vectors are L2-normalised (cosine = inner product). |
| `.fp32.header4096.fbin` | same payload with the header padded to 4096 B, so every row starts 4 KiB-aligned for `O_DIRECT`. `bbq_hnsw --fp-rerank-path` writes it on first use. |
| `.ibin` | `int32 n, int32 k`, then `n*k` int32 neighbour ids (exact cosine top-k; ties broken by id). |
| splits | `seed<S>.cal<C>.test<T>.{cal,test}.{query.fbin,gt.ibin}`, a random permutation of the query set, made by `make_split` in `scripts/run_adcbq_calibrated_threshold_overlap.py` (seed 11 for all paper runs). |

## The BEIR sets of Table 5 (text-embedding-3-small, BGE-large)

Table 5's OpenAI rows and the DBpedia BGE row are BEIR sets prepared for the `openai_beir_11methods_t1` campaign
(manifest: `$BBQ_DATA/openai_beir_11methods_t1/dataset_manifest.json`, built by
`scripts/prepare_openai_beir_11methods_inputs.py`):

| key | corpus | model | dim | base rows | queries (cal / test) | used by |
|---|---|---|---|---|---|---|
| sieun_msmarco_openai1536 | MS MARCO | OpenAI text-embedding-3-small | 1536 | 1,000,000 | 6,980 (1000 / 5980) | Fig 10/21 panel (i), Figs 15, 16 |
| sieun_dbpedia_entity_openai1536 | DBpedia-entity | OpenAI text-embedding-3-small | 1536 | 1,000,000 | 400 (200 / 200) | Fig 10/21 panel (m) |
| sieun_dbpedia_entity_bge_large1024_beir400 | DBpedia-entity | BGE-large-en-v1.5 (BEIR-400) | 1024 | 1,000,000 | 400 (200 / 200) | Fig 10/21 panel (n), once all 12 methods are measured |
| sieun_nq_openai1536, sieun_quora_openai1536 | NQ, Quora | OpenAI text-embedding-3-small | 1536 | 1M / 522,931 | 3,452 / 10,000 | not in the paper figures |

All 12 methods are measured at T = 1 (3 rounds) on the first two. They are collected into
`fig10_overall_recall_qps/results/beir_recall_qps_t1.csv` by `scripts/collect_beir_recall_qps.py`. Only DiskANN-PQ,
DiskANN-RaBitQ, SPANN and PipeANN are measured on the BGE-large BEIR-400 set so far.

## The original 16-configuration matrix (registry: `scripts/run_v051_ssd_odirect_multidataset.py`)

The first full matrix (`diskveclab_full_matrix`, rounds = 1) used the files below. Three of its rows predate the
BEIR sets above:
- `msmarco_openai1536` is the ada-002 `msmarco-v1-openai-ada2-1M` file;
- `dbpedia_openai1536` is `openai-1536-1M` (999k rows, 1,000 queries);
- `dbpedia_bge1024` is `bge-1024-100k` (99k rows, 1,000 queries).

Figs 10/21 now draw panels (i) and (m) from the BEIR sets. Panel (n) still shows `bge-1024-100k` until the BEIR-400
BGE runs are complete.

Counts are taken from `fig10_overall_recall_qps/results/split_manifest.csv`, written by the runner.

| key | corpus | model (as labelled in the runner) | dim | base rows | cal / test queries | base file (under its root) |
|---|---|---|---|---|---|---|
| nq_bge_m3_vanilla | NQ | BGE-M3 | 1024 | 1,000,000 | 1000 / 2452 | `$BBQ_DATA/nq_sieun_adcbq_rabitq/nq_vanilla_train.fbin` |
| nq_qwen06_1024 | NQ | Qwen3-Embedding-0.6B | 1024 | 1,000,000 | 1000 / 2452 | `$BBQ_DATA/nq_sieun_adcbq_rabitq/nq_vanilla-qw06_train.fbin` |
| nq_nemotron4096 | NQ | NVIDIA Nemotron-8B | 4096 | 1,000,000 | 1000 / 2452 | `$BBQ_DATA/nq_sieun_adcbq_rabitq/nq_vanilla-nem8_train.fbin` |
| nq_e5mistral4096 | NQ | E5-Mistral-7B | 4096 | 1,000,000 | 1000 / 2452 | `$BBQ_DATA/e5/nq_vanilla-e5m7_train.fbin` |
| quora_bge_m3_vanilla | Quora | BGE-M3 | 1024 | 522,931 | 1000 / 9000 | `$BBQ_DATA/quora_sieun_adcbq_rabitq/quora_vanilla_train.fbin` |
| quora_qwen1024 | Quora | Qwen3-Embedding-0.6B | 1024 | 522,931 | 1000 / 9000 | `$BBQ_DATA/quora_sieun_adcbq_rabitq/quora_vanilla-qw06_train.fbin` |
| quora_nemotron4096 | Quora | NVIDIA Nemotron-8B | 4096 | 522,931 | 1000 / 9000 | `$BBQ_DATA/quora_sieun_adcbq_rabitq/quora_vanilla-nem8_train.fbin` |
| quora_e5mistral4096 | Quora | E5-Mistral-7B | 4096 | 522,931 | 1000 / 9000 | `$BBQ_DATA/e5/quora_vanilla-e5m7_train.fbin` |
| msmarco_openai1536 | MS MARCO | OpenAI-1536 (file: `msmarco-v1-openai-ada2-1M`) | 1536 | 1,000,000 | 1000 / 5980 | `$ANN_BENCH_DATA/deep_1B/3072/marco/msmarco-v1-openai-ada2-1M_train.fbin` |
| msmarco_bge1024 | MS MARCO | BGE-1024 | 1024 | 1,000,000 | 1000 / 9000 | `$ANN_BENCH_DATA/msmarco/1m/bge_1m/msmarco-bge-1024-train.fbin` |
| msmarco_qwen1024 | MS MARCO | Qwen3-Embedding-0.6B | 1024 | 1,000,000 | 1000 / 5980 | `$BBQ_DATA/qwen/msmarco-1m__vanilla-qwen06b_train.fbin` |
| msmarco_e5mistral4096 | MS MARCO | E5-Mistral-7B | 4096 | 1,000,000 | 1000 / 5980 | `$BBQ_DATA/e5/msmarco-1m__vanilla-e5mistral_train.fbin` |
| dbpedia_openai1536 | DBpedia | OpenAI-1536 (file: `openai-1536-1M`) | 1536 | 999,000 | 200 / 800 | `$ANN_BENCH_DATA/deep_1B/3072/openai/openai-1536-1M-train.fbin` |
| dbpedia_bge1024 | DBpedia | BGE-1024 | 1024 | **99,000** | 200 / 800 | `$ANN_BENCH_DATA/deep_1B/3072/dbpedia-100k/bge/bge-1024-100k_train.fbin` |
| dbpedia_qwen1024 | DBpedia | Qwen3-Embedding-0.6B | 1024 | 1,000,000 | 100 / 300 | `$BBQ_DATA/qwen/dbpedia-entity__vanilla-qwen06b_train.fbin` |
| dbpedia_e5mistral4096 | DBpedia | E5-Mistral-7B | 4096 | 1,000,000 | 100 / 300 | `$BBQ_DATA/e5/dbpedia-entity__vanilla-e5mistral_train.fbin` |

Query and ground-truth files sit next to each base file, with the same stem and a `_query.fbin` / `_neighbors*.ibin`
(or `.ibin`) suffix.

Table 5 describes the BEIR sets above for the OpenAI rows and DBpedia BGE. One row of this matrix still differs from
Table 5: MS MARCO BGE-1024 (`msmarco-bge-1024`) has 10,000 queries; Table 5 gives 6,980 for MS MARCO.

## Other sets

| Used by | Data | Prepared by (in `scripts/`) | Source on the testbed |
|---|---|---|---|
| Fig 17 (MRL 256–4096) | NQ, Qwen3-Embedding-8B, truncated + renormalised per width, GT recomputed per width | `prepare_nq_qwen8_mrl_inputs.py` | `$BBQ_DATA/nq_sieun_adcbq_rabitq` → `$BBQ_DATA/nq_qwen8_mrl_four_methods_t1` (17 GB) |
| Fig 18 (10M / 50M / 100M) | MS MARCO passages, Qwen3-Embedding-0.6B | `prepare_msmarco_qwen1024_100m_scalability_inputs.py` | `$BBQ_DATA2/qwen100m` |
| Fig 19 (1M + 100M inserts) | same 101M pool, 4 KiB-aligned | `prepare_msmarco_qwen1024_101m_dynamic_inputs.py` | `$BBQ_DATA2/{msmarco-qwen1024,qwen100m}` → 414 GB aligned file |
| Fig 15/16 OpenAI-1536 | BEIR MS MARCO, OpenAI | `prepare_openai_beir_11methods_inputs.py` | `$BBQ_DATA/openai_beir_11methods_t1/dataset_manifest.json` |

## Not available from this repository

- **The embedding step.** The scripts that encoded the corpora (model revisions, query instructions, truncation and
  normalisation) are not in this tree. They were run separately by co-authors, and the runners start from the
  `.fbin` files above.
- **Hosting.** The vectors are too large for git. They need either a data release (e.g. a Hugging Face dataset with
  the `.fbin` / `.ibin` files and their sha256) or the encoding scripts plus the exact model revisions.
