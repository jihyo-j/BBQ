# Claim → source map

Status key: **OK** reproduces exactly · **STALE** generating code and data are here
but the manuscript value is from an older data vintage · **UNPINNED** magnitude
reproduces, the exact pooling that produced the printed digits does not ·
**MISSING** no artifact on this machine.

Run `scripts/run_all.sh`; the transcripts in `out/` were produced by it.

## Section 3 — Candidate Preservation and Score Geometry

| Claim | Source | Status |
|---|---|---|
| MS MARCO BGE-M3 CandidateRecall@30 0.848 → 0.947 (sym-BQ → ADC-BQ) | `data/bbqeta_out/msmarco_bgem3.json` | **OK** 0.8477 / 0.9469 |
| MS MARCO BGE-M3 p95 R* 416 → 99 | same | **OK** 416.05 / 99.0 |
| NQ BGE-M3 CandidateRecall@30 0.865 → 0.955 | `data/bbqeta_out/nq_bgem3.json` | **OK** 0.8652 / 0.9549 |
| NQ BGE-M3 p95 R* 367 → 84 | same | **OK** 366.55 / 84.0 |
| GloVe CandidateRecall@30 0.412, GIST 0.000 (ADC-BQ) | `data/bbqeta_out/{glove,gist}.json` | **OK** |
| η orders p95 R*: Spearman **0.921 over 90 pairs**, **0.912** for ADC-BQ | `data/bbqeta_out/*.json` + `scripts/03_eta_spearman.py` (mirrors BBQETA `src/tables.py`) | **STALE** → now **0.959 over 153 pairs**, **0.903** over 19 configurations |
| MRL ladder: f stays 0.775–0.802 | `data/bbqeta_out/dimsweep_msmarco_qw8.json` | **OK** 0.7746–0.8015 |
| MRL ladder: κ 0.782 → 0.958, β 0.17 → 1.07 | same | **OK** 0.7823 → 0.9577, 0.1745 → 1.0690 |
| MRL ladder: p95 R* 2,534 → 124 (d 128→512), 47 → 22 (d 1024→4096) | same | **OK** |
| Table 2 (width vs precision at fixed bytes), all 8 rows | `data/misc/bitsvsdims.txt` | **OK** |
| Quora/MS MARCO β and budget: 0.889/35, 0.896/29, 0.825/47, 0.620/99 | `data/bbqeta_out/*.json` | **OK** |
| Query-instruction ablation: β 0.825→0.523 budget 47→74; β 0.889→0.740 budget 35→30 | `data/misc/placebo.log` | **OK** |
| Variance of log R*: **query 89.4% / embeddings 7.7% / quantizers 1.8%** | `data/perq/*_perq.npz` + `scripts/05_variance_decomposition.py` | **UNPINNED** — many learned-embedding poolings give ≈ 89–90 / 8 / 2, none is identifiable as *the* one |
| MS MARCO Qwen-1024 median/p95/p99 R* = 16/47/87; BGE-M3 = 23/99/195 | `data/misc/p_cdf.dat` | **OK** all six |
| Figure 3(d), six quantizers on the MRL ladder | `data/bbqeta_out/dimperq_msmarco_qw8.json` | **OK** |

## Section 6 — Evaluation

Everything below recomputes from `data/recall-qps_full16_latest_t1.csv`
(3,477 operating points, 16 configurations × 11 methods, single thread).

| Claim | Status |
|---|---|
| BBQ faster than TurboQuant-1bit in 14 of 16 at Recall@10 ≥ 0.99, 7 of 7 at ≥ 0.995 | **OK** |
| NQ Qwen-1024 @0.995: 1,275 vs 636 = 2.00× ; MS MARCO Qwen-1024 @0.99: 1,055 vs 664 = 1.59× | **OK** |
| BBQ reads 45% fewer FP32 vectors than TurboQuant (41.8 vs 76.0) | **OK** (n = 5 configurations record per-query counts) |
| Two losses, both DBpedia: 250 vs 181 and 201 vs 182 QPS; on BGE-1024 BBQ reads 77.5 vs 200 at ef 1024 vs 512 | **OK** |
| vs in-memory FP32 HNSW: faster in 15 of 16, 1.21–2.70× | **OK** |
| max-recall gap vs FP32 HNSW within 0.20 pp, 0.10 pp mean | **OK** |
| OPQ refines fewer candidates than PQ yet is slower in 12 of 16; Quora Qwen-1024 at ef 80: 30 vs 80 reads, 980 vs 1,886 QPS | **OK** |
| Table 7 / Finding 8: BBQ fastest in 58 of 74 cells, median 1.4× over runner-up | **OK** (median 1.44×) |
| Finding 9: 12 of 16 losses unreachable; fastest in 40 of 42 cells with ≥ 0.3 pp headroom | **OK** |
| Ceiling lift over BBQ: OPQ +0.00, PQ −0.04, TQ −0.06 pp; RaBitQ-SSD +0.20 in 15/16; FP32 +0.10 in 15/16 | **OK** |
| RaBitQ-7bit: median 2.10× at ≥0.95, fastest in 14/16 at 0.98, 1.19× at 0.99, reaches 0.995 in 2 of 16 | **OK** |
| SSD-resident graphs 12.7–31.6× slower at 0.99 (median 22.6× DiskANN-PQ, 19.7× PipeANN); IVF median 9.1× | **OK** |
| 10M/50M/100M scaling: 563 / 541 / 539 QPS, 1.98× and 9.59×, max recall 0.9944 | **MISSING** |
| 100M index footprint 31.3 GiB vs RaBitQ-7bit 105.1 GiB (3.36×) | **MISSING** |
| 100M mean FP32 reads 51.7 → 71.5 per query | **MISSING** |
| Table 7 DRAM column (409/473/793 B/vec etc.) | derived: codes `d/8` + ≈280 B graph, the graph term back-computed from the 451 MiB figure of the memory-constrained experiment. Replace with a measured RSS if one exists. |

The csv is 1M-scale only (`nb` = 1,000,000, or 522,931 for Quora), which is why
every 10M/50M/100M number is **MISSING** here. Those need the output of the
large-scale run.

Ablation, calibration, I/O and colocation numbers in Sections 6.3–6.5 come from
their own runs and are not covered by this csv either; they were not re-derived.

## Not in this manuscript at all

`BBQ.11.25` (the 11-25 snapshot, which is what this bundle was checked against)
contains no Haar-rotation, whitening, or text-embedding-3-large width paragraph.
If those sentences exist in a newer draft they were added elsewhere.

Related evidence that *is* here: the Section 3 sweep already measures a Haar
random rotation of the sign code, under the name `tq1` — see
`scripts/04_rotation_ablation.py` and the note at the top of it.
