# Experiments (Sec. 4–6)

One directory per paper figure. Each holds the result files the figure was drawn from, the gnuplot source, a
`regen.sh` that re-draws it from those results, and a README listing the runner scripts and every number the
paper quotes from it.

```
exp/
├── claims.csv          every paper number → (result file, row, column) or a formula over other claims
├── claims.source       which manuscript the "paper" values were transcribed from
├── verify.cpp          checks claims.csv against */results  (make verify)
├── common/
│   ├── verify.h        header-only claim checker: CSV reader, row selectors, derived-claim expressions
│   └── plot/           shared gnuplot style (common.gp) and legend tools
├── MATRIX.md           figure → experiment directory → number of claims
├── DATA.md             every dataset the runners read: files, sizes, splits, provenance
├── figNN_*/, fig_*/   (fig_* = figures added after the 2026-11-22 draft: LUT ablation, query bit-width, all-DRAM)
│   ├── README.md       what was run, which result file backs which claim, caveats
│   ├── results/        result CSV/JSON copied verbatim from the original runs
│   └── plot/           NAME.gp, NAME.dat, make_*.py, regen.sh, NAME.paper.pdf (as printed)
├── scripts/            the runners, flat: they import and call each other by file name
├── baselines/          every compared system: pinned upstream commit + our patch (README.md)
├── envs/               conda environment specs (base / hnsw / rabitq-cfr) used by the runners
├── smoke/              synthetic end-to-end run (make smoke)
├── tools/              setup_workspace.sh, relocate_paths.sh, gen_claims.py, gen_readmes.py
└── env.sh              toolchain, external systems and data locations
```

## Three levels of reproduction

### 1. Verify the numbers (seconds, needs only g++)

`make verify` checks 129 claims against the stored results:
- 98 numbers printed in the text;
- 31 derived claims, i.e. the ratios and percentages the text states (16.8×, "28.0 % fewer reads", …). These are
  recomputed from the *measured* values of the claims they reference.

A claim passes when exactly one result row matches its selector and the value is within one unit of the last
printed digit.

The expected values are transcribed from the manuscript named in `claims.source` (the 19-page draft of 2026-11-22).
Another revision changes the numbers in Figs 15 and 16 (see below). If you check against a different PDF, update
the `paper` values in `tools/gen_claims.py` first.

Current status: **127 / 129**. The two failures are errors in the manuscript:

| claim | paper | data | |
|---|---|---|---|
| I.2 | Intro: "38.0×" fewer page reads than DiskANN (Qwen-1024, ≥ 0.99) | 1,042.0 / 54.3 = **19.2×** | 2,064.7 pages, which gives 38.0×, is DiskANN at Recall@10 0.9959, not at the 0.99 operating point the sentence uses |
| F12.8 | Fig 12(b): 91.79 % of queries | 5,490 / 5,980 = **91.81 %** | rounding |

### 2. Re-draw the figures (a minute; gnuplot ≥ 5.4 with Liberation Sans, python3 + numpy)

`make plots` runs every `regen.sh`. Wherever the figure has a data builder, the regenerated `.dat` file is
line-for-line identical to the shipped one, apart from comment headers. The only exception is Fig 15, where the
paper figure itself is stale (below).

### 3. Re-run the experiments (days; the paper's hardware and data)

The original testbed:
- 2 × Intel Xeon Gold 6442Y (48 cores, AVX-512)
- 1 TiB DDR5
- Samsung PM1743 NVMe
- Ubuntu 24.04, Linux 6.8

Most runs are single-threaded.

```bash
# 1. FAISS: upstream facebookresearch/faiss 46def469, no patch (baselines/patches/faiss/README.md)
# 2. data: see DATA.md
source exp/env.sh                 # edit the roots first
exp/tools/relocate_paths.sh       # rewrite every original absolute path in exp/scripts to the env.sh roots
make workspace                    # exp/experiment/ layout + BBQ binaries under the names the runners call,
                                  # patched RaBitQ headers, patched Qdrant for the TurboQuant FFI
conda env create -f exp/envs/hnsw.yml          # and rabitq-cfr.yml, base.yml, as the runner requires
python3 exp/experiment/<runner>.py             # see each figure README for its runners
```

Build status, checked 2026-11-23 against upstream FAISS `46def469` (no fork):

| Build | Result |
|---|---|
| `bbq_hnsw` | builds |
| `bbq_dynamic_insert` | builds |
| unit test | passes |
| `make smoke` | passes, including O_DIRECT and io_uring rerank |
| `make workspace` | builds 7 BBQ binaries and patches the RaBitQ headers |
| FAISS-based baseline drivers | compile |

The runners are the scripts that produced `results/`. They were changed in exactly two places, after the runs, so
that they build without the development FAISS fork:
- the `xflags` / `block_scales` guard in `faiss_hnsw_bq_budget_paged_codes.cpp`;
- `relocate_paths.sh`, if you run it.

## Paper ↔ data

Found while assembling this artifact and during an external review. None of these is "fixed" in the data:
`verify` and `regen.sh` report what is stored.

### Numbers or statements that do not match the data

| Where | Paper | Data |
|---|---|---|
| Intro | page reads 38.0× lower than DiskANN | 19.2× at the stated operating point (claim I.2) |
| Intro | insertion 1.8–2.2× faster than RaBitQ-7bit | 1.82–2.13×, i.e. "1.8–2.1×" (`fig19_insertion`) |
| Fig 12(b) | 91.79 % | 91.81 % |
| Fig 15 (PDF figure) | TurboQuant on OpenAI-1536 drawn at 80 pages/query | 120 (R = 60 × 2 pages at d = 1536); the text already says 120. `regen.sh` fixes the panel |
| Fig 18 | BBQ footprint 31.3 GiB | not recorded in any result file (RaBitQ-7bit's 105.1 GiB is) |
| Sec 6.8 | inserts "in increments of 25M" | measured at 10M, 30M, 50M and 100M |
| Sec 6.1 | "FAISS 1.14.1" | an unreleased upstream `main` commit (`46def469`, 2026-05-01, version string 1.14.1); pin the commit |
| Sec 6.1 | dimensionality studied with "Qwen3-Embedding 0.6B / 4B / 8B" | Fig 17 uses MRL truncations of the single 8B checkpoint, as Sec 6.7 says |
| Sec 5.3 | separate tuning and calibration query sets | every split is calibration (1,000; 100–200 for DBpedia) + test; ef_search is swept on the test queries |
| Table 5 | MS MARCO: 6,980 queries | MS MARCO BGE-1024 (`msmarco-bge-1024`): 10,000 |

### Mixed sources inside one figure

| Where | What |
|---|---|
| Figs 10 / 21 | Table 5 describes the BEIR sets: text-embedding-3-small for MS MARCO and DBpedia, and BGE-large BEIR-400 for DBpedia. The figure panels (i) MS MARCO OpenAI, (m) DBpedia OpenAI and (n) DBpedia BGE were first drawn from older files (ada-002; 999k/1,000-query DBpedia OpenAI; 99k/1,000-query DBpedia BGE). <br>• Panels (i) and (m) are now redrawn from the BEIR campaign: all 12 methods, T = 1, 3 rounds (`results/beir_recall_qps_t1.csv`). <br>• Panel (n) keeps the old 99k data until 8 methods are run on BGE-large BEIR-400 (see `fig10_overall_recall_qps/README.md`). <br>On the BEIR data, at R@10 ≥ 0.99, BBQ beats TurboQuant on both panels (2,318 vs 1,523 and 1,639 vs 943 QPS), but RaBitQ-7bit beats BBQ on both (3,387 and 1,849 QPS). The Sec 6.3 sentences about these datasets need to be re-read against the new panels. |
| Fig 9 | The text numbers (635 vs 335, 1,547 vs 727 QPS) come from the 2026-11-09 run. The plotted curves come from the 2026-11-10 rerun. Both are kept. |
| Figs 12(b), 13 | Three ground-truth variants of one MS MARCO run are in use, all kept under `fig13_calibration_analysis/results/`: <br>• Fig 12(b) matches `ip_aligned`. <br>• Fig 13's plotted recall curve matches the legacy GT. <br>• The 307 traversal-limited queries in Sec 6.5 come from `exact_cosine_aligned` (the legacy GT gives 454). |

### Comparisons that hold in the data but deserve qualification

| Where | Observation |
|---|---|
| Fig 14, NQ "2.01× faster" | BBQ ef = 512 (651 QPS, R@10 0.9976) vs TurboQuant ef = 1024 (324 QPS, 0.9973). At the same ef = 512, TurboQuant gets 654 QPS at 0.9963. The "42.1 vs 61.3 reads" and "1.77× distance evaluations" compare the same two ef points. |
| Fig 18, "1.98× TurboQuant at 100M" | TurboQuant's best QPS at R@10 ≥ 0.99 is 237 / 457 / 273 at 10M / 50M / 100M. This is not monotone, so the ratio is sensitive to that run. |
| Fig 19 | TurboQuant inserts faster than BBQ at 10M, 30M and 100M. BBQ is faster at 50M only. RaBitQ-1bit has higher recall than BBQ at 10M (0.9899 vs 0.9893) and 50M (0.9878 vs 0.9849). |
| Intro vs Sec 6.8 | The intro's insertion speed-up is against RaBitQ-7bit; Sec 6.8 uses RaBitQ-1bit (2.06× / 1.49×). Both hold, but against different baselines. |
| Fig 10 / 21 | RaBitQ-7bit is faster than BBQ at R@10 ≥ 0.99 on 8 of 16 configurations (all 4 Quora, 3 DBpedia, and MS MARCO OpenAI on the BEIR data). TurboQuant is faster on 2 of 16, plus DBpedia BGE on the old 99k data. |
| Fig 10 / 21 measurement | Points from the original matrix are single runs (`rounds = 1`); the DBpedia Qwen/E5 test sets have 300 queries. The BEIR panels (i), (m) use 3 rounds, as do Figs 17 and 18. |

### Another manuscript revision

A later revision of the manuscript prints different numbers in two places:

**Fig 16:** 233 / 1,031 / 1,192 QPS.
- These are the **ef = 256** rows of `fig16_io_modes/results/qwen1024_summary_t1.csv`.
- At ef = 256, Recall@10 is 0.9897, i.e. below the "Recall@10 ≥ 0.99" the text states.
- The revision in `claims.source` uses ef = 320 (222.0 / 928.0 / 993.5 at 0.9908), which is consistent with the text.

**Sec 6.6, page reads.**
- That revision prints 115.7 / 404.3 / 2,064.0 pages for OpenAI-1536 at 0.98, and 2,064.7 for DiskANN on
  Qwen-1024 at 0.99.
- 404.3 is a Qwen-1024 sweep point.
- 2,064.7 is DiskANN at Recall@10 0.9959.
- 115.7 appears in no result file.
- The revision in `claims.source` matches `operating_points.csv` exactly.

## Provenance notes

- `bbq/src/bbq_hnsw.cpp` is the 2026-11-23 source. It adds the compile-time `ADCBQ_FORCE_FP32_DIRECT_SCALAR` switch
  used only by the LUT ablation (`fig_lut_ablation`); the default build is unchanged. The query bit-width variant is
  `scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp`. Both carry the same upstream-FAISS guard.
- The BBQ binaries used in the paper runs were built from successive snapshots of `bbq/src/bbq_hnsw.cpp`. The final
  ones (`faiss_adcbq_hnsw_same_pool_calibration`, `_equal_pool_qd`, 2026-11-19) match the committed source. Fig 8
  used an earlier snapshot (2026-09-29).
- `bbq/src/bbq_dynamic_insert.cpp` (Fig 19) was built on 2026-11-18. On 2026-11-19 the core gained a
  `fixed_rerank_n` argument, and the call site was updated to pass the previous behaviour.
- `scripts/faiss_hnsw_bq_budget_paged_codes.cpp` and `scripts/faiss_fp_hnsw_resident_from_binary_graph.cpp` were edited
  after the paper runs (2026-11-22 and 2026-11-20). The committed versions are the current ones.
- RaBitQ v0.5.1 binding: the cumulative patch is the 2026-11-17 source state (`baselines/patches/rabitq_v051/`).
- Orchestration scripts (queue supervisors, `wait_*`, `resume_*`) are left out; no runner imports or calls them.

## Not in this artifact

The workload analysis in Sec 2–3 was produced by a separate harness, not by the code in this repository. That covers
Figs 2, 3, 4 and 7, Table 2, the candidate-preservation table, and the variance decomposition. It is the evidence
for the "blessing of dimensionality" claim, so it needs to be added before submission.

Fig 1, Fig 6, Table 1 and Table 3 involve no measurements.
