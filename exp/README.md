# Experiments

One directory per paper figure (`figNN_*`) or text-only experiment (`secN_*`). Each holds:
- the result files the figure or text was drawn from;
- the gnuplot source;
- a `regen.sh` that re-draws the figure from those results;
- a README listing the runner scripts and every number the paper quotes from it.

Figure numbers follow the manuscript named in `claims.source`.

```
exp/
├── claims.csv          every paper number → (result file, row(s), column[, min|max|…]) or a formula over other claims
├── claims.source       which manuscript the "paper" values were transcribed from
├── verify.cpp          checks claims.csv against */results  (make verify)
├── common/
│   ├── verify.h        header-only checker: CSV reader, row selectors, aggregates, derived-claim expressions
│   └── plot/           shared gnuplot style (common.gp) and legend tools
├── MATRIX.md           paper item → experiment directory → number of claims
├── DATA.md             every dataset the runners read: files, sizes, splits, provenance, embedding scripts
├── figNN_*/  secN_*/
│   ├── README.md       what was run, which result file backs which claim, caveats
│   ├── results/        result CSV/JSON copied verbatim from the original runs
│   └── plot/           NAME.gp, NAME.dat, builders, regen.sh, NAME.paper.pdf (as printed)
├── sec2_3_analysis/    the Secs 2-3 workload-analysis harness (exhaustive scoring; Figs 2-4, Tables 2-3)
├── supplementary/      measured but not in the paper
├── data_prep/          embedding generation for the BEIR text-embedding-3 sets
├── scripts/            the runners, flat: they import and call each other by file name
├── baselines/          every compared system: pinned upstream commit + our patch (README.md)
├── envs/               conda environment specs (base / hnsw / rabitq-cfr)
├── smoke/              synthetic end-to-end run (make smoke)
├── tools/              gen_claims.py, gen_readmes.py, setup_workspace.sh, relocate_paths.sh
└── env.sh              toolchain, external systems and data locations
```

## Three levels of reproduction

**1. Verify the numbers (seconds, needs only g++).** `make verify` checks 312 claims transcribed from the manuscript,
covering Secs 1–6 and Appendix B:
- numbers printed in the text and tables;
- ranges, which are checked as the min/max over the rows they summarise;
- ratios and percentages, which are recomputed from the measured values they reference;
- corpus-wide counts and medians (e.g. "14 of 16", "median 2.10×"). These are computed by
  `scripts/summarize_full16.py` from the 16-configuration tables.

A claim passes when its value is within one unit of the last printed digit.

Current status: **303 / 312**. Every failure is a place where the manuscript and the data disagree; they are listed
below.

**2. Re-draw the figures (a minute; gnuplot ≥ 5.4, python3 + numpy).** `make plots` runs every `regen.sh`.

| Figures | Result |
|---|---|
| Figs 5, 7(b), 9, 10–13, 15–17, 20, 21 | The regenerated `.dat` (and the Fig 9/20/21 panel files) are identical to the ones behind the printed figures. |
| Fig 14 | The printed TurboQuant panel is stale; `regen.sh` rebuilds it from the data. |
| Fig 18 | One point was edited by hand; `regen.sh` rebuilds it from the data. |
| Figs 7(a), 8, 19 and the Fig 11(b) clouds | Drawn from coordinates recovered from earlier matplotlib PDFs. Their numbers are checked against the result CSVs instead. |

**3. Re-run the experiments (days; the paper's hardware and data).** The original testbed:
- 2 × Intel Xeon Gold 6442Y (AVX-512)
- 1 TiB DDR5
- Samsung PM1743 NVMe
- Ubuntu 24.04, Linux 6.8

```bash
# FAISS: upstream facebookresearch/faiss 46def469, no patch (baselines/patches/faiss/README.md)
# data:  DATA.md (embeddings: data_prep/beir_embeddings)
source exp/env.sh                 # edit the roots first
exp/tools/relocate_paths.sh       # rewrite every original absolute path in exp/scripts to the env.sh roots
make workspace                    # exp/experiment/ layout, BBQ binaries, patched RaBitQ headers and Qdrant
python3 exp/experiment/<runner>.py             # see each README for its runners
```

## Paper ↔ data

### Claims that fail `verify`

These are the manuscript's errors, or numbers no result file supports.

| claim | where | paper | data |
|---|---|---|---|
| F9.27–28 | Sec 6.2 | RaBitQ-SSD at "9–20× lower throughput" at 0.99 | 3.1–27.5× (median 9.1×, claim S66.5) |
| F11.12 | Fig 11(b) | 91.79 % | 5,490 / 5,980 = 91.81 % |
| F11.16–17 | Fig 11(b) | latency 0.921 / 1.192 ms | 0.904 / 1.165 ms. The printed values come from the earlier matplotlib figure. |
| F18.12 | Fig 18(a) | RaBitQ-1bit at 50M drawn at 0.9852 | 0.9878; no file contains 0.9852 |
| S66.7–8 | Sec 6.6 | SSD-resident graphs "12.7–31.6× slower" at 0.99 | 8.2–66.4× over the 38 (configuration, system) pairs. The paper's definition is not stated and no script produced these numbers. |
| AB.9 | App B | NQ BGE TurboQuant 416 QPS at 0.995 | 352.8 in the table the figures were drawn from. 416 comes from the later rerun of that curve (see below). |

### Mixed sources

- **Figs 9/20 vs the text.** The figures were drawn from `fig09_20_overall_t1/results/recall_qps_as_printed_t1.csv`.
  Four curves were later re-measured with 20 points:
  - NQ BGE TurboQuant and PQ;
  - MS MARCO Qwen OPQ;
  - DBpedia E5 TurboQuant.

  The rerun is `recall_qps_all_latest_t1.csv`. The text uses both versions: Sec 6.2 ("DBpedia E5 TurboQuant 201";
  "all seven" at 0.995) matches the printed table, while Appendix B (AB.9) matches the rerun. Use one version
  throughout.
- **Fig 12 vs Fig 11(b).** Fig 12 and Fig 11(a) use the legacy GT of the MS MARCO run. Fig 11(b) matches the
  `ip_aligned` variant. All three variants are kept under `fig12_calibration/results/`.

### Statements not checkable from any stored result

| Where | Statement |
|---|---|
| Sec 3.2 | text-embedding-3-large width sweep (3072D p95 R* 17–31) |
| Sec 3.2 | Haar-rotation sweep (25,123 → 21) |
| Sec 3.2 | whitening (c_Q 0.74–0.76 on DBpedia) |
| Sec 3.2 | Spearman 0.921 / 0.912 |
| Sec 3.3 | p99 R* = 195 |
| Sec 3.3 | 89.4 % of the variance (`levels.json` gives 85.7 % for its 7 settings) |
| Sec 6.2 | OPQ "fewer candidates than PQ and still slower in 12 of 16" (the definition is not stated) |
| Sec 6.6 | "58 of 74 settings", "40 of 42", "16 settings … 12 exceed its maximum recall" |
| Sec 6.6 | "3–5× more memory", "280 B per vector" (analytic, not measured) |
| Sec 6.5 | BBQ's 31.3 GiB index footprint (RaBitQ-7bit's 105.1 GiB is recorded) |

### Comparisons that hold but deserve qualification

| Where | Qualification |
|---|---|
| Fig 13, NQ "2.01×" | Compares BBQ at ef = 512 with TurboQuant at ef = 1024. At the same ef = 512, TurboQuant gets 654 QPS at Recall@10 0.9963. |
| Fig 17, "1.98× at 100M" | TurboQuant's best QPS at 0.99 is not monotone in scale (237 / 457 / 273 at 10M / 50M / 100M). |
| Fig 18 | TurboQuant inserts faster than BBQ at 10M, 30M and 100M. |
| Sec 5.3 vs runs | ef_search is selected on the test queries: every split is calibration + test, with no separate tuning set. |
| Sec 6.1 | "FAISS 1.14.1" is the unreleased upstream commit `46def469` (version string 1.14.1). |

## Provenance notes

- **BBQ core.** `bbq/src/bbq_hnsw.cpp` is the 2026-11-23 source. It adds the compile-time
  `ADCBQ_FORCE_FP32_DIRECT_SCALAR` switch, used only by the LUT ablation (Fig 7(b)). The query bit-width variant is
  `scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp`. Both build on upstream FAISS.
- **BBQ binaries.** The binaries used in the paper runs were built from successive snapshots of
  `bbq/src/bbq_hnsw.cpp`. Fig 8 used the 2026-09-29 snapshot.
- **Online insertion.** `bbq/src/bbq_dynamic_insert.cpp` (Fig 18) was built on 2026-11-18. Its call into the core was
  updated for an argument added on 2026-11-19, keeping the same behaviour.
- **Baseline drivers.** `scripts/faiss_hnsw_bq_budget_paged_codes.cpp` and
  `scripts/faiss_fp_hnsw_resident_from_binary_graph.cpp` were edited after the paper runs.
- **Secs 2–3.** The harness (`sec2_3_analysis/`), the Fig 9/20 source table and the embedding scripts
  (`data_prep/beir_embeddings/`) are Sieun Kim's. They are included with her permission.
- **Orchestration.** Scripts such as queue supervisors, `wait_*` and `resume_*` are left out; no runner imports or calls
  them.

Fig 1, Fig 6 and Table 1 involve no measurements.
