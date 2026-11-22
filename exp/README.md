# Experiments (Sec. 4–6)

One directory per paper figure. Each holds the result files the figure was drawn from, the gnuplot source, a
`regen.sh` that re-draws it from those results, and a README listing the runner scripts and every number the
paper quotes from it.

```
exp/
├── claims.csv          every paper number → (result file, row, column, value, tolerance)
├── verify.cpp          checks claims.csv against */results  (make verify)
├── common/
│   ├── verify.h        header-only claim checker (CSV reader, row selectors, tolerance check)
│   └── plot/           shared gnuplot style (common.gp) and legend tools
├── MATRIX.md           figure → experiment directory → number of claims
├── figNN_*/
│   ├── README.md       what was run, which result file backs which claim, caveats
│   ├── results/        result CSV/JSON copied verbatim from the original runs
│   └── plot/           NAME.gp, NAME.dat, make_*.py, regen.sh, NAME.paper.pdf (as printed)
├── scripts/            the runners, flat: they import each other as modules
├── baselines/          TurboQuant FFI (Rust, over Qdrant) and patches against every external system
├── smoke/              synthetic end-to-end run (make smoke)
├── tools/              setup_workspace.sh, relocate_paths.sh, gen_claims.py, gen_readmes.py
└── env.sh              toolchain, external systems and data locations
```

## Three levels of reproduction

**1. Verify the numbers (seconds, no hardware requirements).** `make verify` checks the 96 numbers the text quotes
against the stored results. A claim passes when exactly one result row matches its selector and the value is within
one unit of the last printed digit. Current status: 95 / 96. F12.8 fails on purpose: the paper prints
91.79 %, but the data gives 5490 / 5980 = 91.806 %.

**2. Re-draw the figures (a minute; gnuplot ≥ 5.4 with Liberation Sans, python3 + numpy).** `make plots` runs every
`regen.sh`. For each figure with a data builder, the regenerated `.dat` is identical to the shipped one, except in
the cases listed below, where the paper figure itself is stale.

**3. Re-run the experiments (days; the paper's hardware and data).** The original testbed was:
- 2 × Intel Xeon Gold 6442Y (48 cores, AVX-512)
- 1 TiB DDR5
- Samsung PM1743 NVMe
- Ubuntu 24.04, Linux 6.8, FAISS 1.14.1

Most runs are single-threaded.

```bash
source exp/env.sh                       # edit paths first
exp/tools/relocate_paths.sh /home/smrc/samsung-nvme/jihyo "$BBQ_DATA"   # only if your data lives elsewhere
make workspace                          # exp/experiment/ + the BBQ binaries under the names the runners call
python3 exp/experiment/<runner>.py      # see each figure README for its runners
```

The runners are byte-identical to the scripts that produced `results/`. They expect the original layout:
`REPO/"experiment/<name>"` for scripts, sources and binaries. `make workspace` recreates that layout under
`exp/experiment/` with symlinks. Data (embeddings, ground truth, splits, prebuilt indexes) is not in the repository.
The data roots the runners read are listed in `env.sh`.

## Paper ↔ data inconsistencies found while assembling this artifact

These are left as they are, so `verify` and `regen.sh` report the stored data and not the PDF.

| Where | What |
|---|---|
| Fig 12(b) text | 91.79 % should read 91.81 % (claim F12.8). |
| Fig 12(b) / Fig 13 | Three ground-truth variants of the same MS MARCO run are in use. Fig 12(b) matches `ip_aligned`. Fig 13's plotted recall curve matches the legacy GT. The 307 traversal-limited queries quoted in Sec. 6.5 come from `exact_cosine_aligned` (the legacy GT gives 454). All three are kept under `fig13_calibration_analysis/results/`. |
| Fig 9 | The text numbers (635 vs 335, 1,547 vs 727 QPS) come from the 2026-11-09 run. The plotted curves come from the 2026-11-10 rerun. Both are kept. |
| Fig 15 | The paper PDF draws TurboQuant on OpenAI-1536 at 80 pages/query. The data and the text say 120 (R = 60 × 2 pages at d = 1536). `regen.sh` gives the corrected panel. |
| Fig 18 | The 31.3 GiB BBQ footprint is not recorded in any result file. The 105.1 GiB RaBitQ-7bit footprint is (`index_bytes`). |

## Provenance notes

- The BBQ binaries used in the paper runs were built from successive snapshots of `bbq/src/bbq_hnsw.cpp`. The
  final ones (`faiss_adcbq_hnsw_same_pool_calibration`, `_equal_pool_qd`, 2026-11-19) match the committed source.
  Fig 8 used an earlier snapshot (2026-09-29).
- `scripts/faiss_hnsw_bq_budget_paged_codes.cpp` (TurboQuant/PQ/OPQ/sym-BQ search) and
  `scripts/faiss_fp_hnsw_resident_from_binary_graph.cpp` (in-memory FP32 HNSW) were edited after the paper runs
  (2026-11-22 and 2026-11-20). The committed versions are the current ones.
- `baselines/patches/<system>/` holds `git diff` against the commit named in `BASE` for DiskVecLab, PipeANN, SPTAG,
  RaBitQ-SSD, RESQ (MRQ), Qdrant and FAISS. Untracked files in those trees (e.g. DiskVecLab `src/BBQ`) are not
  included.
- Orchestration scripts (queue supervisors, `wait_*`, `resume_*`) are left out. None of the runners imports them.

## Not in this artifact

The workload-analysis section was produced by a separate harness, not by the code in this repository:
- Sec. 2–3: Fig 2, 3, 4 and 7, Table 2, the candidate-preservation table, and the variance decomposition.

Fig 1, Fig 6, Table 1 and Table 3 involve no measurements.
