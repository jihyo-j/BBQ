# Repository notes on this bundle

`README.md`, `CLAIMS.md`, `scripts/` and `out/` are Sieun Kim's evidence bundle (`~sieun/BBQ_evidence`, 2026-11-28),
copied verbatim and included with her permission. `data/` links to the identical files already in this repository
instead of duplicating them:

| bundle path | repository file |
|---|---|
| `data/bbqeta_out/*.json` | `sec2_3_analysis/harness/out/` |
| `data/perq/` | `sec2_3_analysis/harness/work/` |
| `data/recall-qps_full16_latest_t1.csv` | `fig09_20_overall_t1/results/recall_qps_as_printed_t1.csv` (byte-identical) |
| `data/misc/{bitsvsdims.txt,placebo.log}` | `sec2_3_analysis/harness/out/` |
| `data/misc/p_cdf.dat` | `sec2_3_analysis/plot/fig4/p_cdf.dat` |
| `data/misc/quantizers.py.reference` | `sec2_3_analysis/harness/src/quantizers.py` |

Run `scripts/run_all.sh`; script 05 needs numpy. All five transcripts regenerate byte-identically from these links.

The Sec 6.6 method-selection definitions of `02_verify_sec6.py` are ported into `scripts/summarize_full16.py`
(`dbudget_*`, `bbq_over_{diskann_pq,diskann_rabitq,pipeann}_*`), so `make verify` checks them as S66.7-S66.17.

Where this repository reads the transcripts differently from `CLAIMS.md`:

- **"40 of 42" (Finding 9) is not OK as printed.** The bundle's own transcript gives 47 of 50 at the stated 0.3 pp
  headroom. A 0.5 pp headroom gives exactly 40 of 42, and the two exceptions are then the 1.02x Quora E5-4096 ties the
  text names. At 0.3 pp a third exception appears (DBpedia BGE-1024 at 0.99, TurboQuant 1.38x faster). The text should
  say 0.5 pp (claims S66.16-17).
- **"12.7-31.6x slower" is DiskANN-PQ's range only.** The sentence attributes it to all SSD-resident graphs, but
  DiskANN-RaBitQ spans 11.5-29.9x and PipeANN 8.2-66.4x (claims S66.7-8 check the DiskANN-PQ range).
- **"1.21-2.70x vs FP32 HNSW"** is the range over the 15 configurations BBQ wins; the 16th is 0.92x.
- **Variance decomposition (05).** The closest matches come from searching thousands of configuration and quantizer
  subsets, so they are not a reproduction. The harness's own `src/levels.py` reproduces 89.4 / 7.7 / 1.8 exactly once its
  ada-002 setting (`msmarco_ada2`) is left out (claims S3.42-44); with it, the result is 85.7 / 10.3 / 2.1.
- **Configuration set.** Likewise the ADC-BQ eta Spearman reproduces (0.912) on the 18 configurations without the two
  ada-002 ones; the bundle's 0.959 / 0.903 are over all 20.
- **OPQ "12 of 16"** is marked OK in `CLAIMS.md`, but no script in the bundle computes it.
- **Haar rotation, whitening, text-embedding-3-large.** The bundle states that these paragraphs are not in the
  2026-11-25 manuscript. `04_rotation_ablation.py` measures a Haar rotation through the `tq1` arm, but it does not
  produce the printed 25,123 -> 21.
