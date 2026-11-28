# BBQ paper — evidence bundle

Everything needed to re-derive the numbers printed in Sections 3 and 6 of the
BBQ manuscript, with the raw measurements included so the scripts run offline.

    scripts/run_all.sh          regenerate every transcript in out/
    CLAIMS.md                   claim -> source -> status, one row per number

Scripts 01–04 use only the Python standard library. 05 needs numpy.

## Layout

    data/
      recall-qps_full16_latest_t1.csv    Section 6: 3,477 operating points,
                                         16 dataset/embedding configurations
                                         x 11 methods, single thread, held-out
                                         test queries after a 1,000-query
                                         calibration split.
      bbqeta_out/*.json                  Section 3: offline exhaustive-scan
                                         sweep, 20 configurations x up to 10
                                         quantizer arms.  Fields per arm:
                                         kappa, nu, eta, Rstar_{p95,med,mean},
                                         candrec at 20 depths.
      bbqeta_out/dimsweep_msmarco_qw8.json   MRL width ladder, 32..4096.
      bbqeta_out/dimperq_msmarco_qw8.json    the same ladder per quantizer.
      perq/*_perq.npz                    per-query R*, kappa, eta and ranks for
                                         every arm, 20 configurations.
      misc/bitsvsdims.txt                Table 2.
      misc/placebo.log                   query-instruction ablation.
      misc/p_cdf.dat                     per-query P(R)=Pr(R*<=R), Figure 4(a).
      misc/quantizers.py.reference       the arm definitions, for reference.

    scripts/
      _common.py                  loaders, Spearman, eta_agg
      01_verify_sec3.py           28 exact checks against Section 3
      02_verify_sec6.py           every Section 6.2 / 6.5 aggregate
      03_eta_spearman.py          the eta ordering claim, recomputed
      04_rotation_ablation.py     what a random rotation buys a sign code
      05_variance_decomposition.py   the 89.4 / 7.7 / 1.8 split

    out/                          transcripts of the above

## Headline

* **Section 3: 28 of 28 exact checks pass.**  Two sentences do not pin down:
  the eta Spearman (stale — see below) and the variance decomposition
  (magnitude reproduces, exact pooling does not).
* **Section 6.2 and 6.5: every aggregate reproduces** from the csv, including
  the Table 7 / Finding 8 / Finding 9 counts.
* **Section 6.5 scaling numbers (10M/50M/100M) are not derivable here** — the
  csv is 1M-scale only.  That run's output is the one genuinely missing piece.

## The two Section 3 sentences that need attention

**1. The eta Spearman is stale.**  The generating code (`BBQETA src/tables.py`,
mirrored in `scripts/03_eta_spearman.py`) and its input are both present; the
manuscript value simply predates the current sweep.

| | manuscript | BBQETA `RESULTS.md:111` | today |
|---|---|---|---|
| pooled (configuration x quantizer) | 0.921, n=90 | 0.981, n=94 | **0.959, n=153** |
| ADC-BQ across configurations | 0.912 | — | **0.903, n=19** |

Suggested replacement: *"orders p95 R\* with Spearman 0.959 across 153
(configuration, quantizer) pairs, and 0.903 for ADC-BQ across the 19
configurations where kappa is defined."*  (GIST is excluded: every coordinate
is positive, so every sign code is all-+1 and kappa is undefined.)

90 pairs is 15 configurations x 6 arms, but eight different 6-arm subsets land
within 0.006 of 0.921, so the original subset is not recoverable.

**2. The variance decomposition is unpinned.**  89.4 / 7.7 / 1.8 reproduces on
several learned-embedding poolings (e.g. 8.0 / 2.4 / 89.6) but not uniquely,
and no stored file carries those three digits.  Either fix a pooling and
re-state, or soften to the magnitude: query variation dominates both the
embedding and the quantizer term by about an order of magnitude.

## Random rotation

There is no arm called "rotation", but `tq1` is exactly one.  From
`quantizers.py`: TurboQuant-1bit applies a Haar orthogonal `P` (QR of a
Gaussian matrix) and then the MSE-optimal 1-bit codebook, and *on unit-norm
data the global scale is constant, so the ranking of this arm equals rotated
ADC-BQ*.  Every Section 3 metric is rank-based, so the `tq1` column **is** the
random-rotation column, with no centring mixed in.

What it buys (`out/04_rotation_ablation.txt`):

* median p95 R\* **66 → 103** — rotation *costs* at the median, and lowers the
  budget in only 6 of 15 configurations.
* **GIST is rescued**: plain sign codes are degenerate there (all +1, kappa
  undefined, CandidateRecall@30 = 0.000); after rotation kappa = 0.972,
  CandidateRecall@30 = 0.526, p95 R\* falls from 995,299 to 3,136 (317x).
* The losses are concentrated on OpenAI ada2 (2.02x worse on both DBpedia and
  MS MARCO), where kappa drops.
* Mechanism is consistent with the eta framework: beta depends only on the
  exact scores and is invariant to rotation, so a rotation moves eta purely
  through kappa.  Pearson(d kappa, d log p95 R\*) = **-0.699**.

Ingredient ablation, median over the configurations where kappa is defined:

| arm | p95 R\* | CandRec@30 |
|---|--:|--:|
| plain sign (ADC-BQ) | **64.5** | **0.9682** |
| + centring | 102.0 | 0.9407 |
| + centring + scale | 74.1 | 0.9612 |
| + random rotation | 80.0 | 0.9568 |
| centring + rotation + beta (RaBitQ) | 68.3 | 0.9645 |

No added ingredient beats the plain sign code at the median — the offline
counterpart of the Section 6.5 finding that extra quantization information at
equal code size does not raise the recall ceiling.

## Provenance

The Section 3 measurements come from the BBQETA project on this machine
(`~/projects/BBQETA`: `src/measure.py`, `src/add_arms.py`, `src/dimsweep.py`,
`src/bitsvsdims.py`); per-query arrays were written to
`/home/smrc/samsung-nvme/sieun/bbqeta/` and are copied into `data/perq/` here.
The Section 6 csv was produced by the search harness and delivered on
2026-11-26.  Figure sources for the paper are a separate bundle
(`BBQ.11.25/bbq_gp_sources_compact.zip`).
