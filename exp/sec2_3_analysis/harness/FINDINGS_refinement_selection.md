# Refinement selection at a fixed RaBitQ code, score and candidate pool

2026-11-20. `projects/BBQETA`, harness `src/rbq.py`, `src/refsel.py`,
`src/io/refio.c`. Every arm below reads the **same** RaBitQ-1bit code, the
**same** estimated distance, and the **same** candidate pool `L(q)` in the same
order; the arms differ only in which pool members they send to FP32, and in how
those reads are issued. That isolates the selection rule from the code, and
then the I/O structure from the selection rule.

## Setup

* 5 configurations, N = 1M each: MS MARCO / Qwen3-0.6B, NQ / Qwen3-0.6B,
  Quora / BGE-M3, FEVER / BGE-M3 (the hard anchor), MS MARCO / ada-002 (1536d,
  2 pages per vector). `efSearch` ∈ {128, 256, 512}, M = 24, efc = 200, k = 10.
* nq = 4,000 (NQ 3,452), split in half: one half calibrates every knob, the
  other half is scored. 20 random splits; ± is the sd over splits.
* RaBitQ-1bit exactly as published (one global centroid, random rotation,
  `<obar,u>` and `||x-c||` stored per vector):
  `dhat = ||x-c||^2+||q-c||^2-2||x-c||||q-c||<obar,w>/<obar,u>`,
  `B = 2||x-c||||q-c||(eps0/<obar,u>)sqrt((1-<obar,u>^2)/(D-1))`, default
  `eps0 = 1.9`.
* **Target**: `H(q)`, the FP32 top-k *inside* `L(q)`. A query is "preserved"
  when every member of `H(q)` is refined -- equivalently, when the returned
  top-k equals `H(q)`.

Validation: our estimator's ranking matches faiss `IndexRaBitQ(nb=1, qb=0)` at
**0.9998** containment@100 with the rotation aligned (0.71 if faiss is fed
unrotated vectors -- faiss applies no rotation of its own); the distance
decomposition reproduces exact L2 to 2.4e-7; `|dhat-d|/B` has mean 0.80 against
E|N(0,1)| = 0.798.

## 1. The rules, at a matched preservation of 0.99

Every rule is a one-parameter threshold family, so each is calibrated by the
*same* conformal step -- the `ceil((m+1)*0.99)`-th order statistic of the
per-query critical value on the calibration half. No rule is helped or hurt by
a search grid (an earlier grid-based version of this table overstated the
calibrated rule's advantage by 16-20%; it was a grid artefact).

Mean FP32 vectors refined per query, `efSearch` = 256, test half:

| configuration | bound @1.9 (seq) | bound tuned (seq) | bound as margin (1 batch) | **calibrated scalar (1 batch)** | bound 2-round | tuned fixed `R` | per-query floor |
|---|--:|--:|--:|--:|--:|--:|--:|
| MS MARCO / Qwen | 32.3 *(pres .982)* | 36.3 | 41.1 | **40.7** | 49.4 | 59.9 | 17.7 |
| NQ / Qwen | 34.0 *(.984)* | 37.0 | 40.7 | **42.3** | 51.6 | 64.7 | 18.4 |
| Quora / BGE-M3 | 23.2 *(.992)* | 22.3 | 24.1 | **24.9** | 26.8 | 37.5 | 13.9 |
| FEVER / BGE-M3 | 71.8 *(.964)* | 94.9 | 117.3 | **119.7** | 134.6 | 149.2 | 34.3 |
| MS MARCO / ada-002 | 33.1 *(.977)* | 38.4 | 39.3 | **42.6** | 51.4 | 63.7 | 18.6 |

As a ratio to the calibrated scalar margin (all five configurations):

| | sequential bound | bound as margin | 2-round bound | fixed `R` | floor |
|---|--:|--:|--:|--:|--:|
| range over configs | 0.79-0.90x | **0.92-1.01x** | 1.08-1.22x | **1.25-1.53x** | 0.29-0.56x |

Paired over 40 splits (same test half, `out/refsel_paired_ef256.json`):

* **calibrated scalar vs the bound used as a margin: a wash.** The difference
  is -0.4 to +3.6 reads (≤8%); the bound is ahead on 4 of 5 configurations and
  behind on one, and on MS MARCO/Qwen the calibrated rule wins 62% of splits.
* **calibrated scalar vs tuned fixed `R`: 12.4-31.4 fewer reads, 100% of
  splits**, i.e. 1.25-1.53x. This is the comparison that matters for the paper,
  because the Table 7 baselines all use a dataset-tuned fixed `R`.
* **sequential bound vs either single-batch rule: 1.9-24.2 fewer reads, 100% of
  splits** -- but only because it reads one vector at a time (§3).
* The 2-round form is *worse* than both single-batch rules: `theta` from the
  top-k prefix is a weak threshold, and it costs a round.

p95 reads tell the opposite story to the mean: the fixed prefix is constant
(60 / 65 / 38 / 149 / 64) while the adaptive rules reach 84 / 93 / 45 / 241 / 85.
Calibration buys mean reads with tail reads.

Global Recall@10 is pinned by the pool, not by the rule: every arm lands within
0.001 of the in-pool ceiling (.9909 / .9918 / .9933 / .9479 / .9858), so at
these targets the selection rule is a cost lever, not a recall lever.

`efSearch` = 128 and 512 reproduce every ordering (`out/refsel2_*.json`).

### The theorem's rho is per-answer, not per-query

Theorem 1 at rho = 0.01 delivers a per-**answer** recall of 0.9895-0.9905 --
exactly what it promises -- while the share of queries that keep *all* of
`H(q)` is only **0.899-0.912**. Both numbers are correct; the paper should not
let a reader read the first as the second.

## 2. Is the bound conservative? No -- the default is too narrow

`|dhat - d| / B` with `B` taken at `eps0 = 1` is, to two digits, the folded
standard normal:

| | p50 | p95 | p99 | p99.9 | share > 1.9 |
|---|--:|--:|--:|--:|--:|
| `|N(0,1)|` | 0.674 | 1.960 | 2.576 | 3.291 | 5.7% |
| MS MARCO / Qwen | 0.68 | 1.98 | 2.61 | 3.33 | 6.08% |
| NQ / Qwen | 0.71 | 2.04 | 2.67 | 3.39 | 6.83% |
| Quora / BGE-M3 | 0.62 | 1.85 | 2.46 | 3.18 | 4.47% |
| FEVER / BGE-M3 | 0.81 | 2.23 | 2.87 | 3.59 | 10.00% |
| MS MARCO / ada-002 | 0.68 | 1.98 | 2.61 | 3.34 | 5.99% |

So `B` is *one standard deviation* of the estimator error, and `eps0 = 1.9` is a
~94% per-candidate confidence, not slack. Candidates at the top-k boundary
(pool rank k..10k) behave like the rest (p95 1.82-2.29); members of `H(q)` are
mildly easier (p95 1.40-1.74).

The consequence runs the opposite way to the hypothesis: at the default the
**per-query preservation is 0.964-0.992, below the 0.99 target on 4 of 5
configurations**, and reaching 0.99 requires widening to `eps0` = 1.81-2.24
(sequential) or 2.19-2.97 (margin form). The default's error is a recall
shortfall, not a read surcharge -- though the recall it costs is small
(global Recall@10 0.9894 vs 0.9904 on MS MARCO/Qwen), which is presumably why
1.9 survives in practice.

Reads outside `H(q)` at `eps0 = 1.9`: 22.3 / 24.0 / 13.1 / 61.8 / 23.1 per
query, i.e. **57-86% of all reads**. That number alone proves nothing: the
per-query floor also reads 13.9-34.3 vectors to deliver 10 answers. The
informative comparison is to the floor, and every rule sits at 1.8-3.5x it.

### The per-candidate radius carries no information a scalar does not

One family interpolates the two rules: refine iff
`dhat(x) - dhat_k(q) < t * (rad(x)/mean rad)^beta`, with `beta = 0` the
calibrated scalar and `beta = 1` RaBitQ's bound in margin form; `t` is
calibrated conformally at every `beta`. The cost curve is **flat**: the full
sweep moves mean reads by at most 3.3% (ada-002 42.6 -> 39.3), by 0.5-1.0% on
MS MARCO/Qwen and Quora, and the *sign is not consistent* -- the selected
`beta` is 0.2, 1.5, 1.5, 0.5, 1.5 across the five configurations. Whatever the
per-vector `||x-c||` and `<obar,u>` metadata buys, it is not refinement reads.

## 3. Reads are not the cost axis; I/O rounds are

Real device, not a model: the FP32 vectors are written page-aligned to the
NVMe, read with `O_DIRECT` (page cache bypassed) and `io_uring`, exact
distances computed from the bytes read. `src/io/refio.c`, 2,000 held-out
queries, 3 interleaved repetitions, median.

MS MARCO / Qwen3-0.6B, `efSearch` = 256:

| configuration | FP32 reads | rounds | latency / query | p95 | refinement QPS |
|---|--:|--:|--:|--:|--:|
| bound selection + sequential I/O | 38.4 | 38.4 | 2618 us | 5574 | 382 |
| calibrated selection + sequential I/O | 44.1 | 44.1 | 3011 us | 6375 | 332 |
| **calibrated selection + one async batch** | 44.1 | 1.0 | **206 us** | 359 | 4860 |
| bound as margin + one async batch | 52.0 | 1.0 | 229 us | 405 | 4370 |
| bound, 2 rounds | 48.2 | 2.0 | 293 us | 552 | 3418 |
| tuned fixed `R` + one async batch | 63.0 | 1.0 | 246 us | 286 | 4063 |
| bound chunked at 16 (batch per chunk) | 38.7 | 3.0 | 338 us | 694 | 2955 |
| bound chunked at 32 | 43.1 | 1.7 | 267 us | 500 | 3749 |
| bound chunked at 64 | 66.7 | 1.1 | 267 us | 397 | 3745 |

Reading the first three rows the way the question asks:

* changing the **selection rule** at identical I/O mechanics: **1.00-1.19x**
  (MS MARCO 1.15x, ada-002 1.19x, FEVER 1.00x) -- and in the bound's favour;
* changing the **I/O structure** at an identical selection: **11.9-18.2x**
  (MS MARCO 14.6x, ada-002 11.9x, FEVER 18.2x).

A linear model explains the whole table:

```
latency_us = a + c_round * rounds + c_read * reads

all twelve arms        c_round = 62-66 us   c_read = 1.4-3.8 us   R^2 >= 0.999
rounds <= 5 only       c_round = 73-84 us   c_read = 2.0-3.0 us   R^2 >= 0.986
```

One I/O round costs as much as **27-37 additional batched 4 KiB reads** in the
batched regime (16-46 over the full range). That single line explains the whole
table, including why the 2-round bound and the chunked bound lose to a rule
that reads 15-30% *more* in one batch.

**The obvious rebuttal, measured.** RaBitQ's rule can be batched too, by
walking the pool in chunks and updating `theta` once per chunk. At chunk 16 it
keeps its read advantage exactly (38.7 vs 38.4) and cuts latency 7.7x, but it
still needs 3 rounds and lands at 338 us against the calibrated batch's 206 us.
At chunk 64 the first chunk runs with `theta = inf` and reads blow up to 66.7.
So the dependency is relaxable but not removable: the bound's read advantage
and single-round issue are not simultaneously available.

The 1536-d configuration adds the page axis: 2 pages per vector doubles pages
(92.8 vs 46.4) but the batch latency rises only 36% (206 -> 280 us), while the
sequential arms rise in proportion to reads.

## What this says for the paper

1. Do **not** claim that the calibrated rule refines less than RaBitQ's bound.
   At matched preservation it does not: the bound's sequential form reads
   10-21% *fewer*, and its margin form is a wash (≤8%, mixed sign).
2. The defensible selection claim is against the **tuned fixed budget** that the
   Table 7 baselines actually use: **1.25-1.53x fewer FP32 reads at a matched
   0.99 preservation, on 100% of splits**, with the caveat that the fixed
   budget has the better p95.
3. Do **not** attribute the throughput gap to a conservative bound. The bound
   is a 1-sd scale; the shipped `eps0 = 1.9` under-covers rather than
   over-reads; and the per-candidate radius is worth ≤3% of reads with no
   consistent sign.
4. The measured mechanism is the **I/O dependency**: one round ≈ 27-37 batched
   reads on this device, so deferring refinement until after traversal and
   issuing one batch is worth 11.9-18.2x, which is an order of magnitude more
   than any selection rule in this study can move. Section 5.4's framing is the
   one the data supports; Section 4's "refinement policy is the first lever"
   needs to become "refinement *issue structure* is the first lever".

## Caveats

* The pool is fixed from a RaBitQ-estimator traversal for every arm. The
  deployed RaBitQ-SSD baseline touches FP32 *during* traversal, so its true
  read count and latency are not this table's; this is a controlled ablation of
  the selection rule, not a system comparison.
* Shared 8-user box (another user's process was at 684% CPU; the device showed
  ~6% utilisation from other tenants during the runs). Repetitions were
  interleaved across arms and the median reported, but absolute microseconds
  are contaminated; read, page and round counts are exact.
* Single-threaded refinement, queue depth = batch size. Concurrency is not
  swept, and concurrency is exactly where the paper claims further overlap.
* Preservation is defined per query against `H(q)`; recall against the global
  top-k is ceilinged by traversal (0.9479 on FEVER) and is insensitive to the
  rule.
* GloVe / GIST controls were not run -- no cached FP32 graph for them here.

## Files

```
src/rbq.py                RaBitQ codes, factors, bound radius, pools by its estimator
src/refsel.py             the rules, per-query critical values, conformal campaign
src/refsel_figs.py        figs/refsel_f1_cost, _f2_bound, _f3_io
src/io/refio.c            O_DIRECT + io_uring replay              src/io/export_io.py
out/refsel2_*.json        iso-preservation table (20 splits)      out/refccurve_*.json
out/refbeta_*.json        the beta interpolation                  out/refsel_paired_ef256.json
out/refio_*.json          measured latency / reads / pages / rounds
/home/smrc/samsung-nvme/sieun/bbqeta/rbq_*.npz, crit_*.npz, io/
```
