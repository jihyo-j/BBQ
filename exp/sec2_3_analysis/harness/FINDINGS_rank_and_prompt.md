# Two campaigns, 2026-11-18/19

Both ran on the full corpora with the stored vectors, 200 or 2,000 queries,
ADC-BQ unless stated.  Raw data: `rankexp/` (A) and `promptwhy/` (B, C).

---

## A. Score distortion is not a statement about ranks

`c_Q = nu/kappa` is the residual of a **linear fit over the whole corpus**; a
shortlist depends only on the **order** of the quantized score.  Two
interventions separate them, on MS MARCO Qwen-1024, five same-size codes:

| intervention | c_Q | p95 R\* |
|---|--:|--:|
| strictly increasing warp of the score | **x134 to x216** | **unchanged, 0 of 200 queries, all 5 codes** |
| Gaussian noise on the exact top-1000 only | **x1.009** | **43 -> 2,390 (x55)**; at 4x noise x1,400 |

Neither is something a quantizer does.  They bound how far `c_Q` can be read.

**The ordering it costs, paired bootstrap over queries (B=4,000):**

| config | smaller c_Q | larger budget | P |
|---|---|---|--:|
| MS MARCO/Qwen | PQ 0.327 | 41 vs RaBitQ's 32 (0.362) | 0.996 |
| Quora/BGE-M3 | RaBitQ 0.334 | 32 vs TQ's 24 (0.385) | 0.999 |
| Quora/BGE-M3 | PQ 0.291 | 29 vs TQ's 24 (0.385) | 0.994 |

**The residual is Gaussian.**  Excess kurtosis -0.00 to +0.02 and the 1-1e-5
quantile 4.22-4.32 against the Gaussian 4.265, on 10 of 10 (config, quantizer).
Recomputing the model below with the empirical residual instead of Phi changes
nothing by more than 0.1%.  The tail is not the problem.

**A mechanism model that does predict the budget.**  A document overtakes the
j-th answer when its residual beats the gap, so

    Rhat = p95 over (queries, u) of max_j [ 1 + sum_x Phi( -kappa(z_j - z_x)/nu - u ) ]

with u ~ N(0,1) the answer's own residual draw.  Nothing fitted, no beta.

* across 12 configurations, Spearman +0.930 with p95 R\* (c_Q alone +0.566,
  eta +0.916) **and an absolute number**: median |log2(pred/meas)| = 0.293,
  a typical factor of 1.23x over a measured range of 29 to 16,427 candidates
  (worst case MS MARCO/E5-Mistral at 1.97x);
* the MRL ladder, 18,371 -> 20 candidates (919x), tracked within 0.84-1.55x
  **with no beta anywhere**;
* per query inside one configuration: c_Q alone correlates with that query's
  R\* at -0.14 to +0.19 (uninformative); the same c_Q combined with that
  query's own score distribution, +0.49 to +0.70.

**What nothing explains: the quantizer axis.**  Within one embedding, five
same-size codes: bootstrapped Spearman for c_Q is +0.61 [+0.30,+0.90] and
+0.18 [-0.60,+0.70]; for sigma_top +0.81 [+0.50,+1.00] and +0.72 [+0.10,+1.00].
Five items cannot separate them -- the earlier sigma_top = +1.000 was a point
estimate, now withdrawn.  Substituting sigma(z), the residual measured as a
function of the exact score, does not fix it either.  PQ departs furthest from
every predictor (41 measured against 27).

### Prior work (checked 11/19) -- most of this is not new

* **Herman, CoRR abs/2609.00364 (2026)** defines the *score margin*, "the gap
  between the k-th exact score and the deeper field", and a Sect. 5.2 gap model
  `survival ~ Phi((margin + shift)/sd)` -- the same form as the model above.
  It also **explains our open problem**: "near neighbours perturb coherently --
  the naive independent-perturbation model over-predicts fidelity loss 3-4x".
  Our model is that naive one, and PQ, whose cell structure makes the coherence
  largest, is where it fails.
* **ScaNN / anisotropic quantization (Guo et al., ICML 2020)**: that mean score
  error is not what a ranking is graded on is its premise.  Our E1/E2 are a
  sharpened counterexample, not a new claim.
* **Relative contrast (He, Kumar & Chang, ICML 2012)**: the classical ancestor
  of the margin.
* Herman leaves **scalar/binary quantization unanalysed** ("recorded but
  unused") and the word *instruction* does not occur in it.  The 1-bit
  application, the budget axis R\*, and campaign B below are not covered.

---

## B. Why the Qwen instruction costs candidates on Quora

Stored artefact had only (beta, R\*).  Re-encoded both arms with the
instrumented measurement; the gate reproduces the released vectors at 0.9998.

| | beta | c_Q | z10 | eta | p95 R\* |
|---|--:|--:|--:|--:|--:|
| Quora, as prescribed | 0.889 | 0.3974 | 7.79 | 0.490 | 35 |
| Quora, no instruction | 0.740 | 0.3457 | 6.76 | **0.503** | **30** |
| MS MARCO, as prescribed | 0.825 | 0.4189 | 7.49 | 0.549 | 47 |
| MS MARCO, no instruction | 0.523 | 0.3913 | 6.36 | 0.826 | 72 |

**eta gets Quora wrong**: it rises 0.490 -> 0.503 while the budget falls.  The
c_Q drop is real (-13%) but short of the -17% that would have flipped eta.

**Separation moves the same way on both corpora** -- z10, z1-z10, z10-z100 and
beta all fall when the instruction is removed.  What differs is the ratio: on
MS MARCO the gap collapses 34% against a 7% fall in the perturbation (budget
+53%); on Quora the gap falls 18% against 13% (they nearly cancel).

**Why Quora**: `t1` rises 0.832 -> 0.923 when the instruction is removed, while
on MS MARCO it is flat (0.726 -> 0.720).  Quora is duplicate-question
retrieval: query and document are the *same kind of text*, and the documents
carry no instruction, so prepending one displaces the query off the document
manifold and even its true duplicate only reaches 0.83.

**It is a tail effect, not a shift.**  Quantile ratio (none / prescribed) on
Quora: p10 1.00, p25 1.00, p50 0.93, p75 0.89, p90 0.89, **p95 0.86**, p99
0.89.  Per query 50.0% better, 37.9% worse, 12.1% tied, median delta -0.5.
MS MARCO is a genuine location shift: 65.5% worse, median delta +4.0, p50 ratio
1.31.

**Control.**  A mismatched instruction on Quora: beta 0.299, exact top-10
overlap with prescribed 0.066, p95 R\* 356.

**Paper impact.**  The appendix wording "buys no budget at all (30 to 35)" is
wrong twice: it is not neutral (14% worse in the tail) and it is not a shift
(the median barely moves).

---

## C. Is BGE's larger budget the missing prefix?  Half of it, and it is not transferable

Nine configurations re-measured at a matched 2,000 queries (400 for DBpedia):

| corpus | Qwen (+prefix) | BGE | |
|---|--:|--:|---|
| MS MARCO | **47** | 99 (M3) | Qwen 2.1x |
| NQ | **50** | 84 (M3) | Qwen 1.7x |
| Quora | 35 | **29** (M3) | **BGE wins** |
| DBpedia | **50** | 77 (large) / 102 (base) | |

**Decomposition on MS MARCO.**  47 -> 72 (remove Qwen's prefix) -> 99 (BGE-M3).

* the prefix is **48%** of the gap, and it is pure geometry: z10 7.49 -> 6.36;
* at the no-prefix point the two are **geometrically identical** -- z10 6.36 vs
  6.35 -- and the remaining 1.38x in budget is code distortion, c_Q 0.391 vs
  0.450 (**1.15x**).

**On Quora the prefix is almost the whole difference**: 35 -> 30 -> 29, i.e.
83%, and the sign is reversed.

**The reverse test refutes the hypothesis.**  Prepending a prefix to BGE-M3,
whose dense head was trained without one, makes everything worse:

| | Quora p95 R\* | MS MARCO p95 R\* |
|---|--:|--:|
| none (as released) | **29** | **99** |
| BGE-large's own prefix | 53 | 196 |
| Qwen's prefix | 80 | 233 |
| mismatched | 128 | 456 |

All P(improves) = 0.000.  `t1` falls 0.937 -> 0.76 / 0.68 on Quora.  **c_Q
barely moves** (0.397 -> 0.419 / 0.401 / 0.406): a prefix moves the score
geometry and not the code's distortion, which is the same split as the
decomposition above, measured in the other direction.

So the prefix is worth a lot **to the model that was trained with it** and is
negative otherwise.  Corroboration: BGE-large, which *does* ship a prescribed
prefix, still loses to Qwen on DBpedia (77 against 50).

---

## D. Instrumentation

* `src/encoders.py`: `.numpy()` rejects bfloat16, which is what the checkpoint
  loads as on CPU -- the encoder path could not run without a GPU.  Cast to
  fp32 before normalising.  Backup at `encoders.py.bak`.
* `src/encoders.py`: registered `bgem3` (BAAI/bge-m3, CLS pooling, no
  instruction) so the prefix question can be put to a model that never saw one.
* Gate values: Quora/Qwen 0.99983, MS MARCO/Qwen 0.99980, Quora/BGE-M3 0.99999,
  MS MARCO/BGE-M3 0.99998.
* CPU-only encoding: torch 2.15+cpu in `/home/smrc/samsung-nvme/sieun/qvenv`;
  2,000 queries encode in 10-50 s per arm.

## Files

    rankexp/RESULTS.md, DATA.md     campaign A, tables T1-T7
    rankexp/*.json, *_perq.npz      per-query beta, c_Q, R*, ranks
    promptwhy/why_*.json            campaign B summaries
    promptwhy/why_*_perq.npz        per-query arrays, both prompt arms
    promptwhy/stored_arms.json      the nine-configuration table
    promptwhy/bge_*.json            campaign C, the reverse test
