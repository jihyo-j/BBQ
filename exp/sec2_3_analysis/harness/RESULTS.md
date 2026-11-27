# BBQETA — Section 4 re-measured on the datasets Section 6 actually evaluates

2026-11-16. `projects/BBQETA`. The paper's Section 4 (κ, β, η, z₁₀, p95 R\*) was measured
on a config set that does not match Table 7 / Figure 10. This re-measures all of it on the
evaluated datasets, and extends η and R\* from ADC-BQ alone to **eight quantizers**.

## Scope (as agreed)

* **corpus axis** — Qwen3-Embedding-0.6B, 1024D: MS MARCO, NQ, Quora, DBpedia
* **encoder axis** — BGE: MS MARCO / NQ / Quora / FEVER (BGE-M3 1024D), DBpedia (BGE-large 1024D, BGE-base 768D)
* **conventional controls** — GloVe-100, GIST-960
* **quantizers** — Sym-BQ, ADC-BQ, PQ, OPQ, SQ-4bit, SQ-8bit, RaBitQ-1bit, TurboQuant-1bit
* Section 3's end-to-end QPS / I-O figures were **not** re-run (needs jihyo's SSD harness;
  timings on this 8-user box are contaminated anyway — see `lab-thread-cap`).

12 configs × 8 quantizers, nq = 2000 per config (400 where only 400 queries exist),
k = 10, cosine on unit-normalised vectors, exhaustive ranking (no graph).

## Definitions used

| symbol | definition |
|---|---|
| t, a | exact cosine and quantizer score over the **whole** corpus, per query |
| κ | Pearson(t, a) |
| ν | sd of the standardised residual = **√(1−κ²)** (exact under eq. 1) |
| z(r) | (t of the exact rank-r doc − mean_corpus t) / sd_corpus t |
| β | \|OLS slope of z(r) on ln r\|, **ranks 1..100**, per query then averaged |
| η_Q | ν_Q /(κ_Q · β) — β is a property of the exact scores, so it is fixed per corpus |
| r_j | rank of the exact rank-j doc under a; ties broken by ascending id (App. A) |
| R\* | max_{j≤10} r_j; CandidateRecall@R = (1/10)·#{j : r_j ≤ R} |

κ is obtained **exactly** from Gram matrices (Σᵢ tᵢaᵢ = qᵀ(XᵀX̂)q̃ etc.), which avoids
building an N×nq score matrix per quantizer. Cross-products are centred before
accumulation (see *Fixed bug* below).

## Validation against the paper's own numbers

Same corpora, same encoder — the harness reproduces Table 3 / Table 5 / Fig 6:

| quantity | ours | paper |
|---|--:|--:|
| MS MARCO BGE-M3 κ | **0.9117** | 0.913 |
| MS MARCO BGE-M3 p95 R\* | **99** | 103 |
| MS MARCO BGE-M3 CandRec@30 (ADC) | **0.9469** | 0.947 |
| NQ BGE-M3 p95 R\* | **84** | 84 |
| Quora BGE-M3 κ / z₁₀ / p95 R\* | **0.9288 / 7.41 / 29** | 0.929 / 7.41 / 30 |
| FEVER BGE-M3 κ / z₁₀ | **0.9165 / 5.15** | 0.916 / 5.15 |
| GloVe κ / p95 R\* | **0.804 / 17,645** | 0.803 / 17,088 |
| pooled ρ(z₁₀, R\*) | **−0.57** | −0.61 |
| MS MARCO Qwen, grid R reaching CandRec 0.99: OPQ vs ADC-BQ | **30 vs 40** | 25 vs 37 |

Independent cross-checks (`out/crosscheck_msmarco_qw06.json`):
* RaBitQ-1bit arm vs faiss `IndexRaBitQ(nb=1, qb=0)`: containment@80 **0.99920 vs 0.99940**
  (ours 132 B, faiss 136 B) — same agreement BBQFLOW step 9/12 found.
* rank + tie-break vs brute force, 4 arms × 5 queries × 10 ranks: **max \|diff\| = 0**.
* Gram κ vs direct full-corpus Pearson: **1e-8**.

## ⚠ β does not reproduce, and it is the only quantity that does not

With κ, z₁₀, p95 R\* and CandidateRecall all landing on the paper's values, β under the
stated definition lands on three of five and misses two:

| corpus (BGE-M3) | ours (1..100) | paper |
|---|--:|--:|
| MS MARCO | 0.620 | 0.600 |
| FEVER | 0.490 | 0.473 |
| NQ | 0.551 | 0.592 |
| **Quora** | **0.896** | **0.672** |
| **GloVe** | **0.278** | **0.402** |

No fit window reproduces all of them — ranks 1–10, 1–20, 1–50, 1–100, 10–100, 1–1000 and
100–1000 were all scanned, and the disagreement changes sign between Quora and GloVe, so
it is not a window choice, a log base, or mean-vs-median. Since z₁₀ matches to 2 decimals,
the z(r) curve away from r = 10 must differ. **This needs a check against the original
script before Table 3 goes to camera-ready.** Both windows are reported in `out/TABLES.md`
so the sensitivity is visible; η shifts by the β ratio and nothing else.

## Result 1 — the corpus/encoder picture holds, with the budget ordering intact

`out/table3.tex` is a paste-ready Table 3 replacement. Reading it:

* **Encoder dominates corpus.** On the same four corpora, swapping BGE-M3 → Qwen3-0.6B
  moves p95 R\* 99→47 (MS MARCO), 84→50 (NQ), 29→35 (Quora), 77→50 (DBpedia). The Qwen
  configs sit in a tight band (κ 0.922–0.930, p95 R\* 35–50); the BGE configs spread
  29–175.
* **FEVER is still the hard anchor** (η 0.89, p95 R\* 175, CandRec@30 0.899) and Quora
  still the easy one (η 0.45, p95 R\* 29) — the Section 4 story survives the dataset swap.
* **DBpedia BGE-base(768) vs BGE-large(1024)**: κ 0.893→0.911, p95 R\* 102→77 at equal
  β (0.661 / 0.657) — a clean encoder-only move, since the corpus and z₁₀ are identical.
* **η still rank-predicts the budget**: ρ(η, p95 R\*) = **+0.989**, ρ(η, CandRec@80) =
  **−0.964** across 11 configs (the paper reports +0.873 / −0.911 across its 11).

## Result 2 — η separates quantizers, and does it better than it separates corpora

This is the requested bar chart (`figs/fig_eta_rstar_bars.pdf`): η_Q on top, the p95 R\*
it buys below, grouped by quantizer.

MS MARCO / Qwen3-0.6B, 128 B/vector unless noted:

| quantizer | B/vec | κ | η | p95 R\* | CandRec@30 |
|---|--:|--:|--:|--:|--:|
| Sym-BQ | 128 | 0.835 | 0.80 | 140 | 0.937 |
| ADC-BQ | 128 | 0.922 | 0.51 | 47 | 0.984 |
| PQ | 128 | 0.950 | 0.40 | 48 | 0.982 |
| OPQ | 128 | 0.971 | 0.30 | 33 | 0.993 |
| RaBitQ-1bit | 132 | 0.940 | 0.44 | 43 | 0.987 |
| TurboQuant-1bit | 128 | 0.930 | 0.48 | 43 | 0.984 |
| SQ-4bit | 512 | 0.997 | 0.10 | 14 | 1.000 |
| SQ-8bit | 1024 | 1.000 | 0.01 | 10 | 1.000 |

* **ρ(η, p95 R\*) = +0.981 pooled over all 94 (config × quantizer) pairs**, and ≥ +0.71
  within every single config. η is a *stronger* diagnostic across quantizers than across
  corpora — which is the justification for putting it on a per-quantizer axis at all.
* **Where it breaks is ADC-BQ vs PQ, and only when the budget is already small.** PQ has
  the lower η in all 12 configs (0.40 vs 0.51 here), but it only converts that into fewer
  candidates where the budget is large: FEVER −49, DBpedia-BGE-base −34, MS MARCO-BGE-M3
  −18, DBpedia-BGE-large −18, NQ-BGE-M3 −14. On the Qwen-1024 axis and on Quora it is a
  wash or worse — MS MARCO +1, NQ ±0, DBpedia −3, Quora-Qwen **+5**, Quora-BGE-M3 **+6** —
  and on GloVe it is +6,256. So η ranks quantizers reliably *in the large* (ρ = 0.98) but
  must not be used to separate two quantizers whose η differ by less than ~0.2 at
  p95 R\* ≲ 50. This is the quantizer-level version of §4.1's "similar κ does not imply
  similar candidate recall".
* **The 1-bit family is tight**: ADC-BQ / RaBitQ-1bit / TurboQuant-1bit land at 47/43/43
  on MS MARCO-Qwen. RaBitQ's rotation + per-vector β saves 4–17 candidates on eight of the
  ten learned configs and is a wash on Quora-BGE-M3 (+3) and FEVER (+1) — consistent with
  the lab's standing result that the rotation is worth little on qwen/openai embeddings.

## Result 3 — the rotation is what makes 1-bit work on conventional descriptors

| GIST-960 | p95 R\* | CandRec@30 |
|---|--:|--:|
| Sym-BQ | 995,299 (= N) | 0.000 |
| ADC-BQ | 995,299 (= N) | 0.000 |
| TurboQuant-1bit | 3,136 | 0.526 |
| RaBitQ-1bit | 3,797 | 0.444 |
| OPQ | 465 | 0.745 |

GIST coordinates are non-negative, so `sign(x)` is **the same code for every document** and
κ is *undefined*, not small — plain BQ returns a constant score and R\* degenerates to N.
Rotating before taking the sign (RaBitQ, TurboQuant) restores a usable 1-bit code at
320× lower R\*. The paper's "every 960-dim GIST vector maps to the all-+1 code" is thus a
statement about *unrotated* BQ only, and is worth saying that way.

GloVe behaves the same in miniature: Sym-BQ 149,071 → ADC-BQ 17,645 → RaBitQ-1bit 13,358.
Note the reversal of Result 2's ordering here: on the conventional corpora the rotated
1-bit codes match or beat **PQ** at the same byte budget — GloVe 13,358 (RaBitQ) vs 23,901
(PQ), GIST 3,136 (TurboQuant) vs 3,770 (PQ), with RaBitQ a tie at 3,797 — the opposite of
the 14–49 candidate margin PQ holds on the learned corpora.

## Fixed bug worth keeping

The first pass computed per-query variance as E[a²]−E[a]², which on GIST cancels to
rounding noise and produced a **meaningless κ = 0.995 alongside R\* = 995,299**. Cross
products are now centred before accumulation (`measure.grams_c`), so a degenerate code
reports κ = undefined. Non-degenerate configs are bit-identical before and after
(κ 0.9222 / 0.9714 / 0.9400 on MS MARCO-Qwen either way).

## Files

```
src/common.py      configs + loaders (24-thread cap)
src/quantizers.py  8 arms as (transformed query, reconstruction) pairs
src/measure.py     the measurement; Gram-based kappa, exact ranks with id tie-break
src/tables.py      Table A/B/C + table3.tex
src/figures.py     fig_zrank, fig_z10_rstar, fig_eta_rstar_bars, fig_eta_vs_rstar, fig_candrec
src/crosscheck.py  faiss RaBitQ reference, brute-force ranks
out/<key>.json     per-config results      out/TABLES.md  out/table3.tex
/home/smrc/samsung-nvme/sieun/bbqeta/<key>_perq.npz   per-query arrays
```

## Caveats

* DBpedia configs have only 400 queries on disk — p95 R\* there is noisy.
* FEVER ships 6,666 queries here, not the 3,333 of Table 7; we use the first 2,000.
* NQ BGE-M3 is N = 1,000,000 here vs the 500,000 of Table 7.
* TurboQuant-1bit is a reimplementation: random rotation + the MSE-optimal 1-bit codebook,
  whose centroid ±√(2/πd) reproduces the lab's reference codebook
  (`jihyo/turboquant_hnsw/reference_codebooks/codebook_d960_b1.json`, ±0.025758 at d=960).
  On unit-norm data its *ranking* equals rotated ADC-BQ; the published "+" variant's
  per-vector correction is not modelled.
* PQ/OPQ use M = d/8 subquantizers (8 bits each) so the code size matches 1-bit BQ;
  for GloVe d=100 the nearest divisor is M=10 (10 B vs BQ's 12.5 B), i.e. PQ is slightly
  favoured there.
