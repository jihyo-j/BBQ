# Revised Sections 3 and 4

Second pass, after review. Built against the Overleaf snapshot in
`Blessing BQ exp/overleaf`: 19 pages, Findings 1-7 in order on pages 4, 5, 5,
6, 7, 11, 16, and the new table lands as Table 2 on page 5.
`paper_with_revised_sec3.pdf` is that build.

## What the review changed

- **The `1.33x` bound in Finding 4 was wrong and is now `1.40x`.** It came
  from checking only k = 1, 10 and 100. Over all of k = 1..100 the three
  asymmetrically scored 1-bit codes are widest apart at k = 62 on
  MS MARCO BGE-M3, where TurboQuant-1bit needs 534 candidates and RaBitQ-1bit
  380. The finding also said "one 1-bit code for another", which wrongly
  included sym-BQ at 3.44x; it now names the three codes it covers and
  contrasts query quantization separately.
- **The k finding is now stated per answer retained.** Needing more candidates
  for more answers is partly arithmetic, so the number that carries the point
  is R*(k)/k: 1.4, 2.0 and 4.3 candidates per answer at k = 1, 10, 100 on
  MS MARCO Qwen-0.6B, and 1.7, 3.5, 8.6 on MS MARCO BGE-M3.
- **c_Q does not depend on k**, so Section 4 no longer says the distortion
  ordering is read at k = 10; only the budget columns are.
- **The k subsection now comes before the per-query one** (3.1, 3.2, the k
  subsection, then query variation), so the long tail runs straight into
  Section 4's refinement argument.
- **The judged-answer paragraph is cut to two clauses** and now says plainly
  that semantic relevance and FP32 neighbor preservation are different
  questions.
- **The matched-byte numbers are a table** (Table 2) instead of prose.
- **Finding 2 is scoped** to the checkpoint and corpus measured.
- **Section 4 states that only ADC-BQ and TurboQuant share the common search
  path**, so the DiskANN bars in Figure 5 are not read as a controlled
  quantizer comparison.

## Drop in

| file | where |
|---|---|
| `tex/3.BQonModernEmbedding.tex` | replaces the same file |
| `tex/4.SearchEfficiency.tex` | replaces the same file |
| `tex/6.experiment.tex` | replaces the same file -- **only two Finding numbers**, 5->6 and 6->7 |
| `figures/bbq/f3b.pdf` | replaces panel (b); no .tex change needed |

The `\comment{...}` block holding the old Section 3 is left untouched.

## What changed, and why

**Figure 3(b) now runs to k = 100.** Same six configurations, same colours and
markers as (a), same two-strip axis break. The lower strip is logarithmic
because at k = 100 the Transformer budgets reach the hundreds. The statistic
is the mean over queries, not p95: a p95 is fixed by its top 5% and jumps from
one k to the next, which is unreadable across two decades of k. The caption
says so.

**Section 3.2 gains the matched-byte control.** The MRL ladder confounds width
with code length -- a sign code is d/8 bytes, so 32 to 4096 dimensions is 4 to
512 B/vector. A reviewer will say the blessing is just more bits. The new
paragraph holds the bytes fixed and the ground truth fixed (the exact FP32
top-10 of the full 4096D vectors, so every arm is asked for the same answers)
and spends the budget either on width at one bit or on precision at a
truncated width. Finding 2 now carries that.

**Section 3.4 is new: "The Budget Is a Function of k".** R* is reported at a
k the evaluation fixes, and that choice moves the number far more than the
code does. Finding 4.

**Section 3.2's description of panel (d) was stale** -- it said the horizontal
axis lists quantizer variants, where the panel plots c_Q against dimension,
and it explained a "+cs" series the panel does not draw. Rewritten.

**Section 3.1** now says "at k = 10" where it reads the budget off (b),
since (b) is no longer a single-k panel.

**Section 4** recaps the two new results in its opening sentence, scopes
Table 2 to k = 10 with a pointer to 3.4, and its Finding becomes 5.

## Every new number, and where it comes from

All measurements are offline, exhaustive FP32 ranking, no graph search.

**Matched bytes** (`src/bitsvsdims.py` -> `data/bitsvsdims.txt`).
Qwen3-8B on MS MARCO, N = 1M, 500 queries. Ground truth is the exact FP32
top-10 of the full 4096D vectors for every arm. MRL truncation is a prefix,
renormalized. Mean R*:

| budget | 1-bit | 4-bit | 8-bit | PQ |
|---|---|---|---|---|
| 512 B | **13.2** (4096D) | 17.4 (1024D) | 30.8 (512D) | 14.0 (4096D, M=512) |
| 128 B | **30.1** (1024D) | 118.8 (256D) | 555.4 (128D) | 295.4 (4096D, M=128) |

**R*(k)** (`src/kcurve.py` -> `data/kcurve_*.dat`, and `src/p_kcorpus6.dat`
for the panel). 1500 queries per configuration, 1000 on GIST. Mean R* under
ADC-BQ at k = 1 / 10 / 100: Quora/Qwen 1.17 / 17.0 / 294, NQ/Qwen 1.46 / 21.3
/ 487, MS MARCO/Qwen 1.37 / 20.4 / 426, MS MARCO/BGE 1.65 / 34.6 / 864,
GloVe 217 / 4436 / 41562, GIST 508506 / 906521 / 988748. Same corpus, other
codes at k = 100 on MS MARCO/Qwen: TurboQuant-1bit 420, RaBitQ-1bit 365.

**Judged answers per query** (`src/raw_material.py` -> `data/raw_material.txt`).
From the released judgments: MS MARCO 94.4% of queries have exactly one,
NQ 80.7%, Quora 76.9%; DBpedia median 16 with 62.5% above ten. MS MARCO's
judgments come inline with the query file and are already keyed by corpus row
(verified: judged documents sit at median rank 1 and 99.5% inside the top-1000
under the vectors); Quora's needed its corpus id list recovered from the BEIR
parquet.

**Sanity check against the existing text.** The same pipeline reproduces the
numbers already in Section 3: beta 0.825 / 0.889 / 0.620 / 0.896 and p95 R*
47 / 35 / 99 / 29 on MS MARCO-Qwen, Quora-Qwen, MS MARCO-BGE, Quora-BGE.
beta is the mean over queries and R* the p95, which is the convention the
section already uses.

## Things to look at before sending this out

- **Finding numbering shifts.** Section 3 now holds 1-4, so Section 4's is 5
  and Section 6's are 6 and 7. That is the only change to `6.experiment.tex`.
- **A pre-existing LaTeX error is still there**, not introduced here:
  `6.experiment.tex:684` has `\swlee{+ recent VLDB/SIGMOD papers?}}` with one
  closing brace too many ("Too many }'s"). It is also in the original build.
- **A commented-out Korean "Finding 1."** is still sitting in
  `6.experiment.tex`; left alone.
- **Scope of the matched-byte result.** One checkpoint, one corpus, 500
  queries, exhaustive scan. It does not include each arm's query-side
  preprocessing, which Section 4 is where that cost lives. Widening it to a
  second encoder or corpus is the obvious next measurement.
- **The judged-answer paragraph** rests on BEIR's shallow pooling, which the
  text states. If a reviewer objects, that paragraph can be cut without
  touching Finding 4, which stands on the k sweep alone.
