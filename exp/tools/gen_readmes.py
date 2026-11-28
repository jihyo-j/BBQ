#!/usr/bin/env python3
"""Write exp/<fig>/README.md from <fig>/title.txt, <fig>/run.txt, the files on disk and exp/claims.csv."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])

NOTES = {
 "fig05_preprocessing_io": [
  "The ADC-BQ bar uses the *fixed* R=120 (MS MARCO) / R=80 (NQ) operating point at Recall@10 >= 0.99, as in Sec. 4.",
  "`plot_qwen1024_diskann_adaptive_compare_t1.py` merges the four timing campaigns into the five-method CSV used by the figure.",
 ],
 "fig07_adc_lut": [
  "(a) `plot/make_preserve_dat.py` builds `preserve.dat` from the per-query ranks of the Sec 2-3 harness "
  "(`../sec2_3_analysis/harness/work/msmarco_bgem3_perq.npz`, MS MARCO BGE-M3, 2,000 queries, exhaustive 1M scan). "
  "`remeasure_preserve.sh` re-runs that measurement from the raw BGE-M3 vectors (`harness/src/measure.py msmarco_bgem3`) into "
  "`remeasure/` and redraws the panel. The printed panel's coordinates are kept in `plot/preserve.recovered.dat`. They agree with the "
  "stored ranks within 0.003 everywhere except ADC-BQ at R = 10, which is 0.726 printed against 0.716 measured. CandidateRecall@30 "
  "(0.848 / 0.947) is claims S3.1-S3.2.",
  "(b) is rebuilt from `results/quora_bge_m3/paired_ef_comparison_t1.csv`: same graph and efSearch, scalar FP32 ADC "
  "(`bbq/build.sh` with `ADCBQ_FORCE_FP32_DIRECT_SCALAR=1`) vs the int16 nibble LUT, no calibration, no rerank. "
  "The NQ / Quora Qwen-1024 runs of the same ablation are kept in `results/`.",
 ],
 "fig08_graph_build_precision": [
  "`plot/make_index_quality_dat.py` builds `index_quality.dat` from `results/graph_construction_precision/*_paired.csv` "
  "(`run_msmarco_quora_qwen1024_graph_construction_precision.py`). The printed figure's coordinates, recovered from its PDF, are kept "
  "as `plot/index_quality.recovered.dat`. Their Recall@10 values equal the 20 measured efSearch points of all four curves to "
  "five decimals. Their QPS differs by up to about 20 % in either direction: single-thread QPS of a separate measurement, "
  "not a different experiment.",
 ],
 "fig09_20_overall_t1": [
  "**Which table.** The paper's Figs 9 and 20 were drawn from `results/recall_qps_as_printed_t1.csv` (3,477 rows). "
  "`regen.sh` rebuilds both figures from it identically (data and panel files). `results/recall_qps_all_latest_t1.csv` is the "
  "later table in which four curves were re-measured with 20 points (NQ BGE TurboQuant and PQ, MS MARCO Qwen OPQ, DBpedia E5 "
  "TurboQuant); draw it with `CSV=../results/recall_qps_all_latest_t1.csv ./regen.sh`.",
  "The text mixes the two tables: Sec. 6.2's 'DBpedia E5 TurboQuant 201 QPS' and 'all seven above 0.995' hold in the printed "
  "table, while Appendix B's 'NQ BGE TurboQuant 416 QPS at 0.995' comes from the rerun (352.8 in the printed table; claim AB.9).",
  "`results/full16_summary_t1.csv` (printed table) and `full16_summary_rerun_t1.csv` are written by "
  "`exp/scripts/summarize_full16.py`: best QPS per (configuration, method, recall target), maximum recall, and the corpus-wide "
  "counts / medians / ranges the text quotes. SPANN and MRQ (dropped from the figures) are in `results/with_spann_es/` and "
  "`results/mrq_rabitq_ssd_summary_t1.csv`.",
  "Markers and curves are a monotone trend through each method's Pareto frontier (see the docstring of "
  "`plot/update_recall-qps_full16.py`); quote numbers from the CSV, not from the plot.",
  "'RaBitQ-SSD attains the highest maximum recall in 15 of 16' is relative to BBQ (claim F9.26); against all methods it is highest in "
  "11 (FP32 HNSW has the higher maximum in 5). Not reproduced: '9-20x lower throughput at 0.99' (3.1-27.5x, median 9.1x).",
 ],
 "fig10_recall_at_k": [
  "`results/nq_recall_k30_50_100_with_inmemory_fp.csv` adds the in-memory FP32 HNSW curves to the BBQ / TurboQuant / RaBitQ runs.",
  "TurboQuant uses the Pareto-selected fixed R = 160 / 320 / 512 for k = 30 / 50 / 100.",
 ],
 "fig11_ablation": [
  "**Panel (b)** is built by `plot/make_recall_groups.py` from the default (legacy-GT) output of "
  "`run_msmarco_qwen1024_calibration_analysis.py`, `../fig12_calibration/results/legacy_gt/per_query_target099.csv`. It groups the "
  "test queries exactly as the runner does. Every printed number reproduces: 5,489 queries (91.79 %), 51.5 / 75.4 / 119.8 FP32 reads "
  "and 0.921 / 1.035 / 1.192 ms. The `ip_aligned` and `exact_cosine_aligned` groups, which differ slightly (91.81 %, 0.904 ms, ...), "
  "are kept for reference. Fig 11 therefore uses the legacy GT throughout, like Fig 12.",
  "`plot/perquery.gp` (per-query R* vs z10 clouds, DBpedia z10 vs relevant documents) is an earlier Sec 3 figure, not part of Fig 11; "
  "its clouds are raster images reused from that figure.",
  "Fixed R=75 vs calibrated (1,109 vs 1,191 QPS; 26.0-40.5 % fewer reads and 2.7-19.9 % more QPS over the sweep) is the "
  "equal-pool / equal-queue-depth run in `results/equal_pool_qd_ablation/`.",
 ],
 "fig12_calibration": [
  "Both panels use the legacy GT of the MS MARCO run, as does Fig. 11(a): (a) the target sweep (`results/legacy_gt/target_sweep.csv`); "
  "(b) L_visit = 0.0096 (454 queries), L_select = 0.0007 (40 queries), sum 1 - Recall@10 = 0.0103, traversal share 93 % "
  "(`results/legacy_gt/loss_split_target099.csv`, from `per_query_target099.csv` by `plot/make_loss_split.py`).",
  "`plot/calibration.gp` + `make_calibration_dat.py` draw the paper figure. Under legacy GT, L_visit includes 147 queries whose in-pool GT "
  "is not reproduced by exact rerank (legacy GT is inexact at rank 10 for ~2.6 % of MS MARCO Qwen queries). The exact-cosine and ip "
  "variants are kept under `results/exact_cosine_aligned/` and `results/ip_aligned/` (exact cosine: 307 traversal-limited queries).",
 ],
 "fig13_bbq_vs_tq_same_calibration": [
  "NQ '2.01x' compares BBQ at ef=512 (651 QPS, Recall@10 0.9976) with TurboQuant at ef=1024 (324 QPS, 0.9973); at the same ef=512 "
  "TurboQuant reaches 654 QPS at 0.9963. The reads and the 1.77x distance evaluations compare the same two points.",
 ],
 "fig14_page_io": [
  "**Stale panel in the paper.** The paper PDF draws TurboQuant on OpenAI-1536 at 80 pages/query; the data and the text say 120 "
  "(R = 60 vectors x 2 pages at d = 1536, `tq_selected_r_heldout_t1.csv`). No stored file holds an 80-page TurboQuant curve. "
  "Only the TurboQuant curve of panel (a) differs; every other block of the shipped "
  "`page_io.dat` equals what `regen.sh` builds. The shipped `page_io.dat` is the printed one; `make plots` / `regen.sh` "
  "overwrite it with the data.",
 ],
 "fig15_io_modes": [
  "Claims use ef=320 (Qwen) / ef=88 (OpenAI), the first points at Recall@10 >= 0.99 (0.9908 / 0.9902 for all three modes).",
 ],
 "fig16_17_dimension_scale": [
  "Fig. 16 uses the DiskVecLab TurboQuant arm (`turboquant_tq1`); a Qdrant-native TQ1 fixed-R run is kept in `results/alt_qdrant_native_tq1/`.",
  "Fig. 17: TurboQuant's best QPS at Recall@10 >= 0.99 is 237 / 457 / 273 at 10M / 50M / 100M, so the 1.98x at 100M rests on one "
  "non-monotone point. The 105.1 GiB RaBitQ-7bit footprint is `index_bytes` in the 100M summary; BBQ's 31.3 GiB is not recorded.",
 ],
 "fig18_insertion": [
  "Inserts measured at 10M, 30M, 50M and 100M, each from the same 1M index. TurboQuant inserts faster than BBQ at 10M, 30M and 100M.",
  "**One stale point.** Panel (a) plots RaBitQ-1bit at 50M as 0.9852; the 1-bit run (`rabitq_b1_adaptive_c16_random_independent_t32.csv`) "
  "gives 0.9878 (claim F18.12). 0.9852 is `rabitq_recall_at_10` of the earlier RaBitQ-7bit series at 50M in "
  "`frozen_adcbq_vs_rabitq_c16_comparison.csv`: `make_insertion_scale_dat.py` swaps the RaBitQ series to the 1-bit run, and the "
  "printed figure kept the 7-bit value at that one point. Every other point equals the regenerated file. The shipped "
  "`insertion_scale.dat` is the printed one; `make plots` / `regen.sh` overwrite it with the data.",
 ],
 "fig19_colocation": [
  "Data: `results/te3small_ef256_t24/` from `run_msmarco_te3small1536_adcbq_colocation.py` (MS MARCO BEIR, text-embedding-3-small, "
  "the Table 5 set `sieun_msmarco_openai1536`; ef=256, 24 threads, tau_256 = 1.5349 frozen across all 24 configurations; "
  "Recall@10 0.9952 everywhere). QPS is the median of 3 fresh processes after one discarded warm-up process; `validation.json` passed. "
  "`plot/make_colocation_dat.py` builds `colocation.dat` from `summary.csv`.",
  "**Text not yet updated.** Sec 6.5 quotes 16 KB @ 50% DRAM as 112.2 -> 170.6 QPS (+52.0%) with hit ratio 57.6% -> 75.1%. Those "
  "numbers come from the earlier run on the ada-002 MS MARCO file (`msmarco_openai1536`, Recall@10 0.985), kept in "
  "`results/superseded_ada002/`. This run gives 157.0 -> 200.1 QPS (+27.5%) and 57.1% -> 71.9% (claims F19.1-5). The largest gain "
  "is +53.7% at 16 KB @ 25% (97.4 -> 149.6); colocation wins in all 12 (page, ratio) cells, +11.5% to +53.7%. "
  "`plot/colocation.recovered.dat` holds the coordinates of the earlier printed figure.",
  "Colocation is `build_colocation_physical_rows()` in `bbq/src/bbq_hnsw.cpp` (LDG partitioning + page-aware reorder), ported from the "
  "pgvector implementation of Shim et al.",
 ],
 "fig21_overall_t24": [
  "`regen.sh` rebuilds the figure from `results/recall_qps_all_latest_t24.csv` identically. `results/full16_summary_t24.csv` carries the "
  "Appendix B numbers, including the median thread speedups against the single-thread table.",
 ],
 "sec1_hnsw_vs_ivf": [
  "First measured operating point with Recall@10 >= 0.99: HNSW efSearch 512 (0.9928, 9,533 1-bit evaluations) vs IVF nprobe 768 of "
  "977 lists (0.9959, 778,383 codes). The IVF point has higher recall; this is target-threshold evidence, not an iso-recall comparison.",
 ],
 "sec2_3_analysis": [
  "Workload analysis harness of Secs. 2-3 (Sieun Kim, BBQETA): `harness/src` produces `harness/out/*.json` from exhaustive scoring "
  "of 20 configurations x up to 10 quantizers; `harness/work/*_perq.npz` (6.6 MB) holds the per-query R* arrays behind Figs 3-4 and "
  "the Sec 3.3 quantiles. `harness/gp` and `plot/` hold the gnuplot sources of Figs 2-4; `kcurve/` the Fig. 3(a) data.",
  "Sources, per Sieun's map: CandidateRecall / R* / kappa / nu / eta per configuration and quantizer = `out/<corpus>_<enc>.json` "
  "(`src/measure.py`, `src/add_arms.py`); Fig 3(a) = `src/kcurve.py`; Fig 3(b) = `src/export_cq.py`; Fig 3(c) MRL = "
  "`out/dimsweep_msmarco_qw8.json` (`src/dimsweep.py`); Fig 3(d) = `out/dimperq_msmarco_qw8.json` (`src/export_dimperq.py`); "
  "Table 2 = `out/bitsvsdims.txt` (`src/bitsvsdims.py`); query-instruction removal = `out/placebo.json` / `placebo.log`; "
  "Fig 4 = `src/export_fig4.py`.",
  "Two tables feed verify: `tools/flatten_outputs.py` copies every harness value into `results/sec3_values.csv`; "
  "`tools/sec3_derived.py` (needs scipy) recomputes, with the harness's own functions, what the harness prints but does not store: "
  "the eta Spearman of `src/tables.py` `diagnostics()`, per-query median / p95 / p99 R*, and the `src/levels.py` decomposition. "
  "It writes `results/sec3_derived.csv`.",
  "**eta Spearman (S3.18-20) is stale in the text.** The sentence quotes 0.921 over 90 (configuration, quantizer) pairs and "
  "0.912 for ADC-BQ. The current outputs (20 configurations, 155 pairs; GIST's sym-BQ / ADC-BQ drop out because kappa is undefined) "
  "give 0.959 over 153 pairs and 0.903 over 19 configurations. Sieun reports that 90 pairs = 15 configurations x 6 quantizers, "
  "but six different 6-quantizer subsets give 0.919-0.922, so the old pair set cannot be recovered; `RESULTS.md` records a third "
  "generation (0.981 / 94 pairs). Suggested text: *Spearman 0.959 across 153 (configuration, quantizer) pairs, and 0.903 for "
  "ADC-BQ across the 19 configurations where kappa is defined.*",
  "**Variance shares (S3.42-44) not reproduced.** `src/levels.py` gives query 85.7 % / embedding 10.3 % / quantizer 2.1 % for its "
  "7 settings x 5 deployable quantizers; the text says 89.4 / 7.7 / 1.8. No combination of the stored settings and quantizers we "
  "tried gives 89.4 %.",
  "**Not found in any output:** the text-embedding-3-large width sweep (3072D p95 R* 17-31), the Haar-rotation sweep "
  "(25,123 -> 21) and the whitening result (c_Q 0.74-0.76 on DBpedia). The raw text-embedding-3-large vectors exist "
  "(`data_prep/beir_embeddings`). Sieun's evidence bundle (`../supplementary/sieun_evidence`) confirms that the 2026-11-25 "
  "manuscript has no such paragraphs. Its closest evidence is the `tq1` arm (Haar rotation + sign code): on GIST it takes p95 R* "
  "from 995,299 to 3,136, which is not the printed 25,123 -> 21.",
 ],
 "sec5_1_query_bits": [
  "Query bit-width sweep at efSearch 256: 8-bit queries reach Recall@10 0.9934 on NQ against 0.9956 with FP32 queries (claim F7.1-F7.2).",
 ],
 "sec6_3_traversal_efficiency": [
  "`results/time_breakdown_20261124/` is the matched-recall profile quoted in Sec. 6.3 (9,533 vs 7,256 evaluations, 1.31 ms traversal, "
  "0.24 ms exposed refinement, 29.72 vs 1.22 MB nominal payload). The older full curves (Nov 9 / Nov 10 runs) are kept alongside.",
 ],
}

import re
all_claims = list(csv.DictReader(open(EXP / "claims.csv")))
file_of = {r["id"]: r["file"] for r in all_claims if r["file"]}
claims = defaultdict(list)
for r in all_claims:
    if not r["expected"]:
        continue  # reference rows are inputs to derived claims, not claims
    f = r["file"] or next((file_of[t] for t in re.findall(r"[A-Z][0-9.]*[0-9]", r["derive"]) if t in file_of), "")
    claims[f.split("/")[0]].append(r)

rows = []
for d in sorted([p for p in EXP.glob("fig*") if p.is_dir()] + [p for p in EXP.glob("sec*") if p.is_dir()]
                + [p for p in (EXP / "supplementary").glob("*") if p.is_dir() and (p / "title.txt").exists()]):
    title = (d / "title.txt").read_text().strip()
    run = [l for l in (d / "run.txt").read_text().split() if l]
    res = sorted(p.relative_to(d) for p in (d / "results").rglob("*") if p.is_file() and p.suffix.lower() != ".pdf")
    cl = claims.get(str(d.relative_to(EXP)).split("/")[0], [])
    md = [f"# {title}", "",
          "| | |", "|---|---|",
          f"| Verify | `make build/verify && ./build/verify --root exp --only {(lambda ps: max(dict.fromkeys(ps), key=ps.count))([c['id'].split('.')[0] for c in cl])}.`  ({len(cl)} claims here) |" if cl else "| Verify | — |",
          f"| Re-plot | `./exp/{d.relative_to(EXP)}/plot/regen.sh` |" if (d / "plot" / "regen.sh").exists() else "| Re-plot | see `plot/` |", ""]
    md += ["## Re-run the experiment", "",
           "Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). "
           "Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.", ""]
    md += [f"- `exp/scripts/{s}`" if "/" not in s else f"- `exp/{d.relative_to(EXP)}/{s}`" for s in run] + [""]
    md += ["## Stored results", "", "Result files behind the figure (copied verbatim from the original run):", ""]
    md += [f"- `{p}`" for p in res if len(res) <= 40] or [f"- {len(res)} files under `results/`"]
    md += [""]
    if cl:
        md += ["## Claims checked by `verify`", "", "| id | claim | paper |", "|---|---|---|"]
        md += [f"| {c['id']} | {c['claim']}" + (f" (= `{c['derive']}`)" if c['derive'] else "") + f" | {c['expected']} |" for c in cl] + [""]
    if NOTES.get(d.name):
        md += ["## Notes", ""] + [f"- {n}" for n in NOTES[d.name]] + [""]
    (d / "README.md").write_text("\n".join(md))
    rows.append((str(d.relative_to(EXP)), title, len(cl)))

(EXP / "MATRIX.md").write_text("\n".join(
    ["| Experiment | Paper item | Claims |", "|---|---|---|"] +
    [f"| [`{n}`]({n}/README.md) | {t} | {c} |" for n, t, c in rows]) + "\n")
print(f"{len(rows)} READMEs written")
