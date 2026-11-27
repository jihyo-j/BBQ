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
  "(a) `preserve.dat` was recovered from the original figure; its numbers (CandidateRecall@30 0.847 -> 0.947 on MS MARCO BGE-M3, "
  "0.977 -> 0.994 on Quora) are checked against the Sec. 3 harness outputs (claims S3.1-S3.2) in `../sec2_3_analysis`.",
  "(b) is rebuilt from `results/quora_bge_m3/paired_ef_comparison_t1.csv`: same graph and efSearch, scalar FP32 ADC "
  "(`bbq/build.sh` with `ADCBQ_FORCE_FP32_DIRECT_SCALAR=1`) vs the int16 nibble LUT, no calibration, no rerank. "
  "The NQ / Quora Qwen-1024 runs of the same ablation are kept in `results/`.",
 ],
 "fig08_graph_build_precision": [
  "`index_quality.dat` was recovered from the vector coordinates of the original matplotlib PDF; the measured numbers are in "
  "`results/graph_construction_precision/*_paired.csv` and `*_build_times.csv`.",
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
  "**GT variant.** Panel (b) (91.8 % / 51.5 / 75.4 / 119.8) matches the `ip_aligned` per-query groups.",
  "**Not reproduced.** The paper prints 91.79 % (data: 91.81 %), and panel (b)'s latencies 0.921 / 1.035 / 1.192 ms match none of the "
  "three GT variants (ip_aligned: 0.904 / 1.014 / 1.165); they come from the earlier matplotlib figure (`perquery.dat` is recovered).",
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
  "(R = 60 vectors x 2 pages at d = 1536). The shipped `page_io.dat` is the printed one; `regen.sh` rebuilds it from `results/`.",
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
  "**Not reproduced.** Panel (a) plots RaBitQ-1bit at 50M as 0.9852; the data (`rabitq_b1_adaptive_c16_random_independent_t32.csv`) "
  "gives 0.9878 and no file contains 0.9852 (claim F18.12). The shipped `insertion_scale.dat` is the printed one; `regen.sh` gives the data.",
 ],
 "fig19_colocation": [
  "`colocation.dat` was recovered from the original matplotlib PDF; measured QPS and hit ratios are in "
  "`results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv`.",
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
  "Workload analysis harness of Secs. 2-3 (Sieun Kim): `harness/src` produces `harness/out/*.json` from exhaustive scoring of each "
  "configuration; per-query arrays (`*_perq.npz`, 6.6 MB) live with the data (`$BBQ_DATA2/bbqeta`). `harness/gp` and `plot/` hold "
  "the gnuplot sources of Figs 2-4; `kcurve/` the Fig. 3(a) candidate-expansion data; `paper_figures/` the printed panels.",
  "`tools/flatten_outputs.py` flattens every harness output into `results/sec3_values.csv` so that verify can check them "
  "(CandidateRecall, kappa / beta / f across MRL widths, Table 2 from `bitsvsdims.txt`, Table 3 p95 R*).",
  "**Not found in any output:** the text-embedding-3-large width sweep (3072D p95 R* 17-31), the Haar-rotation sweep "
  "(25,123 -> 21), the whitening result (c_Q 0.74-0.76 on DBpedia), p99 R* = 195, the 89.4 % variance share (levels.json gives "
  "85.7 % for its 7 settings) and the Spearman correlations 0.921 / 0.912. The raw text-embedding-3-large vectors exist "
  "(`data_prep/beir_embeddings`).",
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
                + [p for p in (EXP / "supplementary").glob("*") if p.is_dir()]):
    title = (d / "title.txt").read_text().strip()
    run = [l for l in (d / "run.txt").read_text().split() if l]
    res = sorted(p.relative_to(d) for p in (d / "results").rglob("*") if p.is_file())
    pdfs = sorted(p.name for p in (d / "plot").glob("*.paper.pdf"))
    cl = claims.get(str(d.relative_to(EXP)).split("/")[0], [])
    md = [f"# {title}", "",
          "| | |", "|---|---|",
          f"| Verify | `make build/verify && ./build/verify --root exp --only {max(set(c['id'].split('.')[0] for c in cl), key=[c['id'].split('.')[0] for c in cl].count)}.`  ({len(cl)} claims here) |" if cl else "| Verify | — |",
          f"| Re-plot | `./exp/{d.relative_to(EXP)}/plot/regen.sh` |" if (d / "plot" / "regen.sh").exists() else "| Re-plot | see `plot/` |",
          f"| Paper figure | {', '.join(f'`plot/{p}`' for p in pdfs) or '—'} |", ""]
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
