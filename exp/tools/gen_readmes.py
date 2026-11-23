#!/usr/bin/env python3
"""Write exp/<fig>/README.md from <fig>/title.txt, <fig>/run.txt, the files on disk and exp/claims.csv."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])

NOTES = {
 "fig05_preprocessing_io": [
  "The ADC-BQ bar uses the *fixed* R=120 (MS MARCO) / R=80 (NQ) operating point at Recall@10 >= 0.99, as in Sec. 4; "
  "calibrated BBQ is evaluated from Sec. 6 on.",
  "`plot_qwen1024_diskann_adaptive_compare_t1.py` merges the four timing campaigns into the five-method CSV used by the figure.",
 ],
 "fig08_graph_build_precision": [
  "`index_quality.dat` was recovered from the vector coordinates of the original matplotlib PDF; the measured numbers are in "
  "`results/graph_construction_precision/*_paired.csv` and `*_build_times.csv`.",
  "The run used `experiment/faiss_fpgraph_packed_asym_bq_hnsw` built on 2026-09-29 from an earlier snapshot of `bbq/src/bbq_hnsw.cpp` "
  "(`--graph-build fp|binary`).",
 ],
 "fig09_vs_inmemory_fp_hnsw": [
  "**Two runs.** The numbers quoted in Sec. 6.2 (635 vs 335 QPS, 1547 vs 727 QPS) come from `results/paper_run_20261109/`; "
  "the plotted curves come from the 2026-11-10 rerun in `results/clean_rerun_20261110_cpu20/`. Both are kept.",
  "`results/rerun_ndis_20261120/` holds the DBpedia distance-evaluation counters (22,977 vs 6,834) quoted in the text.",
  "`results/calibration/` holds the per-efSearch tau fitted by `fit_pool_top10_conformal` (= `bbq/calibration/calibrate.py`).",
 ],
 "fig10_overall_recall_qps": [
  "**BEIR panels.** Panels (i) MS MARCO OpenAI and (m) DBpedia OpenAI are drawn from the BEIR text-embedding-3-small "
  "sets of Table 5: all 12 methods, T=1, 3 rounds, collected by `exp/scripts/collect_beir_recall_qps.py` into "
  "`results/beir_recall_qps_t1.csv` and applied by `plot/update_beir_panels.py`. Panel (n) DBpedia BGE still shows the "
  "older 99k `bge-1024-100k` data, because only DiskANN-PQ, DiskANN-RaBitQ, SPANN and PipeANN have been run on "
  "BGE-large BEIR-400. To complete it, run on an otherwise idle machine "
  "`run_openai_beir_11methods_t1.py --skip-prepare --datasets sieun_dbpedia_entity_bge_large1024_beir400 "
  "--result-root <experiment>/openai_beir_bge400_t1`, then rerun the collector and `regen.sh`; the panel switches over "
  "automatically once all 12 methods are present.",
  "On the BEIR panels at Recall@10 >= 0.99: MS MARCO OpenAI has RaBitQ-7bit 3,387, BBQ 2,318 and TurboQuant 1,523 QPS; "
  "DBpedia OpenAI has RaBitQ-7bit 1,849, BBQ 1,639 and TurboQuant 943 QPS.",
  "Claims F10.1/F10.2 (TurboQuant 1,869 vs BBQ 1,196 QPS on DBpedia BGE-1024) refer to the older 99k panel (n).",
  "Every point is a single measurement (`rounds = 1`). The DBpedia test sets have 300 (Qwen, E5) or 800 (OpenAI, BGE) "
  "queries, and DBpedia BGE-1024 is a 99k-vector corpus (`bge-1024-100k`). See `../DATA.md`.",
  "At Recall@10 >= 0.99: BBQ is faster than TurboQuant on 13/16 configurations and faster than RaBitQ-7bit on 9/16 "
  "(RaBitQ-7bit wins all 4 Quora and 3 DBpedia configurations).",
  "Baselines run through each system's own search and I/O path; DiskANN/SPANN/PipeANN through DiskVecLab. "
  "Patches against every external tree are in `exp/baselines/patches/`.",
  "`recall-qps_main.dat` starts from curves recovered from the earlier matplotlib PDF (`recall-qps_main_orig.dat`); "
  "`update_recall-qps_main_tq_pq_opq.py` replaces the TQ/PQ/OPQ curves and `add_recall-qps_main_pipeann.py` adds PipeANN. "
  "`regen.sh` reproduces the shipped `.dat` exactly.",
  "MRQ (NQ BGE-M3, 1,657.6 FP reads/query at Recall@10 0.9903) is in `results/mrq_nq_bge_m3_C4096_B320_direct.csv`, line 12.",
 ],
 "fig11_recall_at_k": [
  "Despite its name, `run_msmarco_quora_qwen1024_recall_k30_50_100_t1.py` writes the NQ result (`DATASET_KEYS=(\"nq_qwen06_1024\",)`).",
  "TurboQuant uses the Pareto-selected fixed R = 160 / 320 / 512 for k = 30 / 50 / 100.",
 ],
 "fig12_ablation": [
  "**GT variant.** Panel (b) (91.8 % / 51.5 / 75.4 / 119.8) matches the `ip_aligned` per-query groups; "
  "the `exact_cosine_aligned` variant gives 94.2 % / 51.7 / 84.8 / 120.2. Both files are kept.",
  "**Paper typo.** The text says 91.79 % of queries; the data gives 5490/5980 = 91.806 %, i.e. 91.81 % (claim F12.8 fails on purpose).",
  "Fixed R=75 vs calibrated QPS (1,109 vs 1,191) is the equal-pool / equal-queue-depth run in `results/equal_pool_qd_ablation/`.",
 ],
 "fig13_calibration_analysis": [
  "**GT variant.** The *plotted* final-recall curve (0.9897 at target 0.99) is the legacy-GT sweep in `results/legacy_gt/`; "
  "the 307-of-5,980 traversal-limited count quoted in Sec. 6.5 comes from `results/exact_cosine_aligned/` "
  "(legacy GT reports 454; `ip_aligned` reports 307 plus 146 metric/reference mismatches). FP reads and pool coverage agree "
  "across all three. Panel (a) (L_visit / L_select) is still a placeholder in the paper.",
 ],
 "fig14_bbq_vs_tq_same_calibration": [
  "NQ \"2.01x\" compares BBQ at ef=512 (651 QPS, Recall@10 0.9976) with TurboQuant at ef=1024 (324 QPS, 0.9973). "
  "At the same ef=512, TurboQuant reaches 654 QPS at 0.9963. The reads (61.3 vs 42.1) and distance evaluations "
  "(1.77x) compare the same two points.",
  "`combine_nq_msmarco_calibrated_adcbq_turboquant.py` merges the four per-method runs into `results/summary_t1.csv`.",
 ],
 "fig15_page_io": [
  "**Stale panel in the paper.** The paper PDF draws TurboQuant on OpenAI-1536 at 80 pages/query, but the data and the text say "
  "120 (R = 60 vectors x 2 pages at d = 1536). `regen.sh` rebuilds `page_io.dat` from `results/` and fixes the panel; "
  "the shipped `page_io.dat` is the one in the paper.",
 ],
 "fig16_io_modes": [
  "The claims use ef=320 (Recall@10 0.9908 for all three modes), the first point at or above 0.99. A later manuscript "
  "revision quotes the ef=256 rows (233 / 1,031 / 1,192 QPS), whose recall is 0.9897.",
  "`results/qwen1024_summary_t1.csv` contains both datasets (the OpenAI rows were merged in); "
  "`openai1536_summary_t1.csv` is the separate priority run used as the second input of `make_io_ablation_dat.py`.",
 ],
 "fig17_18_dimension_scale": [
  "Fig. 18: TurboQuant's best QPS at Recall@10 >= 0.99 is 237 / 457 / 273 at 10M / 50M / 100M (3 rounds each), "
  "so the 1.98x at 100M depends on that single non-monotone point.",
  "The TurboQuant arm is the DiskVecLab TQ1 (`method_key = turboquant_tq1`). An alternative Qdrant-native TQ1 fixed-R run is kept "
  "in `results/alt_qdrant_native_tq1/` but is not plotted.",
  "Fig. 18 footprint: RaBitQ-7bit `index_bytes` = 112,834,728,305 B = 105.1 GiB (100M `summary_t1.csv`). "
  "The 31.3 GiB BBQ footprint is not recorded in any result file.",
  "`mrl256` is measured but not plotted.",
 ],
 "fig19_insertion": [
  "Inserts were measured at 10M, 30M, 50M and 100M (the paper text says 25M increments).",
  "TurboQuant inserts faster than BBQ at 10M, 30M and 100M (388 / 1,342 / 5,046 s vs 522 / 1,637 / 5,762 s); "
  "RaBitQ-1bit has higher recall than BBQ at 10M and 50M.",
  "The intro's \"1.8-2.2x vs RaBitQ-7bit\" is 1.82-2.13x in the data (claims I.3, I.4); Sec. 6.8's 2.06x / 1.49x is "
  "against RaBitQ-1bit.",
  "Online insertion: `bbq/src/bbq_dynamic_insert.cpp` (`./bbq/build.sh dynamic`). The 1M base uses a frozen tau; "
  "inserted vectors get sign codes only, no retraining.",
  "Input prep (`prepare_msmarco_qwen1024_101m_dynamic_inputs.py`) materialises a 414 GB aligned FP32 file.",
 ],
 "fig_lut_ablation": [
  "No calibration and no FP rerank: this isolates the scorer. The scalar build is `bbq/build.sh` with "
  "`ADCBQ_FORCE_FP32_DIRECT_SCALAR=1` (FP32 ADC per dimension, no LUT, no SIMD); same graph and efSearch for both.",
  "The runner `run_nq_quora_qwen1024_adcbq_lut_no_calibration_t1.py` was still being extended on 2026-11-24 (a Quora "
  "BGE-M3 run); the stored summary is the NQ / Quora Qwen-1024 run used by the figure.",
 ],
 "fig_query_bits": [
  "Query side only: database codes stay 1-bit sign codes; the query is quantised to b bits (int path: "
  "`scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp`) or kept in FP32. "
  "`query_improvement_transitions_t1.csv` counts the queries whose recall changes between consecutive bit-widths.",
 ],
 "fig_ram_bbq_rbq7": [
  "Everything in DRAM, one search thread. BBQ = calibrated rerank with FP32 vectors in process RAM "
  "(2,452 held-out test queries, QPS = median of 3); RaBitQ-7bit c16 is all 3,452 NQ queries (QPS = mean of 3), "
  "as stated in `make_ram_bbq_rbq7_dat.py`. The query sets differ: compare curves, not individual points.",
 ],
 "fig20_colocation": [
  "`colocation.dat` was recovered from the original matplotlib PDF; the measured QPS and graph page hit ratios are in "
  "`results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv` (4K from `page4k_*`, 8K/16K from `page{8k,16k}_qps_rerun_summary.csv`).",
  "Colocation itself is `build_colocation_physical_rows()` in `bbq/src/bbq_hnsw.cpp` (LDG partitioning + page-aware reorder, "
  "`--colocation-iterations 20 --colocation-partition-size 64`), ported from the pgvector implementation of Shim et al. [28].",
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
for d in sorted(p for p in EXP.glob("fig*") if p.is_dir()):
    title = (d / "title.txt").read_text().strip()
    run = [l for l in (d / "run.txt").read_text().split() if l]
    res = sorted(p.relative_to(d) for p in (d / "results").rglob("*") if p.is_file())
    pdfs = sorted(p.name for p in (d / "plot").glob("*.paper.pdf"))
    cl = claims.get(d.name, [])
    md = [f"# {title}", "",
          "| | |", "|---|---|",
          f"| Verify | `make build/verify && ./build/verify --root exp --only {cl[0]['id'].split('.')[0]}.`  ({len(cl)} claims) |" if cl else "| Verify | — |",
          f"| Re-plot | `./exp/{d.name}/plot/regen.sh` |",
          f"| Paper figure | {', '.join(f'`plot/{p}`' for p in pdfs) or '—'} |", ""]
    md += ["## Re-run the experiment", "",
           "Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). "
           "Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.", ""]
    md += [f"- `exp/scripts/{s}`" for s in run] + [""]
    md += ["## Stored results", "", "Result files behind the figure (copied verbatim from the original run):", ""]
    md += [f"- `{p}`" for p in res if len(res) <= 40] or [f"- {len(res)} files under `results/`"]
    md += [""]
    if cl:
        md += ["## Claims checked by `verify`", "", "| id | claim | paper |", "|---|---|---|"]
        md += [f"| {c['id']} | {c['claim']}" + (f" (= `{c['derive']}`)" if c['derive'] else "") + f" | {c['expected']} |" for c in cl] + [""]
    if NOTES.get(d.name):
        md += ["## Notes", ""] + [f"- {n}" for n in NOTES[d.name]] + [""]
    (d / "README.md").write_text("\n".join(md))
    rows.append((d.name, title, len(cl)))

(EXP / "MATRIX.md").write_text("\n".join(
    ["| Experiment | Paper item | Claims |", "|---|---|---|"] +
    [f"| [`{n}`]({n}/README.md) | {t} | {c} |" for n, t, c in rows]) + "\n")
print(f"{len(rows)} READMEs written")
