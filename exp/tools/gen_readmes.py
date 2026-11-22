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
  "`combine_nq_msmarco_calibrated_adcbq_turboquant.py` merges the four per-method runs into `results/summary_t1.csv`.",
 ],
 "fig15_page_io": [
  "**Stale panel in the paper.** The paper PDF draws TurboQuant on OpenAI-1536 at 80 pages/query, but the data and the text say "
  "120 (R = 60 vectors x 2 pages at d = 1536). `regen.sh` rebuilds `page_io.dat` from `results/` and fixes the panel; "
  "the shipped `page_io.dat` is the one in the paper.",
 ],
 "fig16_io_modes": [
  "`results/qwen1024_summary_t1.csv` contains both datasets (the OpenAI rows were merged in); "
  "`openai1536_summary_t1.csv` is the separate priority run used as the second input of `make_io_ablation_dat.py`.",
 ],
 "fig17_18_dimension_scale": [
  "The TurboQuant arm is the DiskVecLab TQ1 (`method_key = turboquant_tq1`). An alternative Qdrant-native TQ1 fixed-R run is kept "
  "in `results/alt_qdrant_native_tq1/` but is not plotted.",
  "Fig. 18 footprint: RaBitQ-7bit `index_bytes` = 112,834,728,305 B = 105.1 GiB (100M `summary_t1.csv`). "
  "The 31.3 GiB BBQ footprint is not recorded in any result file.",
  "`mrl256` is measured but not plotted.",
 ],
 "fig19_insertion": [
  "Online insertion: `bbq/src/bbq_dynamic_insert.cpp` (`./bbq/build.sh dynamic`). The 1M base uses a frozen tau; "
  "inserted vectors get sign codes only, no retraining.",
  "Input prep (`prepare_msmarco_qwen1024_101m_dynamic_inputs.py`) materialises a 414 GB aligned FP32 file.",
 ],
 "fig20_colocation": [
  "`colocation.dat` was recovered from the original matplotlib PDF; the measured QPS and graph page hit ratios are in "
  "`results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv` (4K from `page4k_*`, 8K/16K from `page{8k,16k}_qps_rerun_summary.csv`).",
  "Colocation itself is `build_colocation_physical_rows()` in `bbq/src/bbq_hnsw.cpp` (LDG partitioning + page-aware reorder, "
  "`--colocation-iterations 20 --colocation-partition-size 64`), ported from the pgvector implementation of Shim et al. [28].",
 ],
}

claims = defaultdict(list)
for r in csv.DictReader(open(EXP / "claims.csv")):
    claims[r["file"].split("/")[0]].append(r)

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
        md += [f"| {c['id']} | {c['claim']} | {c['expected']} |" for c in cl] + [""]
    if NOTES.get(d.name):
        md += ["## Notes", ""] + [f"- {n}" for n in NOTES[d.name]] + [""]
    (d / "README.md").write_text("\n".join(md))
    rows.append((d.name, title, len(cl)))

(EXP / "MATRIX.md").write_text("\n".join(
    ["| Experiment | Paper item | Claims |", "|---|---|---|"] +
    [f"| [`{n}`]({n}/README.md) | {t} | {c} |" for n, t, c in rows]) + "\n")
print(f"{len(rows)} READMEs written")
