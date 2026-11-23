#!/usr/bin/env python3
"""Collect the BEIR text-embedding-3-small / BGE-large campaign into one Recall@10-QPS table (Figs 10, 21).

The 16-configuration matrix (diskveclab_full_matrix) used an older OpenAI ada-002 MS MARCO file, the
999k-row DBpedia OpenAI set and a 99k-row DBpedia BGE set. Table 5 describes the BEIR sets that the
openai_beir_11methods_t1 / onebit_disk_baselines_t1 campaigns measured instead:

    sieun_msmarco_openai1536                    MS MARCO, text-embedding-3-small, 1M, 6,980 queries
    sieun_dbpedia_entity_openai1536             DBpedia-entity, text-embedding-3-small, 1M, 400 queries
    sieun_dbpedia_entity_bge_large1024_beir400  DBpedia-entity, BGE-large-en-v1.5, 1M, 400 queries

This script reads every method's held-out T=1 result for those datasets and writes them in the schema of
recall_qps_12methods_16datasets_t1.csv, with `dataset` set to the figure panel key, so the panels can be
replaced one-for-one. Methods without results for a dataset are reported and skipped.

    python3 collect_beir_recall_qps.py [--experiment DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PANEL_OF = {
    "sieun_msmarco_openai1536": "msmarco_openai1536",
    "sieun_dbpedia_entity_openai1536": "dbpedia_openai1536",
    "sieun_dbpedia_entity_bge_large1024_beir400": "dbpedia_bge1024",
}
# figure block name -> (relative CSV path template, row filter, parameter column)
SOURCES = {
    "adcbq":   ("openai_beir_11methods_t1/four_methods/{ds}/summary_t1.csv", ("method_key", "adc_bq_calibrated"), "search_parameter_value"),
    "tq":      ("openai_beir_11methods_t1/four_methods/{ds}/summary_t1.csv", ("method_key", "turboquant_tq1"), "search_parameter_value"),
    "rbq7":    ("openai_beir_11methods_t1/four_methods/{ds}/summary_t1.csv", ("method_key", "rabitq_b7"), "search_parameter_value"),
    "hnswrbq": ("openai_beir_11methods_t1/four_methods/{ds}/summary_t1.csv", ("method_key", "rabitq_b1_adaptive"), "search_parameter_value"),
    "esbbq":   ("openai_beir_11methods_t1/elasticsearch_bbq/{ds}/all_results_t1.csv", None, "num_candidates"),
    "fpq":     ("openai_beir_11methods_t1/faiss_pq_opq_calibrated/selected_r_heldout_t1.csv", ("method_key", "pq8m128"), "search_parameter_value"),
    "fopq":    ("openai_beir_11methods_t1/faiss_pq_opq_calibrated/selected_r_heldout_t1.csv", ("method_key", "opq8m128"), "search_parameter_value"),
    "rbqssd":  ("openai_beir_11methods_t1/native_matrix/mrq_rabitq_ssd/summary_t1.csv", ("method_key", "rabitq_ssd"), "search_parameter_value"),
    "dannpq":  ("onebit_disk_baselines_t1/all_results_t1.csv", ("family", "diskann_classic"), "Lsearch"),
    "dannrbq": ("onebit_disk_baselines_t1/all_results_t1.csv", ("family", "diskann_rabitq_1bit_adaptive_fp"), "Lsearch"),
    "spann":   ("onebit_disk_baselines_t1/all_results_t1.csv", ("family", "spann_pq_onebit_budget_fixed_fp_rerank80"), None),
    "pipeann": ("onebit_disk_baselines_t1/all_results_t1.csv", ("family", "pipeann_native_rabitq_1bit_c16"), "Lsearch"),
}
# Campaign roots searched in order. The campaign refuses a different dataset selection under one root, so the
# BGE-large BEIR-400 set is run under its own root:
#   python3 run_openai_beir_11methods_t1.py --skip-prepare --datasets sieun_dbpedia_entity_bge_large1024_beir400 \
#       --result-root <experiment>/openai_beir_bge400_t1 --external-root $BBQ_DATA/openai_beir_bge400_t1
CAMPAIGN_ROOTS = ("openai_beir_11methods_t1", "openai_beir_bge400_t1")
FIELDS = ["dataset", "source_dataset", "corpus", "model", "figure_method", "method", "search_parameter_value",
          "recall_at_10", "qps", "rounds", "test_queries", "source_csv"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parents[1] / "experiment")
    ap.add_argument("--out", type=Path, default=Path("beir_recall_qps_t1.csv"))
    a = ap.parse_args()

    out, missing = [], []
    for ds, panel in PANEL_OF.items():
        for fig, (tmpl, flt, param) in SOURCES.items():
            rows, used = [], tmpl.format(ds=ds)
            candidates = [used] + [used.replace(CAMPAIGN_ROOTS[0], alt, 1) for alt in CAMPAIGN_ROOTS[1:]
                                   if used.startswith(CAMPAIGN_ROOTS[0])]
            for rel in candidates:
                path = a.experiment / rel
                if not path.is_file():
                    continue
                for r in csv.DictReader(open(path, newline="")):
                    if r.get("dataset") != ds or (flt and r.get(flt[0]) != flt[1]):
                        continue
                    rows.append(r)
                if rows:
                    used = rel
                    break
            if not rows:
                missing.append((ds, fig))
                continue
            for r in rows:
                out.append({
                    "dataset": panel, "source_dataset": ds, "corpus": r.get("corpus", ""), "model": r.get("model", ""),
                    "figure_method": fig, "method": r.get("method") or r.get("canonical_method") or fig,
                    "search_parameter_value": r.get(param, "") if param else r.get("search_parameter_value", ""),
                    "recall_at_10": r["recall_at_10"], "qps": r["qps"], "rounds": r.get("rounds", ""),
                    "test_queries": r.get("test_queries") or r.get("nq", ""), "source_csv": used,
                })
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {a.out}")
    for ds, fig in missing:
        print(f"MISSING {ds}: {fig}")


if __name__ == "__main__":
    main()
