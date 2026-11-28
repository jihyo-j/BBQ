#!/usr/bin/env python3
"""Recompute every aggregate quoted in Section 6.2 and 6.5 from the raw sweep.

Source
  data/recall-qps_full16_latest_t1.csv
      3,477 measured operating points: 16 dataset/embedding configurations
      x 11 methods, single thread, held-out test queries.

Nothing here is read from the manuscript; the manuscript values are printed
beside each result for comparison.
"""
import os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_sec6, best_at, ceiling, f, ISO_DRAM, SEC6_METHODS

rows, by = load_sec6()
DS = sorted({r["dataset"] for r in rows})
BBQ, TQ, FP32 = "BBQ (calibrated ADC-BQ)", "TurboQuant TQ1+", "FP32 HNSW RAM"
NICE = {"sieun_nq_openai1536": "NQ/OpenAI-1536", "nq_bge_m3_vanilla": "NQ/BGE-1024",
        "nq_qwen06_1024": "NQ/Qwen-1024", "nq_e5mistral4096": "NQ/E5-4096",
        "sieun_quora_openai1536": "Quora/OpenAI-1536", "quora_bge_m3_vanilla": "Quora/BGE-1024",
        "quora_qwen1024": "Quora/Qwen-1024", "quora_e5mistral4096": "Quora/E5-4096",
        "sieun_msmarco_openai1536": "MSMARCO/OpenAI-1536", "msmarco_bge1024": "MSMARCO/BGE-1024",
        "msmarco_qwen1024": "MSMARCO/Qwen-1024", "msmarco_e5mistral4096": "MSMARCO/E5-4096",
        "sieun_dbpedia_entity_openai1536": "DBpedia/OpenAI-1536",
        "sieun_dbpedia_entity_bge_large1024_beir400": "DBpedia/BGE-1024",
        "dbpedia_qwen1024": "DBpedia/Qwen-1024", "dbpedia_e5mistral4096": "DBpedia/E5-4096"}
ORDER = [k for k in NICE if k in DS]

print(f"rows {len(rows)}   configurations {len(DS)}   methods {len({r['method'] for r in rows})}")

print("\n" + "=" * 92)
print("6.2  BBQ vs TurboQuant-1bit")
print("=" * 92)
for thr in (0.99, 0.995):
    w = l = 0
    for ds in ORDER:
        b, t = best_at(by, ds, BBQ, thr), best_at(by, ds, TQ, thr)
        if b and t:
            w += f(b["qps"]) > f(t["qps"]); l += f(b["qps"]) <= f(t["qps"])
    print(f"  Recall@10 >= {thr}: BBQ faster in {w}/{w+l} configurations where both reach it")
print("  [manuscript: 14 of 16 at 0.99, and all seven at 0.995]")
for ds, thr, lbl in [("nq_qwen06_1024", 0.995, "NQ Qwen-1024 @0.995"),
                     ("msmarco_qwen1024", 0.99, "MS MARCO Qwen-1024 @0.99")]:
    b, t = best_at(by, ds, BBQ, thr), best_at(by, ds, TQ, thr)
    print(f"  {lbl:28s} BBQ {f(b['qps']):7.0f}  TQ {f(t['qps']):7.0f}  = {f(b['qps'])/f(t['qps']):.2f}x")
print("  [manuscript: 1,275 vs 636 = 2.00x ; 1,055 vs 664 = 1.59x]")
fb, ft = [], []
for ds in ORDER:
    b, t = best_at(by, ds, BBQ, 0.99), best_at(by, ds, TQ, 0.99)
    if b and t and f(b["fp_vectors_per_query"]) and f(t["fp_vectors_per_query"]):
        fb.append(f(b["fp_vectors_per_query"])); ft.append(f(t["fp_vectors_per_query"]))
print(f"  FP32 reads/query where recorded (n={len(fb)}): BBQ {sum(fb)/len(fb):.1f} vs TQ {sum(ft)/len(ft):.1f}"
      f"  = {100*(1-sum(fb)/sum(ft)):.0f}% fewer    [manuscript: 41.8 vs 76.0, 45%]")

print("\n  losses (TurboQuant faster at Recall@10 >= 0.99):")
for ds in ORDER:
    b, t = best_at(by, ds, BBQ, 0.99), best_at(by, ds, TQ, 0.99)
    if b and t and f(t["qps"]) > f(b["qps"]):
        print(f"    {NICE[ds]:22s} TQ {f(t['qps']):6.0f} vs BBQ {f(b['qps']):6.0f} QPS"
              f"   FP32 reads {t['fp_vectors_per_query'] or '-':>7s} vs {b['fp_vectors_per_query'] or '-':>7s}"
              f"   ef {t['search_parameter_value']:>5s} vs {b['search_parameter_value']:>5s}")
print("  [manuscript: two configurations, both DBpedia; 250 vs 181 and 201 vs 182;")
print("               on BGE-1024 BBQ reads 77.5 vs 200 but needs ef 1024 vs 512]")

print("\n" + "=" * 92)
print("6.2  BBQ vs in-memory FP32 HNSW")
print("=" * 92)
rt = []
for ds in ORDER:
    b, p = best_at(by, ds, BBQ, 0.99), best_at(by, ds, FP32, 0.99)
    if b and p:
        rt.append((f(b["qps"]) / f(p["qps"]), ds))
print(f"  Recall@10 >= 0.99: BBQ faster in {sum(1 for r,_ in rt if r>1)}/{len(rt)},"
      f" range {min(r for r,_ in rt):.2f}-{max(r for r,_ in rt):.2f}x"
      f"     [manuscript: 15 of 16, 1.21-2.70x]")
for r, ds in sorted(rt):
    print(f"    {r:5.2f}x  {NICE[ds]}")
gap = [(ceiling(by, ds, FP32) - ceiling(by, ds, BBQ)) * 100 for ds in ORDER]
print(f"  max-recall gap vs FP32 HNSW: mean {sum(gap)/len(gap):+.2f} pp, worst {max(gap):+.2f} pp"
      f"     [manuscript: within 0.20 pp everywhere, 0.10 pp on average]")

print("\n" + "=" * 92)
print("6.5  Method selection (Table 7, Findings 8 and 9)")
print("=" * 92)
cells = losses = 0
for thr in (0.95, 0.98, 0.99, 0.995, 0.998):
    for ds in ORDER:
        c = sorted(((f(best_at(by, ds, m, thr)["qps"]), m) for m in ISO_DRAM if best_at(by, ds, m, thr)),
                   reverse=True)
        if not c:
            continue
        cells += 1
        if c[0][1] != BBQ:
            losses += 1
print(f"  BBQ fastest in {cells-losses}/{cells} (configuration, recall-target) cells of the d/8-byte class"
      f"     [manuscript: 58 of 74]")
marg = []
for thr in (0.95, 0.98, 0.99, 0.995, 0.998):
    for ds in ORDER:
        c = sorted(((f(best_at(by, ds, m, thr)["qps"]), m) for m in ISO_DRAM if best_at(by, ds, m, thr)),
                   reverse=True)
        if len(c) > 1 and c[0][1] == BBQ:
            marg.append(c[0][0] / c[1][0])
print(f"  median margin of BBQ over the runner-up: {st.median(marg):.2f}x     [manuscript: 1.4x]")

unreach = hi_w = hi_n = 0
for thr in (0.95, 0.98, 0.99, 0.995, 0.998):
    for ds in ORDER:
        c = sorted(((f(best_at(by, ds, m, thr)["qps"]), m) for m in ISO_DRAM if best_at(by, ds, m, thr)),
                   reverse=True)
        if not c:
            continue
        head = (ceiling(by, ds, BBQ) - thr) * 100
        won = c[0][1] == BBQ
        if not won and head < 0:
            unreach += 1
        if head >= 0.3:
            hi_n += 1; hi_w += won
print(f"  of BBQ's {losses} losses, {unreach} are cells it cannot reach at all     [manuscript: 12 of 16]")
print(f"  where BBQ clears the target by >= 0.3 pp it is fastest in {hi_w}/{hi_n}     [manuscript: 40 of 42]")

print("\n  ceiling lift over BBQ (percentage points, mean over the 16 configurations):")
for m in ["Faiss OPQ-PQ d/8", "Faiss PQ d/8", TQ, "RaBitQ-SSD B9", FP32]:
    d = [(ceiling(by, ds, m) - ceiling(by, ds, BBQ)) * 100 for ds in ORDER]
    print(f"    {m:34s} {sum(d)/len(d):+.3f} pp   higher in {sum(1 for x in d if x>0)}/16")
print("  [manuscript: OPQ +0.00, PQ -0.04, TQ -0.06; RaBitQ-SSD +0.20 in 15/16; FP32 +0.10 in 15/16]")

print("\n  RaBitQ-7bit (7d/8 bytes resident) vs BBQ:")
for thr in (0.95, 0.98, 0.99):
    r = []
    for ds in ORDER:
        a, b = best_at(by, ds, "RaBitQ 7-bit (RAM)", thr), best_at(by, ds, BBQ, thr)
        if a and b:
            r.append(f(a["qps"]) / f(b["qps"]))
    print(f"    Recall@10 >= {thr}: median {st.median(r):.2f}x, faster in {sum(1 for x in r if x>1)}/{len(r)}")
c995 = sum(1 for ds in ORDER if ceiling(by, ds, "RaBitQ 7-bit (RAM)") >= 0.995)
c99 = sum(1 for ds in ORDER if ceiling(by, ds, "RaBitQ 7-bit (RAM)") >= 0.99)
print(f"    reaches 0.99 in {c99}/16 and 0.995 in {c995}/16"
      f"     [manuscript: median 2.10x at 0.95, 14/16 fastest at 0.98, 1.19x at 0.99, 0.995 in 2 of 16]")

print("\n  SSD-resident graphs vs BBQ at Recall@10 >= 0.99 (QPS ratio):")
for m in ["DiskANN-PQ d/8", "DiskANN-RaBitQ 1-bit", "PipeANN-RaBitQ 1-bit", "RaBitQ-SSD B9",
          "HNSW RaBitQ 1-bit + adaptive FP"]:
    r = []
    for ds in ORDER:
        a, b = best_at(by, ds, m, 0.99), best_at(by, ds, BBQ, 0.99)
        if a and b:
            r.append(f(b["qps"]) / f(a["qps"]))
    print(f"    {m:34s} n={len(r):2d}  {min(r):5.1f}-{max(r):6.1f}x  median {st.median(r):5.1f}x")
print("  [manuscript: 12.7-31.6x, median 22.6x for DiskANN-PQ and 19.7x for PipeANN; IVF median 9.1x]")

print("\n" + "=" * 92)
print("NOT COVERED BY THIS FILE")
print("=" * 92)
print("  The csv is 1M-scale only (nb = 1,000,000, or 522,931 for Quora).  The 10M/50M/100M")
print("  scaling numbers of Section 6.5 -- 563/541/539 QPS, 31.3 GiB vs 105.1 GiB, 51.7 -> 71.5")
print("  FP32 reads -- come from a separate run whose output is not on this machine.")
