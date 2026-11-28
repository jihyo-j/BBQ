#!/usr/bin/env python3
"""The eta = c_Q / beta ordering claim of Section 3.2, recomputed.

Reproduces BBQETA src/tables.py:diagnostics() on the data as it stands today.
The manuscript sentence ("Spearman 0.921 across 90 (configuration, quantizer)
pairs, and 0.912 for ADC-BQ alone") predates the current sweep and no longer
matches; this script prints the current values and shows why the 90-pair
subset cannot be recovered.
"""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ARMS, NICE, load_sec3, eta_agg, spearman

C = load_sec3()
FULL = sorted(k for k, d in C.items() if len(d["arms"]) == 10)

print("configurations:", len(C), " with all 10 arms:", len(FULL))
print()
print("per (configuration, quantizer): eta and p95 R*")
print(f"  {'configuration':16s} {'quantizer':24s} {'kappa':>7s} {'beta':>6s} {'eta':>7s} {'p95 R*':>10s}")
E, P = [], []
for k in sorted(C):
    for a in ARMS:
        if a not in C[k]["arms"]:
            continue
        arm = C[k]["arms"][a]
        e = eta_agg(C[k], arm)
        kp = arm["kappa"]
        print(f"  {k:16s} {NICE[a][:24]:24s} "
              f"{(f'{kp:7.4f}' if kp == kp else '    nan')} {C[k]['beta']:6.3f} "
              f"{(f'{e:7.3f}' if e == e else '    nan')} {arm['Rstar_p95']:10.1f}")
        if e == e:
            E.append(e); P.append(arm["Rstar_p95"])

print()
print("=" * 78)
print(f"pooled over (configuration, quantizer)   n={len(E):4d}   Spearman = {spearman(E,P):.3f}")
ea, pa = [], []
for k in sorted(C):
    if "adc_bq" in C[k]["arms"]:
        e = eta_agg(C[k], C[k]["arms"]["adc_bq"])
        if e == e:
            ea.append(e); pa.append(C[k]["arms"]["adc_bq"]["Rstar_p95"])
print(f"ADC-BQ across configurations             n={len(ea):4d}   Spearman = {spearman(ea,pa):.3f}")
print("=" * 78)
print("manuscript:  0.921 over 90 pairs, and 0.912 for ADC-BQ")
print("BBQETA RESULTS.md:111 (older vintage):  0.981 over 94 pairs")
print()
print("Why 90 is not recoverable: 90 = 15 configurations x 6 arms, and several")
print("different 6-arm subsets land in the same place --")
hits = []
for comb in itertools.combinations(ARMS, 6):
    E2, P2 = [], []
    for k in FULL:
        for a in comb:
            e = eta_agg(C[k], C[k]["arms"][a])
            if e == e:
                E2.append(e); P2.append(C[k]["arms"][a]["Rstar_p95"])
    if len(E2) >= 88 and abs(spearman(E2, P2) - 0.921) < 0.006:
        hits.append((spearman(E2, P2), len(E2), comb))
for r, n, comb in sorted(hits, reverse=True):
    print(f"   rho={r:.4f} (n={n})   {', '.join(NICE[a].split(' (')[0] for a in comb)}")
if not hits:
    print('   (no 6-arm subset of the current sweep lands within 0.006 of 0.921)')
print()
print("SUGGESTED REPLACEMENT for the Section 3.2 sentence:")
print(f'   "orders p95 R* with Spearman {spearman(E,P):.3f} across {len(E)} (configuration,')
print(f'    quantizer) pairs, and {spearman(ea,pa):.3f} for ADC-BQ across the {len(ea)} configurations')
print('    where kappa is defined."')
print("(GIST is excluded: every sign code is all-+1, so kappa is undefined.)")
