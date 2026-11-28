#!/usr/bin/env python3
"""Check every numeric claim in Section 3 against the stored measurements.

Sources
  data/bbqeta_out/*.json          offline exhaustive-scan sweep (20 configurations x 10 arms)
  data/bbqeta_out/dimsweep_*.json MRL width ladder, Qwen3-8B on MS MARCO
  data/misc/bitsvsdims.txt        Table 2 (width vs precision at a fixed byte budget)
  data/misc/placebo.log           query-instruction ablation
  data/misc/p_cdf.dat             per-query P(R) = Pr(R* <= R), Figure 4(a)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DATA, load_sec3, spearman, eta_agg

C = load_sec3()
ok = bad = 0


def check(label, got, want, tol):
    global ok, bad
    good = got is not None and abs(got - want) <= tol
    ok, bad = ok + good, bad + (not good)
    gs = "   n/a" if got is None else f"{got:9.4f}"
    print(f"  {label:48s} paper {want:<10} data {gs}  {'OK' if good else 'MISMATCH'}")


print("=" * 86)
print("3.1  Candidate preservation under BQ")
print("=" * 86)
for cfg, lbl in [("msmarco_bgem3", "MS MARCO BGE-M3"), ("nq_bgem3", "NQ BGE-M3")]:
    for arm, an in [("sym_bq", "sym-BQ"), ("adc_bq", "ADC-BQ")]:
        a = C[cfg]["arms"][arm]
        w = {"msmarco_bgem3": {"sym_bq": (0.848, 416), "adc_bq": (0.947, 99)},
             "nq_bgem3": {"sym_bq": (0.865, 367), "adc_bq": (0.955, 84)}}[cfg][arm]
        check(f"{lbl} CandRec@30 {an}", a["candrec"]["30"], w[0], 0.0015)
        check(f"{lbl} p95 R* {an}", a["Rstar_p95"], w[1], 1.0)
check("GloVe  CandRec@30 ADC-BQ", C["glove"]["arms"]["adc_bq"]["candrec"]["30"], 0.412, 0.0015)
check("GIST   CandRec@30 ADC-BQ", C["gist"]["arms"]["adc_bq"]["candrec"]["30"], 0.000, 0.0015)

print()
print("=" * 86)
print("3.2  Geometry: the eta = c_Q / beta ordering")
print("=" * 86)
E, P = [], []
for k, d in C.items():
    for a in d["arms"]:
        e = eta_agg(d, d["arms"][a])
        if e == e:
            E.append(e); P.append(d["arms"][a]["Rstar_p95"])
ea, pa, drop = [], [], []
for k, d in C.items():
    if "adc_bq" in d["arms"]:
        e = eta_agg(d, d["arms"]["adc_bq"])
        (ea.append(e) or pa.append(d["arms"]["adc_bq"]["Rstar_p95"])) if e == e else drop.append(k)
print(f"  pooled over (configuration, quantizer): n={len(E)}  Spearman={spearman(E, P):.3f}"
      f"     [manuscript: 0.921 over 90 pairs]")
print(f"  ADC-BQ across configurations:           n={len(ea)}  Spearman={spearman(ea, pa):.3f}"
      f"     [manuscript: 0.912]")
print(f"  excluded (kappa undefined, all-+1 sign codes): {drop}")
print("  >> STALE: the manuscript number predates the current 20-configuration sweep.")
print("     RESULTS.md:111 of the BBQETA project carries a third vintage, 0.981 over 94 pairs.")
print("     90 pairs is reproducible as 15 configurations x some 6 arms, but six different")
print("     6-arm subsets land in 0.919-0.922, so the original subset is not identifiable.")

print()
print("=" * 86)
print("3.2  MRL width ladder (Qwen3-8B on MS MARCO, own ground truth per width)")
print("=" * 86)
rows = {r["d"]: r for r in json.load(open(f"{DATA}/bbqeta_out/dimsweep_msmarco_qw8.json"))["rows"]}
fs = [r["f_sign_cos"] for r in rows.values()]
check("sign alignment f, minimum", min(fs), 0.775, 0.001)
check("sign alignment f, maximum", max(fs), 0.802, 0.001)
check("kappa  ADC-BQ at d=32", rows[32]["arms"]["adc_bq"]["kappa"], 0.782, 0.001)
check("kappa  ADC-BQ at d=4096", rows[4096]["arms"]["adc_bq"]["kappa"], 0.958, 0.001)
check("beta at d=32", rows[32]["beta"], 0.17, 0.005)
check("beta at d=4096", rows[4096]["beta"], 1.07, 0.005)
check("p95 R* ADC-BQ at d=128", rows[128]["arms"]["adc_bq"]["Rstar_p95"], 2534, 1.0)
check("p95 R* ADC-BQ at d=512", rows[512]["arms"]["adc_bq"]["Rstar_p95"], 124, 1.0)
check("p95 R* ADC-BQ at d=1024", rows[1024]["arms"]["adc_bq"]["Rstar_p95"], 47, 1.0)
check("p95 R* ADC-BQ at d=4096", rows[4096]["arms"]["adc_bq"]["Rstar_p95"], 22, 1.0)

print()
print("=" * 86)
print("3.2  Corpus / encoder / query-instruction effects")
print("=" * 86)
for cfg, lbl, wb, wr in [("quora_qw06", "Quora Qwen3-0.6B", 0.889, 35),
                         ("quora_bgem3", "Quora BGE-M3", 0.896, 29),
                         ("msmarco_qw06", "MS MARCO Qwen3-0.6B", 0.825, 47),
                         ("msmarco_bgem3", "MS MARCO BGE-M3", 0.620, 99)]:
    check(f"{lbl} beta", C[cfg]["beta"], wb, 0.001)
    check(f"{lbl} p95 R* ADC-BQ", C[cfg]["arms"]["adc_bq"]["Rstar_p95"], wr, 1.0)
print("  instruction ablation, from data/misc/placebo.log:")
for line in open(f"{DATA}/misc/placebo.log"):
    if line.startswith("RESULT"):
        print("    " + line.rstrip())
print("    [manuscript: MS MARCO beta 0.825 -> 0.523, budget 47 -> 74;")
print("                 Quora    beta 0.889 -> 0.740, budget 35 -> 30]")

print()
print("=" * 86)
print("3.2  Table 2 -- width or precision at a fixed byte budget")
print("=" * 86)
print(open(f"{DATA}/misc/bitsvsdims.txt").read().rstrip())

print()
print("=" * 86)
print("3.4  Long-tailed candidate demand: median / p95 / p99 of R*")
print("=" * 86)
hdr = open(f"{DATA}/misc/p_cdf.dat").readline().lstrip("#").split()
cols = hdr[1:]
R, F = [], [[] for _ in cols]
for line in open(f"{DATA}/misc/p_cdf.dat"):
    if line.startswith("#"):
        continue
    v = line.split()
    R.append(float(v[0]))
    for i in range(len(cols)):
        F[i].append(float(v[i + 1]))


def quant(i, p):
    for r, y in zip(R, F[i]):
        if y >= p:
            return r
    return R[-1]


for i, name in enumerate(cols):
    print(f"  {name:16s} median {quant(i,0.50):6.0f}   p95 {quant(i,0.95):6.0f}   p99 {quant(i,0.99):6.0f}")
print("  [manuscript: MS MARCO Qwen-1024 16 / 47 / 87 ; MS MARCO BGE-M3 23 / 99 / 195]")

print()
print("=" * 86)
print(f"exact-match checks: {ok} OK, {bad} MISMATCH")
print("the eta Spearman sentence is reported separately above and is STALE")
print("=" * 86)
