#!/usr/bin/env python3
"""What a random rotation buys a sign code.

There is no arm literally called "rotation" in the Section 3 sweep, but one of
the arms is exactly that.  From BBQETA src/quantizers.py (copied verbatim into
data/misc/quantizers.py.reference):

    if arm == "tq1":
        # TurboQuant 1-bit: random rotation, then the MSE-optimal 1-bit codebook.
        # On unit-norm data the global scale is constant, so the RANKING of this
        # arm equals rotated ADC-BQ; only the estimate's scale differs.
        P = _rot(d, SEED + 1)                     # Haar orthogonal, via QR
        Xh = sign(X @ P) * sqrt(2/(pi*d))

Every Section 3 metric (R*, CandidateRecall) is rank-based, so the tq1 column
IS the "ADC-BQ + random rotation" column.  No centring is applied, so this
isolates the rotation.  The companion arms complete the ablation:

    adc_bq      plain sign code
    adc_bq_c    + centring
    adc_bq_cb   + centring + per-vector scale
    tq1         + random rotation
    rabitq1     centring + rotation + per-vector beta
"""
import math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_sec3, spearman

C = {k: d for k, d in load_sec3().items() if len(d["arms"]) == 10}
print(f"configurations with the full arm set: {len(C)}")
print()
print(f"{'configuration':16s} {'kappa adc':>9s} {'kappa rot':>9s} | {'CR@30 adc':>9s} {'CR@30 rot':>9s} |"
      f" {'p95R* adc':>10s} {'p95R* rot':>10s} {'ratio':>7s}")
rows = []
for k in sorted(C):
    a, t = C[k]["arms"]["adc_bq"], C[k]["arms"]["tq1"]
    ra, rt = a["Rstar_p95"], t["Rstar_p95"]
    ka = f"{a['kappa']:9.4f}" if a["kappa"] == a["kappa"] else "      nan"
    print(f"{k:16s} {ka} {t['kappa']:9.4f} | {a['candrec']['30']:9.4f} {t['candrec']['30']:9.4f} |"
          f" {ra:10.1f} {rt:10.1f} {rt/ra:7.3f}")
    rows.append((k, a, t))

fin = [(k, a, t) for k, a, t in rows if a["kappa"] == a["kappa"]]
print()
print(f"  median p95 R*     plain sign {st.median(a['Rstar_p95'] for _,a,_ in rows):8.1f}"
      f"   ->  rotated {st.median(t['Rstar_p95'] for _,_,t in rows):8.1f}")
print(f"  rotation lowers p95 R* in {sum(1 for _,a,t in rows if t['Rstar_p95'] < a['Rstar_p95'])}/{len(rows)}"
      f" configurations")
dk = [t["kappa"] - a["kappa"] for _, a, t in fin]
dr = [math.log(t["Rstar_p95"] / a["Rstar_p95"]) for _, a, t in fin]
n = len(dk); mk, mr = sum(dk)/n, sum(dr)/n
num = sum((x-mk)*(y-mr) for x, y in zip(dk, dr))
den = math.sqrt(sum((x-mk)**2 for x in dk) * sum((y-mr)**2 for y in dr))
print(f"  Pearson( d kappa , d log p95 R* ) = {num/den:+.3f}   (n={n}, GIST excluded)")
print("    beta is a property of the exact scores and is unchanged by a rotation, so the")
print("    rotation moves eta = c_Q/beta purely through kappa -- which is what this shows.")

print()
print("  GIST is the one place a rotation is decisive:")
g = C["gist"]["arms"]
print(f"    plain sign : kappa undefined (every coordinate positive -> every code all-+1),"
      f" CandRec@30 {g['adc_bq']['candrec']['30']:.4f}, p95 R* {g['adc_bq']['Rstar_p95']:,.0f}")
print(f"    rotated    : kappa {g['tq1']['kappa']:.4f},"
      f" CandRec@30 {g['tq1']['candrec']['30']:.4f}, p95 R* {g['tq1']['Rstar_p95']:,.0f}"
      f"   ({g['adc_bq']['Rstar_p95']/g['tq1']['Rstar_p95']:.0f}x better)")

print()
print("  ingredient ablation, median over the configurations where kappa is defined:")
for arm, lbl in [("adc_bq", "plain sign (ADC-BQ)"), ("adc_bq_c", "+ centring"),
                 ("adc_bq_cb", "+ centring + scale"), ("tq1", "+ random rotation"),
                 ("rabitq1", "centring + rotation + beta (RaBitQ)")]:
    v = [C[k]["arms"][arm]["Rstar_p95"] for k, a, _ in fin]
    c = [C[k]["arms"][arm]["candrec"]["30"] for k, a, _ in fin]
    print(f"    {lbl:38s} p95 R* {st.median(v):8.1f}   CandRec@30 {st.median(c):.4f}")
print("    No added ingredient beats the plain sign code at the median.  This is the")
print("    offline counterpart of the Section 6.5 result that extra quantization")
print("    information does not raise the recall ceiling.")
