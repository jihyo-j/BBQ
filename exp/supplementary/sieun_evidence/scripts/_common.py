"""Shared loaders for the BBQ evidence scripts.  No third-party dependencies."""
import csv, json, glob, math, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# --- the 10 quantizer arms of the Section 3 offline sweep, in paper order -----
ARMS = ["sym_bq", "adc_bq", "adc_bq_c", "adc_bq_cb", "pq", "opq",
        "sq4", "sq8", "rabitq1", "tq1"]
NICE = {"sym_bq": "Sym-BQ", "adc_bq": "ADC-BQ", "adc_bq_c": "ADC-BQ +centre",
        "adc_bq_cb": "ADC-BQ +centre+scale", "pq": "PQ", "opq": "OPQ",
        "sq4": "SQ-4bit", "sq8": "SQ-8bit", "rabitq1": "RaBitQ-1bit",
        "tq1": "TurboQuant-1bit (= rotated ADC-BQ, by ranking)"}


def load_sec3():
    """key -> configuration record, for the offline exhaustive-scan sweep."""
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, "bbqeta_out", "*.json"))):
        name = os.path.basename(p)[:-5]
        if name.startswith(("dimsweep", "dimperq")):
            continue
        d = json.load(open(p))
        if isinstance(d, dict) and isinstance(d.get("arms"), dict):
            out[d["key"]] = d
    return out


def eta_agg(rec, arm):
    """eta exactly as the paper defines it: nu/(kappa*beta) from the aggregate
    kappa of the arm and the aggregate beta of the configuration (ranks 1-100).
    Mirrors BBQETA src/tables.py:eta_agg."""
    k = arm["kappa"]
    b = rec["beta"]
    if k is None or b is None or k != k or b != b or k <= 0 or b <= 0:
        return float("nan")
    return math.sqrt(max(1.0 - k * k, 0.0)) / (k * b)


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    a, b = rank(x), rank(y)
    n = len(x)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den


def load_sec6():
    """The 16 workload x 11 method Recall@10-QPS sweep behind Figures 9 and 20."""
    rows = list(csv.DictReader(open(os.path.join(DATA, "recall-qps_full16_latest_t1.csv"))))
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["method"]), []).append(r)
    return rows, by


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def best_at(by, ds, method, target):
    """Highest-QPS operating point that reaches Recall@10 >= target."""
    pts = [p for p in by.get((ds, method), [])
           if f(p["recall_at_10"]) is not None and f(p["recall_at_10"]) >= target]
    return max(pts, key=lambda p: f(p["qps"])) if pts else None


def ceiling(by, ds, method):
    return max(f(p["recall_at_10"]) for p in by[(ds, method)])


SEC6_METHODS = ["BBQ (calibrated ADC-BQ)", "TurboQuant TQ1+", "Faiss PQ d/8",
                "Faiss OPQ-PQ d/8", "FP32 HNSW RAM", "RaBitQ 7-bit (RAM)",
                "RaBitQ-SSD B9", "HNSW RaBitQ 1-bit + adaptive FP",
                "DiskANN-PQ d/8", "DiskANN-RaBitQ 1-bit", "PipeANN-RaBitQ 1-bit"]
# everything that keeps d/8 bytes of codes resident (excludes FP32 and 7-bit RAM)
ISO_DRAM = [m for m in SEC6_METHODS if m not in ("FP32 HNSW RAM", "RaBitQ 7-bit (RAM)")]
