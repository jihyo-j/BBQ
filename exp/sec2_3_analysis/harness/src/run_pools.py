import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool
DEV = ["msmarco_qw06","nq_qw06","quora_qw06","dbpedia_qw06",
       "msmarco_bgem3","nq_bgem3","quora_bgem3","fever_bgem3",
       "msmarco_ada2","dbpedia_ada2","dbpedia_bgel","dbpedia_bgeb"]
HELD = ["nq_nem8","quora_nem8","nq_e5m","quora_e5m","dbpedia_e5m"]
GROUPS = dict(dev=DEV, held=HELD)
for g in sys.argv[1:]:
    for k in GROUPS[g]:
        t = time.time()
        try:
            pool.run(k, nq=6000, nproc=16)
            print(f"DONE {k} {time.time()-t:.0f}s", flush=True)
        except Exception as e:
            print(f"FAIL {k} {type(e).__name__}: {e}", flush=True)
