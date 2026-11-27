import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure, pool
HELD = ["nq_nem8","quora_nem8","nq_e5m","quora_e5m","dbpedia_e5m"]
for k in HELD:
    if os.path.exists(f"/home/sieun/projects/BBQETA/out/{k}.json"):
        print(f"SKIP measure {k}", flush=True); continue
    t=time.time()
    try:
        measure.run(k, arms=["adc_bq"])
        print(f"DONE measure {k} {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL measure {k} {type(e).__name__}: {e}", flush=True)
for k in HELD:
    t=time.time()
    try:
        pool.run(k, nq=6000, nproc=16)
        print(f"DONE pool {k} {time.time()-t:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL pool {k} {type(e).__name__}: {e}", flush=True)
