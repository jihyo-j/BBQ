"""Run the refinement replay over all arms, interleaved repetitions.

Interleaving the arms across repetitions keeps a drifting shared device from
being charged to one arm; the reported number is the median repetition.
"""
import os, sys, json, subprocess, time
IOD = "/home/smrc/samsung-nvme/sieun/bbqeta/io"
BIN = "/home/sieun/projects/BBQETA/src/io/refio"
OUT = "/home/sieun/projects/BBQETA/out"


def arms(k):
    a = [("seq_bound", k["seq_bound_default"], 0, "RaBitQ bound @1.9, sequential"),
         ("seq_bound", k["seq_bound_tuned"], 0, "RaBitQ bound tuned, sequential"),
         ("seq_calib", k["calib_tau"], 0, "calibrated prefix, sequential reads"),
         ("batch_calib", k["calib_tau"], 0, "calibrated prefix, one io_uring batch"),
         ("batch_boundM", k["batch_boundM"], 0, "bound as margin, one batch"),
         ("twophase", k["twophase"], 0, "bound, 2 rounds (top-k then batch)"),
         ("batch_fixedR", k["fixedR"], 0, "tuned fixed prefix, one batch")]
    for C in (4, 8, 16, 32, 64):
        a.append(("chunk_bound", k["seq_bound_tuned"], C,
                  f"bound chunked at {C}, batch per chunk"))
    return a


def run(key, ef, nq, reps=3):
    m = json.load(open(f"{IOD}/meta_{key}_ef{ef}.json"))
    res = {}
    for rep in range(reps):
        for arm, param, qd, note in arms(m["knobs"]):
            cmd = [BIN, m["poolfile"], m["vecfile"], str(m["pagesz"]), arm,
                   repr(float(param)), str(nq)] + ([str(qd)] if qd else [])
            o = json.loads(subprocess.run(cmd, capture_output=True, text=True,
                                          check=True).stdout)
            o["note"] = note
            res.setdefault(f"{arm}|{param}|{qd}", []).append(o)
        print(f"  {key} rep{rep} done", flush=True)
    agg = {}
    for k2, v in res.items():
        med = sorted(v, key=lambda x: x["lat_mean_us"])[len(v) // 2]
        agg[k2] = dict(med, lat_mean_us_all=[x["lat_mean_us"] for x in v],
                       qps_all=[x["qps_refine"] for x in v])
    out = dict(key=key, ef=ef, meta=m, arms=agg)
    json.dump(out, open(f"{OUT}/refio_{key}_ef{ef}.json", "w"), indent=1)
    for k2, v in agg.items():
        print(f"{key:14s} {v['note']:42s} reads {v['reads']:7.2f} pages "
              f"{v['pages']:7.2f} rounds {v['rounds']:6.2f} lat {v['lat_mean_us']:8.1f}us "
              f"p95 {v['lat_p95_us']:8.1f} qps {v['qps_refine']:8.0f}", flush=True)
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        key, ef, nq = a.split(":")
        run(key, int(ef), int(nq))
