import os, json, random, time, glob
import numpy as np, tiktoken
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from openai import OpenAI

ROOT = "/root/beir_emb/out_full"; os.makedirs(ROOT, exist_ok=True)
NC = 1_000_000; SEED = 42; SHARD = 10_000; BS = 512; WORKERS = 24
CFG = {"quora": "test", "dbpedia-entity": "test", "nq": "test", "msmarco": "validation"}
MODELS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
client = OpenAI(max_retries=0)
enc = tiktoken.get_encoding("cl100k_base")

def clean(t):
    t = (t or "").strip() or "."
    ids = enc.encode(t, disallowed_special=())
    return enc.decode(ids[:8000]) if len(ids) > 8000 else t

def call(texts, model):
    for a in range(10):
        try:
            r = client.embeddings.create(model=model, input=texts)
            return np.asarray([d.embedding for d in sorted(r.data, key=lambda d: d.index)], np.float32)
        except Exception as e:
            print("retry", a, type(e).__name__, str(e)[:120], flush=True); time.sleep(min(60, 2 ** a))
    raise RuntimeError("embedding failed")

def embed_to(texts, model, path):
    dim = MODELS[model]
    if os.path.exists(path): print("skip", path, flush=True); return
    sd = path + ".shards"; os.makedirs(sd, exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for s in range(0, len(texts), SHARD):
            sp = f"{sd}/{s:09d}.npy"
            if os.path.exists(sp): continue
            chunk = texts[s:s+SHARD]
            parts = list(ex.map(lambda i: call(chunk[i:i+BS], model), range(0, len(chunk), BS)))
            np.save(sp + ".tmp.npy", np.concatenate(parts)); os.rename(sp + ".tmp.npy", sp)
            print(f"{os.path.basename(path)} {s+len(chunk)}/{len(texts)}", flush=True)
    out = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float32, shape=(len(texts), dim))
    for s in range(0, len(texts), SHARD):
        out[s:s+SHARD] = np.load(f"{sd}/{s:09d}.npy")
    out.flush(); del out; os.rename(path + ".tmp", path)
    for f in glob.glob(sd + "/*"): os.remove(f)
    os.rmdir(sd)

for name, split in CFG.items():
    d = f"{ROOT}/{name}"; os.makedirs(d, exist_ok=True)
    q = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qtext = {str(r["_id"]): r["text"] for r in q}
    qr = load_dataset(f"BeIR/{name}-qrels", split=split)
    qids = sorted({str(x) for x in qr["query-id"]} & qtext.keys())
    with open(f"{d}/qrels_{split}.tsv", "w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for a, b, c in zip(qr["query-id"], qr["corpus-id"], qr["score"]): f.write(f"{a}\t{b}\t{c}\n")
    c = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    cid = [str(x) for x in c["_id"]]
    if len(cid) > NC:
        rel = {str(x) for x, s in zip(qr["corpus-id"], qr["score"]) if s > 0}
        must = [i for i, x in enumerate(cid) if x in rel]
        ms = set(must); rest = [i for i in range(len(cid)) if i not in ms]
        idx = sorted(must + random.Random(SEED).sample(rest, NC - len(must)))
        print(f"{name}: {len(must)} relevant docs + {NC-len(must)} random", flush=True)
    else:
        idx = list(range(len(cid)))
    cs = c.select(idx)
    ctexts = [clean(t + " " + x) for t, x in zip(cs["title"], cs["text"])]
    if not os.path.exists(f"{d}/corpus.jsonl"):
        with open(f"{d}/corpus.jsonl", "w") as f:
            for i, t, x in zip(cs["_id"], cs["title"], cs["text"]):
                f.write(json.dumps({"_id": str(i), "title": t, "text": x}, ensure_ascii=False) + "\n")
    with open(f"{d}/queries.jsonl", "w") as f:
        for i in qids: f.write(json.dumps({"_id": i, "text": qtext[i]}, ensure_ascii=False) + "\n")
    qtexts = [clean(qtext[i]) for i in qids]
    print(f"== {name}: {len(qids)} queries, {len(ctexts)} docs", flush=True)
    for m in MODELS:
        embed_to(qtexts, m, f"{d}/queries_{m}.npy")
        embed_to(ctexts, m, f"{d}/corpus_{m}.npy")
print("ALL DONE", flush=True)
