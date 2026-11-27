import json, numpy as np, torch
from sentence_transformers import SentenceTransformer
D = "/root/beir_emb/out_full/msmarco"; M = "all-MiniLM-L6-v2"
model = SentenceTransformer(f"sentence-transformers/{M}", device="cuda")
model.half()
def load(p, f): return [f(json.loads(l)) for l in open(p)]
q = load(f"{D}/queries.jsonl", lambda x: x["text"])
c = load(f"{D}/corpus.jsonl", lambda x: (x["title"] + " " + x["text"]).strip())
for name, texts in [("queries", q), ("corpus", c)]:
    E = model.encode(texts, batch_size=1024, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.save(f"{D}/{name}_{M}.npy", E); print(name, E.shape, flush=True)
