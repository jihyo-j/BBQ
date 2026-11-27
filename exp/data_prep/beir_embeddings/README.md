# BEIR OpenAI embeddings (NQ / Quora / MSMARCO / DBpedia)

Models: `text-embedding-3-small` (1536-d), `text-embedding-3-large` (3072-d). float32, OpenAI 기본 L2 정규화 → 내적 = 코사인.

**추가 (msmarco만):** `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2 정규화, max 256 토큰). 파일 `msmarco/{corpus,queries}_all-MiniLM-L6-v2.npy`, 스크립트 `embed_minilm.py`.

| dataset | queries (split) | corpus |
|---|---|---|
| nq | 3,452 (test) | 1,000,000 / 2,681,468 |
| quora | 10,000 (test) | 522,931 (전체) |
| msmarco | 6,980 (dev) | 1,000,000 / 8,841,823 |
| dbpedia-entity | 400 (test) | 1,000,000 / 4,635,922 |

코퍼스 서브샘플: 해당 split qrels의 정답 문서(score>0) 전부 + 나머지 무작위(seed 42). 원본: HF `BeIR/{name}`, `BeIR/{name}-qrels`.
임베딩 입력 텍스트: `(title + " " + text).strip()`, 8000 토큰 초과 시 절단. 생성 스크립트: `embed_all.py`.

## 파일 (데이터셋 폴더마다)
- `corpus.jsonl`, `queries.jsonl` — `_id` 등. **npy 행 순서 = jsonl 줄 순서**
- `qrels_<split>.tsv` — query-id, corpus-id, score
- `corpus_<model>.npy`, `queries_<model>.npy`

## 로드 예시
```python
import numpy as np, json
d, m = "nq", "text-embedding-3-large"
C = np.load(f"{d}/corpus_{m}.npy", mmap_mode="r")
Q = np.load(f"{d}/queries_{m}.npy")
cids = [json.loads(l)["_id"] for l in open(f"{d}/corpus.jsonl")]
qids = [json.loads(l)["_id"] for l in open(f"{d}/queries.jsonl")]
scores = Q @ C.T   # (nq, nc) — 메모리 크면 청크로
```

무결성 확인: `sha256sum -c SHA256SUMS`
