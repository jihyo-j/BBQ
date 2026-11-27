"""BBQETA - re-measure Section 4 (kappa, beta, eta, R*) of the BBQ paper on the
datasets that Section 6 actually evaluates, and extend eta/R* to every quantizer.

Lab rule: <= 24 threads on this shared server.
"""
import os
NTHREADS = int(os.environ.get("BBQETA_THREADS", "24"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(NTHREADS)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import struct
import numpy as np

JH = "/home/smrc/samsung-nvme/jihyo"
SYN = "/home/smrc/samsung-nvme/sieun/SYN/emb"
KM = "/home/smrc/samsung-nvme/kyungmin/datasets"
AB = "/home/smrc/samsung-nvme/ann-benchmark-data"
WORK = "/home/smrc/samsung-nvme/sieun/bbqeta"
OUT = "/home/sieun/projects/BBQETA/out"
# BEIR text (corpus / queries / qrels) for the corpora whose row order we
# verified against our vectors.  Only Quora is the full BEIR corpus in its
# native order; the other three are 1M subsamples and cannot be mapped back.
BEIR = os.environ.get("BBQETA_BEIR",
    "/tmp/claude-1007/-home-sieun/f2a0a973-d46a-4f9e-8675-19793fb4c84e/scratchpad/beir")
FIGS = "/home/sieun/projects/BBQETA/figs"

SEED = 20261116
NQ_MAX = 2000          # queries used per configuration
K = 10                 # Recall@k / CandidateRecall@R target depth
RANK_FIT = 100         # beta is fit on exact ranks 1..RANK_FIT


# ---------------------------------------------------------------- loaders
def fbin(path, n=None):
    with open(path, "rb") as f:
        N, D = struct.unpack("<II", f.read(8))
    n = N if n is None else min(n, N)
    a = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(N, D))
    return np.array(a[:n], dtype=np.float32, order="C")


def npy(path, n=None):
    a = np.load(path, mmap_mode="r")
    a = a if n is None else a[:n]
    return np.array(a, dtype=np.float32, order="C")


def h5(path, key, n=None):
    import h5py
    with h5py.File(path, "r") as f:
        d = f[key]
        a = d[:] if n is None else d[:n]
    return np.array(a, dtype=np.float32, order="C")


# ---------------------------------------------------------------- configs
# group: 'qwen' = corpus axis at a fixed encoder (Qwen3-0.6B, 1024d)
#        'bge'  = encoder axis
#        'classic' = conventional (non-learned) descriptors
CONFIGS = {
    # ---- Qwen3-Embedding-0.6B, 1024d: the corpus axis -------------------
    "msmarco_qw06": dict(
        corpus="MS MARCO", encoder="Qwen3-0.6B", d=1024, group="qwen",
        base=lambda: fbin(f"{JH}/qwen/msmarco-1m__vanilla-qwen06b_train.fbin"),
        query=lambda: fbin(f"{JH}/qwen/msmarco-1m__vanilla-qwen06b_query.fbin")),
    "nq_qw06": dict(
        corpus="NQ", encoder="Qwen3-0.6B", d=1024, group="qwen",
        base=lambda: fbin(f"{JH}/nq_sieun_adcbq_rabitq/nq_vanilla-qw06_train.fbin"),
        query=lambda: fbin(f"{JH}/nq_sieun_adcbq_rabitq/nq_vanilla-qw06_query.fbin")),
    "quora_qw06": dict(
        corpus="Quora", encoder="Qwen3-0.6B", d=1024, group="qwen",
        base=lambda: fbin(f"{JH}/quora_sieun_adcbq_rabitq/quora_vanilla-qw06_train.fbin"),
        query=lambda: fbin(f"{JH}/quora_sieun_adcbq_rabitq/quora_vanilla-qw06_query.fbin")),
    "dbpedia_qw06": dict(
        corpus="DBpedia", encoder="Qwen3-0.6B", d=1024, group="qwen",
        base=lambda: fbin(f"{JH}/qwen/dbpedia-entity__vanilla-qwen06b_train.fbin"),
        query=lambda: fbin(f"{JH}/qwen/dbpedia-entity__vanilla-qwen06b_query.fbin")),

    # ---- BGE: the encoder axis -----------------------------------------
    "msmarco_bgem3": dict(
        corpus="MS MARCO", encoder="BGE-M3", d=1024, group="bge",
        base=lambda: npy(f"{SYN}/msmarco-1m__vanilla_docs.npy"),
        query=lambda: npy(f"{SYN}/msmarco-1m__vanilla_queries.npy")),
    "nq_bgem3": dict(
        corpus="NQ", encoder="BGE-M3", d=1024, group="bge",
        base=lambda: fbin(f"{JH}/nq_sieun_adcbq_rabitq/nq_vanilla_train.fbin"),
        query=lambda: fbin(f"{JH}/nq_sieun_adcbq_rabitq/nq_vanilla_query.fbin")),
    "quora_bgem3": dict(
        corpus="Quora", encoder="BGE-M3", d=1024, group="bge",
        base=lambda: fbin(f"{JH}/quora_sieun_adcbq_rabitq/quora_vanilla_train.fbin"),
        query=lambda: fbin(f"{JH}/quora_sieun_adcbq_rabitq/quora_vanilla_query.fbin")),
    "fever_bgem3": dict(
        corpus="FEVER", encoder="BGE-M3", d=1024, group="bge",
        base=lambda: fbin(f"{JH}/fever_sieun_adcbq_rabitq/fever_vanilla_train.fbin"),
        query=lambda: fbin(f"{JH}/fever_sieun_adcbq_rabitq/fever_vanilla_query.fbin")),
    "dbpedia_bgel": dict(
        corpus="DBpedia", encoder="BGE-large-en-v1.5", d=1024, group="bge",
        base=lambda: npy(f"{SYN}/dbpedia-entity__vanilla-bgel_docs.npy"),
        query=lambda: npy(f"{SYN}/dbpedia-entity__vanilla-bgel_queries.npy")),
    "dbpedia_bgeb": dict(
        corpus="DBpedia", encoder="BGE-base-en-v1.5", d=768, group="bge",
        base=lambda: npy(f"{SYN}/dbpedia-entity__vanilla-bgeb_docs.npy"),
        query=lambda: npy(f"{SYN}/dbpedia-entity__vanilla-bgeb_queries.npy")),

    "msmarco_ada2": dict(
        corpus="MS MARCO", encoder="ada-002", d=1536, group="bge",
        base=lambda: h5(f"{AB}/data/msmarco-v1-openai-ada2-1M-ip.hdf5", "train"),
        query=lambda: h5(f"{AB}/data/msmarco-v1-openai-ada2-1M-ip.hdf5", "test")),
    "dbpedia_ada2": dict(
        corpus="DBpedia", encoder="ada-002", d=1536, group="bge",
        base=lambda: h5(f"{AB}/deep_1B/dbpedia-openai-1000k-angular.hdf5", "train"),
        query=lambda: h5(f"{AB}/deep_1B/dbpedia-openai-1000k-angular.hdf5", "test")),

    "msmarco_qw8": dict(
        corpus="MS MARCO", encoder="Qwen3-8B", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/qwen/msmarco-1m__vanilla-qwen8b_train.fbin"),
        query=lambda: fbin(f"{JH}/qwen/msmarco-1m__vanilla-qwen8b_query.fbin")),
    "nq_nem8": dict(
        corpus="NQ", encoder="Nemotron", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/nq_sieun_adcbq_rabitq/nq_vanilla-nem8_train.fbin"),
        query=lambda: fbin(f"{JH}/nq_sieun_adcbq_rabitq/nq_vanilla-nem8_query.fbin")),
    "quora_nem8": dict(
        corpus="Quora", encoder="Nemotron", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/quora_sieun_adcbq_rabitq/quora_vanilla-nem8_train.fbin"),
        query=lambda: fbin(f"{JH}/quora_sieun_adcbq_rabitq/quora_vanilla-nem8_query.fbin")),
    "nq_e5m": dict(
        corpus="NQ", encoder="E5-Mistral", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/e5/nq_vanilla-e5m7_train.fbin"),
        query=lambda: fbin(f"{JH}/e5/nq_vanilla-e5m7_query.fbin")),
    "quora_e5m": dict(
        corpus="Quora", encoder="E5-Mistral", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/e5/quora_vanilla-e5m7_train.fbin"),
        query=lambda: fbin(f"{JH}/e5/quora_vanilla-e5m7_query.fbin")),
    "msmarco_e5m": dict(
        corpus="MS MARCO", encoder="E5-Mistral", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/e5/msmarco-1m__vanilla-e5mistral_train.fbin"),
        query=lambda: fbin(f"{JH}/e5/msmarco-1m__vanilla-e5mistral_query.fbin")),
    "dbpedia_e5m": dict(
        corpus="DBpedia", encoder="E5-Mistral", d=4096, group="eval",
        base=lambda: fbin(f"{JH}/e5/dbpedia-entity__vanilla-e5mistral_train.fbin"),
        query=lambda: fbin(f"{JH}/e5/dbpedia-entity__vanilla-e5mistral_query.fbin")),

    # ---- conventional controls -----------------------------------------
    "glove": dict(
        corpus="GloVe", encoder="-", d=100, group="classic",
        base=lambda: h5(f"{KM}/glove-100-angular.hdf5", "train"),
        query=lambda: h5(f"{KM}/glove-100-angular.hdf5", "test")),
    "gist": dict(
        corpus="GIST", encoder="-", d=960, group="classic",
        base=lambda: fbin(f"{JH}/gist_rotated/gist-960_train.fbin"),
        query=lambda: fbin(f"{JH}/gist_rotated/gist-960_query.fbin")),
}

ORDER = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06",
         "msmarco_bgem3", "nq_bgem3", "quora_bgem3", "fever_bgem3",
         "dbpedia_bgel", "dbpedia_bgeb", "dbpedia_ada2", "msmarco_ada2",
         "msmarco_qw8", "nq_nem8", "quora_nem8", "nq_e5m", "quora_e5m", "msmarco_e5m",
         "dbpedia_e5m", "glove", "gist"]


def load(key, nq=NQ_MAX):
    """Unit-normalised base and query matrices; every configuration is scored by
    cosine similarity, matching Section 4 of the paper."""
    c = CONFIGS[key]
    X = c["base"]().astype(np.float32, copy=False)
    Q = c["query"]().astype(np.float32, copy=False)[:nq]
    X = np.require(X, np.float32, ["C", "W", "O"])
    Q = np.require(Q, np.float32, ["C", "W", "O"])
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(1e-30)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-30)
    return X, Q, c


def label(key):
    c = CONFIGS[key]
    return f"{c['corpus']} / {c['encoder']}" if c["encoder"] != "-" else c["corpus"]
