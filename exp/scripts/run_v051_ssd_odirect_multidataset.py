#!/usr/bin/env python3
"""SSD recall/QPS sweep for RaBitQ v0.5.1 and ADC-BQ.

Methods:
  * RaBitQ 1-bit HNSW with adaptive FP reads through synchronous O_DIRECT.
  * ADC-BQ int16/nibble LUT HNSW with fixed FP rerank@80 through
    O_DIRECT + io_uring.
  * RaBitQ total 7-bit HNSW in RAM, without FP reranking.

All indexes use M=24 and efConstruction=200. Search thread count is
configurable and defaults to one.  The script is restartable: a method is
skipped when its CSV already contains all requested efSearch values.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RABITQ_BINDING_DIR = REPO / "experiment/rabitq_v051_adaptive_odirect_build/rabitqlib"
ADCBQ_BIN = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw"
CLUSTER_HELPER = REPO / "experiment/cluster_fbin_ivf256.py"
CLUSTER_PYTHON = Path("/home/jihyo/anaconda3/bin/python")
NVME_INDEX_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/rabitq_v051_ssd_multidataset_indexes"
)
RESULT_DIR = REPO / "experiment/v051_ssd_odirect_multidataset"
FIGURE_DIR = REPO / "experiment/figures/v051_ssd_odirect_multidataset"
DEFAULT_EFS = [64, 128, 256, 512, 1024]
RABITQ_COMMIT = "44f8a607741e9e54f6dda3b38c325071e808aa45"


@dataclass(frozen=True)
class Dataset:
    key: str
    corpus: str
    model: str
    title: str
    train: Path
    query: Path
    gt: Path
    b1_index: Path
    b7_index: Path
    adcbq_index: Path
    index_provenance: str = "existing-compatible"


OPENAI_ROOT = Path("/home/smrc/samsung-nvme/ann-benchmark-data/deep_1B/3072/openai")
MARCO_OPENAI_ROOT = Path("/home/smrc/samsung-nvme/ann-benchmark-data/deep_1B/3072/marco")
QWEN_ROOT = Path("/home/smrc/samsung-nvme/jihyo/qwen")
E5_ROOT = Path("/home/smrc/samsung-nvme/jihyo/e5")
DBP_BGE_ROOT = Path(
    "/home/smrc/samsung-nvme/ann-benchmark-data/deep_1B/3072/dbpedia-100k/bge"
)
MS_BGE_ROOT = Path("/home/smrc/samsung-nvme/ann-benchmark-data/msmarco/1m/bge_1m")
QUORA_BENCH_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/quora_sieun_adcbq_rabitq"
)
NQ_BENCH_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq"
)
JINA_BENCH_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/jina_sieun_adcbq_rabitq"
)
MSCOCO_BENCH_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/mscoco_sieun_adcbq_rabitq"
)
FEVER_BENCH_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/fever_sieun_adcbq_rabitq"
)


def generated_index(key: str, kind: str, dim: int) -> Path:
    root = NVME_INDEX_ROOT / key
    suffix = "index" if kind.startswith("rabitq_") else "faiss"
    path = root / f"{key}.{kind}.d{dim}.M24.efC200.{suffix}"
    # Early runs used a cosmetic .faiss suffix for generated RaBitQ files.
    # Keep those files reusable while using .index for future builds.
    legacy = root / f"{key}.{kind}.d{dim}.M24.efC200.faiss"
    if suffix == "index" and legacy.exists() and not path.exists():
        return legacy
    return path


def ds(
    key: str,
    corpus: str,
    model: str,
    dim: int,
    train: Path,
    query: Path,
    gt: Path,
    b1: Path | None,
    b7: Path | None,
    adc: Path | None,
) -> Dataset:
    return Dataset(
        key=key,
        corpus=corpus,
        model=model,
        title=f"{corpus} {model} ({dim}D)",
        train=train,
        query=query,
        gt=gt,
        b1_index=b1 or generated_index(key, "rabitq_b1_c256", dim),
        b7_index=b7 or generated_index(key, "rabitq_b7_c256", dim),
        adcbq_index=adc or generated_index(key, "fpgraph_adcbq", dim),
    )


DATASETS: dict[str, Dataset] = {}


def add(dataset: Dataset) -> None:
    DATASETS[dataset.key] = dataset


def load_extra_datasets_from_manifest() -> None:
    """Register experiment-local datasets without changing the base matrix.

    The manifest is opt-in through an environment variable so normal runs keep
    the historical registry byte-for-byte.  This is used by queued campaigns
    whose large converted inputs are created only after earlier experiments
    have released their temporary SSD space.
    """
    raw_path = os.environ.get("DISKVECLAB_EXTRA_DATASETS_MANIFEST", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    allow_override = os.environ.get(
        "DISKVECLAB_ALLOW_DATASET_OVERRIDE", ""
    ).strip().lower() in {"1", "true", "yes"}
    for item in payload.get("datasets", []):
        key = str(item["key"])
        dim = int(item["dim"])
        dataset = ds(
            key,
            str(item["corpus"]),
            str(item["model"]),
            dim,
            Path(str(item["train"])),
            Path(str(item["query"])),
            Path(str(item["gt"])),
            Path(str(item["b1_index"])),
            Path(str(item["b7_index"])),
            Path(str(item["adcbq_index"])),
        )
        if key in DATASETS and not allow_override:
            raise ValueError(f"extra dataset key collides with registry: {key}")
        add(dataset)


# DBpedia OpenAI: the local 999K set is the DBpedia 1M embedding set.
for dim in (1536, 3072):
    stem = f"openai-{dim}-1M"
    rabitq_root = Path(f"/home/smrc/samsung-nvme/jihyo/openai{dim}_bits/rabitq_indexes")
    adc = (
        Path(
            "/home/smrc/samsung-nvme/jihyo/fpgraph_adcbq_indexes/"
            "openai-1536-1M.fpgraph_pgbit_binary_hnsw.n999000.d1536.M24.efC200.faiss"
        )
        if dim == 1536
        else None
    )
    add(
        ds(
            f"dbpedia_openai{dim}",
            "DBpedia",
            f"OpenAI-{dim}",
            dim,
            OPENAI_ROOT / f"{stem}-train.fbin",
            OPENAI_ROOT / f"{stem}-query.fbin",
            OPENAI_ROOT / f"{stem}.ibin",
            rabitq_root / f"{stem}-rabitq-hnsw-b1-c256-M24-efC200.index",
            rabitq_root / f"{stem}-rabitq-hnsw-b7-c256-M24-efC200.index",
            adc,
        )
    )

# DBpedia and MSMARCO Qwen.
for corpus, prefix in (("DBpedia", "dbpedia-entity"), ("MSMARCO", "msmarco-1m")):
    for dim, tag in ((1024, "qwen06b"), (2560, "qwen4b"), (4096, "qwen8b")):
        stem = f"{prefix}__vanilla-{tag}"
        key = f"{corpus.lower()}_qwen{dim}"
        rabitq_root = QWEN_ROOT / "indexes/rabitq"
        adc: Path | None = None
        if corpus == "DBpedia" and dim == 1024:
            adc = (
                REPO
                / "indexes/dbpedia_graph_build_bq_adcbq/"
                "dbpedia_qwen1024.fp_graph_bq_storage.n1000000.d1024.M24.efC200.faiss"
            )
        if corpus == "MSMARCO" and dim == 1024:
            adc = Path(
                "/home/smrc/samsung-nvme/jihyo/msmarco_qwen1024_scale_v014/"
                "indexes/faiss_adcbq/"
                "msmarco-qwen1024-1m.fpgraph_pgbit_binary_hnsw.n1000000.d1024.M24.efC200.faiss"
            )
        add(
            ds(
                key,
                corpus,
                f"Qwen-{dim}",
                dim,
                QWEN_ROOT / f"{stem}_train.fbin",
                QWEN_ROOT / f"{stem}_query.fbin",
                QWEN_ROOT / f"{stem}_neighbors.ibin",
                rabitq_root
                / f"{stem}.rabitq_b1_c256.n1000000.d{dim}.M24.efC200.index",
                rabitq_root
                / f"{stem}.rabitq_b7_c256.n1000000.d{dim}.M24.efC200.index",
                adc,
            )
        )

# DBpedia and MSMARCO BGE.
for corpus in ("DBpedia", "MSMARCO"):
    for dim in (768, 1024):
        key = f"{corpus.lower()}_bge{dim}"
        if corpus == "DBpedia":
            train = DBP_BGE_ROOT / f"bge-{dim}-100k_train.fbin"
            query = DBP_BGE_ROOT / f"bge-{dim}-100k_query.fbin"
            gt = DBP_BGE_ROOT / f"bge-{dim}-100k.ibin"
            b1 = (
                REPO
                / "indexes/dbpedia_v051_rabitq1_adcbq_rerank/"
                "dbpedia_bge768.rabitq_v051_b1_c256.n99000.d768.M24.efC200.index"
                if dim == 768
                else None
            )
            b7 = None
            adc = (
                REPO
                / "indexes/dbpedia_graph_build_bq_adcbq/"
                "dbpedia_bge768.fp_graph_bq_storage.n99000.d768.M24.efC200.faiss"
                if dim == 768
                else None
            )
        else:
            train = MS_BGE_ROOT / f"msmarco-bge-{dim}-train.fbin"
            query = MS_BGE_ROOT / f"msmarco-bge-{dim}-query.fbin"
            gt = MS_BGE_ROOT / f"msmarco-bge-{dim}-neighbors.ibin"
            rabitq_root = Path(
                f"/home/smrc/samsung-nvme/jihyo/rabitq_bge{dim}_indexes"
            )
            b1 = rabitq_root / (
                f"msmarco-bge-{dim}-1M.rabitq_b1_c256.n1000000.d{dim}.M24.efC200.index"
            )
            b7 = rabitq_root / (
                f"msmarco-bge-{dim}-1M.rabitq_b7_c256.n1000000.d{dim}.M24.efC200.index"
            )
            adc = Path(
                "/home/smrc/samsung-nvme/jihyo/fpgraph_adcbq_indexes/"
                f"msmarco-bge-{dim}-1M.fpgraph_pgbit_binary_hnsw."
                f"n1000000.d{dim}.M24.efC200.faiss"
            )
        add(ds(key, corpus, f"BGE-{dim}", dim, train, query, gt, b1, b7, adc))

# DBpedia and MSMARCO E5-Mistral (4096D).
for corpus, prefix in (("DBpedia", "dbpedia-entity"), ("MSMARCO", "msmarco-1m")):
    stem = f"{prefix}__vanilla-e5mistral"
    key = f"{corpus.lower()}_e5mistral4096"
    rabitq_root = E5_ROOT / "indexes/rabitq"
    add(
        ds(
            key,
            corpus,
            "E5-Mistral-4096",
            4096,
            E5_ROOT / f"{stem}_train.fbin",
            E5_ROOT / f"{stem}_query.fbin",
            E5_ROOT / f"{stem}_neighbors.ibin",
            rabitq_root
            / f"{stem}.rabitq_b1_c256.n1000000.d4096.M24.efC200.index",
            rabitq_root
            / f"{stem}.rabitq_b7_c256.n1000000.d4096.M24.efC200.index",
            None,
        )
    )

# Only a local MSMARCO OpenAI-1536 set is available (not OpenAI-3072).
add(
    ds(
        "msmarco_openai1536",
        "MSMARCO",
        "OpenAI-1536",
        1536,
        MARCO_OPENAI_ROOT / "msmarco-v1-openai-ada2-1M_train.fbin",
        MARCO_OPENAI_ROOT / "msmarco-v1-openai-ada2-1M_query.fbin",
        MARCO_OPENAI_ROOT / "msmarco-v1-openai-ada2-1M.ibin",
        None,
        None,
        None,
    )
)

# Sieun's Quora BGE-M3 embeddings.  Both arms contain the same Quora documents
# and queries; the SYN arm applies an adapter trained on MSMARCO.
for arm, model in (
    ("vanilla", "BGE-M3 vanilla"),
    ("syn", "BGE-M3 SYN (MSMARCO adapter)"),
):
    add(
        ds(
            f"quora_bge_m3_{arm}",
            "Quora",
            model,
            1024,
            QUORA_BENCH_ROOT / f"quora_{arm}_train.fbin",
            QUORA_BENCH_ROOT / f"quora_{arm}_query.fbin",
            QUORA_BENCH_ROOT / f"quora_{arm}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )

# Sieun's Quora Qwen3 embeddings.
for arm, dim, model in (
    ("vanilla-qw06", 1024, "Qwen3-Embedding-0.6B"),
    ("vanilla-qw4", 2560, "Qwen3-Embedding-4B"),
    ("vanilla-qw8", 4096, "Qwen3-Embedding-8B"),
):
    add(
        ds(
            f"quora_qwen{dim}",
            "Quora",
            model,
            dim,
            QUORA_BENCH_ROOT / f"quora_{arm}_train.fbin",
            QUORA_BENCH_ROOT / f"quora_{arm}_query.fbin",
            QUORA_BENCH_ROOT / f"quora_{arm}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )

# Sieun's Jina v2 base-en embeddings.
for key, corpus, stem in (
    ("dbpedia_jina768", "DBpedia", "dbpedia-entity_vanilla-jina"),
    ("msmarco_jina768", "MSMARCO", "msmarco-1m_vanilla-jina"),
):
    add(
        ds(
            key,
            corpus,
            "Jina-v2-base-en",
            768,
            JINA_BENCH_ROOT / f"{stem}_train.fbin",
            JINA_BENCH_ROOT / f"{stem}_query.fbin",
            JINA_BENCH_ROOT / f"{stem}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )

# Sieun's MSMARCO Jina v4 Matryoshka embeddings.  The prepared fbin files
# contain the first k components of the 2048D output; every benchmark path
# normalizes those prefixes before cosine/IP indexing and search.
for dim in (512, 1024, 2048):
    stem = f"msmarco-1m_vanilla-jv4-d{dim}"
    add(
        ds(
            f"msmarco_jina_v4_{dim}",
            "MSMARCO",
            f"Jina-v4 Matryoshka-{dim}",
            dim,
            JINA_BENCH_ROOT / f"{stem}_train.fbin",
            JINA_BENCH_ROOT / f"{stem}_query.fbin",
            JINA_BENCH_ROOT / f"{stem}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )

# Sieun's Natural Questions BGE-M3 embeddings.  The SYN arm uses the same
# MSMARCO-trained adapter as the Quora SYN arm.
for arm, model in (
    ("vanilla", "BGE-M3 vanilla"),
    ("syn", "BGE-M3 SYN (MSMARCO adapter)"),
):
    add(
        ds(
            f"nq_bge_m3_{arm}",
            "NQ",
            model,
            1024,
            NQ_BENCH_ROOT / f"nq_{arm}_train.fbin",
            NQ_BENCH_ROOT / f"nq_{arm}_query.fbin",
            NQ_BENCH_ROOT / f"nq_{arm}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )

# Sieun's Natural Questions Qwen3 embeddings.  The 4B-1024 arm is the
# separately materialized Matryoshka-truncated output from the 2560D model.
for key, arm, dim, model in (
    ("nq_qwen06_1024", "vanilla-qw06", 1024, "Qwen3-Embedding-0.6B"),
    ("nq_qwen4_2560", "vanilla-qw4", 2560, "Qwen3-Embedding-4B"),
    ("nq_qwen8_4096", "vanilla-qw8", 4096, "Qwen3-Embedding-8B"),
    (
        "nq_qwen4_mrl1024",
        "vanilla-qw4-1024",
        1024,
        "Qwen3-Embedding-4B MRL-1024",
    ),
):
    add(
        ds(
            key,
            "NQ",
            model,
            dim,
            NQ_BENCH_ROOT / f"nq_{arm}_train.fbin",
            NQ_BENCH_ROOT / f"nq_{arm}_query.fbin",
            NQ_BENCH_ROOT / f"nq_{arm}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )

# Sieun's 4096D modern-encoder panel for NQ and Quora.  ``vanilla-nem8``
# is NVIDIA llama-embed-nemotron-8b and ``vanilla-e5m7`` is the
# E5-Mistral-7B arm.  The same registry entries are consumed by the calibrated
# ADC-BQ, RaBitQ, DiskANN, SPANN/PipeANN, and Elasticsearch runners.
for corpus, prefix, root in (
    ("NQ", "nq", NQ_BENCH_ROOT),
    ("Quora", "quora", QUORA_BENCH_ROOT),
):
    for suffix, arm, model in (
        ("nemotron4096", "vanilla-nem8", "NVIDIA Nemotron-8B"),
        ("e5mistral4096", "vanilla-e5m7", "E5-Mistral-7B"),
    ):
        # Nemotron files live with the corpus-specific NQ/Quora panel, while
        # the E5-Mistral panel is stored under the shared E5 root.
        data_root = E5_ROOT if suffix == "e5mistral4096" else root
        add(
            ds(
                f"{corpus.lower()}_{suffix}",
                corpus,
                model,
                4096,
                data_root / f"{prefix}_{arm}_train.fbin",
                data_root / f"{prefix}_{arm}_query.fbin",
                data_root / f"{prefix}_{arm}_neighbors_top100.ibin",
                None,
                None,
                None,
            )
        )

# Sieun's MS COCO and FEVER BGE-M3 embeddings.  MS COCO only has a vanilla
# arm; FEVER also has the MSMARCO-trained SYN adapter arm.
add(
    ds(
        "mscoco_bge_m3_vanilla",
        "MS COCO",
        "BGE-M3 vanilla",
        1024,
        MSCOCO_BENCH_ROOT / "mscoco_vanilla_train.fbin",
        MSCOCO_BENCH_ROOT / "mscoco_vanilla_query.fbin",
        MSCOCO_BENCH_ROOT / "mscoco_vanilla_neighbors_top100.ibin",
        None,
        None,
        None,
    )
)
for arm, model in (
    ("vanilla", "BGE-M3 vanilla"),
    ("syn", "BGE-M3 SYN (MSMARCO adapter)"),
):
    add(
        ds(
            f"fever_bge_m3_{arm}",
            "FEVER",
            model,
            1024,
            FEVER_BENCH_ROOT / f"fever_{arm}_train.fbin",
            FEVER_BENCH_ROOT / f"fever_{arm}_query.fbin",
            FEVER_BENCH_ROOT / f"fever_{arm}_neighbors_top100.ibin",
            None,
            None,
            None,
        )
    )


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as fp:
        header = fp.read(8)
    if len(header) != 8:
        raise ValueError(f"bad matrix header: {path}")
    return struct.unpack("<ii", header)


def read_fbin(path: Path, max_rows: int = 0) -> np.ndarray:
    n, dim = matrix_shape(path)
    rows = min(n, max_rows) if max_rows else n
    payload_bytes = n * dim * np.dtype(np.float32).itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes == 8 + payload_bytes:
        data_offset = 8
    elif actual_bytes == 4096 + payload_bytes:
        data_offset = 4096
    else:
        raise ValueError(
            f"unexpected fbin size: {path} has {actual_bytes} bytes; "
            f"expected {8 + payload_bytes} or {4096 + payload_bytes}"
        )
    with path.open("rb") as fp:
        fp.seek(data_offset)
        data = np.fromfile(fp, dtype=np.float32, count=rows * dim)
    if data.size != rows * dim:
        raise ValueError(f"short fbin: {path}")
    return np.ascontiguousarray(data.reshape(rows, dim), dtype=np.float32)


def read_ibin(path: Path, max_rows: int) -> np.ndarray:
    n, width = matrix_shape(path)
    rows = min(n, max_rows) if max_rows else n
    with path.open("rb") as fp:
        fp.seek(8)
        data = np.fromfile(fp, dtype=np.int32, count=rows * width)
    if data.size != rows * width:
        raise ValueError(f"short ibin: {path}")
    return data.reshape(rows, width)


def normalize_rows(x: np.ndarray, batch: int = 65536) -> None:
    for start in range(0, len(x), batch):
        block = x[start : start + batch]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        block /= norms


def recall_at_10(ids: np.ndarray, gt: np.ndarray) -> float:
    hits = 0
    for got, truth in zip(np.asarray(ids)[:, :10], np.asarray(gt)[:, :10]):
        truth_set = set(map(int, truth))
        hits += sum(int(value) in truth_set for value in got)
    return hits / float(len(ids) * 10)


def import_rabitq():
    sys.path.insert(0, str(RABITQ_BINDING_DIR))
    from _rabitqlib import HnswIndex  # type: ignore

    return HnswIndex


def complete_csv(
    path: Path,
    efs: list[int],
    search_threads: int | None = None,
    rounds: int | None = None,
) -> bool:
    try:
        with path.open(newline="") as fp:
            rows = list(csv.DictReader(fp))
        if sorted(int(row["efSearch"]) for row in rows) != sorted(efs):
            return False
        if search_threads is not None and any(
            int(row.get("search_threads", -1)) != search_threads for row in rows
        ):
            return False
        if rounds is not None and any(int(row.get("rounds", -1)) != rounds for row in rows):
            return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def output_csv(dataset: Dataset, family: str) -> Path:
    return RESULT_DIR / f"{dataset.key}.{family}.M24.efC200.csv"


def validate(dataset: Dataset) -> None:
    for path in (dataset.train, dataset.query, dataset.gt):
        if not path.exists():
            raise FileNotFoundError(path)
    nb, dim = matrix_shape(dataset.train)
    nq, qdim = matrix_shape(dataset.query)
    ngt, width = matrix_shape(dataset.gt)
    if dim != qdim or nq != ngt or width < 10:
        raise ValueError(
            f"shape mismatch {dataset.key}: train={nb}x{dim}, query={nq}x{qdim}, "
            f"gt={ngt}x{width}"
        )
    print(f"DATASET {dataset.key}: train={nb}x{dim}, query={nq}, gt_width={width}", flush=True)


def cluster_files(dataset: Dataset) -> tuple[Path, Path]:
    root = NVME_INDEX_ROOT / dataset.key
    return root / "centroids_c256.npy", root / "cluster_ids_c256.npy"


def cluster_data(
    dataset: Dataset, train: np.ndarray, threads: int
) -> tuple[np.ndarray, np.ndarray, float]:
    centroids_path, ids_path = cluster_files(dataset)
    if centroids_path.exists() and ids_path.exists():
        centroids = np.ascontiguousarray(np.load(centroids_path), dtype=np.float32)
        ids = np.ascontiguousarray(np.load(ids_path), dtype=np.uint32)
        if centroids.shape == (256, train.shape[1]) and ids.shape == (len(train),):
            return centroids, ids, 0.0
    start = time.perf_counter()
    subprocess.run(
        [
            str(CLUSTER_PYTHON),
            str(CLUSTER_HELPER),
            "--train",
            str(dataset.train),
            "--centroids",
            str(centroids_path),
            "--assignments",
            str(ids_path),
            "--threads",
            str(threads),
        ],
        cwd=REPO,
        check=True,
    )
    elapsed = time.perf_counter() - start
    centroids = np.ascontiguousarray(np.load(centroids_path), dtype=np.float32)
    ids = np.ascontiguousarray(np.load(ids_path), dtype=np.uint32)
    if centroids.shape != (256, train.shape[1]) or ids.shape != (len(train),):
        raise ValueError(f"bad cluster helper output for {dataset.key}")
    return centroids, ids, elapsed


def ensure_rabitq_indexes(dataset: Dataset, args: argparse.Namespace) -> None:
    missing: list[tuple[int, Path]] = []
    for bits, path in ((1, dataset.b1_index), (7, dataset.b7_index)):
        if (bits == 1 and "b1" not in args.parts) or (
            bits == 7 and "b7" not in args.parts
        ):
            continue
        if args.rebuild or not path.exists():
            missing.append((bits, path))
    if not missing:
        return
    print(f"[{dataset.key}] load+normalize train for missing RaBitQ indexes", flush=True)
    train = read_fbin(dataset.train)
    normalize_rows(train)
    centroids, cluster_ids, cluster_seconds = cluster_data(
        dataset, train, args.build_threads
    )
    # Clustering is isolated in another Python environment because the FAISS
    # and RaBitQ native modules select incompatible MKL variants in this env.
    HnswIndex = import_rabitq()
    print(f"[{dataset.key}] clustering seconds={cluster_seconds:.2f}", flush=True)
    for bits, path in missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        index = HnswIndex(
            dim=train.shape[1],
            max_elements=train.shape[0],
            M=24,
            ef_construction=200,
            nbits=bits,
            metric="ip",
        )
        print(f"[{dataset.key}] BUILD RaBitQ b{bits}: {path}", flush=True)
        start = time.perf_counter()
        index.build(
            train,
            centroids,
            cluster_ids,
            num_threads=args.build_threads,
            fast_quantization=False,
        )
        index.save(str(path))
        print(
            f"[{dataset.key}] BUILT b{bits} seconds={time.perf_counter()-start:.2f}",
            flush=True,
        )
        del index
        gc.collect()
    del train, centroids, cluster_ids
    gc.collect()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def run_rabitq(dataset: Dataset, bits: int, args: argparse.Namespace) -> Path:
    family = "rabitq_b1_adaptive_odirect" if bits == 1 else "rabitq_b7_ram"
    out = output_csv(dataset, family)
    if complete_csv(out, args.efs, args.search_threads, args.rounds) and not args.force_search:
        print(f"SKIP complete {out}", flush=True)
        return out
    HnswIndex = import_rabitq()
    path = dataset.b1_index if bits == 1 else dataset.b7_index
    start = time.perf_counter()
    index = HnswIndex.load(str(path))
    load_seconds = time.perf_counter() - start
    if index.nbits != bits or index.dim != matrix_shape(dataset.train)[1]:
        raise ValueError(f"incompatible RaBitQ index: {path}")
    expected_clusters = getattr(args, "num_clusters", None)
    if expected_clusters is not None and index.num_clusters != expected_clusters:
        raise ValueError(
            f"unexpected RaBitQ cluster count: {index.num_clusters} "
            f"!= {expected_clusters}: {path}"
        )
    queries = read_fbin(dataset.query, args.max_queries)
    normalize_rows(queries)
    gt = read_ibin(dataset.gt, len(queries))
    rows: list[dict[str, object]] = []
    resume_partial = bool(getattr(args, "resume_partial", False))
    if resume_partial and out.is_file() and not args.force_search:
        try:
            with out.open(newline="", encoding="utf-8") as handle:
                partial = [dict(row) for row in csv.DictReader(handle)]
            requested = set(args.efs)
            if all(
                int(row["efSearch"]) in requested
                and int(row["search_threads"]) == args.search_threads
                and int(row["rounds"]) == args.rounds
                and int(row["bits"]) == bits
                and (
                    expected_clusters is None
                    or int(row.get("num_clusters", -1)) == expected_clusters
                )
                for row in partial
            ):
                rows.extend(partial)
                print(
                    f"RESUME {out}: completed efSearch="
                    f"{sorted(int(row['efSearch']) for row in rows)}",
                    flush=True,
                )
        except (OSError, KeyError, ValueError):
            rows = []
    completed_efs = {int(row["efSearch"]) for row in rows}
    for ef in args.efs:
        if ef in completed_efs:
            continue
        recalls: list[float] = []
        times: list[float] = []
        fp_means: list[float] = []
        fp_mins: list[int] = []
        fp_maxs: list[int] = []
        byte_counts: list[int] = []
        request_counts: list[int] = []
        for round_id in range(args.rounds):
            start = time.perf_counter()
            if bits == 1:
                ids, _, stats = index.search_1bit_adaptive_fp_odirect(
                    queries,
                    str(dataset.train),
                    k=10,
                    ef=ef,
                    num_threads=args.search_threads,
                )
                fp_reads = np.asarray(stats["fp_read_count"], dtype=np.uint64)
                byte_counts.append(int(stats["odirect_read_bytes"]))
                request_counts.append(int(stats["odirect_read_requests"]))
                fp_means.append(float(np.mean(fp_reads)))
                fp_mins.append(int(np.min(fp_reads)))
                fp_maxs.append(int(np.max(fp_reads)))
            else:
                ids, _ = index.search(
                    queries, k=10, ef=ef, num_threads=args.search_threads
                )
            elapsed = time.perf_counter() - start
            recall = recall_at_10(np.asarray(ids), gt)
            recalls.append(recall)
            times.append(elapsed)
            print(
                f"[{dataset.key}] RaBitQ-b{bits} ef={ef} "
                f"round={round_id+1}/{args.rounds} recall={recall:.6f} "
                f"qps={len(queries)/elapsed:.2f}"
                + (f" fp_reads={np.mean(fp_reads):.2f}" if bits == 1 else ""),
                flush=True,
            )
        mean_seconds = float(np.mean(times))
        rows.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method": (
                    "RaBitQ v0.5.1 1-bit + adaptive FP O_DIRECT"
                    if bits == 1
                    else "RaBitQ v0.5.1 total 7-bit (RAM, no rerank)"
                ),
                "library_commit": RABITQ_COMMIT,
                "storage": "SSD O_DIRECT" if bits == 1 else "RAM",
                "io_engine": "sync pread O_DIRECT" if bits == 1 else "none",
                "M": 24,
                "efConstruction": 200,
                "efSearch": ef,
                "k": 10,
                "nq": len(queries),
                "nb": index.max_elements,
                "dim": index.dim,
                "bits": bits,
                "num_clusters": index.num_clusters,
                "rerank_policy": "adaptive_lower_bound" if bits == 1 else "none",
                "rerank_n": "adaptive" if bits == 1 else 0,
                "fp_reads_mean": float(np.mean(fp_means)) if bits == 1 else 0,
                "fp_reads_min": min(fp_mins) if bits == 1 else 0,
                "fp_reads_max": max(fp_maxs) if bits == 1 else 0,
                "direct_bytes_mean": float(np.mean(byte_counts)) if bits == 1 else 0,
                "direct_requests_mean": (
                    float(np.mean(request_counts)) if bits == 1 else 0
                ),
                "rounds": args.rounds,
                "search_threads": args.search_threads,
                "load_seconds": load_seconds,
                "total_seconds_mean": mean_seconds,
                "recall_at_k": float(np.mean(recalls)),
                "qps": len(queries) / mean_seconds,
                "index_bytes": path.stat().st_size,
                "index_path": path,
            }
        )
        if resume_partial:
            rows.sort(key=lambda row: int(row["efSearch"]))
            write_rows(out, rows)
    rows.sort(key=lambda row: int(row["efSearch"]))
    write_rows(out, rows)
    del index, queries, gt
    gc.collect()
    return out


def run_adcbq(dataset: Dataset, args: argparse.Namespace) -> Path:
    out = output_csv(
        dataset, f"adcbq_int16_io_uring_rerank{args.adc_rerank_n}"
    )
    if complete_csv(out, args.efs, args.search_threads, args.rounds) and not args.force_search:
        print(f"SKIP complete {out}", flush=True)
        return out
    dataset.adcbq_index.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ADCBQ_BIN),
        "--dataset",
        dataset.key,
        "--train",
        str(dataset.train),
        "--query",
        str(dataset.query),
        "--gt",
        str(dataset.gt),
        "--index",
        str(dataset.adcbq_index),
        "--out",
        str(out),
        "--m",
        "24",
        "--ef-construction",
        "200",
        "--k",
        "10",
        "--rerank-n",
        str(args.adc_rerank_n),
        "--max-queries",
        str(args.max_queries),
        "--rounds",
        str(args.rounds),
        "--build-threads",
        str(args.build_threads),
        "--search-threads",
        str(args.search_threads),
        "--adc-mode",
        "int16",
        "--lut-mode",
        "nibble",
        "--graph-build",
        "fp",
        "--distance-mode",
        "adcbq",
        "--search-impl",
        "inline",
        "--code-layout",
        "block32",
        "--rerank-dot",
        "fp32",
        "--fp-rerank-source",
        "odirect-uring",
        "--fp-rerank-norms",
        "onfetch",
        "--efs",
        *map(str, args.efs),
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    with out.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if any(int(row.get("search_threads", -1)) != args.search_threads for row in rows):
        for row in rows:
            row["search_threads"] = args.search_threads
        write_rows(out, rows)
    return out


def combine(selected: list[Dataset], args: argparse.Namespace) -> Path:
    out = RESULT_DIR / "all_results.csv"
    families = (
        "rabitq_b1_adaptive_odirect",
        "adcbq_int16_io_uring_rerank80",
        "rabitq_b7_ram",
    )
    rows: list[dict[str, object]] = []
    for dataset in selected:
        for family in families:
            path = output_csv(dataset, family)
            if not complete_csv(path, args.efs):
                continue
            with path.open(newline="") as fp:
                for row in csv.DictReader(fp):
                    if family.startswith("adcbq"):
                        method = "ADC-BQ int16 LUT + FP rerank@80 (O_DIRECT io_uring)"
                        storage = "SSD O_DIRECT"
                        io_engine = "io_uring"
                        rerank_policy = "fixed_top80"
                        fp_reads = 80
                    else:
                        method = row["method"]
                        storage = row["storage"]
                        io_engine = row["io_engine"]
                        rerank_policy = row["rerank_policy"]
                        fp_reads = row["fp_reads_mean"]
                    seconds = float(row["total_seconds_mean"])
                    nq = int(row["nq"])
                    rows.append(
                        {
                            "dataset": dataset.key,
                            "corpus": dataset.corpus,
                            "model": dataset.model,
                            "title": dataset.title,
                            "family": family,
                            "method": method,
                            "storage": storage,
                            "io_engine": io_engine,
                            "M": 24,
                            "efConstruction": 200,
                            "efSearch": row["efSearch"],
                            "k": 10,
                            "nq": nq,
                            "nb": row["nb"],
                            "dim": row["dim"],
                            "rerank_policy": rerank_policy,
                            "fp_reads_mean": fp_reads,
                            "rounds": row["rounds"],
                            "recall_at_10": row["recall_at_k"],
                            "qps": row["qps"],
                            "latency_ms_per_query": seconds * 1000.0 / nq,
                            "index_path": row["index_path"],
                            "source_csv": path,
                        }
                    )
    if not rows:
        raise RuntimeError("no complete result CSVs to combine")
    write_rows(out, rows)
    return out


def plot(combined: Path, selected: list[Dataset]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with combined.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    styles = {
        "rabitq_b1_adaptive_odirect": ("#2ca02c", "^", "-"),
        "adcbq_int16_io_uring_rerank80": ("#d62728", "s", "--"),
        "rabitq_b7_ram": ("#9467bd", "D", "-."),
    }
    labels = {
        "rabitq_b1_adaptive_odirect": "RaBitQ 1-bit + adaptive FP (O_DIRECT)",
        "adcbq_int16_io_uring_rerank80": "ADC-BQ int16 + rerank@80 (io_uring)",
        "rabitq_b7_ram": "RaBitQ total 7-bit (RAM)",
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for corpus in dict.fromkeys(item.corpus for item in selected):
        datasets = [item for item in selected if item.corpus == corpus]
        if not datasets:
            continue
        cols = min(4, len(datasets))
        nrows = (len(datasets) + cols - 1) // cols
        fig, axes = plt.subplots(
            nrows,
            cols,
            figsize=(5.4 * cols, 4.4 * nrows + 1.2),
            squeeze=False,
        )
        for ax, dataset in zip(axes.flat, datasets):
            for family, (color, marker, linestyle) in styles.items():
                group = sorted(
                    (
                        row
                        for row in rows
                        if row["dataset"] == dataset.key and row["family"] == family
                    ),
                    key=lambda row: int(row["efSearch"]),
                )
                if not group:
                    continue
                xs = [float(row["recall_at_10"]) for row in group]
                ys = [float(row["qps"]) for row in group]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=5,
                    label=labels[family],
                )
                for x, y, row in zip(xs, ys, group):
                    ax.annotate(
                        row["efSearch"],
                        (x, y),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )
            ax.set_title(dataset.model)
            ax.set_xlabel("Recall@10")
            ax.set_ylabel("QPS (log scale)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
        for ax in axes.flat[len(datasets) :]:
            ax.axis("off")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(
            f"{corpus}: SSD O_DIRECT reranking vs RAM RaBitQ-7bit\n"
            "HNSW M=24, efC=200, k=10, 1 search thread",
            fontsize=15,
            y=0.985,
        )
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.895),
            ncol=min(3, len(handles)),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.79))
        for extension in ("png", "pdf"):
            path = FIGURE_DIR / f"{corpus.lower()}_recall_qps.{extension}"
            fig.savefig(path, dpi=190 if extension == "png" else None, bbox_inches="tight")
            created.append(path)
            print(f"WROTE {path}", flush=True)
        plt.close(fig)
    return created


def write_summary(combined: Path) -> Path:
    with combined.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    out = RESULT_DIR / "summary.md"
    lines = [
        "# RaBitQ v0.5.1 / ADC-BQ SSD experiment",
        "",
        "M=24, efConstruction=200, k=10, one search thread. ADC-BQ uses an "
        "int16 nibble LUT and fixed FP rerank@80 through O_DIRECT + io_uring. "
        "RaBitQ 1-bit uses adaptive synchronous O_DIRECT FP reads. RaBitQ "
        "total-7-bit runs in RAM without FP reranking.",
        "",
        "MSMARCO OpenAI-3072 is omitted because no local embedding/ground-truth set exists.",
        "",
        "| Dataset | Method | efSearch | Recall@10 | QPS | FP reads/query |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['title']} | {row['method']} | {row['efSearch']} | "
            f"{float(row['recall_at_10']):.6f} | {float(row['qps']):.2f} | "
            f"{float(row['fp_reads_mean']):.2f} |"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"WROTE {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--efs", nargs="+", type=int, default=DEFAULT_EFS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--build-threads", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--adc-rerank-n", type=int, default=80)
    parser.add_argument("--parts", nargs="+", choices=("b1", "adc", "b7"), default=("b1", "adc", "b7"))
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="run requested benchmark parts without overwriting combined reports",
    )
    args = parser.parse_args()

    if args.search_threads < 1:
        parser.error("--search-threads must be positive")
    if args.adc_rerank_n != 80 and not args.skip_report:
        parser.error("non-default --adc-rerank-n requires --skip-report")

    if not any(RABITQ_BINDING_DIR.glob("_rabitqlib*.so")):
        raise FileNotFoundError(f"missing v0.5.1 O_DIRECT binding: {RABITQ_BINDING_DIR}")
    if not ADCBQ_BIN.exists():
        raise FileNotFoundError(ADCBQ_BIN)
    selected = [DATASETS[key] for key in args.datasets]
    for dataset in selected:
        validate(dataset)

    if not args.plot_only:
        for dataset in selected:
            if "b1" in args.parts or "b7" in args.parts:
                ensure_rabitq_indexes(dataset, args)
            if "b1" in args.parts:
                run_rabitq(dataset, 1, args)
            if "adc" in args.parts:
                run_adcbq(dataset, args)
            if "b7" in args.parts:
                run_rabitq(dataset, 7, args)
    if args.skip_report:
        return
    combined = combine(selected, args)
    plot(combined, selected)
    write_summary(combined)


if __name__ == "__main__":
    main()
