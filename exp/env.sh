# Source this before running anything under exp/scripts:   source exp/env.sh
# Defaults are the paths of the original testbed; override them for your machine.

# --- toolchain ---------------------------------------------------------------------------
export FAISS_ROOT="${FAISS_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/faiss}"                       # FAISS 1.14.x source
export FAISS_BUILD="${FAISS_BUILD:-${FAISS_ROOT}/build_hnsw_conda/faiss}"           # holds libfaiss_avx2.so
export BLAS_LIB="${BLAS_LIB:-${CONDA_ROOT:-/home/jihyo/anaconda3}/envs/hnsw/lib}"                  # holds libopenblas.so
export CONDA_LIB="${BLAS_LIB}"                                                      # name used by older scripts

# --- external systems (patched; see exp/baselines/patches/*/BASE) ------------------------
export DISKVECLAB_ROOT="${DISKVECLAB_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/DiskVecLab}"
export PIPEANN_ROOT="${PIPEANN_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/PipeANN}"
export SPTAG_ROOT="${SPTAG_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/SPTAG}"
export RABITQ_SSD_ROOT="${RABITQ_SSD_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/RaBitQ-SSD}"
export RESQ_ROOT="${RESQ_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/RESQ}"
export QDRANT_ROOT="${QDRANT_ROOT:-${WORKSPACE_ROOT:-/home/jihyo/workspace}/qdrant}"

# --- data (embeddings, ground truth, indexes; all on the NVMe under test) ---------------
export NVME_ROOT="${NVME_ROOT:-/home/smrc/samsung-nvme}"
export BBQ_DATA="${BBQ_DATA:-${NVME_ROOT}/jihyo}"                  # most corpora, aligned FP32 sidecars, indexes
export BBQ_DATA2="${BBQ_DATA2:-${NVME_ROOT}/sieun}"                # MS MARCO 100M (qwen100m), msmarco-qwen1024, BEIR OpenAI
export ANN_BENCH_DATA="${ANN_BENCH_DATA:-${NVME_ROOT}/ann-benchmark-data}"

# --- Python environments (specs in exp/envs/*.yml) ---------------------------------------
export CONDA_ROOT="${CONDA_ROOT:-/home/jihyo/anaconda3}"          # base env (python 3.12): plotting, data prep
#   ${CONDA_ROOT}/envs/hnsw        python 3.11: FAISS-side runners
#   ${CONDA_ROOT}/envs/rabitq-cfr  python 3.11: RaBitQ v0.5.1 binding
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/jihyo/workspace}"  # parent of the external system checkouts

# The runners were written against the original paths. After editing the variables above, rewrite every
# hard-coded prefix in exp/scripts in one go (idempotent; prints what it changed):
#   exp/tools/relocate_paths.sh
