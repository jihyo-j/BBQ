# Source this before running anything under exp/scripts:   source exp/env.sh
# Defaults are the paths of the original testbed; override them for your machine.

# --- toolchain ---------------------------------------------------------------------------
export FAISS_ROOT="${FAISS_ROOT:-/home/jihyo/workspace/faiss}"                       # FAISS 1.14.x source
export FAISS_BUILD="${FAISS_BUILD:-${FAISS_ROOT}/build_hnsw_conda/faiss}"           # holds libfaiss_avx2.so
export BLAS_LIB="${BLAS_LIB:-/home/jihyo/anaconda3/envs/hnsw/lib}"                  # holds libopenblas.so
export CONDA_LIB="${BLAS_LIB}"                                                      # name used by older scripts

# --- external systems (patched; see exp/baselines/patches/*/BASE) ------------------------
export DISKVECLAB_ROOT="${DISKVECLAB_ROOT:-/home/jihyo/workspace/DiskVecLab}"
export PIPEANN_ROOT="${PIPEANN_ROOT:-/home/jihyo/workspace/PipeANN}"
export SPTAG_ROOT="${SPTAG_ROOT:-/home/jihyo/workspace/SPTAG}"
export RABITQ_SSD_ROOT="${RABITQ_SSD_ROOT:-/home/jihyo/workspace/RaBitQ-SSD}"
export RESQ_ROOT="${RESQ_ROOT:-/home/jihyo/workspace/RESQ}"
export QDRANT_ROOT="${QDRANT_ROOT:-/home/jihyo/workspace/qdrant}"

# --- data (embeddings, ground truth, indexes; all on the NVMe under test) ---------------
export BBQ_DATA="${BBQ_DATA:-/home/smrc/samsung-nvme/jihyo}"       # most corpora, aligned FP32 sidecars, indexes
export BBQ_DATA2="${BBQ_DATA2:-/home/smrc/samsung-nvme/sieun}"     # MS MARCO 100M (qwen100m), msmarco-qwen1024, BEIR OpenAI
export ANN_BENCH_DATA="${ANN_BENCH_DATA:-/home/smrc/samsung-nvme/ann-benchmark-data}"

# The runners were written against the paths above. If yours differ, rewrite them once:
#   exp/tools/relocate_paths.sh /home/smrc/samsung-nvme/jihyo "$BBQ_DATA"
