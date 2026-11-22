#!/usr/bin/env bash
# Rewrite the original testbed's absolute paths in exp/scripts to the locations set in exp/env.sh.
#   source exp/env.sh && exp/tools/relocate_paths.sh            # all prefixes at once
#   exp/tools/relocate_paths.sh OLD_PREFIX NEW_PREFIX            # a single prefix
# Longer prefixes are rewritten first, so e.g. .../samsung-nvme/jihyo is mapped before .../samsung-nvme.
# The runners are otherwise byte-identical to the ones that produced results/; this is the one sanctioned edit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../scripts"

rewrite() {  # OLD NEW
  [[ "$1" == "$2" ]] && return 0
  local files; files=$(grep -rlF -- "$1" . || true)
  [[ -z "$files" ]] && return 0
  echo "$files" | xargs sed -i "s|$1|$2|g"
  printf '%-48s -> %-40s %3d files\n' "$1" "$2" "$(echo "$files" | wc -l)"
}

if [[ $# -eq 2 ]]; then rewrite "$1" "$2"; exit 0; fi
[[ $# -eq 0 ]] || { echo "usage: $0 [OLD_PREFIX NEW_PREFIX]" >&2; exit 2; }
: "${BBQ_DATA:?source exp/env.sh first}"

# original prefix                          variable from env.sh
while read -r old new; do rewrite "$old" "$new"; done <<MAP
/home/smrc/samsung-nvme/ann-benchmark-data ${ANN_BENCH_DATA}
/home/smrc/samsung-nvme/jihyo              ${BBQ_DATA}
/home/smrc/samsung-nvme/sieun              ${BBQ_DATA2}
/home/smrc/samsung-nvme                    ${NVME_ROOT}
/home/jihyo/workspace/DiskVecLab           ${DISKVECLAB_ROOT}
/home/jihyo/workspace/RaBitQ-SSD           ${RABITQ_SSD_ROOT}
/home/jihyo/workspace/PipeANN              ${PIPEANN_ROOT}
/home/jihyo/workspace/SPTAG                ${SPTAG_ROOT}
/home/jihyo/workspace/RESQ                 ${RESQ_ROOT}
/home/jihyo/workspace/qdrant               ${QDRANT_ROOT}
/home/jihyo/workspace/faiss                ${FAISS_ROOT}
/home/jihyo/anaconda3                      ${CONDA_ROOT}
MAP
left=$( (grep -rhoE '/home/(smrc|jihyo)/[^"'"'"' ]*' . || true) | sort -u | wc -l)
echo "remaining original absolute paths: ${left}"
