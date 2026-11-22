#!/usr/bin/env bash
# Rewrite a hard-coded absolute path prefix in every runner under exp/scripts.
#   exp/tools/relocate_paths.sh OLD_PREFIX NEW_PREFIX
# The runners are kept byte-identical to the ones that produced results/; this is the one sanctioned edit.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 OLD_PREFIX NEW_PREFIX" >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/../scripts"
n=$(grep -rlF -- "$1" . | wc -l)
grep -rlF -- "$1" . | xargs -r sed -i "s|$1|$2|g"
echo "rewrote '$1' -> '$2' in ${n} files"
