#!/bin/sh
# Build the paper out of tree and drop the PDF and a source zip in releases/<v>.
# TeX Live 2026 lives off /home because / has no room.
set -e
V=${1:-v16}
ROOT=/home/sieun/projects/BBQETA
TL=/home/smrc/samsung-nvme/sieun/texlive/2026/bin/x86_64-linux
PATH=$TL:$PATH
export PATH
B=$(mktemp -d)
cp -r "$ROOT/paper/." "$B/"
cd "$B"
xelatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true
bibtex paper > /dev/null 2>&1 || true
xelatex -interaction=nonstopmode paper.tex > /dev/null 2>&1 || true
xelatex -interaction=nonstopmode paper.tex > build.log 2>&1 || true
echo "--- $(grep -o 'Output written on paper.pdf ([0-9]* pages' build.log)"
echo "--- undefined: $(grep -c 'Reference .* undefined\|Citation .* undefined' build.log)"
mkdir -p "$ROOT/releases/$V"
cp paper.pdf "$ROOT/releases/$V/BBQ_${V}_full.pdf"
cd "$ROOT/paper"
rm -f "$ROOT/releases/$V/bbq_${V}_src.zip"
zip -qr "$ROOT/releases/$V/bbq_${V}_src.zip" . -x '*.aux' '*.log' '*.out' '*.bbl' '*.blg'
echo "--- wrote $ROOT/releases/$V/BBQ_${V}_full.pdf and bbq_${V}_src.zip"
rm -rf "$B"
