#!/bin/sh
# Render one figure block in the paper's own class, so what we look at is what
# the paper would print.  usage: preview.sh <body.tex> <out.png>
set -e
TL=/home/smrc/samsung-nvme/sieun/texlive/2026/bin/x86_64-linux
PATH=$TL:$PATH; export PATH
GP=/home/sieun/projects/BBQETA/gp
B=$(mktemp -d)
cat > "$B/prev.tex" <<'EOF'
\documentclass[sigconf, nonacm]{acmart}
\settopmatter{printacmref=false}
\usepackage{graphicx}\usepackage{subcaption}\usepackage{xspace}
\def\bbq{BBQ\xspace}\def\msmarco{MS MARCO\xspace}\def\quora{Quora\xspace}
\def\nq{NQ\xspace}\def\dbp{DBpedia\xspace}\def\adcbq{ADC-BQ\xspace}
\def\m{\texttt{M}\xspace}
\pagestyle{empty}
\settopmatter{printfolios=false}
\title{preview}
\author{a}\affiliation{\institution{x}\country{}}
\begin{document}
\maketitle
\input{body}
\null
\end{document}
EOF
cp "$1" "$B/body.tex"
cd "$B"
ln -s "$GP" figures_gp
xelatex -interaction=nonstopmode prev.tex > prev.log 2>&1 || true
grep -E "^! |Undefined control|Warning: Reference" prev.log | head -5 || true
cp "$B/prev.pdf" "${2%.png}.pdf"
python3 /home/sieun/projects/BBQETA/gp/shot.py "$B/prev.pdf" "$2" ${3:-0 0 1 1}
echo "wrote $2"
rm -rf "$B"
