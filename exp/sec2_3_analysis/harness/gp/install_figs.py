#!/usr/bin/env python3
"""Install the rebuilt panels and swap the four figure environments over.

Each figure becomes: a row of panel PDFs, then the legend strip under the
panels it actually describes, then one subcaption per panel.  The old
one-PDF-per-figure artwork is kept under figures-v2/old/.

Run from gp/.  Edits both the live paper tree and the Blessing BQ copy.
"""
import re, shutil, pathlib, sys

GP   = pathlib.Path("/home/sieun/projects/BBQETA/gp")
TREES = [pathlib.Path("/home/sieun/projects/BBQETA/paper"),
         pathlib.Path("/home/sieun/Blessing BQ")]
PANELS = ["f1a.pdf","f1b.pdf","f1_legend.pdf",
          "f2a.pdf","f2b.pdf","f2_legend.pdf",
          "f3a.pdf","f3b.pdf","f3c.pdf","f3d.pdf",
          "f3_legend.pdf","f3q_legend.pdf",
          "f4a.pdf","f4b.pdf","f4_legend.pdf"]
REPLACED = ["embuse.pdf","capacity4096_crop.pdf","mrl_scoreerr.pdf",
            "fig4.pdf","perq.pdf"]
SRC = ["house.gp","legend.gp","fig1.gp","fig3.gp","fig4p.gp","cut_fig2.py","fig2leg.gp",
       "make.sh","preview.sh","shot.py","crop.py","checkfit.py","legcheck.py",
       "fixwidths.py","install_figs.py","boxoff.py",
       "w_f1_legend.gp","w_f2_legend.gp","w_f3_legend.gp","w_f3d_legend.gp","w_f4_legend.gp"]

# Each panel's plot box sits right of its own canvas centre: the rotated y
# label and the widest tick label need ~0.41in on the left, the last x tick
# label only 0.09in on the right.  Centring a subcaption on the canvas
# therefore centres it on whitespace, and the box reads as shifted right --
# which is what it looked like.  So each subcaption is shifted by its own
# panel's measured box offset (gp/boxoff.py), in printed points.
# `oneside` because acmart is a twoside class, where the caption package reads
# margin={a,b} as {inner,outer} and silently mirrors the shift on even pages --
# which put Figure 4's captions 11pt to the LEFT of its boxes.
SUB = (r"""\begin{subfigure}[t]{%s}\captionsetup{oneside,margin={%.0fpt,0pt}}"""
       r"""\caption{%s}\label{%s}\end{subfigure}""")

def row(panels, width):
    return "\n".join(r"    \includegraphics[width=%s]{figures-v2/%s}%s"
                     % (width, p, r"\hfill" if i < len(panels)-1 else "")
                     for i, p in enumerate(panels))

def subs(items, width):
    return "\n".join("    " + SUB % (width, 2*off, cap, lab) +
                     (r"\hfill" if i < len(items)-1 else "")
                     for i, (cap, lab, off) in enumerate(items))

COL = r"0.49\columnwidth"
FOUR = r"0.245\textwidth"

BLOCKS = {}

BLOCKS["fig:embuse"] = (r"""\begin{figure}[t]
    \centering
    \captionsetup[subfigure]{skip=1pt,justification=centering}
""" + row(["f1a.pdf","f1b.pdf"], COL) + r"""
    \par\vspace{1.5pt}
    \includegraphics[width=\columnwidth]{figures-v2/f1_legend.pdf}
    \par
""" + subs([("ST namespace downloads","fig:embuse:a",13.3),
            ("VLDB evaluation mix","fig:embuse:b",13.3)], COL) + r"""
    \caption{Embedding activity and ANN evaluation datasets. (a) Annual
    downloads in Hugging Face's \texttt{sentence-transformers}
    namespace~\cite{xcidHfDownloads}; the 2025 total is $13.3\times$ the 2022
    one. (b) Representation shares across
    193 evaluation groups from 40 VLDB papers; labels give the Transformer and
    modern-API share.}
    \Description{(a) A line rising from 0.21 billion downloads in 2022 to 2.75
    billion in 2025 in the Hugging Face sentence-transformers namespace.
    (b) 100 per cent stacked bars by year for the representations used in VLDB
    ANN evaluations; the Transformer and modern-API share rises from zero
    before 2023 to about a fifth in 2025.}
    \label{fig:embuse}
\end{figure}""")

BLOCKS["fig:motiv"] = (r"""\begin{figure}[t]
    \centering
    \captionsetup[subfigure]{skip=1pt,justification=centering}
""" + row(["f2a.pdf","f2b.pdf"], COL) + r"""
    \par\vspace{1.5pt}
    \includegraphics[width=\columnwidth]{figures-v2/f2_legend.pdf}
    \par
""" + subs([(r"Index size","fig:motiv:a",7.6),
            (r"Score error","fig:motiv:b",10.8)], COL) + r"""
    \caption{Memory pressure and BQ accuracy at higher dimensions. (a) Index
    size at $d=4096$, including HNSW graph storage (208\,B/vector,
    $\m{}=24$); the dotted drops mark where each representation reaches the
    1007\,GiB budget.
    (b) \adcbq{} score distortion $c_Q$ across MRL dimensions of Qwen3-8B on
    \msmarco{}, compared with an isotropic reference (dashed).}
    \Description{Left: log-log index size against number of vectors for FP32,
    4-bit and 1-bit codes, each crossing the DRAM budget at a different corpus
    size. Right: the relative score error of a sign code falling from 0.80 at
    32 dimensions to 0.30 at 4096, against a flat isotropic reference near
    0.80.}
    \label{fig:motiv}
\end{figure}""")

BLOCKS["fig:profile"] = (r"""\begin{figure*}[t]
    \centering
    \captionsetup[subfigure]{skip=1pt,justification=centering}
    % the keys sit above the pair each describes, which is also what lets the
    % row below them be panels alone; (b) keys itself inside the panel
    \includegraphics[width=0.4967\textwidth]{figures-v2/f3_legend.pdf}\hfill
    \includegraphics[width=0.4967\textwidth]{figures-v2/f3q_legend.pdf}
    \par\vspace{1pt}
""" + row(["f3a.pdf","f3b.pdf","f3c.pdf","f3d.pdf"], FOUR) + r"""
    \par
""" + subs([("Exact score profile","fig:profile:a",11.4),
            ("MRL dimension","fig:profile:b",11.4),
            (r"Budget vs $k$","fig:profile:c",11.4),
            ("Quantizer distortion","fig:profile:d",11.4)], FOUR) + r"""
    \caption{Score geometry and candidate preservation. The key above
    (a)--(b) identifies configurations and the one above (c)--(d) quantizers;
    (b) keys its own metrics. An open marker is a configuration, a filled one a
    quantizer. In (b) every quantity is normalized to
    $d=32$, with ground truth recomputed for each truncated, renormalized
    representation. (c) reads the mean budget past the $k=10$
    convention on \msmarco{}: left of the break $k$ is linear, right of it
    logarithmic. A learned rotation buys $1.02\times$ over \adcbq{} at $k=1$ and
    $1.55\times$ at $k=100$, and quantizing the query costs $1.34\times$ at $k=1$
    and $3.44\times$ at $k=100$, so what a quantizer comparison reports is set
    by the $k$ it is run at as much as by the codes. (d) walks the MRL ladder for the same six codes and
    reports the distortion itself, $c_Q=\nu/\kappa$. Ground truth uses exact
    FP32 cosine ranking; budgets are p95 over queries except in (c), which is
    the mean over 1500 queries because a p95 is set by its top 5\% and jumps
    from one $k$ to the next.}
    \Description{Exact score profile by rank, relative error against Matryoshka
    dimension, the candidate budget against how many answers are kept, and the
    quantizer error ladder across dimensions.}
    \label{fig:profile}
\end{figure*}""")

BLOCKS["fig:perq"] = (r"""\begin{figure}[t]
    \centering
    \captionsetup[subfigure]{skip=1pt,justification=centering}
""" + row(["f4a.pdf","f4b.pdf"], COL) + r"""
    \par\vspace{1.5pt}
    \includegraphics[width=\columnwidth]{figures-v2/f4_legend.pdf}
    \par
""" + subs([(r"Coverage vs $R$","fig:perq:a",11.4),
            ("Fixed vs oracle","fig:perq:b",11.4)], COL) + r"""
    \caption{Per-query candidate requirement under \adcbq{}. (a) The fraction
    of queries whose whole exact top-10 lies in the top $R$. (b) The fixed
    budget $R_c$ reaching a coverage target, divided by the oracle mean
    $\frac1n\sum_i\min(R^{*}_i,R_c)$ at the same coverage. Both axes count
    candidates, not measured I/O.}
    \Description{Cumulative distribution of the per-query candidate budget, and
    the ratio between a fixed budget and an oracle per-query budget at matched
    coverage.}
    \label{fig:perq}
\end{figure}""")

FILES = {"fig:embuse": "1.intro.tex", "fig:motiv": "2.background.tex",
         "fig:profile": "3.BQonModernEmbedding.tex",
         "fig:perq": "3.BQonModernEmbedding.tex"}

def swap(text, label, block):
    """Replace the uncommented figure environment carrying `label`."""
    pat = re.compile(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", re.S)
    hits = [m for m in pat.finditer(text)
            if re.search(r"\\label\{" + re.escape(label) + r"\}", m.group(0))
            and not m.group(0).lstrip().startswith("%")]
    if len(hits) != 1:
        raise SystemExit("%s: matched %d figure environments" % (label, len(hits)))
    m = hits[0]
    return text[:m.start()] + block + text[m.end():]

for tree in TREES:
    new = tree / "figures-v2"
    (new / "old").mkdir(parents=True, exist_ok=True)
    (new / "src").mkdir(parents=True, exist_ok=True)
    for p in PANELS:
        shutil.copy2(GP / p, new / p)
    for s in SRC:
        shutil.copy2(GP / s, new / "src" / s)
    shutil.copy2(GP / ".." / "src" / "export_perq.py", new / "src" / "export_perq.py")
    for o in REPLACED:
        old = tree / "figures" / "bbq" / o
        if old.exists():
            shutil.copy2(old, new / "old" / o)
    for label, fname in FILES.items():
        f = tree / fname
        t = f.read_text()
        t = swap(t, label, BLOCKS[label])
        f.write_text(t)
    # the one dangling reference that points at the figure we just rebuilt
    bg = tree / "2.background.tex"
    t = bg.read_text().replace(r"\ref{fig:capacity}", r"\ref{fig:motiv}")
    bg.write_text(t)
    print("installed into", tree)
