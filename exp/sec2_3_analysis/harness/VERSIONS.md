# BBQ paper — version index

One directory per delivered version under `releases/`. The live source is
`paper/`; it is the next version in progress and the only tree that gets
edited. Nothing else in this project holds a copy of the paper.

| version | date | artefact | what it was |
|---|---|---|---|
| `_early` | 2026-11-16 | `bbq_section4.zip` | first Section-4 rewrite handed over, with the measurement harness |
| `v11` | 2026-11-17 | `bbq_section4_v11.zip` | Section 4 only |
| `v12` | 2026-11-17 | `bbq_section4_v12.zip`, `PREVIEW_section4_v12.pdf` | Section 4, longer |
| `v13` | 2026-11-18 | `bbq_section4_v13.zip` (+ `_1.69p` variant), two previews | Section 4 with the $c_Q$ figure |
| `v14` | 2026-11-18 | `bbq_section4_v14.zip`, `PREVIEW_s4_v14.pdf` | Section 4 as sent before the swap |
| `v14_base` | 2026-11-18 | `BBQ_개선안.zip` | the full source the reviewed *BBQ … (14).pdf* was built from |
| `v15` | 2026-11-18 | `BBQ_v15_full.pdf`, `bbq_v15_changed.zip`, `bbq_v15_sec34.zip`, previews, `notes/` | Sections 3 and 4 swapped and rebuilt — **the accepted base** |
| `v16` | in progress | `paper/` | reviewer items ①②③ + priorities 2 and 3 |

`releases/v15/notes/` keeps the v15 changelog, the apply note, and
`3.Mechanism.NOT_ADOPTED.tex` — the Gaussian-model Section 3 that was written
and rejected. It is kept because the rejection is a decision, not a draft.

## Building

No TeX on this host originally; TeX Live 2026 is installed out of tree at
`/home/smrc/samsung-nvme/sieun/texlive/2026`.

    export PATH=/home/smrc/samsung-nvme/sieun/texlive/2026/bin/x86_64-linux:$PATH
    cd paper && xelatex paper && bibtex paper && xelatex paper && xelatex paper

`acmart` is TeX Live 2026's v2.21; the v15 PDF was built against v1.83, so page
breaks can differ by a line. Both come out at 20 pages.

## Measurement

`src/` is the analysis harness (Sections 3 and 4, and the v16 additions);
`out/`, `gp/`, `calib/`, `rankexp/` are its outputs. Large intermediates live in
`/home/smrc/samsung-nvme/sieun/bbqeta`.
