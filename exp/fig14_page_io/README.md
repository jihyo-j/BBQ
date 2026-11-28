# Fig 14 / Sec 6.4: 4-KiB SSD pages per query vs Recall@10 (MS MARCO OpenAI-1536, Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F14.`  (24 claims here) |
| Re-plot | `./exp/fig14_page_io/plot/regen.sh` |
| Paper figure | `plot/page_io.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/plot_page_io_text_embedding_3_small.py`
- `exp/scripts/plot_msmarco_four_method_page_io.py`
- `exp/scripts/run_onebit_disk_baselines_t1.py`
- `exp/scripts/run_openai_beir_11methods_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/page_io/normalized_page_io.csv`
- `results/page_io/operating_points.csv`
- `results/page_io/page_io.pdf`
- `results/page_io/page_io.png`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| I.10 | page-read reduction vs DiskANN-RaBitQ, Qwen-1024 @0.99 (= `F14.10/F14.9`) | 16.8 |
| I.11 | page-read reduction vs DiskANN, Qwen-1024 @0.99 (= `F14.11/F14.9`) | 19.2 |
| F14.1 | BBQ pages, OpenAI-1536 @0.98 | 64.5 |
| F14.2 | TurboQuant pages, OpenAI-1536 @0.98 | 120 |
| F14.3 | DiskANN-RaBitQ pages, OpenAI-1536 @0.98 | 184.4 |
| F14.4 | DiskANN pages, OpenAI-1536 @0.98 | 184.2 |
| F14.5 | BBQ pages, OpenAI-1536 @0.99 | 67.6 |
| F14.6 | DiskANN-RaBitQ pages, OpenAI-1536 @0.99 | 342.4 |
| F14.7 | DiskANN pages, OpenAI-1536 @0.99 | 373.9 |
| F14.8 | BBQ pages, Qwen-1024 @0.98 | 46.2 |
| F14.9 | BBQ pages, Qwen-1024 @0.99 | 54.3 |
| F14.10 | DiskANN-RaBitQ pages, Qwen-1024 @0.99 | 913.6 |
| F14.11 | DiskANN pages, Qwen-1024 @0.99 | 1042 |
| F14.12 | TurboQuant pages, Qwen-1024 @0.99 | 60 |
| F14.13 | DiskANN-RaBitQ pages, Qwen-1024 @0.98 | 341 |
| F14.14 | DiskANN pages, Qwen-1024 @0.98 | 373.2 |
| F14.15 | TurboQuant / BBQ pages, OpenAI @0.98 (low end of 1.86-2.86x) (= `F14.2/F14.1`) | 1.86 |
| F14.16 | DiskANN / BBQ pages, OpenAI @0.98 (high end) (= `F14.3/F14.1`) | 2.86 |
| F14.17 | BBQ page growth 0.98->0.99, OpenAI (= `F14.5/F14.1`) | 1.05 |
| F14.18 | BBQ page growth 0.98->0.99, Qwen (= `F14.9/F14.8`) | 1.18 |
| F14.19 | DiskANN-RaBitQ growth, OpenAI (low end of 1.86-2.68x) (= `F14.6/F14.3`) | 1.86 |
| F14.20 | DiskANN-RaBitQ growth, Qwen (high end) (= `F14.10/F14.13`) | 2.68 |
| F14.21 | DiskANN growth, OpenAI (low end of 2.03-2.79x) (= `F14.7/F14.4`) | 2.03 |
| F14.22 | DiskANN growth, Qwen (high end) (= `F14.11/F14.14`) | 2.79 |

## Notes

- **Stale panel in the paper.** The paper PDF draws TurboQuant on OpenAI-1536 at 80 pages/query; the data and the text say 120 (R = 60 vectors x 2 pages at d = 1536, `tq_selected_r_heldout_t1.csv`). No stored file holds an 80-page TurboQuant curve. Only the TurboQuant curve of panel (a) differs; every other block of the shipped `page_io.dat` equals what `regen.sh` builds. The shipped `page_io.dat` is the printed one; `make plots` / `regen.sh` overwrite it with the data.
