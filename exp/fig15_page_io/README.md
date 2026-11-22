# Fig 15 / Sec 6.6: 4-KiB SSD pages per query vs Recall@10 (MS MARCO OpenAI-1536, Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F15.`  (11 claims) |
| Re-plot | `./exp/fig15_page_io/plot/regen.sh` |
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
| F15.1 | BBQ pages/query, OpenAI-1536 @0.98 | 64.5 |
| F15.2 | TurboQuant pages/query, OpenAI-1536 @0.98 | 120.0 |
| F15.3 | DiskANN-RaBitQ pages/query, OpenAI-1536 @0.98 | 184.4 |
| F15.4 | DiskANN pages/query, OpenAI-1536 @0.98 | 184.2 |
| F15.5 | BBQ pages/query, OpenAI-1536 @0.99 | 67.6 |
| F15.6 | DiskANN-RaBitQ pages/query, OpenAI-1536 @0.99 | 342.4 |
| F15.7 | DiskANN pages/query, OpenAI-1536 @0.99 | 373.9 |
| F15.8 | BBQ pages/query, Qwen-1024 @0.98 | 46.2 |
| F15.9 | BBQ pages/query, Qwen-1024 @0.99 | 54.3 |
| F15.10 | DiskANN-RaBitQ pages/query, Qwen-1024 @0.99 | 913.6 |
| F15.11 | DiskANN pages/query, Qwen-1024 @0.99 | 1042.0 |

## Notes

- **Stale panel in the paper.** The paper PDF draws TurboQuant on OpenAI-1536 at 80 pages/query, but the data and the text say 120 (R = 60 vectors x 2 pages at d = 1536). `regen.sh` rebuilds `page_io.dat` from `results/` and fixes the panel; the shipped `page_io.dat` is the one in the paper.
