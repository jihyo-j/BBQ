# Fig 16: sequential vs async batch vs async batch + overlap (MS MARCO; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F16.`  (10 claims) |
| Re-plot | `./exp/fig16_io_modes/plot/regen.sh` |
| Paper figure | `plot/io_ablation.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_bbq_calibrated_io_modes_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/openai1536_summary_t1.csv`
- `results/qwen1024_summary_t1.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F16.1 | Sequential QPS, Qwen-1024 | 222.0 |
| F16.2 | Async Batch QPS, Qwen-1024 | 928.0 |
| F16.3 | Async Batch + Overlap QPS, Qwen-1024 | 993.5 |
| F16.4 | Sequential QPS, OpenAI-1536 | 376.9 |
| F16.5 | Async Batch QPS, OpenAI-1536 | 1898.3 |
| F16.6 | Async Batch + Overlap QPS, OpenAI-1536 | 2322.8 |
| D16.1 | Async Batch / Sequential QPS, Qwen-1024 (= `F16.2/F16.1`) | 4.18 |
| D16.2 | Async Batch / Sequential QPS, OpenAI-1536 (= `F16.5/F16.4`) | 5.04 |
| D16.3 | overlap gain over Async Batch, Qwen-1024 (%) (= `(F16.3/F16.2-1)*100`) | 7.1 |
| D16.4 | overlap gain over Async Batch, OpenAI-1536 (%) (= `(F16.6/F16.5-1)*100`) | 22.4 |

## Notes

- The claims use ef=320 (Recall@10 0.9908 for all three modes), the first point at or above 0.99. A later manuscript revision quotes the ef=256 rows (233 / 1,031 / 1,192 QPS), whose recall is 0.9897.
- `results/qwen1024_summary_t1.csv` contains both datasets (the OpenAI rows were merged in); `openai1536_summary_t1.csv` is the separate priority run used as the second input of `make_io_ablation_dat.py`.
