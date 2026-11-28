# Fig 15 / Sec 6.4: sequential vs async batch vs async batch + overlap SSD refinement (MS MARCO; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F15.`  (10 claims here) |
| Re-plot | `./exp/fig15_io_modes/plot/regen.sh` |

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
| F15.1 | Sequential QPS, Qwen-1024 | 222 |
| F15.2 | Async Batch QPS, Qwen-1024 | 928 |
| F15.3 | Async Batch + Overlap QPS, Qwen-1024 | 993.5 |
| F15.4 | Sequential QPS, OpenAI-1536 | 376.9 |
| F15.5 | Async Batch QPS, OpenAI-1536 | 1898.3 |
| F15.6 | Async Batch + Overlap QPS, OpenAI-1536 | 2322.8 |
| F15.7 | batching speedup, Qwen (= `F15.2/F15.1`) | 4.18 |
| F15.8 | batching speedup, OpenAI (= `F15.5/F15.4`) | 5.04 |
| F15.9 | overlap gain, Qwen (%) (= `(F15.3/F15.2-1)*100`) | 7.1 |
| F15.10 | overlap gain, OpenAI (%) (= `(F15.6/F15.5-1)*100`) | 22.4 |

## Notes

- Claims use ef=320 (Qwen) / ef=88 (OpenAI), the first points at Recall@10 >= 0.99 (0.9908 / 0.9902 for all three modes).
