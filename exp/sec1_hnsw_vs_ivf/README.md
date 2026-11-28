# Sec 1: HNSW vs IVF candidate generation at Recall@10 >= 0.99 (MS MARCO Qwen-1024, 1-bit ADC, T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only I.`  (5 claims here) |
| Re-plot | `./exp/sec1_hnsw_vs_ivf/plot/regen.sh` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_calibrated_adcbq_ivf_vs_rabitq_ssd.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/README.md`
- `results/candidate_generation.csv`
- `results/first_measured_point_at_recall099.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| I.1 | HNSW 1-bit distance evaluations at first point >= 0.99 (MS MARCO-1M) | 9500 |
| I.2 | HNSW Recall@10 at that point | 0.9928 |
| I.3 | IVF codes scanned per query | 778000 |
| I.4 | IVF nprobe (of 977 lists) | 768 |
| I.5 | IVF Recall@10 at that point | 0.9959 |

## Notes

- First measured operating point with Recall@10 >= 0.99: HNSW efSearch 512 (0.9928, 9,533 1-bit evaluations) vs IVF nprobe 768 of 977 lists (0.9959, 778,383 codes). The IVF point has higher recall; this is target-threshold evidence, not an iso-recall comparison.
