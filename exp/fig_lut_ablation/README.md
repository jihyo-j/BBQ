# Sec 5.1: ADC-BQ with the int16 nibble LUT vs scalar FP32 ADC (no calibration, no rerank; NQ / Quora Qwen-1024; T=1)

| | |
|---|---|
| Verify | — |
| Re-plot | `./exp/fig_lut_ablation/plot/regen.sh` |
| Paper figure | `plot/lut_ablation.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_nq_quora_qwen1024_adcbq_lut_no_calibration_t1.py`
- `exp/scripts/run_nq_quora_qwen1024_adcbq_lut_ablation_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/manifest.json`
- `results/paired_ef_comparison_t1.csv`
- `results/summary_t1.csv`

## Notes

- No calibration and no FP rerank: this isolates the scorer. The scalar build is `bbq/build.sh` with `ADCBQ_FORCE_FP32_DIRECT_SCALAR=1` (FP32 ADC per dimension, no LUT, no SIMD); same graph and efSearch for both.
- The runner `run_nq_quora_qwen1024_adcbq_lut_no_calibration_t1.py` was still being extended on 2026-11-24 (a Quora BGE-M3 run); the stored summary is the NQ / Quora Qwen-1024 run used by the figure.
