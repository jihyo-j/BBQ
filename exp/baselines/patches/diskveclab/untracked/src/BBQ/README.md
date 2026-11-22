# Calibrated ADC-BQ (BBQ) adapter

This directory registers BBQ as a separate DiskVecLab method. It does **not**
replace DiskANN's quantizer: that would measure `DiskANN+ADC-BQ`, while BBQ's
query path is RAM HNSW traversal with one-bit codes followed by one calibrated
FP32 refinement batch.

The adapter reuses the implementation in a sibling `RaBitQ-Library` checkout:

- FP-built HNSW;
- one-bit sign codes;
- asymmetric int16 nibble LUT (`8d` query bytes);
- held-out calibration of the query-dependent threshold;
- FP32 reranking through mmap, pread, O_DIRECT, or O_DIRECT+io_uring.

## Manifest

Copy `manifest.example.json` and replace the paths. Inputs use DiskANN's binary
matrix format: two int32 values `(rows, dimensions)` followed by row-major data.
Base/query matrices contain float32 values and ground truth contains int32 IDs.

For cosine search, provide unit-normalized base and query vectors and compute the
ground truth from those same files.

## Run

Validate the plan without building or searching:

```bash
python3 src/BBQ/run_calibrated_adcbq.py \
  --manifest src/BBQ/manifest.example.json --plan
```

Run calibration and the held-out benchmark:

```bash
python3 src/BBQ/run_calibrated_adcbq.py \
  --manifest /path/to/modern_embeddings.json \
  --efs 128 256 512 \
  --targets 0.95 0.99 \
  --calibration-queries 1000 \
  --rounds 3 \
  --fp-source odirect-uring
```

`BBQ_ROOT` or `--bbq-root` selects the RaBitQ-Library checkout. The default is
`../RaBitQ-Library` relative to this DiskVecLab checkout. Its build script accepts
`FAISS_ROOT`, `FAISS_BUILD`, `CONDA_LIB`, and `CXX` through the environment.

For every `(dataset, efSearch, target mean recall)` pair the runner:

1. creates deterministic calibration/test query splits;
2. records the top-`efSearch` ADC-BQ traversal candidates;
3. fits the smallest threshold `tau` attaining the requested mean candidate
   recall on calibration queries, counting traversal misses as misses;
4. derives a query-specific candidate count
   `R(q) = |{x: d_ADC(q,x) <= d_k(q) + tau * scale(q)}|`;
5. submits those FP32 reads as a batch and reports held-out Recall@k, QPS,
   latency, refinement counts, and bytes.

The compact common result is written to `<output_root>/bbq_results.csv`; raw
per-run output and calibration JSON files remain in each dataset output folder.
