# HNSW vs IVF candidate generation

Small appendix analysis on MS MARCO Qwen-1024 (1M vectors, 5,980 test
queries). Both methods use 1-bit ADC, exact FP32 refinement, a target coverage
of 0.99 calibrated separately for every search-width setting, and one search
thread.

- HNSW work is `ndis_mean`: upper-layer plus base-layer 1-bit distance
  evaluations.
- IVF work is `mean_scanned_codes`: 1-bit codes scanned in the selected lists.
- FP32 refinement evaluations are reported separately and are not included in
  either work count.
- IVF uses `nlist=977` (the existing `ceil(N/1024)` configuration). Under this
  calibration protocol, only `nprobe=768,896,977` were target-feasible among
  the measured settings; lower `nprobe` settings are therefore omitted rather
  than mixed with the fixed-4096 fallback measurements.

Run `python3 make_plot.py` to regenerate the CSV and PDF/PNG figure from the
existing raw measurements.

At the first measured operating point with test Recall@10 at least 0.99, HNSW
uses `efSearch=512`, reaches 0.992793 recall, and performs 9,532.71 1-bit
distance evaluations per query. IVF uses `nprobe=768/977`, reaches 0.995920
recall, and scans 778,383 codes per query, or 81.65x as many evaluations.
This is target-threshold evidence rather than an exact iso-recall comparison:
the IVF point has higher recall, and the IVF conclusion is specific to the
measured `nlist=977` configuration rather than all possible IVF designs.
