# RaBitQ adaptive O_DIRECT I/O timing build

This benchmark-only Python binding extends the aligned-4K RaBitQ v0.5.1
adaptive-FP binding with one statistic, `odirect_read_ns`.  The counter sums
only wall-clock nanoseconds spent in the synchronous `pread` syscalls used to
fetch exact FP vectors.  It excludes query rotation, traversal, FP vector
normalization, exact distance arithmetic, and Python conversion.

The binary was built from `/tmp/rabitq-v051-adaptive` using its
`build_odirect` CMake build directory.  The source-level incremental change is
recorded in `io_wait_timing.patch`.
