# RaBitQ v0.5.1 adaptive O_DIRECT + online add binding

This CPython 3.11 binding combines the existing aligned-4KiB adaptive FP
reranking path with online `HnswIndex.add()` support. The add path retains the
already-built centroids and graph, quantizes each inserted vector with its
provided frozen cluster ID, and connects it to the existing graph without
reclustering or rebuilding.

Validated locally with a `build -> add -> search_1bit_adaptive_fp_odirect`
smoke test. The test confirmed that inserted labels are returned and that the
binding reports FP read counts, O_DIRECT bytes/requests/time, and a 4096-byte
fbin data offset.
