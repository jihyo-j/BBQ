# FAISS

**No patch is needed.** BBQ (`bbq/src`) and every FAISS-based driver in `exp/scripts` build against upstream
FAISS at the commit in `BASE` (facebookresearch/faiss `main` of 2026-05-01, version string 1.14.1):

```bash
git clone https://github.com/facebookresearch/faiss.git && cd faiss && git checkout $(sed -n 's/^commit: //p' BASE)
cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=avx2 -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OpenBLAS
cmake --build build -j --target faiss_avx2          # -> build/faiss/libfaiss_avx2.a
export FAISS_ROOT=$PWD FAISS_BUILD=$PWD/build/faiss
```

Checked on 2026-11-23: with this build, `make test` and `make smoke` (all four steps, including the O_DIRECT and
io_uring rerank) pass, giving the results shown in the top-level README.

The paper runs linked against a development fork of that commit. The fork's diff is
`dev_fork_vs_upstream.patch`, kept for transparency. It adds:
- an adaptive-ef HNSW path;
- "xflag" binary-code experiments, which add the `xflags` / `block_scales` fields to `IndexBinaryFlat`.

None of these is used by BBQ or the baselines. The only places the code touches those fields are guarded by a
compile-time member check (`bbq_compat::*_fork_payloads`).
